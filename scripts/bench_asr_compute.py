"""ASR / VAD 推論計算時間的細部量測（backlog B8：多執行緒、ORT session 選項）。

不經 pipeline、不模擬即時，只量計算本身。每句音訊整條以 10 ms chunk 送入 ASR stream 再 finalize，記錄：

- encoder：每次 encoder 推論（一個 chunk，X-ASR 480 ms 為 61 frame）的耗時。
- 搜尋：每個 chunk 的 greedy search（joiner + decoder）耗時。
- finalize：補零解碼剩餘音訊的耗時（= A2 停頓時產生暫定結果的計算，直接加在結果延遲上）。
- 重新解碼：另開 stream 一次送入整句音訊的耗時（= A2 恢復說話時的計算，加在恢復後的 partial 上）。
- VAD：Silero 每個 32 ms frame 的耗時。
- 結果：全部句子的 token 序列雜湊，以及和第一個設定不同的句數（多執行緒、圖最佳化可能改變結果）。
- 記憶體：載入後與量測結束時的 RSS、physical footprint（vmmap，與 scripts/probe_footprint.py 相同）。

每個設定在獨立子程序中執行；`--repeat N` 時各設定交錯重跑 N 輪（減少背景負載的偏差），計時取各輪中位數。

    uv run scripts/bench_asr_compute.py                         # 執行緒 1 / 2 / 4，預設 session 選項
    uv run scripts/bench_asr_compute.py --threads 1 --preset default extended no-arena no-spin
    uv run scripts/bench_asr_compute.py --manifest data/aishell1_align_300/manifest.jsonl --limit 0
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# session 選項預設組：名稱 → 對 ort.SessionOptions 的修改（ASR 三個 session 與 VAD 都套用）
PRESETS: dict[str, Callable[[Any], None]] = {
    "default": lambda o: None,  # ORT_ENABLE_ALL、arena 開、prepacking 開、spinning 開
    "extended": lambda o: setattr(o, "graph_optimization_level", _level("ORT_ENABLE_EXTENDED")),
    "basic": lambda o: setattr(o, "graph_optimization_level", _level("ORT_ENABLE_BASIC")),
    "no-arena": lambda o: setattr(o, "enable_cpu_mem_arena", False),
    "no-mem-pattern": lambda o: setattr(o, "enable_mem_pattern", False),
    "no-prepack": lambda o: o.add_session_config_entry("session.disable_prepacking", "1"),
    "no-spin": lambda o: o.add_session_config_entry("session.intra_op.allow_spinning", "0"),
    "denormal-zero": lambda o: o.add_session_config_entry("session.set_denormal_as_zero", "1"),
    # 不是 session 選項：greedy search 退回逐 frame 呼叫 joiner（對照批次 joiner）
    "per-frame-joiner": lambda o: None,
}


def _level(name: str) -> Any:
    import onnxruntime as ort  # type: ignore[import-untyped]

    return getattr(ort.GraphOptimizationLevel, name)


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
    }


def _footprint_snapshot() -> Callable[[str], dict[str, Any]]:
    """沿用 scripts/probe_footprint.py 的 vmmap 量測（以檔案路徑載入，scripts/ 不是套件）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_footprint", Path(__file__).with_name("probe_footprint.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot: Callable[[str], dict[str, Any]] = module.snapshot
    return snapshot


def worker(model: str, threads: int, preset: str, manifest: Path, limit: int) -> dict[str, Any]:
    import onnxruntime as ort

    from src.asr import TRANSDUCER_SPECS, OrtTransducerConfig, ZipformerOrtAsr
    from src.vad import SileroVadModel, load_wav

    snapshot = _footprint_snapshot()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    PRESETS[preset](opts)

    files, tail_padding_s = TRANSDUCER_SPECS[model.removesuffix("-ort")]
    asr = ZipformerOrtAsr(
        files,
        OrtTransducerConfig(num_threads=threads, tail_padding_s=tail_padding_s),
        sess_options=opts,
    )
    if preset == "per-frame-joiner":
        asr._batch_joiner = False
    vad = SileroVadModel(num_threads=threads, sess_options=opts)
    loaded = snapshot("loaded")

    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
    rows = rows[:limit] if limit else rows
    wavs = [load_wav(manifest.parent / r["wav"]) for r in rows]

    # 包住 encoder 與 greedy search，逐次計時
    enc_ms: list[float] = []
    search_ms: list[float] = []
    encoder, run_enc = asr._encoder, asr._encoder.run

    def timed_encoder(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = run_enc(*args, **kwargs)
        enc_ms.append((time.perf_counter() - t0) * 1000)
        return out

    from src.asr.zipformer_ort import ZipformerOrtStream

    greedy = ZipformerOrtStream._greedy_search

    def timed_greedy(self: ZipformerOrtStream, encoder_out: Any) -> None:
        t0 = time.perf_counter()
        greedy(self, encoder_out)
        search_ms.append((time.perf_counter() - t0) * 1000)

    def decode() -> tuple[list[list[int]], list[float], float]:
        tokens: list[list[int]] = []
        finalize_ms: list[float] = []
        accept_s = 0.0
        for w in wavs:
            stream = asr.create_stream()
            t0 = time.perf_counter()
            for i in range(0, len(w), 160):
                stream.accept_waveform(w[i : i + 160])
            t1 = time.perf_counter()
            stream.finalize()
            accept_s += t1 - t0
            finalize_ms.append((time.perf_counter() - t1) * 1000)
            tokens.append(list(stream._tokens))
        return tokens, finalize_ms, accept_s

    decode()  # 暖機：整個集合跑一次
    encoder.run = timed_encoder
    ZipformerOrtStream._greedy_search = timed_greedy  # type: ignore[method-assign]
    enc_ms.clear()
    search_ms.clear()
    t0 = time.perf_counter()
    tokens, finalize_ms, accept_s = decode()
    total_s = time.perf_counter() - t0
    encoder.run = run_enc
    ZipformerOrtStream._greedy_search = greedy  # type: ignore[method-assign]

    redecode_ms: list[float] = []
    for w in wavs:
        stream = asr.create_stream()
        t1 = time.perf_counter()
        stream.accept_waveform(w)
        redecode_ms.append((time.perf_counter() - t1) * 1000)

    vad_ms: list[float] = []
    frame = vad.frame_size
    for w in wavs:
        vad.reset()
        for i in range(0, len(w) - frame + 1, frame):
            t1 = time.perf_counter()
            vad(w[i : i + frame])
            vad_ms.append((time.perf_counter() - t1) * 1000)

    audio_s = sum(len(w) for w in wavs) / 16000
    end = snapshot("end")
    return {
        "model": model,
        "threads": threads,
        "preset": preset,
        "batch_joiner": asr._batch_joiner,
        "num_utterances": len(wavs),
        "audio_s": audio_s,
        "asr_rtf": total_s / audio_s,
        "encoder_ms": _summary(enc_ms),
        "search_ms": _summary(search_ms),
        "finalize_ms": _summary(finalize_ms),
        "redecode_ms": _summary(redecode_ms),
        "redecode_ms_per_audio_s": sum(redecode_ms) / audio_s,
        "vad_frame_ms": _summary(vad_ms),
        "vad_rtf": sum(vad_ms) / 1000 / audio_s,
        "tokens_md5": hashlib.md5(json.dumps(tokens).encode()).hexdigest(),
        "tokens": tokens,
        "memory_mb": {
            "loaded_rss": loaded["rss"],
            "loaded_footprint": loaded["footprint"],
            "end_rss": end["rss"],
            "end_footprint": end["footprint"],
            "peak_rss": end["peak_rss"],
            "footprint_peak": end["footprint_peak"],
        },
        "accept_s": accept_s,
    }


def _median_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """各輪計時取中位數（記憶體、結果取第一輪）。"""
    out = dict(runs[0])
    out["asr_rtf"] = statistics.median(r["asr_rtf"] for r in runs)
    out["vad_rtf"] = statistics.median(r["vad_rtf"] for r in runs)
    out["redecode_ms_per_audio_s"] = statistics.median(r["redecode_ms_per_audio_s"] for r in runs)
    for key in ("encoder_ms", "search_ms", "finalize_ms", "redecode_ms", "vad_frame_ms"):
        out[key] = {k: statistics.median(r[key][k] for r in runs) for k in runs[0][key]}
    out["repeats"] = len(runs)
    out["tokens_stable_across_repeats"] = len({r["tokens_md5"] for r in runs}) == 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "data/aishell1_test_200/manifest.jsonl"
    )
    p.add_argument("--limit", type=int, default=60, help="只用前 N 句（0 = 全部）")
    p.add_argument(
        "--model",
        default="x-asr-480ms-int8-ort",
        choices=["x-asr-480ms-int8-ort", "x-asr-160ms-int8-ort"],
    )
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--preset", nargs="+", default=["default"], choices=sorted(PRESETS))
    p.add_argument("--repeat", type=int, default=1, help="各設定交錯重跑幾輪，計時取中位數")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval/results")
    p.add_argument(
        "--worker", nargs=3, metavar=("MODEL", "THREADS", "PRESET"), help=argparse.SUPPRESS
    )
    args = p.parse_args()

    if args.worker:
        model, threads, preset = args.worker
        print(json.dumps(worker(model, int(threads), preset, args.manifest, args.limit)))
        return 0

    configs = [(t, s) for t in args.threads for s in args.preset]
    runs: dict[tuple[int, str], list[dict[str, Any]]] = {c: [] for c in configs}
    for r in range(args.repeat):
        for threads, preset in configs:
            cmd = [sys.executable, __file__, "--manifest", str(args.manifest)]
            cmd += ["--limit", str(args.limit), "--worker", args.model, str(threads), preset]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            runs[(threads, preset)].append(json.loads(out.splitlines()[-1]))
            print(f"  round {r + 1}/{args.repeat}: threads={threads} {preset}", file=sys.stderr)

    results = [_median_run(runs[c]) for c in configs]
    ref = results[0]["tokens"]
    print(
        "| threads | preset | ASR RTF | encoder mean / P90 (ms) | search mean (ms) "
        "| finalize mean / P90 (ms) | 重新解碼 (ms / 音訊 s) | VAD (ms / frame) "
        "| footprint 載入後 / 峰值 (MB) | 峰值 RSS (MB) | 結果與第一列不同的句數 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for res in results:
        res["utterances_differ_from_first"] = sum(a != b for a, b in zip(ref, res["tokens"]))
        m = res["memory_mb"]
        print(
            f"| {res['threads']} | {res['preset']} | {res['asr_rtf']:.4f} "
            f"| {res['encoder_ms']['mean']:.2f} / {res['encoder_ms']['p90']:.2f} "
            f"| {res['search_ms']['mean']:.2f} "
            f"| {res['finalize_ms']['mean']:.1f} / {res['finalize_ms']['p90']:.1f} "
            f"| {res['redecode_ms_per_audio_s']:.1f} | {res['vad_frame_ms']['mean']:.3f} "
            f"| {m['loaded_footprint']:.0f} / {m['footprint_peak']:.0f} | {m['peak_rss']:.0f} "
            f"| {res['utterances_differ_from_first']} |"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.results_dir / f"bench_compute_{stamp}_{args.manifest.parent.name}.json"
    for res in results:
        res.pop("tokens")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果：{path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
