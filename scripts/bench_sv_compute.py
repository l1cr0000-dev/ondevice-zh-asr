"""SenseVoice（onnxruntime 版）每次整段辨識的計算時間：音訊長度 × 執行緒數 × session 選項。

不經 pipeline、不模擬即時，只量「整段辨識一次」本身。偽串流的 partial、A2 停頓時的暫定結果、
finalize 都是這個計算（對已累積的音訊整段辨識），所以它直接加在首字、暫定結果與說完 → 完整結果的延遲上。

- 從 manifest 取音訊長度 ≥ L 的句子，截取前 L 秒（與偽串流 partial 看到的前綴相同），每個長度最多 `--per-length` 句。
- 分別計時：fbank + LFR + CMVN（`features`）、模型推論（`session.run`）、整次辨識（含 CTC greedy），
  以及整次辨識的程序 CPU 時間（`cpu`，所有執行緒合計，含 spinning；多執行緒省下牆鐘時間的 CPU 代價）。
- 結果：全部片段的辨識文字雜湊，以及和第一個設定不同的片段數（多執行緒可能改變浮點累加順序）。

每個設定在獨立子程序中執行（先整批暖機一次）；`--repeat N` 時各設定交錯重跑 N 輪，計時取各輪中位數。

    uv run scripts/bench_sv_compute.py                                   # 執行緒 1 / 2 / 4，預設 session 選項
    uv run scripts/bench_sv_compute.py --threads 1 2 4 --preset default no-spin --repeat 3
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

# session 選項預設組：名稱 → 對 ort.SessionOptions 的修改
PRESETS: dict[str, Callable[[Any], None]] = {
    "default": lambda o: None,  # spinning 開
    "no-spin": lambda o: o.add_session_config_entry("session.intra_op.allow_spinning", "0"),
}


def _summary(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
    }


def worker(
    threads: int, preset: str, manifest: Path, lengths: list[float], per_length: int
) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    from src.asr import SenseVoiceOrtAsr
    from src.audio import SAMPLE_RATE
    from src.vad import load_wav

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    PRESETS[preset](opts)
    asr = SenseVoiceOrtAsr(sess_options=opts)

    rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    wavs = [load_wav(manifest.parent / r["wav"]) for r in rows]
    clips: dict[float, list[Any]] = {}
    for length in lengths:
        n = round(length * SAMPLE_RATE)
        clips[length] = [w[:n] for w in wavs if len(w) >= n][:per_length]

    def run_all() -> tuple[dict[float, dict[str, list[float]]], list[str]]:
        times: dict[float, dict[str, list[float]]] = {}
        texts: list[str] = []
        for length, xs in clips.items():
            t = times[length] = {"features": [], "session": [], "total": [], "cpu": []}
            for audio in xs:
                t0 = time.perf_counter()
                feats = asr.features(audio)
                t1 = time.perf_counter()
                c1 = time.process_time()
                logits = asr.logits(audio)  # 內含 features；session 時間以差值估計
                t2 = time.perf_counter()
                text = "" if logits is None else asr.decode(logits)
                t3 = time.perf_counter()
                c3 = time.process_time()
                t["features"].append((t1 - t0) * 1000)
                t["session"].append((t2 - t1 - (t1 - t0)) * 1000)
                t["total"].append((t3 - t1) * 1000)
                t["cpu"].append((c3 - c1) * 1000)
                texts.append(text)
                del feats
        return times, texts

    run_all()  # 暖機
    times, texts = run_all()
    return {
        "threads": threads,
        "preset": preset,
        "clips": {str(k): len(v) for k, v in clips.items()},
        "ms": {str(k): {p: _summary(v) for p, v in t.items()} for k, t in times.items()},
        "texts_md5": hashlib.md5(json.dumps(texts).encode()).hexdigest(),
        "texts": texts,
    }


def _median_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(runs[0])
    out["ms"] = {
        length: {
            part: {k: statistics.median(r["ms"][length][part][k] for r in runs) for k in stats}
            for part, stats in parts.items()
        }
        for length, parts in runs[0]["ms"].items()
    }
    out["repeats"] = len(runs)
    out["texts_stable_across_repeats"] = len({r["texts_md5"] for r in runs}) == 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "data/aishell1_test_200/manifest.jsonl"
    )
    p.add_argument("--lengths", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    p.add_argument("--per-length", type=int, default=40, help="每個長度最多幾個片段")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--preset", nargs="+", default=["default"], choices=sorted(PRESETS))
    p.add_argument("--repeat", type=int, default=1, help="各設定交錯重跑幾輪，計時取中位數")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval/results")
    p.add_argument("--worker", nargs=2, metavar=("THREADS", "PRESET"), help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        threads, preset = args.worker
        res = worker(int(threads), preset, args.manifest, args.lengths, args.per_length)
        print(json.dumps(res, ensure_ascii=False))
        return 0

    configs = [(t, s) for t in args.threads for s in args.preset]
    runs: dict[tuple[int, str], list[dict[str, Any]]] = {c: [] for c in configs}
    for r in range(args.repeat):
        for threads, preset in configs:
            cmd = [sys.executable, __file__, "--manifest", str(args.manifest)]
            cmd += ["--lengths", *map(str, args.lengths), "--per-length", str(args.per_length)]
            cmd += ["--worker", str(threads), preset]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            runs[(threads, preset)].append(json.loads(out.splitlines()[-1]))
            print(f"  round {r + 1}/{args.repeat}: threads={threads} {preset}", file=sys.stderr)

    results = [_median_run(runs[c]) for c in configs]
    ref = results[0]["texts"]
    lengths = list(results[0]["ms"])
    clips = results[0]["clips"]
    print(
        "整段辨識 mean / P90 (ms)；片段數 "
        + "、".join(f"{float(k):g} s {v}" for k, v in clips.items())
    )
    print(
        "| threads | preset | "
        + " | ".join(f"{float(k):g} s" for k in lengths)
        + " | 其中推論 (最長) | CPU 時間 (最長) | 結果與第一列不同的片段 | 各輪結果一致 |"
    )
    print("|---|---|" + "---|" * (len(lengths) + 4))
    for res in results:
        res["texts_differ_from_first"] = sum(a != b for a, b in zip(ref, res["texts"]))
        cells = " | ".join(
            f"{res['ms'][k]['total']['mean']:.1f} / {res['ms'][k]['total']['p90']:.1f}"
            for k in lengths
        )
        sess = res["ms"][lengths[-1]]["session"]["mean"]
        cpu = res["ms"][lengths[-1]]["cpu"]["mean"]
        print(
            f"| {res['threads']} | {res['preset']} | {cells} | {sess:.1f} | {cpu:.1f} "
            f"| {res['texts_differ_from_first']} | {res['texts_stable_across_repeats']} |"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.results_dir / f"bench_sv_compute_{stamp}_{args.manifest.parent.name}.json"
    for res in results:
        res.pop("texts")
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果：{path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
