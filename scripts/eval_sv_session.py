"""全鏈路評測（eval/run_eval.py 的 worker）換 SenseVoice（onnxruntime 版）的 session 選項，比較記憶體、CPU 與延遲。

session 選項預設組與 `scripts/bench_sv_compute.py` 相同（`PRESETS`）。每個（集合, 預設組）在獨立子程序中執行：
子程序把 `SenseVoiceOrtAsr` 換成套用該預設組的版本（只影響這個子程序），再呼叫 `run_eval.worker`，
其餘設定都是 run_eval 的預設值（預設模型、A2、chunk 10 ms、暖機句數）。另外記：

- 程序 CPU 時間（user + sys，所有執行緒合計）：暖機結束（`reset_compute_stats`）→ 全部句子跑完，
  除以音訊總長（含補靜音）= 每秒音訊花多少 CPU 秒；與 RTF（牆鐘）對照可看出多執行緒 / spinning 的代價。
- 跑完後的 RSS / physical footprint 與程序生命期的 footprint 峰值（vmmap，`scripts/probe_footprint.py`）。
- 對齊集（id 在 data/aishell1_align_300/alignments.jsonl）另算「說完 → 完整結果」（`eval.pause`）。
- 最終文字與第一個預設組不同的句數（逐句比對）。

`--repeat N` 時各集合 × 預設組交錯重跑 N 輪，數字取各輪中位數；逐句結果另存 JSON（每輪每設定一份）。

    uv run scripts/eval_sv_session.py --preset default no-arena --repeat 3
    uv run scripts/eval_sv_session.py --manifest data/aishell1_align_300_short3/manifest.jsonl \\
        data/aishell1_align_300/manifest.jsonl --preset default no-arena
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # 共用其他腳本的函式（probe_footprint 等）

from bench_sv_compute import PRESETS

from eval.run_eval import SMOKE_MANIFEST, RunConfig

ALIGNMENTS = REPO_ROOT / "data" / "aishell1_align_300" / "alignments.jsonl"


def default_config(manifest: Path) -> RunConfig:
    """run_eval 不加參數時的設定（單一組合），只換 manifest。"""
    from eval.run_eval import parse_args
    from src.asr import default_threads

    a = parse_args(["--manifest", str(manifest)])
    model = a.model[0]
    return RunConfig(
        model=model,
        manifest=str(manifest.resolve()),
        threads=a.threads[0] or default_threads(model),
        chunk_ms=a.chunk_ms[0],
        pre_roll_ms=a.pre_roll_ms[0],
        vad_threshold=a.threshold,
        vad_neg_threshold=a.neg_threshold,
        vad_min_speech_ms=a.min_speech_ms,
        vad_min_silence_ms=a.min_silence_ms[0],
        vad_speech_pad_ms=a.speech_pad_ms,
        warmup=a.warmup,
        lead_silence_ms=a.lead_silence_ms,
        tail_silence_ms=a.tail_silence_ms,
        pause_ms=a.pause_ms[0],
        provisional=a.provisional[0],
    )


def worker(cfg: RunConfig, preset: str) -> dict[str, Any]:
    from bench_sv_compute import session_options
    from probe_footprint import snapshot

    import src.asr.sense_voice as sv
    from eval import run_eval

    cpu_mark: list[float] = []

    class PresetAsr(sv.SenseVoiceOrtAsr):
        def __init__(self, config: sv.SenseVoiceConfig | None = None, **kw: Any) -> None:
            opts, run_opts = session_options((config or sv.SenseVoiceConfig()).num_threads, preset)
            super().__init__(config, sess_options=opts, run_options=run_opts, **kw)

        def reset_compute_stats(self) -> None:  # run_eval 在暖機結束時呼叫一次
            super().reset_compute_stats()
            cpu_mark.append(time.process_time())

    # src.asr 的工廠以模組屬性名稱取用 SenseVoiceOrtAsr，換掉兩處即可
    import src.asr as asr_pkg

    sv.SenseVoiceOrtAsr = PresetAsr  # type: ignore[misc, assignment]
    asr_pkg.SenseVoiceOrtAsr = PresetAsr  # type: ignore[misc, assignment]
    raw = run_eval.worker(cfg)
    if not cpu_mark:
        raise RuntimeError(f"{cfg.model} 不是 SenseVoice onnxruntime 版，預設組沒有套用")
    raw["cpu_s"] = time.process_time() - cpu_mark[-1]
    raw["memory_end"] = snapshot("跑完")
    raw["asr_name"] += f" preset={preset}"
    return raw


def _true_ends() -> dict[str, float]:
    if not ALIGNMENTS.is_file():
        return {}
    ends = {}
    for line in ALIGNMENTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        ends[row["id"]] = row["chars"][-1]["end"]
    return ends


def metrics(res: dict[str, Any], raw: dict[str, Any], ends: dict[str, float]) -> dict[str, Any]:
    from eval.pause import full_result_latency_s

    m = res["metrics"]
    utts = res["utterances"]
    processed = sum(u.get("processed_s", u["duration_s"]) for u in utts)
    lat = [
        x * 1000
        for u in utts
        if u["id"] in ends and (x := full_result_latency_s(u, ends[u["id"]])) is not None
    ]
    end = raw["memory_end"]
    return {
        "cer": m["cer"] * 100,
        "rtf": m["rtf"],
        "cpu_per_audio_s": raw["cpu_s"] / processed,
        "first_token_mean": m["first_token_latency_ms"]["mean"],
        "provisional_mean": (m["provisional"] or {}).get("redecode_ms", {}).get("mean"),
        "full_result_mean": statistics.fmean(lat) if lat else None,
        "full_result_p90": sorted(lat)[int(len(lat) * 0.9)] if lat else None,
        "after_load_rss": m["memory_mb"]["after_load_rss"],
        "peak_rss": m["memory_mb"]["peak_rss"],
        "footprint_end": end["footprint"],
        "footprint_peak": end["footprint_peak"],
    }


def main() -> int:
    if "--worker" in sys.argv[1:]:
        job = json.loads(sys.stdin.read())
        print(json.dumps(worker(RunConfig(**job["config"]), job["preset"])))
        return 0

    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, nargs="+", default=[SMOKE_MANIFEST])
    p.add_argument("--preset", nargs="+", default=["default", "no-arena"], choices=sorted(PRESETS))
    p.add_argument("--repeat", type=int, default=1, help="交錯重跑幾輪，數字取中位數")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval/results/sv_session")
    args = p.parse_args()

    from eval.run_eval import environment, summarize_run

    env = environment()
    ends = _true_ends()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    keys = [(m, s) for m in args.manifest for s in args.preset]
    runs: dict[tuple[Path, str], list[dict[str, Any]]] = {k: [] for k in keys}
    hyps: dict[tuple[Path, str], list[list[str]]] = {k: [] for k in keys}
    loads: list[float] = []
    for r in range(args.repeat):
        for manifest, preset in keys:
            cfg = default_config(manifest)
            loads.append(os.getloadavg()[0])
            proc = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps({"config": asdict(cfg), "preset": preset}),
                capture_output=True,
                text=True,
                check=False,
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"{manifest.parent.name} {preset} 子程序失敗：\n{proc.stderr}")
            raw = json.loads(proc.stdout.splitlines()[-1])
            res = summarize_run(cfg, raw, env)
            res["session_preset"] = preset
            out = args.results_dir / f"{stamp}_{manifest.parent.name}_{preset}_r{r + 1}.json"
            out.write_text(json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
            runs[(manifest, preset)].append(metrics(res, raw, ends))
            hyps[(manifest, preset)].append([u["hyp"] for u in res["utterances"]])
            print(
                f"  round {r + 1}/{args.repeat}: {manifest.parent.name} {preset}", file=sys.stderr
            )

    print(
        f"load average（每次啟動子程序時的 1 分鐘值）{min(loads):.1f}–{max(loads):.1f}；"
        f"{args.repeat} 輪中位數"
    )
    print(
        "| 集合 | 預設組 | CER % | RTF | CPU 秒 / 音訊秒 | 首字 mean | 暫定結果計算 mean | "
        "說完 → 完整結果 mean / P90 | 載入後 RSS | peak RSS | 跑完 footprint | footprint 峰值 | "
        "與第一個預設組不同的句數（各輪） |"
    )
    print("|---|---|" + "---|" * 11)

    def med(xs: list[dict[str, Any]], k: str) -> float | None:
        vals = [x[k] for x in xs if x[k] is not None]
        return statistics.median(vals) if vals else None

    def fmt(x: float | None, spec: str = ".0f") -> str:
        return "—" if x is None else format(x, spec)

    for manifest, preset in keys:
        xs = runs[(manifest, preset)]
        ref = hyps[(manifest, args.preset[0])][0]
        diff = " / ".join(
            str(sum(a != b for a, b in zip(ref, h))) for h in hyps[(manifest, preset)]
        )
        print(
            f"| {manifest.parent.name} | {preset} | {fmt(med(xs, 'cer'), '.2f')} "
            f"| {fmt(med(xs, 'rtf'), '.4f')} | {fmt(med(xs, 'cpu_per_audio_s'), '.4f')} "
            f"| {fmt(med(xs, 'first_token_mean'))} | {fmt(med(xs, 'provisional_mean'))} "
            f"| {fmt(med(xs, 'full_result_mean'))} / {fmt(med(xs, 'full_result_p90'))} "
            f"| {fmt(med(xs, 'after_load_rss'))} | {fmt(med(xs, 'peak_rss'))} "
            f"| {fmt(med(xs, 'footprint_end'))} | {fmt(med(xs, 'footprint_peak'))} | {diff} |"
        )
    summary = args.results_dir / f"{stamp}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "environment": env,
                "load_average": [min(loads), max(loads)],
                "runs": [
                    {"manifest": str(m), "preset": s, "rounds": runs[(m, s)]} for m, s in keys
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n總表：{summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
