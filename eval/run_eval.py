"""全鏈路評測：VAD + 串流 ASR 在評測集上的 CER / RTF / 首字延遲 / 峰值記憶體 / 模型大小 / 切段數。

用法：
    uv run scripts/prepare_eval_data.py      # 先準備資料（一次即可）
    uv run eval/run_eval.py                  # 預設模型與 pipeline 設定
    uv run eval/run_eval.py --model zipformer-zh-14m --min-silence-ms 500 --pre-roll-ms 200
    # 參數掃描：多個值取笛卡兒積，每個組合各存一份結果，最後印出總表
    uv run eval/run_eval.py --model x-asr-480ms-int8 zipformer-zh-14m \\
        --min-silence-ms 100 300 500 800 --pre-roll-ms 0 100 200 300 --tag sweep

每個組合在獨立子程序中執行（模型載入、推論、量峰值記憶體），父程序只負責 CER 與彙整，
避免評測框架本身（jiwer、OpenCC）的記憶體算進去。結果存成 eval/results/<日期>_<設定名>.json，
內含完整設定、機器資訊、commit、彙總指標與逐句結果。指標定義見 docs/eval.md。
"""

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "data" / "aishell1_test_200" / "manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


@dataclass(frozen=True)
class RunConfig:
    """一次評測的完整設定（全部寫入結果 JSON）。"""

    model: str
    manifest: str
    threads: int
    chunk_ms: int
    pre_roll_ms: int
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    warmup: int

    @property
    def name(self) -> str:
        return (
            f"{self.model}_sil{self.vad_min_silence_ms}_pre{self.pre_roll_ms}"
            f"_chunk{self.chunk_ms}_t{self.threads}"
        )


# ---------- 子程序：載入模型、跑 pipeline、量記憶體 ----------


def _rss_mb() -> float:
    import psutil  # type: ignore[import-untyped]

    return float(psutil.Process().memory_info().rss) / 2**20


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 單位是 bytes，Linux 是 KB
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def worker(cfg: RunConfig) -> dict[str, Any]:
    from src.asr import SAMPLE_RATE, load_asr
    from src.pipeline import PipelineConfig, transcribe
    from src.vad import SileroVadModel, VadConfig, load_wav

    rss_baseline = _rss_mb()
    manifest_path = Path(cfg.manifest)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    pipe_cfg = PipelineConfig(
        vad=VadConfig(
            threshold=cfg.vad_threshold,
            min_speech_ms=cfg.vad_min_speech_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
            speech_pad_ms=cfg.vad_speech_pad_ms,
        ),
        pre_roll_ms=cfg.pre_roll_ms,
    )

    t0 = time.perf_counter()
    vad_model = SileroVadModel(num_threads=cfg.threads)
    asr = load_asr(cfg.model, num_threads=cfg.threads)
    load_s = time.perf_counter() - t0
    rss_loaded = _rss_mb()
    files = [vad_model.model_path, *asr.model_files()]

    def run(path: Path) -> tuple[float, float, list[Any]]:
        audio = load_wav(path)  # 讀檔不計時
        start = time.perf_counter()
        utts = transcribe(audio, vad_model, asr, pipe_cfg, chunk_ms=cfg.chunk_ms)
        return len(audio) / SAMPLE_RATE, time.perf_counter() - start, utts

    for row in rows[: cfg.warmup]:
        run(manifest_path.parent / row["wav"])

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        duration, compute, utts = run(manifest_path.parent / row["wav"])
        results.append(
            {
                "id": row["id"],
                "ref": row["text"],
                "hyp": " ".join(u.text for u in utts if u.text),
                "duration_s": duration,
                "compute_s": compute,
                "segments": [
                    {
                        "text": u.text,
                        "speech_start_s": u.speech_start_s,
                        "speech_end_s": u.speech_end_s,
                        "vad_start_delay_s": u.start_emitted_s - u.speech_start_s,
                        "first_token_latency_s": u.first_token_latency_s,
                        "tail_latency_s": u.tail_latency_s,
                    }
                    for u in utts
                ],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {cfg.name}: {i + 1}/{len(rows)}", file=sys.stderr)

    return {
        "asr_name": asr.name,
        "load_s": load_s,
        "memory_mb": {
            "baseline_rss": rss_baseline,
            "after_load_rss": rss_loaded,
            "peak_rss": _peak_rss_mb(),
        },
        "model_files": [
            {"path": str(p.relative_to(REPO_ROOT)), "bytes": p.stat().st_size} for p in files
        ],
        "utterances": results,
    }


# ---------- 父程序：設定、CER、彙整、輸出 ----------


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict[str, Any]:
    status = _cmd("git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no")
    versions: dict[str, str | None] = {}
    for pkg in ("onnxruntime", "sherpa-onnx", "numpy", "jiwer", "opencc"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "commit": _cmd("git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"),
        "dirty": bool(status) and status != "unknown",
        "machine": _cmd("sysctl", "-n", "hw.model"),
        "chip": _cmd("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": versions,
    }


def _precision(path: str) -> str:
    name = Path(path).name
    if not name.endswith(".onnx"):
        return "-"
    return "INT8" if ".int8." in name else "FP32"


def _first_with_text(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((s for s in segments if s["text"].strip()), None)


def summarize_run(cfg: RunConfig, raw: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    from eval.metrics import corpus_cer, summarize
    from eval.text_norm import normalize
    from src.vad import VAD_MODEL_NAME

    utts = raw["utterances"]
    cer = corpus_cer([u["ref"] for u in utts], [u["hyp"] for u in utts])
    total_audio = sum(u["duration_s"] for u in utts)
    total_compute = sum(u["compute_s"] for u in utts)

    first_token_ms: list[float] = []  # 每條音訊一個值：第一個有字的句段
    tail_ms: list[float] = []  # 每條音訊一個值：最後一個句段
    vad_delay_ms: list[float] = []
    seg_counts: list[int] = []
    for u in utts:
        segs = u["segments"]
        seg_counts.append(len(segs))
        first = _first_with_text(segs)
        if first is not None and first["first_token_latency_s"] is not None:
            first_token_ms.append(first["first_token_latency_s"] * 1000)
            vad_delay_ms.append(first["vad_start_delay_s"] * 1000)
        if segs and segs[-1]["tail_latency_s"] is not None:
            tail_ms.append(segs[-1]["tail_latency_s"] * 1000)
        u["ref_norm"] = normalize(u["ref"])
        u["hyp_norm"] = normalize(u["hyp"])

    seg_hist: dict[str, int] = {}
    for n in seg_counts:
        key = str(n) if n < 3 else "3+"
        seg_hist[key] = seg_hist.get(key, 0) + 1

    files = [{**f, "precision": _precision(f["path"])} for f in raw["model_files"]]
    model_bytes = sum(f["bytes"] for f in files)
    return {
        "name": cfg.name,
        "date": date.today().isoformat(),
        "config": {
            **asdict(cfg),
            "asr_name": raw["asr_name"],
            "vad_model": VAD_MODEL_NAME,
            "timing": "simulated-realtime",
        },
        "environment": env,
        "metrics": {
            "num_utterances": len(utts),
            "audio_s": total_audio,
            "cer": cer.cer,
            "cer_counts": asdict(cer),
            "rtf": total_compute / total_audio,
            "first_token_latency_ms": summarize(first_token_ms),
            "no_text_utterances": len(utts) - len(first_token_ms),
            "vad_start_delay_ms": summarize(vad_delay_ms),
            "tail_latency_ms": summarize(tail_ms),
            "segments_per_utterance": {
                "mean": sum(seg_counts) / len(seg_counts),
                "max": max(seg_counts),
                "histogram": dict(sorted(seg_hist.items())),
            },
            "memory_mb": raw["memory_mb"],
            "model_size_mb": model_bytes / 2**20,
            "load_s": raw["load_s"],
        },
        "model_files": files,
        "utterances": utts,
    }


def run_one(cfg: RunConfig) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(asdict(cfg)),
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg.name} 子程序失敗：\n{proc.stderr}")
    sys.stderr.write(proc.stderr)
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


# ---------- 輸出 ----------


def _fmt(x: float | None, spec: str = ".0f") -> str:
    return "-" if x is None else format(x, spec)


TABLE_HEADER = (
    "| model | min_sil | pre_roll | CER % | RTF | first-token mean / P90 (ms) "
    "| tail mean (ms) | seg/utt (max) | peak RSS (MB) | model (MB) |"
)


def table_row(res: dict[str, Any]) -> str:
    c, m = res["config"], res["metrics"]
    ft, seg = m["first_token_latency_ms"], m["segments_per_utterance"]
    return (
        f"| {c['model']} | {c['vad_min_silence_ms']} | {c['pre_roll_ms']} "
        f"| {m['cer'] * 100:.2f} | {m['rtf']:.4f} | {_fmt(ft['mean'])} / {_fmt(ft['p90'])} "
        f"| {_fmt(m['tail_latency_ms']['mean'])} | {seg['mean']:.2f} ({seg['max']}) "
        f"| {m['memory_mb']['peak_rss']:.0f} | {m['model_size_mb']:.1f} |"
    )


def print_detail(res: dict[str, Any]) -> None:
    from src.cli import display_width

    c, m, env = res["config"], res["metrics"], res["environment"]
    ft, tail, cc = m["first_token_latency_ms"], m["tail_latency_ms"], m["cer_counts"]
    seg = m["segments_per_utterance"]
    vad = (
        f"threshold={c['vad_threshold']} min_speech={c['vad_min_speech_ms']}ms"
        f" min_silence={c['vad_min_silence_ms']}ms pad={c['vad_speech_pad_ms']}ms"
    )
    dirty = " (dirty)" if env["dirty"] else ""
    counts = (
        f"S={cc['substitutions']} D={cc['deletions']} I={cc['insertions']} N={cc['ref_tokens']}"
    )
    ftl = (
        f"mean {_fmt(ft['mean'])}  P50 {_fmt(ft['p50'])}  P90 {_fmt(ft['p90'])}"
        f"  P99 {_fmt(ft['p99'])}  (n={ft['n']}，無輸出 {m['no_text_utterances']} 句)"
    )
    mem = m["memory_mb"]
    mem_s = (
        f"peak RSS {mem['peak_rss']:.0f}"
        f"  (載入前 {mem['baseline_rss']:.0f}，載入後 {mem['after_load_rss']:.0f})"
    )
    lines = [
        ("設定", res["name"]),
        ("ASR", c["asr_name"]),
        ("VAD", vad),
        ("pre-roll / chunk", f"{c['pre_roll_ms']} ms / {c['chunk_ms']} ms，暖機 {c['warmup']} 句"),
        (
            "環境",
            f"{env['chip']} ({env['machine']}) macOS {env['macos']} commit {env['commit']}{dirty}",
        ),
        ("句數 / 音訊", f"{m['num_utterances']} 句 / {m['audio_s'] / 60:.1f} 分鐘"),
        ("CER", f"{m['cer'] * 100:.2f}%  ({counts})"),
        ("RTF", f"{m['rtf']:.4f}"),
        ("首字延遲 (ms)", ftl),
        ("  其中 VAD 觸發 (ms)", f"mean {_fmt(m['vad_start_delay_ms']['mean'])}"),
        ("尾字延遲 (ms)", f"mean {_fmt(tail['mean'])}  P90 {_fmt(tail['p90'])}"),
        ("每條切段數", f"mean {seg['mean']:.2f}  max {seg['max']}  分佈 {seg['histogram']}"),
        ("記憶體 (MB)", mem_s),
        ("模型大小 (MB)", f"{m['model_size_mb']:.1f}"),
    ]
    width = max(display_width(k) for k, _ in lines)
    print("-" * 72)
    for k, v in lines:
        print(f"{k}{' ' * (width - display_width(k))}  {v}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.cli import add_pipeline_args, non_negative_int, positive_int

    p = argparse.ArgumentParser(
        description="全鏈路評測（VAD + 串流 ASR）。多個值的參數取笛卡兒積做掃描。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="評測集 manifest")
    add_pipeline_args(p, sweep=True)
    p.add_argument(
        "--warmup", type=non_negative_int, default=3, help="正式計時前先跑幾句暖機（不計入）"
    )
    p.add_argument(
        "--limit", type=positive_int, help="只評測前 N 句（除錯用，結果名稱會加上 _limN）"
    )
    p.add_argument("--tag", default="", help="掃描總表檔名的標籤")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="結果輸出目錄")
    return p.parse_args(argv)


def main() -> int:
    if "--worker" in sys.argv[1:]:
        cfg = RunConfig(**json.loads(sys.stdin.read()))
        print(json.dumps(worker(cfg)))
        return 0

    args = parse_args()
    if not args.manifest.is_file():
        print(
            f"錯誤：找不到 {args.manifest}，請先執行 uv run scripts/prepare_eval_data.py",
            file=sys.stderr,
        )
        return 1
    manifest = args.manifest.resolve()
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        print(f"錯誤：{manifest} 是空的", file=sys.stderr)
        return 1
    if args.limit:
        lines = lines[: args.limit]
        limited = args.results_dir / ".tmp" / f"{manifest.parent.name}_lim{args.limit}.jsonl"
        limited.parent.mkdir(parents=True, exist_ok=True)
        # wav 路徑相對 manifest，改成絕對路徑
        rows = [json.loads(line) for line in lines]
        for r in rows:
            r["wav"] = str(manifest.parent / r["wav"])
        limited.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest = limited

    configs = [
        RunConfig(
            model=model,
            manifest=str(manifest),
            threads=threads,
            chunk_ms=chunk,
            pre_roll_ms=pre,
            vad_threshold=args.threshold,
            vad_min_speech_ms=args.min_speech_ms,
            vad_min_silence_ms=sil,
            vad_speech_pad_ms=args.speech_pad_ms,
            warmup=args.warmup,
        )
        for model, threads, chunk, sil, pre in itertools.product(
            args.model, args.threads, args.chunk_ms, args.min_silence_ms, args.pre_roll_ms
        )
    ]
    env = environment()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    suffix = f"_lim{args.limit}" if args.limit else ""
    summaries: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name}", file=sys.stderr)
        res = summarize_run(cfg, run_one(cfg), env)
        out = args.results_dir / f"{today}_{cfg.name}{suffix}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_detail(res)
        print(f"結果：{out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
        summaries.append({k: res[k] for k in ("name", "config", "metrics")})

    print("-" * 72)
    print(TABLE_HEADER)
    print("|" + "---|" * (TABLE_HEADER.count("|") - 1))
    for s in summaries:
        print(table_row(s))
    if len(summaries) > 1:
        tag = f"_{args.tag}" if args.tag else ""
        out = args.results_dir / f"{today}_summary{tag}{suffix}.json"
        out.write_text(
            json.dumps({"environment": env, "runs": summaries}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"總表：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
