"""全鏈路評測：VAD + 串流 ASR 在評測集上的 CER / RTF / 首字延遲 / 峰值記憶體 / 模型大小 / 切段數。

用法：
    uv run scripts/prepare_eval_data.py      # 先準備資料（一次即可）：AISHELL-1 test 隨機 3000 句
    uv run eval/run_eval.py                  # 預設模型與 pipeline 設定（兩段門檻 A2），3000 句正式評測集
    uv run eval/run_eval.py --min-silence-ms 500 --pause-ms 0   # 舊的單一門檻 500 ms
    uv run eval/run_eval.py --smoke          # 快速冒煙：固定 seed 抽樣的 200 句子集
    uv run eval/run_eval.py --jobs 4         # 4 個子程序並行（RTF / 峰值 RSS 不是正式值）
    uv run eval/run_eval.py --model zipformer-zh-14m --pre-roll-ms 200
    # 參數掃描：多個值取笛卡兒積，每個組合各存一份結果，最後印出總表
    uv run eval/run_eval.py --model x-asr-480ms-int8 zipformer-zh-14m \\
        --min-silence-ms 100 300 500 800 --pause-ms 0 --pre-roll-ms 0 100 200 300 --tag sweep
    # SenseVoice 熱詞（只改解碼）：beam × 加分掃描
    uv run eval/run_eval.py --manifest data/aishell1_align_300/manifest.jsonl \
        --model sensevoice-small-int8-ort --hotwords hotwords.txt --beam 4 8 --hotword-score 1 2

每個組合在獨立子程序中執行（模型載入、推論、量峰值記憶體），父程序只負責 CER 與彙整，
避免評測框架本身（jiwer、OpenCC）的記憶體算進去。結果存成 eval/results/<日期>_<設定名>.json，
內含完整設定、機器資訊、commit、彙總指標（CER 附 95% bootstrap 信賴區間）與逐句結果。
指標定義見 docs/eval.md。
"""

import argparse
import itertools
import json
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = REPO_ROOT / "data" / "aishell1_test_3000" / "manifest.jsonl"
SMOKE_MANIFEST = REPO_ROOT / "data" / "aishell1_test_200" / "manifest.jsonl"
BOOTSTRAP_RESAMPLES = 10000
RESULTS_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_LEAD_SILENCE_MS = 1000
DEFAULT_TAIL_SILENCE_MS = 1500


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
    lead_silence_ms: int = DEFAULT_LEAD_SILENCE_MS  # 每條音訊前補的數位靜音（端到端首字延遲的起點）
    blank_penalty: float = 0.0  # ASR 解碼的 blank penalty（0 = 不調整）
    # 每條音訊後補的數位靜音：句尾靜音夠長，VAD 才會以靜音計時（而非音訊結束）收尾，尾字延遲才不被截斷
    tail_silence_ms: int = 0
    # VAD 停頓事件的靜音長度：滿此值先輸出暫定結果，滿 min_silence 才定稿（0 = 不用暫定結果）
    pause_ms: int = 0
    provisional: str = "redecode"  # 暫定結果的做法：redecode / finalize（見 pipeline 模組說明）
    vad_neg_threshold: float | None = None  # None = threshold - 0.15（VadConfig 預設）
    # SenseVoice（onnxruntime 版）解碼：beam 1 且沒有熱詞時是 greedy；hotwords 是熱詞檔路徑（"" = 不用）
    beam: int = 1
    hotwords: str = ""
    hotword_score: float = 1.5

    @property
    def name(self) -> str:
        lead = f"_lead{self.lead_silence_ms}" if self.lead_silence_ms else ""
        lead += f"_tail{self.tail_silence_ms}" if self.tail_silence_ms else ""
        lead += f"_pause{self.pause_ms}" if self.pause_ms else ""
        lead += "fin" if self.pause_ms and self.provisional == "finalize" else ""
        lead += f"_bp{self.blank_penalty:g}" if self.blank_penalty else ""
        lead += f"_th{self.vad_threshold:g}" if self.vad_threshold != 0.5 else ""
        lead += f"_neg{self.vad_neg_threshold:g}" if self.vad_neg_threshold is not None else ""
        lead += f"_beam{self.beam}" if self.beam != 1 else ""
        lead += f"_hw-{Path(self.hotwords).stem}{self.hotword_score:g}" if self.hotwords else ""
        return (
            f"{self.model}_sil{self.vad_min_silence_ms}_pre{self.pre_roll_ms}"
            f"_chunk{self.chunk_ms}_t{self.threads}{lead}"
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


def worker(cfg: RunConfig, shard: int = 0, num_shards: int = 1) -> dict[str, Any]:
    """跑 manifest 中第 `shard` 份（共 `num_shards` 份，連續切分）的句子。"""
    import numpy as np

    from src.asr import SAMPLE_RATE, DecodeOptions, load_asr
    from src.cli import read_hotwords
    from src.pipeline import PipelineConfig, transcribe
    from src.vad import SileroVadModel, VadConfig, load_wav

    rss_baseline = _rss_mb()
    manifest_path = Path(cfg.manifest)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows = rows[len(rows) * shard // num_shards : len(rows) * (shard + 1) // num_shards]
    pipe_cfg = PipelineConfig(
        vad=VadConfig(
            threshold=cfg.vad_threshold,
            neg_threshold=cfg.vad_neg_threshold,
            min_speech_ms=cfg.vad_min_speech_ms,
            min_silence_ms=cfg.vad_min_silence_ms,
            speech_pad_ms=cfg.vad_speech_pad_ms,
            pause_ms=cfg.pause_ms or None,
        ),
        pre_roll_ms=cfg.pre_roll_ms,
        provisional="finalize" if cfg.provisional == "finalize" else "redecode",
    )

    t0 = time.perf_counter()
    vad_model = SileroVadModel(num_threads=cfg.threads)
    decode = DecodeOptions(
        blank_penalty=cfg.blank_penalty,
        beam=cfg.beam,
        hotwords=read_hotwords(Path(cfg.hotwords) if cfg.hotwords else None),
        hotword_score=cfg.hotword_score,
    )
    asr = load_asr(cfg.model, num_threads=cfg.threads, decode=decode)
    load_s = time.perf_counter() - t0
    rss_loaded = _rss_mb()
    files = [vad_model.model_path, *asr.model_files()]

    lead = cfg.lead_silence_ms * SAMPLE_RATE // 1000
    lead_s = lead / SAMPLE_RATE
    tail = cfg.tail_silence_ms * SAMPLE_RATE // 1000

    def shift(t: float | None) -> float | None:
        return None if t is None else t - lead_s

    def run(path: Path) -> tuple[float, float, list[Any]]:
        audio = load_wav(path)  # 讀檔與補靜音不計時
        audio = np.concatenate(
            [np.zeros(lead, dtype=audio.dtype), audio, np.zeros(tail, dtype=audio.dtype)]
        )
        start = time.perf_counter()
        utts = transcribe(audio, vad_model, asr, pipe_cfg, chunk_ms=cfg.chunk_ms)
        return len(audio) / SAMPLE_RATE, time.perf_counter() - start, utts

    for row in rows[: cfg.warmup]:
        run(manifest_path.parent / row["wav"])
    asr.reset_compute_stats()  # 計算量統計不含暖機（與 RTF 不同，暖機句子不重複計入）

    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        duration, compute, utts = run(manifest_path.parent / row["wav"])
        results.append(
            {
                "id": row["id"],
                "ref": row["text"],
                "hyp": " ".join(u.text for u in utts if u.text),
                "duration_s": duration - lead_s - tail / SAMPLE_RATE,  # 原始音訊長度
                "processed_s": duration,  # 實際送入 pipeline 的長度（含補靜音），RTF 的分母
                "compute_s": compute,
                # 連續多句集（prepare_continuous_data.py）才有：每句的真實起訖點與句間停頓
                **{k: row[k] for k in ("sentences", "pauses_s") if k in row},
                # 位置與時刻都換回原始音訊的時間軸（補靜音結束 = 0）
                "segments": [
                    {
                        "text": u.text,
                        "speech_start_s": u.speech_start_s - lead_s,
                        "speech_end_s": None if u.speech_end_s is None else u.speech_end_s - lead_s,
                        "vad_start_delay_s": u.start_emitted_s - u.speech_start_s,
                        "first_token_latency_s": u.first_token_latency_s,
                        "e2e_first_token_latency_s": (
                            None if u.first_token_s is None else u.first_token_s - lead_s
                        ),
                        "first_token_audio_s": (
                            None
                            if u.first_token_audio_s is None
                            else u.first_token_audio_s - lead_s
                        ),
                        "tail_latency_s": u.tail_latency_s,
                        # 模擬即時的時刻，換回原始音訊時間軸：VAD 結束事件、最終文字就緒、最終結果
                        "end_emitted_s": shift(u.end_emitted_s),
                        "final_text_ready_s": shift(u.final_text_ready_s),
                        "final_s": shift(u.final_s),
                        "num_partials": len(u.partials),
                        # 每次 partial 變化的時刻與文字（穩定度指標，eval/stability.py）
                        "partials": [
                            {"emitted_s": p.emitted_s - lead_s, "text": p.text} for p in u.partials
                        ],
                        "provisionals": [
                            {
                                "emitted_s": p.emitted_s - lead_s,
                                "audio_end_s": p.audio_end_s - lead_s,
                                "text": p.text,
                            }
                            for p in u.provisionals
                        ],
                        "resume_redecode_s": u.resume_redecode_s,
                    }
                    for u in utts
                ],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {cfg.name}: {i + 1}/{len(rows)}", file=sys.stderr)

    return {
        "asr_name": asr.name,
        "asr_compute": asr.compute_stats(),
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


def _speaker(uid: str) -> str | None:
    """AISHELL 句子 ID（如 BAC009S0764W0121）中的說話人 S0764；其他格式回傳 None。"""
    m = re.search(r"S\d{4}", uid)
    return m.group(0) if m else None


def cer_confidence(utts: list[dict[str, Any]]) -> dict[str, Any]:
    """CER 的 95% 百分位 bootstrap 信賴區間：以句重抽，以及以說話人重抽（ID 無法解析時為 None）。"""
    from eval.metrics import bootstrap_cer_ci, utterance_counts

    errors, refs = utterance_counts([u["ref"] for u in utts], [u["hyp"] for u in utts])
    speakers = [_speaker(u["id"]) for u in utts]
    by_speaker = None
    if all(speakers) and len(set(speakers)) > 1:
        by_speaker = list(
            bootstrap_cer_ci(errors, refs, [s or "" for s in speakers], BOOTSTRAP_RESAMPLES)
        )
    return {
        "level": 0.95,
        "method": "percentile bootstrap",
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": 0,
        "utterance": list(bootstrap_cer_ci(errors, refs, None, BOOTSTRAP_RESAMPLES)),
        "speaker": by_speaker,
        "num_speakers": len(set(speakers)) if by_speaker else None,
    }


def summarize_run(
    cfg: RunConfig, raw: dict[str, Any], env: dict[str, Any], jobs: int = 1
) -> dict[str, Any]:
    from eval.metrics import corpus_cer, summarize
    from eval.text_norm import normalize
    from src.vad import VAD_MODEL_NAME

    utts = raw["utterances"]
    cer = corpus_cer([u["ref"] for u in utts], [u["hyp"] for u in utts])
    total_audio = sum(u["duration_s"] for u in utts)
    total_processed = sum(u.get("processed_s", u["duration_s"]) for u in utts)
    total_compute = sum(u["compute_s"] for u in utts)

    first_token_ms: list[float] = []  # 每條音訊一個值：第一個有字的句段
    e2e_first_token_ms: list[float] = []  # 同一句段，起點改為補靜音結束
    tail_ms: list[float] = []  # 每條音訊一個值：最後一個句段
    # 同一句段：最終文字就緒 → 最終結果（= 若一就緒就輸出，最終結果可提早多少）
    ready_lead_ms: list[float] = []
    ready_before_end = 0  # 最終文字在 VAD 結束事件前已出現在 partial 中的句數
    vad_delay_ms: list[float] = []
    seg_counts: list[int] = []
    for u in utts:
        segs = u["segments"]
        seg_counts.append(len(segs))
        first = _first_with_text(segs)
        if first is not None and first["first_token_latency_s"] is not None:
            first_token_ms.append(first["first_token_latency_s"] * 1000)
            vad_delay_ms.append(first["vad_start_delay_s"] * 1000)
            if first.get("e2e_first_token_latency_s") is not None:
                e2e_first_token_ms.append(first["e2e_first_token_latency_s"] * 1000)
        if segs and segs[-1]["tail_latency_s"] is not None:
            tail_ms.append(segs[-1]["tail_latency_s"] * 1000)
        last = segs[-1] if segs else None
        if last is not None and last.get("final_text_ready_s") is not None and last["text"]:
            ready_lead_ms.append((last["final_s"] - last["final_text_ready_s"]) * 1000)
            ready_before_end += last["final_text_ready_s"] < last["end_emitted_s"]
        u["ref_norm"] = normalize(u["ref"])
        u["hyp_norm"] = normalize(u["hyp"])

    seg_hist: dict[str, int] = {}
    for n in seg_counts:
        key = str(n) if n < 3 else "3+"
        seg_hist[key] = seg_hist.get(key, 0) + 1

    from eval.segmentation import summarize_provisionals
    from eval.stability import summarize_stability

    segmentation = None
    if utts and all("sentences" in u for u in utts):
        from eval.segmentation import summarize_segmentation

        segmentation = summarize_segmentation(utts)

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
            # 同一設定的句子切成幾份、各在一個子程序中跑（> 1 時 RTF / 峰值 RSS 不是正式值）
            "num_shards": raw.get("num_shards", 1),
            "parallel_jobs": jobs,
        },
        "environment": env,
        "metrics": {
            "num_utterances": len(utts),
            "audio_s": total_audio,
            "cer": cer.cer,
            "cer_ci95": cer_confidence(utts),
            "cer_counts": asdict(cer),
            "rtf": total_compute / total_processed,
            # 後端提供時：重新辨識的次數與音訊總長，及其相對送入音訊（含補靜音）的比例，與機器負載無關
            "asr_compute": (
                None
                if raw.get("asr_compute") is None
                else {
                    **raw["asr_compute"],
                    "decoded_audio_ratio": raw["asr_compute"]["decoded_audio_s"] / total_processed,
                }
            ),
            "first_token_latency_ms": summarize(first_token_ms),
            "e2e_first_token_latency_ms": summarize(e2e_first_token_ms),
            "no_text_utterances": len(utts) - len(first_token_ms),
            "vad_start_delay_ms": summarize(vad_delay_ms),
            "tail_latency_ms": summarize(tail_ms),
            "final_text_ready_lead_ms": summarize(ready_lead_ms),
            "final_text_ready_before_end": ready_before_end,
            "segments_per_utterance": {
                "mean": sum(seg_counts) / len(seg_counts),
                "max": max(seg_counts),
                "histogram": dict(sorted(seg_hist.items())),
            },
            "segmentation": segmentation,
            "provisional": summarize_provisionals(utts),
            "stability": summarize_stability(utts),
            "memory_mb": raw["memory_mb"],
            "model_size_mb": model_bytes / 2**20,
            "load_s": raw["load_s"],
        },
        "model_files": files,
        "utterances": utts,
    }


def run_shard(cfg: RunConfig, shard: int, num_shards: int) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps({"config": asdict(cfg), "shard": shard, "num_shards": num_shards}),
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


def merge_shards(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """合併同一設定各份的結果：逐句結果依 manifest 順序串接；記憶體與載入時間取各份最大值。"""
    first = parts[0]
    return {
        "asr_name": first["asr_name"],
        "asr_compute": (
            None
            if first.get("asr_compute") is None
            else {k: sum(p["asr_compute"][k] for p in parts) for k in first["asr_compute"]}
        ),
        "load_s": max(p["load_s"] for p in parts),
        "memory_mb": {k: max(p["memory_mb"][k] for p in parts) for k in first["memory_mb"]},
        "model_files": first["model_files"],
        "utterances": [u for p in parts for u in p["utterances"]],
        "num_shards": len(parts),
    }


def run_all(configs: list[RunConfig], jobs: int) -> Iterator[tuple[RunConfig, dict[str, Any]]]:
    """以最多 `jobs` 個子程序並行執行；設定數少於 jobs 時把每個設定的句子切成多份。依設定順序產出結果。"""
    from concurrent.futures import ThreadPoolExecutor

    shards = max(1, -(-jobs // len(configs)))  # ceil
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            [pool.submit(run_shard, cfg, i, shards) for i in range(shards)] for cfg in configs
        ]
        for cfg, fs in zip(configs, futures, strict=True):
            yield cfg, merge_shards([f.result() for f in fs])


# ---------- 輸出 ----------


def _fmt(x: float | None, spec: str = ".0f") -> str:
    return "-" if x is None else format(x, spec)


def _ci(ci: list[float] | None) -> str:
    return "-" if ci is None else f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]"


TABLE_HEADER = (
    "| model | min_sil | pre_roll | CER % | CER 95% CI (utt) | RTF | first-token mean / P90 (ms) "
    "| e2e first-token mean / P90 (ms) "
    "| tail mean (ms) | seg/utt (max) | peak RSS (MB) | model (MB) |"
)


def table_row(res: dict[str, Any]) -> str:
    c, m = res["config"], res["metrics"]
    ft, seg = m["first_token_latency_ms"], m["segments_per_utterance"]
    e2e = m.get("e2e_first_token_latency_ms") or {"mean": None, "p90": None}
    bp = c.get("blank_penalty") or 0
    model = f"{c['model']} bp={bp:g}" if bp else c["model"]
    model += f" beam={c['beam']}" if c.get("beam", 1) != 1 else ""
    model += f" hw={c['hotword_score']:g}" if c.get("hotwords") else ""
    return (
        f"| {model} | {c['vad_min_silence_ms']} | {c['pre_roll_ms']} "
        f"| {m['cer'] * 100:.2f} | {_ci(m['cer_ci95']['utterance'])} | {m['rtf']:.4f} "
        f"| {_fmt(ft['mean'])} / {_fmt(ft['p90'])} "
        f"| {_fmt(e2e['mean'])} / {_fmt(e2e['p90'])} "
        f"| {_fmt(m['tail_latency_ms']['mean'])} | {seg['mean']:.2f} ({seg['max']}) "
        f"| {m['memory_mb']['peak_rss']:.0f} | {m['model_size_mb']:.1f} |"
    )


def print_detail(res: dict[str, Any]) -> None:
    from src.cli import display_width

    c, m, env = res["config"], res["metrics"], res["environment"]
    ft, tail, cc = m["first_token_latency_ms"], m["tail_latency_ms"], m["cer_counts"]
    e2e = m["e2e_first_token_latency_ms"]
    seg, ci = m["segments_per_utterance"], m["cer_ci95"]
    vad = (
        f"threshold={c['vad_threshold']} neg_threshold={c.get('vad_neg_threshold')}"
        f" min_speech={c['vad_min_speech_ms']}ms"
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
    e2e_s = (
        f"mean {_fmt(e2e['mean'])}  P50 {_fmt(e2e['p50'])}  P90 {_fmt(e2e['p90'])}"
        f"  P99 {_fmt(e2e['p99'])}（起點：前補 {c['lead_silence_ms']} ms 靜音的結束點）"
    )
    ci_s = (
        f"以句重抽 {_ci(ci['utterance'])}  以說話人重抽 {_ci(ci['speaker'])}"
        f"（{ci['num_speakers'] or '-'} 位）"
    )
    ready = m.get("final_text_ready_lead_ms") or {"mean": None, "p50": None, "n": 0}
    ready_s = (
        f"mean {_fmt(ready['mean'])}  P50 {_fmt(ready['p50'])}"
        f"（VAD 結束前已就緒 {m.get('final_text_ready_before_end', '-')} / {ready['n']} 句）"
    )
    jobs, shards = c.get("parallel_jobs", 1), c.get("num_shards", 1)
    parallel = (
        "1 個子程序（RTF / 峰值 RSS 為正式值）"
        if jobs == 1
        else f"{jobs} 個子程序並行，本設定切 {shards} 份（RTF / 峰值 RSS 非正式值）"
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
        ("並行", parallel),
        ("前 / 後補靜音", f"{c['lead_silence_ms']} / {c.get('tail_silence_ms', 0)} ms"),
        (
            "環境",
            f"{env['chip']} ({env['machine']}) macOS {env['macos']} commit {env['commit']}{dirty}",
        ),
        ("句數 / 音訊", f"{m['num_utterances']} 句 / {m['audio_s'] / 60:.1f} 分鐘"),
        ("CER", f"{m['cer'] * 100:.2f}%  ({counts})"),
        ("  95% 信賴區間", ci_s),
        ("RTF", f"{m['rtf']:.4f}"),
        ("端到端首字延遲 (ms)", e2e_s),
        ("首字延遲 VAD 起點 (ms)", ftl),
        ("  其中 VAD 觸發 (ms)", f"mean {_fmt(m['vad_start_delay_ms']['mean'])}"),
        ("尾字延遲 (ms)", f"mean {_fmt(tail['mean'])}  P90 {_fmt(tail['p90'])}"),
        ("  最終文字提早就緒 (ms)", ready_s),
        ("每條切段數", f"mean {seg['mean']:.2f}  max {seg['max']}  分佈 {seg['histogram']}"),
        ("記憶體 (MB)", mem_s),
        ("模型大小 (MB)", f"{m['model_size_mb']:.1f}"),
    ]
    pv = m.get("provisional")
    if pv is not None:
        rd = pv["redecode_ms"]
        lines.append(
            (
                "暫定結果",
                (
                    f"{pv['provisionals']} 個（被取代 {pv['superseded']}），最後一個與最終結果相同 "
                    f"{pv['last_matches_final'] * 100:.1f}%，產生暫定結果 mean {_fmt(rd['mean'])} / "
                    f"P90 {_fmt(rd['p90'])} ms，恢復時重新解碼 {pv['resume_redecode_ms']['n']} 次 "
                    f"mean {_fmt(pv['resume_redecode_ms']['mean'])} / "
                    f"P90 {_fmt(pv['resume_redecode_ms']['p90'])} ms"
                ),
            )
        )
    st = m.get("stability")
    if st is not None:
        sd = st["stable_delay_ms"]
        lines.append(
            (
                "partial 穩定度",
                (
                    f"修正 {st['unstable_update_rate'] * 100:.1f}% 的更新，被改掉的字 / 最終字數 "
                    f"{st['revised_token_ratio'] * 100:.1f}%（每條 mean "
                    f"{st['revised_tokens_per_utterance']['mean']:.2f}），前綴穩定延遲 mean "
                    f"{_fmt(sd['mean'])} / P90 {_fmt(sd['p90'])} ms"
                ),
            )
        )
    sg = m.get("segmentation")
    if sg is not None:
        by = sg["boundary_detected_by_pause"]
        tails = sg["tail_latency_ms_by_pause"]
        ft = sg["first_token_latency_true_onset_ms"]
        lines += [
            (
                "斷句（連續多句）",
                (
                    f"{sg['files']} 條 / {sg['sentences']} 句，完全切對 {sg['exact_files']} 條，"
                    f"句中誤切 {sg['sentences_split_inside']} 句，漏句 {sg['sentences_missed']} 句"
                ),
            ),
            (
                "  句間切開率",
                "  ".join(f"{k}s {v['rate'] * 100:.0f}%" for k, v in by.items()),
            ),
            (
                "  每句尾字 mean (ms)",
                "  ".join(
                    f"{k}{'s' if k != 'end' else ''} {_fmt(v['mean'])}" for k, v in tails.items()
                ),
            ),
            (
                "  每句結果 mean (ms)",
                "  ".join(
                    f"{k}{'s' if k != 'end' else ''} {_fmt(v['mean'])}"
                    for k, v in sg["result_latency_ms_by_pause"].items()
                ),
            ),
            (
                "  首字（真實起點, ms）",
                f"mean {_fmt(ft['mean'])}  P90 {_fmt(ft['p90'])}  (n={ft['n']})",
            ),
        ]
    width = max(display_width(k) for k, _ in lines)
    print("-" * 72)
    for k, v in lines:
        print(f"{k}{' ' * (width - display_width(k))}  {v}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.cli import add_decode_args, add_pipeline_args, non_negative_int, positive_int
    from src.pipeline import PipelineConfig

    p = argparse.ArgumentParser(
        description="全鏈路評測（VAD + 串流 ASR）。多個值的參數取笛卡兒積做掃描。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_mutually_exclusive_group()
    data.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="評測集 manifest")
    data.add_argument(
        "--smoke",
        dest="manifest",
        action="store_const",
        const=SMOKE_MANIFEST,
        help=f"改用 200 句子集快速冒煙（{SMOKE_MANIFEST.relative_to(REPO_ROOT)}）",
    )
    add_pipeline_args(p, sweep=True)
    add_decode_args(p, sweep=True)
    p.add_argument(
        "--warmup", type=non_negative_int, default=3, help="正式計時前先跑幾句暖機（不計入）"
    )
    p.add_argument(
        "--limit", type=positive_int, help="只評測前 N 句（除錯用，結果名稱會加上 _limN）"
    )
    p.add_argument(
        "--lead-silence-ms",
        type=non_negative_int,
        default=DEFAULT_LEAD_SILENCE_MS,
        help="每條音訊前補的靜音長度；端到端首字延遲以補靜音的結束點為起點（0 = 不補，結果名稱不加 _lead）",
    )
    p.add_argument(
        "--tail-silence-ms",
        type=non_negative_int,
        default=DEFAULT_TAIL_SILENCE_MS,
        help="每條音訊後補的靜音長度，讓句子以 VAD 靜音計時收尾、尾字延遲不被音訊結束截斷"
        "（0 = 不補，結果名稱不加 _tail）",
    )
    p.add_argument(
        "--blank-penalty",
        type=float,
        nargs="+",
        default=[0.0],
        help="ASR 解碼時從 blank 減去的分數（可給多個值掃描；非 0 時結果名稱加 _bp<值>）",
    )
    p.add_argument(
        "--jobs",
        type=positive_int,
        default=1,
        help="並行的子程序數（每個仍是 --threads 執行緒）。設定數少於此值時把句子切成多份。"
        "> 1 時 CPU 互搶、可能排到效能核心以外，RTF 與峰值 RSS 不能當正式值；CER 不受影響",
    )
    defaults = PipelineConfig()
    p.add_argument(
        "--pause-ms",
        type=non_negative_int,
        nargs="+",
        default=[defaults.vad.pause_ms or 0],
        help="靜音滿此值先輸出暫定結果、滿 --min-silence-ms 才定稿（0 = 不用暫定結果；可給多個值掃描，"
        "不小於 min_silence 的組合略過；非 0 時結果名稱加 _pause<值>）",
    )
    p.add_argument(
        "--provisional",
        nargs="+",
        choices=["redecode", "finalize"],
        default=[defaults.provisional],
        help="暫定結果的做法（--pause-ms 非 0 時）：redecode 停頓時重新解碼整句；finalize 停頓時 "
        "finalize 原 stream、恢復說話時才重新解碼（結果名稱加 fin）。可給多個值掃描。"
        "退回舊的單一門檻：--min-silence-ms 500 --pause-ms 0",
    )
    p.add_argument("--tag", default="", help="掃描總表檔名的標籤")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="結果輸出目錄")
    return p.parse_args(argv)


def main() -> int:
    if "--worker" in sys.argv[1:]:
        job = json.loads(sys.stdin.read())
        print(json.dumps(worker(RunConfig(**job["config"]), job["shard"], job["num_shards"])))
        return 0

    args = parse_args()
    if not args.manifest.is_file():
        hint = " -n 200" if args.manifest == SMOKE_MANIFEST else ""
        print(
            f"錯誤：找不到 {args.manifest}，請先執行 uv run scripts/prepare_eval_data.py{hint}",
            file=sys.stderr,
        )
        return 1
    manifest = args.manifest.resolve()
    dataset = manifest.parent.name  # 結果檔名前綴，不同評測集的結果不互相覆寫
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

    # pause 0（不用暫定結果）時 provisional 沒有作用，統一記成 redecode 並去重
    # 沒有熱詞時 hotword_score 沒有作用，統一記成預設值並去重
    from src.asr import DecodeOptions, default_threads

    default_score = DecodeOptions().hotword_score
    combos = dict.fromkeys(
        (
            model,
            threads_arg or default_threads(model),
            chunk,
            sil,
            pre,
            bp,
            pause,
            mode if pause else "redecode",
            beam,
            score if args.hotwords else default_score,
        )
        for model, threads_arg, chunk, sil, pre, bp, pause, mode, beam, score in itertools.product(
            args.model,
            args.threads,
            args.chunk_ms,
            args.min_silence_ms,
            args.pre_roll_ms,
            args.blank_penalty,
            args.pause_ms,
            args.provisional,
            args.beam,
            args.hotword_score,
        )
        if not pause or pause < sil
    )
    configs = [
        RunConfig(
            model=model,
            manifest=str(manifest),
            threads=threads,
            chunk_ms=chunk,
            pre_roll_ms=pre,
            vad_threshold=args.threshold,
            vad_neg_threshold=args.neg_threshold,
            vad_min_speech_ms=args.min_speech_ms,
            vad_min_silence_ms=sil,
            vad_speech_pad_ms=args.speech_pad_ms,
            warmup=args.warmup,
            lead_silence_ms=args.lead_silence_ms,
            tail_silence_ms=args.tail_silence_ms,
            blank_penalty=bp,
            pause_ms=pause,
            provisional=mode,
            beam=beam,
            hotwords=str(args.hotwords.resolve()) if args.hotwords else "",
            hotword_score=score,
        )
        for model, threads, chunk, sil, pre, bp, pause, mode, beam, score in combos
    ]
    env = environment()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    suffix = f"_lim{args.limit}" if args.limit else ""
    summaries: list[dict[str, Any]] = []
    print(f"{len(configs)} 個設定，{args.jobs} 個子程序並行", file=sys.stderr)
    for cfg, raw in run_all(configs, args.jobs):
        res = summarize_run(cfg, raw, env, args.jobs)
        out = args.results_dir / f"{today}_{dataset}_{cfg.name}{suffix}.json"
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
        out = args.results_dir / f"{today}_{dataset}_summary{tag}{suffix}.json"
        out.write_text(
            json.dumps({"environment": env, "runs": summaries}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"總表：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
