"""VAD 觸發時機掃描（B2）：只跑 VAD，量 min_speech_ms × threshold 對觸發延遲、切段與誤觸發的影響。

Silero 的逐 frame 語音機率與狀態機參數無關，因此每條音訊只推論一次，把機率快取起來
（`CachedProbs`），再用不同的 `VadConfig` 重播狀態機。其餘 VAD 參數取 pipeline 預設（A2：
min_silence 1000、pause 150、pad 30）；前後補的數位靜音與 `run_eval.py` 相同（1000 / 1500 ms），
所以句段與 pipeline 跑出來的逐條相同。

指標（以 MFA 真實起訖點為準，`eval.segmentation`）：
- 觸發延遲：句子是某個句段的第一句時，VAD 發出「開始」事件的時刻 − 句子真實首字起點。pipeline 最早
  在這個時刻才建立 ASR stream。
- 起點偏移：句段起點（含 speech_pad_ms）− 真實首字起點（負 = 切在首字之前）。
- 句中誤切、漏句、句間切開率、誤觸發（與任何句子都不重疊的句段）。

對齊集（單句）用 `alignments.jsonl` 的首字起點 / 末字結束當作一句；連續集（含車內噪音版）用 manifest
的 `sentences`。

用法：
    uv run scripts/vad_trigger_sweep.py      # 預設：對齊 300 句 + 連續集 + 車內噪音 5 個 SNR
    uv run scripts/vad_trigger_sweep.py --min-speech-ms 250 0 --threshold 0.5 \\
        --manifest data/aishell1_continuous_300_tcar_snr-5/manifest.jsonl
"""

import argparse
import itertools
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import summarize
from eval.segmentation import match_file, summarize_segmentation
from src.audio import SAMPLE_RATE
from src.pipeline import PipelineConfig
from src.vad import StreamingVad, VadConfig, VadEvent

DATA = REPO_ROOT / "data"
DEFAULT_MANIFESTS = [
    DATA / "aishell1_align_300" / "manifest.jsonl",
    DATA / "aishell1_continuous_300" / "manifest.jsonl",
    *(
        DATA / f"aishell1_continuous_300_tcar_snr{snr}" / "manifest.jsonl"
        for snr in (20, 10, 5, 0, -5)
    ),
]
LEAD_MS = 1000
TAIL_MS = 1500
GUARD_S = 0.2  # 非語音區離句子真實起訖點的距離（避開 MFA 邊界誤差與首尾的弱音）


class CachedProbs:
    """依序回傳預先算好的逐 frame 語音機率（`FrameClassifier`）；`reset()` 從頭重播。"""

    def __init__(self, probs: Sequence[float], frame_size: int) -> None:
        self._probs = list(probs)
        self._frame_size = frame_size
        self._i = 0

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def __call__(self, frame: npt.NDArray[np.float32]) -> float:
        prob = self._probs[self._i]
        self._i += 1
        return prob

    def reset(self) -> None:
        self._i = 0


def frame_probs(audio: npt.NDArray[np.float32], model: Any) -> list[float]:
    """整段音訊的逐 frame 機率（最後不足一個 frame 的部分補零，與 `StreamingVad.flush` 相同）。"""
    model.reset()
    size = model.frame_size
    n = -(-len(audio) // size)
    padded = np.zeros(n * size, dtype=np.float32)
    padded[: len(audio)] = audio
    return [float(model(padded[i * size : (i + 1) * size])) for i in range(n)]


def replay(
    probs: Sequence[float], frame_size: int, config: VadConfig, n_samples: int
) -> list[VadEvent]:
    vad = StreamingVad(CachedProbs(probs, frame_size), config)
    whole = (n_samples // frame_size) * frame_size
    events = vad.accept_waveform(np.zeros(whole, dtype=np.float32))
    if n_samples > whole:
        events += vad.accept_waveform(np.zeros(n_samples - whole, dtype=np.float32))
    return events + vad.flush()


def segments_from_events(events: Sequence[VadEvent], offset_s: float) -> list[dict[str, Any]]:
    """事件 → `eval.segmentation` 的句段格式（時間軸減去 `offset_s`，即前補靜音的長度）。"""
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.kind == "start":
            out.append(
                {
                    "speech_start_s": ev.sample / SAMPLE_RATE - offset_s,
                    "speech_end_s": None,
                    "start_emitted_s": ev.detected_at / SAMPLE_RATE - offset_s,
                }
            )
        elif ev.kind == "end" and out:
            out[-1]["speech_end_s"] = ev.sample / SAMPLE_RATE - offset_s
    return out


def trigger_stats(
    sentences: Sequence[dict[str, Any]], segments: Sequence[dict[str, Any]]
) -> tuple[list[float], list[float]]:
    """句子是句段第一句時：（觸發延遲 ms、起點偏移 ms），各一個值。"""
    trig: list[float] = []
    offset: list[float] = []
    for s, r in zip(sentences, match_file(sentences, segments), strict=True):
        if r["starts_segment"]:
            seg = segments[r["segments"][0]]
            trig.append((seg["start_emitted_s"] - s["start_s"]) * 1000)
            offset.append((seg["speech_start_s"] - s["start_s"]) * 1000)
    return trig, offset


def nonspeech_probs(
    probs: Sequence[float],
    frame_size: int,
    sentences: Sequence[dict[str, Any]],
    lead_s: float,
    duration_s: float,
    guard_s: float = GUARD_S,
) -> list[float]:
    """原始音訊中離每句真實起訖點都至少 `guard_s` 的 frame 的語音機率（不含前後補的數位靜音）。"""
    frame_s = frame_size / SAMPLE_RATE
    out: list[float] = []
    for i, prob in enumerate(probs):
        start = i * frame_s - lead_s
        end = start + frame_s
        if start < 0 or end > duration_s:
            continue
        if all(end <= s["start_s"] - guard_s or start >= s["end_s"] + guard_s for s in sentences):
            out.append(prob)
    return out


def nonspeech_margin(values: Sequence[float], thresholds: Sequence[float]) -> dict[str, Any]:
    """非語音 frame 的機率分布：P99 / 最大值，與超過各閾值的 frame 數（每個都可能單獨觸發）。"""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "frames": len(arr),
        "p99": float(np.percentile(arr, 99)) if len(arr) else None,
        "max": float(arr.max()) if len(arr) else None,
        "over": {f"{th:g}": int((arr >= th).sum()) for th in thresholds},
    }


def load_files(manifest: Path) -> list[dict[str, Any]]:
    """manifest → 每條 {wav, sentences, pauses_s}；單句集的句子起訖取 alignments.jsonl。"""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    align_path = manifest.parent / "alignments.jsonl"
    spans: dict[str, tuple[float, float]] = {}
    if align_path.is_file():
        for line in align_path.read_text(encoding="utf-8").splitlines():
            a = json.loads(line)
            spans[a["id"]] = (a["chars"][0]["start"], a["chars"][-1]["end"])
    files = []
    for r in rows:
        if "sentences" in r:
            sentences, pauses = r["sentences"], r.get("pauses_s", [])
        else:
            start, end = spans[r["id"]]
            sentences, pauses = [{"start_s": start, "end_s": end}], []
        files.append(
            {"wav": manifest.parent / r["wav"], "sentences": sentences, "pauses_s": pauses}
        )
    return files


def sweep(manifest: Path, min_speech: Sequence[int], thresholds: Sequence[float]) -> dict[str, Any]:
    from src.vad import SileroVadModel, load_wav

    base = PipelineConfig().vad
    model = SileroVadModel(num_threads=1)
    lead = LEAD_MS * SAMPLE_RATE // 1000
    tail = TAIL_MS * SAMPLE_RATE // 1000
    cached = []
    background: list[float] = []
    for f in load_files(manifest):
        audio = load_wav(f["wav"])
        duration_s = len(audio) / SAMPLE_RATE
        audio = np.concatenate([np.zeros(lead, np.float32), audio, np.zeros(tail, np.float32)])
        probs = frame_probs(audio, model)
        cached.append((f, probs, len(audio)))
        background += nonspeech_probs(
            probs, model.frame_size, f["sentences"], LEAD_MS / 1000, duration_s
        )

    out: dict[str, Any] = {"nonspeech": nonspeech_margin(background, thresholds)}
    for th, ms in itertools.product(thresholds, min_speech):
        cfg = VadConfig(
            threshold=th,
            min_speech_ms=ms,
            min_silence_ms=base.min_silence_ms,
            speech_pad_ms=base.speech_pad_ms,
            pause_ms=base.pause_ms,
        )
        files, trig, offset = [], [], []
        for f, probs, n in cached:
            segs = segments_from_events(replay(probs, model.frame_size, cfg, n), LEAD_MS / 1000)
            files.append({**f, "segments": segs})
            t, o = trigger_stats(f["sentences"], segs)
            trig += t
            offset += o
        seg = summarize_segmentation(files)
        out[f"th{th:g}_ms{ms}"] = {
            "threshold": th,
            "min_speech_ms": ms,
            "trigger_latency_ms": summarize(trig),
            "start_offset_ms": summarize(offset),
            "segments": sum(len(f["segments"]) for f in files),
            **{
                k: seg[k]
                for k in (
                    "files",
                    "sentences",
                    "exact_files",
                    "boundary_detected",
                    "sentences_split_inside",
                    "sentences_missed",
                    "false_triggers",
                )
            },
        }
    return out


def _fmt(x: float | None, spec: str = ".0f") -> str:
    return "—" if x is None else format(x, spec)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, nargs="+", default=DEFAULT_MANIFESTS)
    p.add_argument("--min-speech-ms", type=int, nargs="+", default=[250, 160, 96, 64, 32, 0])
    p.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.6])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results")
    args = p.parse_args()

    report: dict[str, Any] = {}
    for manifest in args.manifest:
        name = manifest.parent.name
        res = sweep(manifest, args.min_speech_ms, args.threshold)
        report[name] = dict(res)
        print(f"\n### {name}\n")
        print(
            "| threshold | min_speech | 觸發延遲 mean / P50 / P90 (ms) | 起點偏移 mean (ms) | 句段 | "
            "句中誤切 | 漏句 | 句間切開率 | 誤觸發 |"
        )
        print("|" + "---|" * 9)
        bg = res.pop("nonspeech")
        print(
            f"非語音區（離句子 ≥ {GUARD_S:g} s）{bg['frames']} 個 frame：機率 P99 {_fmt(bg['p99'], '.3f')}"
            f"、最大 {_fmt(bg['max'], '.3f')}；超過閾值的 frame "
            + "、".join(f"≥{k}：{v}" for k, v in bg["over"].items())
            + "\n"
        )
        for r in res.values():
            t = r["trigger_latency_ms"]
            b = r["boundary_detected"]
            print(
                f"| {r['threshold']:g} | {r['min_speech_ms']} | {_fmt(t['mean'])} / "
                f"{_fmt(t['p50'])} / {_fmt(t['p90'])} | {_fmt(r['start_offset_ms']['mean'])} | "
                f"{r['segments']} | {r['sentences_split_inside']} | {r['sentences_missed']} | "
                f"{'—' if b is None else f'{b * 100:.1f}%'} | {r['false_triggers']} |"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{date.today().isoformat()}_vad_trigger_sweep.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n數字存於 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
