"""scripts/vad_trigger_sweep.py：快取機率重播狀態機、句段轉換、觸發延遲與非語音區的機率。"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.audio import SAMPLE_RATE
from src.vad import StreamingVad, VadConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME = 512  # Silero 16 kHz 的 frame（32 ms）


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vad_trigger_sweep", REPO_ROOT / "scripts" / "vad_trigger_sweep.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load()


def _probs(speech: list[tuple[int, int]], n: int) -> list[float]:
    """n 個 frame，speech 區間（frame 索引，含頭不含尾）為 0.9，其餘 0.05。"""
    out = [0.05] * n
    for a, b in speech:
        out[a:b] = [0.9] * (b - a)
    return out


def test_replay_matches_streaming_vad_with_same_probs() -> None:
    # 語音 frame 20–40、靜音 40 frame（1.28 s）、再一段 frame 80–90
    probs = _probs([(20, 40), (80, 90)], 140)
    n = 140 * FRAME - 100  # 最後不足一個 frame，flush 補零
    cfg = VadConfig(min_speech_ms=250, min_silence_ms=1000, pause_ms=150)
    expected = StreamingVad(sweep.CachedProbs(probs, FRAME), cfg)
    events = expected.accept_waveform(np.zeros(n, np.float32)) + expected.flush()
    assert sweep.replay(probs, FRAME, cfg, n) == events
    assert [e.kind for e in events if e.kind in ("start", "end")] == [
        "start",
        "end",
        "start",
        "end",
    ]


@pytest.mark.parametrize(("min_speech", "frames_after_onset"), [(250, 8), (96, 3), (32, 1), (0, 1)])
def test_lower_min_speech_triggers_earlier(min_speech: int, frames_after_onset: int) -> None:
    probs = _probs([(20, 60)], 120)
    cfg = VadConfig(min_speech_ms=min_speech, min_silence_ms=1000)
    events = sweep.replay(probs, FRAME, cfg, 120 * FRAME)
    start = next(e for e in events if e.kind == "start")
    # 開始事件在第 20 個 frame 之後滿 min_speech 的那個 frame 結束時發出（ceil(ms / 32)，至少 1 個）
    assert start.detected_at == (20 + frames_after_onset) * FRAME


def test_isolated_blip_only_triggers_with_low_min_speech() -> None:
    # 2 frame（64 ms）的突發，前後各 1 s 以上靜音：min_speech 250 丟棄，0 時成為一個誤觸發句段
    probs = _probs([(40, 42)], 120)
    for ms, n_seg in ((250, 0), (64, 1), (0, 1)):
        cfg = VadConfig(min_speech_ms=ms, min_silence_ms=1000)
        segs = sweep.segments_from_events(sweep.replay(probs, FRAME, cfg, 120 * FRAME), 0.0)
        assert len(segs) == n_seg


def test_segments_and_trigger_stats() -> None:
    lead = 1.0
    sr = SAMPLE_RATE
    probs = _probs([(40, 80)], 150)  # 補靜音後 1.28–2.56 s → 原始時間軸 0.28–1.56 s
    cfg = VadConfig(min_speech_ms=0, min_silence_ms=1000, speech_pad_ms=30)
    segs = sweep.segments_from_events(sweep.replay(probs, FRAME, cfg, 150 * FRAME), lead)
    assert len(segs) == 1
    seg = segs[0]
    assert seg["speech_start_s"] == pytest.approx(40 * FRAME / sr - lead - 0.03)
    assert seg["start_emitted_s"] == pytest.approx(41 * FRAME / sr - lead)
    assert seg["speech_end_s"] == pytest.approx(80 * FRAME / sr - lead + 0.03)
    sentences = [{"start_s": 0.25, "end_s": 1.5}]
    trig, offset = sweep.trigger_stats(sentences, segs)
    assert trig == pytest.approx([(seg["start_emitted_s"] - 0.25) * 1000])
    assert offset == pytest.approx([(seg["speech_start_s"] - 0.25) * 1000])


def test_nonspeech_probs_excludes_padding_and_guard() -> None:
    frame_s = FRAME / SAMPLE_RATE
    lead = 10 * frame_s
    probs = [float(i) for i in range(40)]  # 機率 = frame 索引，方便檢查取到哪些
    # 原始音訊 25 frame（frame 10–34），句子在原始時間軸 frame 10–15
    sentences = [{"start_s": 10 * frame_s, "end_s": 15 * frame_s}]
    got = sweep.nonspeech_probs(probs, FRAME, sentences, lead, 25 * frame_s, guard_s=2 * frame_s)
    # 原始 frame 0–7 與 17–24（離句子至少 2 frame）→ 補靜音後 frame 10–17、27–34
    assert got == [float(i) for i in [*range(10, 18), *range(27, 35)]]
    margin = sweep.nonspeech_margin([0.1, 0.55, 0.7], [0.5, 0.6])
    assert margin["over"] == {"0.5": 2, "0.6": 1} and margin["max"] == pytest.approx(0.7)
