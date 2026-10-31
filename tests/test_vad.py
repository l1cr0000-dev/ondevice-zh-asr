"""VAD 測試：真實模型的端到端行為 + 用假分類器驗證狀態機與參數。"""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from src.vad import (
    DEFAULT_MODEL_PATH,
    SAMPLE_RATE,
    SileroVadModel,
    SpeechSegment,
    StreamingVad,
    VadConfig,
    detect_segments,
    events_to_segments,
    load_wav,
)

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURES / "vad_two_utterances.wav"
# 起止誤差容忍度：frame 32 ms + padding 30 ms + TTS 首尾弱音，留一些餘裕
BOUNDARY_TOL_S = 0.1


@pytest.fixture(scope="module")
def model() -> SileroVadModel:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("VAD 模型不存在，先執行 bash scripts/download_models.sh")
    return SileroVadModel()


def stream(vad: StreamingVad, audio: npt.NDArray[np.float32], chunk: int) -> list[SpeechSegment]:
    vad.reset()
    events = []
    for i in range(0, len(audio), chunk):
        events += vad.accept_waveform(audio[i : i + chunk])
    events += vad.flush()
    assert [e.kind for e in events] == ["start", "end"] * (len(events) // 2)
    return events_to_segments(events)


# --- 真實模型 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(3 * SAMPLE_RATE, dtype=np.float32),
        # -60 dBFS 白噪音，接近安靜房間的底噪
        np.random.default_rng(0).normal(0, 1e-3, 3 * SAMPLE_RATE).astype(np.float32),
    ],
    ids=["zeros", "noise-60dB"],
)
def test_silence_has_no_output(model: SileroVadModel, audio: npt.NDArray[np.float32]) -> None:
    assert detect_segments(audio, model) == []
    assert stream(StreamingVad(model), audio, chunk=160) == []


def test_speech_segments_detected(model: SileroVadModel) -> None:
    truth = json.loads(SPEECH_WAV.with_suffix(".json").read_text())["segments"]
    segments = detect_segments(load_wav(SPEECH_WAV), model)

    assert len(segments) == len(truth)
    for seg, ref in zip(segments, truth, strict=True):
        assert seg.start_s == pytest.approx(ref["start"], abs=BOUNDARY_TOL_S)
        assert seg.end_s == pytest.approx(ref["end"], abs=BOUNDARY_TOL_S)


@pytest.mark.parametrize("chunk", [160, 512, 1000, 4096])
def test_streaming_matches_offline(model: SileroVadModel, chunk: int) -> None:
    audio = load_wav(SPEECH_WAV)
    offline = detect_segments(audio, model)
    assert stream(StreamingVad(model), audio, chunk) == offline


def test_streaming_event_latency(model: SileroVadModel) -> None:
    config = VadConfig()
    vad = StreamingVad(model, config)
    audio = load_wav(SPEECH_WAV)
    events = vad.accept_waveform(audio) + vad.flush()
    for ev in events:
        if ev.kind == "start":
            # 至少要等 min_speech_ms 才能確認開始
            assert ev.latency_s >= config.min_speech_ms / 1000
        assert ev.detected_at >= ev.sample


# --- 狀態機（假分類器，每個 frame 的機率預先指定）-------------------------------


FRAME = 512
FRAME_MS = FRAME * 1000 // SAMPLE_RATE  # 32


class ScriptedClassifier:
    frame_size = FRAME

    def __init__(self, probs: list[float]) -> None:
        self.probs = probs
        self.i = 0

    def __call__(self, frame: npt.NDArray[np.float32]) -> float:
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self) -> None:
        self.i = 0


def run(probs: list[float], config: VadConfig) -> list[tuple[int, int]]:
    """回傳以 frame 為單位的 (起, 迄)，padding 設 0 時邊界剛好落在 frame 上。"""
    audio = np.zeros(len(probs) * FRAME, dtype=np.float32)
    segs = detect_segments(audio, ScriptedClassifier(probs), config)
    return [(s.start // FRAME, s.end // FRAME) for s in segs]


NO_PAD = {"speech_pad_ms": 0}


def test_basic_segment() -> None:
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 10
    assert run(probs, VadConfig(**NO_PAD)) == [(5, 15)]


def test_min_speech_drops_short_burst() -> None:
    probs = [0.0] * 5 + [0.9] * 5 + [0.0] * 10  # 160 ms 語音
    assert run(probs, VadConfig(min_speech_ms=250, **NO_PAD)) == []
    assert run(probs, VadConfig(min_speech_ms=150, **NO_PAD)) == [(5, 10)]


def test_pause_event_once_per_pause_and_not_a_boundary() -> None:
    # 兩次 128 ms 靜音（各發一次停頓事件），最後長靜音結束；停頓不影響切段
    probs = [0.9] * 10 + [0.0] * 4 + [0.9] * 10 + [0.0] * 4 + [0.9] * 10 + [0.0] * 20
    config = VadConfig(min_silence_ms=300, pause_ms=96, **NO_PAD)
    audio = np.zeros(len(probs) * FRAME, dtype=np.float32)
    vad = StreamingVad(ScriptedClassifier(probs), config)
    events = vad.accept_waveform(audio) + vad.flush()
    assert [e.kind for e in events] == ["start", "pause", "pause", "pause", "end"]
    pauses = [e for e in events if e.kind == "pause"]
    assert [e.sample // FRAME for e in pauses] == [10, 24, 38]  # 停頓起點
    assert [e.detected_at // FRAME for e in pauses] == [13, 27, 41]  # 靜音滿 96 ms
    assert run(probs, config) == run(probs, VadConfig(min_silence_ms=300, **NO_PAD)) == [(0, 38)]


def test_pause_must_be_shorter_than_min_silence() -> None:
    with pytest.raises(ValueError):
        VadConfig(min_silence_ms=300, pause_ms=300)


def test_min_silence_merges_short_gap() -> None:
    probs = [0.9] * 10 + [0.0] * 3 + [0.9] * 10 + [0.0] * 10  # 中間 96 ms 靜音
    assert run(probs, VadConfig(min_silence_ms=200, **NO_PAD)) == [(0, 23)]
    assert run(probs, VadConfig(min_silence_ms=64, **NO_PAD)) == [(0, 10), (13, 23)]


def test_threshold_and_hysteresis() -> None:
    probs = [0.0] * 5 + [0.6] * 10 + [0.4] * 3 + [0.6] * 10 + [0.0] * 10
    # 0.4 落在 [neg_threshold=0.35, threshold=0.5) 之間：不算語音也不算靜音，不會切段
    assert run(probs, VadConfig(threshold=0.5, min_silence_ms=64, **NO_PAD)) == [(5, 28)]
    # neg_threshold 拉到 threshold，0.4 就被當成靜音而切成兩段
    split = VadConfig(threshold=0.5, neg_threshold=0.5, min_silence_ms=64, **NO_PAD)
    assert run(probs, split) == [(5, 15), (18, 28)]
    assert run(probs, VadConfig(threshold=0.7, **NO_PAD)) == []


def test_padding_clamped_to_audio_and_previous_segment() -> None:
    probs = [0.9] * 10 + [0.0] * 4 + [0.9] * 10
    config = VadConfig(min_silence_ms=100, speech_pad_ms=100)
    audio = np.zeros(len(probs) * FRAME, dtype=np.float32)
    first, second = detect_segments(audio, ScriptedClassifier(probs), config)
    assert first.start == 0
    assert second.start >= first.end
    assert second.end == len(audio)


def test_speech_until_end_is_closed_by_flush() -> None:
    assert run([0.0] * 5 + [0.9] * 10, VadConfig(**NO_PAD)) == [(5, 15)]


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 0.0}, {"threshold": 1.0}, {"neg_threshold": 0.8}, {"min_speech_ms": -1}],
)
def test_invalid_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        VadConfig(**kwargs)  # type: ignore[arg-type]


def test_load_wav_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "8k.wav"
    sf.write(str(path), np.zeros(8000, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="16000"):
        load_wav(path)
