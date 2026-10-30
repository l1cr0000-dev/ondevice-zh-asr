"""語音活動偵測（Silero VAD，ONNX Runtime 後端）。

離線：`detect_segments(audio, model, config)` → `list[SpeechSegment]`
串流：`StreamingVad(model, config)`，`accept_waveform(chunk)` / `flush()` → `list[VadEvent]`
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

from src.vad.detector import (
    FrameClassifier,
    SpeechSegment,
    StreamingVad,
    VadConfig,
    VadEvent,
    detect_segments,
    events_to_segments,
)
from src.vad.silero import DEFAULT_MODEL_PATH, SAMPLE_RATE, SileroVadModel

__all__ = [
    "DEFAULT_MODEL_PATH",
    "SAMPLE_RATE",
    "FrameClassifier",
    "SileroVadModel",
    "SpeechSegment",
    "StreamingVad",
    "VadConfig",
    "VadEvent",
    "detect_segments",
    "events_to_segments",
    "load_wav",
]


def load_wav(path: str | Path) -> npt.NDArray[np.float32]:
    """讀取 16 kHz wav 為 float32 單聲道；多聲道取平均。不做重取樣，取樣率不符直接報錯。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} 取樣率為 {sr} Hz，VAD 需要 {SAMPLE_RATE} Hz，請先重取樣")
    audio: npt.NDArray[np.float32] = data.mean(axis=1)
    return audio
