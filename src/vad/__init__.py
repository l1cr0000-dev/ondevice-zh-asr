"""語音活動偵測（Silero VAD，ONNX Runtime 後端）。

離線：`detect_segments(audio, model, config)` → `list[SpeechSegment]`
串流：`StreamingVad(model, config)`，`accept_waveform(chunk)` / `flush()` → `list[VadEvent]`
"""

from src.audio import SAMPLE_RATE, load_wav
from src.vad.detector import (
    FrameClassifier,
    SpeechSegment,
    StreamingVad,
    VadConfig,
    VadEvent,
    detect_segments,
    events_to_segments,
)
from src.vad.silero import DEFAULT_MODEL_PATH, VAD_MODEL_NAME, SileroVadModel

__all__ = [
    "DEFAULT_MODEL_PATH",
    "SAMPLE_RATE",
    "VAD_MODEL_NAME",
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
