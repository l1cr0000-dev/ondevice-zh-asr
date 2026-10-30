"""串接 VAD 與串流 ASR：音訊 chunk → VAD → 每句一個 ASR stream → 整句結果與時間戳。

`StreamingPipeline(vad_model, asr, config)`，`accept_waveform(chunk)` / `flush()` → `list[PipelineEvent]`，
完成的句子累積在 `pipeline.utterances`。離線處理整段音訊用 `transcribe(audio, ...)`。
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from src.asr import SAMPLE_RATE, StreamingAsr
from src.pipeline.pipeline import (
    Partial,
    PipelineConfig,
    PipelineEvent,
    StreamingPipeline,
    Utterance,
)
from src.vad import FrameClassifier

__all__ = [
    "Partial",
    "PipelineConfig",
    "PipelineEvent",
    "StreamingPipeline",
    "Utterance",
    "transcribe",
]


def transcribe(
    audio: npt.NDArray[np.floating],
    vad_model: FrameClassifier,
    asr: StreamingAsr,
    config: PipelineConfig | None = None,
    chunk_ms: int = 10,
    clock: Callable[[], float] | None = None,
) -> list[Utterance]:
    """整段音訊依 `chunk_ms` 切塊送入 pipeline，回傳每句結果。時間戳為模擬即時（見 pipeline 模組說明）。"""
    if chunk_ms <= 0:
        raise ValueError("chunk_ms 必須為正數")
    pipeline = (
        StreamingPipeline(vad_model, asr, config)
        if clock is None
        else StreamingPipeline(vad_model, asr, config, clock=clock)
    )
    size = chunk_ms * SAMPLE_RATE // 1000
    for i in range(0, len(audio), size):
        pipeline.accept_waveform(audio[i : i + size])
    pipeline.flush()
    return pipeline.utterances
