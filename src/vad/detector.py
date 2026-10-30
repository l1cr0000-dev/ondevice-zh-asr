"""VAD 狀態機：把逐 frame 的語音機率轉成語音開始 / 結束事件。

離線與串流共用同一個狀態機（離線 = 整段送入後 flush），保證兩者輸出一致。

判定規則（參數見 `VadConfig`）：
- 機率 >= threshold 視為語音 frame，< neg_threshold 視為靜音 frame，介於兩者之間維持目前狀態（遲滯）。
- 語音段內部的靜音短於 min_silence_ms 不切段；靜音持續滿 min_silence_ms 才結束。
- 語音段（含其中的短靜音）總長滿 min_speech_ms 才確認開始；不足的直接丟棄。
- 輸出的起點往前、終點往後各延伸 speech_pad_ms，但不跨越前一段的終點、也不超出音訊範圍。

因此串流的「開始」事件最早在語音起點後 min_speech_ms 才發出，「結束」事件在語音終點後
min_silence_ms 才發出。事件的 `sample` 是時間軸上的位置，`detected_at` 是發出事件時已送入的樣本數，
兩者之差就是 VAD 本身造成的延遲。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from src.vad.silero import SAMPLE_RATE


class FrameClassifier(Protocol):
    """逐 frame 回傳語音機率的模型後端。"""

    @property
    def frame_size(self) -> int: ...

    def __call__(self, frame: npt.NDArray[np.float32]) -> float: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class VadConfig:
    threshold: float = 0.5
    neg_threshold: float | None = None  # None 時取 max(threshold - 0.15, 0.01)，與上游一致
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    speech_pad_ms: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold 必須在 (0, 1) 內，收到 {self.threshold}")
        if self.neg_threshold is not None and not 0.0 < self.neg_threshold <= self.threshold:
            raise ValueError("neg_threshold 必須在 (0, threshold] 內")
        if min(self.min_speech_ms, self.min_silence_ms, self.speech_pad_ms) < 0:
            raise ValueError("時間長度參數不可為負")

    @property
    def effective_neg_threshold(self) -> float:
        if self.neg_threshold is not None:
            return self.neg_threshold
        return max(self.threshold - 0.15, 0.01)


@dataclass(frozen=True)
class VadEvent:
    kind: Literal["start", "end"]
    sample: int  # 事件在音訊時間軸上的位置（已含 padding）
    detected_at: int  # 發出事件時已送入的樣本數

    @property
    def time_s(self) -> float:
        return self.sample / SAMPLE_RATE

    @property
    def latency_s(self) -> float:
        """事件發出時刻相對事件位置的延遲（演算法延遲，不含計算時間）。"""
        return (self.detected_at - self.sample) / SAMPLE_RATE


@dataclass(frozen=True)
class SpeechSegment:
    start: int  # 樣本索引，含
    end: int  # 樣本索引，不含

    @property
    def start_s(self) -> float:
        return self.start / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.end / SAMPLE_RATE


class _State(Enum):
    SILENCE = "silence"
    PENDING = "pending"  # 偵測到語音但還不滿 min_speech_ms
    SPEECH = "speech"


def _ms_to_samples(ms: int) -> int:
    return ms * SAMPLE_RATE // 1000


class StreamingVad:
    """逐 chunk 送入音訊，回傳這次新產生的事件。chunk 長度任意，內部緩衝到 frame 大小。"""

    def __init__(self, model: FrameClassifier, config: VadConfig | None = None) -> None:
        self.model = model
        self.config = config or VadConfig()
        self._min_speech = _ms_to_samples(self.config.min_speech_ms)
        self._min_silence = _ms_to_samples(self.config.min_silence_ms)
        self._pad = _ms_to_samples(self.config.speech_pad_ms)
        self._neg_threshold = self.config.effective_neg_threshold
        self.reset()

    def reset(self) -> None:
        self.model.reset()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._received = 0  # 已送入的樣本數（不含 flush 補的零）
        self._processed = 0  # 已送進模型的樣本數
        self._state = _State.SILENCE
        self._speech_start = 0  # 目前（候選）語音段的起點，未加 padding
        self._last_speech_end = 0  # 最後一個語音 frame 的終點
        self._prev_end = 0  # 上一段輸出的終點（已加 padding），避免段落重疊
        self.last_prob = 0.0

    def accept_waveform(self, chunk: npt.NDArray[np.floating]) -> list[VadEvent]:
        """送入一段 float 音訊（16 kHz、單聲道、範圍約 [-1, 1]）。"""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._received += len(chunk)
        self._buffer = np.concatenate([self._buffer, chunk])
        events: list[VadEvent] = []
        frame_size = self.model.frame_size
        n_frames = len(self._buffer) // frame_size
        for i in range(n_frames):
            events.extend(self._step(self._buffer[i * frame_size : (i + 1) * frame_size]))
        self._buffer = self._buffer[n_frames * frame_size :]
        return events

    def flush(self) -> list[VadEvent]:
        """輸入結束：補零處理剩餘樣本，並結束仍在進行中的語音段。之後需 `reset()` 才能重用。"""
        events: list[VadEvent] = []
        if len(self._buffer):
            frame = np.zeros(self.model.frame_size, dtype=np.float32)
            frame[: len(self._buffer)] = self._buffer
            self._buffer = np.zeros(0, dtype=np.float32)
            events.extend(self._step(frame))
        total = self._received
        if self._state is _State.SPEECH:
            events.append(self._end_event(min(self._last_speech_end + self._pad, total), total))
        # PENDING 代表候選段到結尾都不滿 min_speech_ms，直接丟棄
        self._state = _State.SILENCE
        return events

    def _start_event(self, detected_at: int) -> VadEvent:
        self._state = _State.SPEECH
        start = max(self._speech_start - self._pad, self._prev_end, 0)
        return VadEvent("start", start, detected_at)

    def _end_event(self, end: int, detected_at: int) -> VadEvent:
        self._state = _State.SILENCE
        self._prev_end = end
        return VadEvent("end", end, detected_at)

    def _step(self, frame: npt.NDArray[np.float32]) -> Iterator[VadEvent]:
        prob = self.model(frame)
        self.last_prob = prob
        frame_start = self._processed
        self._processed += len(frame)
        frame_end = self._processed
        detected_at = min(frame_end, self._received)

        is_speech = prob >= self.config.threshold
        is_silence = prob < self._neg_threshold

        if self._state is _State.SILENCE:
            if is_speech:
                self._state = _State.PENDING
                self._speech_start = frame_start
                self._last_speech_end = frame_end
        elif is_speech or not is_silence:
            # 語音 frame，或遲滯區間內：延續目前的語音（候選）段
            if is_speech:
                self._last_speech_end = frame_end
        else:
            silence = frame_end - self._last_speech_end
            if silence >= self._min_silence:
                if self._state is _State.SPEECH:
                    # padding 不超過已收到的音訊（speech_pad_ms > min_silence_ms 時才會被截）
                    end = min(self._last_speech_end + self._pad, detected_at)
                    yield self._end_event(end, detected_at)
                else:
                    self._state = _State.SILENCE  # 候選段太短，丟棄
                return

        if (
            self._state is _State.PENDING
            and self._last_speech_end - self._speech_start >= self._min_speech
        ):
            yield self._start_event(detected_at)


def events_to_segments(events: list[VadEvent]) -> list[SpeechSegment]:
    segments: list[SpeechSegment] = []
    start: int | None = None
    for ev in events:
        if ev.kind == "start":
            start = ev.sample
        elif start is not None:
            segments.append(SpeechSegment(start, ev.sample))
            start = None
    return segments


def detect_segments(
    audio: npt.NDArray[np.floating], model: FrameClassifier, config: VadConfig | None = None
) -> list[SpeechSegment]:
    """離線介面：輸入整段 16 kHz 單聲道音訊，回傳語音段。"""
    vad = StreamingVad(model, config)
    events = vad.accept_waveform(audio) + vad.flush()
    return events_to_segments(events)
