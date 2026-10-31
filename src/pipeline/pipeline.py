"""VAD + 串流 ASR：逐 chunk 送入音訊，VAD 偵測到語音開始就建立 ASR stream，語音中持續送入，
語音結束時 finalize 並輸出整句結果。

pre-roll：VAD 的「開始」事件要等語音持續 min_speech_ms 才發出，事件位置（`VadEvent.sample`，已含
speech_pad_ms）也可能切在首字的弱起音上。因此 pipeline 保留最近一段音訊，建立 stream 時從
`事件位置 - pre_roll_ms` 開始送入 ASR；但不早於上一句已送入 ASR 的終點，避免同一段音訊辨識兩次。

暫定結果（`VadConfig.pause_ms` 有設定時）：句中靜音滿 pause_ms，VAD 發出停頓事件，pipeline 另開一個 ASR stream，
把這句目前為止送入的全部音訊重新解碼並 finalize，輸出暫定結果；原 stream 不中斷。之後若又開始說話，同一句
繼續送入原 stream，最終結果會取代（修正）暫定結果；若靜音滿 min_silence_ms，照常 finalize 原 stream 定稿。
因此最終結果的斷句由較長的 min_silence_ms 決定（少切碎），使用者則在 pause_ms 後就先看到結果。
（另開 stream 是因為串流 ASR 的狀態無法複製；代價是重新解碼的計算量與句長成正比。）

時間記錄（全部以秒為單位、以音訊開頭為 0）分兩類：
- 音訊時間軸上的位置：`speech_start_s`、`speech_end_s`、`audio_start_s`。
- 「模擬即時」的發出時刻：`*_emitted_s`。假設音訊依實際時長節奏到達，一個 chunk 要等最後一個
  樣本到達才能送入；送入後的計算時間以 `clock` 實際量測並累加。若計算比音訊慢，會自然積壓到
  後面的 chunk。因此不需要真的 sleep，也能得到與即時餵入一致的延遲（chunk 大小會影響結果，
  報告時要註明）。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from src.asr import SAMPLE_RATE, AsrStream, StreamingAsr
from src.vad import FrameClassifier, StreamingVad, VadConfig, VadEvent


def _default_vad() -> VadConfig:
    # AISHELL-1 test 200 句掃描（docs/progress.md）：min_silence_ms 100 → 500，X-ASR CER −1.8 個百分點、
    # 每條平均切段數 1.74 → 1.19；500 → 800 無顯著差異，但每句最終結果要多等 300 ms。
    # 其餘參數維持上游 Silero 預設值。VAD 模組本身的預設值不變。
    return VadConfig(min_silence_ms=500)


@dataclass(frozen=True)
class PipelineConfig:
    vad: VadConfig = field(default_factory=_default_vad)
    # 送入 ASR 的音訊比 VAD 起點（已含 speech_pad_ms）再往前多取的長度。
    # AISHELL-1 掃描：X-ASR 0 → 100 ms CER −0.8 個百分點，100 / 200 / 300 ms 無顯著差異；
    # 更長的 pre-roll 讓更多句子的首字要多等一個 480 ms chunk（P90 不穩），故取 100 ms。
    pre_roll_ms: int = 100

    def __post_init__(self) -> None:
        if self.pre_roll_ms < 0:
            raise ValueError("pre_roll_ms 不可為負")


@dataclass(frozen=True)
class Partial:
    emitted_s: float
    text: str


@dataclass(frozen=True)
class Provisional:
    emitted_s: float  # 暫定結果可用的時刻（含重新解碼的計算時間）
    audio_end_s: float  # 重新解碼涵蓋到的音訊終點（停頓事件發出時已送入的位置）
    text: str


@dataclass
class Utterance:
    """一句話的結果。`*_s` 的意義見模組說明。"""

    index: int
    speech_start_s: float  # VAD 起點（含 speech_pad_ms）
    audio_start_s: float  # 實際送入 ASR 的第一個樣本（含 pre-roll）
    start_emitted_s: float  # VAD 發出「開始」事件、建立 ASR stream 的時刻
    first_token_s: float | None = None  # 第一個非空 partial 出現的時刻
    # 第一個 token 在音訊時間軸上的位置（ASR 後端回報的 token 時間戳）；後端不支援時為 None
    first_token_audio_s: float | None = None
    speech_end_s: float | None = None  # VAD 終點（含 speech_pad_ms）
    end_emitted_s: float | None = None  # VAD 發出「結束」事件的時刻
    final_s: float | None = None  # finalize 完成、最終文字可用的時刻
    text: str = ""
    partials: list[Partial] = field(default_factory=list)
    provisionals: list[Provisional] = field(default_factory=list)

    @property
    def first_token_latency_s(self) -> float | None:
        """首字延遲：語音起點（VAD 起點）→ 第一個非空 partial。"""
        if self.first_token_s is None:
            return None
        return self.first_token_s - self.speech_start_s

    @property
    def tail_latency_s(self) -> float | None:
        """尾字延遲：語音終點（VAD 終點）→ 最終結果。含 VAD 的 min_silence_ms 等待。"""
        if self.final_s is None or self.speech_end_s is None:
            return None
        return self.final_s - self.speech_end_s

    @property
    def final_text_ready_s(self) -> float | None:
        """最終文字就緒時刻：與最終結果完全相同的文字第一次出現在 partial 中的時刻。

        partial 從未等於最終結果（例如句尾的字要 finalize 補零後才解出）時取 `final_s`。
        與 `end_emitted_s` 比較，可知 VAD 判定結束前 partial 是否已含完整句子。
        """
        if self.final_s is None:
            return None
        return next((p.emitted_s for p in self.partials if p.text == self.text), self.final_s)


@dataclass(frozen=True)
class PipelineEvent:
    kind: Literal["start", "partial", "provisional", "final"]
    utterance: Utterance


class _AudioHistory:
    """保留最近 `capacity` 個樣本，以絕對樣本索引取用。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data = np.zeros(0, dtype=np.float32)
        self.start = 0  # _data[0] 的絕對樣本索引

    @property
    def end(self) -> int:
        return self.start + len(self._data)

    def append(self, chunk: npt.NDArray[np.float32]) -> None:
        self._data = np.concatenate([self._data, chunk])

    def trim(self) -> None:
        excess = len(self._data) - self.capacity
        if excess > 0:
            self._data = self._data[excess:]
            self.start += excess

    def slice(self, begin: int, end: int) -> npt.NDArray[np.float32]:
        if begin < self.start or end > self.end:
            raise IndexError(f"[{begin}, {end}) 超出保留範圍 [{self.start}, {self.end})")
        return self._data[begin - self.start : end - self.start]


class StreamingPipeline:
    """逐 chunk 送入 → 回傳這次新產生的事件（開始 / partial 變化 / 整句結果）。

    pipeline = StreamingPipeline(vad_model, asr)
    for chunk in chunks:
        for ev in pipeline.accept_waveform(chunk):
            ...
    for ev in pipeline.flush():
        ...
    pipeline.utterances  # 所有已完成的句子
    """

    def __init__(
        self,
        vad_model: FrameClassifier,
        asr: StreamingAsr,
        config: PipelineConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if asr.sample_rate != SAMPLE_RATE:
            raise ValueError(f"ASR 取樣率 {asr.sample_rate} 與 VAD {SAMPLE_RATE} 不符")
        self.config = config or PipelineConfig()
        self.vad = StreamingVad(vad_model, self.config.vad)
        self.asr = asr
        self._clock = clock
        self._pre_roll = self.config.pre_roll_ms * SAMPLE_RATE // 1000
        vad_cfg = self.config.vad
        # 「開始」事件最多落後事件位置 min_speech + pad + 一個 frame；再加 pre-roll 與一些餘裕
        lookback_ms = vad_cfg.min_speech_ms + vad_cfg.speech_pad_ms + self.config.pre_roll_ms + 100
        self._history_capacity = lookback_ms * SAMPLE_RATE // 1000 + vad_model.frame_size
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._history = _AudioHistory(self._history_capacity)
        self._received = 0
        self._now = 0.0  # 模擬即時的目前時刻（秒）
        self._call_t0 = 0.0
        self._call_base = 0.0
        self._stream: AsrStream | None = None
        self._current: Utterance | None = None
        self._fed_until = 0  # 目前 / 上一句已送入 ASR 的終點（絕對樣本索引）
        self._utt_audio: list[
            npt.NDArray[np.float32]
        ] = []  # 這句已送入 ASR 的音訊（暫定結果重新解碼用）
        self.utterances: list[Utterance] = []

    @property
    def in_speech(self) -> bool:
        return self._stream is not None

    def accept_waveform(self, chunk: npt.NDArray[np.floating]) -> list[PipelineEvent]:
        """送入一段 16 kHz 單聲道 float 音訊，長度任意。"""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._received += len(chunk)
        # chunk 的最後一個樣本到達後才能處理
        self._begin_call(max(self._now, self._received / SAMPLE_RATE))
        self._history.append(chunk)
        events = self._handle(self.vad.accept_waveform(chunk))
        if self._stream is not None:
            events.extend(self._feed(self._received))
        self._history.trim()
        self._end_call()
        return events

    def flush(self) -> list[PipelineEvent]:
        """輸入結束：結束仍在進行中的句子。之後需 `reset()` 才能重用。"""
        self._begin_call(self._now)
        events = self._handle(self.vad.flush())
        self._end_call()
        return events

    # ----- 內部 -----

    def _begin_call(self, base: float) -> None:
        self._call_base = base
        self._call_t0 = self._clock()

    def _elapsed(self) -> float:
        return self._call_base + (self._clock() - self._call_t0)

    def _end_call(self) -> None:
        self._now = self._elapsed()

    def _handle(self, vad_events: list[VadEvent]) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        for ev in vad_events:
            if ev.kind == "start":
                events.append(self._start(ev))
                events.extend(self._feed(ev.detected_at))
            elif ev.kind == "pause":
                events.extend(self._feed(ev.detected_at))
                events.append(self._provisional())
            else:
                events.extend(self._feed(ev.detected_at))
                events.append(self._finish(ev))
        return events

    def _start(self, ev: VadEvent) -> PipelineEvent:
        assert self._stream is None, "VAD 事件應成對出現"
        begin = max(ev.sample - self._pre_roll, self._fed_until, self._history.start)
        self._stream = self.asr.create_stream()
        self._fed_until = begin
        self._current = Utterance(
            index=len(self.utterances),
            speech_start_s=ev.sample / SAMPLE_RATE,
            audio_start_s=begin / SAMPLE_RATE,
            start_emitted_s=self._elapsed(),
        )
        return PipelineEvent("start", self._current)

    def _feed(self, until: int) -> list[PipelineEvent]:
        """把 [_fed_until, until) 送入目前的 stream，partial 有變化時回傳事件。"""
        stream, utt = self._stream, self._current
        if stream is None or utt is None or until <= self._fed_until:
            return []
        audio = self._history.slice(self._fed_until, until)
        stream.accept_waveform(audio)
        if self.config.vad.pause_ms is not None:
            self._utt_audio.append(audio.copy())
        self._fed_until = until
        text = stream.get_partial_result()
        last = utt.partials[-1].text if utt.partials else ""
        if text == last:
            return []
        now = self._elapsed()
        utt.partials.append(Partial(now, text))
        if utt.first_token_s is None and text.strip():
            utt.first_token_s = now
            utt.first_token_audio_s = self._first_token_audio_s(stream, utt)
        return [PipelineEvent("partial", utt)]

    @staticmethod
    def _first_token_audio_s(stream: AsrStream, utt: Utterance) -> float | None:
        # 首字時刻已在呼叫前取得；讀時間戳的耗時計入之後的計算時間（每句只讀一次）
        stamps = stream.get_token_timestamps()
        return utt.audio_start_s + stamps[0] if stamps else None

    def _provisional(self) -> PipelineEvent:
        utt = self._current
        assert utt is not None, "停頓事件只在語音段中發出"
        stream = self.asr.create_stream()
        stream.accept_waveform(np.concatenate(self._utt_audio))
        text = stream.finalize()
        utt.provisionals.append(Provisional(self._elapsed(), self._fed_until / SAMPLE_RATE, text))
        return PipelineEvent("provisional", utt)

    def _finish(self, ev: VadEvent) -> PipelineEvent:
        stream, utt = self._stream, self._current
        assert stream is not None and utt is not None, "VAD 事件應成對出現"
        utt.speech_end_s = ev.sample / SAMPLE_RATE
        utt.end_emitted_s = self._elapsed()
        utt.text = stream.finalize()
        utt.final_s = self._elapsed()
        if utt.first_token_s is None and utt.text.strip():
            # partial 一直為空、直到 finalize 才出字（例如語音很短）
            utt.first_token_s = utt.final_s
            utt.first_token_audio_s = self._first_token_audio_s(stream, utt)
        self._stream = None
        self._current = None
        self._utt_audio = []
        self.utterances.append(utt)
        return PipelineEvent("final", utt)
