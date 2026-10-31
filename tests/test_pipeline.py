"""pipeline 測試。

- 路由與時間戳：用能量判斷的假 VAD 與記錄輸入的假 ASR，精確檢查 pre-roll、句間不重疊、時間順序。
- 整合測試：真實 Silero VAD + 預設 ASR 模型，跑 tests/fixtures 的兩句合成語音。需先執行
  bash scripts/download_models.sh，否則跳過。
"""

import json
from itertools import pairwise
from pathlib import Path

import jiwer
import numpy as np
import numpy.typing as npt
import pytest

from src.asr import DEFAULT_MODEL, SAMPLE_RATE, AsrStream, StreamingAsr, load_asr
from src.pipeline import PipelineConfig, StreamingPipeline, Utterance, transcribe
from src.vad import DEFAULT_MODEL_PATH, SileroVadModel, VadConfig, load_wav

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURES / "vad_two_utterances.wav"
# fixture 標註是繁體，模型輸出簡體
REFERENCES = ["今天天气很好", "我们去公园散步吧"]
BOUNDARY_TOL_S = 0.1


# ---------- 假後端 ----------


class _EnergyClassifier:
    """RMS 超過門檻即視為語音（機率 1），否則 0。"""

    frame_size = 512

    def __call__(self, frame: npt.NDArray[np.float32]) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > 0.01 else 0.0

    def reset(self) -> None:
        pass


class _RecordingStream(AsrStream):
    """記錄收到的所有樣本；每收到 0.1 s 非零音訊輸出一個字。"""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[npt.NDArray[np.float32]] = []

    @property
    def audio(self) -> npt.NDArray[np.float32]:
        return np.concatenate(self.received) if self.received else np.zeros(0, np.float32)

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        self.received.append(chunk.copy())

    def _partial(self) -> str:
        return "字" * (int(np.count_nonzero(self.audio)) // (SAMPLE_RATE // 10))

    def _finalize(self) -> str:
        return self._partial() + "。"

    def get_token_timestamps(self) -> list[float]:
        return [0.25] * len(self._partial())  # 固定值，檢查 pipeline 換算到音訊時間軸


class _RecordingAsr(StreamingAsr):
    def __init__(self) -> None:
        self.streams: list[_RecordingStream] = []

    @property
    def name(self) -> str:
        return "recording"

    def create_stream(self) -> _RecordingStream:
        self.streams.append(_RecordingStream())
        return self.streams[-1]


def _tones(*spans: tuple[float, float], total_s: float) -> npt.NDArray[np.float32]:
    """在 [start, end) 秒放常數 0.1（每個樣本都非零，方便假 ASR 計數），其餘為 0。"""
    audio = np.zeros(int(total_s * SAMPLE_RATE), dtype=np.float32)
    for start, end in spans:
        audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.1
    return audio


def _run_fake(
    audio: npt.NDArray[np.float32], config: PipelineConfig, chunk_ms: int = 10
) -> tuple[list[Utterance], _RecordingAsr]:
    asr = _RecordingAsr()
    # 固定時鐘：計算時間為 0，發出時刻 = chunk 到齊的音訊時刻
    utts = transcribe(audio, _EnergyClassifier(), asr, config, chunk_ms, clock=lambda: 0.0)
    return utts, asr


def _sample(t: float) -> int:
    return round(t * SAMPLE_RATE)


# ---------- 路由與時間戳（假後端） ----------


@pytest.mark.parametrize("pre_roll_ms", [0, 200, 500])
def test_asr_receives_audio_from_pre_roll(pre_roll_ms: int) -> None:
    audio = _tones((1.0, 2.0), total_s=3.0)
    config = PipelineConfig(VadConfig(min_silence_ms=200), pre_roll_ms=pre_roll_ms)
    (utt,), asr = _run_fake(audio, config)

    begin = _sample(utt.speech_start_s) - pre_roll_ms * SAMPLE_RATE // 1000
    assert _sample(utt.audio_start_s) == begin
    fed = asr.streams[0].audio
    # 送入 ASR 的正是原音訊從 pre-roll 起點開始的連續一段，沒有缺漏或重複
    np.testing.assert_array_equal(fed, audio[begin : begin + len(fed)])
    # 涵蓋整段語音
    assert begin + len(fed) >= _sample(2.0)
    assert utt.text == "字" * 10 + "。"


def test_pre_roll_does_not_overlap_previous_utterance() -> None:
    # 兩段只隔 0.4 s，比 pre-roll 短：第二句的 pre-roll 要被截在第一句送入的終點
    audio = _tones((1.0, 1.5), (1.9, 2.5), total_s=3.5)
    config = PipelineConfig(VadConfig(min_silence_ms=200), pre_roll_ms=1000)
    utts, asr = _run_fake(audio, config)

    assert len(utts) == 2
    first_end = _sample(utts[0].audio_start_s) + len(asr.streams[0].audio)
    assert _sample(utts[1].audio_start_s) == first_end
    total_fed = sum(len(s.audio) for s in asr.streams)
    np.testing.assert_array_equal(
        np.concatenate([s.audio for s in asr.streams]),
        audio[_sample(utts[0].audio_start_s) :][:total_fed],
    )


def test_timestamps_are_ordered_and_consistent() -> None:
    audio = _tones((1.0, 2.0), (3.0, 3.6), total_s=4.5)
    config = PipelineConfig(VadConfig(min_silence_ms=200, min_speech_ms=100))
    utts, _ = _run_fake(audio, config)

    assert len(utts) == 2
    for utt, (start, end) in zip(utts, [(1.0, 2.0), (3.0, 3.6)], strict=True):
        assert utt.first_token_s is not None and utt.speech_end_s is not None
        assert utt.end_emitted_s is not None and utt.final_s is not None
        assert utt.speech_start_s == pytest.approx(start, abs=0.07)
        assert utt.speech_end_s == pytest.approx(end, abs=0.07)
        assert utt.audio_start_s <= utt.speech_start_s < utt.start_emitted_s
        assert utt.start_emitted_s <= utt.first_token_s <= utt.end_emitted_s
        assert utt.speech_end_s < utt.end_emitted_s <= utt.final_s
        # 計算時間為 0 時，發出時刻落在 10 ms chunk 邊界上
        for t in (utt.start_emitted_s, utt.first_token_s, utt.final_s):
            assert round(t * 100) == pytest.approx(t * 100, abs=1e-6)
        assert [p.emitted_s for p in utt.partials] == sorted(p.emitted_s for p in utt.partials)
    assert utts[0].final_s is not None and utts[0].final_s < utts[1].start_emitted_s


def test_first_token_audio_position_uses_stream_offset() -> None:
    audio = _tones((1.0, 2.0), total_s=3.0)
    (utt,), _ = _run_fake(audio, PipelineConfig())
    assert utt.first_token_audio_s == pytest.approx(utt.audio_start_s + 0.25)


def test_compute_time_is_added_to_emitted_time() -> None:
    audio = _tones((1.0, 2.0), total_s=3.0)
    ticks = iter(np.arange(0, 1e6, 1e-3))  # 每呼叫一次 clock 前進 1 ms
    (utt,) = transcribe(
        audio, _EnergyClassifier(), _RecordingAsr(), PipelineConfig(), clock=lambda: next(ticks)
    )
    assert utt.end_emitted_s is not None and utt.final_s is not None
    assert utt.final_s > utt.end_emitted_s


@pytest.mark.parametrize("chunk_ms", [10, 37, 320, 4000])
def test_fed_audio_independent_of_chunk_size(chunk_ms: int) -> None:
    audio = _tones((1.0, 2.0), (2.6, 3.3), total_s=4.0)
    config = PipelineConfig(VadConfig(min_silence_ms=200))
    ref_utts, ref_asr = _run_fake(audio, config, chunk_ms=10)
    utts, asr = _run_fake(audio, config, chunk_ms=chunk_ms)

    assert [u.text for u in utts] == [u.text for u in ref_utts]
    assert [u.audio_start_s for u in utts] == [u.audio_start_s for u in ref_utts]
    for a, b in zip(asr.streams, ref_asr.streams, strict=True):
        np.testing.assert_array_equal(a.audio, b.audio)


def test_silence_creates_no_stream() -> None:
    utts, asr = _run_fake(np.zeros(3 * SAMPLE_RATE, dtype=np.float32), PipelineConfig())
    assert utts == [] and asr.streams == []


def test_speech_until_end_is_finalized_by_flush() -> None:
    pipeline = StreamingPipeline(_EnergyClassifier(), _RecordingAsr(), clock=lambda: 0.0)
    audio = _tones((1.0, 2.0), total_s=2.0)
    events = pipeline.accept_waveform(audio)
    assert pipeline.in_speech and "final" not in [e.kind for e in events]
    events = pipeline.flush()
    assert [e.kind for e in events][-1] == "final"
    assert not pipeline.in_speech
    assert len(pipeline.utterances) == 1


# ---------- 整合測試（真實模型） ----------


@pytest.fixture(scope="module")
def models() -> tuple[SileroVadModel, StreamingAsr]:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("VAD 模型不存在，先執行 bash scripts/download_models.sh")
    try:
        return SileroVadModel(), load_asr(DEFAULT_MODEL, num_threads=1)
    except FileNotFoundError:
        pytest.skip("ASR 模型不存在，先執行 bash scripts/download_models.sh")


def test_end_to_end_two_utterances(models: tuple[SileroVadModel, StreamingAsr]) -> None:
    vad_model, asr = models
    truth = json.loads(SPEECH_WAV.with_suffix(".json").read_text())["segments"]
    utts = transcribe(load_wav(SPEECH_WAV), vad_model, asr, chunk_ms=10)

    assert len(utts) == len(truth)
    for utt, ref, ref_text in zip(utts, truth, REFERENCES, strict=True):
        assert utt.first_token_s is not None and utt.speech_end_s is not None
        assert utt.final_s is not None and utt.end_emitted_s is not None
        cer = jiwer.cer(ref_text, utt.text)
        assert cer <= 0.15, f"CER={cer:.3f}，hyp={utt.text}"
        assert utt.speech_start_s == pytest.approx(ref["start"], abs=BOUNDARY_TOL_S)
        assert utt.speech_end_s == pytest.approx(ref["end"], abs=BOUNDARY_TOL_S)
        assert utt.audio_start_s < utt.speech_start_s < utt.start_emitted_s
        assert utt.start_emitted_s <= utt.first_token_s <= utt.end_emitted_s <= utt.final_s
        # 首字延遲：至少要等 VAD 確認 + ASR 第一個 chunk，但不該超過 1.5 s
        assert utt.first_token_latency_s is not None
        assert 0.3 < utt.first_token_latency_s < 1.5
        # 模型回報的首字位置在送入的音訊範圍內，且不晚於首字實際出現的時刻
        assert utt.first_token_audio_s is not None
        assert utt.audio_start_s <= utt.first_token_audio_s <= utt.first_token_s
        # partial 不倒退，且最後一個 partial 是最終結果的前綴
        assert all(len(a.text) <= len(b.text) for a, b in pairwise(utt.partials))
        assert utt.text.startswith(utt.partials[-1].text)
    assert utts[0].final_s is not None and utts[0].final_s < utts[1].start_emitted_s


def test_end_to_end_independent_of_chunk_size(
    models: tuple[SileroVadModel, StreamingAsr],
) -> None:
    vad_model, asr = models
    audio = load_wav(SPEECH_WAV)
    results = {
        ms: [(u.text, u.audio_start_s) for u in transcribe(audio, vad_model, asr, chunk_ms=ms)]
        for ms in (10, 37, 320)
    }
    assert len({tuple(r) for r in results.values()}) == 1, results
