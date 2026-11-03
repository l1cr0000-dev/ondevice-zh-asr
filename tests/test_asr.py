"""串流 ASR 測試。

- 介面約定：用假後端驗證，不需要模型。
- 真實模型：每個已登記模型各跑一遍，用 zh-14M 壓縮檔自帶的 test_wavs（WenetSpeech 片段，
  不進 git），需先執行 bash scripts/download_models.sh，否則跳過。
"""

from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import jiwer
import numpy as np
import numpy.typing as npt
import pytest

from src.asr import (
    DEFAULT_MODEL,
    SAMPLE_RATE,
    AsrStream,
    PartialSchedule,
    SenseVoiceConfig,
    SenseVoiceOrtAsr,
    StreamingAsr,
    available_models,
    default_threads,
    load_asr,
)
from src.asr.sense_voice import SenseVoiceStream
from src.asr.sherpa import X_ASR_480MS, ZIPFORMER_ZH_14M, SherpaTransducerAsr
from src.asr.zipformer_ort import CoreMLOptions, ZipformerOrtAsr, coreml_available, tokens_to_text
from src.paths import MODELS_DIR
from src.vad import load_wav

# 兩個模型都用 zh-14M 壓縮檔自帶的 0.wav / 1.wav，因為只有這兩句有人工標註
TEST_WAVS = MODELS_DIR / ZIPFORMER_ZH_14M.model_dir / "test_wavs"

# 人工標註（WenetSpeech 原始參考文本，含語氣詞）
REFERENCES = {
    "0.wav": "对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢",
    "1.wav": "重点呢想谈三个问题首先呢就是这一轮全球金融动荡的表现",
}

# sherpa-onnx 官方文件中 zh-14M（FP32，greedy search）對 0.wav 的輸出：
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html
OFFICIAL_OUTPUT_14M_0 = "对我做了介绍那么我想说的是大家如果对我的研究感兴趣"

# 各模型在 REFERENCES 上允許的最大 CER（目前實測值附在後面）
MAX_CER = {
    # 只用 WenetSpeech 訓練，會省略部分語氣詞：0.wav 3/28、1.wav 2/26
    "zipformer-zh-14m": 0.15,
    # 兩句皆 0/N。上限 0.05 也用來抓句尾補零不足：1.wav 漏掉「表现」時 CER = 2/26 ≈ 0.077
    "x-asr-480ms-int8": 0.05,
    # 同一模型直接用 onnxruntime，結果與上一行相同
    "x-asr-480ms-int8-ort": 0.05,
    # 0.wav 1/28（漏「啊」）、1.wav 0/26。同樣用來抓句尾補零不足
    "x-asr-160ms-int8": 0.05,
    # 同一模型直接用 onnxruntime
    "x-asr-160ms-int8-ort": 0.05,
}


# ---------- 介面約定（假後端） ----------


class _EchoStream(AsrStream):
    """每收到一個 chunk 就輸出一個字，方便檢查約定。"""

    def __init__(self) -> None:
        super().__init__()
        self.n_chunks = 0

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        assert chunk.dtype == np.float32 and chunk.ndim == 1
        self.n_chunks += 1

    def _partial(self) -> str:
        return "字" * self.n_chunks

    def _finalize(self) -> str:
        return self._partial() + "。"


class _EchoAsr(StreamingAsr):
    @property
    def name(self) -> str:
        return "echo"

    def create_stream(self) -> _EchoStream:
        return _EchoStream()


def test_contract_partial_then_final() -> None:
    stream = _EchoAsr().create_stream()
    assert stream.get_partial_result() == ""
    stream.accept_waveform(np.zeros(160, dtype=np.float64))  # 非 float32 也接受，內部轉換
    stream.accept_waveform(np.zeros((1, 160), dtype=np.float32))  # 會攤平成一維
    assert stream.get_partial_result() == "字字"
    assert not stream.is_finalized
    assert stream.finalize() == "字字。"
    assert stream.is_finalized
    assert stream.get_partial_result() == "字字。"


def test_contract_finalized_stream_rejects_input() -> None:
    stream = _EchoAsr().create_stream()
    stream.finalize()
    with pytest.raises(RuntimeError):
        stream.accept_waveform(np.zeros(160, dtype=np.float32))
    with pytest.raises(RuntimeError):
        stream.finalize()


def test_load_asr_unknown_name() -> None:
    assert set(MAX_CER) <= set(available_models())
    with pytest.raises(ValueError, match="未知模型"):
        load_asr("no-such-model")


# ---------- 真實模型 ----------


@pytest.fixture(scope="module", params=sorted(MAX_CER))
def asr(request: pytest.FixtureRequest) -> StreamingAsr:
    if not TEST_WAVS.is_dir():
        pytest.skip("模型未下載，請先執行 bash scripts/download_models.sh")
    try:
        return load_asr(request.param, num_threads=1)
    except FileNotFoundError:
        pytest.skip(f"{request.param} 未下載，請先執行 bash scripts/download_models.sh")


def _chunks(audio: npt.NDArray[np.float32], chunk_ms: int) -> Iterator[npt.NDArray[np.float32]]:
    size = SAMPLE_RATE * chunk_ms // 1000
    for i in range(0, len(audio), size):
        yield audio[i : i + size]


def _transcribe(asr: StreamingAsr, wav: Path, chunk_ms: int = 100) -> str:
    stream = asr.create_stream()
    for chunk in _chunks(load_wav(wav), chunk_ms):
        stream.accept_waveform(chunk)
    return stream.finalize()


def test_blank_penalty_is_passed_to_backend() -> None:
    if not TEST_WAVS.is_dir():
        pytest.skip("模型未下載，請先執行 bash scripts/download_models.sh")
    asr = load_asr("zipformer-zh-14m", num_threads=1, blank_penalty=1.5)
    assert isinstance(asr, SherpaTransducerAsr) and asr.config.blank_penalty == 1.5
    assert "blank_penalty=1.5" in asr.name
    assert "blank_penalty" not in load_asr("zipformer-zh-14m", num_threads=1).name


def test_14m_matches_official_output() -> None:
    if not TEST_WAVS.is_dir():
        pytest.skip("模型未下載，請先執行 bash scripts/download_models.sh")
    asr = load_asr("zipformer-zh-14m", num_threads=1)
    assert _transcribe(asr, TEST_WAVS / "0.wav") == OFFICIAL_OUTPUT_14M_0


@pytest.mark.parametrize("wav", sorted(REFERENCES))
def test_cer_against_reference(asr: StreamingAsr, request: pytest.FixtureRequest, wav: str) -> None:
    max_cer = MAX_CER[request.node.callspec.params["asr"]]
    hyp = _transcribe(asr, TEST_WAVS / wav)
    cer = jiwer.cer(REFERENCES[wav], hyp)
    assert cer <= max_cer, f"{wav}: CER={cer:.3f}，hyp={hyp}"


@pytest.mark.parametrize("wav", sorted(REFERENCES))
def test_result_independent_of_chunk_size(asr: StreamingAsr, wav: str) -> None:
    results = {ms: _transcribe(asr, TEST_WAVS / wav, ms) for ms in (10, 37, 320, 100_000)}
    assert len(set(results.values())) == 1, results


def test_partial_results_are_streaming(asr: StreamingAsr) -> None:
    """只送入前半段音訊時就要有 partial，且是最終結果的前綴。"""
    audio = load_wav(TEST_WAVS / "0.wav")
    stream = asr.create_stream()
    partials: list[str] = []
    for chunk in _chunks(audio[: len(audio) // 2], 10):
        stream.accept_waveform(chunk)
        partials.append(stream.get_partial_result())
    for chunk in _chunks(audio[len(audio) // 2 :], 10):
        stream.accept_waveform(chunk)
    final = stream.finalize()

    half = partials[-1]
    assert 0 < len(half) < len(final)
    assert final.startswith(half)
    # partial 隨音訊增加而變長，不會倒退
    assert all(len(a) <= len(b) for a, b in pairwise(partials))


def test_streams_are_independent(asr: StreamingAsr) -> None:
    expected = _transcribe(asr, TEST_WAVS / "0.wav")
    a, b = asr.create_stream(), asr.create_stream()
    for chunk in _chunks(load_wav(TEST_WAVS / "0.wav"), 100):
        a.accept_waveform(chunk)
    assert b.get_partial_result() == ""
    assert b.finalize() == ""
    assert a.finalize() == expected


# ---------- 非串流 SenseVoice ----------


@pytest.fixture(scope="module")
def sense_voice() -> StreamingAsr:
    try:
        return load_asr("sensevoice-small-int8", num_threads=1)
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")


@pytest.mark.parametrize("wav", sorted(REFERENCES))
def test_sense_voice_cer_and_chunk_independence(sense_voice: StreamingAsr, wav: str) -> None:
    # 兩句皆 0/N（language=auto，含語氣詞）
    results = {ms: _transcribe(sense_voice, TEST_WAVS / wav, ms) for ms in (10, 320, 100_000)}
    assert len(set(results.values())) == 1, results
    assert jiwer.cer(REFERENCES[wav], results[10]) <= 0.05


def test_sense_voice_has_no_partial_unless_requested(sense_voice: StreamingAsr) -> None:
    audio = load_wav(TEST_WAVS / "0.wav")
    stream = sense_voice.create_stream()
    for chunk in _chunks(audio, 10):
        stream.accept_waveform(chunk)
    assert stream.get_partial_result() == ""
    assert stream.finalize()

    pseudo = load_asr("sensevoice-small-int8-p480", num_threads=1).create_stream()
    partials = []
    for chunk in _chunks(
        audio[: SAMPLE_RATE * 2], 10
    ):  # 2 s：0.48 / 0.96 / 1.44 / 1.92 s 各重新辨識一次
        pseudo.accept_waveform(chunk)
        partials.append(pseudo.get_partial_result())
    assert partials[-1] and len(set(partials)) >= 2


def test_sense_voice_rejects_blank_penalty() -> None:
    with pytest.raises(ValueError, match="blank_penalty"):
        load_asr("sensevoice-small-int8", blank_penalty=1.0)


@pytest.mark.parametrize("wav", sorted(REFERENCES))
def test_sense_voice_ort_matches_sherpa(sense_voice: StreamingAsr, wav: str) -> None:
    # 直接用 onnxruntime 的版本：前處理與解碼照 sherpa-onnx 實作，結果要逐字相同（300 句對齊集也逐句相同）
    ort_asr = load_asr("sensevoice-small-int8-ort", num_threads=1)
    assert _transcribe(ort_asr, TEST_WAVS / wav) == _transcribe(sense_voice, TEST_WAVS / wav)


def test_sense_voice_ort_lfr_shape_and_edges() -> None:
    try:
        asr = load_asr("sensevoice-small-int8-ort", num_threads=1)
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")
    assert isinstance(asr, SenseVoiceOrtAsr)
    # 1 s 音訊：snip_edges 下 fbank 98 個 frame → LFR ceil(98 / 6) = 17 個 560 維 frame
    feats = asr.features(
        np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1
    )
    assert feats.shape == (17, 560)
    assert asr.features(np.zeros(100, dtype=np.float32)).shape == (0, 560)  # 不足一個 frame


# ---------- X-ASR 直接用 onnxruntime ----------

# X-ASR 壓縮檔自帶的中英混合測試音訊（0–3.wav），用來比對英文單字與空白的處理
X_ASR_TEST_WAVS = MODELS_DIR / X_ASR_480MS.model_dir / "test_wavs"


def test_tokens_to_text_matches_sherpa_spacing() -> None:
    # 中文之間不加空白，英文詞前後加空白，BPE 片段直接相接
    tokens = ["▁昨", "▁天", "▁是", "▁MO", "N", "DA", "Y", "▁IS", "▁礼", "▁拜"]
    assert tokens_to_text(tokens) == "昨天是 MONDAY IS 礼拜"
    assert tokens_to_text([]) == ""


def _transcribe_with_timestamps(asr: StreamingAsr, wav: Path) -> tuple[str, list[float] | None]:
    stream = asr.create_stream()
    for chunk in _chunks(load_wav(wav), 10):
        stream.accept_waveform(chunk)
    return stream.finalize(), stream.get_token_timestamps()


@pytest.mark.parametrize("model", ["x-asr-480ms-int8", "x-asr-160ms-int8"])
@pytest.mark.parametrize("blank_penalty", [0.0, 1.5])
def test_x_asr_ort_matches_sherpa(model: str, blank_penalty: float) -> None:
    # 前處理、encoder 串流與 greedy search 照 sherpa-onnx 實作：文字與 token 時間要逐字相同
    # （200 句冒煙集也逐句相同，見 docs/progress.md 2026-11-01）
    try:
        sherpa = load_asr(model, num_threads=1, blank_penalty=blank_penalty)
    except FileNotFoundError:
        pytest.skip(f"{model} 未下載，請先執行 bash scripts/download_models.sh")
    ort_asr = load_asr(f"{model}-ort", num_threads=1, blank_penalty=blank_penalty)
    wavs = sorted(X_ASR_TEST_WAVS.glob("*.wav")) + [TEST_WAVS / w for w in sorted(REFERENCES)]
    for wav in wavs:
        expected = _transcribe_with_timestamps(sherpa, wav)
        got = _transcribe_with_timestamps(ort_asr, wav)
        assert got[0] == expected[0], wav
        assert got[1] == pytest.approx(expected[1]), wav


def test_x_asr_ort_batched_joiner_matches_per_frame() -> None:
    # greedy search 把一個 chunk 的 frame 批次送進 joiner；決策必須與逐 frame 呼叫完全相同
    try:
        asr = load_asr("x-asr-480ms-int8-ort", num_threads=1, blank_penalty=1.5)
    except FileNotFoundError:
        pytest.skip("x-asr-480ms-int8 未下載，請先執行 bash scripts/download_models.sh")
    assert isinstance(asr, ZipformerOrtAsr)
    assert asr._batch_joiner, "ORT 圖最佳化後的 joiner 應逐列獨立（見 src/asr/zipformer_ort.py）"
    wavs = sorted(X_ASR_TEST_WAVS.glob("*.wav")) + [TEST_WAVS / w for w in sorted(REFERENCES)]
    batched = [_transcribe_with_timestamps(asr, wav) for wav in wavs]
    asr._batch_joiner = False
    per_frame = [_transcribe_with_timestamps(asr, wav) for wav in wavs]
    assert batched == per_frame


def test_x_asr_ort_rejects_sess_options_with_coreml() -> None:
    with pytest.raises(ValueError, match="不能同時給"):
        ZipformerOrtAsr(X_ASR_480MS, sess_options=object(), coreml=CoreMLOptions())


@pytest.mark.skipif(not coreml_available(), reason="onnxruntime 沒有 CoreMLExecutionProvider")
def test_x_asr_ort_coreml_encoder_close_to_cpu() -> None:
    # encoder 改用 CoreML EP（編譯約 15 s）：INT8 運算仍在 CPU、其餘在 Core ML 以 FP16 / FP32 計算，
    # 結果不保證逐位元相同，只要求與 CPU 版的文字接近（docs/progress.md 2026-11-02）
    if not (MODELS_DIR / X_ASR_480MS.model_dir).is_dir():
        pytest.skip("x-asr-480ms-int8 未下載，請先執行 bash scripts/download_models.sh")
    cpu = ZipformerOrtAsr(X_ASR_480MS)
    coreml = ZipformerOrtAsr(X_ASR_480MS, coreml=CoreMLOptions("CPUOnly"))
    assert "CoreMLExecutionProvider" in coreml._encoder.get_providers()
    assert coreml._decoder.get_providers() == ["CPUExecutionProvider"]
    wavs = sorted(X_ASR_TEST_WAVS.glob("*.wav"))
    expected = [_transcribe_with_timestamps(cpu, wav)[0] for wav in wavs]
    got = [_transcribe_with_timestamps(coreml, wav)[0] for wav in wavs]
    assert jiwer.cer(expected, got) < 0.05


# ---------- 偽串流的重新辨識時機（不需模型） ----------


def _decode_points(schedule: PartialSchedule | None, seconds: float, chunk_ms: int) -> list[float]:
    """以 chunk_ms 送入 seconds 秒音訊，回傳每次重新辨識時已累積的秒數。"""
    points: list[float] = []

    def recognize(audio: npt.NDArray[np.float32]) -> str:
        points.append(len(audio) / SAMPLE_RATE)
        return "字" * len(points)

    stream = SenseVoiceStream(recognize, schedule)
    for chunk in _chunks(np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), chunk_ms):
        stream.accept_waveform(chunk)
    assert stream.num_decodes == len(points)
    return points


def test_partial_schedule_every_first_and_max() -> None:
    assert _decode_points(None, 3.0, 10) == []
    assert _decode_points(PartialSchedule(0.48), 2.0, 10) == pytest.approx([0.48, 0.96, 1.44, 1.92])
    assert _decode_points(PartialSchedule(0.96, first_s=0.48), 3.0, 10) == pytest.approx(
        [0.48, 1.44, 2.40]
    )
    assert _decode_points(PartialSchedule(0.48, max_s=1.5), 3.0, 10) == pytest.approx(
        [0.48, 0.96, 1.44]
    )
    # 一次送入一大塊（例如恢復說話時重新解碼）：只辨識一次，之後的時機從目前長度往後排
    assert _decode_points(PartialSchedule(0.48), 2.0, 2000) == pytest.approx([2.0])
    with pytest.raises(ValueError):
        PartialSchedule(0.0)


def test_partial_schedule_text_and_finalize() -> None:
    calls: list[int] = []

    def recognize(audio: npt.NDArray[np.float32]) -> str:
        calls.append(len(audio))
        return f"{len(audio)}"

    stream = SenseVoiceStream(recognize, PartialSchedule(0.48, max_s=0.5))
    for chunk in _chunks(np.zeros(SAMPLE_RATE, dtype=np.float32), 10):
        stream.accept_waveform(chunk)
    assert stream.get_partial_result() == str(int(0.48 * SAMPLE_RATE))  # 超過 max_s 後不再更新
    assert stream.finalize() == str(SAMPLE_RATE)  # finalize 一律整段辨識
    assert len(calls) == 2


def test_sense_voice_compute_stats_count_every_decode() -> None:
    try:
        asr = load_asr("sensevoice-small-int8-ort-p480", num_threads=1)
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")
    assert asr.compute_stats() == {"num_decodes": 0, "decoded_audio_s": 0.0}
    stream = asr.create_stream()
    for chunk in _chunks(np.zeros(SAMPLE_RATE, dtype=np.float32), 10):
        stream.accept_waveform(chunk)
    stream.finalize()
    # 0.48 / 0.96 s 兩次偽 partial + finalize 整段 1 s
    assert asr.compute_stats() == pytest.approx({"num_decodes": 3, "decoded_audio_s": 2.44})
    asr.reset_compute_stats()
    assert asr.compute_stats() == {"num_decodes": 0, "decoded_audio_s": 0.0}


@pytest.mark.parametrize("ms", [240, 320, 400, 480])
def test_sense_voice_zh_models(ms: int) -> None:
    # 語言固定 zh 的偽串流（docs/progress.md 2026-11-02「SenseVoice 語言固定 zh 與第一次 partial」）
    try:
        asr = load_asr(f"sensevoice-small-int8-ort-p{ms}-960-max3-zh", num_threads=1)
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")
    assert isinstance(asr, SenseVoiceOrtAsr)
    assert asr.config.language == "zh"
    assert asr.config.schedule == PartialSchedule(0.96, first_s=ms / 1000, max_s=3.0)
    assert "language=zh" in asr.name


def test_sense_voice_zh_keeps_english_words() -> None:
    # 中英混合：固定 zh 不會把整句的英文詞認成中文；短前綴不再猜成英文碎片（auto 是「yah」）
    try:
        zh = SenseVoiceOrtAsr(SenseVoiceConfig(language="zh"))
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")
    if not X_ASR_TEST_WAVS.is_dir():
        pytest.skip("x-asr-480ms-int8 未下載，請先執行 bash scripts/download_models.sh")
    text = _transcribe(zh, X_ASR_TEST_WAVS / "0.wav")
    assert "monday" in text and "tomorrow" in text
    head = zh.create_stream()
    head.accept_waveform(load_wav(X_ASR_TEST_WAVS / "3.wav")[: int(0.48 * SAMPLE_RATE)])
    assert not head.finalize().isascii()


def test_default_threads_per_model() -> None:
    # SenseVoice onnxruntime 版預設 2 執行緒（結果不變、較快），其餘維持 1（X-ASR 多執行緒無收益）
    assert default_threads(DEFAULT_MODEL) == 2
    assert default_threads("sensevoice-small-int8-ort") == 2
    assert default_threads("x-asr-480ms-int8-ort") == 1
    assert default_threads("sensevoice-small-int8") == 1  # sherpa-onnx 版未量測，維持 1
