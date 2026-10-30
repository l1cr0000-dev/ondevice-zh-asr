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

from src.asr import SAMPLE_RATE, AsrStream, StreamingAsr, available_models, load_asr
from src.asr.sherpa import ZIPFORMER_ZH_14M
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
