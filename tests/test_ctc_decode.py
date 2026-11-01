"""CTC 解碼（src.asr.ctc_decode）：greedy、prefix beam search、熱詞自動機的加分帳。

純演算法部分用人造的 log 機率，不需要模型；最後幾個測試用 SenseVoice 模型（未下載則跳過）。
"""

import numpy as np
import numpy.typing as npt
import pytest

from src.asr import DecodeOptions, SenseVoiceOrtAsr, load_asr
from src.asr.ctc_decode import (
    BLANK,
    HotwordGraph,
    HotwordState,
    ctc_greedy,
    ctc_prefix_beam_search,
    tokenize_hotword,
)
from src.asr.sherpa import ZIPFORMER_ZH_14M
from src.paths import MODELS_DIR
from src.vad import load_wav

VOCAB = 8


def _frames(*rows: dict[int, float]) -> npt.NDArray[np.float32]:
    """每個 frame 給 {token: 機率}，其餘 token 平分剩下的機率，回傳 log 機率。"""
    out = np.empty((len(rows), VOCAB), dtype=np.float32)
    for i, row in enumerate(rows):
        rest = (1 - sum(row.values())) / (VOCAB - len(row))
        p = np.full(VOCAB, rest)
        for t, v in row.items():
            p[t] = v
        out[i] = np.log(p)
    return out


def _walk(graph: HotwordGraph, tokens: list[int]) -> tuple[float, HotwordState]:
    total, state = 0.0, graph.start
    for t in tokens:
        delta, state = graph.step(state, t)
        total += delta
    return total, state


def test_ctc_greedy_merges_repeats_and_drops_blank() -> None:
    lp = _frames({1: 0.9}, {1: 0.9}, {BLANK: 0.9}, {1: 0.9}, {2: 0.9}, {BLANK: 0.9})
    assert ctc_greedy(lp) == [1, 1, 2]


def test_beam_search_accepts_unnormalized_logits() -> None:
    rng = np.random.default_rng(0)
    lp = _frames(*({int(t): 0.6} for t in rng.integers(0, VOCAB, size=20)))
    graph = HotwordGraph([[1, 2]], score=1.0)
    assert ctc_prefix_beam_search(lp * 1.0 + 7.0, 4, graph) == ctc_prefix_beam_search(lp, 4, graph)


def test_beam_search_without_hotwords_matches_greedy_on_peaky_frames() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        ids = rng.integers(0, VOCAB, size=30)
        lp = _frames(*({int(t): 0.95} for t in ids))
        for beam in (1, 4, 8):
            assert ctc_prefix_beam_search(lp, beam) == ctc_greedy(lp)


def test_beam_search_sums_alignments() -> None:
    # 每個 frame 的 argmax 都是 blank，greedy 輸出空；但「1」的各種對齊加總後機率較高
    lp = _frames({BLANK: 0.54, 1: 0.44}, {BLANK: 0.54, 1: 0.44})
    assert ctc_greedy(lp) == []
    assert ctc_prefix_beam_search(lp, beam=4) == [1]  # 0.44² + 2·0.44·0.54 ≈ 0.67 > 0.54² ≈ 0.29


def test_hotword_graph_scoring() -> None:
    g = HotwordGraph([[1, 2, 3], [1, 2, 3, 4], [5, 6]], score=1.5)
    # 拼完熱詞：永久加分 = 長度 × score，結束時不扣
    total, state = _walk(g, [7, 1, 2, 3, 7])
    assert total == pytest.approx(4.5) and g.finalize(state) == 0.0
    # 拼完較短的再拼較長的：兩個都算
    total, state = _walk(g, [1, 2, 3, 4])
    assert total == pytest.approx(4.5 + 1.5) and g.finalize(state) == 0.0
    # 拼完較短的、較長的拼到一半結束：扣回較長那一截的暫時加分
    total, state = _walk(g, [1, 2, 3])
    assert total + g.finalize(state) == pytest.approx(4.5)
    # 拼到一半失敗：加分全部扣回
    total, state = _walk(g, [1, 2, 7])
    assert total == pytest.approx(0.0) and state[0] is g.root
    # 拼到一半結束：finalize 扣回
    total, state = _walk(g, [5])
    assert total == pytest.approx(1.5) and g.finalize(state) == pytest.approx(-1.5)
    # 失敗時沿 fail 退到另一個熱詞的字首（1 2 | 1 2 3）
    total, state = _walk(g, [1, 2, 1, 2, 3])
    assert total + g.finalize(state) == pytest.approx(4.5)


def test_hotword_flips_near_tie_and_score_zero_does_not() -> None:
    # 第 2 個字 3 vs 4 幾乎平手（0.5 vs 0.4），模型偏好 3
    lp = _frames({1: 0.95}, {BLANK: 0.95}, {3: 0.5, 4: 0.4}, {BLANK: 0.95})
    assert ctc_prefix_beam_search(lp, beam=4) == [1, 3]
    assert ctc_prefix_beam_search(lp, 4, HotwordGraph([[1, 4]], score=1.0)) == [1, 4]
    assert ctc_prefix_beam_search(lp, 4, HotwordGraph([[1, 4]], score=0.0)) == [1, 3]
    # 拼到一半的熱詞（1 4 6）不能因為暫時加分就贏：結束時會扣回
    assert ctc_prefix_beam_search(lp, 4, HotwordGraph([[1, 4, 6]], score=1.0)) == [1, 3]


def test_hotword_does_not_delete_following_token() -> None:
    # 熱詞「1 2」後面接 3：拼完後接任何字都不該被扣分（舊的 strict mode 會讓不出字的前綴暫時領先）
    lp = _frames({1: 0.9}, {BLANK: 0.9}, {2: 0.9}, {3: 0.6, BLANK: 0.38}, {BLANK: 0.9})
    graph = HotwordGraph([[1, 2]], score=3.0)
    assert ctc_prefix_beam_search(lp, 1, graph) == [1, 2, 3]
    assert ctc_prefix_beam_search(lp, 4, graph) == [1, 2, 3]


def test_blank_skip_frames_still_allow_repeat_of_last_token() -> None:
    # 中間兩個 frame blank 機率極高被跳過；跳過時仍要保留「重複最後一個 token」的路徑
    lp = _frames({1: 0.99}, {BLANK: 0.9995}, {BLANK: 0.9995}, {2: 0.99})
    assert ctc_prefix_beam_search(lp, 4) == [1, 2] == ctc_greedy(lp)


def test_tokenize_hotword() -> None:
    vocab = {"北": 1, "京": 2, "▁hello": 3, "▁he": 4, "llo": 5, "▁ok": 6}
    assert tokenize_hotword("北京", vocab) == [1, 2]
    assert tokenize_hotword("hello", vocab) == [3]  # 最長匹配
    assert tokenize_hotword("北京 ok", vocab) == [1, 2, 6]
    assert tokenize_hotword("南京", vocab) is None
    assert tokenize_hotword("  ", vocab) is None


# ---------- SenseVoice 模型 ----------

TEST_WAVS = MODELS_DIR / ZIPFORMER_ZH_14M.model_dir / "test_wavs"


@pytest.fixture(scope="module")
def sense_voice_ort() -> SenseVoiceOrtAsr:
    try:
        asr = load_asr("sensevoice-small-int8-ort", num_threads=1)
    except FileNotFoundError:
        pytest.skip("SenseVoice 未下載，請先執行 bash scripts/download_models.sh")
    assert isinstance(asr, SenseVoiceOrtAsr)
    if not (TEST_WAVS / "0.wav").is_file():
        pytest.skip("zh-14M 測試音訊未下載")
    return asr


def test_sense_voice_default_decoder_is_greedy(sense_voice_ort: SenseVoiceOrtAsr) -> None:
    # 預設設定的解碼與改動前的 greedy 程式逐字相同（舊程式照抄在此當對照），名稱也不變
    assert sense_voice_ort.decoder.greedy
    assert "beam" not in sense_voice_ort.name and "hotword" not in sense_voice_ort.name
    for wav in ("0.wav", "1.wav"):
        logits = sense_voice_ort.logits(load_wav(TEST_WAVS / wav))
        assert logits is not None
        ids = logits.argmax(axis=-1)
        tokens: list[int] = []
        prev = -1
        for y in ids.tolist():
            if y != 0 and y != prev:
                tokens.append(y)
            prev = y
        old = "".join(sense_voice_ort._tokens[t] for t in tokens[4:]).replace("▁", " ").strip()
        assert sense_voice_ort.decode(logits) == old
        # 前 4 個 frame 是語言 / 情緒 / 事件 / ITN 的特殊 token，beam search 直接略過它們
        assert logits[:4].argmax(axis=-1).tolist() == tokens[:4]
        # 不帶熱詞的 beam search 在這兩句與 greedy 相同
        assert sense_voice_ort.make_decoder(beam=8).decode(logits) == old


def test_sense_voice_hotword_options_reach_decoder(sense_voice_ort: SenseVoiceOrtAsr) -> None:
    opts = DecodeOptions(beam=4, hotwords=("金融动荡", "不存在的詞☃"), hotword_score=2.0)
    for name in ("sensevoice-small-int8-ort", "sensevoice-small-int8-ort-p480"):
        asr = load_asr(name, decode=opts)
        assert isinstance(asr, SenseVoiceOrtAsr)
        assert asr.decoder.beam == 4 and asr.decoder.graph is not None
        assert asr.decoder.graph.num_words == 1 and asr.decoder.skipped_hotwords == ["不存在的詞☃"]
        assert "beam=4 hotwords=1 hotword_score=2" in asr.name
    stream = asr.create_stream()
    stream.accept_waveform(load_wav(TEST_WAVS / "1.wav"))
    assert "金融动荡" in stream.finalize()


@pytest.mark.parametrize(
    "name", ["sensevoice-small-int8", "x-asr-480ms-int8-ort", "zipformer-zh-14m"]
)
def test_hotwords_rejected_by_other_backends(name: str) -> None:
    with pytest.raises(ValueError, match="熱詞"):
        load_asr(name, decode=DecodeOptions(hotwords=("北京",)))
    with pytest.raises(ValueError, match="熱詞"):
        load_asr(name, decode=DecodeOptions(beam=4))
