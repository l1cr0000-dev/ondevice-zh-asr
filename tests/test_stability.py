"""partial 穩定度：純追加、改最後一字、整句重寫、暫定結果被取代等手造案例。"""

from typing import Any

import pytest

from eval.stability import display_sequence, segment_stability, summarize_stability


def _seg(
    partials: list[tuple[float, str]],
    final: tuple[float, str],
    provisionals: list[tuple[float, str]] | None = None,
) -> dict[str, Any]:
    return {
        "partials": [{"emitted_s": t, "text": x} for t, x in partials],
        "provisionals": [
            {"emitted_s": t, "audio_end_s": t, "text": x} for t, x in provisionals or []
        ],
        "final_s": final[0],
        "text": final[1],
    }


def _stab(seg: dict[str, Any]) -> Any:
    return segment_stability(display_sequence(seg))


def test_append_only_has_no_revision() -> None:
    s = _stab(
        _seg([(1.0, "今天"), (1.5, "今天天气"), (2.0, "今天天气很好")], (2.5, "今天天气很好"))
    )
    assert s.updates == 3  # 最終結果與最後一個 partial 相同，略過
    assert s.unstable == 0 and s.revised_tokens == 0
    assert s.final_tokens == 6
    # 每個字一出現就是對的
    assert s.stable_delay_s == pytest.approx([0.0] * 6)


def test_last_char_changed() -> None:
    s = _stab(_seg([(1.0, "今天天"), (1.5, "今天天汽")], (2.0, "今天天气")))
    assert s.unstable == 1 and s.revised_tokens == 1
    assert s.by_kind["partial"] == [2, 1, 1]  # 兩個 partial 被取代，第二個被最終結果改掉 1 字
    # 第 4 個字 1.5 s 第一次出現、2.0 s 才是對的
    assert s.stable_delay_s == pytest.approx([0.0, 0.0, 0.0, 0.5])


def test_whole_sentence_rewrite() -> None:
    s = _stab(_seg([(1.0, "北京"), (1.5, "背景音乐")], (2.0, "北京音乐")))
    # 北京 → 背景音乐：改 2 字；背景音乐 → 北京音乐：改 4 字（共同前綴 0）
    assert s.unstable == 2 and s.revised_tokens == 6
    # 「北」「京」1.0 s 出現，1.5 s 被改，2.0 s 才穩定；「音」「乐」1.5 s 出現、2.0 s 穩定
    assert s.stable_delay_s == pytest.approx([1.0, 1.0, 0.5, 0.5])


def test_punctuation_and_script_changes_are_not_revisions() -> None:
    s = _stab(_seg([(1.0, "今天"), (1.5, "今天，")], (2.0, "今天。天氣")))
    assert s.updates == 2 and s.unstable == 0


def test_deletion_counts_as_revision() -> None:
    s = _stab(_seg([(1.0, "我想去")], (2.0, "我想")))
    assert s.unstable == 1 and s.revised_tokens == 1


def test_provisional_superseded_by_resumed_speech() -> None:
    # A2：停頓出暫定結果「打开空」，恢復說話後重新解碼成「打开空调」（純追加）再改成「打开空调温度」
    seg = _seg(
        [(1.0, "打开"), (2.0, "打开空调"), (2.5, "打开空调温度")],
        (3.5, "打开空调温度"),
        provisionals=[(1.5, "打开空")],
    )
    seq = display_sequence(seg)
    assert [d.kind for d in seq] == ["partial", "provisional", "partial", "partial"]
    s = segment_stability(seq)
    assert s.unstable == 0
    assert s.by_kind["provisional"] == [1, 0, 0]

    # 暫定結果被改字
    seg = _seg(
        [(1.0, "打开"), (2.0, "打开空调")], (3.0, "打开空调"), provisionals=[(1.5, "打开孔")]
    )
    s = _stab(seg)
    assert s.by_kind["provisional"] == [1, 1, 1]


def test_summary() -> None:
    utts = [
        {"segments": [_seg([(1.0, "今天"), (1.5, "今天天气")], (2.0, "今天天气"))]},
        {"segments": [_seg([(1.0, "北京"), (1.5, "背景音乐")], (2.0, "北京音乐"))]},
    ]
    st = summarize_stability(utts)
    assert st is not None
    # 更新：第一條 1 次（今天 → 今天天气），第二條 2 次；修正 2 次
    assert st["updates"] == 3 and st["unstable_updates"] == 2
    assert st["unstable_update_rate"] == pytest.approx(2 / 3)
    assert st["revised_tokens"] == 6
    assert st["revised_token_ratio"] == pytest.approx(6 / 8)
    assert st["utterances_with_revision"] == pytest.approx(0.5)
    assert st["stable_on_first_ratio"] == pytest.approx(4 / 8)
    # 舊結果沒有逐次 partial 紀錄
    assert summarize_stability([{"segments": [{"final_s": 1.0, "text": "好"}]}]) is None
