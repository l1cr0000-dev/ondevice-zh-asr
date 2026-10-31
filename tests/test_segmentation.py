"""連續多句集的斷句指標：用手造的句段檢查合併、句中誤切、漏句與延遲的歸屬。"""

from typing import Any

import pytest

from eval.segmentation import match_file, summarize_provisionals, summarize_segmentation

# 三句話：0–2 s、2.3–4 s（停頓 0.3 s）、5–6 s（停頓 1.0 s）
SENTENCES = [
    {"start_s": 0.0, "end_s": 2.0},
    {"start_s": 2.3, "end_s": 4.0},
    {"start_s": 5.0, "end_s": 6.0},
]


def _seg(start: float, end: float, final: float, first: float) -> dict[str, Any]:
    return {
        "speech_start_s": start,
        "speech_end_s": end,
        "final_s": final,
        "e2e_first_token_latency_s": first,
    }


def test_exact_segmentation() -> None:
    segs = [_seg(-0.03, 2.03, 2.6, 1.0), _seg(2.27, 4.03, 4.6, 3.3), _seg(4.97, 6.03, 6.6, 6.0)]
    rows = match_file(SENTENCES, segs)
    assert [r["segments"] for r in rows] == [[0], [1], [2]]
    assert not any(r["merged_with_next"] for r in rows)
    assert [r["tail_latency_s"] for r in rows] == pytest.approx([0.6, 0.6, 0.6])
    assert [r["first_token_latency_s"] for r in rows] == pytest.approx([1.0, 1.0, 1.0])


def test_short_pause_merged_delays_first_sentence() -> None:
    # 前兩句併成一段：第一句的完整文字要等到第二句結束後才出現；第二句沒有自己的首字
    segs = [_seg(-0.03, 4.03, 4.6, 1.0), _seg(4.97, 6.03, 6.6, 6.0)]
    rows = match_file(SENTENCES, segs)
    assert rows[0]["merged_with_next"] and not rows[1]["merged_with_next"]
    assert rows[0]["tail_latency_s"] == pytest.approx(2.6)
    assert rows[1]["first_token_latency_s"] is None and not rows[1]["starts_segment"]


def test_split_inside_and_missed() -> None:
    # 第一句在 1.0 s 被切開；第三句完全沒偵測到；句段邊緣只碰到下一句 30 ms 不算重疊
    segs = [_seg(0.0, 1.0, 1.5, 0.8), _seg(1.3, 2.33, 2.9, 1.9), _seg(2.4, 4.0, 4.5, 3.2)]
    rows = match_file(SENTENCES, segs)
    assert rows[0]["segments"] == [0, 1] and not rows[0]["merged_with_next"]
    assert rows[0]["tail_latency_s"] == pytest.approx(0.9)
    assert rows[2]["segments"] == [] and rows[2]["tail_latency_s"] is None

    s = summarize_segmentation([{"sentences": SENTENCES, "segments": segs, "pauses_s": [0.3, 1.0]}])
    assert s["sentences_split_inside"] == 1 and s["sentences_missed"] == 1
    assert s["exact_files"] == 0
    assert s["boundary_detected_by_pause"] == {
        "0.3": {"n": 1, "rate": 1.0},
        "1": {"n": 1, "rate": 1.0},
    }
    assert list(s["tail_latency_ms_by_pause"]) == ["0.3", "1"]  # 最後一句漏掉，沒有尾字延遲


def test_result_latency_uses_provisional_covering_the_sentence() -> None:
    # 前兩句併成一段，但第一句結束後 0.3 s 的停頓產生了暫定結果：第一句的結果延遲取暫定結果
    merged = _seg(-0.03, 4.03, 4.6, 1.0)
    merged["text"] = "AB"
    merged["provisionals"] = [
        {"emitted_s": 1.0, "audio_end_s": 0.9, "text": "a"},  # 句中停頓，不涵蓋第一句
        {"emitted_s": 2.25, "audio_end_s": 2.2, "text": "A"},
        {"emitted_s": 4.4, "audio_end_s": 4.3, "text": "AB"},
    ]
    rows = match_file(SENTENCES[:2], [merged])
    assert rows[0]["tail_latency_s"] == pytest.approx(2.6)
    assert rows[0]["result_latency_s"] == pytest.approx(0.25)
    assert rows[1]["result_latency_s"] == pytest.approx(0.4)

    p = summarize_provisionals([{"segments": [merged, _seg(4.97, 6.03, 6.6, 6.0)]}])
    assert p is not None
    assert p["provisionals"] == 3 and p["superseded"] == 2 and p["segments_with_provisional"] == 1
    assert p["last_matches_final"] == 1.0
    assert p["redecode_ms"]["mean"] == pytest.approx(250 / 3)
    assert summarize_provisionals([{"segments": [_seg(0.0, 1.0, 1.5, 0.5)]}]) is None
