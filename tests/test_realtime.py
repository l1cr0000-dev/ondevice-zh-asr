"""eval/realtime.py 測試：長串流排列、時鐘換算（ADC 時刻 → t0、demo_mic 的估計）與逐句延遲指標。"""

import pytest

from eval.realtime import (
    FileSpan,
    SegmentTimes,
    adc_t0,
    assign_segments,
    clock_offset,
    demo_mic_t0,
    file_latencies,
    layout_spans,
)
from src.asr import SAMPLE_RATE


def test_layout_spans_positions() -> None:
    lead, tail = SAMPLE_RATE, SAMPLE_RATE // 2  # 1 s / 0.5 s
    spans, total = layout_spans(
        [("a", 2 * SAMPLE_RATE, 0.2, 1.8), ("b", SAMPLE_RATE, 0.1, 0.9)], lead, tail
    )
    assert total == (1 + 2 + 0.5 + 1 + 1 + 0.5) * SAMPLE_RATE
    a, b = spans
    assert (a.region_start_s, a.region_end_s) == (0.0, 3.5)
    assert a.onset_s == pytest.approx(1.2) and a.end_s == pytest.approx(2.8)
    assert (b.region_start_s, b.region_end_s) == (3.5, 6.0)
    assert b.onset_s == pytest.approx(4.6) and b.end_s == pytest.approx(5.4)


def test_adc_t0_uses_median() -> None:
    block = 160
    starts = [k * block for k in range(5)]
    adc = [100.0 + s / SAMPLE_RATE for s in starts]
    adc[2] += 0.05  # 個別 block 的時間戳抖動不影響
    assert adc_t0(adc, starts) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        adc_t0([], [])


def test_clock_offset() -> None:
    assert clock_offset([10.0, 11.0, 12.5], [5.0, 6.0, 7.0]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        clock_offset([1.0], [])


def test_demo_mic_t0_misses_delivery_delay() -> None:
    # 真實 t0 = 100；callback 在 block 擷取完後 4 ms（加上 0–2 ms 抖動）才被呼叫
    block = 160
    ends = [(k + 1) * block for k in range(10)]
    arrivals = [100.0 + e / SAMPLE_RATE + 0.004 + (k % 3) * 0.001 for k, e in enumerate(ends)]
    est = demo_mic_t0(arrivals, ends)
    assert est - 100.0 == pytest.approx(0.004)  # 取最小值濾掉抖動，但交付延遲本身被算進 t0


def _span(
    region: tuple[float, float] = (0.0, 10.0), onset: float = 1.0, end: float = 3.0
) -> FileSpan:
    return FileSpan("x", region[0], region[1], onset, end)


def test_assign_segments_by_vad_start() -> None:
    spans = [_span((0.0, 5.0)), _span((5.0, 10.0))]
    segs = [
        SegmentTimes("a", 1.0, 2.0, 3.0),
        SegmentTimes("b", 5.0, 6.0, 7.0),
        SegmentTimes("c", 11.0, None, None),
    ]
    out = assign_segments(spans, segs)
    assert [s.text for s in out[0]] == ["a"]
    assert [s.text for s in out[1]] == ["b"]  # 區間左閉右開；區間外的丟棄


def test_file_latencies_provisional_covers_end() -> None:
    seg = SegmentTimes("今天", 0.9, 2.0, 4.2, provisionals=[(3.3, 3.1)])
    lat = file_latencies(_span(onset=1.0, end=3.0), [seg])
    assert lat["first_token_vad_ms"] == pytest.approx(1100)
    assert lat["first_token_true_ms"] == pytest.approx(1000)
    assert lat["result_ms"] == pytest.approx(300)  # 暫定結果涵蓋到末字結束，比定稿早
    assert lat["final_ms"] == pytest.approx(1200)


def test_file_latencies_half_provisional_and_empty_first() -> None:
    # 第一個句段沒字（噪音觸發）；暫定結果只涵蓋半句 → 完整結果取最終結果
    noise = SegmentTimes("", 0.2, None, 0.5)
    seg = SegmentTimes("今天天气", 0.9, 1.8, 4.0, provisionals=[(2.4, 2.2)])
    lat = file_latencies(_span(onset=1.0, end=3.0), [noise, seg])
    assert lat["first_token_vad_ms"] == pytest.approx(900)
    assert lat["result_ms"] == pytest.approx(1000)
    assert file_latencies(_span(), []) == dict.fromkeys(lat)
