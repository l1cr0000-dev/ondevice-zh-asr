"""真實時鐘路徑（麥克風 / 軟體回放）延遲量測的時間換算與逐句指標（B0，scripts/mic_latency.py 用）。

時間軸：所有時刻都換成「音訊時間軸」上的秒數，0 = 第 0 個樣本被擷取（ADC）的時刻。
- 模擬路徑：pipeline 的 `*_emitted_s` 本來就在這個時間軸上（假設樣本依實際時長到達、零交付延遲）。
- 真實時鐘路徑：事件的牆鐘時刻（`time.perf_counter()`）減掉 t0（第 0 個樣本的牆鐘擷取時刻）。
  t0 由 PortAudio 的 `inputBufferAdcTime`（該 block 第一個樣本的擷取時刻）推得；軟體回放則是回放起點。

指標（每條音訊一個值；與 eval/run_eval.py、scripts/pause_analysis.py 的定義相同）：
- 首字延遲（VAD 起點）：第一個有字句段的首字時刻 − VAD 起點（含 speech_pad_ms）。
- 首字延遲（真實起點）：同一首字時刻 − 強制對齊（MFA）的第一個字開始。
- 說完 → 完整結果：真實末字結束 → 第一個涵蓋整句的結果（`audio_end_s` 不早於末字結束的暫定結果，
  或最後一個句段的最終結果，取早者）。
- 說完 → 定稿：真實末字結束 → 最後一個句段的最終結果（含 min_silence 等待）。
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.asr import SAMPLE_RATE


@dataclass(frozen=True)
class FileSpan:
    """長串流中的一條音訊。全部是音訊時間軸上的秒數。"""

    id: str
    region_start_s: float  # 這條音訊（含前補靜音）的起點；句段以 VAD 起點落在哪個區間歸屬
    region_end_s: float
    onset_s: float  # 強制對齊的第一個字開始
    end_s: float  # 強制對齊的最後一個字結束


@dataclass
class SegmentTimes:
    """一個句段的時刻，全部在音訊時間軸上（秒）。"""

    text: str
    speech_start_s: float  # VAD 起點（含 speech_pad_ms）
    first_token_s: float | None
    final_s: float | None
    # (暫定結果時刻, 重新解碼涵蓋到的音訊終點)
    provisionals: list[tuple[float, float]] = field(default_factory=list)


def layout_spans(
    rows: Sequence[tuple[str, int, float, float]], lead: int, tail: int
) -> tuple[list[FileSpan], int]:
    """把多條音訊排成一條長串流：每條前補 `lead`、後補 `tail` 個樣本的靜音。

    rows：(id, 樣本數, 對齊首字開始秒數, 對齊末字結束秒數)，秒數以該條音訊開頭為 0。
    回傳每條的 FileSpan 與串流總樣本數。
    """
    spans: list[FileSpan] = []
    pos = 0
    for uid, n, onset, end in rows:
        audio_start = pos + lead
        nxt = audio_start + n + tail
        spans.append(
            FileSpan(
                id=uid,
                region_start_s=pos / SAMPLE_RATE,
                region_end_s=nxt / SAMPLE_RATE,
                onset_s=audio_start / SAMPLE_RATE + onset,
                end_s=audio_start / SAMPLE_RATE + end,
            )
        )
        pos = nxt
    return spans, pos


def adc_t0(adc_times: Sequence[float], block_starts: Sequence[int]) -> float:
    """第 0 個樣本的擷取時刻：每個 block 的 ADC 時刻 − 該 block 起始樣本的時長，取中位數（濾掉個別抖動）。"""
    if not adc_times or len(adc_times) != len(block_starts):
        raise ValueError("adc_times 與 block_starts 需等長且非空")
    return statistics.median(
        t - s / SAMPLE_RATE for t, s in zip(adc_times, block_starts, strict=True)
    )


def clock_offset(wall: Sequence[float], stream_time: Sequence[float]) -> float:
    """PortAudio 串流時鐘 → 牆鐘（perf_counter）的偏移：callback 內同時讀到的兩個時刻相減，取中位數。

    macOS Core Audio 下兩者是同一個時鐘（偏移只有 callback 內兩次讀時間之間的幾十 µs）。
    """
    if not wall or len(wall) != len(stream_time):
        raise ValueError("wall 與 stream_time 需等長且非空")
    return statistics.median(w - s for w, s in zip(wall, stream_time, strict=True))


def demo_mic_t0(arrivals: Sequence[float], block_ends: Sequence[int]) -> float:
    """scripts/demo_mic.py 的 t0 估計：min(callback 時刻 − 已收樣本時長)。

    假設 callback 在 block 最後一個樣本擷取的瞬間被呼叫，因此不含驅動 / 硬體的交付延遲，
    比以 ADC 時刻推得的 t0 晚，算出的延遲偏小。
    """
    if not arrivals or len(arrivals) != len(block_ends):
        raise ValueError("arrivals 與 block_ends 需等長且非空")
    return min(t - e / SAMPLE_RATE for t, e in zip(arrivals, block_ends, strict=True))


def assign_segments(
    spans: Sequence[FileSpan], segments: Sequence[SegmentTimes]
) -> list[list[SegmentTimes]]:
    """依 VAD 起點把句段歸到所屬的音訊（區間 [region_start, region_end)）；區間外的丟棄。"""
    out: list[list[SegmentTimes]] = [[] for _ in spans]
    for seg in segments:
        for i, sp in enumerate(spans):
            if sp.region_start_s <= seg.speech_start_s < sp.region_end_s:
                out[i].append(seg)
                break
    return out


def file_latencies(span: FileSpan, segs: Sequence[SegmentTimes]) -> dict[str, float | None]:
    """一條音訊的延遲指標（ms），定義見模組說明；算不出的為 None。"""
    result: dict[str, float | None] = {
        "first_token_vad_ms": None,
        "first_token_true_ms": None,
        "result_ms": None,
        "final_ms": None,
    }
    first = next((s for s in segs if s.text.strip() and s.first_token_s is not None), None)
    if first is not None and first.first_token_s is not None:
        result["first_token_vad_ms"] = (first.first_token_s - first.speech_start_s) * 1000
        result["first_token_true_ms"] = (first.first_token_s - span.onset_s) * 1000
    ready = [t for s in segs for t, end in s.provisionals if end >= span.end_s]
    if segs and segs[-1].final_s is not None:
        ready.append(segs[-1].final_s)
        result["final_ms"] = (segs[-1].final_s - span.end_s) * 1000
    if ready:
        result["result_ms"] = (min(ready) - span.end_s) * 1000
    return result
