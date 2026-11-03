"""暫定結果（停頓事件）的分析指標，供 scripts/pause_analysis.py 使用（backlog B9）。"""

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

BOUNDARY_WINDOW_S = 0.45  # 句末之後多久內的暫定結果算句間（pause ≤ 224 ms + pad + 發音收尾）


def longest_run_below(probs: npt.NDArray[np.float64], threshold: float) -> int:
    """機率連續低於 threshold 的最長 frame 數。"""
    best = cur = 0
    for below in probs < threshold:
        cur = cur + 1 if below else 0
        best = max(best, cur)
    return best


def is_boundary_provisional(audio_end_s: float, sentences: list[dict[str, Any]]) -> bool:
    """暫定結果的音訊終點落在某句真實末字之後 BOUNDARY_WINDOW_S 內（或最後一句之後）= 句間停頓。"""
    return audio_end_s >= sentences[-1]["end_s"] or any(
        s["end_s"] <= audio_end_s <= s["end_s"] + BOUNDARY_WINDOW_S for s in sentences
    )


def max_char_gap(chars: list[dict[str, Any]]) -> float:
    """MFA 字級對齊中，相鄰兩字之間最長的空隙（前一字結束 → 下一字起點，秒）；只有一個字時為 0。"""
    return max((b["start"] - a["end"] for a, b in itertools.pairwise(chars)), default=0.0)


def final_latency_s(utterance: dict[str, Any], true_end_s: float) -> float | None:
    """說完 → 定稿：真實末字結束 → 最後一個句段的最終結果（單句集）。沒有句段或沒有定稿時為 None。

    與 `tail_latency_s`（VAD 終點起算）不同，起點不依賴 VAD；最後一個句段若是句中誤切出來的，
    它的最終結果仍是這句最後一個字定稿的時刻。
    """
    segs = utterance["segments"]
    if not segs or segs[-1]["final_s"] is None:
        return None
    return float(segs[-1]["final_s"] - true_end_s)


def decoded_audio_ratio(result: dict[str, Any]) -> float:
    """解碼的音訊總長 / 音訊總長：原 stream 解碼一次，每次恢復說話再從句段開頭重新解碼到停頓處。"""
    total = sum(u["duration_s"] for u in result["utterances"])
    redecoded = sum(
        p["audio_end_s"] - seg["speech_start_s"]
        for u in result["utterances"]
        for seg in u["segments"]
        for p in seg["provisionals"][: len(seg["resume_redecode_s"])]
    )
    return float(1 + redecoded / total)


def full_result_latency_s(utt: dict[str, Any], true_end_s: float) -> float | None:
    """說完 → 完整結果：真實末字結束 → 第一個涵蓋整句的結果（音訊終點不早於真實末字的暫定結果，
    或最後一個句段的最終結果，取早者）。都沒有時回傳 None。"""
    ready = [
        p["emitted_s"]
        for seg in utt["segments"]
        for p in seg["provisionals"]
        if p["audio_end_s"] >= true_end_s
    ]
    if utt["segments"] and utt["segments"][-1]["final_s"] is not None:
        ready.append(utt["segments"][-1]["final_s"])
    return min(ready) - true_end_s if ready else None
