"""暫定結果（停頓事件）的分析指標，供 scripts/pause_analysis.py 使用（backlog B9）。"""

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
