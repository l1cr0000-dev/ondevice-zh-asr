"""eval/pause.py：scripts/pause_analysis.py 的計算部分。"""

import numpy as np

from eval.pause import decoded_audio_ratio, is_boundary_provisional, longest_run_below


def test_longest_run_below() -> None:
    probs = np.array([0.9, 0.2, 0.3, 0.6, 0.1, 0.1, 0.1, 0.9])
    assert longest_run_below(probs, 0.5) == 3
    assert longest_run_below(probs, 0.7) == 6  # 0.6 也算低於 0.7
    assert longest_run_below(probs, 0.05) == 0


def test_is_boundary_provisional() -> None:
    sents = [{"start_s": 0.0, "end_s": 2.0}, {"start_s": 2.2, "end_s": 4.0}]
    assert is_boundary_provisional(2.3, sents)  # 第一句末字後 0.3 s（已進入下一句也算句間）
    assert not is_boundary_provisional(1.5, sents)  # 第一句中間
    assert not is_boundary_provisional(3.0, sents)  # 第二句中間
    assert is_boundary_provisional(4.2, sents)  # 最後一句之後


def test_decoded_audio_ratio_counts_only_resumed_provisionals() -> None:
    seg = {
        "speech_start_s": 1.0,
        "provisionals": [{"audio_end_s": 3.0}, {"audio_end_s": 6.0}],
        "resume_redecode_s": [0.1],  # 只恢復一次：第二個暫定結果沒有被重新解碼
    }
    result = {"utterances": [{"duration_s": 10.0, "segments": [seg]}]}
    assert decoded_audio_ratio(result) == 1 + 2.0 / 10.0
