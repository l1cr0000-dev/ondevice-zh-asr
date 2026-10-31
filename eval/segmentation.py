"""連續多句評測集（B4）的斷句指標：把 pipeline 的句段對應到已知的參考句子。

參考句子的起訖點是真實語音起訖（MFA），不依賴 VAD。句段與句子「重疊」指句段的 [speech_start_s,
speech_end_s] 與句子的真實語音區間交集超過 `MIN_OVERLAP_S`（避免 VAD padding 碰到邊緣就算）。

- 句間邊界被合併：有一個句段同時與前後兩句重疊。
- 句中誤切：一句話與 2 個以上句段重疊。
- 漏句：一句話沒有任何句段與它重疊。
- 每句尾字延遲：句子真實末字結束 → 與它重疊的最後一個句段的最終結果。兩句被合併時，前一句的完整文字
  要等合併後的句段結束才出現，延遲自然變大，這就是合併的代價。依「這句之後的停頓長度」分組報告。
- 每句結果延遲（有暫定結果時）：句子真實末字結束 → 第一個涵蓋整句的結果（暫定結果或最終結果，取早者）。
  暫定結果要涵蓋整句：它所在句段與這句重疊，且重新解碼到的音訊終點不早於這句的末字結束。
- 每句首字延遲（真實起點）：句子是某個句段的第一句時（沒有和前一句合併），句段的首字時刻 → 真實首字起點。
  與 VAD 無關（backlog B5）。
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from eval.metrics import summarize

MIN_OVERLAP_S = 0.05
LAST = "end"  # 每條最後一句（後面沒有句間停頓）的分組名稱


def _overlap(seg: dict[str, Any], start: float, end: float) -> float:
    seg_end = float(seg["speech_end_s"]) if seg["speech_end_s"] is not None else float("inf")
    return min(seg_end, end) - max(float(seg["speech_start_s"]), start)


def pause_key(pause_s: float | None) -> str:
    return LAST if pause_s is None else f"{pause_s:g}"


def match_file(
    sentences: Sequence[dict[str, Any]], segments: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """每句一筆：對應的句段索引、之後的停頓、是否與下一句合併、尾字 / 首字延遲（秒）。"""
    covers = [
        [
            j
            for j, seg in enumerate(segments)
            if _overlap(seg, s["start_s"], s["end_s"]) > MIN_OVERLAP_S
        ]
        for s in sentences
    ]
    out: list[dict[str, Any]] = []
    for i, s in enumerate(sentences):
        segs = covers[i]
        nxt = sentences[i + 1] if i + 1 < len(sentences) else None
        merged_next = nxt is not None and bool(set(segs) & set(covers[i + 1]))
        merged_prev = i > 0 and bool(set(segs) & set(covers[i - 1]))
        tail = first = result = None
        if segs:
            last = segments[segs[-1]]
            if last.get("final_s") is not None:
                tail = last["final_s"] - s["end_s"]
            ready = [
                p["emitted_s"]
                for p in last.get("provisionals", [])
                if p["audio_end_s"] >= s["end_s"]
            ]
            if tail is not None or ready:
                result = (
                    min([*ready, *([last["final_s"]] if tail is not None else [])]) - s["end_s"]
                )
            head = segments[segs[0]]
            if not merged_prev and head.get("e2e_first_token_latency_s") is not None:
                first = head["e2e_first_token_latency_s"] - s["start_s"]
        out.append(
            {
                "segments": segs,
                "pause_after_s": None if nxt is None else nxt["start_s"] - s["end_s"],
                "merged_with_next": merged_next,
                "starts_segment": bool(segs) and not merged_prev,
                "tail_latency_s": tail,
                "result_latency_s": result,
                "first_token_latency_s": first,
            }
        )
    return out


def summarize_segmentation(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """`files` 每筆含 `sentences`、`segments`、`pauses_s`（句間停頓的標稱值，用來分組）。"""
    boundaries: dict[str, list[bool]] = defaultdict(list)  # 停頓 → 是否被切開
    tails: dict[str, list[float]] = defaultdict(list)
    results: dict[str, list[float]] = defaultdict(list)
    first_ms: list[float] = []
    n_sent = split_inside = missed = exact_files = 0
    for f in files:
        rows = match_file(f["sentences"], f["segments"])
        exact = len(f["segments"]) == len(rows)
        for i, r in enumerate(rows):
            n_sent += 1
            missed += not r["segments"]
            split_inside += len(r["segments"]) > 1
            exact &= len(r["segments"]) == 1 and not r["merged_with_next"]
            key = pause_key(f["pauses_s"][i] if i < len(f["pauses_s"]) else None)
            if key != LAST:
                boundaries[key].append(not r["merged_with_next"])
            if r["tail_latency_s"] is not None:
                tails[key].append(r["tail_latency_s"] * 1000)
            if r["result_latency_s"] is not None:
                results[key].append(r["result_latency_s"] * 1000)
            if r["first_token_latency_s"] is not None:
                first_ms.append(r["first_token_latency_s"] * 1000)
        exact_files += exact
    order = sorted((k for k in tails if k != LAST), key=float) + [LAST]
    all_b = [b for v in boundaries.values() for b in v]
    return {
        "files": len(files),
        "sentences": n_sent,
        "exact_files": exact_files,  # 句段與句子一一對應的條數
        "boundary_detected": sum(all_b) / len(all_b) if all_b else None,
        "boundary_detected_by_pause": {
            k: {"n": len(boundaries[k]), "rate": sum(boundaries[k]) / len(boundaries[k])}
            for k in order
            if boundaries.get(k)
        },
        "sentences_split_inside": split_inside,
        "sentences_missed": missed,
        "tail_latency_ms_by_pause": {k: summarize(tails[k]) for k in order if k in tails},
        "tail_latency_ms": summarize([t for v in tails.values() for t in v]),
        # 暫定或最終結果，取早者；沒有暫定結果時與 tail_latency_ms_by_pause 相同
        "result_latency_ms_by_pause": {k: summarize(results[k]) for k in order if k in results},
        "first_token_latency_true_onset_ms": summarize(first_ms),
    }


def summarize_provisionals(utterances: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """暫定結果的統計（任何評測集）；沒有任何暫定結果時回傳 None。

    - 被取代：暫定結果之後同一句段又開始說話（不是句段的最後一個暫定結果），文字通常只是半句。
    - 最後一個暫定結果與最終結果相同的比例：靜音一路到定稿時，暫定結果是否就是最終文字。
    - 重新解碼耗時：暫定結果時刻 − 停頓事件時已送入的音訊終點（≈ 計算時間）。
    """
    total = superseded = segments = matched = 0
    compute_ms: list[float] = []
    for u in utterances:
        for seg in u["segments"]:
            provs = seg.get("provisionals", [])
            if not provs:
                continue
            segments += 1
            total += len(provs)
            superseded += len(provs) - 1
            matched += provs[-1]["text"] == seg["text"]
            compute_ms.extend((p["emitted_s"] - p["audio_end_s"]) * 1000 for p in provs)
    if not total:
        return None
    return {
        "provisionals": total,
        "segments_with_provisional": segments,
        "superseded": superseded,
        "last_matches_final": matched / segments,
        "redecode_ms": summarize(compute_ms),
    }
