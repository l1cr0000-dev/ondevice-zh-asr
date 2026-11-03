"""短停頓分析（backlog B9）：停頓事件抓不抓得到短的句間停頓，以及暫定結果的代價。只量測，不改演算法。

兩個子命令：
- screen：在連續多句集上逐 frame 跑 Silero，量每個句間停頓（真實末字結束 → 下一句真實首字起點，
  前後各放寬 150 ms）與每句句中（去掉頭尾 150 ms）「機率連續低於 x 的最長長度」。停頓事件要等這段
  長度滿 pause_ms（以 32 ms frame 為單位進位）才發出，所以可預先估計各 pause_ms / 機率門檻下，
  句間停頓抓到的比例與句中出現停頓事件的比例，不必跑 ASR。
- cost：讀 run_eval 的結果 JSON（對齊集或連續集，第一個當對照），印出每個設定的：
  - 結果延遲：真實末字結束 → 第一個涵蓋整句的結果（暫定或最終，取早者）。對齊集的真實末字取自
    alignments.jsonl；連續集取 run_eval 已算好的 metrics.segmentation。
  - 暫定結果的修正：對齊集為「先出現半句暫定結果（重新解碼到的音訊終點早於真實末字）」的句子比例；
    連續集把暫定結果分成句間（音訊終點落在某句真實末字之後 0.45 s 內，或最後一句之後）與句中。
  - 計算量（與機器負載無關）：解碼的音訊總長 / 音訊總長 = 1 + 恢復說話時重新解碼的音訊長度比例。
    同機有其他工作時 RTF 不穩，以這個比較計算量。
  - 最終文字與對照不同的條數、CER、RTF。

用法：
    uv run scripts/pause_analysis.py screen
    uv run scripts/pause_analysis.py cost eval/results/<對照>.json eval/results/<設定>.json ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.pause import (
    decoded_audio_ratio,
    full_result_latency_s,
    is_boundary_provisional,
    longest_run_below,
)

CONTINUOUS_MANIFEST = REPO_ROOT / "data" / "aishell1_continuous_300" / "manifest.jsonl"
ALIGNMENTS = REPO_ROOT / "data" / "aishell1_align_300" / "alignments.jsonl"
MARGIN_S = 0.15  # screen：句間停頓前後放寬、句中去掉頭尾的長度
PROB_THRESHOLDS = (0.35, 0.5, 0.65, 0.8, 0.9)
# 32 ms frame 數：96–224 ms；pause 120 / 150 / 190 / 200 分別要 4 / 5 / 6 / 7 個 frame
RUN_FRAMES = (3, 4, 5, 6, 7)


def screen() -> None:
    from src.audio import load_wav
    from src.vad import SileroVadModel

    model = SileroVadModel()
    frame = model.frame_size
    frame_s = frame / 16000
    gaps: list[tuple[float, dict[float, int]]] = []
    inside: list[dict[float, int]] = []
    for line in CONTINUOUS_MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        audio = load_wav(CONTINUOUS_MANIFEST.parent / row["wav"])
        model.reset()
        n = len(audio) // frame
        probs = np.array([model(audio[i * frame : (i + 1) * frame]) for i in range(n)])
        t0 = np.arange(n) * frame_s

        def runs(lo: float, hi: float) -> dict[float, int]:
            window = probs[(t0 + frame_s > lo) & (t0 < hi)]  # noqa: B023
            return {x: longest_run_below(window, x) for x in PROB_THRESHOLDS}

        sents = row["sentences"]
        for i, pause in enumerate(row["pauses_s"]):
            gaps.append(
                (pause, runs(sents[i]["end_s"] - MARGIN_S, sents[i + 1]["start_s"] + MARGIN_S))
            )
        inside.extend(runs(s["start_s"] + MARGIN_S, s["end_s"] - MARGIN_S) for s in sents)

    pauses = sorted({p for p, _ in gaps})[:3]
    print(f"{len(gaps)} 個句間停頓、{len(inside)} 句；連續低於門檻的最長長度 ≥ N 個 frame 的比例")
    print("| 機率門檻 | 最長 ≥ (ms) | " + " | ".join(f"句間 {p:g} s" for p in pauses) + " | 句中 |")
    print("|---|---|" + "---|" * (len(pauses) + 1))
    for x in PROB_THRESHOLDS:
        for k in RUN_FRAMES:
            hit = [np.mean([r[x] >= k for p, r in gaps if p == pause]) for pause in pauses]
            fa = np.mean([r[x] >= k for r in inside])
            cells = " | ".join(f"{h:.0%}" for h in hit)
            print(f"| < {x:g} | {k * frame_s * 1000:.0f} | {cells} | {fa:.0%} |")


def _align_cost(result: dict[str, Any], ends: dict[str, float]) -> str:
    lat: list[float] = []
    half_utts = half = 0
    for u in result["utterances"]:
        true_end = ends[u["id"]]
        n_half = sum(
            p["audio_end_s"] < true_end for seg in u["segments"] for p in seg["provisionals"]
        )
        latency = full_result_latency_s(u, true_end)
        if latency is not None:
            lat.append(latency * 1000)
        half += n_half
        half_utts += n_half > 0
    n = len(result["utterances"])
    return (
        f"結果延遲 mean / P50 / P90 {np.mean(lat):.0f} / {np.percentile(lat, 50):.0f} / "
        f"{np.percentile(lat, 90):.0f} ms（n={len(lat)}）；半句暫定結果 {half} 個、"
        f"{half_utts}/{n} 句（{half_utts / n:.0%}）"
    )


def _continuous_cost(result: dict[str, Any]) -> str:
    sg = result["metrics"]["segmentation"]
    boundary = mid = 0
    for u in result["utterances"]:
        for seg in u["segments"]:
            for p in seg["provisionals"]:
                if is_boundary_provisional(p["audio_end_s"], u["sentences"]):
                    boundary += 1
                else:
                    mid += 1
    lat = " | ".join(
        f"{k} {v['mean']:.0f}/{v['p90']:.0f}" for k, v in sg["result_latency_ms_by_pause"].items()
    )
    cut = " ".join(f"{k} {v['rate']:.0%}" for k, v in sg["boundary_detected_by_pause"].items())
    return (
        f"結果延遲 mean/P90（依停頓）{lat}\n    句間切開 {cut}；句中誤切 "
        f"{sg['sentences_split_inside']}；暫定結果 句間 {boundary} / 句中 {mid}"
    )


def cost(paths: list[Path]) -> None:
    ends: dict[str, float] = {}
    if ALIGNMENTS.is_file():
        for line in ALIGNMENTS.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            ends[row["id"]] = row["chars"][-1]["end"]
    base: dict[str, str] | None = None
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        m = result["metrics"]
        hyps = {u["id"]: u["hyp"] for u in result["utterances"]}
        diff = "—" if base is None else str(sum(hyps[k] != base.get(k) for k in hyps))
        base = base or hyps
        pv = m["provisional"] or {}
        resume = [
            t for u in result["utterances"] for s in u["segments"] for t in s["resume_redecode_s"]
        ]
        if m.get("segmentation"):
            detail = _continuous_cost(result)
        elif all(u["id"] in ends for u in result["utterances"]):
            detail = _align_cost(result, ends)
        else:
            detail = "（沒有真實末字時間，只列共同指標）"
        print(path.name)
        print(
            f"    CER {m['cer'] * 100:.2f}%；RTF {m['rtf']:.4f}；解碼音訊 / 音訊 "
            f"{decoded_audio_ratio(result):.3f}；暫定結果 {pv.get('provisionals', 0)} 個（被取代 "
            f"{pv.get('superseded', 0)}）；恢復時重新解碼 {len(resume)} 次"
            + (f" mean {np.mean(resume) * 1000:.0f} ms" if resume else "")
            + f"；最終文字與對照不同 {diff} 條"
        )
        print(f"    {detail}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("screen", help="VAD 機率篩選：句間 / 句中低機率段的長度分佈（連續多句集）")
    c = sub.add_parser("cost", help="暫定結果的延遲與代價（run_eval 結果 JSON，第一個當對照）")
    c.add_argument("results", type=Path, nargs="+", help="run_eval 的結果 JSON")
    args = p.parse_args()
    if args.command == "screen":
        if not CONTINUOUS_MANIFEST.is_file():
            print(
                f"錯誤：找不到 {CONTINUOUS_MANIFEST}，先執行 uv run scripts/prepare_continuous_data.py",
                file=sys.stderr,
            )
            return 1
        screen()
    else:
        missing = [str(r) for r in args.results if not r.is_file()]
        if missing:
            print(f"錯誤：找不到 {', '.join(missing)}", file=sys.stderr)
            return 1
        cost(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
