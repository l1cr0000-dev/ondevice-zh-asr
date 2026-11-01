"""偽串流降計算量的取捨表：讀 run_eval 的結果 JSON，同一批句子上逐項比較。只讀結果，不跑模型。

偽串流 = SenseVoice Small 每隔一段時間把整段音訊重新辨識一次當 partial（`sensevoice-small-int8-ort-p*`）；
對照 = 預設的真串流 X-ASR 480 ms（`x-asr-480ms-int8-ort`）。全部用同一個斷句設定（預設 A2）：

- 對齊集 / 短句子集（`scripts/prepare_short_subset.py`，MFA 語音長度 ≤ 3 s）：以 MFA 真實首字起點起算的
  首字延遲、真實末字結束 →「第一個涵蓋整句的結果」（暫定或最終）。
- 連續多句集：每句真實起點的首字延遲（句子是句段的第一句時）與每句的結果延遲。
- 其他評測集（例如冒煙集）：端到端首字延遲（補靜音結束起算，只適合相對比較）。

首字延遲報兩種終點：
- partial：第一個非空 partial（現行 `first_token_latency` 的口徑）。
- 可見：第一次看到任何文字 = 非空 partial、暫定結果、最終結果取最早。沒有偽 partial 的 SenseVoice
  只在停頓時（暫定結果）才出字，要用這個口徑比較才公平。

CER 差是同一批句子的成對 bootstrap（以句重抽 10000 次），以 `--baseline` 為 A。

需要（預設 A2，jobs 1）：
    uv run scripts/prepare_short_subset.py
    uv run eval/run_eval.py --manifest data/aishell1_align_300_short3/manifest.jsonl \\
        --model x-asr-480ms-int8-ort sensevoice-small-int8-ort-p480 ...
    uv run eval/run_eval.py --manifest data/aishell1_align_300/manifest.jsonl --limit 100 --model ...
    uv run eval/run_eval.py --manifest data/aishell1_continuous_300/manifest.jsonl --limit 25 --model ...
用法：
    uv run scripts/pseudo_stream_compare.py            # 印出 Markdown 表，數字另存 eval/results/
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import paired_bootstrap_cer_diff, summarize, utterance_counts
from eval.segmentation import match_file

ALIGN = "aishell1_align_300"
# 依要求只跑小量：短句子集全部 100 句、對齊集前 100 句、連續集前 25 條
DEFAULT_RUNS = [f"{ALIGN}_short3", f"{ALIGN}:lim100", "aishell1_continuous_300:lim25"]
DEFAULT_CONFIG = "sil1000_pre100_chunk10_t1_lead1000_tail1500_pause150fin"
DEFAULT_MODELS = [
    "sensevoice-small-int8-ort",
    "sensevoice-small-int8-ort-p480",
    "sensevoice-small-int8-ort-p960",
    "sensevoice-small-int8-ort-p480-960",
    "sensevoice-small-int8-ort-p480-max3",
]


def _first_visible_s(seg: dict[str, Any]) -> float | None:
    """句段第一次出現非空文字的時刻（partial / 暫定結果 / 最終結果，取早者；原始音訊時間軸）。"""
    first = seg["e2e_first_token_latency_s"]
    times = [] if first is None else [first]
    times += [p["emitted_s"] for p in seg["provisionals"] if p["text"].strip()]
    if seg["text"].strip() and seg["final_s"] is not None:
        times.append(seg["final_s"])
    return min(times) if times else None


def _first_text_segment(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((s for s in segments if _first_visible_s(s) is not None), None)


def _result_s(seg: dict[str, Any], true_end: float) -> float | None:
    """第一個涵蓋到真實末字結束的結果（暫定結果的音訊終點不早於末字結束，或最終結果）。"""
    ready = [p["emitted_s"] for p in seg["provisionals"] if p["audio_end_s"] >= true_end]
    if seg["final_s"] is not None:
        ready.append(seg["final_s"])
    return min(ready) if ready else None


def _ms(values: list[float]) -> dict[str, Any]:
    return summarize([v * 1000 for v in values])


def _rtf(utts: list[dict[str, Any]]) -> float:
    return float(sum(u["compute_s"] for u in utts) / sum(u["processed_s"] for u in utts))


def _cer_block(base: list[dict[str, Any]], utts: list[dict[str, Any]]) -> dict[str, Any]:
    assert [u["id"] for u in base] == [u["id"] for u in utts], "兩個結果的句子不同"
    refs = [u["ref"] for u in utts]
    err_b, n = utterance_counts(refs, [u["hyp"] for u in base])
    err, _ = utterance_counts(refs, [u["hyp"] for u in utts])
    diff, lo, hi = paired_bootstrap_cer_diff(err_b, err, n)
    return {
        "n_utts": len(utts),
        "cer": sum(err) / sum(n),
        "diff_vs_baseline": diff,
        "diff_ci95": [lo, hi],
        "errors": sum(err),
        "ref_tokens": sum(n),
    }


def single_sentence(
    base: dict[str, Any],
    res: dict[str, Any],
    align: dict[str, tuple[float, float]] | None,
) -> dict[str, Any]:
    """冒煙集 / 對齊集（含短句子集）：每條一句。"""
    utts = res["utterances"]
    out: dict[str, Any] = {"cer": _cer_block(base["utterances"], utts), "rtf": _rtf(utts)}
    e2e_partial: list[float] = []
    e2e_visible: list[float] = []
    for u in utts:
        seg = next((s for s in u["segments"] if s["e2e_first_token_latency_s"] is not None), None)
        if seg is not None:
            e2e_partial.append(seg["e2e_first_token_latency_s"])
        first = _first_text_segment(u["segments"])
        if first is not None:
            e2e_visible.append(_first_visible_s(first) or 0.0)
    out["e2e_first_token_ms"] = {"partial": _ms(e2e_partial), "visible": _ms(e2e_visible)}
    if align is not None:
        onset_partial: list[float] = []
        onset_visible: list[float] = []
        result: list[float] = []
        for u in utts:
            onset, end = align[u["id"]]
            seg = next(
                (s for s in u["segments"] if s["e2e_first_token_latency_s"] is not None), None
            )
            if seg is not None:
                onset_partial.append(seg["e2e_first_token_latency_s"] - onset)
            first = _first_text_segment(u["segments"])
            if first is not None:
                onset_visible.append((_first_visible_s(first) or 0.0) - onset)
            if u["segments"]:
                r = _result_s(u["segments"][-1], end)
                if r is not None:
                    result.append(r - end)
        out["true_onset_first_token_ms"] = {
            "partial": _ms(onset_partial),
            "visible": _ms(onset_visible),
        }
        out["result_latency_ms"] = _ms(result)
    return out


def continuous(base: dict[str, Any], res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"cer": _cer_block(base["utterances"], res["utterances"])}
    out["rtf"] = _rtf(res["utterances"])
    partial: list[float] = []
    visible: list[float] = []
    result: list[float] = []
    for u in res["utterances"]:
        rows = match_file(u["sentences"], u["segments"])
        for s, r in zip(u["sentences"], rows, strict=True):
            if r["result_latency_s"] is not None:
                result.append(r["result_latency_s"])
            if not r["starts_segment"]:
                continue
            if r["first_token_latency_s"] is not None:
                partial.append(r["first_token_latency_s"])
            t = _first_visible_s(u["segments"][r["segments"][0]])
            if t is not None:
                visible.append(t - s["start_s"])
    out["true_onset_first_token_ms"] = {"partial": _ms(partial), "visible": _ms(visible)}
    out["result_latency_ms"] = _ms(result)
    return out


def _load(results_dir: Path, day: str, dataset: str, model: str, config: str) -> dict[str, Any]:
    path = results_dir / f"{day}_{dataset}_{model}_{config}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先跑 run_eval（見 --help）")
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return res


def _f(x: float | None, spec: str = ".0f") -> str:
    return "—" if x is None else format(x, spec)


def _lat(block: dict[str, Any]) -> str:
    return f"{_f(block['mean'])} / {_f(block['p90'])}"


def _cer_cell(c: dict[str, Any], is_base: bool) -> str:
    if is_base:
        return f"{c['cer'] * 100:.2f}"
    lo, hi = c["diff_ci95"]
    return f"{c['cer'] * 100:.2f}（{c['diff_vs_baseline'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]）"


def print_tables(report: dict[str, Any], baseline: str) -> None:
    for dataset, rows in report["datasets"].items():
        first = next(iter(rows.values()))
        has_align = "true_onset_first_token_ms" in first
        print(f"\n### {dataset}（{first['cer']['n_utts']} 條）\n")
        head = "| 模型 | CER %（差 [95% CI]） |"
        head += (
            " 首字 真實起點：partial mean / 可見 mean / 可見 P90 |"
            if has_align
            else " 首字 e2e：partial mean / 可見 mean / 可見 P90 |"
        )
        head += " 說完 → 完整結果 mean / P90 |" if has_align else ""
        head += " RTF | 峰值 RSS | 模型 MB |"
        print(head)
        print("|" + "---|" * head.count(" |"))
        for model, r in rows.items():
            cells = [model, _cer_cell(r["cer"], model == baseline)]
            ft = r["true_onset_first_token_ms" if has_align else "e2e_first_token_ms"]
            cells.append(
                f"{_f(ft['partial']['mean'])} / {_f(ft['visible']['mean'])} / "
                f"{_f(ft['visible']['p90'])}"
            )
            if has_align:
                cells.append(_lat(r["result_latency_ms"]))
            cells += [f"{r['rtf']:.3f}", _f(r.get("peak_rss_mb")), _f(r.get("model_mb"), ".1f")]
            print("| " + " | ".join(cells) + " |")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date", default=date.today().isoformat(), help="結果檔名的日期")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="結果檔名中模型之後的設定名")
    p.add_argument("--baseline", default="x-asr-480ms-int8-ort", help="真串流（CER 差的 A）")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="偽串流 / 非串流設定")
    p.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="評測集目錄名，可加「:<後綴>」對應結果檔名的後綴（例如 run_eval --limit 100 的 lim100）",
    )
    args = p.parse_args()

    align: dict[str, tuple[float, float]] = {}
    for line in (REPO_ROOT / "data" / ALIGN / "alignments.jsonl").read_text().splitlines():
        row = json.loads(line)
        align[row["id"]] = (row["chars"][0]["start"], row["chars"][-1]["end"])

    report: dict[str, Any] = {"date": args.date, "config": args.config, "datasets": {}}
    for run in args.runs:
        dataset, _, suffix = run.partition(":")
        config = args.config + (f"_{suffix}" if suffix else "")
        base = _load(args.results_dir, args.date, dataset, args.baseline, config)
        rows: dict[str, Any] = {}
        for model in [args.baseline, *args.models]:
            res = _load(args.results_dir, args.date, dataset, model, config)
            if "continuous" in dataset:
                r = continuous(base, res)
            else:
                r = single_sentence(base, res, align if dataset.startswith(ALIGN) else None)
            r["peak_rss_mb"] = res["metrics"]["memory_mb"]["peak_rss"]
            r["model_mb"] = res["metrics"]["model_size_mb"]
            r["commit"] = res["environment"].get("commit")
            rows[model] = r
        report["datasets"][run] = rows

    print_tables(report, args.baseline)
    out = args.results_dir / f"{args.date}_pseudo_stream_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n數字存於 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
