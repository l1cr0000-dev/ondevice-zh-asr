"""SenseVoice 偽串流：語言（auto / zh）、第一次 partial 的時機與顯示規則的比較表。只讀結果，不跑模型。

讀 run_eval 的結果 JSON，同一批句子上比較：
- CER 與成對 bootstrap 差（以 `--baseline` 為 A，以句重抽 10000 次）。
- 第一個 partial 的品質（每個句段第一個顯示的 partial）：正規化後全是 ASCII（英文 / 數字碎片）的比例、
  是最終文字前綴的比例（顯示的字之後都沒被改）、第一個字與最終文字第一個字相同的比例。
- 可見首字：第一次看到任何文字（顯示的 partial、暫定結果、最終結果取最早）。對齊集與短句子集以 MFA
  首字起點起算；連續集以每句真實起點起算（句子是句段第一句時）；冒煙集以補靜音結束起算（只看相對差）。
- 穩定度（`eval.stability.summarize_stability`）：修正率、被改字 / 最終字（前綴口徑 / 對齊口徑）。
- RTF 與重新辨識音訊比（`metrics.asr_compute.decoded_audio_ratio`，與機器負載無關）。
- VAD 觸發與首字（以 VAD 起點起算）：每條第一個有字的句段的 `vad_start_delay_s` / `first_token_latency_s`。

顯示規則（`eval.stability.FirstPartialRule`）只改畫面要不要顯示句段的第一個 partial，由結果 JSON 模擬：
不顯示的 partial 從句段中拿掉，之後的指標全部照常計算。辨識結果與 CER 不受影響。

短句子集（≤ 3 s，`scripts/prepare_short_subset.py`）直接取對齊集結果中的同一批句子，不另外跑。

需要（預設 A2，`--tag` 隨意）：
    uv run eval/run_eval.py --smoke --model <模型...> --jobs 2
    uv run eval/run_eval.py --manifest data/aishell1_align_300/manifest.jsonl --model <模型...> --jobs 2
    uv run eval/run_eval.py --manifest data/aishell1_continuous_300/manifest.jsonl --model <模型...> --jobs 2
用法：
    uv run scripts/first_partial_compare.py            # 印出 Markdown 表，數字另存 eval/results/
    # 設定名不同的結果（例如 VAD min_speech 掃描，B2）：模型後加 @<設定名後綴>
    uv run scripts/first_partial_compare.py --baseline sensevoice-small-int8-ort-p480-960-max3-zh \
        --models sensevoice-small-int8-ort-p320-960-max3-zh@_ms0 --runs aishell1_align_300 \
        --config sil1000_pre100_chunk10_t2_lead1000_tail1500_pause150fin --name vad_trigger
"""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import paired_bootstrap_cer_diff, summarize, utterance_counts
from eval.segmentation import match_file
from eval.stability import FirstPartialRule, apply_first_partial_rule, summarize_stability
from eval.text_norm import tokenize

ALIGN = "aishell1_align_300"
SHORT = f"{ALIGN}_short3"
DEFAULT_RUNS = ["aishell1_test_200", ALIGN, SHORT, "aishell1_continuous_300"]
DEFAULT_CONFIG = "sil1000_pre100_chunk10_t1_lead1000_tail1500_pause150fin"
DEFAULT_BASELINE = "sensevoice-small-int8-ort-p480-960-max3"
DEFAULT_MODELS = [f"sensevoice-small-int8-ort-p{ms}-960-max3-zh" for ms in (480, 400, 320)]
RULES = {
    "none": FirstPartialRule(),
    "ascii": FirstPartialRule(drop_ascii=True),
    "cjk2": FirstPartialRule(min_cjk=2, drop_ascii=True),
    "cjk3": FirstPartialRule(min_cjk=3, drop_ascii=True),
}


def _first_partial(seg: dict[str, Any]) -> dict[str, Any] | None:
    return next((p for p in seg["partials"] if p["text"].strip()), None)


def _first_visible_s(seg: dict[str, Any]) -> float | None:
    """句段第一次出現非空文字的時刻（顯示的 partial / 暫定結果 / 最終結果取早者）。"""
    p = _first_partial(seg)
    times = [] if p is None else [p["emitted_s"]]
    times += [x["emitted_s"] for x in seg["provisionals"] if x["text"].strip()]
    if seg["text"].strip() and seg["final_s"] is not None:
        times.append(seg["final_s"])
    return min(times) if times else None


def first_partial_quality(utts: list[dict[str, Any]]) -> dict[str, Any]:
    """每個有 partial 的句段的第一個 partial：全 ASCII、是最終文字的前綴、第一個字正確的比例。"""
    n = ascii_ = prefix = first_ok = 0
    for u in utts:
        for seg in u["segments"]:
            p = _first_partial(seg)
            if p is None:
                continue
            tok = tokenize(p["text"])
            final = tokenize(seg["text"])
            if not tok:
                continue
            n += 1
            ascii_ += all(t.isascii() for t in tok)
            prefix += final[: len(tok)] == tok
            first_ok += bool(final) and final[0] == tok[0]
    return {
        "segments": n,
        "ascii_ratio": ascii_ / n if n else 0.0,
        "prefix_ratio": prefix / n if n else 0.0,
        "first_char_ratio": first_ok / n if n else 0.0,
    }


def visible_first_ms(
    utts: list[dict[str, Any]], onset: Callable[[str], float] | None, continuous: bool
) -> list[float]:
    out: list[float] = []
    for u in utts:
        if continuous:
            rows = match_file(u["sentences"], u["segments"])
            for s, r in zip(u["sentences"], rows, strict=True):
                if not r["starts_segment"]:
                    continue
                t = _first_visible_s(u["segments"][r["segments"][0]])
                if t is not None:
                    out.append((t - s["start_s"]) * 1000)
            continue
        times = [t for t in map(_first_visible_s, u["segments"]) if t is not None]
        if times:
            out.append((times[0] - (onset(u["id"]) if onset else 0.0)) * 1000)
    return out


def vad_first_ms(utts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """每條第一個有字的句段：VAD 觸發延遲與首字延遲（都以 VAD 起點起算，ms）。"""
    trig: list[float] = []
    first: list[float] = []
    for u in utts:
        seg = next((s for s in u["segments"] if s["text"].strip()), None)
        if seg is None or seg["first_token_latency_s"] is None:
            continue
        trig.append(seg["vad_start_delay_s"] * 1000)
        first.append(seg["first_token_latency_s"] * 1000)
    return {"vad_start_delay_ms": summarize(trig), "first_token_vad_ms": summarize(first)}


def _cer(base: list[dict[str, Any]], utts: list[dict[str, Any]]) -> dict[str, Any]:
    assert [u["id"] for u in base] == [u["id"] for u in utts], "兩個結果的句子不同"
    refs = [u["ref"] for u in utts]
    err_b, n = utterance_counts(refs, [u["hyp"] for u in base])
    err, _ = utterance_counts(refs, [u["hyp"] for u in utts])
    diff, lo, hi = paired_bootstrap_cer_diff(err_b, err, n)
    same = sum(a["hyp"] == b["hyp"] for a, b in zip(base, utts, strict=True))
    return {
        "n_utts": len(utts),
        "cer": sum(err) / sum(n),
        "diff_vs_baseline": diff,
        "diff_ci95": [lo, hi],
        "same_hyp_ratio": same / len(utts),
    }


def _with_rule(utts: list[dict[str, Any]], rule: FirstPartialRule) -> list[dict[str, Any]]:
    return [
        {**u, "segments": [apply_first_partial_rule(s, rule) for s in u["segments"]]} for u in utts
    ]


def analyse(
    base: list[dict[str, Any]],
    utts: list[dict[str, Any]],
    onset: Callable[[str], float] | None,
    continuous: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {"cer": _cer(base, utts)}
    out["rtf"] = sum(u["compute_s"] for u in utts) / sum(u["processed_s"] for u in utts)
    out.update(vad_first_ms(utts))
    out["rules"] = {}
    for name, rule in RULES.items():
        shown = _with_rule(utts, rule)
        st = summarize_stability(shown) or {}
        out["rules"][name] = {
            "first_partial": first_partial_quality(shown),
            "visible_first_ms": summarize(visible_first_ms(shown, onset, continuous)),
            "unstable_update_rate": st.get("unstable_update_rate"),
            "revised_token_ratio": st.get("revised_token_ratio"),
            "revised_token_ratio_aligned": st.get("revised_token_ratio_aligned"),
            "utterances_with_revision": st.get("utterances_with_revision"),
        }
    return out


def _load(results_dir: Path, day: str, dataset: str, model: str, config: str) -> dict[str, Any]:
    """`model` 可帶 `@<後綴>`：結果檔名的設定名 = config + 後綴（例如 `@_ms0`）。"""
    model, _, suffix = model.partition("@")
    config += suffix
    path = results_dir / f"{day}_{dataset}_{model}_{config}.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}，請先跑 run_eval（見 --help）")
    res: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return res


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _short(model: str) -> str:
    return model.removeprefix("sensevoice-small-int8-ort-").replace("@_", " ")


def _diff(d: dict[str, Any] | None) -> str:
    if d is None:
        return "—"
    lo, hi = d["ci95"]
    return f"{d['mean']:+.0f} [{lo:+.0f}, {hi:+.0f}]"


def paired_diff_ms(
    base: dict[str, float], other: dict[str, float], resamples: int = 10000
) -> dict[str, Any] | None:
    """兩個設定在同一批句子上的平均差（other − base）與以句重抽的 95% 信賴區間。"""
    ids = sorted(base.keys() & other.keys())
    if not ids:
        return None
    d = np.array([other[i] - base[i] for i in ids])
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    return {
        "n": len(ids),
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def visible_first_by_id(
    utts: list[dict[str, Any]], onset: Callable[[str], float] | None
) -> dict[str, float]:
    """單句集：句子 id → 可見首字（ms）。"""
    out: dict[str, float] = {}
    for u in utts:
        times = [t for t in map(_first_visible_s, u["segments"]) if t is not None]
        if times:
            out[u["id"]] = (times[0] - (onset(u["id"]) if onset else 0.0)) * 1000
    return out


def print_tables(report: dict[str, Any], baseline: str) -> None:
    for dataset, rows in report["datasets"].items():
        n = next(iter(rows.values()))["cer"]["n_utts"]
        print(f"\n### {dataset}（{n} 條）\n")
        print(
            "| 設定 | CER %（差 [95% CI]） | 最終文字相同 | RTF | 音訊比 | VAD 觸發 / 首字（VAD 起點）"
            " mean (ms) | 第一個 partial：ASCII / 前綴一致 / 首字正確 | 可見首字 mean / P50 / P90 (ms) | "
            "可見首字差 [95% CI] | 修正率 | 被改字（前綴 / 對齊） |"
        )
        print("|" + "---|" * 11)
        for model, r in rows.items():
            c = r["cer"]
            cer = f"{c['cer'] * 100:.2f}"
            if model != baseline:
                lo, hi = c["diff_ci95"]
                cer += f"（{c['diff_vs_baseline'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]）"
            ac = r.get("asr_compute") or {}
            ratio = ac.get("decoded_audio_ratio")
            x = r["rules"]["none"]
            fp, v = x["first_partial"], x["visible_first_ms"]
            print(
                f"| {_short(model)} | {cer} | {_pct(c['same_hyp_ratio'])} | {r['rtf']:.3f} | "
                f"{'—' if ratio is None else f'{ratio:.2f}'} | "
                f"{r['vad_start_delay_ms']['mean']:.0f} / {r['first_token_vad_ms']['mean']:.0f} | "
                f"{_pct(fp['ascii_ratio'])} / {_pct(fp['prefix_ratio'])} / "
                f"{_pct(fp['first_char_ratio'])} | "
                f"{v['mean']:.0f} / {v['p50']:.0f} / {v['p90']:.0f} | {_diff(r.get('visible_diff_ms'))} | "
                f"{_pct(x['unstable_update_rate'])} | {_pct(x['revised_token_ratio'])} / "
                f"{_pct(x['revised_token_ratio_aligned'])} |"
            )
        print(
            "\n| 設定 | 顯示規則 | 第一個顯示的 partial：ASCII / 前綴一致 / 首字正確 | "
            "可見首字 mean / P90 (ms)（差） | 修正率 | 被改字（前綴 / 對齊） | 有修正的條 |"
        )
        print("|" + "---|" * 7)
        for model, r in rows.items():
            none = r["rules"]["none"]["visible_first_ms"]["mean"]
            for name, x in r["rules"].items():
                fp, v = x["first_partial"], x["visible_first_ms"]
                print(
                    f"| {_short(model)} | {RULES[name].describe()} | "
                    f"{_pct(fp['ascii_ratio'])} / {_pct(fp['prefix_ratio'])} / "
                    f"{_pct(fp['first_char_ratio'])} | {v['mean']:.0f} / {v['p90']:.0f}"
                    f"（{v['mean'] - none:+.0f}） | {_pct(x['unstable_update_rate'])} | "
                    f"{_pct(x['revised_token_ratio'])} / {_pct(x['revised_token_ratio_aligned'])} | "
                    f"{_pct(x['utterances_with_revision'])} |"
                )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date", default=date.today().isoformat(), help="結果檔名的日期")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="結果檔名中模型之後的設定名")
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="CER 差的 A（現行預設）")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="比較的設定")
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="評測集目錄名")
    p.add_argument("--name", default="first_partial_compare", help="輸出 JSON 檔名（日期之後）")
    args = p.parse_args()

    align: dict[str, float] = {}
    for line in (REPO_ROOT / "data" / ALIGN / "alignments.jsonl").read_text().splitlines():
        row = json.loads(line)
        align[row["id"]] = row["chars"][0]["start"]
    short_ids = {
        json.loads(line)["id"]
        for line in (REPO_ROOT / "data" / SHORT / "manifest.jsonl").read_text().splitlines()
    }

    report: dict[str, Any] = {"date": args.date, "config": args.config, "datasets": {}}
    for dataset in args.runs:
        # 短句子集取對齊集結果中的同一批句子
        source = ALIGN if dataset == SHORT else dataset
        ids = short_ids if dataset == SHORT else None

        def keep(u: dict[str, Any], ids: set[str] | None = ids) -> bool:
            return ids is None or u["id"] in ids

        onset = align.__getitem__ if source == ALIGN else None
        continuous = "continuous" in dataset
        base = [
            u
            for u in _load(args.results_dir, args.date, source, args.baseline, args.config)[
                "utterances"
            ]
            if keep(u)
        ]
        rows: dict[str, Any] = {}
        base_visible = visible_first_by_id(base, onset)
        for model in [args.baseline, *args.models]:
            res = _load(args.results_dir, args.date, source, model, args.config)
            utts = [u for u in res["utterances"] if keep(u)]
            r = analyse(base, utts, onset, continuous)
            if not continuous and model != args.baseline:
                r["visible_diff_ms"] = paired_diff_ms(
                    base_visible, visible_first_by_id(utts, onset)
                )
            # 音訊比是整個結果檔的統計，短句子集不適用
            r["asr_compute"] = None if dataset == SHORT else res["metrics"].get("asr_compute")
            r["commit"] = res["environment"].get("commit")
            rows[model] = r
        report["datasets"][dataset] = rows

    print_tables(report, args.baseline)
    out = args.results_dir / f"{args.date}_{args.name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n數字存於 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
