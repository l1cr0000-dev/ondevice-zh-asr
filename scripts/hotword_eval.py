"""SenseVoice 熱詞（CTC prefix beam search + 熱詞加分，只改解碼）在小集合上的效果：beam × 加分掃描。

只量 ASR 解碼本身：每句整段音訊跑一次模型（不經 VAD，與 pipeline 中 finalize 的整段辨識相同），
同一份 logits 依序用每個設定解碼，因此各設定的差異只來自解碼。基準是 beam 1、加分 0（= greedy，
與 `sensevoice-small-int8-ort` 預設逐句相同）。

熱詞清單 = 目標熱詞（評測集參考文本中的專有名詞）+ 干擾熱詞（不在任何句子中）。檢查：
每個目標熱詞至少出現在一句參考文本中、干擾熱詞一句都不出現，否則報錯。

指標（文字都經過 eval.text_norm 正規化、去空白後比對）：
- 熱詞召回率：參考文本中每次出現的目標熱詞，hyp 同句也出現（逐句取 min(ref 次數, hyp 次數)）的比例。
- 熱詞句 / 非熱詞句 CER（含任一目標熱詞的句子為熱詞句），附 bootstrap 95% CI，
  以及對基準的成對差（eval.metrics.paired_bootstrap_cer_diff）。
- 誤觸發：hyp 出現某熱詞的次數超過 ref 的部分（目標熱詞與干擾熱詞分開算）。
- 每句解碼耗時（只含解碼，不含模型）與解碼 RTF；另列模型（特徵 + encoder）的 RTF 當對照。
  機器上有其他工作時絕對值不準，只看同批相對差。

用法：
    uv run scripts/hotword_eval.py                       # 300 句對齊集，beam 1 4 8 × 加分 0 1 1.5 2 3
    uv run scripts/hotword_eval.py --manifest data/aishell1_test_200/manifest.jsonl \\
        --targets eval/hotwords/aishell1_test_200_targets.txt
    uv run scripts/hotword_eval.py --beam 4 --score 1.5 --show-changes   # 印出與基準不同的句子
"""

import argparse
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    bootstrap_cer_ci,
    paired_bootstrap_cer_diff,
    summarize,
    utterance_counts,
)
from eval.text_norm import normalize
from src.cli import positive_int, read_hotwords

DEFAULT_MANIFEST = REPO_ROOT / "data" / "aishell1_align_300" / "manifest.jsonl"
HOTWORD_DIR = REPO_ROOT / "eval" / "hotwords"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def _plain(text: str) -> str:
    return normalize(text).replace(" ", "")


def count_hits(words: list[str], refs: list[str], hyps: list[str]) -> dict[str, dict[str, int]]:
    """每個詞：ref 出現次數、命中次數（逐句 min）、多出來的次數（逐句 hyp − ref 的正值 = 誤觸發）。"""
    out: dict[str, dict[str, int]] = {}
    for w in words:
        ref_n = hit = extra = 0
        for r, h in zip(refs, hyps, strict=True):
            a, b = r.count(w), h.count(w)
            ref_n += a
            hit += min(a, b)
            extra += max(0, b - a)
        out[w] = {"ref": ref_n, "hit": hit, "false": extra}
    return out


def _cer_block(
    idx: list[int], errors: list[int], base_errors: list[int], refs_n: list[int]
) -> dict[str, Any] | None:
    if not idx:
        return None
    e = [errors[i] for i in idx]
    b = [base_errors[i] for i in idx]
    n = [refs_n[i] for i in idx]
    diff, lo, hi = paired_bootstrap_cer_diff(b, e, n)
    return {
        "num_utterances": len(idx),
        "ref_tokens": sum(n),
        "errors": sum(e),
        "cer": sum(e) / sum(n),
        "cer_ci95": list(bootstrap_cer_ci(e, n)),
        "diff_vs_base": diff,
        "diff_ci95": [lo, hi],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SenseVoice 熱詞（只改解碼）的 beam × 加分掃描：召回率、熱詞句 / 非熱詞句 CER、"
        "誤觸發、解碼耗時。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="評測集 manifest")
    p.add_argument(
        "--targets",
        type=Path,
        default=HOTWORD_DIR / "aishell1_align_300_targets.txt",
        help="目標熱詞檔（必須出現在參考文本中）",
    )
    p.add_argument(
        "--distractors",
        type=Path,
        default=HOTWORD_DIR / "distractors.txt",
        help="干擾熱詞檔（不得出現在參考文本中）",
    )
    p.add_argument("--beam", type=positive_int, nargs="+", default=[1, 4, 8], help="beam 寬度")
    p.add_argument(
        "--score",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 1.5, 2.0, 3.0],
        help="熱詞每 token 加分",
    )
    p.add_argument("--threads", type=positive_int, default=1, help="模型推論執行緒數")
    p.add_argument("--limit", type=positive_int, help="只跑前 N 句（除錯用）")
    p.add_argument("--show-changes", action="store_true", help="印出每個設定與基準不同的句子")
    p.add_argument(
        "--out", type=Path, help="結果 JSON；預設 eval/results/<日期>_<資料集>_hotword.json"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from src.asr import SenseVoiceConfig, SenseVoiceOrtAsr
    from src.vad import load_wav

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = rows[: args.limit] if args.limit else rows
    refs = [_plain(r["text"]) for r in rows]
    targets = [_plain(w) for w in read_hotwords(args.targets)]
    distractors = [_plain(w) for w in read_hotwords(args.distractors)]
    missing = [w for w in targets if not any(w in r for r in refs)]
    present = [w for w in distractors if any(w in r for r in refs)]
    if present:
        print(f"錯誤：干擾熱詞出現在參考文本中 {present}", file=sys.stderr)
        return 1
    if missing:
        # 例如拿對齊集的熱詞清單去跑冒煙集，量「熱詞清單對無關句子」的誤改；這些詞的效果等同干擾熱詞
        print(f"注意：{len(missing)} 個目標熱詞不在此評測集的參考文本中", file=sys.stderr)
    hotwords = tuple(dict.fromkeys(targets + distractors))
    hot_idx = [i for i, r in enumerate(refs) if any(w in r for w in targets)]
    cold_idx = [i for i in range(len(rows)) if i not in set(hot_idx)]

    asr = SenseVoiceOrtAsr(SenseVoiceConfig(num_threads=args.threads))
    # 加分 0 時不建熱詞圖（= 不帶熱詞的 beam search）；beam 1 加分 0 = greedy 基準
    settings = list(itertools.product(args.beam, args.score))
    if (1, 0.0) not in settings:
        settings.insert(0, (1, 0.0))
    settings.sort()
    decoders = {(b, s): asr.make_decoder(b, hotwords if s else (), s) for b, s in settings}
    skipped = asr.make_decoder(1, hotwords, 1.0).skipped_hotwords

    hyps: dict[tuple[int, float], list[str]] = {k: [] for k in settings}
    decode_ms: dict[tuple[int, float], list[float]] = {k: [] for k in settings}
    model_s = audio_s = 0.0
    for i, row in enumerate(rows):
        audio = load_wav(
            Path(row["wav"])
            if Path(row["wav"]).is_absolute()
            else args.manifest.parent / row["wav"]
        )
        audio_s += len(audio) / asr.sample_rate
        t0 = time.perf_counter()
        logits = asr.logits(audio)
        model_s += time.perf_counter() - t0
        for k, dec in decoders.items():
            t0 = time.perf_counter()
            text = "" if logits is None else dec.decode(logits)
            decode_ms[k].append((time.perf_counter() - t0) * 1000)
            hyps[k].append(text)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr)

    raw_refs = [r["text"] for r in rows]
    base = hyps[(1, 0.0)]
    base_errors, ref_n = utterance_counts(raw_refs, base)
    base_plain = [_plain(h) for h in base]
    base_target_hits = count_hits(targets, refs, base_plain)
    runs: list[dict[str, Any]] = []
    for k in settings:
        plain = [_plain(h) for h in hyps[k]]
        errors, _ = utterance_counts(raw_refs, hyps[k])
        t_hits = count_hits(targets, refs, plain)
        d_hits = count_hits(distractors, refs, plain)
        ref_total = sum(v["ref"] for v in t_hits.values())
        runs.append(
            {
                "beam": k[0],
                "score": k[1],
                "decoder": dec_desc if (dec_desc := decoders[k].describe()) else " greedy",
                "recall": sum(v["hit"] for v in t_hits.values()) / ref_total if ref_total else None,
                "target_occurrences": ref_total,
                "false_triggers_target": sum(v["false"] for v in t_hits.values()),
                "false_triggers_distractor": sum(v["false"] for v in d_hits.values()),
                "cer_all": _cer_block(list(range(len(rows))), errors, base_errors, ref_n),
                "cer_hot": _cer_block(hot_idx, errors, base_errors, ref_n),
                "cer_cold": _cer_block(cold_idx, errors, base_errors, ref_n),
                "changed_vs_base": sum(a != b for a, b in zip(hyps[k], base, strict=True)),
                "changed_cold_vs_base": sum(hyps[k][i] != base[i] for i in cold_idx),
                "decode_ms": summarize(decode_ms[k]),
                "decode_rtf": sum(decode_ms[k]) / 1000 / audio_s,
                "target_hits": t_hits,
                "distractor_false": {w: v["false"] for w, v in d_hits.items() if v["false"]},
                "hyps": hyps[k],
            }
        )

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout.strip()
    result = {
        "date": date.today().isoformat(),
        "commit": commit,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "manifest": str(args.manifest),
        "asr_name": asr.name,
        "threads": args.threads,
        "num_utterances": len(rows),
        "hot_utterances": len(hot_idx),
        "targets": targets,
        "distractors": distractors,
        "skipped_hotwords": skipped,
        "base_missed_targets": {w: v for w, v in base_target_hits.items() if v["hit"] < v["ref"]},
        "audio_s": audio_s,
        "model_rtf": model_s / audio_s,
        "ids": [r["id"] for r in rows],
        "refs": raw_refs,
        "runs": runs,
    }
    out = (
        args.out
        or RESULTS_DIR / f"{date.today().isoformat()}_{args.manifest.parent.name}_hotword.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"{len(rows)} 句（熱詞句 {len(hot_idx)}），目標熱詞 {len(targets)} 個（出現 "
        f"{runs[0]['target_occurrences']} 次，greedy 漏認 {len(result['base_missed_targets'])} 個詞），"
        f"干擾熱詞 {len(distractors)} 個；模型 RTF {result['model_rtf']:.4f}"
    )
    print(
        "| beam | score | 召回 % | 熱詞句 CER % [CI] | Δ vs greedy [CI] | 非熱詞句 CER % | Δ [CI] "
        "| 非熱詞句改變 | 誤觸發 目標/干擾 | 解碼 ms mean/P90 | 解碼 RTF |"
    )
    print("|" + "---|" * 11)
    pct = 100

    def ci(x: list[float]) -> str:
        return f"[{x[0] * pct:+.2f}, {x[1] * pct:+.2f}]"

    def cer(b: dict[str, Any] | None, with_ci: bool) -> str:
        if b is None:
            return "- | -"
        own = f" [{b['cer_ci95'][0] * pct:.2f}, {b['cer_ci95'][1] * pct:.2f}]" if with_ci else ""
        return f"{b['cer'] * pct:.2f}{own} | {b['diff_vs_base'] * pct:+.2f} {ci(b['diff_ci95'])}"

    for r in runs:
        recall = "-" if r["recall"] is None else f"{r['recall'] * pct:.1f}"
        print(
            f"| {r['beam']} | {r['score']:g} | {recall} "
            f"| {cer(r['cer_hot'], True)} | {cer(r['cer_cold'], False)} "
            f"| {r['changed_cold_vs_base']} "
            f"| {r['false_triggers_target']}/{r['false_triggers_distractor']} "
            f"| {r['decode_ms']['mean']:.1f}/{r['decode_ms']['p90']:.1f} | {r['decode_rtf']:.4f} |"
        )
    if args.show_changes:
        for r in runs:
            print(f"\n== beam {r['beam']} score {r['score']:g}")
            for i, (a, b) in enumerate(zip(base, r["hyps"], strict=True)):
                if a != b:
                    tag = "熱" if i in set(hot_idx) else "非"
                    print(f"  [{tag}] ref {refs[i]}\n       base {a}\n       new  {b}")
    print(f"結果：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
