"""短句子集：從 300 句 MFA 對齊集取真實語音長度（首字起點 → 末字結束）不超過 `--max-s` 的句子。

車艙指令多是 1–3 秒的短句；AISHELL 的句子平均約 4–5 秒，短句子集用來看短句上的延遲與計算量。
只寫 manifest（wav 路徑沿用對齊集的），不複製音訊。輸出目錄名中的數字是 `--max-s`。

用法：
    uv run scripts/prepare_short_subset.py               # → data/aishell1_align_300_short3/（100 句）
    uv run eval/run_eval.py --manifest data/aishell1_align_300_short3/manifest.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--align-dir", type=Path, default=REPO_ROOT / "data" / "aishell1_align_300")
    p.add_argument("--max-s", type=float, default=3.0, help="MFA 語音長度上限（秒）")
    p.add_argument("--out", type=Path, default=None, help="預設 <align-dir>_short<max-s>")
    args = p.parse_args()

    spans: dict[str, float] = {}
    for line in (args.align_dir / "alignments.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        spans[row["id"]] = row["chars"][-1]["end"] - row["chars"][0]["start"]
    rows = [
        json.loads(line)
        for line in (args.align_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    keep = [r for r in rows if spans[r["id"]] <= args.max_s]
    out = args.out or args.align_dir.with_name(f"{args.align_dir.name}_short{args.max_s:g}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8"
    )
    lengths = sorted(spans[r["id"]] for r in keep)
    meta = {
        "source": str(args.align_dir.relative_to(REPO_ROOT)),
        "max_speech_s": args.max_s,
        "n": len(keep),
        "speech_s_min_median_max": [lengths[0], lengths[len(lengths) // 2], lengths[-1]],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(keep)} / {len(rows)} 句 → {out}；語音長度 {meta['speech_s_min_median_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
