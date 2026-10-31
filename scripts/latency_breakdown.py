"""首字延遲拆解：以 MFA 字級對齊的真實首字起點，把「VAD 起點 → 首字出現」拆成三段。

    ① 真實首字起點 − VAD 判定起點             （VAD 切點與真實起音的差，通常為負：VAD 含 padding）
    ② 模型首字時間戳 − 真實首字起點           （模型把第一個字放在音訊的哪裡，40 ms 解析度）
    ③ 首字實際出現的時刻 − 模型首字時間戳     （為了輸出這個字，還要等多少音訊到達 + 計算時間）

① + ② + ③ = 首字延遲（VAD 起點）。另報 ②′ = 模型首字時間戳 − 真實首字「結束」點。
只用每條第一個有字的句段；首字與參考首字不同（辨識錯、或第一段是誤觸發）的句子不計入，數量另列。

需要：
    uv run scripts/prepare_alignment.py                                   # 對齊結果
    uv run eval/run_eval.py --manifest data/aishell1_align_300/manifest.jsonl \\
        --model x-asr-480ms-int8 x-asr-160ms-int8                         # 含 first_token_audio_s 的結果
用法：
    uv run scripts/latency_breakdown.py   # 印出表格，圖存 docs/latency_breakdown.png，數字存 eval/results/
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.text_norm import normalize

MODELS = {"x-asr-480ms-int8": "X-ASR 480 ms", "x-asr-160ms-int8": "X-ASR 160 ms"}
PARTS = [
    ("vad_to_onset", "① VAD start → true first-char onset"),
    ("onset_to_stamp", "② true onset → model token timestamp"),
    ("stamp_to_emit", "③ token timestamp → first token shown"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def latest_result(results_dir: Path, dataset: str, model: str, config: str) -> Path:
    files = sorted(results_dir.glob(f"*_{dataset}_{model}_{config}.json"))
    if not files:
        raise FileNotFoundError(f"找不到 {results_dir}/*_{dataset}_{model}_{config}.json")
    return files[-1]


def breakdown(result: dict[str, Any], align: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    skipped = {"no_alignment": 0, "no_text": 0, "first_char_mismatch": 0, "no_timestamp": 0}
    for u in result["utterances"]:
        a = align.get(u["id"])
        if a is None:
            skipped["no_alignment"] += 1
            continue
        seg = next((s for s in u["segments"] if s["text"].strip()), None)
        if seg is None:
            skipped["no_text"] += 1
            continue
        hyp, ref = normalize(seg["text"]).split(), normalize(u["ref"]).split()
        if not hyp or hyp[0] != ref[0]:
            skipped["first_char_mismatch"] += 1
            continue
        if seg.get("first_token_audio_s") is None:
            skipped["no_timestamp"] += 1
            continue
        onset, offset = a["chars"][0]["start"], a["chars"][0]["end"]
        emitted = seg["e2e_first_token_latency_s"]  # 原始音訊時間軸上的出現時刻
        rows.append(
            {
                "vad_to_onset": onset - seg["speech_start_s"],
                "onset_to_stamp": seg["first_token_audio_s"] - onset,
                "stamp_to_emit": emitted - seg["first_token_audio_s"],
                "offset_to_stamp": seg["first_token_audio_s"] - offset,
                "total": seg["first_token_latency_s"],
                "first_char_duration": offset - onset,
            }
        )
    stats = {}
    for key in rows[0] if rows else []:
        v = np.array([r[key] for r in rows]) * 1000
        stats[key] = {
            "mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
        }
    return {"n": len(rows), "skipped": skipped, "stats_ms": stats}


def plot(out: dict[str, dict[str, Any]], path: Path, title: str) -> None:
    """每個模型一列的瀑布圖：從 VAD 起點（0）依序接上三段平均值；上方細線標出首字本身（MFA）。"""
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(out)
    for row, model in enumerate(names):
        s = out[model]["stats_ms"]
        t = 0.0
        for (key, label), color in zip(PARTS, COLORS, strict=True):
            w = s[key]["mean"]
            # ① 通常為負（VAD 起點早於首字），② 會從 ① 的終點往右畫而蓋住它，所以 ① 畫成上層細條
            first = key == PARTS[0][0]
            ax.barh(row, w, left=t, height=0.18 if first else 0.42, color=color,
                    edgecolor=SURFACE, linewidth=0 if first else 2, zorder=3 if first else 2,
                    label=label if row == 0 else None)  # fmt: skip
            if abs(w) >= 60:
                ax.text(t + w / 2, row, f"{w:.0f}", ha="center", va="center", fontsize=8,
                        color="white")  # fmt: skip
            else:  # 太窄的段把數字標在外側
                ax.text(min(t, t + w) - 6, row, f"{w:.0f}", ha="right", va="center", fontsize=8,
                        color=TEXT)  # fmt: skip
            t += w
        ax.text(t + 8, row, f"= {t:.0f} ms", va="center", fontsize=9, color=TEXT)
        onset = s["vad_to_onset"]["mean"]
        dur = s["first_char_duration"]["mean"]
        y = row - 0.32
        ax.plot([onset, onset + dur], [y, y], color=TEXT_2, lw=2, solid_capstyle="butt")
        ax.text(onset + dur + 6, y, "first char (MFA, mean)", va="center", fontsize=7,
                color=TEXT_2)  # fmt: skip
    ax.axvline(0, color=TEXT_2, lw=1)
    ax.set_yticks(range(len(names)), [f"{MODELS[m]} (n={out[m]['n']})" for m in names])
    ax.set_ylim(len(names) - 0.5, -0.75)
    ax.set_xlim(-120, None)
    ax.set_xlabel("ms relative to VAD speech start (means; the three parts sum to the total)",
                  color=TEXT_2)  # fmt: skip
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    fig.legend(frameon=False, fontsize=8, loc="lower center", ncol=2, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="以字級對齊拆解首字延遲。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--align-dir", type=Path, default=REPO_ROOT / "data" / "aishell1_align_300")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    p.add_argument("--config", default="sil500_pre100_chunk10_t1_lead1000")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "latency_breakdown.png")
    args = p.parse_args()

    align_file = args.align_dir / "alignments.jsonl"
    if not align_file.is_file():
        print(f"錯誤：找不到 {align_file}，請先執行 scripts/prepare_alignment.py", file=sys.stderr)
        return 1
    align = {
        a["id"]: a
        for a in (json.loads(line) for line in align_file.read_text(encoding="utf-8").splitlines())
    }
    out: dict[str, dict[str, Any]] = {}
    env: set[str] = set()
    for model in MODELS:
        try:
            path = latest_result(args.results_dir, args.align_dir.name, model, args.config)
        except FileNotFoundError as e:
            print(f"錯誤：{e}，請先跑 run_eval.py（見 --help）", file=sys.stderr)
            return 1
        res = json.loads(path.read_text(encoding="utf-8"))
        out[model] = breakdown(res, align)
        env_info = res["environment"]
        dirty = " dirty" if env_info["dirty"] else ""
        env.add(f"{env_info['chip']} @ {env_info['commit']}{dirty}")
        print(f"{model}: {path.name}，計入 {out[model]['n']} 句，略過 {out[model]['skipped']}")

    print("\n| 模型 | 段 | mean | P50 | P90 |\n|---|---|---|---|---|")
    keys = [k for k, _ in PARTS] + ["offset_to_stamp", "total", "first_char_duration"]
    for model, o in out.items():
        for k in keys:
            s = o["stats_ms"][k]
            print(f"| {MODELS[model]} | {k} | {s['mean']:.0f} | {s['p50']:.0f} | {s['p90']:.0f} |")

    title = (
        f"First-token latency breakdown, AISHELL-1 test {len(align)} utts (MFA), {'; '.join(env)}"
    )
    plot(out, args.out, title)
    stats_file = (
        args.results_dir / f"{date.today().isoformat()}_{args.align_dir.name}_breakdown.json"
    )
    stats_file.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n圖：{args.out}\n數字：{stats_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
