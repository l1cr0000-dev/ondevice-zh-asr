"""blank_penalty 掃描的取捨圖：CER vs 端到端首字延遲，每個模型一條線、每個點一個 penalty。

讀 `eval/run_eval.py --blank-penalty ...` 的結果 JSON（檔名 `..._lead1000_bp<值>.json`，0 時沒有後綴）。
點 = 平均，誤差線 = CER 以句重抽的 95% 信賴區間。CER 失控的 penalty（例如 8）用 `--penalties` 排除後另在文件列出。

用法：
    uv run scripts/plot_blank_penalty.py     # 300 句對齊集，penalty 0 1 2 4 6，輸出 docs/blank_penalty_tradeoff.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = {"x-asr-480ms-int8": "X-ASR 480 ms", "x-asr-160ms-int8": "X-ASR 160 ms"}
COLORS = {"x-asr-480ms-int8": "#2a78d6", "x-asr-160ms-int8": "#eb6834"}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def result_path(results_dir: Path, dataset: str, model: str, config: str, bp: float) -> Path:
    suffix = f"_bp{bp:g}" if bp else ""
    files = sorted(results_dir.glob(f"*_{dataset}_{model}_{config}{suffix}.json"))
    if not files:
        raise FileNotFoundError(f"找不到 {results_dir}/*_{dataset}_{model}_{config}{suffix}.json")
    return files[-1]


def plot(results: dict[str, dict[float, dict[str, Any]]], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for model, by_bp in results.items():
        color = COLORS[model]
        xs, ys = [], []
        for bp, res in sorted(by_bp.items()):
            m = res["metrics"]
            x, y = m["e2e_first_token_latency_ms"]["mean"], m["cer"] * 100
            lo, hi = (v * 100 for v in m["cer_ci95"]["utterance"])
            ax.plot([x, x], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([x], [y], s=56, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
            # 兩條線的點會重疊（例如 480 ms 的 4 與 160 ms 的 1），480 ms 的標籤放在點下方
            offset = (6, -12) if model == "x-asr-480ms-int8" else (6, 4)
            ax.annotate(f"{bp:g}", (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                        color=TEXT_2)  # fmt: skip
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=color, lw=2, zorder=1, label=MODELS[model])
    ax.set_xlabel("End-to-end first-token latency, mean (ms)", color=TEXT_2)
    ax.set_ylabel("CER (%)  -  bar: 95% CI", color=TEXT_2)
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right",
              title="label = blank_penalty", title_fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="畫 blank_penalty 掃描的 CER vs 端到端首字延遲取捨圖。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    p.add_argument("--dataset", default="aishell1_align_300", help="結果檔名中的資料集名稱")
    p.add_argument("--config", default="sil500_pre100_chunk10_t1_lead1000")
    p.add_argument("--penalties", type=float, nargs="+", default=[0, 1, 2, 4, 6])
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "blank_penalty_tradeoff.png")
    args = p.parse_args()

    results: dict[str, dict[float, dict[str, Any]]] = {}
    envs: set[str] = set()
    try:
        for model in MODELS:
            for bp in args.penalties:
                path = result_path(args.results_dir, args.dataset, model, args.config, bp)
                res = json.loads(path.read_text(encoding="utf-8"))
                results.setdefault(model, {})[bp] = res
                env = res["environment"]
                envs.add(f"{env['chip']} @ {env['commit']}{' dirty' if env['dirty'] else ''}")
    except FileNotFoundError as e:
        print(f"錯誤：{e}，請先跑 run_eval.py --blank-penalty ...", file=sys.stderr)
        return 1
    n = {r["metrics"]["num_utterances"] for by_bp in results.values() for r in by_bp.values()}
    title = (
        f"blank_penalty sweep, AISHELL-1 test {'/'.join(map(str, sorted(n)))} utts, "
        f"{'; '.join(sorted(envs))}"
    )
    plot(results, args.out, title)
    print(f"圖：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
