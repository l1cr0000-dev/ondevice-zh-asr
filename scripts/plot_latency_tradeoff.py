"""chunk 大小 vs CER vs 端到端首字延遲的取捨圖（backlog B1）。

讀 `eval/run_eval.py` 的結果 JSON，每個模型一個點：橫軸端到端首字延遲（點 = 平均，右側細線延伸到 P90），
縱軸 CER（誤差線 = 以句重抽的 95% 信賴區間），點旁標註 chunk 大小。同一系列（X-ASR）的點以線相連。
每個模型取結果目錄中日期最新的 `<日期>_<資料集>_<模型>_<設定>.json`。

用法：
    uv run scripts/plot_latency_tradeoff.py          # 3000 句集、預設設定，輸出 docs/latency_tradeoff.png
    uv run scripts/plot_latency_tradeoff.py --dataset aishell1_test_200 --out /tmp/t.png
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

# 模型名稱 → (圖例系列, 顯示名稱, chunk ms)。chunk 取自 encoder metadata 的 decode_chunk_len（10 ms / frame）。
MODELS: dict[str, tuple[str, str, int]] = {
    "x-asr-160ms-int8": ("X-ASR (INT8)", "X-ASR 160 ms", 160),
    "x-asr-480ms-int8": ("X-ASR (INT8)", "X-ASR 480 ms", 480),
    "zipformer-zh-14m": ("zh-14M (FP32)", "zh-14M 320 ms", 320),
}
SERIES_COLORS = {"X-ASR (INT8)": "#2a78d6", "zh-14M (FP32)": "#eb6834"}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def latest_result(results_dir: Path, dataset: str, model: str, config: str) -> Path:
    files = sorted(results_dir.glob(f"*_{dataset}_{model}_{config}.json"))
    if not files:
        raise FileNotFoundError(
            f"找不到 {model} 的結果（{results_dir}/*_{dataset}_{model}_{config}.json），"
            "請先執行 eval/run_eval.py"
        )
    return files[-1]


def plot(results: dict[str, dict[str, Any]], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    by_series: dict[str, list[tuple[float, float]]] = {}
    for model, res in results.items():
        series, label, _ = MODELS[model]
        m = res["metrics"]
        e2e = m["e2e_first_token_latency_ms"]
        cer = m["cer"] * 100
        lo, hi = (v * 100 for v in m["cer_ci95"]["utterance"])
        color = SERIES_COLORS[series]
        # 平均到 P90 的細線、CER 信賴區間、平均的點（帶表面色描邊，與線重疊時分得開）
        ax.plot([e2e["mean"], e2e["p90"]], [cer, cer], color=color, lw=1, alpha=0.6, zorder=2)
        ax.plot([e2e["mean"]] * 2, [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(
            [e2e["mean"]], [cer], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3
        )
        ax.annotate(
            label,
            (e2e["mean"], cer),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
        )
        by_series.setdefault(series, []).append((e2e["mean"], cer))
    for series, pts in by_series.items():
        pts.sort()
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            color=SERIES_COLORS[series], lw=2, zorder=1, label=series,
        )  # fmt: skip
        if len(pts) == 1:  # 單點系列也要出現在圖例
            ax.lines[-1].set_marker("o")
    ax.set_xlabel(
        "End-to-end first-token latency (ms)  -  dot: mean, thin line: to P90", color=TEXT_2
    )
    ax.set_ylabel("CER (%)  -  bar: 95% CI", color=TEXT_2)
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="畫 chunk 大小 vs CER vs 端到端首字延遲的取捨圖。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    p.add_argument("--dataset", default="aishell1_test_3000", help="結果檔名中的資料集名稱")
    p.add_argument(
        "--config",
        default="sil500_pre100_chunk10_t1_lead1000",
        help="結果檔名中模型名稱之後的設定部分",
    )
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "latency_tradeoff.png")
    args = p.parse_args()

    try:
        paths = {
            m: latest_result(args.results_dir, args.dataset, m, args.config) for m in args.models
        }
    except FileNotFoundError as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1
    results = {m: json.loads(path.read_text(encoding="utf-8")) for m, path in paths.items()}
    envs = {
        (r["environment"]["commit"] + (" dirty" if r["environment"]["dirty"] else ""),
         r["environment"]["chip"])
        for r in results.values()
    }  # fmt: skip
    n = {r["metrics"]["num_utterances"] for r in results.values()}
    title = (
        f"AISHELL-1 test, {'/'.join(map(str, sorted(n)))} utts, VAD sil 500 ms, pre-roll 100 ms, "
        f"1 thread, {'; '.join(f'{chip} @ {c}' for c, chip in sorted(envs))}"
    )
    plot(results, args.out, title)
    for m, path in paths.items():
        print(f"{m}: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"圖：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
