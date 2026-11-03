"""比對 Mac app 的 --headless（牆鐘即時節奏）與 cpp/build/ozasr_cli（模擬值）的輸出。

用法：
  cpp/build/ozasr_cli --models-dir models --tail-silence-ms 1000 --warmup 1 --list list.txt > cli.jsonl
  build/mac/OzAsr.app/Contents/MacOS/OzAsrMac --headless --models models --warmup 1 --list list.txt > app.jsonl
  uv run scripts/compare_mac_app.py cli.jsonl app.jsonl
定稿文字與句段數必須逐字相同；另列牆鐘 vs 模擬的首字 / 尾字延遲、RTF、RSS、載入時間。
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    files: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o["type"] == "file":
            files[o["wav"]] = o["utterances"]
        elif o["type"] == "summary":
            summary = o
    return files, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cli_jsonl", type=Path)
    ap.add_argument("app_jsonl", type=Path)
    args = ap.parse_args()
    cli, cs = load(args.cli_jsonl)
    app, a_s = load(args.app_jsonl)
    if cli.keys() != app.keys():
        print("兩邊的檔案集合不同", file=sys.stderr)
        return 1
    n_utt = n_diff = 0
    cut_max = 0.0
    sim_first, sim_final, wall_first, wall_final = [], [], [], []
    for wav in cli:
        cu, au = cli[wav], app[wav]
        if len(cu) != len(au):
            print(f"句段數不同 {wav}: CLI {len(cu)} / app {len(au)}")
            n_diff += 1
            continue
        for c, a in zip(cu, au):
            n_utt += 1
            if c["text"] != a["text"]:
                n_diff += 1
                print(f"文字不同 {wav}#{c['index']}: CLI「{c['text']}」/ app「{a['text']}」")
            cut_max = max(cut_max, abs(c["speech_start_s"] - a["speech_start_s"]),
                          abs((c["speech_end_s"] or 0) - (a["speech_end_s"] or 0)))
            sim_first.append(c["first_token_s"] - c["speech_start_s"]) if c.get("first_token_s") is not None else None
            sim_final.append(c["final_s"] - c["speech_end_s"]) if c.get("final_s") is not None else None
            if a["first_token_latency_s"] is not None:
                wall_first.append(a["first_token_latency_s"])
            if a["final_latency_s"] is not None:
                wall_final.append(a["final_latency_s"])

    def stat(xs: list[float]) -> str:
        xs = sorted(xs)
        return f"均值 {statistics.mean(xs)*1000:.0f} ms / 中位 {statistics.median(xs)*1000:.0f} ms / p95 {xs[int(0.95*(len(xs)-1))]*1000:.0f} ms"

    print(f"\n檔案 {len(cli)}，句段 {n_utt}，文字不同 {n_diff}，VAD 切點最大差 {cut_max*1000:.1f} ms")
    print(f"首字延遲  CLI 模擬：{stat(sim_first)}\n          app 牆鐘：{stat(wall_first)}")
    print(f"尾字延遲  CLI 模擬：{stat(sim_final)}\n          app 牆鐘：{stat(wall_final)}")
    for k in ("rtf", "load_s", "rss_after_load_mb", "peak_rss_mb"):
        print(f"{k:18s} CLI {cs.get(k, float('nan')):.4f}   app {a_s.get(k, float('nan')):.4f}")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())
