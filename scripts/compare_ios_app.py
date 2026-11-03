"""比對 iPhone app 自動驗證輸出（simulator / 真機取回的 JSON）與 Mac 參考輸出的定稿文字與延遲。

用法：
  uv run scripts/compare_ios_app.py ref.jsonl ios.json [--label 說明]
ref 可以是 cpp/build/ozasr_cli（模擬即時）或 OzAsrMac --headless（牆鐘）的輸出；檔案以檔名（basename）對應。
定稿文字與句段數必須逐字相同；另列首字 / 尾字 / 放開→定稿延遲、RTF、載入時間、RSS。
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
            files[Path(o["wav"]).name] = o["utterances"]
        elif o["type"] == "summary":
            summary = o
    return files, summary


def lat(files: dict[str, list[dict[str, Any]]], key: str) -> list[float]:
    return [u[key] for us in files.values() for u in us if u.get(key) is not None]


def fmt(v: list[float]) -> str:
    return f"均值 {statistics.mean(v) * 1000:.0f} ms / 最大 {max(v) * 1000:.0f} ms" if v else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("ios", type=Path)
    args = ap.parse_args()
    ref, rs = load(args.ref)
    ios, isum = load(args.ios)
    common = sorted(ref.keys() & ios.keys())
    if not common:
        print("沒有共同檔案", file=sys.stderr)
        return 2
    diff = 0
    for name in common:
        a = [u["text"] for u in ref[name]]
        b = [u["text"] for u in ios[name]]
        if a != b:
            diff += 1
            print(f"不同 {name}\n  ref: {a}\n  ios: {b}")
        for ua, ub in zip(ref[name], ios[name], strict=False):
            if abs(ua["speech_start_s"] - ub["speech_start_s"]) > 1e-3 or abs(
                (ua["speech_end_s"] or 0) - (ub["speech_end_s"] or 0)
            ) > 1e-3:
                print(f"VAD 切點不同 {name}")
    print(f"共同檔案 {len(common)}，定稿文字或句段數不同：{diff} / {len(common)}")
    for k, label in [
        ("first_token_latency_s", "首字"),
        ("final_latency_s", "尾字"),
        ("release_to_final_latency_s", "放開→定稿"),
    ]:
        print(f"{label}：ref {fmt(lat(ref, k))}　ios {fmt(lat(ios, k))}")
    for k in ["rtf", "load_s", "rss_after_load_mb", "peak_rss_mb"]:
        print(f"{k}：ref {rs.get(k)}　ios {isum.get(k)}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
