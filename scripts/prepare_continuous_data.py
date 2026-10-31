"""合成連續多句評測集（backlog B4）：同一說話人的多句 AISHELL 句子以已知長度的停頓串接。

來源是 300 句 MFA 字級對齊集（`scripts/prepare_alignment.py`），每句的真實首字起點與末字結束點已知，
因此合成後每句在新音訊中的語音起訖點也精確已知，不依賴 VAD：
- 每句只保留 [首字起點 − margin, 末字結束 + margin] 的原始音訊（margin 預設 50 ms，保留弱起音 / 收尾）。
- 句間停頓 = 前一句末字結束 → 下一句首字起點，依 `--pauses` 循環指定（固定 seed 打亂後依序分配，
  各停頓長度的數量相同）。扣掉兩邊 margin 的部分填數位靜音（全零）。
- 每條第一句前、最後一句後只有 margin；評測時 run_eval 另補前後靜音。

已知限制：margin 內是原始錄音的底噪，中間是數位零，底噪會不連續；AISHELL 錄音安靜，影響應很小。
句中的自然停頓保留原樣，用來量句中誤切。

用法：
    uv run scripts/prepare_continuous_data.py          # → data/aishell1_continuous_300/
    uv run eval/run_eval.py --manifest data/aishell1_continuous_300/manifest.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio import SAMPLE_RATE, load_wav

ALIGN_DIR = REPO_ROOT / "data" / "aishell1_align_300"
DEFAULT_OUT = REPO_ROOT / "data" / "aishell1_continuous_300"
DEFAULT_PAUSES = [0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
SEED = 20261101


def speaker(uid: str) -> str:
    m = re.search(r"S\d{4}", uid)
    if m is None:
        raise ValueError(f"無法從 {uid} 解析說話人")
    return m.group(0)


def chunk_sentences(ids: list[str], per_file: int) -> list[list[str]]:
    """切成每組 per_file 句；最後不足的一組若只剩 1 句就併入前一組（每條至少 2 句才有句間停頓）。"""
    groups = [ids[i : i + per_file] for i in range(0, len(ids), per_file)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())
    return groups


def build(
    rows: list[dict[str, Any]],
    align: dict[str, dict[str, Any]],
    out_dir: Path,
    per_file: int,
    pauses: list[float],
    margin_s: float,
    seed: int,
) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[str]] = defaultdict(list)
    row_by_id = {r["id"]: r for r in rows}
    for r in rows:
        by_speaker[speaker(r["id"])].append(r["id"])
    rng = random.Random(seed)
    files: list[list[str]] = []
    for spk in sorted(by_speaker):
        ids = sorted(by_speaker[spk])
        rng.shuffle(ids)
        files.extend(chunk_sentences(ids, per_file))
    n_pauses = sum(len(f) - 1 for f in files)
    plan = [pauses[i % len(pauses)] for i in range(n_pauses)]
    rng.shuffle(plan)

    margin = int(margin_s * SAMPLE_RATE)
    (out_dir / "wav").mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for ids in files:
        pieces: list[npt.NDArray[np.float32]] = []
        sentences: list[dict[str, Any]] = []
        file_pauses: list[float] = []
        pos = 0  # 目前已寫入的樣本數
        for k, uid in enumerate(ids):
            audio = load_wav(Path(row_by_id[uid]["wav"]))
            chars = align[uid]["chars"]
            onset = round(chars[0]["start"] * SAMPLE_RATE)
            offset = round(chars[-1]["end"] * SAMPLE_RATE)
            lead = min(margin, onset)
            trail = min(margin, len(audio) - offset)
            if k > 0:
                pause = plan.pop()
                prev_trail = sentences[-1]["_trail"]
                gap = round(pause * SAMPLE_RATE) - prev_trail - lead
                if gap < 0:
                    raise ValueError(f"停頓 {pause} s 短於兩邊 margin，請調小 --margin-ms")
                pieces.append(np.zeros(gap, dtype=np.float32))
                pos += gap
                file_pauses.append(pause)
            pieces.append(audio[onset - lead : offset + trail])
            start = pos + lead
            end = start + (offset - onset)
            pos += lead + (offset - onset) + trail
            sentences.append(
                {
                    "id": uid,
                    "text": row_by_id[uid]["text"],
                    "start_s": start / SAMPLE_RATE,  # 真實首字起點（MFA）
                    "end_s": end / SAMPLE_RATE,  # 真實末字結束（MFA）
                    "_trail": trail,
                }
            )
        wav = np.concatenate(pieces)
        fid = "_".join([speaker(ids[0]), *(uid[-5:] for uid in ids)])
        sf.write(str(out_dir / "wav" / f"{fid}.wav"), wav, SAMPLE_RATE, subtype="PCM_16")
        for s in sentences:
            del s["_trail"]
        manifest.append(
            {
                "id": fid,
                "wav": f"wav/{fid}.wav",
                "text": "  ".join(s["text"] for s in sentences),
                "duration_s": len(wav) / SAMPLE_RATE,
                "sentences": sentences,
                "pauses_s": file_pauses,
            }
        )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(
        description="由 MFA 對齊集合成連續多句評測集（B4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--align-dir", type=Path, default=ALIGN_DIR, help="prepare_alignment.py 的輸出")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="輸出目錄")
    p.add_argument("--per-file", type=int, default=4, help="每條音訊的句數")
    p.add_argument(
        "--pauses", type=float, nargs="+", default=DEFAULT_PAUSES, help="句間停頓長度（秒）"
    )
    p.add_argument("--margin-ms", type=int, default=50, help="每句真實起訖點外保留的原始音訊")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    manifest_path, align_path = (
        args.align_dir / "manifest.jsonl",
        args.align_dir / "alignments.jsonl",
    )
    if not manifest_path.is_file() or not align_path.is_file():
        print(f"錯誤：找不到 {args.align_dir}，請先執行 uv run scripts/prepare_alignment.py")
        return 1
    if args.per_file < 2:
        print("錯誤：--per-file 至少 2")
        return 1
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    align = {
        (a := json.loads(line))["id"]: a
        for line in align_path.read_text(encoding="utf-8").splitlines()
    }
    rows = [r for r in rows if r["id"] in align]
    manifest = build(
        rows, align, args.out, args.per_file, args.pauses, args.margin_ms / 1000, args.seed
    )
    (args.out / "manifest.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8"
    )
    pause_count: dict[str, int] = defaultdict(int)
    for m in manifest:
        for x in m["pauses_s"]:
            pause_count[f"{x:g}"] += 1
    meta = {
        "source": str(args.align_dir.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "per_file": args.per_file,
        "margin_ms": args.margin_ms,
        "files": len(manifest),
        "sentences": sum(len(m["sentences"]) for m in manifest),
        "pauses": dict(sorted(pause_count.items(), key=lambda kv: float(kv[0]))),
        "duration_s": sum(m["duration_s"] for m in manifest),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
