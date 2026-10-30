"""準備評測資料：從 AISHELL-1 test set（7176 句）以固定 seed 隨機抽樣，下載 wav 並寫出 manifest。

用法：
    uv run scripts/prepare_eval_data.py                 # 預設：200 句、seed 20261030
    uv run scripts/prepare_eval_data.py -n 500 --seed 1 --out data/aishell1_test_500

資料來源（固定 revision，確保每次抽到同樣的句子、拿到同樣的檔案）：
- 音訊：Hugging Face `shenyunhang/AISHELL-1`，保留官方 `data_aishell/wav/test/<說話人>/<句子>.wav`
  目錄結構，可逐條下載；每個檔案以 Hugging Face 提供的 LFS SHA-256 校驗。
  （官方 `AISHELL/AISHELL-1` repo 只上傳了 100 位說話人，沒有 test set；OpenSLR SLR33 只提供 15 GB
  的完整壓縮檔。）
- 標註：同 repo 的 `data_aishell/transcript/aishell_transcript_v0.8.txt`（與官方檔案大小相同），
  以下方寫死的 SHA-256 校驗。

抽樣：把 test set 全部句子 ID 排序後，`random.Random(seed).sample(ids, n)`，結果再依 ID 排序。
抽樣只依賴 ID 清單與 seed，不依賴下載順序。若 test set 句數不是 7176 會直接報錯。

輸出（data/ 不進 git）：
    <out>/wav/<句子ID>.wav
    <out>/manifest.jsonl   每行 {"id", "wav"（相對 manifest）, "text", "duration_s", "sha256"}
    <out>/meta.json        來源、revision、seed、抽樣數等

可重複執行：已存在且校驗通過的 wav 會跳過。
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import soundfile as sf  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]

HF_REPO = "shenyunhang/AISHELL-1"
HF_REVISION = "2724409d538167445e43ebf846990319f12a1cbf"
TEST_DIR = "data_aishell/wav/test"
TRANSCRIPT_PATH = "data_aishell/transcript/aishell_transcript_v0.8.txt"
TRANSCRIPT_SHA256 = "b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4"
TEST_SET_SIZE = 7176

DEFAULT_N = 200
DEFAULT_SEED = 20261030
DEFAULT_OUT = REPO_ROOT / "data" / "aishell1_test_200"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從 AISHELL-1 test set 以固定 seed 抽樣、下載 wav 並寫出 manifest。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", "--num", type=int, default=DEFAULT_N, help="抽樣句數")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="隨機種子")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="輸出目錄")
    p.add_argument("--jobs", type=int, default=8, help="並行下載數")
    return p.parse_args()


def _get(url: str, retries: int = 3) -> tuple[bytes, dict[str, str]]:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read(), dict(r.headers)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_tree(path: str) -> list[dict[str, Any]]:
    """列出 repo 某目錄（固定 revision），處理分頁。"""
    url: str | None = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}/{path}"
    items: list[dict[str, Any]] = []
    while url:
        body, headers = _get(url)
        items.extend(json.loads(body))
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        url = m.group(1) if m else None
    return items


def resolve_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{path}"


def load_transcripts() -> dict[str, str]:
    data, _ = _get(resolve_url(TRANSCRIPT_PATH))
    actual = hashlib.sha256(data).hexdigest()
    if actual != TRANSCRIPT_SHA256:
        raise RuntimeError(f"標註檔 checksum 不符：預期 {TRANSCRIPT_SHA256}，實際 {actual}")
    out: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def sample_ids(ids: list[str], n: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(ids), n))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: str, dest: Path, sha256: str) -> bool:
    """下載並校驗；已存在且校驗通過回傳 False（跳過）。"""
    if dest.is_file() and _sha256_file(dest) == sha256:
        return False
    data, _ = _get(resolve_url(path))
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"{path} checksum 不符：預期 {sha256}，實際 {actual}")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True


def main() -> int:
    args = parse_args()
    if not 0 < args.num <= TEST_SET_SIZE:
        print(f"錯誤：-n 必須在 1–{TEST_SET_SIZE}", file=sys.stderr)
        return 2

    print(f"列出 {HF_REPO}@{HF_REVISION[:10]} 的 test set …", file=sys.stderr)
    files: dict[str, dict[str, Any]] = {}
    for spk in list_tree(TEST_DIR):
        for f in list_tree(spk["path"]):
            if f["path"].endswith(".wav"):
                files[Path(f["path"]).stem] = f
    transcripts = load_transcripts()
    pool = [i for i in files if i in transcripts]
    if len(pool) != TEST_SET_SIZE:
        print(f"錯誤：test set 有標註的句子 {len(pool)} 句，預期 {TEST_SET_SIZE}", file=sys.stderr)
        return 1

    chosen = sample_ids(pool, args.num, args.seed)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    def fetch(uid: str) -> bool:
        f = files[uid]
        return download(f["path"], wav_dir / f"{uid}.wav", f["lfs"]["oid"])

    with ThreadPoolExecutor(args.jobs) as ex:
        fetched = sum(ex.map(fetch, chosen))
    print(
        f"下載 {fetched} 個、跳過 {len(chosen) - fetched} 個（已存在且校驗通過）", file=sys.stderr
    )

    total = 0.0
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as fp:
        for uid in chosen:
            info = sf.info(str(wav_dir / f"{uid}.wav"))
            if info.samplerate != 16000 or info.channels != 1:
                raise RuntimeError(
                    f"{uid}: {info.samplerate} Hz / {info.channels} ch，預期 16 kHz 單聲道"
                )
            total += info.duration
            row = {
                "id": uid,
                "wav": f"wav/{uid}.wav",
                "text": transcripts[uid],
                "duration_s": round(info.duration, 4),
                "sha256": files[uid]["lfs"]["oid"],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "AISHELL-1 test",
        "source": f"https://huggingface.co/datasets/{HF_REPO}",
        "revision": HF_REVISION,
        "transcript": TRANSCRIPT_PATH,
        "transcript_sha256": TRANSCRIPT_SHA256,
        "pool_size": len(pool),
        "num": args.num,
        "seed": args.seed,
        "sampling": "sorted(random.Random(seed).sample(sorted(ids), num))",
        "speakers": sorted({uid[6:11] for uid in chosen}),
        "total_duration_s": round(total, 2),
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"完成：{args.num} 句、{total / 60:.1f} 分鐘、{len(meta['speakers'])} 位說話人 → {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
