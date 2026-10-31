"""字級強制對齊集：從 3000 句評測集抽 300 句，用 Montreal Forced Aligner（MFA）取得每個字的起止時間。

用途是拆解首字延遲（docs/latency-analysis.md）：需要與 VAD、ASR 都無關的「真實首字起點」。

MFA 不放進專案依賴（需要 conda 的 Kaldi 執行檔），先另外建環境；中文模型下載到 models/mfa/：
    conda create -n mfa --override-channels -c conda-forge montreal-forced-aligner=3.4.3
    B=https://github.com/MontrealCorpusTools/mfa-models/releases/download
    curl -L -o models/mfa/mandarin_mfa.zip $B/acoustic-mandarin_mfa-v3.0.0/mandarin_mfa.zip
    curl -L -o models/mfa/mandarin_china_mfa.dict \
        $B/dictionary-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.dict
    curl -L -o models/mfa/g2p_mandarin_china_mfa.zip \
        $B/g2p-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.zip
（`mfa model download` 也可以；網路不穩時它沒有重試。上游沒有公布 checksum，下方 MODEL_SHA256 是
2026-10-31 下載時的值，執行前會校驗。）
模型：Mandarin MFA acoustic model v3.0.0（GMM-HMM，CC BY 4.0，訓練資料為 Common Voice zh-CN / zh-TW、
aidatatang_200zh、AISHELL-3、THCHS-30、GlobalPhone，不含 AISHELL-1）。

用法：
    uv run scripts/prepare_alignment.py              # 抽樣、建語料、跑 MFA、轉成字級結果、匯出 10 句供人工抽查
    uv run scripts/prepare_alignment.py -n 300 --check 10 --jobs 4

輸出 data/aishell1_align_300/（不進 git）：
    manifest.jsonl     與評測集相同格式（run_eval.py --manifest 可直接用）
    alignments.jsonl   每句 {"id", "chars": [{"char", "start", "end"}], "phones": [...]}，秒，原始音訊時間軸
    check/             抽查用：<id>.wav、<id>.TextGrid（Praat）、<id>.txt（Audacity 標籤，檔案 → 匯入 → 標籤）
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data" / "aishell1_test_3000" / "manifest.jsonl"
SEED = 20261031
SILENCE_LABELS = {"", "<eps>", "sil", "sp", "spn"}
MFA_MODELS = REPO_ROOT / "models" / "mfa"
MODEL_SHA256 = {
    "mandarin_mfa.zip": "c4c91c499ada0de35346481d4af69c7328075ddb0ce16a8cfb19fc74d0401199",
    "mandarin_china_mfa.dict": "0881d4e338e26509d022992cb5b897ead24a176448f534481238cb3a7fe949f0",
    "g2p_mandarin_china_mfa.zip": "005a9cc869d606a19ae3338cdcdd283c8b6384968cfed76c0e30fa58137a6034",
}


def check_models() -> str | None:
    """校驗 models/mfa/ 的模型檔；有問題時回傳錯誤訊息。"""
    import hashlib

    for name, expected in MODEL_SHA256.items():
        path = MFA_MODELS / name
        if not path.is_file():
            return f"找不到 {path}，下載方式見本檔說明（--help）"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            return f"{path} 的 SHA-256 是 {actual}，預期 {expected}"
    return None


def sample_rows(n: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in rows}
    picked = sorted(random.Random(SEED).sample(sorted(by_id), n))
    return [by_id[i] for i in picked]


def build_corpus(rows: list[dict[str, Any]], out: Path) -> None:
    """manifest + MFA 語料（每位說話人一個目錄，wav 以 hardlink 共用，.lab 為逐字空白分隔）。"""
    corpus = out / "mfa_corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    lines = []
    for r in rows:
        src = SOURCE.parent / r["wav"]
        chars = [c for c in r["text"] if not c.isspace()]
        speaker = corpus / r["id"][6:11]  # BAC009S0764W0121 → S0764
        speaker.mkdir(parents=True, exist_ok=True)
        os.link(src, speaker / f"{r['id']}.wav")
        # 以字為單位對齊：字層的邊界即每個字的起止，不必從音素再切字
        (speaker / f"{r['id']}.lab").write_text(" ".join(chars) + "\n", encoding="utf-8")
        lines.append(json.dumps({**r, "wav": str(src)}, ensure_ascii=False))
    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_mfa(out: Path, jobs: int) -> None:
    cmd = [
        "conda", "run", "-n", "mfa", "--no-capture-output",
        "mfa", "align", str(out / "mfa_corpus"), str(MFA_MODELS / "mandarin_china_mfa.dict"),
        str(MFA_MODELS / "mandarin_mfa.zip"), str(out / "mfa_output"),
        # 詞典沒有的字以 G2P 產生發音（300 句的 1051 個字都在詞典裡，保留以備換句子）
        "--g2p_model_path", str(MFA_MODELS / "g2p_mandarin_china_mfa.zip"),
        # .lab 已逐字以空白分隔，關掉 MFA 內建的中文分詞（否則需另裝 spacy-pkuseg 等套件）
        "--no_tokenization",
        "--output_format", "json", "--clean", "--overwrite", "-j", str(jobs),
    ]  # fmt: skip
    print("執行：" + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _intervals(tier: dict[str, Any]) -> list[tuple[float, float, str]]:
    return [(float(s), float(e), str(lab)) for s, e, lab in tier["entries"]]


def convert(rows: list[dict[str, Any]], out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """MFA JSON → 字級結果。字數與參考不符（含漏對齊）的句子列入 failed。"""
    results, failed = [], []
    for r in rows:
        path = out / "mfa_output" / r["id"][6:11] / f"{r['id']}.json"
        if not path.is_file():
            failed.append(r["id"])
            continue
        tiers = json.loads(path.read_text(encoding="utf-8"))["tiers"]
        words = [w for w in _intervals(tiers["words"]) if w[2] not in SILENCE_LABELS]
        phones = [p for p in _intervals(tiers["phones"]) if p[2] not in SILENCE_LABELS]
        ref = [c for c in r["text"] if not c.isspace()]
        # MFA 會把全形英數轉成半形小寫（例如 ｋ → k），比對前兩邊做同樣的轉換
        if [_fold(w[2]) for w in words] != [_fold(c) for c in ref]:
            failed.append(r["id"])
            continue
        results.append(
            {
                "id": r["id"],
                "chars": [{"char": c, "start": s, "end": e} for s, e, c in words],
                "phones": [{"phone": p, "start": s, "end": e} for s, e, p in phones],
            }
        )
    return results, failed


def _textgrid(duration: float, tiers: dict[str, list[tuple[float, float, str]]]) -> str:
    """Praat long-format TextGrid，區間之間的空白補成空標籤。"""
    out = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0", f"xmax = {duration}", "tiers? <exists>", f"size = {len(tiers)}", "item []:",
    ]  # fmt: skip
    for k, (name, spans) in enumerate(tiers.items(), 1):
        filled: list[tuple[float, float, str]] = []
        t = 0.0
        for s, e, lab in spans:
            if s > t:
                filled.append((t, s, ""))
            filled.append((s, e, lab))
            t = e
        if t < duration:
            filled.append((t, duration, ""))
        out += [
            f"    item [{k}]:", '        class = "IntervalTier"', f'        name = "{name}"',
            "        xmin = 0", f"        xmax = {duration}",
            f"        intervals: size = {len(filled)}",
        ]  # fmt: skip
        for i, (s, e, lab) in enumerate(filled, 1):
            out += [
                f"        intervals [{i}]:", f"            xmin = {s}", f"            xmax = {e}",
                f'            text = "{lab}"',
            ]  # fmt: skip
    return "\n".join(out) + "\n"


def export_check(
    results: list[dict[str, Any]], rows: list[dict[str, Any]], out: Path, n: int
) -> None:
    import soundfile as sf  # type: ignore[import-untyped]

    check = out / "check"
    shutil.rmtree(check, ignore_errors=True)
    check.mkdir()
    wav_of = {r["id"]: SOURCE.parent / r["wav"] for r in rows}
    for res in random.Random(SEED).sample(results, min(n, len(results))):
        uid, wav = res["id"], wav_of[res["id"]]
        shutil.copy(wav, check / f"{uid}.wav")
        duration = sf.info(str(wav)).duration
        chars = [(c["start"], c["end"], c["char"]) for c in res["chars"]]
        phones = [(p["start"], p["end"], p["phone"]) for p in res["phones"]]
        (check / f"{uid}.TextGrid").write_text(
            _textgrid(duration, {"chars": chars, "phones": phones}), encoding="utf-8"
        )
        (check / f"{uid}.txt").write_text(
            "".join(f"{s:.3f}\t{e:.3f}\t{lab}\n" for s, e, lab in chars), encoding="utf-8"
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="抽樣 AISHELL-1 test 句子並以 MFA 做字級強制對齊。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-n", type=int, default=300, help="句數（從 3000 句評測集以固定 seed 抽樣）")
    p.add_argument("--check", type=int, default=10, help="匯出幾句供人工抽查")
    p.add_argument("--jobs", type=int, default=4, help="MFA 平行工作數")
    p.add_argument("--skip-mfa", action="store_true", help="沿用既有的 MFA 輸出，只重新轉換與匯出")
    args = p.parse_args()

    if not SOURCE.is_file():
        print(
            f"錯誤：找不到 {SOURCE}，請先執行 uv run scripts/prepare_eval_data.py", file=sys.stderr
        )
        return 1
    if not args.skip_mfa:
        if shutil.which("conda") is None:
            print("錯誤：找不到 conda，MFA 的安裝方式見本檔說明（--help）", file=sys.stderr)
            return 1
        if (err := check_models()) is not None:
            print(f"錯誤：{err}", file=sys.stderr)
            return 1
    out = REPO_ROOT / "data" / f"aishell1_align_{args.n}"
    out.mkdir(parents=True, exist_ok=True)
    rows = sample_rows(args.n)
    if not args.skip_mfa:
        build_corpus(rows, out)
        run_mfa(out, args.jobs)
    results, failed = convert(rows, out)
    (out / "alignments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    (out / "meta.json").write_text(
        json.dumps(
            {"source": str(SOURCE.relative_to(REPO_ROOT)), "seed": SEED, "n": args.n,
             "aligned": len(results), "failed": failed,
             "mfa": "3.4.3, acoustic mandarin_mfa v3.0.0, dictionary / g2p mandarin_china_mfa"},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    export_check(results, rows, out, args.check)
    print(f"對齊成功 {len(results)} / {len(rows)} 句，失敗 {len(failed)} 句：{failed[:10]}")
    print(f"輸出：{out.relative_to(REPO_ROOT)}（抽查檔在 check/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
