"""記憶體分階段探查（backlog B11）：RSS、physical footprint 與 malloc 的存活配置 / 碎片。

physical footprint 是 iOS jetsam 判斷記憶體壓力的依據，比 RSS 更接近裝置上的實際代價。數字取自
macOS `vmmap --summary <自己的 pid>`：
- footprint：Physical footprint（目前 / 程序生命期峰值）。
- malloc 存活 / 碎片：MALLOC ZONE 表 TOTAL 列的 BYTES ALLOCATED / FRAG SIZE（已釋放但未歸還系統）。

兩種模式，都在獨立子程序中執行：

    uv run scripts/probe_footprint.py --model sensevoice-small-int8 x-asr-480ms-int8
        經專案的 ASR 後端載入，記錄 啟動 → import → 載入後 → 依序解碼 3 / 5 / 10 / 20 / 30 s 音訊後。
    uv run scripts/probe_footprint.py --ort-direct models/<目錄>/model.int8.onnx --ort-versions 1.28.0 1.31.1
        用指定版本的 onnxruntime（`uv run --no-project --with onnxruntime==<版本>` 臨時環境，不動專案依賴）
        直接載入單一 onnx 檔，比較「從檔案路徑」與「先讀進記憶體再建 session」（sherpa-onnx 非串流模型的
        做法，見 csrc/offline-sense-voice-model.cc 的 ReadFile）兩種載入方式。只載入、不推論。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WAV_DIR = REPO_ROOT / "data" / "aishell1_test_200" / "wav"
DURATIONS_S = [3, 5, 10, 20, 30]
_UNIT = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}


def _mb(token: str) -> float:
    m = re.match(r"([\d.]+)([KMG]?)", token)
    if m is None:
        raise ValueError(f"無法解析 vmmap 數值：{token}")
    return float(m.group(1)) * _UNIT.get(m.group(2), 1 / 2**20)


def snapshot(label: str) -> dict[str, Any]:
    import resource

    import psutil  # type: ignore[import-untyped]

    out = subprocess.run(
        ["vmmap", "--summary", str(os.getpid())], capture_output=True, text=True, check=True
    ).stdout
    lines = out.splitlines()
    footprint = next(_mb(x.split()[-1]) for x in lines if x.startswith("Physical footprint:"))
    peak = next(_mb(x.split()[-1]) for x in lines if x.startswith("Physical footprint (peak):"))
    zone = next(i for i, x in enumerate(lines) if x.startswith("MALLOC ZONE"))
    total = next(x for x in lines[zone:] if x.startswith("TOTAL")).split()
    return {
        "label": label,
        "rss": psutil.Process().memory_info().rss / 2**20,
        "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # macOS 單位 bytes
        "footprint": footprint,
        "footprint_peak": peak,
        "malloc_allocated": _mb(total[6]),
        "malloc_frag": _mb(total[7]),
    }


def worker_asr(model: str) -> dict[str, Any]:
    snaps = [snapshot("啟動")]
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    from src.asr import load_asr
    from src.audio import SAMPLE_RATE, load_wav

    snaps.append(snapshot("import 後"))
    asr = load_asr(model)
    snaps.append(snapshot("載入後"))
    pool = np.concatenate([load_wav(p) for p in sorted(WAV_DIR.glob("*.wav"))[:20]])
    for sec in DURATIONS_S:
        audio = pool[: sec * SAMPLE_RATE]
        stream = asr.create_stream()
        for i in range(0, len(audio), SAMPLE_RATE // 100):
            stream.accept_waveform(audio[i : i + SAMPLE_RATE // 100])
        stream.finalize()
        snaps.append(snapshot(f"解碼 {sec} s 後"))
    return {"model": model, "snapshots": snaps}


def worker_ort(path: str, mode: str) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    before = snapshot("載入前")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    if mode == "bytes":
        with open(path, "rb") as f:
            buf = f.read()
        session = ort.InferenceSession(buf, opts, providers=["CPUExecutionProvider"])
        del buf
    else:
        session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    after = snapshot("載入後")
    del session
    return {"ort": ort.__version__, "mode": mode, "before": before, "after": after}


def _run(cmd: list[str], job: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [*cmd, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"子程序失敗：\n{proc.stderr}")
    result: dict[str, Any] = json.loads(proc.stdout.splitlines()[-1])
    return result


def main() -> int:
    if "--worker" in sys.argv[1:]:
        job = json.loads(sys.stdin.read())
        res = (
            worker_asr(job["model"])
            if job["kind"] == "asr"
            else worker_ort(job["path"], job["mode"])
        )
        print(json.dumps(res))
        return 0
    if sys.platform != "darwin":
        print("錯誤：需要 macOS 的 vmmap", file=sys.stderr)
        return 1
    p = argparse.ArgumentParser(
        description="分階段量 RSS / physical footprint / malloc 碎片（backlog B11）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", nargs="+", default=["sensevoice-small-int8", "x-asr-480ms-int8"])
    p.add_argument("--ort-direct", type=Path, help="改為直接用 onnxruntime 載入這個 onnx 檔")
    p.add_argument("--ort-versions", nargs="+", default=["1.28.0", "1.31.1"])
    args = p.parse_args()

    if args.ort_direct:
        print("| ORT | 載入方式 | ΔRSS | Δfootprint | footprint 峰值 | Δmalloc 存活 | Δ碎片 |")
        print("|---|---|---|---|---|---|---|")
        for version in args.ort_versions:
            cmd = ["uv", "run", "--no-project", "--quiet", "--with", f"onnxruntime=={version}",
                   "--with", "psutil", "python", "-I"]  # fmt: skip
            for mode in ("path", "bytes"):
                r = _run(cmd, {"kind": "ort", "path": str(args.ort_direct.resolve()), "mode": mode})
                b, a = r["before"], r["after"]
                print(
                    f"| {r['ort']} | {'檔案路徑' if mode == 'path' else '先讀進記憶體'} "
                    f"| {a['rss'] - b['rss']:.0f} | {a['footprint'] - b['footprint']:.0f} "
                    f"| {a['footprint_peak']:.0f} | {a['malloc_allocated'] - b['malloc_allocated']:.0f} "
                    f"| {a['malloc_frag'] - b['malloc_frag']:.0f} |"
                )
        return 0

    for model in args.model:
        r = _run([sys.executable, "-I"], {"kind": "asr", "model": model})
        print(f"\n{model}（MB）\n")
        print("| 階段 | RSS | peak RSS | footprint | footprint 峰值 | malloc 存活 | 碎片 |")
        print("|---|---|---|---|---|---|---|")
        for s in r["snapshots"]:
            print(
                f"| {s['label']} | {s['rss']:.0f} | {s['peak_rss']:.0f} | {s['footprint']:.0f} "
                f"| {s['footprint_peak']:.0f} | {s['malloc_allocated']:.0f} | {s['malloc_frag']:.0f} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
