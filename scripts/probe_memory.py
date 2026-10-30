"""記憶體探查（backlog B11）：比較 ONNX Runtime session 設定對 ASR 載入後 RSS 與峰值 RSS 的影響。

每個變體在獨立子程序中執行，依序記錄 RSS：直譯器啟動（只有標準庫）→ import numpy / sherpa-onnx 後
→ 載入 ASR 模型後 → 解碼 N 句後，以及 peak RSS（ru_maxrss）。不經過 VAD / pipeline，
整句送入後補零 finalize，只為了讓推論時的記憶體配置發生；辨識結果不評分。

變體以 sherpa-onnx 的 provider 設定檔傳給 ONNX Runtime（`provider="cpu:<設定檔>"`，
sherpa-onnx 2.0.1 `csrc/session.cc` 的 `GetSessionOptionsImpl` 解析）：

    default     不給設定檔（與 src/asr 相同）
    no-arena    EnableCpuMemArena=0
    no-prepack  SessionConfig.session.disable_prepacking=1
    no-opt      GraphOptimizationLevel=0（ORT_DISABLE_ALL）

另外 `--ort-direct` 用 onnxruntime Python 套件直接逐檔載入 encoder / decoder / joiner
（arena 開 / 關），列出每個檔案載入增加的 RSS，與經過 sherpa-onnx 載入的結果對照。

用法：
    uv run scripts/probe_memory.py                        # X-ASR，全部變體，30 句
    uv run scripts/probe_memory.py --model zipformer-zh-14m --variants default no-arena -n 100
    uv run scripts/probe_memory.py --ort-direct
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV_DIR = REPO_ROOT / "data" / "aishell1_test_200" / "wav"

VARIANTS: dict[str, list[str]] = {
    "default": [],
    "no-arena": ["EnableCpuMemArena=0"],
    "no-prepack": ["SessionConfig.session.disable_prepacking=1"],
    "no-opt": ["GraphOptimizationLevel=0"],
}


def _rss_mb() -> float:
    import psutil  # type: ignore[import-untyped]

    return float(psutil.Process().memory_info().rss) / 2**20


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # macOS bytes、Linux KB


def worker_sherpa(job: dict[str, Any]) -> dict[str, Any]:
    rss_start = _rss_mb()
    import numpy as np
    import sherpa_onnx  # type: ignore[import-untyped]

    sys.path.insert(0, str(REPO_ROOT))
    from src.asr import TRANSDUCER_SPECS
    from src.audio import SAMPLE_RATE, load_wav

    rss_imported = _rss_mb()
    files, tail_padding_s = TRANSDUCER_SPECS[job["model"]]
    paths = files.paths()
    with tempfile.TemporaryDirectory() as tmp:
        provider = "cpu"
        if job["config_lines"]:
            cfg = Path(tmp) / "provider.cfg"
            cfg.write_text("\n".join(job["config_lines"]) + "\n")
            provider = f"cpu:{cfg}"  # 設定檔只在建立 session 時讀取
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=1,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=False,
            provider=provider,
        )
    rss_loaded = _rss_mb()
    padding = np.zeros(int(tail_padding_s * SAMPLE_RATE), dtype=np.float32)
    for wav in job["wavs"]:
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, load_wav(wav))
        stream.accept_waveform(SAMPLE_RATE, padding)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
    return {
        "start": rss_start,
        "imported": rss_imported,
        "loaded": rss_loaded,
        "decoded": _rss_mb(),
        "peak": _peak_rss_mb(),
    }


def worker_ort(job: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    sys.path.insert(0, str(REPO_ROOT))
    from src.asr import TRANSDUCER_SPECS

    paths = TRANSDUCER_SPECS[job["model"]][0].paths()
    opts: Any = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = job["arena"]
    sessions: list[Any] = []  # 保留參照，避免載入下一個檔案前被釋放
    out: dict[str, Any] = {"files": {}}
    for part in ("encoder", "decoder", "joiner"):
        before = _rss_mb()
        sessions.append(
            ort.InferenceSession(str(paths[part]), opts, providers=["CPUExecutionProvider"])
        )
        out["files"][paths[part].name] = {
            "file_mb": paths[part].stat().st_size / 2**20,
            "rss_delta_mb": _rss_mb() - before,
        }
    return out


def run(job: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"子程序失敗：\n{proc.stderr}")
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


def parse_args() -> argparse.Namespace:
    sys.path.insert(0, str(REPO_ROOT))
    from src.asr import DEFAULT_MODEL, available_models
    from src.cli import positive_int

    p = argparse.ArgumentParser(
        description="比較 ONNX Runtime session 設定對 ASR 載入後與峰值 RSS 的影響（backlog B11）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=available_models(), default=DEFAULT_MODEL, help="ASR 模型")
    p.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS), help="設定變體"
    )
    p.add_argument(
        "-n", "--num", type=positive_int, default=30, help="解碼句數（依檔名排序取前 N）"
    )
    p.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR, help="16 kHz wav 目錄")
    p.add_argument(
        "--ort-direct", action="store_true", help="改用 onnxruntime 直接逐檔載入（arena 開 / 關）"
    )
    return p.parse_args()


def main() -> int:
    if "--worker" in sys.argv[1:]:
        job = json.loads(sys.stdin.read())
        print(json.dumps(worker_ort(job) if job["kind"] == "ort" else worker_sherpa(job)))
        return 0

    args = parse_args()
    if args.ort_direct:
        for arena in (True, False):
            res = run({"kind": "ort", "model": args.model, "arena": arena})
            for name, f in res["files"].items():
                print(
                    f"arena={'on ' if arena else 'off'}  {name:<36} 檔案 {f['file_mb']:6.1f} MB"
                    f"  載入 RSS +{f['rss_delta_mb']:.0f} MB"
                )
        return 0

    wavs = [str(p) for p in sorted(args.wav_dir.glob("*.wav"))[: args.num]]
    if not wavs:
        print(
            f"錯誤：{args.wav_dir} 沒有 wav，請先執行 uv run scripts/prepare_eval_data.py",
            file=sys.stderr,
        )
        return 1
    print(f"模型 {args.model}，解碼 {len(wavs)} 句（{args.wav_dir}）；單位 MB")
    print("| 變體 | 啟動 | import 後 | 載入後 | 解碼後 | peak | 載入增加 |")
    print("|---|---|---|---|---|---|---|")
    for name in args.variants:
        job = {"kind": "sherpa", "model": args.model, "config_lines": VARIANTS[name], "wavs": wavs}
        r = run(job)
        print(
            f"| {name} | {r['start']:.0f} | {r['imported']:.0f} | {r['loaded']:.0f} "
            f"| {r['decoded']:.0f} | {r['peak']:.0f} | {r['loaded'] - r['imported']:.0f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
