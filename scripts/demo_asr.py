"""對一個 wav 模擬串流辨識：逐 chunk 送入，即時印出 partial 結果，最後印出最終結果。

用法：
    uv run scripts/demo_asr.py                         # 用模型自帶的測試音訊，依實際時長節奏送入
    uv run scripts/demo_asr.py input.wav --chunk-ms 100
    uv run scripts/demo_asr.py input.wav --fast        # 不等待，盡快送完（看計算速度）

wav 需為 16 kHz（多聲道會取平均）。模型需先以 bash scripts/download_models.sh 下載。
這裡印出的時間與 RTF 只是單次量測，正式指標請用 eval/ 的腳本。
"""

import argparse
import sys
import time
from pathlib import Path

# 讓 `uv run scripts/demo_asr.py` 能直接 import src（專案本身沒有安裝成套件）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import DEFAULT_MODEL, SAMPLE_RATE, available_models, load_asr
from src.asr.sherpa import ZIPFORMER_ZH_14M
from src.audio import load_wav
from src.cli import positive_int
from src.paths import MODELS_DIR

DEFAULT_WAV = MODELS_DIR / ZIPFORMER_ZH_14M.model_dir / "test_wavs" / "0.wav"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="串流 ASR demo：輸入 wav，模擬即時逐 chunk 送入，印出 partial 與最終結果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("wav", nargs="?", type=Path, default=DEFAULT_WAV, help="16 kHz wav 檔")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=available_models(), help="ASR 模型")
    p.add_argument("--threads", type=positive_int, help="推論執行緒數；不給時用模型預設")
    p.add_argument("--chunk-ms", type=positive_int, default=10, help="每次送入的音訊長度")
    p.add_argument("--fast", action="store_true", help="不依音訊時長節奏等待，盡快送完所有 chunk")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        # 先載入模型：預設 wav 在模型目錄內，模型未下載時應提示下載而不是「找不到 wav」
        t_load = time.perf_counter()
        asr = load_asr(args.model, num_threads=args.threads)
        load_s = time.perf_counter() - t_load
        audio = load_wav(args.wav)
    except (ValueError, FileNotFoundError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1

    duration = len(audio) / SAMPLE_RATE
    chunk = args.chunk_ms * SAMPLE_RATE // 1000
    print(f"檔案：{args.wav}（{duration:.2f} s）")
    print(f"模型：{asr.name}（載入 {load_s * 1000:.0f} ms）")
    print(f"chunk = {args.chunk_ms} ms，{'盡快送入' if args.fast else '依實際時長節奏送入'}\n")
    print(f"{'音訊 (s)':>8}  partial")

    stream = asr.create_stream()
    compute_s = 0.0
    last = ""
    first_partial_at: float | None = None
    start = time.perf_counter()
    for i in range(0, len(audio), chunk):
        if not args.fast:
            # 等到這個 chunk 在真實世界中「錄完」的時刻才送入
            delay = start + i / SAMPLE_RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        t0 = time.perf_counter()
        stream.accept_waveform(audio[i : i + chunk])
        compute_s += time.perf_counter() - t0
        partial = stream.get_partial_result()
        if partial != last:
            fed_s = min(i + chunk, len(audio)) / SAMPLE_RATE
            if first_partial_at is None:
                first_partial_at = fed_s
            print(f"{fed_s:>8.2f}  {partial}", flush=True)
            last = partial

    t0 = time.perf_counter()
    final = stream.finalize()
    finalize_s = time.perf_counter() - t0
    compute_s += finalize_s

    print(f"\n最終結果：{final}")
    if first_partial_at is not None:
        print(
            f"第一個 partial 出現在送入 {first_partial_at:.2f} s 音訊時（相對檔案開頭，非語音起點）"
        )
    print(f"finalize 耗時 {finalize_s * 1000:.0f} ms")
    print(
        f"計算時間 {compute_s * 1000:.0f} ms，RTF {compute_s / duration:.4f}"
        "（只計 accept_waveform + finalize，不含模型載入與等待）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
