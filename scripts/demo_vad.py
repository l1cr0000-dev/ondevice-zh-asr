"""對一個 wav 跑 VAD，印出語音段時間。

用法：
    uv run scripts/demo_vad.py                       # 用內附的測試音訊
    uv run scripts/demo_vad.py input.wav --threshold 0.6 --min-silence-ms 300
    uv run scripts/demo_vad.py input.wav --stream    # 串流模式：逐 chunk 送入，印出事件與延遲

wav 需為 16 kHz（多聲道會取平均）。模型需先以 bash scripts/download_models.sh 下載。
"""

import argparse
import sys
import time
from pathlib import Path

# 讓 `uv run scripts/demo_vad.py` 能直接 import src（專案本身沒有安裝成套件）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.vad import (
    DEFAULT_MODEL_PATH,
    SAMPLE_RATE,
    SileroVadModel,
    StreamingVad,
    VadConfig,
    detect_segments,
    load_wav,
)

DEFAULT_WAV = REPO_ROOT / "tests" / "fixtures" / "vad_two_utterances.wav"


def parse_args() -> argparse.Namespace:
    defaults = VadConfig()
    p = argparse.ArgumentParser(
        description="Silero VAD demo：輸入 wav，印出語音段起止時間。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("wav", nargs="?", type=Path, default=DEFAULT_WAV, help="16 kHz wav 檔")
    p.add_argument("--threshold", type=float, default=defaults.threshold, help="語音機率閾值")
    p.add_argument(
        "--neg-threshold",
        type=float,
        default=None,
        help="結束語音的機率閾值（遲滯）；預設 threshold - 0.15",
    )
    p.add_argument(
        "--min-speech-ms", type=int, default=defaults.min_speech_ms, help="最短語音長度"
    )
    p.add_argument(
        "--min-silence-ms", type=int, default=defaults.min_silence_ms, help="切段所需最短靜音"
    )
    p.add_argument(
        "--speech-pad-ms", type=int, default=defaults.speech_pad_ms, help="語音段前後延伸"
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Silero VAD ONNX 模型")
    p.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op 執行緒數")
    p.add_argument("--stream", action="store_true", help="以串流方式逐 chunk 送入並印出事件")
    p.add_argument("--chunk-ms", type=int, default=10, help="串流模式的 chunk 長度")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = VadConfig(
            threshold=args.threshold,
            neg_threshold=args.neg_threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
        model = SileroVadModel(args.model, num_threads=args.threads)
        audio = load_wav(args.wav)
    except (ValueError, FileNotFoundError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1

    duration = len(audio) / SAMPLE_RATE
    print(f"檔案：{args.wav}（{duration:.2f} s）")
    print(f"設定：{config}")

    t0 = time.perf_counter()
    if args.stream:
        vad = StreamingVad(model, config)
        chunk = args.chunk_ms * SAMPLE_RATE // 1000
        print(f"\n串流模式，chunk = {args.chunk_ms} ms")
        print(f"{'事件':<6}{'位置 (s)':>10}{'發出於 (s)':>12}{'延遲 (ms)':>11}")
        for i in range(0, len(audio) + 1, chunk):
            events = vad.accept_waveform(audio[i : i + chunk]) if i < len(audio) else vad.flush()
            for ev in events:
                print(
                    f"{ev.kind:<6}{ev.time_s:>10.3f}{ev.detected_at / SAMPLE_RATE:>12.3f}"
                    f"{ev.latency_s * 1000:>11.0f}"
                )
    else:
        segments = detect_segments(audio, model, config)
        print(f"\n偵測到 {len(segments)} 段語音")
        print(f"{'#':>3}{'開始 (s)':>10}{'結束 (s)':>10}{'長度 (s)':>10}")
        for n, seg in enumerate(segments, 1):
            print(f"{n:>3}{seg.start_s:>10.3f}{seg.end_s:>10.3f}{seg.end_s - seg.start_s:>10.3f}")
    elapsed = time.perf_counter() - t0
    print(f"\n處理時間 {elapsed * 1000:.0f} ms，RTF {elapsed / duration:.4f}（不含模型載入）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
