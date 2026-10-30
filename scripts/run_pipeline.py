"""VAD + 串流 ASR pipeline：輸入 wav，輸出每句結果與時間資訊（JSON）。

用法：
    uv run scripts/run_pipeline.py                          # 用 tests/fixtures 的兩句測試音訊
    uv run scripts/run_pipeline.py input.wav -o out.json
    uv run scripts/run_pipeline.py input.wav --min-silence-ms 400 --partials

時間欄位以秒為單位、以音訊開頭為 0。`*_emitted_s` / `first_token_s` / `final_s` 是「模擬即時」的時刻：
假設音訊依實際時長節奏、每 --chunk-ms 送入一次，再加上實際量測的計算時間（不真的 sleep）。
這裡的數字只是單次量測，正式指標請用 eval/ 的腳本。
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# 讓 `uv run scripts/run_pipeline.py` 能直接 import src（專案本身沒有安裝成套件）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import SAMPLE_RATE, load_asr
from src.cli import add_pipeline_args, pipeline_config_from_args
from src.pipeline import Utterance, transcribe
from src.vad import VAD_MODEL_NAME, SileroVadModel, load_wav

DEFAULT_WAV = REPO_ROOT / "tests" / "fixtures" / "vad_two_utterances.wav"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VAD + 串流 ASR：輸入 wav，模擬即時逐 chunk 送入，輸出每句結果與時間戳（JSON）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("wav", nargs="?", type=Path, default=DEFAULT_WAV, help="16 kHz wav 檔")
    p.add_argument("-o", "--output", type=Path, help="輸出 JSON 路徑；不給則印到 stdout")
    add_pipeline_args(p)
    p.add_argument("--partials", action="store_true", help="輸出每句的 partial 變化歷程")
    return p.parse_args()


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def utterance_json(utt: Utterance, with_partials: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "index": utt.index,
        "text": utt.text,
        "speech_start_s": _round(utt.speech_start_s),
        "first_token_s": _round(utt.first_token_s),
        "speech_end_s": _round(utt.speech_end_s),
        "final_s": _round(utt.final_s),
        "audio_start_s": _round(utt.audio_start_s),
        "start_emitted_s": _round(utt.start_emitted_s),
        "end_emitted_s": _round(utt.end_emitted_s),
        "first_token_latency_ms": _round(
            None if utt.first_token_latency_s is None else utt.first_token_latency_s * 1000
        ),
        "tail_latency_ms": _round(
            None if utt.tail_latency_s is None else utt.tail_latency_s * 1000
        ),
    }
    if with_partials:
        out["partials"] = [{"emitted_s": _round(p.emitted_s), "text": p.text} for p in utt.partials]
    return out


def main() -> int:
    args = parse_args()
    try:
        config = pipeline_config_from_args(args)
        audio = load_wav(args.wav)
        vad_model = SileroVadModel(num_threads=args.threads)
        asr = load_asr(args.model, num_threads=args.threads)
    except (ValueError, FileNotFoundError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1

    duration = len(audio) / SAMPLE_RATE
    t0 = time.perf_counter()
    utterances = transcribe(audio, vad_model, asr, config, chunk_ms=args.chunk_ms)
    compute_s = time.perf_counter() - t0

    result = {
        "wav": str(args.wav),
        "duration_s": round(duration, 3),
        "settings": {
            "asr": asr.name,
            "vad": VAD_MODEL_NAME,
            "vad_config": asdict(config.vad),
            "pre_roll_ms": config.pre_roll_ms,
            "chunk_ms": args.chunk_ms,
            "threads": args.threads,
            "timing": "simulated-realtime",
        },
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "compute_s": round(compute_s, 4),
        "rtf": round(compute_s / duration, 4) if duration > 0 else None,
        "utterances": [utterance_json(u, args.partials) for u in utterances],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"已寫入 {args.output}（{len(utterances)} 句）", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
