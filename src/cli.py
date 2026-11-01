"""命令列腳本共用的參數定義與終端顯示工具（scripts/、eval/run_eval.py）。"""

import argparse
import unicodedata
from typing import Any

from src.asr import DEFAULT_MODEL, available_models
from src.pipeline import PipelineConfig
from src.vad import VadConfig


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"必須為正整數，收到 {value}")
    return n


def non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"不可為負，收到 {value}")
    return n


def add_pipeline_args(p: argparse.ArgumentParser, sweep: bool = False) -> None:
    """加入 pipeline（VAD + ASR）的共用參數，預設值取自 `PipelineConfig()`。

    `sweep=True` 時 --model / --threads / --chunk-ms / --pre-roll-ms / --min-silence-ms 可給多個值，
    值與預設都是 list（評測腳本取笛卡兒積）。
    """
    defaults = PipelineConfig()
    vad = defaults.vad

    def multi(default: object) -> dict[str, Any]:
        return {"nargs": "+", "default": [default]} if sweep else {"default": default}

    p.add_argument("--model", choices=available_models(), help="ASR 模型", **multi(DEFAULT_MODEL))
    p.add_argument("--threads", type=positive_int, help="VAD / ASR 推論執行緒數", **multi(1))
    p.add_argument("--chunk-ms", type=positive_int, help="每次送入的音訊長度", **multi(10))
    p.add_argument(
        "--pre-roll-ms",
        type=non_negative_int,
        help="送入 ASR 時比 VAD 起點再往前多取的音訊長度",
        **multi(defaults.pre_roll_ms),
    )
    p.add_argument("--threshold", type=float, default=vad.threshold, help="VAD 語音閾值")
    p.add_argument(
        "--min-speech-ms", type=non_negative_int, default=vad.min_speech_ms, help="VAD 最短語音"
    )
    p.add_argument(
        "--min-silence-ms",
        type=non_negative_int,
        help="VAD 切段靜音",
        **multi(vad.min_silence_ms),
    )
    p.add_argument(
        "--speech-pad-ms", type=non_negative_int, default=vad.speech_pad_ms, help="VAD 前後延伸"
    )
    if not sweep:  # 評測腳本自己定義可掃描的版本
        p.add_argument(
            "--pause-ms",
            type=non_negative_int,
            default=0,
            help="靜音滿此值先輸出暫定結果、滿 --min-silence-ms 才定稿（0 = 關閉；需小於 min_silence）",
        )
        p.add_argument(
            "--provisional",
            choices=["redecode", "finalize"],
            default=defaults.provisional,
            help="暫定結果的做法：redecode 停頓時重新解碼整句；finalize 停頓時 finalize、恢復說話時才重新解碼",
        )


def pipeline_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """由 `add_pipeline_args(sweep=False)` 解析出的參數建立設定；數值不合法時丟 `ValueError`。"""
    return PipelineConfig(
        vad=VadConfig(
            threshold=args.threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            pause_ms=args.pause_ms or None,
        ),
        pre_roll_ms=args.pre_roll_ms,
        provisional=args.provisional,
    )


def char_width(ch: str) -> int:
    """終端顯示寬度：全形 / 寬字元（中文）為 2，其餘為 1。"""
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def display_width(text: str) -> int:
    return sum(char_width(c) for c in text)
