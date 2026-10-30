"""串流語音辨識。

介面見 `src.asr.base`：`StreamingAsr.create_stream()` → `AsrStream`，
`accept_waveform(chunk)` / `get_partial_result()` / `finalize()`。

以名稱載入已登記的模型：`load_asr("x-asr-480ms-int8", num_threads=1)`（不給名稱時用 `DEFAULT_MODEL`）。
新增後端時：實作 `StreamingAsr` / `AsrStream`，再到 `_REGISTRY` 登記一個名稱。
"""

from collections.abc import Callable

from src.asr.base import SAMPLE_RATE, AsrStream, StreamingAsr
from src.asr.sherpa import (
    X_ASR_480MS,
    ZIPFORMER_ZH_14M,
    SherpaTransducerAsr,
    SherpaTransducerConfig,
    TransducerFiles,
)

__all__ = [
    "DEFAULT_MODEL",
    "SAMPLE_RATE",
    "AsrStream",
    "SherpaTransducerAsr",
    "SherpaTransducerConfig",
    "StreamingAsr",
    "TransducerFiles",
    "available_models",
    "load_asr",
]

DEFAULT_MODEL = "x-asr-480ms-int8"

_REGISTRY: dict[str, Callable[[int], StreamingAsr]] = {
    "zipformer-zh-14m": lambda threads: SherpaTransducerAsr(
        ZIPFORMER_ZH_14M, SherpaTransducerConfig(num_threads=threads)
    ),
    # chunk 48 frame（480 ms）+ 右側 context 13 frame。實測補 1.2 s 仍會漏句尾的字，
    # 1.4–3.0 s 結果相同；取 1.6 s 留餘量，finalize 計算量與 1.4 s 相同。
    "x-asr-480ms-int8": lambda threads: SherpaTransducerAsr(
        X_ASR_480MS, SherpaTransducerConfig(num_threads=threads, tail_padding_s=1.6)
    ),
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def load_asr(name: str = DEFAULT_MODEL, num_threads: int = 1) -> StreamingAsr:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"未知模型 {name!r}，可用：{', '.join(available_models())}") from None
    return factory(num_threads)
