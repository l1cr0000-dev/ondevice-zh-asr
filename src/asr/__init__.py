"""串流語音辨識。

介面見 `src.asr.base`：`StreamingAsr.create_stream()` → `AsrStream`，
`accept_waveform(chunk)` / `get_partial_result()` / `finalize()`。

以名稱載入已登記的模型：`load_asr("x-asr-480ms-int8", num_threads=1)`（不給名稱時用 `DEFAULT_MODEL`）。
新增後端時：實作 `StreamingAsr` / `AsrStream`，再到 `_REGISTRY` 登記一個名稱。
"""

from collections.abc import Callable

from src.asr.base import AsrStream, StreamingAsr
from src.asr.sherpa import (
    X_ASR_160MS,
    X_ASR_480MS,
    ZIPFORMER_ZH_14M,
    SherpaTransducerAsr,
    SherpaTransducerConfig,
    TransducerFiles,
)
from src.audio import SAMPLE_RATE

__all__ = [
    "DEFAULT_MODEL",
    "SAMPLE_RATE",
    "TRANSDUCER_SPECS",
    "AsrStream",
    "SherpaTransducerAsr",
    "SherpaTransducerConfig",
    "StreamingAsr",
    "TransducerFiles",
    "available_models",
    "load_asr",
]

DEFAULT_MODEL = "x-asr-480ms-int8"

# sherpa-onnx transducer 模型：名稱 → (模型檔, 句尾補零秒數)。也供 scripts/probe_memory.py 直接載入。
TRANSDUCER_SPECS: dict[str, tuple[TransducerFiles, float]] = {
    "zipformer-zh-14m": (ZIPFORMER_ZH_14M, 0.66),
    # chunk 48 frame（480 ms）+ 右側 context 13 frame。實測補 1.2 s 仍會漏句尾的字，
    # 1.4–3.0 s 結果相同；取 1.6 s 留餘量，finalize 計算量與 1.4 s 相同。
    "x-asr-480ms-int8": (X_ASR_480MS, 1.6),
    # chunk 16 frame（160 ms）+ 右側 context 13 frame。AISHELL 200 句 + 4 個官方測試音訊整句解碼：
    # 補 1.6 s 仍有 3/204 句漏句尾的字，2.0–4.0 s 結果完全相同；取 2.0 s（finalize 約 121 ms）。
    "x-asr-160ms-int8": (X_ASR_160MS, 2.0),
}


def _transducer(name: str) -> Callable[[int, float], StreamingAsr]:
    files, tail_padding_s = TRANSDUCER_SPECS[name]
    return lambda threads, blank_penalty: SherpaTransducerAsr(
        files,
        SherpaTransducerConfig(
            num_threads=threads, tail_padding_s=tail_padding_s, blank_penalty=blank_penalty
        ),
    )


_REGISTRY: dict[str, Callable[[int, float], StreamingAsr]] = {
    name: _transducer(name) for name in TRANSDUCER_SPECS
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def load_asr(
    name: str = DEFAULT_MODEL, num_threads: int = 1, blank_penalty: float = 0.0
) -> StreamingAsr:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"未知模型 {name!r}，可用：{', '.join(available_models())}") from None
    return factory(num_threads, blank_penalty)
