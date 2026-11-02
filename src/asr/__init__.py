"""串流語音辨識。

介面見 `src.asr.base`：`StreamingAsr.create_stream()` → `AsrStream`，
`accept_waveform(chunk)` / `get_partial_result()` / `finalize()`。

以名稱載入已登記的模型：`load_asr("x-asr-480ms-int8", num_threads=1)`（不給名稱時用 `DEFAULT_MODEL`）。
解碼參數（blank penalty、beam、熱詞）以 `DecodeOptions` 傳入；不適用於該模型的參數會報錯。
新增後端時：實作 `StreamingAsr` / `AsrStream`，再到 `_REGISTRY` 登記一個名稱。
"""

from collections.abc import Callable
from dataclasses import dataclass

from src.asr.base import AsrStream, StreamingAsr
from src.asr.sense_voice import (
    PartialSchedule,
    SenseVoiceAsr,
    SenseVoiceConfig,
    SenseVoiceOrtAsr,
)
from src.asr.sherpa import (
    X_ASR_160MS,
    X_ASR_480MS,
    ZIPFORMER_ZH_14M,
    SherpaTransducerAsr,
    SherpaTransducerConfig,
    TransducerFiles,
)
from src.asr.zipformer_ort import (
    CoreMLComputeUnits,
    CoreMLOptions,
    OrtTransducerConfig,
    ZipformerOrtAsr,
)
from src.audio import SAMPLE_RATE

__all__ = [
    "DEFAULT_MODEL",
    "SAMPLE_RATE",
    "TRANSDUCER_SPECS",
    "AsrStream",
    "CoreMLOptions",
    "DecodeOptions",
    "OrtTransducerConfig",
    "PartialSchedule",
    "SenseVoiceAsr",
    "SenseVoiceConfig",
    "SenseVoiceOrtAsr",
    "SherpaTransducerAsr",
    "SherpaTransducerConfig",
    "StreamingAsr",
    "TransducerFiles",
    "ZipformerOrtAsr",
    "available_models",
    "default_threads",
    "load_asr",
]

# 2026-11-02 使用者決定改為 SenseVoice 偽串流（docs/progress.md「決定：預設改為 SenseVoice 偽串流」）：
# 首字約 0.45 s（X-ASR 約 1 s）、CER 持平或略好、支援熱詞；代價是 partial 跳動較多、結果慢數十 ms、
# 記憶體較大、授權依據較弱。X-ASR 保留為 `x-asr-480ms-int8-ort`。
# 2026-11-02 再改為語言固定 zh 的版本：第一個 partial 的英文碎片 15–32% → 0–1.5%，CER 不變。
DEFAULT_MODEL = "sensevoice-small-int8-ort-p480-960-max3-zh"

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


@dataclass(frozen=True)
class DecodeOptions:
    """模型共用的解碼參數。預設值 = 各模型原本的解碼方式。"""

    blank_penalty: float = 0.0  # 只適用於 transducer
    beam: int = 1  # 只適用於 SenseVoice onnxruntime 版；1 = greedy
    hotwords: tuple[str, ...] = ()  # 同上
    hotword_score: float = 1.5


_Factory = Callable[[int, DecodeOptions], StreamingAsr]


def _no_ctc_options(opts: DecodeOptions) -> None:
    if opts.beam != 1 or opts.hotwords:
        raise ValueError("beam / 熱詞目前只支援 sensevoice-small-int8-ort*")


def _transducer(name: str) -> _Factory:
    files, tail_padding_s = TRANSDUCER_SPECS[name]

    def factory(threads: int, opts: DecodeOptions) -> StreamingAsr:
        _no_ctc_options(opts)
        return SherpaTransducerAsr(
            files,
            SherpaTransducerConfig(
                num_threads=threads, tail_padding_s=tail_padding_s, blank_penalty=opts.blank_penalty
            ),
        )

    return factory


def _transducer_ort(name: str, coreml: CoreMLOptions | None = None) -> _Factory:
    files, tail_padding_s = TRANSDUCER_SPECS[name]

    def factory(threads: int, opts: DecodeOptions) -> StreamingAsr:
        _no_ctc_options(opts)
        return ZipformerOrtAsr(
            files,
            OrtTransducerConfig(
                num_threads=threads, tail_padding_s=tail_padding_s, blank_penalty=opts.blank_penalty
            ),
            coreml=coreml,
        )

    return factory


# 第二階段前哨實驗：X-ASR 480 ms 的 encoder 改用 onnxruntime CoreML EP（MLProgram），
# 名稱後綴 → compute units。比 CPU 慢約 3 倍（docs/progress.md 2026-11-02），只供量測。
_COREML_UNITS: dict[str, CoreMLComputeUnits] = {
    "cpu": "CPUOnly",
    "gpu": "CPUAndGPU",
    "ane": "CPUAndNeuralEngine",
    "all": "ALL",
}


def _sense_voice(
    partial_every_s: float | None,
    ort: bool = False,
    first_s: float | None = None,
    max_s: float | None = None,
    language: str = "auto",
) -> _Factory:
    def factory(threads: int, opts: DecodeOptions) -> StreamingAsr:
        if opts.blank_penalty:
            raise ValueError("blank_penalty 只適用於 transducer 模型")
        if not ort:
            _no_ctc_options(opts)
        config = SenseVoiceConfig(
            num_threads=threads,
            language=language,
            partial_every_s=partial_every_s,
            partial_first_s=first_s,
            partial_max_s=max_s,
            beam=opts.beam,
            hotwords=opts.hotwords,
            hotword_score=opts.hotword_score,
        )
        return SenseVoiceOrtAsr(config) if ort else SenseVoiceAsr(config)

    return factory


_REGISTRY: dict[str, _Factory] = {
    **{name: _transducer(name) for name in TRANSDUCER_SPECS},
    # 同一個 X-ASR 模型直接用 onnxruntime（專案依賴的版本）推論，結果與 sherpa-onnx 版逐句相同
    "x-asr-480ms-int8-ort": _transducer_ort("x-asr-480ms-int8"),
    # 160 ms 版同樣做法；只在測試音訊上與 sherpa-onnx 版比對過，未跑冒煙集（docs/progress.md 2026-11-01）
    "x-asr-160ms-int8-ort": _transducer_ort("x-asr-160ms-int8"),
    **{
        f"x-asr-480ms-int8-ort-coreml-{suffix}": _transducer_ort(
            "x-asr-480ms-int8", CoreMLOptions(units)
        )
        for suffix, units in _COREML_UNITS.items()
    },
    # 非串流：說話中沒有 partial，首字要等到 finalize（VAD 結束或暫定結果）
    "sensevoice-small-int8": _sense_voice(None),
    # 同上，但每累積 0.48 s 音訊就整段重新辨識一次當 partial（與 X-ASR 480 ms 的更新頻率相同）
    "sensevoice-small-int8-p480": _sense_voice(0.48),
    # 同一個模型直接用 onnxruntime（專案依賴的版本）推論，避開 sherpa-onnx 內附 ORT 1.28 的記憶體開銷
    "sensevoice-small-int8-ort": _sense_voice(None, ort=True),
    "sensevoice-small-int8-ort-p480": _sense_voice(0.48, ort=True),
    # 偽串流的降計算量選項（docs/progress.md 2026-11-01 偽串流 vs 真串流）：
    # 間隔加倍
    "sensevoice-small-int8-ort-p960": _sense_voice(0.96, ort=True),
    # 第一次 partial 仍在 0.48 s（首字不變），之後每 0.96 s 才更新
    "sensevoice-small-int8-ort-p480-960": _sense_voice(0.96, ort=True, first_s=0.48),
    # 每 0.48 s，但累積超過 3 s 就不再重新辨識（車艙指令多在 3 s 內；長句段只等暫定結果 / 定稿）
    "sensevoice-small-int8-ort-p480-max3": _sense_voice(0.48, ort=True, max_s=3.0),
    # 兩者合併：首字同 p480，短句計算量同 p480-960，長句段的最壞情況有上限
    "sensevoice-small-int8-ort-p480-960-max3": _sense_voice(
        0.96, ort=True, first_s=0.48, max_s=3.0
    ),
    # 語言固定為中文（-zh；其餘同上），並掃第一次 partial 的時機 0.32 / 0.40 / 0.48 s
    # （docs/progress.md 2026-11-02「SenseVoice 語言固定 zh 與第一次 partial」）。
    # language=zh 時模型不再自行判斷語言，短音訊的第一個 partial 不會猜成英文碎片
    **{
        f"sensevoice-small-int8-ort-p{ms}-960-max3-zh": _sense_voice(
            0.96, ort=True, first_s=ms / 1000, max_s=3.0, language="zh"
        )
        for ms in (320, 400, 480)
    },
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def default_threads(name: str) -> int:
    """模型的預設推論執行緒數。

    SenseVoice onnxruntime 版每次整段推論，2 執行緒結果逐句不變、說完 → 完整結果快 30–50 ms
    （docs/progress.md 2026-11-02「SenseVoice 偽串流的執行緒數」）；其餘維持 1
    （X-ASR 多執行緒沒有收益且會改變結果，B8）。
    """
    return 2 if name.startswith("sensevoice-small-int8-ort") else 1


def load_asr(
    name: str = DEFAULT_MODEL,
    num_threads: int | None = None,
    blank_penalty: float = 0.0,
    decode: DecodeOptions | None = None,
) -> StreamingAsr:
    """`num_threads` 不給時用 `default_threads(name)`。

    `blank_penalty` 是 `DecodeOptions.blank_penalty` 的簡寫（給了 `decode` 時以 `decode` 為準）。
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"未知模型 {name!r}，可用：{', '.join(available_models())}") from None
    threads = num_threads or default_threads(name)
    return factory(threads, decode or DecodeOptions(blank_penalty=blank_penalty))
