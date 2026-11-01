"""sherpa-onnx 非串流 SenseVoice 後端。

非串流模型一次看完整段音訊才辨識，沒有真正的 partial。實作成 `AsrStream` 的方式：
- `accept_waveform` 只累積音訊；`finalize` 時把累積的音訊整段辨識一次。
- 選用 `partial_every_s`：每多累積這麼長的音訊，就把目前為止的全部音訊整段辨識一次當 partial
  （假的串流）。計算量隨句長成長，計入 `accept_waveform` 的耗時，因此會反映在模擬即時的延遲上。
  None = 不產生 partial（首字要等到 finalize）。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import sherpa_onnx  # type: ignore[import-untyped]

from src.asr.base import AsrStream, StreamingAsr
from src.audio import SAMPLE_RATE
from src.paths import MODELS_DIR

SENSE_VOICE_DIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"


@dataclass(frozen=True)
class SenseVoiceConfig:
    num_threads: int = 1
    # "auto" 讓模型自己判斷語言（與 X-ASR 一樣不預設語言）；也可指定 zh / en / yue / ja / ko
    language: str = "auto"
    # 反正規化（數字轉阿拉伯數字、加標點）。AISHELL 標註是中文數字，評測的正規化不做 ITN，故關閉
    use_itn: bool = False
    partial_every_s: float | None = None


class SenseVoiceStream(AsrStream):
    def __init__(self, recognizer: "sherpa_onnx.OfflineRecognizer", partial_every_s: float | None):
        super().__init__()
        self._recognizer = recognizer
        self._chunks: list[npt.NDArray[np.float32]] = []
        self._n = 0  # 已累積的樣本數
        self._every = None if partial_every_s is None else int(partial_every_s * SAMPLE_RATE)
        self._next_partial = self._every
        self._partial_text = ""

    def _decode(self) -> str:
        if not self._n:
            return ""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.concatenate(self._chunks))
        self._recognizer.decode_stream(stream)
        return str(stream.result.text)

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        self._chunks.append(chunk.copy())
        self._n += len(chunk)
        if self._next_partial is not None and self._every and self._n >= self._next_partial:
            self._partial_text = self._decode()
            self._next_partial = (self._n // self._every + 1) * self._every

    def _partial(self) -> str:
        return self._partial_text

    def _finalize(self) -> str:
        return self._decode()


class SenseVoiceAsr(StreamingAsr):
    def __init__(self, config: SenseVoiceConfig | None = None, models_dir: Path = MODELS_DIR):
        self.config = config or SenseVoiceConfig()
        root = models_dir / SENSE_VOICE_DIR
        self._paths = {"model": root / "model.int8.onnx", "tokens": root / "tokens.txt"}
        missing = [str(p) for p in self._paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "找不到模型檔，請先執行 bash scripts/download_models.sh：\n  "
                + "\n  ".join(missing)
            )
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self._paths["model"]),
            tokens=str(self._paths["tokens"]),
            num_threads=self.config.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            language=self.config.language,
            use_itn=self.config.use_itn,
            provider="cpu",
        )

    @property
    def name(self) -> str:
        every = self.config.partial_every_s
        return (
            f"sherpa-onnx/{SENSE_VOICE_DIR}/model.int8.onnx threads={self.config.num_threads}"
            f" language={self.config.language} itn={self.config.use_itn}"
            + (f" partial_every={every:g}s" if every else "")
        )

    def model_files(self) -> list[Path]:
        return list(self._paths.values())

    def create_stream(self) -> SenseVoiceStream:
        return SenseVoiceStream(self._recognizer, self.config.partial_every_s)
