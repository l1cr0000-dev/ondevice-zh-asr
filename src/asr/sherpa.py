"""sherpa-onnx 串流 transducer 後端（Zipformer 等）。"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import sherpa_onnx  # type: ignore[import-untyped]

from src.asr.base import AsrStream, StreamingAsr
from src.audio import SAMPLE_RATE
from src.paths import MODELS_DIR


@dataclass(frozen=True)
class TransducerFiles:
    """模型目錄（相對 models/）與其中的檔名。"""

    model_dir: str
    encoder: str
    decoder: str
    joiner: str
    tokens: str = "tokens.txt"

    def paths(self, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
        root = models_dir / self.model_dir
        return {
            "encoder": root / self.encoder,
            "decoder": root / self.decoder,
            "joiner": root / self.joiner,
            "tokens": root / self.tokens,
        }


@dataclass(frozen=True)
class SherpaTransducerConfig:
    num_threads: int = 1
    decoding_method: str = "greedy_search"
    # 輸入結束時補的靜音長度。encoder 以固定 chunk 處理且需要右側 context，不補會漏掉句尾的字。
    # 這段補零是演算法的一部分，會計入尾字延遲的計算量。所需長度隨模型 chunk 大小而異，
    # 各模型的值在 `src.asr._REGISTRY` 設定；0.66 s 是 sherpa-onnx 官方範例的值。
    tail_padding_s: float = 0.66


ZIPFORMER_ZH_14M = TransducerFiles(
    model_dir="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
    encoder="encoder-epoch-99-avg-1.onnx",
    decoder="decoder-epoch-99-avg-1.onnx",
    joiner="joiner-epoch-99-avg-1.onnx",
)

X_ASR_160MS = TransducerFiles(
    model_dir="sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05",
    encoder="encoder.int8.onnx",
    decoder="decoder.onnx",  # 官方 INT8 版的 decoder 仍為 FP32
    joiner="joiner.int8.onnx",
)

X_ASR_480MS = TransducerFiles(
    model_dir="sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05",
    encoder="encoder.int8.onnx",
    decoder="decoder.onnx",  # 官方 INT8 版的 decoder 仍為 FP32
    joiner="joiner.int8.onnx",
)


class SherpaTransducerStream(AsrStream):
    def __init__(self, recognizer: "sherpa_onnx.OnlineRecognizer", tail_padding_s: float) -> None:
        super().__init__()
        self._recognizer = recognizer
        self._stream = recognizer.create_stream()
        self._tail_padding = np.zeros(int(tail_padding_s * SAMPLE_RATE), dtype=np.float32)

    def _decode_ready(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        self._stream.accept_waveform(SAMPLE_RATE, chunk)
        self._decode_ready()

    def _partial(self) -> str:
        return str(self._recognizer.get_result(self._stream))

    def get_token_timestamps(self) -> list[float]:
        # sherpa-onnx 以 encoder 輸出 frame 為單位（X-ASR / zh-14M 為 40 ms），
        # 是 token 被輸出的那個 frame，不一定是該字的聲學起點
        result = json.loads(self._recognizer.get_result_as_json_string(self._stream))
        return [float(t) for t in result["timestamps"]]

    def _finalize(self) -> str:
        if len(self._tail_padding):
            self._stream.accept_waveform(SAMPLE_RATE, self._tail_padding)
        self._stream.input_finished()
        self._decode_ready()
        return str(self._recognizer.get_result(self._stream))


class SherpaTransducerAsr(StreamingAsr):
    def __init__(
        self,
        files: TransducerFiles,
        config: SherpaTransducerConfig | None = None,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.files = files
        self.config = config or SherpaTransducerConfig()
        paths = self._paths = self.files.paths(models_dir)
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "找不到模型檔，請先執行 bash scripts/download_models.sh：\n  "
                + "\n  ".join(missing)
            )
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=self.config.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method=self.config.decoding_method,
            enable_endpoint_detection=False,  # 斷句交給 VAD
            provider="cpu",
        )

    @property
    def name(self) -> str:
        return (
            f"sherpa-onnx/{self.files.model_dir}/{self.files.encoder}"
            f" threads={self.config.num_threads} {self.config.decoding_method}"
        )

    def model_files(self) -> list[Path]:
        return list(self._paths.values())

    def create_stream(self) -> SherpaTransducerStream:
        return SherpaTransducerStream(self._recognizer, self.config.tail_padding_s)
