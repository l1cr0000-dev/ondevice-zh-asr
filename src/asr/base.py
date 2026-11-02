"""串流 ASR 的抽象介面。各後端（sherpa-onnx、SenseVoice、自行量化版、Core ML……）都實作這兩個類別。

使用方式（一句話 = 一個 stream）：

    asr = load_asr("x-asr-480ms-int8")
    stream = asr.create_stream()
    for chunk in chunks:                 # 16 kHz float32 單聲道，長度任意
        stream.accept_waveform(chunk)    # 內部會把已足夠的音訊解碼完
        print(stream.get_partial_result())
    text = stream.finalize()             # 輸入結束，回傳最終結果

約定：
- `accept_waveform` 回傳時，已送入的音訊能解碼的部分都已解碼完，因此計時包含計算時間，
  `get_partial_result` 只是讀取目前結果、不做計算。
- `finalize` 只能呼叫一次；之後再呼叫 `accept_waveform` / `finalize` 會報錯，`get_partial_result`
  回傳最終結果。下一句請重新 `create_stream()`。
- 非串流模型（例如 SenseVoice）也可以實作：`accept_waveform` 只累積音訊、partial 回傳空字串或
  自行定期重新解碼，`finalize` 時才完整辨識。
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.audio import SAMPLE_RATE


class AsrStream(ABC):
    """一句話的辨識狀態。由 `StreamingAsr.create_stream()` 建立。"""

    def __init__(self) -> None:
        self._final: str | None = None

    @property
    def is_finalized(self) -> bool:
        return self._final is not None

    def accept_waveform(self, chunk: npt.NDArray[np.floating]) -> None:
        """送入一段 16 kHz、單聲道、範圍約 [-1, 1] 的 float 音訊。"""
        if self._final is not None:
            raise RuntimeError("stream 已 finalize，請重新 create_stream()")
        self._accept(np.asarray(chunk, dtype=np.float32).reshape(-1))

    def get_partial_result(self) -> str:
        """目前為止的辨識結果（finalize 後回傳最終結果）。"""
        if self._final is not None:
            return self._final
        return self._partial()

    def finalize(self) -> str:
        """標記輸入結束，解碼剩餘音訊並回傳最終結果。"""
        if self._final is not None:
            raise RuntimeError("stream 已 finalize")
        self._final = self._finalize()
        return self._final

    def get_token_timestamps(self) -> list[float] | None:
        """目前結果中每個 token 在此 stream 音訊中的位置（秒，以第一個送入的樣本為 0）。

        用來拆解首字延遲：模型「認為」第一個字在音訊的哪裡。後端不支援時回傳 None。
        """
        return None

    @abstractmethod
    def _accept(self, chunk: npt.NDArray[np.float32]) -> None: ...

    @abstractmethod
    def _partial(self) -> str: ...

    @abstractmethod
    def _finalize(self) -> str: ...


class StreamingAsr(ABC):
    """已載入的 ASR 模型。模型只載入一次，每句話各自 `create_stream()`。"""

    sample_rate: int = SAMPLE_RATE

    @property
    @abstractmethod
    def name(self) -> str:
        """模型與設定的識別字串，寫入評測報告用。"""

    @abstractmethod
    def create_stream(self) -> AsrStream: ...

    def model_files(self) -> list[Path]:
        """推論所需的全部模型檔（含 tokens / 詞表），評測用來計算模型大小。"""
        return []

    def compute_stats(self) -> dict[str, float] | None:
        """與機器負載無關的計算量統計（自上次 `reset_compute_stats` 起、所有 stream 合計）。

        例如非串流模型重新辨識的次數與音訊總長。後端不提供時回傳 None。
        """
        return None

    def reset_compute_stats(self) -> None:
        """清除 `compute_stats` 的累計（評測在暖機後呼叫）。"""
