"""Silero VAD 的 ONNX Runtime 後端：每次輸入一個固定長度的 frame，回傳語音機率。

模型 I/O（v6.2.3，16 kHz）：
- input  float32 [1, 64 + 512]：前 64 個樣本是上一個 frame 的尾端（context），後 512 是本 frame
- state  float32 [2, 1, 128]：RNN 狀態，跨 frame 傳遞
- sr     int64 []：取樣率
- output float32 [1, 1]：本 frame 的語音機率；stateN 為下一步的 state

context 的處理與上游 `silero_vad.utils_vad.OnnxWrapper` 一致，少了它機率會明顯偏移。
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32 ms
CONTEXT_SIZE = 64

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "silero-vad-v6.2.3" / "silero_vad.onnx"


class SileroVadModel:
    """有狀態的 frame 分類器。換成 Core ML 後端時，實作同樣的 `frame_size` / `__call__` / `reset`。"""

    frame_size = FRAME_SIZE

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, num_threads: int = 1) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"找不到 VAD 模型：{model_path}。請先執行 bash scripts/download_models.sh"
            )
        opts: Any = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    def __call__(self, frame: npt.NDArray[np.float32]) -> float:
        if frame.shape != (FRAME_SIZE,):
            raise ValueError(f"frame 長度必須是 {FRAME_SIZE}，收到 {frame.shape}")
        x = np.concatenate([self._context, frame])[np.newaxis, :]
        prob, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = frame[-CONTEXT_SIZE:].copy()
        return float(prob[0, 0])
