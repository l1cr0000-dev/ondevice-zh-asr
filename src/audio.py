"""音訊共用設定與讀檔。VAD、ASR、pipeline 與腳本都以這裡的取樣率為準。"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

SAMPLE_RATE = 16000


def load_wav(path: str | Path) -> npt.NDArray[np.float32]:
    """讀取 16 kHz wav 為 float32 單聲道；多聲道取平均。不做重取樣，取樣率不符直接報錯。

    檔案不存在時丟 `FileNotFoundError`，無法解碼或取樣率不符時丟 `ValueError`。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音訊檔：{path}")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.SoundFileError as e:
        raise ValueError(f"無法讀取音訊檔 {path}：{e}") from e
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} 取樣率為 {sr} Hz，需要 {SAMPLE_RATE} Hz，請先重取樣")
    audio: npt.NDArray[np.float32] = data.mean(axis=1)
    return audio
