"""src.audio：讀檔的錯誤類型（腳本只接 FileNotFoundError / ValueError，其他例外會變成 traceback）。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from src.audio import SAMPLE_RATE, load_wav


def test_load_wav_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="找不到音訊檔"):
        load_wav(tmp_path / "nope.wav")


def test_load_wav_not_audio(tmp_path: Path) -> None:
    path = tmp_path / "text.wav"
    path.write_text("not a wav", encoding="utf-8")
    with pytest.raises(ValueError, match="無法讀取音訊檔"):
        load_wav(path)


def test_load_wav_averages_channels(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    stereo = np.stack([np.full(160, 0.5), np.full(160, -0.25)], axis=1).astype(np.float32)
    sf.write(str(path), stereo, SAMPLE_RATE, subtype="FLOAT")
    audio = load_wav(path)
    assert audio.dtype == np.float32
    assert audio.shape == (160,)
    np.testing.assert_allclose(audio, 0.125)
