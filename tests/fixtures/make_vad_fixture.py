"""產生 VAD 測試用的 wav：用 macOS `say` 合成兩句中文，修掉頭尾靜音後放到已知位置。

輸出 tests/fixtures/vad_two_utterances.wav（16 kHz、單聲道、int16），以及同名 .json 記錄
每句語音在時間軸上的真實起止（秒）。檔案已提交，只有改內容時才需要重跑：

    uv run tests/fixtures/make_vad_fixture.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

SAMPLE_RATE = 16000
VOICE = "Tingting"
UTTERANCES = ["今天天氣很好", "我們去公園散步吧"]
GAP_S = 1.0  # 開頭、句間、結尾的靜音長度
OUT_DIR = Path(__file__).resolve().parent
NAME = "vad_two_utterances"


def synthesize(text: str) -> npt.NDArray[np.float32]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "say.wav"
        subprocess.run(
            ["say", "-v", VOICE, "-o", str(path), f"--data-format=LEI16@{SAMPLE_RATE}", text],
            check=True,
        )
        audio, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE
    # 以 -40 dBFS 為界修掉 TTS 自帶的頭尾靜音，讓真實起止有明確定義
    voiced = np.flatnonzero(np.abs(audio) > 10 ** (-40 / 20))
    trimmed: npt.NDArray[np.float32] = audio[voiced[0] : voiced[-1] + 1]
    return trimmed


def main() -> None:
    gap = np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32)
    parts: list[npt.NDArray[np.float32]] = [gap]
    truth: list[dict[str, float | str]] = []
    pos = len(gap)
    for text in UTTERANCES:
        clip = synthesize(text)
        truth.append(
            {"text": text, "start": pos / SAMPLE_RATE, "end": (pos + len(clip)) / SAMPLE_RATE}
        )
        parts += [clip, gap]
        pos += len(clip) + len(gap)
    audio = np.concatenate(parts)
    sf.write(str(OUT_DIR / f"{NAME}.wav"), audio, SAMPLE_RATE, subtype="PCM_16")
    meta = {"voice": VOICE, "sample_rate": SAMPLE_RATE, "segments": truth}
    (OUT_DIR / f"{NAME}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
