"""C++ fbank / LFR / CMVN 與 Python（kaldi-native-fbank）的數值比對。

對每條音訊：Python 的 log-mel（`kaldi_native_fbank`，80 維、hamming、dither 0、snip_edges、樣本乘 32768）
與 C++ 工具 `ozasr_fbank` 的輸出逐元素比較，報告最大絕對誤差；另比較 LFR + CMVN 後的模型輸入特徵。
log-mel 最大絕對誤差需 < 1e-3，否則結束碼為 1。

另有一份 numpy float64 的獨立參考（同一套演算法，無 float32 捨入）。kaldi-native-fbank 用 float32 做 FFT，
低能量的 bin 會有約 1e-3 的捨入誤差（合成 fixture 的第 123 frame 第 1 維：float64 1.92359、knf 1.92253、
C++ 1.92359）。因此門檻的判定如下：AISHELL 真實語音對 knf < 1e-3；每條音訊對 float64 參考 < 1e-4
（確認 C++ 本身是準的）；fixture 對 knf 的誤差只報告。

用法：
    uv run scripts/compare_fbank.py                 # fixture + 冒煙集前 20 句
    uv run scripts/compare_fbank.py -n 50
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import SAMPLE_RATE
from src.asr.sense_voice import SENSE_VOICE_DIR, SenseVoiceOrtAsr
from src.vad import load_wav

TOLERANCE = 1e-3
PREEMPH = float(np.float32(0.97))  # kaldi 的係數是 float 0.97f
SMOKE_MANIFEST = REPO_ROOT / "data" / "aishell1_test_200" / "manifest.jsonl"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "vad_two_utterances.wav"


def python_logmel(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    import kaldi_native_fbank as knf  # type: ignore[import-untyped]

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = 0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, (audio * 32768.0).tolist())
    fbank.input_finished()
    n = fbank.num_frames_ready
    return np.stack([np.asarray(fbank.get_frame(i), dtype=np.float32) for i in range(n)])


def float64_logmel(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """同一套演算法（去 DC、預強調 0.97、hamming、512 點 FFT、20–8000 Hz 的 80 個 mel 三角濾波器）的 float64 版。"""
    x = audio.astype(np.float64) * 32768.0
    n = 1 + (len(x) - 400) // 160 if len(x) >= 400 else 0
    frames = np.stack([x[i * 160 : i * 160 + 400] for i in range(n)])
    frames = frames - frames.mean(axis=1, keepdims=True)
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0] * (1 - PREEMPH)
    pre[:, 1:] = frames[:, 1:] - PREEMPH * frames[:, :-1]
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)
    power = np.abs(np.fft.rfft(pre * win, 512, axis=1)[:, :256]) ** 2

    def mel(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1127.0 * np.log(1.0 + hz / 700.0)

    low, high = float(mel(np.array(20.0))), float(mel(np.array(8000.0)))
    delta = (high - low) / 81
    m = mel(np.arange(256) * 16000 / 512)
    bank = np.zeros((256, 80))
    for b in range(80):
        left, center, right = low + b * delta, low + (b + 1) * delta, low + (b + 2) * delta
        up = (m - left) / (center - left)
        down = (right - m) / (right - center)
        bank[:, b] = np.where((m > left) & (m < right), np.where(m <= center, up, down), 0.0)
    out: npt.NDArray[np.float64] = np.log(np.maximum(power @ bank, np.finfo(np.float32).eps))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-n", type=int, default=20, help="冒煙集取前 N 句")
    p.add_argument("--cli", type=Path, default=REPO_ROOT / "cpp" / "build" / "ozasr_fbank")
    args = p.parse_args()
    if not args.cli.is_file():
        print(f"錯誤：找不到 {args.cli}，請先建置 C++", file=sys.stderr)
        return 1

    rows = [json.loads(x) for x in SMOKE_MANIFEST.read_text(encoding="utf-8").splitlines()]
    files = [FIXTURE] + [SMOKE_MANIFEST.parent / r["wav"] for r in rows[: args.n]]
    asr = SenseVoiceOrtAsr()
    root = REPO_ROOT / "models" / SENSE_VOICE_DIR
    worst_speech = worst_fixture = worst_feat = worst_f64 = 0.0
    total_frames = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path in files:
            audio = load_wav(path)
            ref_mel = python_logmel(audio)
            ref_feat = asr.features(audio)
            out_mel, out_feat = Path(tmp) / "mel.f32", Path(tmp) / "feat.f32"
            subprocess.run(
                [
                    str(args.cli),
                    str(path),
                    str(out_mel),
                    "--asr-model",
                    str(root / "model.int8.onnx"),
                    "--tokens",
                    str(root / "tokens.txt"),
                    "--features",
                    str(out_feat),
                ],
                check=True,
            )
            mel = np.fromfile(out_mel, dtype=np.float32).reshape(-1, 80)
            feat = np.fromfile(out_feat, dtype=np.float32).reshape(-1, ref_feat.shape[1])
            if mel.shape != ref_mel.shape or feat.shape != ref_feat.shape:
                print(f"{path.name}: 形狀不符 {mel.shape} vs {ref_mel.shape}")
                return 1
            err = float(np.abs(mel - ref_mel).max())
            if path == FIXTURE:
                worst_fixture = err
            else:
                worst_speech = max(worst_speech, err)
            worst_f64 = max(worst_f64, float(np.abs(mel - float64_logmel(audio)).max()))
            worst_feat = max(worst_feat, float(np.abs(feat - ref_feat).max()))
            total_frames += len(mel)
    print(f"{len(files)} 條音訊、{total_frames} 個 frame")
    print(
        f"log-mel 對 knf 最大絕對誤差：AISHELL 語音 {worst_speech:.2e}（門檻 {TOLERANCE:.0e}）、fixture {worst_fixture:.2e}"
    )
    print(f"log-mel 對 float64 參考最大絕對誤差：{worst_f64:.2e}（門檻 1e-04）")
    print(f"LFR + CMVN 特徵對 Python 最大絕對誤差：{worst_feat:.2e}")
    return 0 if worst_speech < TOLERANCE and worst_f64 < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
