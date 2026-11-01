"""非串流 SenseVoice 後端：sherpa-onnx 版（`SenseVoiceAsr`）與直接用 onnxruntime 的版本（`SenseVoiceOrtAsr`）。

兩者用同一個模型檔、同一套串流包裝（`SenseVoiceStream`），只差「整段辨識一次」怎麼做。
onnxruntime 版存在的原因：sherpa-onnx 2.0.1 內附 ORT 1.28.2，載入 SenseVoice 時留下約 500 MB 的
記憶體開銷（docs/progress.md 2026-11-01），直接用專案依賴的 onnxruntime（≥ 1.30）可以避開。

非串流模型一次看完整段音訊才辨識，沒有真正的 partial。實作成 `AsrStream` 的方式：
- `accept_waveform` 只累積音訊；`finalize` 時把累積的音訊整段辨識一次。
- 選用 `partial_every_s`：每多累積這麼長的音訊，就把目前為止的全部音訊整段辨識一次當 partial
  （假的串流，偽串流）。計算量隨句長成長，計入 `accept_waveform` 的耗時，因此會反映在模擬即時的延遲上。
  None = 不產生 partial（首字要等到 finalize 或 pipeline 的暫定結果）。
- 降計算量的兩個選項（`PartialSchedule`）：`partial_first_s` 讓第一次 partial 與之後的間隔分開設定
  （首字只取決於第一次，之後的更新可以稀疏）；`partial_max_s` 讓累積音訊超過這個長度後不再產生 partial，
  只等 finalize / 暫定結果，限制長句段的最壞計算量。
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    partial_first_s: float | None = None  # None = partial_every_s
    partial_max_s: float | None = None  # None = 不限

    @property
    def schedule(self) -> "PartialSchedule | None":
        if self.partial_every_s is None:
            return None
        return PartialSchedule(self.partial_every_s, self.partial_first_s, self.partial_max_s)

    def describe(self) -> str:
        sched = self.schedule
        return f"threads={self.num_threads} language={self.language} itn={self.use_itn}" + (
            f" {sched.describe()}" if sched else ""
        )


@dataclass(frozen=True)
class PartialSchedule:
    """偽 partial 的重新辨識時機，以這個 stream 已累積的音訊長度（秒）計：

    first_s、first_s + every_s、first_s + 2 × every_s……，超過 max_s 之後不再重新辨識。
    """

    every_s: float
    first_s: float | None = None  # None = every_s
    max_s: float | None = None  # None = 不限

    def __post_init__(self) -> None:
        if self.every_s <= 0 or (self.first_s is not None and self.first_s <= 0):
            raise ValueError("partial 間隔必須為正")

    def next_sample(self, n: int) -> int | None:
        """已累積 n 個樣本（且已處理完 n 之前的時機）時，下一次重新辨識的樣本數；None = 不再辨識。"""
        first = round((self.every_s if self.first_s is None else self.first_s) * SAMPLE_RATE)
        every = round(self.every_s * SAMPLE_RATE)
        nxt = first if n < first else first + ((n - first) // every + 1) * every
        if self.max_s is not None and nxt > self.max_s * SAMPLE_RATE:
            return None
        return nxt

    def describe(self) -> str:
        out = f"partial_every={self.every_s:g}s"
        out += f" partial_first={self.first_s:g}s" if self.first_s is not None else ""
        out += f" partial_max={self.max_s:g}s" if self.max_s is not None else ""
        return out


class SenseVoiceStream(AsrStream):
    """累積音訊；`recognize(整段音訊) → 文字` 由後端提供。"""

    def __init__(
        self,
        recognize: Callable[[npt.NDArray[np.float32]], str],
        schedule: PartialSchedule | None,
    ) -> None:
        super().__init__()
        self._recognize = recognize
        self._chunks: list[npt.NDArray[np.float32]] = []
        self._n = 0  # 已累積的樣本數
        self._schedule = schedule
        self._next_partial = None if schedule is None else schedule.next_sample(0)
        self._partial_text = ""
        self.num_decodes = 0  # 整段辨識的次數（含 finalize）

    def _decode(self) -> str:
        if not self._n:
            return ""
        self.num_decodes += 1
        return self._recognize(np.concatenate(self._chunks))

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        self._chunks.append(chunk.copy())
        self._n += len(chunk)
        nxt = self._next_partial
        if self._schedule is not None and nxt is not None and self._n >= nxt:
            self._partial_text = self._decode()
            self._next_partial = self._schedule.next_sample(self._n)

    def _partial(self) -> str:
        return self._partial_text

    def _finalize(self) -> str:
        return self._decode()


class _WholeUtteranceAsr(StreamingAsr):
    """兩個後端共用：整段辨識的計數（`compute_stats`）與建立 `SenseVoiceStream`。"""

    config: SenseVoiceConfig
    _num_decodes = 0
    _decoded_s = 0.0

    def _recognize_audio(self, audio: npt.NDArray[np.float32]) -> str:
        raise NotImplementedError

    def _counted(self, audio: npt.NDArray[np.float32]) -> str:
        self._num_decodes += 1
        self._decoded_s += len(audio) / SAMPLE_RATE
        return self._recognize_audio(audio)

    def compute_stats(self) -> dict[str, float]:
        return {
            "num_decodes": self._num_decodes,
            "decoded_audio_s": self._decoded_s,
        }

    def reset_compute_stats(self) -> None:
        self._num_decodes = 0
        self._decoded_s = 0.0

    def create_stream(self) -> SenseVoiceStream:
        return SenseVoiceStream(self._counted, self.config.schedule)


class SenseVoiceAsr(_WholeUtteranceAsr):
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
        return f"sherpa-onnx/{SENSE_VOICE_DIR}/model.int8.onnx {self.config.describe()}"

    def model_files(self) -> list[Path]:
        return list(self._paths.values())

    def _recognize_audio(self, audio: npt.NDArray[np.float32]) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._recognizer.decode_stream(stream)
        return str(stream.result.text)


class SenseVoiceOrtAsr(_WholeUtteranceAsr):
    """直接用 onnxruntime 跑 SenseVoice，前處理與解碼照 sherpa-onnx 2.0.1 的實作：

    - fbank：kaldi-native-fbank（sherpa-onnx 用的同一個函式庫），80 維、hamming 窗、dither 0、
      snip_edges；metadata `normalize_samples=0` 時樣本先乘 32768（int16 範圍）。
      （`offline-recognizer-sense-voice-impl.h` 的 InitFeatConfig）
    - LFR：每 6 個 frame 取一次、拼接以它為中心的 7 個 frame（左 3 右 3，超出邊界重複首 / 末 frame），
      輸出 ceil(T / 6) 個 560 維 frame（`csrc/lfr.cc`）。之後 CMVN：(x + neg_mean) * inv_stddev。
    - 模型輸入另有 language（metadata `lang_<語言>`）與 text_norm（with_itn / without_itn）。
    - CTC greedy：逐 frame argmax、合併重複、去掉 blank（id 0），再丟掉前 4 個 token
      （語言、情緒、事件、ITN 標記）。token 中的「▁」換成空白。
    """

    def __init__(self, config: SenseVoiceConfig | None = None, models_dir: Path = MODELS_DIR):
        import kaldi_native_fbank as knf  # type: ignore[import-untyped]
        import onnxruntime as ort  # type: ignore[import-untyped]

        self.config = config or SenseVoiceConfig()
        root = models_dir / SENSE_VOICE_DIR
        self._paths = {"model": root / "model.int8.onnx", "tokens": root / "tokens.txt"}
        missing = [str(p) for p in self._paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "找不到模型檔，請先執行 bash scripts/download_models.sh：\n  "
                + "\n  ".join(missing)
            )
        opts: Any = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.num_threads
        opts.inter_op_num_threads = 1
        # 以檔案路徑建立 session（不先讀進記憶體），見模組說明
        self._session: Any = ort.InferenceSession(
            str(self._paths["model"]), opts, providers=["CPUExecutionProvider"]
        )
        meta = self._session.get_modelmeta().custom_metadata_map
        self._lfr_m = int(meta["lfr_window_size"])
        self._lfr_n = int(meta["lfr_window_shift"])
        self._neg_mean = np.array(meta["neg_mean"].split(","), dtype=np.float32)
        self._inv_std = np.array(meta["inv_stddev"].split(","), dtype=np.float32)
        self._scale = 1.0 if int(meta["normalize_samples"]) else 32768.0
        self._language = np.array([int(meta[f"lang_{self.config.language}"])], dtype=np.int32)
        itn = meta["with_itn"] if self.config.use_itn else meta["without_itn"]
        self._text_norm = np.array([int(itn)], dtype=np.int32)
        self._tokens: dict[int, str] = {}
        for line in self._paths["tokens"].read_text(encoding="utf-8").splitlines():
            sym, idx = line.rsplit(" ", 1)
            self._tokens[int(idx)] = sym
        fbank = knf.FbankOptions()
        fbank.frame_opts.samp_freq = SAMPLE_RATE
        fbank.frame_opts.dither = 0.0
        fbank.frame_opts.window_type = "hamming"
        fbank.frame_opts.snip_edges = True
        fbank.mel_opts.num_bins = 80
        fbank.mel_opts.high_freq = 0
        self._fbank_opts = fbank
        self._knf = knf

    def features(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """整段音訊 → LFR + CMVN 後的 (T', 560) 特徵。"""
        fbank = self._knf.OnlineFbank(self._fbank_opts)
        fbank.accept_waveform(SAMPLE_RATE, (audio * self._scale).tolist())
        fbank.input_finished()
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, 80 * self._lfr_m), dtype=np.float32)
        feats = np.stack([np.asarray(fbank.get_frame(i), dtype=np.float32) for i in range(n)])
        out_frames = 1 + (n - 1) // self._lfr_n
        idx = np.arange(out_frames)[:, None] * self._lfr_n - (self._lfr_m - 1) // 2
        idx = np.clip(idx + np.arange(self._lfr_m)[None, :], 0, n - 1)
        lfr = feats[idx].reshape(out_frames, -1)
        out: npt.NDArray[np.float32] = ((lfr + self._neg_mean) * self._inv_std).astype(np.float32)
        return out

    def _recognize_audio(self, audio: npt.NDArray[np.float32]) -> str:
        x = self.features(audio)
        if not len(x):
            return ""
        (logits,) = self._session.run(
            ["logits"],
            {
                "x": x[None],
                "x_length": np.array([len(x)], dtype=np.int32),
                "language": self._language,
                "text_norm": self._text_norm,
            },
        )
        ids = logits[0].argmax(axis=-1)
        tokens: list[int] = []
        prev = -1
        for y in ids.tolist():
            if y != 0 and y != prev:
                tokens.append(y)
            prev = y
        return "".join(self._tokens[t] for t in tokens[4:]).replace("▁", " ").strip()

    @property
    def name(self) -> str:
        return f"onnxruntime/{SENSE_VOICE_DIR}/model.int8.onnx {self.config.describe()}"

    def model_files(self) -> list[Path]:
        return list(self._paths.values())
