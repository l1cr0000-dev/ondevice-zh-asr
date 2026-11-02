"""串流 Zipformer2 transducer 直接用 onnxruntime 推論（不經 sherpa-onnx）。

存在的原因與 `SenseVoiceOrtAsr` 相同：sherpa-onnx 2.0.1 內附 ORT 1.28.2，載入時留下大量記憶體開銷
（docs/progress.md 2026-11-01），改用專案依賴的 onnxruntime（≥ 1.30）以檔案路徑載入可以避開。

前處理、encoder 串流與解碼照 sherpa-onnx 2.0.1 的實作（`online-recognizer-transducer-impl.h`、
`online-zipformer2-transducer-model.cc`、`online-transducer-greedy-search-decoder.cc`）：

- fbank：kaldi-native-fbank 串流（`OnlineFbank`），80 維、povey 窗、dither 0、snip_edges=False、
  low_freq 20、high_freq -400；樣本維持 [-1, 1]（sherpa-onnx 的 normalize_samples=True）。
- encoder：每次取 T（metadata，X-ASR 480 ms 為 61 = 48 + 右側 context 13）個 frame，前進
  decode_chunk_len（48）個；要「已處理 + T < 可用 frame 數」才送（嚴格小於，與 sherpa-onnx 的
  IsReady 相同）。狀態初始為全零，每次以 encoder 輸出的 `new_<名稱>` 取代。
- greedy search：decoder 初始輸入為 [-1, blank]（context_size 2）；每個 encoder frame 跑一次 joiner，
  blank_penalty > 0 時從 blank 的 logit 減去，取 argmax；不是 blank / <unk> 就輸出並更新 decoder。
  每個 frame 最多輸出一個 token。
  實作上把 chunk 中尚未解碼的 frame 一次送進 joiner（decoder_out 相同），取第一個輸出 token 的 frame，
  更新 decoder 後從下一個 frame 再批次送入；與逐 frame 呼叫的決策完全相同，呼叫次數由每 chunk 12 次
  降到「1 + 輸出 token 數」。前提是 joiner 的批次輸出與逐列輸出逐位元相同：joiner 的
  DynamicQuantizeLinear 依 ONNX 定義是整個 tensor 共用一個 scale，但 ORT 圖最佳化（EXTENDED 以上）
  把它與 MatMulInteger 融合成 DynamicQuantizeMatMul，arm64 上是逐列量化，批次不影響結果
  （關掉最佳化時逐列與批次的 logit 最多差 0.16）。載入時以隨機輸入自我檢查，不成立就退回逐 frame。
- CoreML Execution Provider（選用，`CoreMLOptions`；預設只用 CPU）：第二階段前哨實驗
  （docs/progress.md 2026-11-02）。encoder 的批次維度 N 是符號維度，ORT 推不出大部分張量的形狀，
  CoreML EP 只接得下 1880 / 7431 個節點；把 N 固定為 1（`add_free_dimension_override_by_name`）
  後 3915 / 4828 個，但 INT8 的 DynamicQuantizeLinear / MatMulInteger 不被 CoreML EP 支援，
  每層都被切開（約 415 個 partition）。joiner 的 N 隨批次 joiner 變動，不能固定。
- 文字：token 開頭的「▁」視為空白，但前後都是非 ASCII（中文）時省略，最後去掉首尾空白
  （例：「昨天是 MONDAY TODAY IS 礼拜二」）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.asr.base import AsrStream, StreamingAsr
from src.asr.sherpa import TransducerFiles
from src.audio import SAMPLE_RATE
from src.paths import MODELS_DIR

_FRAME_SHIFT_S = 0.01  # fbank frame shift
_SUBSAMPLING = 4  # encoder 輸出一個 frame = 4 個 fbank frame（40 ms）


@dataclass(frozen=True)
class OrtTransducerConfig:
    num_threads: int = 1
    # 意義與值同 `SherpaTransducerConfig.tail_padding_s`（各模型的值在 `src.asr.TRANSDUCER_SPECS`）
    tail_padding_s: float = 0.66
    blank_penalty: float = 0.0


CoreMLComputeUnits = Literal["CPUOnly", "CPUAndGPU", "CPUAndNeuralEngine", "ALL"]
CoreMLModelFormat = Literal["MLProgram", "NeuralNetwork"]
OrtPart = Literal["encoder", "decoder", "joiner"]


@dataclass(frozen=True)
class CoreMLOptions:
    """onnxruntime CoreML Execution Provider 的設定（只給量測用，見模組說明）。"""

    compute_units: CoreMLComputeUnits = "ALL"
    model_format: CoreMLModelFormat = "MLProgram"
    parts: tuple[OrtPart, ...] = ("encoder",)  # 用 CoreML EP 的 session，其餘只用 CPU
    # encoder / decoder 的批次維度 N 固定為 1（單一 stream 一定是 1；joiner 不固定）
    fix_batch_dim: bool = True
    # 其他 provider 選項，例如 ("RequireStaticInputShapes", "1")、("ModelCacheDirectory", 路徑)
    extra: tuple[tuple[str, str], ...] = ()

    def providers(self) -> list[Any]:
        opts = {"MLComputeUnits": self.compute_units, "ModelFormat": self.model_format}
        return [("CoreMLExecutionProvider", {**opts, **dict(self.extra)}), "CPUExecutionProvider"]


def coreml_available() -> bool:
    import onnxruntime as ort  # type: ignore[import-untyped]

    return "CoreMLExecutionProvider" in ort.get_available_providers()


def tokens_to_text(tokens: list[str]) -> str:
    """token 序列 → 文字，與 sherpa-onnx 的輸出相同（見模組說明）。"""
    out = ""
    for tok in tokens:
        if tok.startswith("▁"):
            rest = tok[1:]
            cjk_join = out and not out[-1].isascii() and not rest[:1].isascii()
            out += rest if cjk_join else " " + rest
        else:
            out += tok
    return out.strip()


class ZipformerOrtStream(AsrStream):
    def __init__(self, asr: "ZipformerOrtAsr") -> None:
        super().__init__()
        self._asr = asr
        self._fbank = asr._knf.OnlineFbank(asr._fbank_opts)
        self._states = [s.copy() for s in asr._init_states]
        self._decoder_out = asr._init_decoder_out
        self._tokens: list[int] = []
        self._frames: list[int] = []  # 每個 token 所在的 encoder 輸出 frame
        self._processed = 0  # 已送入 encoder 的 fbank frame（以 chunk shift 計）
        self._frame_offset = 0  # 已輸出的 encoder frame 數
        self._tail_padding = np.zeros(int(asr.config.tail_padding_s * SAMPLE_RATE), np.float32)

    def _decode_ready(self) -> None:
        asr = self._asr
        while self._processed + asr._chunk_size < self._fbank.num_frames_ready:
            start = self._processed
            x = np.stack(
                [self._fbank.get_frame(i) for i in range(start, start + asr._chunk_size)]
            ).astype(np.float32)
            self._processed += asr._chunk_shift
            outputs = asr._encoder.run(
                asr._encoder_outputs,
                {asr._x_name: x[None], **dict(zip(asr._state_names, self._states, strict=True))},
            )
            self._states = outputs[1:]
            encoder_out = outputs[0][0]
            self._greedy_search(encoder_out)
            self._frame_offset += len(encoder_out)

    def _greedy_search(self, encoder_out: npt.NDArray[np.float32]) -> None:
        """一個 chunk 的 greedy search（見模組說明：批次送入 joiner，決策與逐 frame 相同）。"""
        asr = self._asr
        t = 0
        while t < len(encoder_out):
            rows = encoder_out[t:] if asr._batch_joiner else encoder_out[t : t + 1]
            logits = asr.run_joiner(rows, self._decoder_out)
            if asr.config.blank_penalty > 0:
                logits[:, 0] -= asr.config.blank_penalty
            ys = logits.argmax(axis=1)
            (emit,) = np.nonzero((ys != 0) & (ys != asr._unk_id))
            if len(emit) == 0:
                t += len(rows)
                continue
            t += int(emit[0])
            self._tokens.append(int(ys[emit[0]]))
            self._frames.append(self._frame_offset + t)
            self._decoder_out = asr.run_decoder(
                [*asr._context_prefix, *self._tokens][-asr._context_size :]
            )
            t += 1

    def _accept(self, chunk: npt.NDArray[np.float32]) -> None:
        self._fbank.accept_waveform(SAMPLE_RATE, chunk.tolist())
        self._decode_ready()

    def _partial(self) -> str:
        return tokens_to_text([self._asr._symbols[t] for t in self._tokens])

    def get_token_timestamps(self) -> list[float]:
        # 與 sherpa-onnx 相同：token 被輸出的那個 encoder frame（40 ms 為單位）
        return [f * _FRAME_SHIFT_S * _SUBSAMPLING for f in self._frames]

    def _finalize(self) -> str:
        if len(self._tail_padding):
            self._fbank.accept_waveform(SAMPLE_RATE, self._tail_padding.tolist())
        self._fbank.input_finished()
        self._decode_ready()
        return self._partial()


class ZipformerOrtAsr(StreamingAsr):
    def __init__(
        self,
        files: TransducerFiles,
        config: OrtTransducerConfig | None = None,
        models_dir: Path = MODELS_DIR,
        sess_options: Any = None,
        coreml: CoreMLOptions | None = None,
    ) -> None:
        """`sess_options`：自訂的 `ort.SessionOptions`（量測 session 選項用，見
        scripts/bench_asr_compute.py）；給定時執行緒數以它為準。
        `coreml`：`parts` 中的 session 改用 CoreML EP（不在 CoreML 上的節點仍由 CPU 執行）。"""
        import kaldi_native_fbank as knf  # type: ignore[import-untyped]
        import onnxruntime as ort

        self.files = files
        self.config = config or OrtTransducerConfig()
        self.coreml = coreml
        if coreml is not None and sess_options is not None:
            raise ValueError("sess_options 與 coreml 不能同時給")
        paths = self._paths = files.paths(models_dir)
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "找不到模型檔，請先執行 bash scripts/download_models.sh：\n  "
                + "\n  ".join(missing)
            )

        def new_options() -> Any:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.config.num_threads
            opts.inter_op_num_threads = 1
            return opts

        shared_opts: Any = sess_options if sess_options is not None else new_options()

        def session(name: OrtPart) -> Any:
            opts, providers = shared_opts, ["CPUExecutionProvider"]
            if coreml is not None and name in coreml.parts:
                providers = coreml.providers()
                if coreml.fix_batch_dim and name != "joiner":
                    opts = new_options()
                    opts.add_free_dimension_override_by_name("N", 1)
            # 以檔案路徑建立 session（不先讀進記憶體）
            return ort.InferenceSession(str(paths[name]), opts, providers=providers)

        self._encoder = session("encoder")
        self._decoder = session("decoder")
        self._joiner = session("joiner")

        meta = self._encoder.get_modelmeta().custom_metadata_map
        if meta.get("model_type") != "zipformer2":
            raise ValueError(
                f"只支援串流 zipformer2，{paths['encoder']} 是 {meta.get('model_type')}"
            )
        self._chunk_size = int(meta["T"])
        self._chunk_shift = int(meta["decode_chunk_len"])
        self._context_size = int(self._decoder.get_modelmeta().custom_metadata_map["context_size"])
        # 尚未輸出任何 token 時 decoder 的輸入（sherpa-onnx 的 GetEmptyResult）
        self._context_prefix = [-1] * (self._context_size - 1) + [0]

        inputs = self._encoder.get_inputs()
        self._x_name: str = inputs[0].name
        self._state_names: list[str] = [i.name for i in inputs[1:]]
        self._encoder_outputs = [self._encoder.get_outputs()[0].name] + [
            f"new_{n}" for n in self._state_names
        ]
        self._init_states = [
            np.zeros(
                [1 if isinstance(d, str) else d for d in i.shape],
                dtype=np.int64 if i.type == "tensor(int64)" else np.float32,
            )
            for i in inputs[1:]
        ]

        self._symbols: dict[int, str] = {}
        for line in paths["tokens"].read_text(encoding="utf-8").splitlines():
            sym, idx = line.rsplit(" ", 1)
            self._symbols[int(idx)] = sym
        self._unk_id = next((i for i, s in self._symbols.items() if s == "<unk>"), -1)

        fbank = knf.FbankOptions()
        fbank.frame_opts.samp_freq = SAMPLE_RATE
        fbank.frame_opts.dither = 0.0
        fbank.frame_opts.snip_edges = False
        fbank.frame_opts.window_type = "povey"
        fbank.mel_opts.num_bins = 80
        fbank.mel_opts.low_freq = 20.0
        fbank.mel_opts.high_freq = -400.0
        self._fbank_opts = fbank
        self._knf = knf
        self._init_decoder_out = self.run_decoder(self._context_prefix)
        self._batch_joiner = self._joiner_is_row_independent()

    def _joiner_is_row_independent(self) -> bool:
        """joiner 批次輸出是否與逐列輸出逐位元相同（見模組說明）。各列刻意用不同的數值範圍。"""
        rng = np.random.default_rng(0)
        dim = self._init_decoder_out.shape[-1]
        rows = (rng.standard_normal((8, dim)) * np.linspace(0.2, 3.0, 8)[:, None]).astype(
            np.float32
        )
        batch = self.run_joiner(rows, self._init_decoder_out)
        single = np.concatenate([self.run_joiner(r[None], self._init_decoder_out) for r in rows])
        return bool(np.array_equal(batch, single))

    def run_joiner(
        self, encoder_out: npt.NDArray[np.float32], decoder_out: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """encoder_out [N, D] 各列與同一個 decoder_out [1, D] 的 logit [N, vocab]。"""
        (out,) = self._joiner.run(
            ["logit"],
            {
                "encoder_out": encoder_out,
                "decoder_out": np.repeat(decoder_out, len(encoder_out), axis=0),
            },
        )
        result: npt.NDArray[np.float32] = out
        return result

    def run_decoder(self, context: list[int]) -> npt.NDArray[np.float32]:
        (out,) = self._decoder.run(["decoder_out"], {"y": np.array([context], dtype=np.int64)})
        result: npt.NDArray[np.float32] = out
        return result

    @property
    def name(self) -> str:
        return (
            f"onnxruntime/{self.files.model_dir}/{self.files.encoder}"
            f" threads={self.config.num_threads} greedy_search"
            + (f" blank_penalty={self.config.blank_penalty:g}" if self.config.blank_penalty else "")
            + (
                f" coreml={self.coreml.compute_units}/{self.coreml.model_format}"
                f"/{'+'.join(self.coreml.parts)}"
                if self.coreml
                else ""
            )
        )

    def model_files(self) -> list[Path]:
        return list(self._paths.values())

    def create_stream(self) -> ZipformerOrtStream:
        return ZipformerOrtStream(self)
