"""麥克風即時辨識 demo：麥克風 → VAD → 串流 ASR，終端即時顯示 partial，句子結束後輸出最終結果與首字延遲。

用法：
    uv run scripts/demo_mic.py                       # 預設輸入裝置
    uv run scripts/demo_mic.py --list-devices        # 列出輸入裝置
    uv run scripts/demo_mic.py --device 2
    uv run scripts/demo_mic.py --model sensevoice-small-int8-p480
    # 預設是兩個門檻（方案 A2）：靜音 150 ms 先給暫定結果，1000 ms 才定稿；退回舊的單一門檻 500 ms：
    uv run scripts/demo_mic.py --min-silence-ms 500 --pause-ms 0

Ctrl+C 結束，印出本次統計（句數、首字延遲與「說完 → 結果」的平均 / P50 / P90）。

「說完 → 結果」：VAD 語音終點（含 speech_pad_ms）對應的樣本到達時刻 → 這句的完整文字顯示的時刻
（有暫定結果時取最後一個暫定結果，否則取最終結果）。暫定結果以「（暫定）」標示，之後若又開始說話會被修正。

首字延遲在這裡是**牆鐘時間**：語音起點（VAD 起點，含 speech_pad_ms）對應的樣本到達時刻 → 第一個非空
partial 顯示的時刻。樣本到達時刻以 PortAudio callback 被呼叫的時間推算，不含音訊硬體本身的輸入延遲
（開頭會印出 PortAudio 回報的輸入延遲供參考）。這是單次體感量測，正式指標請用 eval/ 的腳本。

macOS 麥克風權限：第一次執行時系統會詢問是否允許「執行這個指令的 App」（Terminal / iTerm / Claude 等）
使用麥克風。若曾拒絕，錄到的會是全零的靜音，腳本會提示；到「系統設定 → 隱私權與安全性 → 麥克風」
打開該 App 後，需重新啟動該 App 才會生效。
"""

import argparse
import importlib
import queue
import shutil
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import numpy.typing as npt

# 讓 `uv run scripts/demo_mic.py` 能直接 import src（專案本身沒有安裝成套件）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import SAMPLE_RATE, load_asr
from src.cli import add_pipeline_args, char_width, pipeline_config_from_args
from src.pipeline import PipelineEvent, StreamingPipeline
from src.vad import SileroVadModel

# sounddevice 沒有型別標註
sd: Any = importlib.import_module("sounddevice")

# 開始錄音後這麼久都是全零樣本，就判定為沒有麥克風權限（或裝置靜音）
_SILENCE_CHECK_S = 2.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="麥克風即時辨識：VAD + 串流 ASR，即時顯示 partial 與每句首字延遲。Ctrl+C 結束。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", help="輸入裝置編號或名稱（見 --list-devices）；不給用系統預設")
    p.add_argument("--list-devices", action="store_true", help="列出輸入裝置後結束")
    add_pipeline_args(p)  # --chunk-ms 在這裡也是每次從麥克風讀取的長度
    return p.parse_args()


def list_input_devices() -> None:
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "*" if i == default_in else " "
            print(f"{mark} [{i}] {d['name']}  default_sr={d['default_samplerate']:.0f}")


def parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


# ----- 終端顯示 -----


def fit_tail(text: str, width: int) -> str:
    """取 text 的尾端，使顯示寬度不超過 width（中文字寬 2）。超出時前面加「…」。"""
    total = 0
    for i in range(len(text) - 1, -1, -1):
        total += char_width(text[i])
        if total > width - 1:
            return "…" + text[i + 1 :]
    return text


class LineWriter:
    """同一行刷新 partial；`commit` 把目前這行換成最終內容並換行。"""

    def __init__(self) -> None:
        self.isatty = sys.stdout.isatty()

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size((80, 24)).columns - 1)

    def update(self, text: str) -> None:
        if not self.isatty:
            return  # 非終端（例如重導向到檔案）只輸出最終結果
        sys.stdout.write("\r\033[K" + fit_tail(text, self._width()))
        sys.stdout.flush()

    def clear(self) -> None:
        if self.isatty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def commit(self, text: str) -> None:
        self.clear()
        print(text, flush=True)


# ----- 主流程 -----


@dataclass
class Session:
    """把 pipeline 事件轉成終端輸出，並記錄每句的牆鐘首字延遲。"""

    writer: LineWriter
    # 音訊第 0 個樣本對應的 perf_counter 時刻（由 callback 到達時間推算）
    t0: float = float("inf")
    first_token_wall: dict[int, float] = field(default_factory=dict)
    provisional_wall: dict[int, float] = field(default_factory=dict)  # 最後一個暫定結果的顯示時刻
    latencies_ms: list[float] = field(default_factory=list)
    result_ms: list[float] = field(default_factory=list)

    def handle(self, ev: PipelineEvent, now: float, interrupted: bool = False) -> None:
        utt = ev.utterance
        if ev.kind == "start":
            self.writer.update("…")
        elif ev.kind == "partial":
            text = utt.partials[-1].text
            if text.strip() and utt.index not in self.first_token_wall:
                self.first_token_wall[utt.index] = now
            self.writer.update(text or "…")
        elif ev.kind == "provisional":
            text = utt.provisionals[-1].text
            if text.strip():
                self.first_token_wall.setdefault(utt.index, now)  # 非串流模型的第一個可見文字
                self.provisional_wall[utt.index] = now
            self.writer.update(f"{text}（暫定）")
        elif ev.kind == "final":
            if not utt.text.strip():
                self.writer.clear()  # 噪音觸發 VAD 但沒辨識出字，不算一句
                return
            first = self.first_token_wall.get(utt.index, now)  # 只在 finalize 才出字
            latency_ms = (first - (self.t0 + utt.speech_start_s)) * 1000
            self.latencies_ms.append(latency_ms)
            tag = "（中斷）" if interrupted else ""
            detail = f"首字 {latency_ms:.0f} ms"
            if utt.speech_end_s is not None and not interrupted:
                # 停頓後沒有恢復時，最後一個暫定結果就是最終文字
                shown = self.provisional_wall.get(utt.index, now)
                if utt.provisionals and utt.provisionals[-1].text != utt.text:
                    shown = now
                result = (shown - (self.t0 + utt.speech_end_s)) * 1000
                self.result_ms.append(result)
                detail += f"，說完 → 結果 {result:.0f} ms"
            self.writer.commit(f"[{len(self.latencies_ms)}] {utt.text}{tag}  （{detail}）")

    def print_summary(self) -> None:
        print("\n== 本次統計 ==")
        n = len(self.latencies_ms)
        print(f"句數：{n}")
        if n == 0:
            return
        for name, values in (("首字延遲", self.latencies_ms), ("說完 → 結果", self.result_ms)):
            if not values:
                continue
            line = f"{name}：平均 {statistics.fmean(values):.0f} ms"
            if len(values) >= 2:
                p50, p90 = np.percentile(values, [50, 90])
                line += f"，P50 / P90 {p50:.0f} / {p90:.0f} ms（最小 {min(values):.0f}、最大 {max(values):.0f}）"
            print(line)


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return 0

    device = parse_device(args.device)
    try:
        sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=SAMPLE_RATE)
        config = pipeline_config_from_args(args)
        vad_model = SileroVadModel(num_threads=args.threads)
        asr = load_asr(args.model, num_threads=args.threads)
    except (ValueError, FileNotFoundError, sd.PortAudioError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1
    pipeline = StreamingPipeline(vad_model, asr, config)

    blocks: queue.Queue[tuple[npt.NDArray[np.float32], float]] = queue.Queue()
    overflows = 0

    def callback(indata: npt.NDArray[np.float32], frames: int, time_info: Any, status: Any) -> None:
        nonlocal overflows
        arrival = time.perf_counter()
        if status.input_overflow:
            overflows += 1
        blocks.put((indata[:, 0].copy(), arrival))

    # Ctrl+C 只設旗標，讓主迴圈在兩個 chunk 之間乾淨地停下，避免打斷 pipeline 內部狀態
    stop = threading.Event()

    def on_sigint(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    session = Session(LineWriter())
    received = 0
    peak = 0.0
    warned_silent = False
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=args.chunk_ms * SAMPLE_RATE // 1000,
            device=device,
            channels=1,
            dtype="float32",
            latency="low",
            callback=callback,
        )
    except sd.PortAudioError as e:
        print(f"錯誤：無法開啟輸入裝置：{e}", file=sys.stderr)
        return 1
    dev_name = sd.query_devices(stream.device)["name"]
    print(
        f"模型 {asr.name}｜執行緒 {args.threads}｜裝置 {dev_name}｜"
        f"PortAudio 輸入延遲 {stream.latency * 1000:.0f} ms",
        file=sys.stderr,
    )
    print("開始說話吧，Ctrl+C 結束。\n", file=sys.stderr)

    with stream:
        while not stop.is_set():
            try:
                chunk, arrival = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            received += len(chunk)
            # callback 呼叫時刻 − 已收音訊長度，取最小值當作第 0 個樣本的時刻，濾掉排程抖動
            session.t0 = min(session.t0, arrival - received / SAMPLE_RATE)
            peak = max(peak, float(np.abs(chunk).max(initial=0.0)))
            if not warned_silent and peak == 0.0 and received >= _SILENCE_CHECK_S * SAMPLE_RATE:
                warned_silent = True
                session.writer.commit(
                    "⚠️  收到的音訊全是 0：可能沒有麥克風權限。到「系統設定 → 隱私權與安全性 → 麥克風」"
                    "允許執行這個指令的 App（Terminal / iTerm / Claude 等），然後重新啟動該 App。"
                )
            for ev in pipeline.accept_waveform(chunk):
                session.handle(ev, time.perf_counter())

    # 麥克風已關閉；結束說到一半的句子
    for ev in pipeline.flush():
        session.handle(ev, time.perf_counter(), interrupted=True)
    session.writer.clear()
    if overflows:
        print(
            f"\n注意：輸入緩衝溢位 {overflows} 次（處理跟不上即時，延遲數字偏高）", file=sys.stderr
        )
    session.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
