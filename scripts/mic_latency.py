"""B0：麥克風路徑的真實端到端延遲 vs 模擬即時。同一段音訊分別跑三條路徑，比較首字延遲與結果延遲。

- sim：模擬即時（與 eval/run_eval.py 相同）。依 block 切塊送入，時刻以「樣本依實際時長到達」推算，
  不 sleep、沒有交付延遲。另跑一次 `--chunk-ms`（預設 10 ms）當作正式評測的基準。
- thread：不出聲的軟體路徑。另一條執行緒照真實時鐘、每 block 時長送一次 block 到 queue，主執行緒的
  取出 → pipeline 迴圈與 scripts/demo_mic.py 相同。量「真實時鐘 + 執行緒排程 + queue」多出的延遲。
- portaudio：真的開啟輸入裝置（sounddevice / PortAudio），callback 的節奏、緩衝與時間戳都是真的，但
  **丟棄麥克風收到的聲音**（不儲存），改成依序送入檔案音訊。比 thread 多了驅動 / PortAudio 的緩衝與
  交付延遲。延遲起點用 PortAudio 回報的 `inputBufferAdcTime`（block 第一個樣本的擷取時刻）推算。
  不含聲學路徑（喇叭 → 空氣 → 麥克風）與 Core Audio 未計入 ADC 時間戳的硬體延遲（若有）。

參考音訊：MFA 對齊集（data/aishell1_align_300）前 `--n` 句，每句前補 `--lead-ms`、後補 `--tail-ms` 靜音，
接成一條長串流（真實路徑每句要花真實時間：30 句約 3.5 分鐘 / 每個 block 大小）。指標定義見 eval/realtime.py。

用法：
    uv run scripts/mic_latency.py                    # 30 句，sim / thread × block 160/320/512/1024（不碰麥克風）
    uv run scripts/mic_latency.py --n 20 --blocksize 160
    # portaudio 路徑會開啟麥克風：需使用者同意才執行，且要加 --allow-mic
    uv run scripts/mic_latency.py --modes sim thread portaudio --allow-mic

結果存成 eval/results/<日期>_mic-latency_<tag>.json，並印出總表。portaudio 路徑**預設不跑**：它會開啟輸入裝置
（macOS 顯示麥克風使用中），需使用者同意才執行，並明確給 `--modes ... portaudio --allow-mic`。沒有麥克風權限時
callback 收到全零，不影響量測（聲音本來就丟棄）。
"""

import argparse
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import summarize
from eval.realtime import (
    FileSpan,
    SegmentTimes,
    adc_t0,
    assign_segments,
    clock_offset,
    demo_mic_t0,
    file_latencies,
    layout_spans,
)
from src.asr import SAMPLE_RATE, StreamingAsr, load_asr
from src.cli import add_pipeline_args, pipeline_config_from_args, positive_int
from src.pipeline import PipelineConfig, PipelineEvent, StreamingPipeline, Utterance
from src.vad import SileroVadModel, load_wav

# sounddevice 沒有型別標註
sd: Any = importlib.import_module("sounddevice")

ALIGN_DIR = REPO_ROOT / "data" / "aishell1_align_300"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
METRICS = ("first_token_vad_ms", "first_token_true_ms", "result_ms", "final_ms")
METRIC_NAMES = {
    "first_token_vad_ms": "首字（VAD 起點）",
    "first_token_true_ms": "首字（真實起點）",
    "result_ms": "說完 → 完整結果",
    "final_ms": "說完 → 定稿",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="麥克風路徑真實延遲 vs 模擬即時（B0）：sim / thread / portaudio 三條路徑比較",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=positive_int, default=30, help="對齊集前幾句（最多 300）")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["sim", "thread", "portaudio"],
        default=["sim", "thread"],
        help="portaudio 會開啟麥克風，需使用者同意，且要加 --allow-mic",
    )
    p.add_argument(
        "--allow-mic", action="store_true", help="允許 portaudio 路徑開啟輸入裝置（需使用者同意）"
    )
    p.add_argument(
        "--blocksize",
        type=positive_int,
        nargs="+",
        default=[160, 320, 512, 1024],
        help="每個 block 的樣本數（16 kHz；160 = demo_mic 預設的 10 ms）",
    )
    p.add_argument("--lead-ms", type=positive_int, default=1000, help="每句前補的靜音")
    p.add_argument(
        "--tail-ms", type=positive_int, default=1500, help="每句後補的靜音（需 > min_silence）"
    )
    p.add_argument("--device", help="portaudio 路徑的輸入裝置編號或名稱；不給用系統預設")
    p.add_argument("--list-devices", action="store_true", help="列出輸入裝置後結束")
    p.add_argument("--tag", default="", help="結果檔名後綴")
    add_pipeline_args(p)  # --chunk-ms：sim 路徑額外跑的正式評測基準 chunk
    return p.parse_args()


# ----- 參考音訊 -----


def load_reference(
    n: int, lead_ms: int, tail_ms: int
) -> tuple[npt.NDArray[np.float32], list[FileSpan]]:
    manifest = [
        json.loads(x) for x in (ALIGN_DIR / "manifest.jsonl").read_text("utf-8").splitlines()
    ]
    aligns = {
        r["id"]: r["chars"]
        for r in map(json.loads, (ALIGN_DIR / "alignments.jsonl").read_text("utf-8").splitlines())
    }
    audios: list[npt.NDArray[np.float32]] = []
    rows: list[tuple[str, int, float, float]] = []
    for row in manifest[:n]:
        audio = load_wav(row["wav"])
        chars = aligns[row["id"]]
        audios.append(audio)
        rows.append((row["id"], len(audio), chars[0]["start"], chars[-1]["end"]))
    lead = lead_ms * SAMPLE_RATE // 1000
    tail = tail_ms * SAMPLE_RATE // 1000
    spans, total = layout_spans(rows, lead, tail)
    stream = np.zeros(total, dtype=np.float32)
    pos = 0
    for audio in audios:
        stream[pos + lead : pos + lead + len(audio)] = audio
        pos += lead + len(audio) + tail
    return stream, spans


# ----- 事件記錄 -----


@dataclass
class WallLog:
    """真實時鐘路徑：記錄每個事件被主迴圈處理的牆鐘時刻（與 demo_mic 的 Session 相同時機）。"""

    first_token: dict[int, float] = field(default_factory=dict)
    provisional: dict[int, list[float]] = field(default_factory=dict)
    final: dict[int, float] = field(default_factory=dict)

    def handle(self, ev: PipelineEvent, now: float) -> None:
        utt = ev.utterance
        if ev.kind == "partial" and utt.partials[-1].text.strip():
            self.first_token.setdefault(utt.index, now)
        elif ev.kind == "provisional":
            self.provisional.setdefault(utt.index, []).append(now)
        elif ev.kind == "final":
            self.final[utt.index] = now
            if utt.text.strip():
                self.first_token.setdefault(utt.index, now)  # 只在 finalize 才出字


def segments_simulated(utts: Sequence[Utterance]) -> list[SegmentTimes]:
    """pipeline 自己的模擬即時時刻（sim 路徑；真實路徑中則是「同一次執行若沒有交付延遲」的時刻）。"""
    return [
        SegmentTimes(
            text=u.text,
            speech_start_s=u.speech_start_s,
            first_token_s=u.first_token_s,
            final_s=u.final_s,
            provisionals=[(p.emitted_s, p.audio_end_s) for p in u.provisionals],
        )
        for u in utts
    ]


def segments_wall(utts: Sequence[Utterance], log: WallLog, t0: float) -> list[SegmentTimes]:
    out = []
    for u in utts:
        ft = log.first_token.get(u.index)
        fin = log.final.get(u.index)
        provs = log.provisional.get(u.index, [])
        out.append(
            SegmentTimes(
                text=u.text,
                speech_start_s=u.speech_start_s,
                first_token_s=None if ft is None else ft - t0,
                final_s=None if fin is None else fin - t0,
                provisionals=[
                    (w - t0, p.audio_end_s) for w, p in zip(provs, u.provisionals, strict=True)
                ],
            )
        )
    return out


# ----- 三條路徑 -----


@dataclass
class BlockRecord:
    start: int  # block 第一個樣本的索引
    end: int
    arrival: float  # callback / 回放執行緒放進 queue 的牆鐘時刻
    adc: float  # block 第一個樣本的擷取時刻（牆鐘）
    dequeued: float = 0.0
    processed: float = 0.0


@dataclass
class RunOutput:
    utterances: list[Utterance]
    t0: float = 0.0  # 第 0 個樣本擷取時刻（牆鐘）；sim 為 0
    log: WallLog | None = None
    blocks: list[BlockRecord] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    compute_s: float = (
        0.0  # pipeline 的總計算時間（sim：整次執行；真實路徑：各 block 處理時間總和）
    )


def run_sim(
    audio: npt.NDArray[np.float32], block: int, make: Callable[[], StreamingPipeline]
) -> RunOutput:
    pipeline = make()
    t = time.perf_counter()
    for i in range(0, len(audio), block):
        pipeline.accept_waveform(audio[i : i + block])
    pipeline.flush()
    return RunOutput(pipeline.utterances, compute_s=time.perf_counter() - t)


def consume(
    pipeline: StreamingPipeline,
    blocks: "queue.Queue[BlockRecord | None]",
    chunks: dict[int, npt.NDArray[np.float32]],
    log: WallLog,
) -> list[BlockRecord]:
    """主迴圈：與 demo_mic 相同的「queue 取出 → accept_waveform → 處理事件」；None 表示輸入結束。"""
    done: list[BlockRecord] = []
    while True:
        rec = blocks.get(timeout=5.0)
        if rec is None:
            break
        rec.dequeued = time.perf_counter()
        for ev in pipeline.accept_waveform(chunks.pop(rec.start)):
            log.handle(ev, time.perf_counter())
        rec.processed = time.perf_counter()
        done.append(rec)
    for ev in pipeline.flush():
        log.handle(ev, time.perf_counter())
    return done


def run_thread(
    audio: npt.NDArray[np.float32], block: int, make: Callable[[], StreamingPipeline]
) -> RunOutput:
    pipeline = make()
    blocks: queue.Queue[BlockRecord | None] = queue.Queue()
    chunks: dict[int, npt.NDArray[np.float32]] = {}
    t0 = time.perf_counter() + 0.1

    def produce() -> None:
        for start in range(0, len(audio), block):
            end = min(start + block, len(audio))
            # block 最後一個樣本「擷取」完的時刻才送出
            delay = t0 + end / SAMPLE_RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            chunks[start] = audio[start:end].copy()
            blocks.put(BlockRecord(start, end, time.perf_counter(), t0 + start / SAMPLE_RATE))
        blocks.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    log = WallLog()
    recs = consume(pipeline, blocks, chunks, log)
    producer.join()
    return RunOutput(pipeline.utterances, t0, log, recs)


def run_portaudio(
    audio: npt.NDArray[np.float32],
    block: int,
    make: Callable[[], StreamingPipeline],
    device: int | str | None,
) -> RunOutput:
    pipeline = make()
    blocks: queue.Queue[BlockRecord | None] = queue.Queue()
    chunks: dict[int, npt.NDArray[np.float32]] = {}
    stream_times: list[tuple[float, float, float]] = []  # (牆鐘, currentTime, inputBufferAdcTime)
    pos = 0
    overflows = 0
    finished = False

    def callback(indata: npt.NDArray[np.float32], frames: int, time_info: Any, status: Any) -> None:
        nonlocal pos, overflows, finished
        arrival = time.perf_counter()
        if status.input_overflow:
            overflows += 1
        if finished:
            return
        stream_times.append((arrival, time_info.currentTime, time_info.inputBufferAdcTime))
        # 丟棄麥克風的聲音，改送檔案音訊（同樣長度）
        end = min(pos + frames, len(audio))
        chunks[pos] = audio[pos:end].copy()
        blocks.put(BlockRecord(pos, end, arrival, time_info.inputBufferAdcTime))
        pos = end
        if pos >= len(audio):
            finished = True
            blocks.put(None)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=block,
        device=device,
        channels=1,
        dtype="float32",
        latency="low",
        callback=callback,
    )
    log = WallLog()
    with stream:
        recs = consume(pipeline, blocks, chunks, log)
    offset = clock_offset([w for w, _, _ in stream_times], [c for _, c, _ in stream_times])
    for r in recs:
        r.adc += offset
    t0 = adc_t0([r.adc for r in recs], [r.start for r in recs])
    info = {
        "device": sd.query_devices(stream.device)["name"],
        "portaudio_input_latency_ms": stream.latency * 1000,
        "clock_offset_us": offset * 1e6,
        "callback_minus_adc_ms": summarize([(w - (a + offset)) * 1000 for w, _, a in stream_times]),
        "overflows": overflows,
    }
    return RunOutput(pipeline.utterances, t0, log, recs, info)


# ----- 彙整 -----


def per_file(
    spans: Sequence[FileSpan], segs: Sequence[SegmentTimes]
) -> list[dict[str, float | None]]:
    return [
        file_latencies(sp, s) for sp, s in zip(spans, assign_segments(spans, segs), strict=True)
    ]


def column(rows: Sequence[dict[str, float | None]], key: str) -> list[float]:
    return [v for r in rows if (v := r[key]) is not None]


def summarize_run(
    name: str, out: RunOutput, spans: Sequence[FileSpan], block: int
) -> dict[str, Any]:
    if out.log is None:
        files = per_file(spans, segments_simulated(out.utterances))
        internal = None
    else:
        files = per_file(spans, segments_wall(out.utterances, out.log, out.t0))
        internal = per_file(spans, segments_simulated(out.utterances))
    res: dict[str, Any] = {
        "name": name,
        "blocksize": block,
        "hyps": [
            " ".join(u.text for u in segs if u.text)
            for segs in assign_segments(spans, segments_simulated(out.utterances))
        ],
        "files": files,
        "metrics": {k: summarize(column(files, k)) for k in METRICS},
        "info": out.info,
        "compute_s": out.compute_s or sum(r.processed - r.dequeued for r in out.blocks),
    }
    if internal is not None:
        res["internal_files"] = internal
        # 同一次執行中，牆鐘時刻 − pipeline 自己的模擬時刻：交付延遲 + queue 等待造成的落後
        res["wall_minus_internal_ms"] = {
            k: summarize(
                [
                    w - s
                    for a, b in zip(files, internal, strict=True)
                    if (w := a[k]) is not None and (s := b[k]) is not None
                ]
            )
            for k in METRICS
        }
        recs = out.blocks
        res["blocks"] = {
            "n": len(recs),
            # block 最後一個樣本擷取完 → 放進 queue（驅動 / callback 或回放執行緒的交付延遲）
            "delivery_ms": summarize(
                [(r.arrival - (out.t0 + r.end / SAMPLE_RATE)) * 1000 for r in recs]
            ),
            "queue_wait_ms": summarize([(r.dequeued - r.arrival) * 1000 for r in recs]),
            "process_ms": summarize([(r.processed - r.dequeued) * 1000 for r in recs]),
            # demo_mic 的 t0 估計比 ADC 推得的 t0 晚多少（demo_mic 顯示的延遲就少算這麼多）
            "demo_mic_t0_bias_ms": (
                demo_mic_t0([r.arrival for r in recs], [r.end for r in recs]) - out.t0
            )
            * 1000,
        }
    return res


def paired(a: dict[str, Any], b: dict[str, Any], key: str) -> dict[str, float | int | None]:
    """同一句 b − a（ms）。"""
    return summarize(
        [
            y - x
            for ra, rb in zip(a["files"], b["files"], strict=True)
            if (x := ra[key]) is not None and (y := rb[key]) is not None
        ]
    )


def load_snapshot() -> dict[str, Any]:
    top = subprocess.run(
        ["ps", "-Ao", "pcpu,comm", "-r"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[1:6]
    return {"loadavg": os.getloadavg(), "top_cpu": [x.strip() for x in top]}


def environment() -> dict[str, Any]:
    def cmd(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "commit": cmd("git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"),
        "dirty": bool(
            cmd("git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no")
        ),
        "chip": cmd("sysctl", "-n", "machdep.cpu.brand_string"),
        "model": cmd("sysctl", "-n", "hw.model"),
        "macos": cmd("sw_vers", "-productVersion"),
        "cpu_count": os.cpu_count(),
    }


def fmt(s: dict[str, Any]) -> str:
    if not s or s.get("n", 0) == 0:
        return "—"
    return f"{s['mean']:.0f} / {s['p50']:.0f} / {s['p90']:.0f}"


def print_table(results: Sequence[dict[str, Any]]) -> None:
    head = "| 路徑 | block | " + " | ".join(METRIC_NAMES[k] for k in METRICS) + " |"
    print("\n延遲 mean / P50 / P90（ms）")
    print(head)
    print("|" + "---|" * (len(METRICS) + 2))
    for r in results:
        cells = " | ".join(fmt(r["metrics"][k]) for k in METRICS)
        print(f"| {r['name']} | {r['blocksize']} | {cells} |")
    print("\n真實時鐘路徑的拆解（ms，mean / P50 / P90）")
    print(
        "| 路徑 | block | 交付延遲 | queue 等待 | 每 block 處理 | demo_mic t0 少算 | 首字 牆鐘 − 內部模擬 | 完整結果 牆鐘 − 內部模擬 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if "blocks" not in r:
            continue
        b = r["blocks"]
        w = r["wall_minus_internal_ms"]
        print(
            f"| {r['name']} | {r['blocksize']} | {fmt(b['delivery_ms'])} | {fmt(b['queue_wait_ms'])} | "
            f"{fmt(b['process_ms'])} | {b['demo_mic_t0_bias_ms']:.1f} | {fmt(w['first_token_vad_ms'])} | "
            f"{fmt(w['result_ms'])} |"
        )


def main() -> int:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0
    if "portaudio" in args.modes and not args.allow_mic:
        print("錯誤：portaudio 路徑會開啟麥克風，需使用者同意後加 --allow-mic", file=sys.stderr)
        return 1
    device = (
        None
        if args.device is None
        else (int(args.device) if args.device.isdigit() else args.device)
    )
    config: PipelineConfig = pipeline_config_from_args(args)
    if config.vad.min_silence_ms >= args.tail_ms:
        print("錯誤：--tail-ms 需大於 min_silence，句子才會以靜音收尾", file=sys.stderr)
        return 1
    audio, spans = load_reference(min(args.n, 300), args.lead_ms, args.tail_ms)
    vad_model = SileroVadModel(num_threads=args.threads)
    asr: StreamingAsr = load_asr(args.model, num_threads=args.threads)

    def make() -> StreamingPipeline:
        return StreamingPipeline(vad_model, asr, config)

    # 暖機：第一句跑一次
    run_sim(audio[: int(spans[0].region_end_s * SAMPLE_RATE)], 160, make)
    real_s = len(audio) / SAMPLE_RATE
    print(
        f"{len(spans)} 句，串流 {real_s:.0f} s；真實路徑每個 block 大小約 {real_s / 60:.1f} 分鐘",
        file=sys.stderr,
    )

    results: list[dict[str, Any]] = []
    base_block = args.chunk_ms * SAMPLE_RATE // 1000
    if "sim" in args.modes and base_block not in args.blocksize:
        results.append(summarize_run("sim", run_sim(audio, base_block, make), spans, base_block))
    for block in args.blocksize:
        for mode in args.modes:
            before = load_snapshot()
            t = time.perf_counter()
            if mode == "sim":
                out = run_sim(audio, block, make)
            elif mode == "thread":
                out = run_thread(audio, block, make)
            else:
                out = run_portaudio(audio, block, make, device)
            res = summarize_run(mode, out, spans, block)
            res["load_before"] = before
            res["load_after"] = load_snapshot()
            res["elapsed_s"] = time.perf_counter() - t
            results.append(res)
            print(
                f"  {mode} block {block}: {res['elapsed_s']:.0f} s，load {before['loadavg'][0]:.2f}",
                file=sys.stderr,
            )

    # 與同 block 的 sim 逐句成對比較
    sims = {r["blocksize"]: r for r in results if r["name"] == "sim"}
    for r in results:
        s = sims.get(r["blocksize"])
        if r["name"] != "sim" and s is not None:
            r["minus_sim_ms"] = {k: paired(s, r, k) for k in METRICS}
            # 真實路徑中 pipeline 自己的模擬時刻 − sim：同樣的計算在即時節奏下變慢多少（與交付延遲無關）
            internal = {"files": r["internal_files"]}
            r["internal_minus_sim_ms"] = {k: paired(s, internal, k) for k in METRICS}
            r["hyp_diff_vs_sim"] = sum(a != b for a, b in zip(s["hyps"], r["hyps"], strict=True))

    print_table(results)
    print("\n真實 − 同 block 的 sim（逐句成對，ms，mean / P50 / P90）；辨識結果不同的句數")
    for r in results:
        if "minus_sim_ms" in r:
            cells = "，".join(f"{METRIC_NAMES[k]} {fmt(r['minus_sim_ms'][k])}" for k in METRICS)
            print(f"  {r['name']} {r['blocksize']}：{cells}；結果不同 {r['hyp_diff_vs_sim']} 句")
            cells = "，".join(
                f"{METRIC_NAMES[k]} {fmt(r['internal_minus_sim_ms'][k])}" for k in METRICS
            )
            print(f"    其中 內部模擬 − sim（計算變慢）：{cells}")
    print(
        "\n總計算時間（s）："
        + "，".join(f"{r['name']} {r['blocksize']} {r['compute_s']:.2f}" for r in results)
    )
    for r in results:
        if r["info"]:
            i = r["info"]
            print(
                f"  portaudio {r['blocksize']}：裝置 {i['device']}，PortAudio 回報輸入延遲 "
                f"{i['portaudio_input_latency_ms']:.0f} ms，callback − ADC {fmt(i['callback_minus_adc_ms'])}，"
                f"時鐘偏移 {i['clock_offset_us']:.0f} µs，溢位 {i['overflows']}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RESULTS_DIR / f"{date.today().isoformat()}_mic-latency_n{len(spans)}{tag}.json"
    payload = {
        "config": {**vars(args), "pipeline": asdict(config), "asr_name": asr.name, "n": len(spans)},
        "environment": environment(),
        "spans": [asdict(s) for s in spans],
        "results": results,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    print(f"\n結果：{path.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
