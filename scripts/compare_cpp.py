"""C++ 辨識核心（cpp/）與 Python 參考實作的對齊驗證。

兩邊用同一批音訊、同一組預設設定（SenseVoice 偽串流 p480-960-max3-zh、A2 兩段門檻、2 執行緒、
10 ms chunk 模擬即時）各跑一次，逐項比較：
- VAD 切點（每句的 speech_start_s / speech_end_s，單位 frame = 32 ms）；
- 每句的 partial 序列、暫定結果序列、定稿文字（逐字）；
- 整個事件序列（類型、句子編號、文字、事件發出時已送入的音訊位置）；
- 事件的模擬即時時刻 emitted_s（含實測計算時間，兩邊必然不同，只報差距）；
- 語料級 CER（兩邊各算一次）、RTF、峰值 RSS（各在獨立程序量測）。

用法：
    cmake -S cpp -B cpp/build -DORT_ROOT=third_party/onnxruntime-osx-arm64-1.31.1 && cmake --build cpp/build -j
    uv run scripts/compare_cpp.py                       # fixture + 200 句冒煙集
    uv run scripts/compare_cpp.py --set fixture
    uv run scripts/compare_cpp.py --set smoke --limit 20

冒煙集沿用 eval/run_eval.py 的設定：每條音訊前補 1000 ms、後補 1500 ms 靜音，前 3 句暖機不計入。
結果寫到 eval/results/cpp_compare_<集合>.json（不進 git）。只跑小集合，不要拿 3000 句來跑。
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CLI = REPO_ROOT / "cpp" / "build" / "ozasr_cli"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "vad_two_utterances.wav"
SMOKE_MANIFEST = REPO_ROOT / "data" / "aishell1_test_200" / "manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SAMPLE_RATE = 16000
FRAME_S = 0.032

# 各集合的設定：（補前靜音 ms, 補後靜音 ms, 暖機句數）
SETS = {"fixture": (0, 0, 0), "smoke": (1000, 1500, 3)}


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


# ---------- Python 參考實作（獨立子程序，量測才不受本程序影響） ----------


def py_worker(spec_path: Path, out_path: Path) -> None:
    import numpy as np

    from src.asr import DEFAULT_MODEL, load_asr
    from src.pipeline import PipelineConfig, PipelineEvent, StreamingPipeline
    from src.vad import SileroVadModel, load_wav

    spec = json.loads(spec_path.read_text())
    threads, chunk_ms = spec["threads"], spec["chunk_ms"]
    lead = spec["lead_ms"] * SAMPLE_RATE // 1000
    tail = spec["tail_ms"] * SAMPLE_RATE // 1000
    chunk = chunk_ms * SAMPLE_RATE // 1000

    vad_model = SileroVadModel(num_threads=threads)
    asr = load_asr(DEFAULT_MODEL, num_threads=threads)
    config = PipelineConfig()

    def run(path: str) -> dict[str, Any]:
        audio = load_wav(path)
        audio = np.concatenate([np.zeros(lead, np.float32), audio, np.zeros(tail, np.float32)])
        pipeline = StreamingPipeline(vad_model, asr, config)
        events: list[dict[str, Any]] = []
        seen: dict[tuple[str, int], int] = {}

        def collect(evs: list[PipelineEvent], pos: int) -> None:
            for ev in evs:
                u = ev.utterance
                k = seen.get((ev.kind, u.index), 0)
                seen[(ev.kind, u.index)] = k + 1
                d: dict[str, Any] = {
                    "type": ev.kind,
                    "utterance": u.index,
                    "audio_pos_s": pos / SAMPLE_RATE,
                    "text": "",
                    "emitted_s": u.start_emitted_s,
                    "audio_end_s": None,
                }
                if ev.kind == "partial":
                    d["text"], d["emitted_s"] = u.partials[k].text, u.partials[k].emitted_s
                elif ev.kind == "provisional":
                    p = u.provisionals[k]
                    d["text"], d["emitted_s"], d["audio_end_s"] = p.text, p.emitted_s, p.audio_end_s
                elif ev.kind == "final":
                    d["text"], d["emitted_s"] = u.text, u.final_s
                events.append(d)

        t0 = time.perf_counter()
        pos = 0
        for i in range(0, len(audio), chunk):
            piece = audio[i : i + chunk]
            pos += len(piece)
            collect(pipeline.accept_waveform(piece), pos)
        collect(pipeline.flush(), pos)
        compute = time.perf_counter() - t0
        return {
            "wav": path,
            "duration_s": len(audio) / SAMPLE_RATE,
            "compute_s": compute,
            "events": events,
            "utterances": [
                {
                    "index": u.index,
                    "text": u.text,
                    "speech_start_s": u.speech_start_s,
                    "speech_end_s": u.speech_end_s,
                    "audio_start_s": u.audio_start_s,
                }
                for u in pipeline.utterances
            ],
        }

    rss_loaded = peak_rss_mb()
    results = []
    for i, path in enumerate(spec["files"]):
        r = run(path)
        if i == spec["warmup"] - 1:
            asr.reset_compute_stats()
        if i >= spec["warmup"]:
            results.append(r)
    audio_s = sum(r["duration_s"] for r in results)
    compute_s = sum(r["compute_s"] for r in results)
    stats = asr.compute_stats() or {}
    out = {
        "files": results,
        "summary": {
            "files": len(results),
            "audio_s": audio_s,
            "compute_s": compute_s,
            "rtf": compute_s / audio_s,
            "rss_after_load_mb": rss_loaded,
            "peak_rss_mb": peak_rss_mb(),
            "num_decodes": stats.get("num_decodes"),
            "decoded_audio_s": stats.get("decoded_audio_s"),
        },
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False))


def run_python(spec: dict[str, Any], work: Path) -> dict[str, Any]:
    spec_path, out_path = work / "py_spec.json", work / "py_out.json"
    spec_path.write_text(json.dumps(spec))
    subprocess.run(
        [sys.executable, str(Path(__file__)), "--py-worker", str(spec_path), str(out_path)],
        check=True,
    )
    out: dict[str, Any] = json.loads(out_path.read_text())
    return out


# ---------- C++ ----------


def run_cpp(spec: dict[str, Any], work: Path, cli: Path) -> dict[str, Any]:
    list_path = work / "files.txt"
    list_path.write_text("\n".join(spec["files"]) + "\n")
    cmd = [
        str(cli),
        "--models-dir",
        str(REPO_ROOT / "models"),
        "--list",
        str(list_path),
        "--warmup",
        str(spec["warmup"]),
        "--lead-silence-ms",
        str(spec["lead_ms"]),
        "--tail-silence-ms",
        str(spec["tail_ms"]),
        "--chunk-ms",
        str(spec["chunk_ms"]),
        "--threads",
        str(spec["threads"]),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    files, summary = [], {}
    for line in proc.stdout.splitlines():
        d = json.loads(line)
        if d["type"] == "file":
            files.append(d)
        elif d["type"] == "summary":
            summary = d
    if not summary:  # 單一檔案沒有 summary 行
        audio_s = sum(f["duration_s"] for f in files)
        compute_s = sum(f["compute_s"] for f in files)
        summary = {"files": len(files), "audio_s": audio_s, "compute_s": compute_s}
        summary["rtf"] = compute_s / audio_s
    return {"files": files, "summary": summary}


# ---------- 比較 ----------


def ev_key(e: dict[str, Any]) -> tuple[Any, ...]:
    return (e["type"], e["utterance"], e["text"], round(e["audio_pos_s"] * SAMPLE_RATE))


def texts(events: list[dict[str, Any]], kind: str, utt: int) -> list[str]:
    return [e["text"] for e in events if e["type"] == kind and e["utterance"] == utt]


def compare_file(py: dict[str, Any], cpp: dict[str, Any]) -> dict[str, Any]:
    """單一檔案的差異。"""
    pu, cu = py["utterances"], cpp["utterances"]
    out: dict[str, Any] = {
        "wav": py["wav"],
        "num_utt": (len(pu), len(cu)),
        "vad_diff_frames": 0.0,
        "partial_diff": [],
        "provisional_diff": [],
        "final_diff": [],
        "event_seq_equal": [ev_key(e) for e in py["events"]] == [ev_key(e) for e in cpp["events"]],
    }
    for a, b in zip(pu, cu, strict=False):
        for k in ("speech_start_s", "speech_end_s", "audio_start_s"):
            out["vad_diff_frames"] = max(
                out["vad_diff_frames"], abs(a[k] - b[k]) / FRAME_S if b[k] is not None else 1e9
            )
        i = a["index"]
        if texts(py["events"], "partial", i) != texts(cpp["events"], "partial", i):
            out["partial_diff"].append(
                (i, texts(py["events"], "partial", i), texts(cpp["events"], "partial", i))
            )
        if texts(py["events"], "provisional", i) != texts(cpp["events"], "provisional", i):
            out["provisional_diff"].append(
                (i, texts(py["events"], "provisional", i), texts(cpp["events"], "provisional", i))
            )
        if a["text"] != b["text"]:
            out["final_diff"].append((i, a["text"], b["text"]))
    if len(pu) != len(cu):
        out["vad_diff_frames"] = 1e9
    return out


def time_deltas(py: dict[str, Any], cpp: dict[str, Any]) -> list[float]:
    """事件序列一致時，逐事件的 emitted_s 差（C++ − Python，秒）。"""
    if [ev_key(e) for e in py["events"]] != [ev_key(e) for e in cpp["events"]]:
        return []
    return [
        c["emitted_s"] - p["emitted_s"] for p, c in zip(py["events"], cpp["events"], strict=True)
    ]


def cer_of(files: list[dict[str, Any]], refs: list[str]) -> tuple[float, int, int]:
    from eval.metrics import corpus_cer

    hyps = [" ".join(u["text"] for u in f["utterances"] if u["text"]) for f in files]
    r = corpus_cer(refs, hyps)
    return r.cer, r.errors, r.ref_tokens


def report(name: str, py: dict[str, Any], cpp: dict[str, Any], refs: list[str] | None) -> None:
    diffs = [compare_file(p, c) for p, c in zip(py["files"], cpp["files"], strict=True)]
    n = len(diffs)
    deltas = [d for p, c in zip(py["files"], cpp["files"], strict=True) for d in time_deltas(p, c)]

    def count(pred: Any) -> int:
        return sum(1 for d in diffs if pred(d))

    n_utt_py = sum(len(f["utterances"]) for f in py["files"])
    n_utt_cpp = sum(len(f["utterances"]) for f in cpp["files"])
    vad_worst = max((d["vad_diff_frames"] for d in diffs), default=0.0)
    print(f"\n===== {name}：{n} 條音訊，句段數 Python {n_utt_py} / C++ {n_utt_cpp} =====")
    rows = [
        ("句段數不同的音訊", count(lambda d: d["num_utt"][0] != d["num_utt"][1])),
        (
            "VAD 切點差 > 1 frame 的音訊",
            count(lambda d: d["vad_diff_frames"] > 1.0),
        ),
        ("partial 序列不同的句段", sum(len(d["partial_diff"]) for d in diffs)),
        ("暫定結果不同的句段", sum(len(d["provisional_diff"]) for d in diffs)),
        ("定稿文字不同的句段", sum(len(d["final_diff"]) for d in diffs)),
        ("事件序列（含音訊位置）不同的音訊", count(lambda d: not d["event_seq_equal"])),
    ]
    for label, v in rows:
        print(f"  {label:<28}{v:>6}")
    print(f"  VAD 切點最大差距: {vad_worst:.3f} frame（{vad_worst * FRAME_S * 1000:.1f} ms）")
    if deltas:
        mean_ms = statistics.mean(deltas) * 1000
        med_ms = statistics.median(deltas) * 1000
        abs_ms = statistics.mean(abs(d) for d in deltas) * 1000
        print(
            f"  事件 emitted_s（C++ − Python，{len(deltas)} 個事件）：中位數 {med_ms:+.1f} ms，"
            f"平均 {mean_ms:+.1f} ms，平均絕對 {abs_ms:.1f} ms（含實測計算時間，預期不同）"
        )
    shown = 0
    for d in diffs:
        for kind in ("partial_diff", "provisional_diff", "final_diff"):
            for item in d[kind]:
                if shown < 8:
                    print(
                        f"  例 [{Path(d['wav']).name} {kind} 句 {item[0]}] Python={item[1]!r} C++={item[2]!r}"
                    )
                    shown += 1
        if (
            not d["event_seq_equal"]
            and shown < 8
            and not any(d[k] for k in ("partial_diff", "provisional_diff", "final_diff"))
        ):
            print(f"  例 [{Path(d['wav']).name}] 事件序列不同（文字相同，音訊位置不同）")
            shown += 1

    if refs is not None:
        cer_py, e_py, n_ref = cer_of(py["files"], refs)
        cer_cpp, e_cpp, _ = cer_of(cpp["files"], refs)
        print(
            f"  CER: Python {cer_py * 100:.2f}%（{e_py}/{n_ref}） C++ {cer_cpp * 100:.2f}%（{e_cpp}/{n_ref}）"
        )
    ps, cs = py["summary"], cpp["summary"]
    print("  | | Python | C++ |\n  |---|---|---|")
    print(f"  | RTF | {ps['rtf']:.4f} | {cs['rtf']:.4f} |")
    print(f"  | 載入後 RSS (MB) | {ps['rss_after_load_mb']:.0f} | {cs['rss_after_load_mb']:.0f} |")
    print(f"  | 峰值 RSS (MB) | {ps['peak_rss_mb']:.0f} | {cs['peak_rss_mb']:.0f} |")
    if ps.get("num_decodes") is not None and cs.get("num_decodes") is not None:
        print(
            f"  | 整段推論次數 / 音訊總長 (s) | {ps['num_decodes']} / {ps['decoded_audio_s']:.0f}"
            f" | {cs['num_decodes']} / {cs['decoded_audio_s']:.0f} |"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"cpp_compare_{name}.json").write_text(
        json.dumps(
            {"diffs": diffs, "python": py["summary"], "cpp": cpp["summary"]},
            ensure_ascii=False,
            indent=1,
        )
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--set", nargs="+", choices=sorted(SETS), default=list(SETS), help="要跑的集合")
    p.add_argument("--limit", type=int, help="冒煙集只跑前 N 句（除錯用）")
    p.add_argument("--cli", type=Path, default=DEFAULT_CLI, help="C++ 命令列工具路徑")
    p.add_argument("--threads", type=int, default=2, help="兩邊的 ASR / VAD 執行緒數")
    p.add_argument("--chunk-ms", type=int, default=10)
    p.add_argument(
        "--py-worker", nargs=2, type=Path, metavar=("SPEC", "OUT"), help=argparse.SUPPRESS
    )
    args = p.parse_args()
    if args.py_worker:
        py_worker(*args.py_worker)
        return 0
    if not args.cli.is_file():
        print(f"錯誤：找不到 {args.cli}，請先建置 C++（見 README 快速開始）", file=sys.stderr)
        return 1

    for name in args.set:
        lead, tail, warmup = SETS[name]
        refs: list[str] | None = None
        if name == "fixture":
            files = [str(FIXTURE)]
        else:
            rows = [json.loads(x) for x in SMOKE_MANIFEST.read_text(encoding="utf-8").splitlines()]
            if args.limit:
                rows = rows[: args.limit + warmup]
            files = [str(SMOKE_MANIFEST.parent / r["wav"]) for r in rows]
            refs = [r["text"] for r in rows[warmup:]]
        spec = {
            "files": files,
            "lead_ms": lead,
            "tail_ms": tail,
            "warmup": warmup,
            "threads": args.threads,
            "chunk_ms": args.chunk_ms,
        }
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            print(f"[{name}] 執行 Python 參考實作……", file=sys.stderr)
            py = run_python(spec, work)
            print(f"[{name}] 執行 C++ ……", file=sys.stderr)
            cpp = run_cpp(spec, work, args.cli)
        report(name, py, cpp, refs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
