"""斷句 / 尾字延遲診斷（backlog B3、B9）：何時判定一句話結束。只量測，不改演算法。

三部分：
1. 句尾時間線（對齊集）：以 MFA 的真實末字結束點為 0，比較 VAD 終點、VAD 結束事件、最終文字就緒
   （最終文字第一次出現在 partial）、最終結果的時刻。需要 run_eval 在對齊集上、後補靜音的結果。
2. 不 finalize 的對照（對齊集）：整條音訊 + 後補靜音送入同一個 ASR stream（不經 VAD、不 finalize），
   記錄 partial 何時變成完整句子、何時不再變化。回答「只靠持續送靜音，partial 要多久才完整」。
   時刻以音訊到達時間計（不含計算時間，chunk 10 ms）。
3. 停頓分佈：
   - 句中停頓：AISHELL 每條都是單句，VAD 以最細設定（min_silence / min_speech / pad = 0）
     切出的語音段之間的靜音都是句中停頓。以此預測各 min_silence 下會被切開的比例。
   - 實際切點：run_eval 結果中被切成多段的句子，切點的靜音長度（相鄰句段間距 + 2 × pad）。
   - min_silence 掃描：同一批 3000 句在各 min_silence 下的成對 CER 差，分被切開 / 未切開兩組。
   - 句間停頓：只有自錄音訊（data/my_voice*.wav）的少數樣本，列出 VAD 間隔與各段文字供人工判讀。

需要：
    uv run scripts/prepare_alignment.py
    uv run eval/run_eval.py                                                  # 3000 句（第 3 部分）
    uv run eval/run_eval.py --min-silence-ms 300 800 1000 1500 --jobs 2      # 掃描（可選）
    uv run eval/run_eval.py --manifest data/aishell1_align_300/manifest.jsonl \\
        --model x-asr-480ms-int8 x-asr-160ms-int8                            # 對齊集（第 1 部分）
用法：
    uv run scripts/endpoint_diagnosis.py     # 印出表格，數字存 eval/results/<日期>_endpoint_diagnosis.json
"""

import argparse
import itertools
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from eval.metrics import corpus_cer, summarize
from src.asr import SAMPLE_RATE, load_asr
from src.vad import SileroVadModel, VadConfig, detect_segments, load_wav

MODELS = ["x-asr-480ms-int8", "x-asr-160ms-int8"]
ALIGN_DIR = REPO_ROOT / "data" / "aishell1_align_300"
MAIN_MANIFEST = REPO_ROOT / "data" / "aishell1_test_3000" / "manifest.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SELF_RECORDED = [REPO_ROOT / "data" / "my_voice.wav", REPO_ROOT / "data" / "my_voice_script.wav"]
THRESHOLDS_MS = [200, 300, 400, 500, 600, 800, 1000, 1500]
SWEEP_MS = [300, 500, 800, 1000, 1500]
BOOTSTRAP = 10000
FINEST_VAD = VadConfig(min_speech_ms=0, min_silence_ms=0, speech_pad_ms=0)


def latest_result(dataset: str, model: str, config: str) -> Path:
    files = sorted(RESULTS_DIR.glob(f"*_{dataset}_{model}_{config}.json"))
    if not files:
        raise FileNotFoundError(f"找不到 {RESULTS_DIR}/*_{dataset}_{model}_{config}.json")
    return files[-1]


def ms(values: list[float]) -> dict[str, float | int | None]:
    out = summarize([v * 1000 for v in values])
    if values:
        out["p10"] = float(np.percentile(values, 10) * 1000)
    return out


def fmt(s: dict[str, Any]) -> str:
    if not s["n"]:
        return "-"
    return f"{s['mean']:.0f} / {s['p10']:.0f} / {s['p50']:.0f} / {s['p90']:.0f}"


# ---------- 1. 句尾時間線 ----------


def tail_timeline(result: dict[str, Any], align: dict[str, Any]) -> dict[str, Any]:
    cols: dict[str, list[float]] = {
        k: [] for k in ("vad_end", "end_event", "text_ready", "final", "ready_to_final")
    }
    ready_before_event = 0
    skipped = 0
    for u in result["utterances"]:
        a = align.get(u["id"])
        segs = [s for s in u["segments"] if s["text"].strip()]
        if a is None or not segs or segs[-1].get("final_s") is None:
            skipped += 1
            continue
        seg, true_end = segs[-1], a["chars"][-1]["end"]
        cols["vad_end"].append(seg["speech_end_s"] - true_end)
        cols["end_event"].append(seg["end_emitted_s"] - true_end)
        cols["text_ready"].append(seg["final_text_ready_s"] - true_end)
        cols["final"].append(seg["final_s"] - true_end)
        cols["ready_to_final"].append(seg["final_s"] - seg["final_text_ready_s"])
        ready_before_event += seg["final_text_ready_s"] < seg["end_emitted_s"]
    return {
        "n": len(cols["final"]),
        "skipped": skipped,
        "ready_before_end_event": ready_before_event,
        **{k: ms(v) for k, v in cols.items()},
    }


# ---------- 2. 不 finalize 的對照 ----------


def no_finalize(model: str, align: dict[str, Any], tail_s: float, chunk_ms: int) -> dict[str, Any]:
    asr = load_asr(model)
    rows = [json.loads(line) for line in (ALIGN_DIR / "manifest.jsonl").read_text().splitlines()]
    size = chunk_ms * SAMPLE_RATE // 1000
    complete: list[float] = []  # partial 第一次等於 finalize 結果的時刻（相對真實末字結束）
    last_change: list[float] = []  # partial 最後一次變化的時刻
    missing_at: dict[int, list[int]] = {500: [], 1000: []}  # 末字結束後 N ms 時 partial 少幾個字
    never = 0  # 送完後補靜音 partial 仍不等於 finalize 結果
    for r in rows:
        audio = load_wav(ALIGN_DIR / r["wav"])
        audio = np.concatenate([audio, np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)])
        true_end = align[r["id"]]["chars"][-1]["end"]
        stream = asr.create_stream()
        timeline: list[tuple[float, str]] = []
        for i in range(0, len(audio), size):
            stream.accept_waveform(audio[i : i + size])
            text = stream.get_partial_result()
            if not timeline or text != timeline[-1][1]:
                timeline.append((min(i + size, len(audio)) / SAMPLE_RATE, text))
        final = stream.finalize()
        hit = next((t for t, text in timeline if text == final), None)
        if hit is None:
            never += 1
        else:
            complete.append(hit - true_end)
        last_change.append(timeline[-1][0] - true_end)
        for delay, out in missing_at.items():
            t = true_end + delay / 1000
            text = next((x for tt, x in reversed(timeline) if tt <= t), "")
            out.append(len(final) - len(text))
    return {
        "n": len(rows),
        "tail_silence_s": tail_s,
        "never_complete": never,
        "complete": ms(complete),
        "last_change": ms(last_change),
        "missing_chars": {
            f"{d}ms": {
                "mean": float(np.mean(v)),
                "zero": sum(x == 0 for x in v),
                "hist": {str(k): v.count(k) for k in sorted(set(v))},
            }
            for d, v in missing_at.items()
        },
    }


# ---------- 3. 停頓分佈 ----------


def inner_gaps(audio: npt.NDArray[np.float32], vad: SileroVadModel) -> list[float]:
    segs = detect_segments(audio, vad, FINEST_VAD)
    return [(b.start - a.end) / SAMPLE_RATE for a, b in itertools.pairwise(segs)]


def within_sentence_pauses(manifest: Path, vad: SileroVadModel) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    all_gaps: list[float] = []
    max_gap: list[float] = []
    for r in rows:
        gaps = inner_gaps(load_wav(manifest.parent / r["wav"]), vad)
        all_gaps.extend(gaps)
        max_gap.append(max(gaps, default=0.0))
    arr, mx = np.asarray(all_gaps), np.asarray(max_gap)
    return {
        "utterances": len(rows),
        "gaps": len(all_gaps),
        "gaps_ms": ms([g for g in all_gaps if g >= 0.1]),
        # 句中停頓 ≥ T 的個數，以及含至少一個 ≥ T 停頓的句子比例（≈ min_silence = T 時被切開的比例）
        "by_threshold": {
            str(t): {
                "gaps": int((arr >= t / 1000).sum()),
                "utterances_split": float((mx >= t / 1000).mean()),
            }
            for t in THRESHOLDS_MS
        },
    }


def split_points(result: dict[str, Any]) -> dict[str, Any]:
    pad = result["config"]["vad_speech_pad_ms"] / 1000
    gaps: list[float] = []
    split: list[dict[str, Any]] = []
    whole: list[dict[str, Any]] = []
    for u in result["utterances"]:
        segs = u["segments"]
        (split if len(segs) > 1 else whole).append(u)
        for a, b in itertools.pairwise(segs):
            gaps.append(b["speech_start_s"] - a["speech_end_s"] + 2 * pad)

    def cer(us: list[dict[str, Any]]) -> float | None:
        return corpus_cer([u["ref"] for u in us], [u["hyp"] for u in us]).cer if us else None

    return {
        "split_utterances": len(split),
        "total": len(result["utterances"]),
        "gaps_ms": ms(gaps),
        "cer_split": cer(split),
        "cer_whole": cer(whole),
    }


def silence_sweep(config: str) -> dict[str, Any] | None:
    """各 min_silence 相對 500 ms 的成對 CER 差（以句重抽），分「500 ms 時被切開」與「未切開」兩組。"""
    from eval.metrics import utterance_counts

    runs: dict[int, dict[str, Any]] = {}
    for sil in SWEEP_MS:
        try:
            path = latest_result(
                "aishell1_test_3000", MODELS[0], config.replace("sil500", f"sil{sil}")
            )
        except FileNotFoundError:
            continue
        runs[sil] = json.loads(path.read_text())
    if 500 not in runs or len(runs) < 2:
        return None
    ids = [u["id"] for u in runs[500]["utterances"]]
    errors: dict[int, npt.NDArray[np.int64]] = {}
    for sil, r in runs.items():
        us = r["utterances"]
        if [u["id"] for u in us] != ids:
            raise ValueError(f"sil{sil} 的句子與 sil500 不同")
        e, n = utterance_counts([u["ref"] for u in us], [u["hyp"] for u in us])
        errors[sil] = np.asarray(e)
    refs = np.asarray(n)
    split = np.array([len(u["segments"]) > 1 for u in runs[500]["utterances"]])
    rng = np.random.default_rng(0)

    def paired(sil: int, mask: npt.NDArray[np.bool_]) -> list[float]:
        idx = np.flatnonzero(mask)
        diff = errors[sil] - errors[500]
        boot = rng.choice(idx, size=(BOOTSTRAP, len(idx)))
        ci = np.percentile(diff[boot].sum(axis=1) / refs[boot].sum(axis=1), [2.5, 97.5])
        return [float(diff[idx].sum() / refs[idx].sum()), float(ci[0]), float(ci[1])]

    out: dict[str, Any] = {"split_at_500": int(split.sum()), "runs": {}}
    for sil, r in runs.items():
        m = r["metrics"]
        out["runs"][str(sil)] = {
            "cer": m["cer"],
            "split_rate": sum(len(u["segments"]) > 1 for u in r["utterances"]) / len(ids),
            "tail_latency_ms": m["tail_latency_ms"],
            "parallel_jobs": r["config"].get("parallel_jobs", 1),
            "vs_500_all": paired(sil, np.ones_like(split)),
            "vs_500_split": paired(sil, split),
            "vs_500_unsplit": paired(sil, ~split),
        }
    return out


def self_recorded(vad: SileroVadModel, model: str) -> list[dict[str, Any]]:
    from src.pipeline import PipelineConfig, transcribe

    asr = load_asr(model)
    out = []
    for path in SELF_RECORDED:
        if not path.is_file():
            continue
        audio = load_wav(path)
        segs = detect_segments(audio, vad, FINEST_VAD)
        # 文字用 min_silence 100 ms 切段的結果，方便判讀每個間隔是句間還是句中
        utts = transcribe(audio, vad, asr, PipelineConfig(vad=VadConfig(min_silence_ms=100)))
        out.append(
            {
                "file": path.name,
                "gaps": [
                    {
                        "at_s": round(a.end / SAMPLE_RATE, 2),
                        "ms": (b.start - a.end) * 1000 // SAMPLE_RATE,
                    }
                    for a, b in itertools.pairwise(segs)
                    if b.start - a.end >= 0.1 * SAMPLE_RATE
                ],
                "segments": [{"start_s": round(u.speech_start_s, 2), "text": u.text} for u in utts],
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--config", default="sil500_pre100_chunk10_t1_lead1000_tail1500")
    p.add_argument("--tail-s", type=float, default=2.0, help="第 2 部分後補的靜音（秒）")
    p.add_argument("--skip-no-finalize", action="store_true", help="略過第 2 部分（需跑 ASR）")
    args = p.parse_args()

    align = {
        (a := json.loads(line))["id"]: a
        for line in (ALIGN_DIR / "alignments.jsonl").read_text().splitlines()
    }
    report: dict[str, Any] = {"config": args.config, "tail_timeline": {}, "no_finalize": {}}

    print("== 1. 句尾時間線（對齊集，相對真實末字結束，ms：mean / P10 / P50 / P90）")
    for model in MODELS:
        res = json.loads(latest_result("aishell1_align_300", model, args.config).read_text())
        t = report["tail_timeline"][model] = tail_timeline(res, align)
        print(f"-- {model}（n={t['n']}，略過 {t['skipped']}）")
        for k, label in [
            ("vad_end", "VAD 終點（含 pad）"),
            ("end_event", "VAD 結束事件"),
            ("text_ready", "最終文字就緒"),
            ("final", "最終結果"),
            ("ready_to_final", "就緒 → 最終結果"),
        ]:
            print(f"   {label:<16} {fmt(t[k])}")
        print(f"   結束事件前已就緒  {t['ready_before_end_event']} / {t['n']}")

    if not args.skip_no_finalize:
        print("\n== 2. 不 finalize、持續送靜音（相對真實末字結束，ms：mean / P10 / P50 / P90）")
        for model in MODELS:
            nf = report["no_finalize"][model] = no_finalize(model, align, args.tail_s, 10)
            print(
                f"-- {model}（n={nf['n']}，補 {args.tail_s} s 後仍不完整 {nf['never_complete']}）"
            )
            print(f"   partial 變完整     {fmt(nf['complete'])}")
            print(f"   partial 最後變化   {fmt(nf['last_change'])}")
            for d, m in nf["missing_chars"].items():
                print(f"   末字後 {d} 少的字數 mean {m['mean']:.2f}，完整 {m['zero']}，{m['hist']}")

    vad = SileroVadModel()
    print("\n== 3a. 句中停頓（3000 句，VAD 最細設定）")
    w = report["within_sentence"] = within_sentence_pauses(MAIN_MANIFEST, vad)
    print(f"   停頓 ≥ 100 ms：{fmt(w['gaps_ms'])}（n={w['gaps_ms']['n']}）")
    for t, v in w["by_threshold"].items():
        print(
            f"   ≥ {t:>4} ms：{v['gaps']:>5} 個，含此停頓的句子 {v['utterances_split'] * 100:5.1f}%"
        )

    res = json.loads(latest_result("aishell1_test_3000", MODELS[0], args.config).read_text())
    s = report["split_points"] = split_points(res)
    print("\n== 3b. 實際切點（3000 句，預設設定）")
    print(f"   被切開 {s['split_utterances']} / {s['total']} 句；切點靜音 {fmt(s['gaps_ms'])}")
    print(f"   CER 被切開 {s['cer_split'] * 100:.2f}%  未切開 {s['cer_whole'] * 100:.2f}%")

    sw = report["silence_sweep"] = silence_sweep(args.config)
    if sw is not None:
        print(f"\n== 3d. min_silence 掃描（3000 句；500 ms 時被切開 {sw['split_at_500']} 句）")
        print("   CER 差為相對 500 ms 的成對差（百分點）[95% CI]")

        def pp(v: list[float]) -> str:
            return f"{v[0] * 100:+.2f} [{v[1] * 100:+.2f}, {v[2] * 100:+.2f}]"

        for sil, v in sw["runs"].items():
            t = v["tail_latency_ms"]
            print(
                f"   {sil:>4} ms：CER {v['cer'] * 100:.2f}%  切開 {v['split_rate'] * 100:4.1f}%"
                f"  尾字 {t['mean']:.0f} / {t['p90']:.0f}  jobs {v['parallel_jobs']}"
                f"  全部 {pp(v['vs_500_all'])}  被切開 {pp(v['vs_500_split'])}"
                f"  未切開 {pp(v['vs_500_unsplit'])}"
            )

    print("\n== 3c. 自錄音訊的停頓（VAD 最細設定，≥ 100 ms）")
    report["self_recorded"] = self_recorded(vad, MODELS[0])
    for f in report["self_recorded"]:
        print(f"-- {f['file']}")
        print("   間隔：" + "、".join(f"{g['at_s']}s {g['ms']}ms" for g in f["gaps"]))
        for seg in f["segments"]:
            print(f"   [{seg['start_s']:6.2f}] {seg['text']}")

    out = RESULTS_DIR / f"{date.today().isoformat()}_endpoint_diagnosis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n結果：{out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
