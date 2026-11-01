"""評測模組測試：文字正規化、語料級 CER 與信賴區間、分位數、端到端（小 manifest）。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.metrics import (
    bootstrap_cer_ci,
    corpus_cer,
    paired_bootstrap_cer_diff,
    summarize,
    utterance_counts,
)
from eval.run_eval import RunConfig, cer_confidence, summarize_run
from eval.text_norm import normalize, tokenize
from src.vad import DEFAULT_MODEL_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


# ---------- 正規化 ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("今天，天氣很好！", "今 天 天 气 很 好"),  # 去標點、繁轉簡
        ("楼市  调控  供  的", "楼 市 调 控 供 的"),  # AISHELL 標註的分詞空白
        ("ＡＢＣ１２３", "abc123"),  # 全形轉半形、轉小寫
        ("你好 CLOUD 你是谁", "你 好 cloud 你 是 谁"),  # 英文詞是一個 token
        ("hello,world", "hello world"),  # 標點換成空白，不黏詞
        ("「引號」（括號）《書名》……—~", "引 号 括 号 书 名"),
        ("１００％", "100"),  # 符號（S*）也去掉
        ("", ""),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_tokenize_ignores_whitespace_kinds() -> None:
    assert tokenize("中　文\t字\n") == ["中", "文", "字"]


# ---------- CER ----------


def test_corpus_cer_is_not_mean_of_sentence_cer() -> None:
    # 句 1：4 字錯 1；句 2：1 字錯 1。語料級 = 2/5，逐句平均 = 0.625
    res = corpus_cer(["今天天气", "好"], ["今天天汽", "坏"])
    assert (res.substitutions, res.deletions, res.insertions, res.ref_tokens) == (2, 0, 0, 5)
    assert res.cer == pytest.approx(0.4)


def test_corpus_cer_counts_deletions_and_insertions() -> None:
    res = corpus_cer(["我们去公园", "你好"], ["我们去公园散步", ""])
    assert (res.substitutions, res.deletions, res.insertions) == (0, 2, 2)
    assert res.ref_tokens == 7


def test_corpus_cer_applies_same_normalization() -> None:
    assert corpus_cer(["語音，辨識"], ["语音辨识。"]).errors == 0


def test_corpus_cer_rejects_empty_reference() -> None:
    with pytest.raises(ValueError):
        corpus_cer(["，。"], ["你好"])


def test_utterance_counts_sum_to_corpus_counts() -> None:
    refs = ["今天天气", "好", "我们去公园", "你好"]
    hyps = ["今天天汽", "坏", "我们去公园散步", ""]
    errors, n = utterance_counts(refs, hyps)
    assert errors == [1, 1, 2, 2] and n == [4, 1, 5, 2]
    total = corpus_cer(refs, hyps)
    assert (sum(errors), sum(n)) == (total.errors, total.ref_tokens)


def test_bootstrap_ci_contains_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    refs = rng.integers(5, 20, size=300)
    errors = rng.binomial(refs, 0.1)
    point = errors.sum() / refs.sum()
    lo, hi = bootstrap_cer_ci(list(errors), list(refs), n_resamples=2000)
    assert lo < point < hi and hi - lo < 0.05
    assert (lo, hi) == bootstrap_cer_ci(list(errors), list(refs), n_resamples=2000)


def test_bootstrap_ci_by_group_is_wider_when_groups_differ() -> None:
    # 10 位說話人，每人 30 句，CER 依說話人在 2%–20% 之間：同一說話人的句子高度相關
    refs, errors, groups = [], [], []
    for spk, rate in enumerate(np.linspace(0.02, 0.2, 10)):
        for _ in range(30):
            refs.append(10)
            errors.append(round(10 * rate))
            groups.append(f"S{spk:04d}")
    lo_u, hi_u = bootstrap_cer_ci(errors, refs, n_resamples=2000)
    lo_g, hi_g = bootstrap_cer_ci(errors, refs, groups, n_resamples=2000)
    assert hi_g - lo_g > 2 * (hi_u - lo_u)


def test_cer_confidence_groups_by_aishell_speaker() -> None:
    utts = [
        {
            "id": f"BAC009S07{6 + i % 2}4W{i:04d}",
            "ref": "今天天气",
            "hyp": "今天天汽" if i % 3 else "今天天气",
        }
        for i in range(12)
    ]
    ci = cer_confidence(utts)
    assert ci["num_speakers"] == 2 and ci["speaker"] is not None
    assert ci["utterance"][0] <= ci["utterance"][1]
    assert cer_confidence([{"id": "fx", "ref": "好", "hyp": "好"}])["speaker"] is None


def test_summarize() -> None:
    s = summarize([float(x) for x in range(1, 11)])
    assert s["n"] == 10 and s["mean"] == pytest.approx(5.5)
    assert s["p50"] == pytest.approx(5.5) and s["p90"] == pytest.approx(9.1)
    assert summarize([])["mean"] is None


# ---------- 彙整 ----------


def _cfg(manifest: str = "m.jsonl") -> RunConfig:
    return RunConfig(
        model="fake",
        manifest=manifest,
        threads=1,
        chunk_ms=10,
        pre_roll_ms=100,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=100,
        vad_speech_pad_ms=30,
        warmup=0,
    )


def _seg(text: str, ftl: float | None, tail: float | None) -> dict[str, object]:
    return {
        "text": text,
        "speech_start_s": 0.0,
        "speech_end_s": 1.0,
        "vad_start_delay_s": 0.3,
        "first_token_latency_s": ftl,
        "e2e_first_token_latency_s": None if ftl is None else ftl + 0.2,
        "tail_latency_s": tail,
    }


def test_run_config_name_marks_non_default_options() -> None:
    assert _cfg().name.endswith("_t1_lead1000")
    assert RunConfig(**{**_cfg().__dict__, "blank_penalty": 2.0}).name.endswith("_lead1000_bp2")
    tail = RunConfig(**{**_cfg().__dict__, "tail_silence_ms": 1500, "blank_penalty": 2.0})
    assert tail.name.endswith("_lead1000_tail1500_bp2")
    vad = RunConfig(**{**_cfg().__dict__, "vad_threshold": 0.6, "vad_neg_threshold": 0.5})
    assert vad.name.endswith("_lead1000_th0.6_neg0.5")
    hw = RunConfig(**{**_cfg().__dict__, "beam": 4, "hotwords": "/x/cabin.txt", "hotword_score": 2.0})
    assert hw.name.endswith("_lead1000_beam4_hw-cabin2")


def test_summarize_run_latency_uses_first_segment_with_text() -> None:
    raw = {
        "asr_name": "fake",
        "load_s": 0.0,
        "memory_mb": {"baseline_rss": 1.0, "after_load_rss": 2.0, "peak_rss": 3.0},
        "model_files": [{"path": "a.int8.onnx", "bytes": 2**20}, {"path": "t.txt", "bytes": 0}],
        "utterances": [
            {  # 第一段是沒有字的誤觸發，首字延遲取第二段；尾字延遲取最後一段
                "id": "a", "ref": "你好", "hyp": "你好", "duration_s": 2.0, "compute_s": 0.1,
                "segments": [_seg("", None, 0.1), _seg("你好", 0.5, 0.2)],
            },
            {
                "id": "b", "ref": "再见", "hyp": "", "duration_s": 2.0, "compute_s": 0.1,
                "segments": [],
            },
        ],
    }  # fmt: skip
    res = summarize_run(_cfg(), raw, {"commit": "x"})
    m = res["metrics"]
    assert m["cer"] == pytest.approx(0.5)
    assert m["rtf"] == pytest.approx(0.05)
    assert m["first_token_latency_ms"]["n"] == 1
    assert m["first_token_latency_ms"]["mean"] == pytest.approx(500)
    assert m["e2e_first_token_latency_ms"]["mean"] == pytest.approx(700)
    assert m["tail_latency_ms"]["mean"] == pytest.approx(200)
    assert m["no_text_utterances"] == 1
    assert m["segments_per_utterance"] == {"mean": 1.0, "max": 2, "histogram": {"0": 1, "2": 1}}
    assert m["model_size_mb"] == pytest.approx(1.0)
    assert res["model_files"][0]["precision"] == "INT8"


# ---------- 端到端 ----------


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason="需先執行 download_models.sh")
def test_run_eval_end_to_end(tmp_path: Path) -> None:
    wav = FIXTURES / "vad_two_utterances.wav"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "fx", "wav": str(wav), "text": "今天天氣很好。我們去公園散步吧。"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    out_dir = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval" / "run_eval.py"), "--manifest", str(manifest),
         "--warmup", "0", "--results-dir", str(out_dir)],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    (result_file,) = out_dir.glob("*.json")
    res = json.loads(result_file.read_text(encoding="utf-8"))
    m = res["metrics"]
    assert m["cer"] <= 0.15
    # fixture 兩句間隔 1.0 s，預設 min_silence 1000 ms 不切開，第一句以暫定結果先給出
    assert m["segments_per_utterance"]["max"] == 1
    assert 0 < m["first_token_latency_ms"]["mean"] < 2000
    assert res["config"]["lead_silence_ms"] == 1000 and res["config"]["tail_silence_ms"] == 1500
    assert result_file.stem.endswith("_lead1000_tail1500_pause150fin")  # 預設兩段門檻（A2，pause 150）
    (utt,) = res["utterances"]
    assert utt["processed_s"] == pytest.approx(utt["duration_s"] + 2.5, abs=1e-3)
    # 句尾補足靜音後，最後一句由 VAD 靜音計時收尾：尾字延遲 ≈ min_silence（預設 1000 ms）
    assert m["tail_latency_ms"]["mean"] >= 1000
    for seg in utt["segments"]:
        assert seg["end_emitted_s"] <= seg["final_s"]
        assert seg["final_text_ready_s"] <= seg["final_s"]
    # 端到端起點早於 VAD 起點（fixture 開頭有靜音），所以端到端延遲較大
    assert m["e2e_first_token_latency_ms"]["mean"] > m["first_token_latency_ms"]["mean"]
    assert m["memory_mb"]["peak_rss"] > m["memory_mb"]["baseline_rss"] > 0
    assert m["model_size_mb"] > 1
    assert set(res["config"]) >= {"model", "pre_roll_ms", "chunk_ms", "vad_min_silence_ms"}


def test_paired_bootstrap_cer_diff() -> None:
    refs = [10] * 50
    a = [1] * 50
    diff, lo, hi = paired_bootstrap_cer_diff(a, a, refs, n_resamples=500)
    assert diff == lo == hi == 0.0  # 逐句相同 → 差為 0，區間退化
    b = [2] * 25 + [1] * 25  # B 在一半的句子多錯 1 字
    diff, lo, hi = paired_bootstrap_cer_diff(a, b, refs, n_resamples=2000)
    assert diff == pytest.approx(25 / 500)
    assert 0 < lo < diff < hi
    with pytest.raises(ValueError):
        paired_bootstrap_cer_diff(a, b[:-1], refs)
