"""評測模組測試：文字正規化、語料級 CER、分位數、端到端（小 manifest）。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.metrics import corpus_cer, summarize
from eval.run_eval import RunConfig, summarize_run
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
        "tail_latency_s": tail,
    }


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
    assert m["segments_per_utterance"]["max"] == 2  # fixture 是兩句
    assert 0 < m["first_token_latency_ms"]["mean"] < 2000
    assert m["memory_mb"]["peak_rss"] > m["memory_mb"]["baseline_rss"] > 0
    assert m["model_size_mb"] > 1
    assert set(res["config"]) >= {"model", "pre_roll_ms", "chunk_ms", "vad_min_silence_ms"}
