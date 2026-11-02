"""src.cli 的共用參數，以及腳本對錯誤輸入的處理（要給錯誤訊息與非零狀態碼，不能是 traceback）。"""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from src.cli import (
    add_decode_args,
    add_pipeline_args,
    display_width,
    pipeline_config_from_args,
    read_hotwords,
)
from src.pipeline import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser(sweep: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_pipeline_args(p, sweep=sweep)
    return p


def test_defaults_match_pipeline_config() -> None:
    args = _parser().parse_args([])
    assert pipeline_config_from_args(args) == PipelineConfig()
    assert args.threads is None and args.chunk_ms == 10  # None = 模型預設（src.asr.default_threads）


def test_single_threshold_fallback() -> None:
    args = _parser().parse_args(["--min-silence-ms", "500", "--pause-ms", "0"])
    config = pipeline_config_from_args(args)
    assert config.vad.min_silence_ms == 500 and config.vad.pause_ms is None


def test_neg_threshold() -> None:
    assert pipeline_config_from_args(_parser().parse_args([])).vad.neg_threshold is None
    args = _parser().parse_args(["--neg-threshold", "0.5"])
    assert pipeline_config_from_args(args).vad.effective_neg_threshold == 0.5


def test_sweep_args_are_lists() -> None:
    args = _parser(sweep=True).parse_args(["--min-silence-ms", "100", "500"])
    assert args.min_silence_ms == [100, 500]
    assert args.pre_roll_ms == [PipelineConfig().pre_roll_ms]
    assert isinstance(args.threshold, float)  # 非掃描參數維持單一值


@pytest.mark.parametrize(
    "argv", [["--chunk-ms", "0"], ["--threads", "-1"], ["--pre-roll-ms", "-10"]]
)
def test_rejects_invalid_numbers(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(argv)
    assert exc.value.code == 2


def test_invalid_vad_threshold_raises_value_error() -> None:
    args = _parser().parse_args(["--threshold", "1.5"])
    with pytest.raises(ValueError):
        pipeline_config_from_args(args)


def test_display_width() -> None:
    assert display_width("ab中文") == 6


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )


def test_run_pipeline_missing_wav_reports_error(tmp_path: Path) -> None:
    proc = _run("scripts/run_pipeline.py", str(tmp_path / "nope.wav"))
    assert proc.returncode == 1
    assert "找不到音訊檔" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_run_eval_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("\n", encoding="utf-8")
    proc = _run("eval/run_eval.py", "--manifest", str(manifest), "--results-dir", str(tmp_path))
    assert proc.returncode == 1
    assert "是空的" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not list(tmp_path.glob("*.json"))


def test_decode_args_and_hotword_file(tmp_path: Path) -> None:
    p = argparse.ArgumentParser()
    add_decode_args(p, sweep=True)
    args = p.parse_args([])
    assert args.hotwords is None and args.beam == [1] and args.hotword_score == [1.5]
    args = p.parse_args(["--beam", "4", "8", "--hotword-score", "1", "2"])
    assert args.beam == [4, 8] and args.hotword_score == [1.0, 2.0]
    f = tmp_path / "hw.txt"
    f.write_text("# 註解\n北京\n\n  上海  # 行尾註解\n北京\n", encoding="utf-8")
    assert read_hotwords(f) == ("北京", "上海")
    assert read_hotwords(None) == ()
