"""C++ 辨識核心（cpp/）的整合測試：C++ 執行檔沒建置時整組跳過，建置了就和 Python 版比對 fixture。"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.asr import DEFAULT_MODEL, load_asr
from src.paths import MODELS_DIR
from src.pipeline import transcribe
from src.vad import DEFAULT_MODEL_PATH, SileroVadModel, load_wav

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD = REPO_ROOT / "cpp" / "build"
CLI = BUILD / "ozasr_cli"
WAV = REPO_ROOT / "tests" / "fixtures" / "vad_two_utterances.wav"
FRAME_S = 0.032

pytestmark = pytest.mark.skipif(
    not CLI.is_file(),
    reason="C++ 尚未建置：cmake -S cpp -B cpp/build -DORT_ROOT=... && cmake --build cpp/build",
)


@pytest.fixture(scope="module")
def cpp_result() -> dict[str, Any]:
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("模型不存在，先執行 bash scripts/download_models.sh")
    out = subprocess.run(
        [str(CLI), "--models-dir", str(MODELS_DIR), str(WAV)],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, Any] = json.loads(out.stdout.splitlines()[0])
    return result


def test_ctest_unit_tests_pass() -> None:
    ctest = shutil.which("ctest")
    if ctest is None:
        pytest.skip("找不到 ctest")
    subprocess.run([ctest, "--test-dir", str(BUILD), "--output-on-failure"], check=True)


def test_fixture_matches_python(cpp_result: dict[str, Any]) -> None:
    py = transcribe(load_wav(WAV), SileroVadModel(num_threads=2), load_asr(DEFAULT_MODEL))
    cpp = cpp_result["utterances"]
    assert len(cpp) == len(py) == 2
    for c, p in zip(cpp, py, strict=True):
        # VAD 切點差距在一個 frame 以內
        assert abs(c["speech_start_s"] - p.speech_start_s) <= FRAME_S
        assert abs(c["speech_end_s"] - p.speech_end_s) <= FRAME_S
        # 定稿文字逐字相同
        assert c["text"] == p.text
    events = cpp_result["events"]
    for u, p in enumerate(py):
        assert [e["text"] for e in events if e["type"] == "partial" and e["utterance"] == u] == [
            x.text for x in p.partials
        ]
        assert [
            e["text"] for e in events if e["type"] == "provisional" and e["utterance"] == u
        ] == [x.text for x in p.provisionals]
    # 事件類型順序：每句 start → partial… → provisional → final
    kinds = [e["type"] for e in events if e["utterance"] == 0]
    assert kinds[0] == "start" and kinds[-1] == "final" and "provisional" in kinds
