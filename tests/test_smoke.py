"""佔位測試：確認套件可匯入。各模組實作後換成真正的最小測試。"""

import importlib

import pytest


@pytest.mark.parametrize("module", ["src.vad", "src.asr", "src.pipeline", "eval"])
def test_module_importable(module: str) -> None:
    importlib.import_module(module)
