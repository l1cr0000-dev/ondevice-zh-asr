"""環境檢查：印出套件版本、onnxruntime execution providers、麥克風裝置列表。

用法：uv run scripts/check_env.py
任何一項失敗時以非零狀態碼結束。
"""

import importlib
import importlib.metadata
import platform
import subprocess
import sys
from typing import Any

# (PyPI 套件名, import 名)
PACKAGES: list[tuple[str, str]] = [
    ("onnxruntime", "onnxruntime"),
    ("sherpa-onnx", "sherpa_onnx"),
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"),
    ("jiwer", "jiwer"),
    ("psutil", "psutil"),
]


def section(title: str) -> None:
    print(f"\n== {title} ==")


def chip_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown"


def check_system() -> None:
    section("系統")
    print(f"Python      {sys.version.split()[0]} ({sys.executable})")
    print(f"macOS       {platform.mac_ver()[0] or 'N/A'}")
    print(f"架構        {platform.machine()}")
    print(f"晶片        {chip_name()}")


def check_packages() -> bool:
    section("套件版本")
    ok = True
    for dist, module in PACKAGES:
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(dist)
            print(f"{dist:<12} {version}")
        except Exception as e:  # noqa: BLE001 - 任何匯入錯誤都要回報
            print(f"{dist:<12} 失敗：{type(e).__name__}: {e}")
            ok = False
    return ok


def check_onnxruntime() -> bool:
    section("onnxruntime execution providers")
    try:
        ort: Any = importlib.import_module("onnxruntime")
        providers: list[str] = ort.get_available_providers()
    except Exception as e:  # noqa: BLE001
        print(f"失敗：{type(e).__name__}: {e}")
        return False
    for p in providers:
        print(f"- {p}")
    return True


def check_microphones() -> bool:
    section("音訊輸入裝置")
    try:
        sd: Any = importlib.import_module("sounddevice")
        devices: list[dict[str, Any]] = list(sd.query_devices())
        default_in: int = sd.default.device[0]
    except Exception as e:  # noqa: BLE001
        print(f"失敗：{type(e).__name__}: {e}")
        return False
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print("找不到輸入裝置")
        return False
    for i, d in inputs:
        mark = "*" if i == default_in else " "
        print(
            f"{mark} [{i}] {d['name']}  "
            f"channels={d['max_input_channels']}  "
            f"default_sr={d['default_samplerate']:.0f}"
        )
    print("（* = 預設輸入裝置；列出裝置不需要麥克風權限，實際錄音時 macOS 才會詢問）")
    return True


def main() -> int:
    check_system()
    results = [check_packages(), check_onnxruntime(), check_microphones()]
    ok = all(results)
    section("結果")
    print("OK" if ok else "有項目失敗，見上方輸出")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
