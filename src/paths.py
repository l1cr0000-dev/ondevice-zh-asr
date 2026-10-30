"""專案目錄。全部以本檔位置推算，不依賴目前工作目錄。"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
