#!/usr/bin/env bash
# 下載推論所需模型到 models/，並校驗 SHA-256。已存在且 checksum 正確的檔案會跳過。
#
# 用法：bash scripts/download_models.sh [--force]
#
# 模型清單（來源與版本固定在下方，升版時同步更新 URL、版本與 checksum）：
#
#   Silero VAD
#     版本：v6.2.3（git tag，commit 5cd7945676）
#     來源：https://github.com/snakers4/silero-vad
#     授權：MIT
#     檔案：models/silero-vad-v6.2.3/silero_vad.onnx（16 kHz / 8 kHz，FP32，約 2.3 MB）

set -euo pipefail

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

FORCE=0
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --force) FORCE=1 ;;
        *) echo "未知參數：$arg" >&2; usage >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

sha256_of() {
    shasum -a 256 "$1" | awk '{print $1}'
}

# fetch <URL> <目的地（相對 models/）> <預期 sha256>
fetch() {
    local url="$1" dest="$MODELS_DIR/$2" expected="$3"
    if [[ $FORCE -eq 0 && -f "$dest" && "$(sha256_of "$dest")" == "$expected" ]]; then
        echo "已存在且校驗通過：$2"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    local tmp="$dest.part"
    echo "下載 $2"
    echo "  來源：$url"
    curl --fail --location --silent --show-error --retry 3 -o "$tmp" "$url"
    local actual
    actual="$(sha256_of "$tmp")"
    if [[ "$actual" != "$expected" ]]; then
        rm -f "$tmp"
        echo "checksum 不符：$2" >&2
        echo "  預期 $expected" >&2
        echo "  實際 $actual" >&2
        exit 1
    fi
    mv "$tmp" "$dest"
    echo "  完成，sha256=$actual"
}

SILERO_VAD_VERSION="v6.2.3"
fetch \
    "https://raw.githubusercontent.com/snakers4/silero-vad/${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad.onnx" \
    "silero-vad-${SILERO_VAD_VERSION}/silero_vad.onnx" \
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

echo "全部模型就緒：$MODELS_DIR"
