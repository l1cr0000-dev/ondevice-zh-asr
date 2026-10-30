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
#
#   串流 Zipformer transducer（中文，14M 參數）
#     版本：sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23（sherpa-onnx release tag asr-models）
#     來源：https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#     訓練資料：WenetSpeech；授權：Apache-2.0
#     檔案：models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/
#           encoder/decoder/joiner 各有 FP32（約 53 MB）與 INT8（約 24 MB）兩版，另含 tokens.txt、test_wavs/
#
#   X-ASR 串流 Zipformer transducer（中英，480 ms chunk，官方 INT8 版）
#     版本：sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05（release tag asr-models）
#     來源：https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#           原始模型 https://github.com/Gilgamesh-J/X-ASR
#     檔案：models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05/
#           encoder / joiner 為 INT8、decoder 為 FP32（合計約 161 MB），另含 tokens.txt、bpe.model、test_wavs/

set -euo pipefail

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
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

# fetch_archive <URL> <解壓後目錄名（相對 models/）> <壓縮檔預期 sha256>
# 壓縮檔校驗通過才解壓；解壓完成後寫入 .sha256 標記並刪除壓縮檔，標記吻合即跳過。
fetch_archive() {
    local url="$1" name="$2" expected="$3"
    local dir="$MODELS_DIR/$name" stamp="$MODELS_DIR/$name/.sha256"
    if [[ $FORCE -eq 0 && -f "$stamp" && "$(cat "$stamp")" == "$expected" ]]; then
        echo "已存在且校驗通過：$name/"
        return
    fi
    local archive="$name.tar.bz2"
    fetch "$url" "$archive" "$expected"
    rm -rf "$dir"
    echo "  解壓到 $name/"
    tar -xjf "$MODELS_DIR/$archive" -C "$MODELS_DIR"
    if [[ ! -d "$dir" ]]; then
        echo "解壓後找不到目錄：$name/" >&2
        exit 1
    fi
    echo "$expected" > "$stamp"
    rm -f "$MODELS_DIR/$archive"
}

SILERO_VAD_VERSION="v6.2.3"
fetch \
    "https://raw.githubusercontent.com/snakers4/silero-vad/${SILERO_VAD_VERSION}/src/silero_vad/data/silero_vad.onnx" \
    "silero-vad-${SILERO_VAD_VERSION}/silero_vad.onnx" \
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

ZIPFORMER_ZH_14M="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
fetch_archive \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ZIPFORMER_ZH_14M}.tar.bz2" \
    "$ZIPFORMER_ZH_14M" \
    "2cbd71b640d9c37d3784f29367333a4577b0398b62e9deeed418170b081cba8b"

X_ASR_480MS="sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05"
fetch_archive \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${X_ASR_480MS}.tar.bz2" \
    "$X_ASR_480MS" \
    "14919f03b74d9001fc8f2689c2ab1ea42c436db269e8c809144492f41612bfff"

echo "全部模型就緒：$MODELS_DIR"
