#!/usr/bin/env bash
# 一條命令建置 iPhone app：C++ 核心（iOS 靜態庫）→ 暫存模型與測試 wav → xcodebuild。
# 用法：scripts/build_ios_app.sh --simulator | --device [--bundle-id ID] [--team ID] [--skip-core]
#   --simulator   iPhone 模擬器版（arm64），ad-hoc 簽名（不需 Apple ID）；輸出 build/ios/simulator/OzAsrIOS.app
#   --device      真機版，Xcode 自動簽名（預設 Team 736HS34H28，-allowProvisioningUpdates）；輸出 build/ios/device/OzAsrIOS.app
#   --bundle-id   預設 com.lcrljw.ozasr（免費帳號的 bundle ID 需全域唯一，被占用就換一個）
# 模型與 wav 不進 git：每次建置前複製到 build/ios/stage/oz，Xcode 以資料夾參照打包進 app。
# 內附 wav：tests/fixtures/vad_two_utterances.wav + data/aishell1_test_200 的前 15 句（冒煙集）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET=""
BUNDLE_ID="com.lcrljw.ozasr"
TEAM="736HS34H28"
SKIP_CORE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --simulator) TARGET=simulator ;;
    --device) TARGET=device ;;
    --bundle-id) BUNDLE_ID="$2"; shift ;;
    --team) TEAM="$2"; shift ;;
    --skip-core) SKIP_CORE=1 ;;
    *) echo "未知參數：$1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$TARGET" ]] || { echo "用法：$0 --simulator | --device" >&2; exit 2; }

SV=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
for f in "models/silero-vad-v6.2.3/silero_vad.onnx" "models/$SV/model.int8.onnx" "models/$SV/tokens.txt" \
         "tests/fixtures/vad_two_utterances.wav" "data/aishell1_test_200/manifest.jsonl"; do
  [[ -e "$ROOT/$f" ]] || { echo "缺少 $f" >&2; exit 1; }
done

if [[ $SKIP_CORE == 0 ]]; then
  echo "== 1/3 C++ 核心（iOS）"
  "$ROOT/scripts/build_ios_core.sh"
fi

echo "== 2/3 暫存模型與測試 wav"
STAGE="$ROOT/build/ios/stage/oz"
rm -rf "$STAGE"
mkdir -p "$STAGE/models/silero-vad-v6.2.3" "$STAGE/models/$SV" "$STAGE/wavs"
cp "$ROOT/models/silero-vad-v6.2.3/silero_vad.onnx" "$STAGE/models/silero-vad-v6.2.3/"
cp "$ROOT/models/$SV/model.int8.onnx" "$ROOT/models/$SV/tokens.txt" "$STAGE/models/$SV/"
cp "$ROOT/tests/fixtures/vad_two_utterances.wav" "$STAGE/wavs/00_vad_two_utterances.wav"
python3 - "$ROOT" "$STAGE/wavs" <<'PY'
import json, shutil, sys
root, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{root}/data/aishell1_test_200/manifest.jsonl")][:15]
for i, r in enumerate(rows, 1):
    shutil.copy(f"{root}/data/aishell1_test_200/{r['wav']}", f"{out}/{i:02d}_{r['id']}.wav")
PY

echo "== 3/3 xcodebuild（$TARGET）"
OUT="$ROOT/build/ios/$TARGET"
rm -rf "$OUT"
if [[ $TARGET == simulator ]]; then
  xcodebuild -project "$ROOT/apps/ios/OzAsrIOS.xcodeproj" -target OzAsrIOS -configuration Release \
    -sdk iphonesimulator -arch arm64 OZ_BUNDLE_ID="$BUNDLE_ID" \
    CONFIGURATION_BUILD_DIR="$OUT" CODE_SIGN_STYLE=Manual CODE_SIGN_IDENTITY=- DEVELOPMENT_TEAM= \
    build | tail -25
else
  # 以 -destination id=UDID 讓 Xcode 把真機 UDID 加進描述檔（免費帳號需要）；DEVICE_UDID 可覆寫
  xcodebuild -project "$ROOT/apps/ios/OzAsrIOS.xcodeproj" -scheme OzAsrIOS -configuration Release \
    -destination "id=${DEVICE_UDID:-00008110-000A50213A61401E}" OZ_BUNDLE_ID="$BUNDLE_ID" DEVELOPMENT_TEAM="$TEAM" \
    CONFIGURATION_BUILD_DIR="$OUT" -allowProvisioningUpdates \
    build | tail -25
fi
echo "完成：$OUT/OzAsrIOS.app（$(du -sh "$OUT/OzAsrIOS.app" | cut -f1)）bundle id: $BUNDLE_ID"
