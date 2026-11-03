#!/usr/bin/env bash
# 一條命令從頭建置 macOS app：C++ 核心（cmake）→ Swift（swift build）→ 組裝 .app（嵌入 dylib、Info.plist、ad-hoc 簽名）。
# 用法：scripts/build_mac_app.sh [--bundle-models]
#   --bundle-models   把 models/ 裡執行需要的三個檔案複製進 app 的 Resources/models（約 +230 MB，不進 git）
# 輸出：build/mac/OzAsr.app
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_ROOT="${ORT_ROOT:-$ROOT/third_party/onnxruntime-osx-arm64-1.31.1}"
CPP_BUILD="$ROOT/cpp/build"
APP="$ROOT/build/mac/OzAsr.app"
BUNDLE_MODELS=0
[[ "${1:-}" == "--bundle-models" ]] && BUNDLE_MODELS=1

[[ -f "$ORT_ROOT/lib/libonnxruntime.1.dylib" ]] || { echo "找不到 onnxruntime：$ORT_ROOT（可用 ORT_ROOT 指定）" >&2; exit 1; }

echo "== 1/3 C++ 核心"
cmake -S "$ROOT/cpp" -B "$CPP_BUILD" -DORT_ROOT="$ORT_ROOT" >/dev/null
cmake --build "$CPP_BUILD" -j

echo "== 2/3 Swift app"
cd "$ROOT/apps/macos"
swift build -c release \
  -Xlinker -L"$CPP_BUILD" -Xlinker -lozasr \
  -Xlinker -rpath -Xlinker @executable_path/../Frameworks
BIN="$(swift build -c release --show-bin-path)/OzAsrMac"

echo "== 3/3 組裝 $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Frameworks" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/OzAsrMac"
cp "$ROOT/apps/macos/Info.plist" "$APP/Contents/Info.plist"
cp "$CPP_BUILD/libozasr.dylib" "$APP/Contents/Frameworks/"
# libozasr 依賴 @rpath/libonnxruntime.1.dylib；換成 bundle 內的 rpath（@loader_path = Frameworks/）
cp -L "$ORT_ROOT/lib/libonnxruntime.1.dylib" "$APP/Contents/Frameworks/libonnxruntime.1.dylib"
OZ="$APP/Contents/Frameworks/libozasr.dylib"
for rp in $(otool -l "$OZ" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; f=0}'); do
  install_name_tool -delete_rpath "$rp" "$OZ"
done
install_name_tool -add_rpath @loader_path "$OZ"

if [[ $BUNDLE_MODELS == 1 ]]; then
  M="$APP/Contents/Resources/models"
  SV=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
  mkdir -p "$M/silero-vad-v6.2.3" "$M/$SV"
  cp "$ROOT/models/silero-vad-v6.2.3/silero_vad.onnx" "$M/silero-vad-v6.2.3/"
  cp "$ROOT/models/$SV/model.int8.onnx" "$ROOT/models/$SV/tokens.txt" "$M/$SV/"
fi

# ad-hoc 簽名（不需要 Apple ID）：先簽 dylib，再簽 app
codesign --force --sign - "$APP/Contents/Frameworks/"*.dylib
codesign --force --sign - "$APP"

echo "完成：$APP（$(du -sh "$APP" | cut -f1)）"
