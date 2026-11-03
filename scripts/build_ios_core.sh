#!/usr/bin/env bash
# 把 C++ 辨識核心編成 iOS 靜態庫：ios-arm64（真機）與 ios-arm64-simulator 兩個切片，並包成 xcframework。
# 用法：scripts/build_ios_core.sh
# 環境變數：ORT_IOS_XCFRAMEWORK（預設 third_party/onnxruntime-ios-1.30.0/onnxruntime.xcframework）、IOS_MIN（預設 16.0）
# 輸出：build/ios/core/ozasr.xcframework，以及各切片 build/ios/core/{device,simulator}/libozasr.a（核心 + C API 併成一個 .a，不含 onnxruntime）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_XC="${ORT_IOS_XCFRAMEWORK:-$ROOT/third_party/onnxruntime-ios-1.30.0/onnxruntime.xcframework}"
IOS_MIN="${IOS_MIN:-16.0}"
OUT="$ROOT/build/ios/core"
[[ -d "$ORT_XC/ios-arm64/onnxruntime.framework" ]] || { echo "找不到 onnxruntime iOS xcframework：$ORT_XC" >&2; exit 1; }

build_slice() {  # name sysroot ort-slice
  local name=$1 sysroot=$2 slice=$3 dir="$OUT/$1"
  cmake -S "$ROOT/cpp" -B "$dir/cmake" \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT="$sysroot" -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" -DCMAKE_BUILD_TYPE=Release \
    -DORT_FRAMEWORK="$ORT_XC/$slice/onnxruntime.framework" >/dev/null
  cmake --build "$dir/cmake" -j
  libtool -static -o "$dir/libozasr.a" "$dir/cmake/libozasr.a" "$dir/cmake/libozasr_core.a"
}

rm -rf "$OUT"
build_slice device iphoneos ios-arm64
build_slice simulator iphonesimulator ios-arm64_x86_64-simulator

# 標頭：只放純 C API（include/ozasr/ozasr_c_api.h）
xcodebuild -create-xcframework \
  -library "$OUT/device/libozasr.a" -headers "$ROOT/cpp/include" \
  -library "$OUT/simulator/libozasr.a" -headers "$ROOT/cpp/include" \
  -output "$OUT/ozasr.xcframework" >/dev/null
echo "完成：$OUT/ozasr.xcframework"
