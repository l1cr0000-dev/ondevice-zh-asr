cmake_minimum_required(VERSION 3.20)
project(ozasr VERSION 0.1.0 LANGUAGES CXX)

# 離線低延遲中文 ASR 辨識核心（C++17 + onnxruntime）。三平台共用，各平台只寫收音與介面。
# 建置：cmake -S cpp -B cpp/build -DORT_ROOT=<onnxruntime 套件目錄> && cmake --build cpp/build -j
#   ORT_ROOT 需含 include/onnxruntime_cxx_api.h 與 lib/libonnxruntime.*

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# macOS / Linux：ORT_ROOT 指向預編譯套件（含 include/ 與 lib/）。
# iOS（CMAKE_SYSTEM_NAME=iOS）：ORT_FRAMEWORK 指向 xcframework 內對應切片的 onnxruntime.framework（靜態 framework，
#   二進位本身是 .a）；只建靜態庫 libozasr.a，不建 dylib / CLI / 測試。完整流程見 scripts/build_ios_core.sh。
set(ORT_ROOT "" CACHE PATH "onnxruntime 預編譯套件的根目錄（含 include/ 與 lib/）")
set(ORT_FRAMEWORK "" CACHE PATH "iOS：onnxruntime.framework 目錄（靜態）")
if(IOS)
  if(NOT ORT_FRAMEWORK)
    message(FATAL_ERROR "iOS 建置請以 -DORT_FRAMEWORK=<…/onnxruntime.framework> 指定 onnxruntime")
  endif()
  add_library(onnxruntime_imported STATIC IMPORTED)
  set_target_properties(onnxruntime_imported PROPERTIES
    IMPORTED_LOCATION "${ORT_FRAMEWORK}/onnxruntime"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_FRAMEWORK}/Headers")
else()
  if(NOT ORT_ROOT)
    message(FATAL_ERROR "請以 -DORT_ROOT=<onnxruntime 套件目錄> 指定 onnxruntime")
  endif()
  find_library(ORT_LIB NAMES onnxruntime HINTS "${ORT_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)
  add_library(onnxruntime_imported SHARED IMPORTED)
  set_target_properties(onnxruntime_imported PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")
endif()

add_compile_options(-Wall -Wextra)

add_library(ozasr_core STATIC
  src/fbank.cc src/ctc.cc src/vad.cc src/sense_voice.cc src/pipeline.cc)
target_include_directories(ozasr_core PUBLIC src)
target_link_libraries(ozasr_core PUBLIC onnxruntime_imported)
set_target_properties(ozasr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(IOS)
  # iOS：C API 與核心分別是靜態庫；腳本再用 libtool 併成單一 libozasr.a（不含 onnxruntime，由 app 另外連結）
  add_library(ozasr STATIC src/c_api.cc)
  target_include_directories(ozasr PUBLIC include)
  target_link_libraries(ozasr PRIVATE ozasr_core)
  return()
endif()

# 純 C API（Swift / Kotlin JNI 用）
add_library(ozasr SHARED src/c_api.cc)
target_include_directories(ozasr PUBLIC include)
target_link_libraries(ozasr PRIVATE ozasr_core)
set_target_properties(ozasr PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")

add_executable(ozasr_cli tools/ozasr_cli.cc)
target_link_libraries(ozasr_cli PRIVATE ozasr_core)
set_target_properties(ozasr_cli PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")

add_executable(ozasr_fbank tools/ozasr_fbank.cc)
target_link_libraries(ozasr_fbank PRIVATE ozasr_core)
set_target_properties(ozasr_fbank PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")

enable_testing()
foreach(t fbank vad ctc pipeline c_api)
  add_executable(test_${t} tests/test_${t}.cc)
  target_link_libraries(test_${t} PRIVATE ozasr_core)
  set_target_properties(test_${t} PROPERTIES BUILD_RPATH "${ORT_LIB_DIR}")
  add_test(NAME ${t} COMMAND test_${t})
endforeach()
target_link_libraries(test_c_api PRIVATE ozasr)
