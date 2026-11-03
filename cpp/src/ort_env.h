// 全程式共用一個 Ort::Env。
#pragma once

#include <onnxruntime_cxx_api.h>

namespace ozasr {

inline Ort::Env& ort_env() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ozasr");
  return env;
}

}  // namespace ozasr
