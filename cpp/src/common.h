// 共用常數與小工具。
#pragma once

#include <cstddef>
#include <cstdint>

namespace ozasr {

constexpr int kSampleRate = 16000;

inline int64_t ms_to_samples(int ms) { return static_cast<int64_t>(ms) * kSampleRate / 1000; }

}  // namespace ozasr
