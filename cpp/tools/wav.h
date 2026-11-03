// 最小的 wav 讀取：PCM 16/24/32 與 float32，多聲道取平均，要求 16 kHz。
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ozasr {

inline std::vector<float> load_wav(const std::string& path, int expected_rate = 16000) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("找不到音訊檔：" + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  auto u16 = [&](size_t o) { return static_cast<uint32_t>(buf[o] | (buf[o + 1] << 8)); };
  auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
  if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) || std::memcmp(buf.data() + 8, "WAVE", 4))
    throw std::runtime_error("不是 RIFF/WAVE：" + path);
  uint32_t fmt = 0, channels = 0, rate = 0, bits = 0;
  size_t data_off = 0, data_len = 0;
  for (size_t o = 12; o + 8 <= buf.size();) {
    const uint32_t len = u32(o + 4);
    if (!std::memcmp(buf.data() + o, "fmt ", 4)) {
      fmt = u16(o + 8);
      channels = u16(o + 10);
      rate = u32(o + 12);
      bits = u16(o + 22);
      if (fmt == 0xFFFE && len >= 26) fmt = u16(o + 8 + 24);  // WAVE_FORMAT_EXTENSIBLE
    } else if (!std::memcmp(buf.data() + o, "data", 4)) {
      data_off = o + 8;
      data_len = std::min<size_t>(len, buf.size() - data_off);
      break;
    }
    o += 8 + len + (len & 1);
  }
  if (!data_off || !channels) throw std::runtime_error("wav 缺少 fmt / data：" + path);
  if (static_cast<int>(rate) != expected_rate)
    throw std::runtime_error(path + " 取樣率為 " + std::to_string(rate) + " Hz，需要 16000 Hz");
  const size_t bytes = bits / 8;
  const size_t frames = data_len / (bytes * channels);
  std::vector<float> out(frames);
  for (size_t i = 0; i < frames; ++i) {
    double sum = 0;
    for (uint32_t c = 0; c < channels; ++c) {
      const uint8_t* p = buf.data() + data_off + (i * channels + c) * bytes;
      float v;
      if (fmt == 3 && bits == 32) {
        std::memcpy(&v, p, 4);
      } else if (fmt == 1 && bits == 16) {
        v = static_cast<int16_t>(p[0] | (p[1] << 8)) / 32768.0f;
      } else if (fmt == 1 && bits == 24) {
        int32_t s = (p[0] << 8) | (p[1] << 16) | (static_cast<int32_t>(p[2]) << 24);
        v = static_cast<float>(s >> 8) / 8388608.0f;
      } else if (fmt == 1 && bits == 32) {
        int32_t s;
        std::memcpy(&s, p, 4);
        v = static_cast<float>(s / 2147483648.0);
      } else {
        throw std::runtime_error("不支援的 wav 格式（fmt=" + std::to_string(fmt) + ", bits=" +
                                 std::to_string(bits) + "）");
      }
      sum += v;
    }
    out[i] = static_cast<float>(sum / channels);
  }
  return out;
}

}  // namespace ozasr
