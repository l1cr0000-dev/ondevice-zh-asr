// 特徵前處理：kaldi 風格 fbank（80 維、hamming、dither 0、snip_edges）→ LFR → CMVN。
// 與 kaldi-native-fbank / sherpa-onnx 的 SenseVoice 前處理對齊，不依賴第三方程式碼。
#pragma once

#include <vector>

namespace ozasr {

struct FbankOptions {
  int sample_rate = 16000;
  int frame_length = 400;  // 25 ms
  int frame_shift = 160;   // 10 ms
  int fft_size = 512;      // 進位到 2 的次方
  int num_bins = 80;
  float low_freq = 20.0f;
  float preemph = 0.97f;
};

class Fbank {
 public:
  explicit Fbank(const FbankOptions& opts = FbankOptions());

  // snip_edges：只輸出完整的 frame
  int num_frames(size_t num_samples) const;

  // samples 的範圍是 int16 尺度（呼叫端自行乘 32768）。輸出 num_frames × num_bins 的 log-mel。
  void compute(const float* samples, size_t n, std::vector<float>& out) const;

  int num_bins() const { return opts_.num_bins; }

 private:
  struct Filter {
    int first = 0;
    std::vector<double> weights;
  };
  void fft(std::vector<double>& re, std::vector<double>& im) const;

  FbankOptions opts_;
  std::vector<double> window_;
  std::vector<double> cos_, sin_;
  std::vector<int> bitrev_;
  std::vector<Filter> filters_;
};

// LFR（每 shift 個 frame 取一次，拼接以它為中心的 m 個 frame，超出邊界重複首 / 末 frame）
// 後接 CMVN：(x + neg_mean) * inv_std。輸出 ceil(T / shift) × (dim * m)。
void lfr_cmvn(const std::vector<float>& feats, int num_frames, int dim, int m, int shift,
              const std::vector<float>& neg_mean, const std::vector<float>& inv_std,
              std::vector<float>& out, int& out_frames);

}  // namespace ozasr
