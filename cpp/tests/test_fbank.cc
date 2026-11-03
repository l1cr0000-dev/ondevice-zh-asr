#include <cmath>
#include <vector>

#include "check.h"
#include "fbank.h"

using namespace ozasr;

static std::vector<float> test_signal(int n) {
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = static_cast<float>((0.3 * std::sin(2 * M_PI * 440 * i / 16000) +
                               0.1 * std::sin(2 * M_PI * 1234 * i / 16000)) * 32768.0);
  return x;
}

int main() {
  Fbank fb;
  // snip_edges：只輸出完整 frame
  CHECK_EQ(fb.num_frames(399), 0);
  CHECK_EQ(fb.num_frames(400), 1);
  CHECK_EQ(fb.num_frames(559), 1);
  CHECK_EQ(fb.num_frames(560), 2);
  CHECK_EQ(fb.num_frames(1600), 8);

  // 與 kaldi-native-fbank 的數值（同一個合成訊號）：frame 0 與 5 的第 0/5/10/20/40/79 維
  const std::vector<float> x = test_signal(1600);
  std::vector<float> out;
  fb.compute(x.data(), x.size(), out);
  CHECK_EQ(out.size(), static_cast<size_t>(8 * 80));
  const int dims[6] = {0, 5, 10, 20, 40, 79};
  const float g0[6] = {11.9754f, 11.8139f, 11.6183f, 13.6093f, 13.511f, 12.1175f};
  const float g5[6] = {11.7692f, 12.1083f, 12.3545f, 14.8655f, 12.1609f, 9.9875f};
  for (int k = 0; k < 6; ++k) {
    CHECK_NEAR(out[0 * 80 + dims[k]], g0[k], 1e-3);
    CHECK_NEAR(out[5 * 80 + dims[k]], g5[k], 1e-3);
  }

  // 全零輸入 → log(float epsilon)
  std::vector<float> zeros(400, 0.0f);
  fb.compute(zeros.data(), zeros.size(), out);
  CHECK_NEAR(out[0], std::log(1.1920929e-7), 1e-4);

  // LFR：10 個 frame、dim 1、m 7、shift 6 → 2 個輸出 frame，邊界重複首 / 末 frame
  std::vector<float> feats = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, lfr;
  std::vector<float> mean(7, 0.0f), inv(7, 1.0f);
  int frames = 0;
  lfr_cmvn(feats, 10, 1, 7, 6, mean, inv, lfr, frames);
  CHECK_EQ(frames, 2);
  const float expect[14] = {0, 0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9};
  for (int i = 0; i < 14; ++i) CHECK_EQ(lfr[i], expect[i]);
  // 13 個 frame → ceil(13 / 6) = 3，最後一個窗口超出末端的部分重複末 frame
  feats.assign(13, 1.0f);
  lfr_cmvn(feats, 13, 1, 7, 6, mean, inv, lfr, frames);
  CHECK_EQ(frames, 3);
  // CMVN：(x + neg_mean) * inv_std，依 LFR 拼接後的維度索引
  std::vector<float> mean2(14), inv2(14);
  for (int i = 0; i < 14; ++i) { mean2[i] = -static_cast<float>(i); inv2[i] = 0.5f; }
  feats.assign(8, 10.0f);
  lfr_cmvn(feats, 4, 2, 7, 6, mean2, inv2, lfr, frames);  // dim 2 → 4 frame / 2 維，輸出 1 frame × 14
  CHECK_EQ(frames, 1);
  for (int i = 0; i < 14; ++i) CHECK_NEAR(lfr[i], (10.0f - i) * 0.5f, 1e-6);
  TEST_MAIN_END();
}
