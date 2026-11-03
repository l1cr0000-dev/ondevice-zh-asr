#include <vector>

#include "check.h"
#include "ctc.h"

using namespace ozasr;

int main() {
  // 6 frame × 5 類，argmax = 0 2 2 0 2 3
  const int best[6] = {0, 2, 2, 0, 2, 3};
  std::vector<float> logits(6 * 5, 0.0f);
  for (int t = 0; t < 6; ++t) logits[t * 5 + best[t]] = 1.0f;
  CHECK_EQ(ctc_greedy(logits.data(), 6, 5), (std::vector<int>{2, 2, 3}));  // 中間隔著 blank 的重複算兩個

  // 平手取第一個（與 numpy argmax 相同）
  std::vector<float> tie = {1.0f, 1.0f, 0.0f};
  CHECK_EQ(ctc_greedy(tie.data(), 1, 3).size(), static_cast<size_t>(0));  // argmax = 0 = blank
  std::vector<float> tie2 = {0.0f, 1.0f, 1.0f};
  CHECK_EQ(ctc_greedy(tie2.data(), 1, 3), (std::vector<int>{1}));

  // 全 blank
  std::vector<float> blank(3 * 4, 0.0f);
  for (int t = 0; t < 3; ++t) blank[t * 4] = 5.0f;
  CHECK(ctc_greedy(blank.data(), 3, 4).empty());

  // token → 文字：丟前 4 個、「▁」→ 空白、去頭尾空白
  const std::vector<std::string> tokens = {"<blank>", "<a>", "<b>", "<c>", "<d>", "\xE2\x96\x81hi", "x", "你", "好"};
  CHECK_EQ(tokens_to_text({1, 2, 3, 4, 5, 6}, tokens), std::string("hix"));
  CHECK_EQ(tokens_to_text({1, 2, 3, 4, 7, 8}, tokens), std::string("你好"));
  CHECK_EQ(tokens_to_text({1, 2, 3, 4, 7, 5, 8}, tokens), std::string("你 hi好"));
  CHECK_EQ(tokens_to_text({1, 2, 3}, tokens), std::string(""));

  GreedyDecoder dec(tokens);
  std::vector<float> l(8 * 9, 0.0f);
  const int seq[8] = {1, 2, 3, 4, 7, 7, 0, 8};
  for (int t = 0; t < 8; ++t) l[t * 9 + seq[t]] = 1.0f;
  CHECK_EQ(dec.decode(l.data(), 8, 9), std::string("你好"));
  TEST_MAIN_END();
}
