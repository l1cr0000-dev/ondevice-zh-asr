// CTC 解碼。目前只有 greedy；熱詞（prefix beam search + Aho-Corasick）見 CtcDecoder 的擴充點。
#pragma once

#include <string>
#include <vector>

namespace ozasr {

// logits：frames × vocab（未經 softmax）。逐 frame argmax（平手取第一個）、合併重複、去掉 blank（id 0）。
std::vector<int> ctc_greedy(const float* logits, int frames, int vocab);

// SenseVoice 的 token 序列 → 文字：丟掉前 4 個 token（語言、情緒、事件、ITN 標記），
// 「▁」換成空白，去頭尾空白。
std::string tokens_to_text(const std::vector<int>& ids, const std::vector<std::string>& tokens);

// 解碼器介面：logits → 文字。熱詞版本之後從這裡派生（beam > 1 時略過前 4 個 frame 再做 prefix beam search），
// 辨識核心不需改動。
class CtcDecoder {
 public:
  virtual ~CtcDecoder() = default;
  virtual std::string decode(const float* logits, int frames, int vocab) const = 0;
};

class GreedyDecoder : public CtcDecoder {
 public:
  explicit GreedyDecoder(std::vector<std::string> tokens) : tokens_(std::move(tokens)) {}
  std::string decode(const float* logits, int frames, int vocab) const override;

 private:
  std::vector<std::string> tokens_;
};

}  // namespace ozasr
