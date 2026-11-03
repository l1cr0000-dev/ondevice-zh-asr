#include "ctc.h"

namespace ozasr {

std::vector<int> ctc_greedy(const float* logits, int frames, int vocab) {
  std::vector<int> ids;
  int prev = -1;
  for (int t = 0; t < frames; ++t) {
    const float* row = logits + static_cast<size_t>(t) * vocab;
    int best = 0;
    for (int v = 1; v < vocab; ++v)
      if (row[v] > row[best]) best = v;
    if (best != 0 && best != prev) ids.push_back(best);
    prev = best;
  }
  return ids;
}

std::string tokens_to_text(const std::vector<int>& ids, const std::vector<std::string>& tokens) {
  static const std::string kSpaceMark = "\xE2\x96\x81";  // U+2581 ▁
  std::string raw;
  for (size_t i = 4; i < ids.size(); ++i) {
    const int id = ids[i];
    if (id >= 0 && static_cast<size_t>(id) < tokens.size()) raw += tokens[id];
  }
  std::string text;
  for (size_t i = 0; i < raw.size();) {
    if (raw.compare(i, kSpaceMark.size(), kSpaceMark) == 0) {
      text += ' ';
      i += kSpaceMark.size();
    } else {
      text += raw[i++];
    }
  }
  const char* ws = " \t\n\r\f\v";
  const size_t b = text.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  return text.substr(b, text.find_last_not_of(ws) - b + 1);
}

std::string GreedyDecoder::decode(const float* logits, int frames, int vocab) const {
  return tokens_to_text(ctc_greedy(logits, frames, vocab), tokens_);
}

}  // namespace ozasr
