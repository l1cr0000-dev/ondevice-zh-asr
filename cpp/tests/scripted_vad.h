// 依腳本回傳機率的假 VAD 後端（測試用）。
#pragma once

#include <vector>

#include "vad.h"

struct ScriptedVad : ozasr::FrameClassifier {
  std::vector<float> probs;
  size_t i = 0;
  explicit ScriptedVad(std::vector<float> p) : probs(std::move(p)) {}
  int frame_size() const override { return 512; }
  float operator()(const float*) override { return i < probs.size() ? probs[i++] : (++i, 0.0f); }
  void reset() override { i = 0; }
};

// 5 靜音 + 20 語音 + 6 靜音 + 5 語音 + 30 靜音（frame 數），共 66 frame
inline std::vector<float> two_part_script() {
  std::vector<float> p;
  p.insert(p.end(), 5, 0.0f);
  p.insert(p.end(), 20, 0.9f);
  p.insert(p.end(), 6, 0.0f);
  p.insert(p.end(), 5, 0.9f);
  p.insert(p.end(), 30, 0.0f);
  return p;
}
