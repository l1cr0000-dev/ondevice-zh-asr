// SenseVoice Small INT8（onnxruntime）與偽串流 stream。對應 src/asr/sense_voice.py 的 SenseVoiceOrtAsr / SenseVoiceStream。
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ctc.h"
#include "fbank.h"

namespace ozasr {

// 偽 partial 的重新辨識時機（以 stream 已累積的樣本數計）：first、first + every、…；超過 max 之後不再辨識。
struct PartialSchedule {
  double every_s = 0.96;
  double first_s = 0.48;  // <= 0 時等於 every_s
  double max_s = 3.0;     // <= 0 = 不限

  // 已累積 n 個樣本時，下一次重新辨識的樣本數；無下一次時回傳 nullopt
  std::optional<int64_t> next_sample(int64_t n) const;
};

struct SenseVoiceConfig {
  std::string model_path;
  std::string tokens_path;
  std::string language = "zh";  // metadata 的 lang_<language>
  bool use_itn = false;
  int num_threads = 2;
  bool allow_spinning = false;  // false → session.intra_op.allow_spinning=0
  // 解碼擴充點：預設 greedy；熱詞版本在這裡注入自己的 CtcDecoder（需要 tokens，見 SenseVoiceAsr::tokens()）
};

// 整段音訊 → 文字
class Recognizer {
 public:
  virtual ~Recognizer() = default;
  virtual std::string recognize(const std::vector<float>& audio) = 0;
};

class SenseVoiceAsr : public Recognizer {
 public:
  explicit SenseVoiceAsr(const SenseVoiceConfig& config);
  ~SenseVoiceAsr() override;
  std::string recognize(const std::vector<float>& audio) override;

  // 前處理（供測試與對齊腳本）：音訊（[-1, 1]）→ LFR + CMVN 後的 (T', 560) 特徵
  void features(const float* audio, size_t n, std::vector<float>& out, int& frames) const;
  const std::vector<std::string>& tokens() const { return tokens_; }
  void set_decoder(std::unique_ptr<CtcDecoder> decoder) { decoder_ = std::move(decoder); }

  int64_t num_decodes() const { return num_decodes_; }
  double decoded_audio_s() const { return decoded_samples_ / 16000.0; }
  void reset_stats() { num_decodes_ = 0, decoded_samples_ = 0; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Fbank fbank_;
  int lfr_m_ = 7, lfr_n_ = 6;
  std::vector<float> neg_mean_, inv_std_;
  float scale_ = 32768.0f;
  int32_t language_ = 0, text_norm_ = 0;
  std::vector<std::string> tokens_;
  std::unique_ptr<CtcDecoder> decoder_;
  int64_t num_decodes_ = 0, decoded_samples_ = 0;
};

// 一句話的辨識狀態：累積音訊，依 schedule 重新辨識整段當 partial；finalize 時整段辨識一次。
class AsrStream {
 public:
  AsrStream(Recognizer* recognizer, const std::optional<PartialSchedule>& schedule);
  void accept_waveform(const float* chunk, size_t n);
  const std::string& partial() const { return partial_; }
  bool finalized() const { return finalized_; }
  std::string finalize();
  // finalize 後回傳最終結果，否則目前的 partial
  const std::string& result() const { return finalized_ ? final_ : partial_; }

 private:
  std::string decode();

  Recognizer* recognizer_;
  std::optional<PartialSchedule> schedule_;
  std::vector<float> audio_;
  std::optional<int64_t> next_partial_;
  std::string partial_, final_;
  bool finalized_ = false;
};

}  // namespace ozasr
