// Silero VAD（32 ms frame、state 張量）與遲滯雙閾值狀態機。對應 src/vad/{silero,detector}.py。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ozasr {

// 逐 frame 回傳語音機率的模型後端（換 Core ML 等後端時實作此介面）
class FrameClassifier {
 public:
  virtual ~FrameClassifier() = default;
  virtual int frame_size() const = 0;
  virtual float operator()(const float* frame) = 0;
  virtual void reset() = 0;
};

// Silero v6 ONNX：input [1, 64 + 512]（前 64 個是上一個 frame 的尾端）、state [2,1,128]、sr int64 scalar
class SileroVad : public FrameClassifier {
 public:
  SileroVad(const std::string& model_path, int num_threads);
  ~SileroVad() override;
  int frame_size() const override { return 512; }
  float operator()(const float* frame) override;
  void reset() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct VadConfig {
  float threshold = 0.5f;
  float neg_threshold = -1.0f;  // < 0 時取 max(threshold - 0.15, 0.01)
  int min_speech_ms = 250;
  int min_silence_ms = 100;
  int speech_pad_ms = 30;
  int pause_ms = 0;  // 0 = 不發停頓事件；否則需小於 min_silence_ms

  float effective_neg_threshold() const;
  void validate() const;  // 不合法時丟 std::invalid_argument
};

struct VadEvent {
  enum Kind { Start, Pause, Resume, End };
  Kind kind;
  int64_t sample;       // 事件在音訊時間軸上的位置（已含 padding）
  int64_t detected_at;  // 發出事件時已送入的樣本數
};

class StreamingVad {
 public:
  // model 的生命週期由呼叫端管理
  StreamingVad(FrameClassifier* model, const VadConfig& config);
  void reset();
  // 送入 float 音訊（16 kHz 單聲道），回傳這次新產生的事件
  std::vector<VadEvent> accept_waveform(const float* samples, size_t n);
  // 輸入結束：補零處理剩餘樣本，結束進行中的語音段
  std::vector<VadEvent> flush();
  // 強制結束進行中的語音段（按住說話放開時用）：Speech 狀態立刻發 End，Pending 候選丟棄；
  // 與 flush() 不同，不補零處理剩餘樣本，之後可以直接繼續送入音訊。
  std::vector<VadEvent> force_end();
  float last_prob() const { return last_prob_; }

 private:
  enum class State { Silence, Pending, Speech };
  void step(const float* frame, std::vector<VadEvent>& events);
  VadEvent start_event(int64_t detected_at);
  VadEvent end_event(int64_t end, int64_t detected_at);

  FrameClassifier* model_;
  VadConfig config_;
  int64_t min_speech_, min_silence_, pad_, pause_;
  float neg_threshold_;
  std::vector<float> buffer_;
  int64_t received_ = 0, processed_ = 0;
  State state_ = State::Silence;
  int64_t speech_start_ = 0, last_speech_end_ = 0, prev_end_ = 0;
  bool paused_ = false;
  float last_prob_ = 0.0f;
};

}  // namespace ozasr
