// VAD + 偽串流 ASR 的 pipeline：逐 chunk 送入 → 回傳開始 / partial / 暫定 / 定稿事件。
// 對應 src/pipeline/pipeline.py（provisional="finalize" 與 "redecode" 兩種做法都實作）。
//
// 時間分兩類（秒，以音訊開頭為 0）：
//  - 音訊時間軸上的位置：speech_start_s、speech_end_s、audio_start_s、audio_pos_s。決定性，與機器無關。
//  - 「模擬即時」的發出時刻 emitted_s：假設音訊依實際時長節奏到達，一個 chunk 要等最後一個樣本到達
//    才能處理，處理的計算時間以 clock 實際量測並累加（與 Python 版相同）。
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "sense_voice.h"
#include "vad.h"

namespace ozasr {

struct PipelineConfig {
  VadConfig vad{0.5f, -1.0f, 250, 700, 30, 150};  // 預設：A2 兩段門檻（暫定 150 ms、定稿 700 ms）
  int pre_roll_ms = 100;
  enum Provisional { Finalize, Redecode } provisional = Finalize;
  std::optional<PartialSchedule> schedule = PartialSchedule{0.96, 0.48, 3.0};
};

const double kNone = std::numeric_limits<double>::quiet_NaN();

struct PipelineEvent {
  enum Kind { Start, Partial, Provisional, Final };
  Kind kind;
  int utterance;           // 句子編號（從 0 起）
  std::string text;        // Start 為空；其餘為該事件的文字
  double emitted_s;        // 模擬即時的發出時刻
  double audio_pos_s;      // 事件發出時已送入的音訊長度（決定性）
  double speech_start_s;   // VAD 起點（含 pad）
  double audio_start_s;    // 實際送入 ASR 的第一個樣本（含 pre-roll）
  double speech_end_s = kNone;  // 只有 Final
  double audio_end_s = kNone;   // 只有 Provisional：重新解碼涵蓋到的音訊終點
};

struct Utterance {
  int index = 0;
  double speech_start_s = 0, audio_start_s = 0, start_emitted_s = 0;
  double first_token_s = kNone, speech_end_s = kNone, end_emitted_s = kNone, final_s = kNone;
  std::string text;
  struct Partial { double emitted_s; std::string text; };
  struct Provisional { double emitted_s, audio_end_s; std::string text; };
  std::vector<Partial> partials;
  std::vector<Provisional> provisionals;
  std::vector<double> resume_redecode_s;
};

class Pipeline {
 public:
  using Clock = std::function<double()>;  // 單調時鐘（秒）
  // vad_model 與 recognizer 的生命週期由呼叫端管理
  Pipeline(FrameClassifier* vad_model, Recognizer* recognizer, const PipelineConfig& config,
           Clock clock = nullptr);
  void reset();
  std::vector<PipelineEvent> accept_waveform(const float* samples, size_t n);
  std::vector<PipelineEvent> flush();
  // 立刻結束目前句子並定稿（按住說話放開）：不等靜音門檻；若句子在停頓中則直接以暫定結果定稿。
  // 與 flush() 不同，之後可直接繼續 accept_waveform，不需要 reset。沒有進行中的句子時回傳空。
  std::vector<PipelineEvent> finalize_utterance();
  const std::vector<Utterance>& utterances() const { return utterances_; }
  bool in_speech() const { return stream_ != nullptr; }

 private:
  struct History {
    std::vector<float> data;
    int64_t start = 0;
    int64_t end() const { return start + static_cast<int64_t>(data.size()); }
  };

  void begin_call(double base);
  double elapsed() const;
  void end_call() { now_ = elapsed(); }
  void handle(const std::vector<VadEvent>& vad_events, std::vector<PipelineEvent>& out);
  void start(const VadEvent& ev, std::vector<PipelineEvent>& out);
  void feed(int64_t until, std::vector<PipelineEvent>& out);
  void emit_partial(std::vector<PipelineEvent>& out);
  void provisional(std::vector<PipelineEvent>& out);
  void resume(std::vector<PipelineEvent>& out);
  void finish(const VadEvent& ev, std::vector<PipelineEvent>& out);
  PipelineEvent make_event(PipelineEvent::Kind kind, std::string text, double emitted_s) const;

  Recognizer* recognizer_;
  PipelineConfig config_;
  Clock clock_;
  StreamingVad vad_;
  int64_t pre_roll_, history_capacity_;

  History history_;
  int64_t received_ = 0, fed_until_ = 0;
  double now_ = 0, call_t0_ = 0, call_base_ = 0;
  std::unique_ptr<AsrStream> stream_;
  std::optional<Utterance> current_;
  bool held_ = false;
  std::vector<float> utt_audio_;
  std::vector<Utterance> utterances_;
};

}  // namespace ozasr
