#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "common.h"

namespace ozasr {

namespace {
double steady_seconds() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

Pipeline::Pipeline(FrameClassifier* vad_model, Recognizer* recognizer, const PipelineConfig& config,
                   Clock clock)
    : recognizer_(recognizer),
      config_(config),
      clock_(clock ? std::move(clock) : Clock(steady_seconds)),
      vad_(vad_model, config.vad) {
  if (config_.pre_roll_ms < 0) throw std::invalid_argument("pre_roll_ms 不可為負");
  pre_roll_ = ms_to_samples(config_.pre_roll_ms);
  // 「開始」事件最多落後事件位置 min_speech + pad + 一個 frame；再加 pre-roll 與一些餘裕
  const int lookback_ms =
      config_.vad.min_speech_ms + config_.vad.speech_pad_ms + config_.pre_roll_ms + 100;
  history_capacity_ = ms_to_samples(lookback_ms) + vad_model->frame_size();
  reset();
}

void Pipeline::reset() {
  vad_.reset();
  history_ = History();
  received_ = fed_until_ = 0;
  now_ = call_t0_ = call_base_ = 0;
  stream_.reset();
  current_.reset();
  held_ = false;
  utt_audio_.clear();
  utterances_.clear();
}

void Pipeline::begin_call(double base) {
  call_base_ = base;
  call_t0_ = clock_();
}

double Pipeline::elapsed() const { return call_base_ + (clock_() - call_t0_); }

PipelineEvent Pipeline::make_event(PipelineEvent::Kind kind, std::string text,
                                   double emitted_s) const {
  const Utterance& u = *current_;
  PipelineEvent ev{kind, u.index, std::move(text), emitted_s,
                   static_cast<double>(received_) / kSampleRate, u.speech_start_s, u.audio_start_s};
  return ev;
}

std::vector<PipelineEvent> Pipeline::accept_waveform(const float* samples, size_t n) {
  received_ += static_cast<int64_t>(n);
  begin_call(std::max(now_, static_cast<double>(received_) / kSampleRate));
  history_.data.insert(history_.data.end(), samples, samples + n);
  std::vector<PipelineEvent> events;
  handle(vad_.accept_waveform(samples, n), events);
  if (stream_) feed(received_, events);
  const int64_t excess = static_cast<int64_t>(history_.data.size()) - history_capacity_;
  if (excess > 0) {
    history_.data.erase(history_.data.begin(), history_.data.begin() + excess);
    history_.start += excess;
  }
  end_call();
  return events;
}

std::vector<PipelineEvent> Pipeline::flush() {
  begin_call(now_);
  std::vector<PipelineEvent> events;
  handle(vad_.flush(), events);
  end_call();
  return events;
}

void Pipeline::handle(const std::vector<VadEvent>& vad_events, std::vector<PipelineEvent>& out) {
  for (const VadEvent& ev : vad_events) {
    switch (ev.kind) {
      case VadEvent::Start:
        start(ev, out);
        feed(ev.detected_at, out);
        break;
      case VadEvent::Pause:
        feed(ev.detected_at, out);
        provisional(out);
        break;
      case VadEvent::Resume:
        feed(ev.detected_at, out);
        resume(out);
        break;
      case VadEvent::End:
        feed(ev.detected_at, out);
        finish(ev, out);
        break;
    }
  }
}

void Pipeline::start(const VadEvent& ev, std::vector<PipelineEvent>& out) {
  if (stream_) throw std::logic_error("VAD 事件應成對出現");
  const int64_t begin = std::max({ev.sample - pre_roll_, fed_until_, history_.start});
  stream_ = std::make_unique<AsrStream>(recognizer_, config_.schedule);
  fed_until_ = begin;
  Utterance u;
  u.index = static_cast<int>(utterances_.size());
  u.speech_start_s = static_cast<double>(ev.sample) / kSampleRate;
  u.audio_start_s = static_cast<double>(begin) / kSampleRate;
  u.start_emitted_s = elapsed();
  current_ = std::move(u);
  out.push_back(make_event(PipelineEvent::Start, "", current_->start_emitted_s));
}

void Pipeline::feed(int64_t until, std::vector<PipelineEvent>& out) {
  if (!stream_ || !current_ || until <= fed_until_) return;
  if (fed_until_ < history_.start || until > history_.end())
    throw std::out_of_range("音訊超出保留範圍");
  const float* p = history_.data.data() + (fed_until_ - history_.start);
  const size_t n = static_cast<size_t>(until - fed_until_);
  if (config_.vad.pause_ms > 0) utt_audio_.insert(utt_audio_.end(), p, p + n);
  fed_until_ = until;
  if (held_) return;  // 原 stream 已 finalize，只緩衝，恢復說話時一起重新解碼
  stream_->accept_waveform(p, n);
  emit_partial(out);
}

void Pipeline::emit_partial(std::vector<PipelineEvent>& out) {
  Utterance& u = *current_;
  const std::string& text = stream_->partial();
  const std::string last = u.partials.empty() ? "" : u.partials.back().text;
  if (text == last) return;
  const double now = elapsed();
  u.partials.push_back({now, text});
  const bool has_text = text.find_first_not_of(" \t\n\r\f\v") != std::string::npos;
  if (std::isnan(u.first_token_s) && has_text) u.first_token_s = now;
  out.push_back(make_event(PipelineEvent::Partial, text, now));
}

void Pipeline::provisional(std::vector<PipelineEvent>& out) {
  if (!current_ || !stream_) throw std::logic_error("停頓事件只在語音段中發出");
  std::string text;
  if (config_.provisional == PipelineConfig::Finalize) {
    text = stream_->finalize();
    held_ = true;
  } else {
    AsrStream s(recognizer_, config_.schedule);
    s.accept_waveform(utt_audio_.data(), utt_audio_.size());
    text = s.finalize();
  }
  const double audio_end = static_cast<double>(fed_until_) / kSampleRate;
  const double now = elapsed();
  current_->provisionals.push_back({now, audio_end, text});
  PipelineEvent ev = make_event(PipelineEvent::Provisional, text, now);
  ev.audio_end_s = audio_end;
  out.push_back(std::move(ev));
}

void Pipeline::resume(std::vector<PipelineEvent>& out) {
  if (!held_ || !current_) return;
  const double t0 = elapsed();
  stream_ = std::make_unique<AsrStream>(recognizer_, config_.schedule);
  stream_->accept_waveform(utt_audio_.data(), utt_audio_.size());
  held_ = false;
  current_->resume_redecode_s.push_back(elapsed() - t0);
  emit_partial(out);
}

void Pipeline::finish(const VadEvent& ev, std::vector<PipelineEvent>& out) {
  if (!stream_ || !current_) throw std::logic_error("VAD 事件應成對出現");
  Utterance& u = *current_;
  u.speech_end_s = static_cast<double>(ev.sample) / kSampleRate;
  u.end_emitted_s = elapsed();
  // "finalize" 做法且停頓後沒有恢復：原 stream 已 finalize，以暫定結果定稿
  u.text = stream_->finalized() ? stream_->result() : stream_->finalize();
  u.final_s = elapsed();
  const bool has_text = u.text.find_first_not_of(" \t\n\r\f\v") != std::string::npos;
  if (std::isnan(u.first_token_s) && has_text) u.first_token_s = u.final_s;
  PipelineEvent e = make_event(PipelineEvent::Final, u.text, u.final_s);
  e.speech_end_s = u.speech_end_s;
  out.push_back(std::move(e));
  utterances_.push_back(std::move(u));
  stream_.reset();
  current_.reset();
  held_ = false;
  utt_audio_.clear();
}

}  // namespace ozasr
