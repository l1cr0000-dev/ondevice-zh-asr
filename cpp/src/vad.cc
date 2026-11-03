#include "vad.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

#include "common.h"
#include "ort_env.h"

namespace ozasr {

// ---------- Silero ----------

constexpr int kFrame = 512;
constexpr int kContext = 64;

struct SileroVad::Impl {
  Ort::Session session{nullptr};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<float, 2 * 128> state{};
  std::array<float, kContext + kFrame> input{};
  int64_t sr = kSampleRate;
  std::string out_prob, out_state;
};

SileroVad::SileroVad(const std::string& model_path, int num_threads) : impl_(new Impl) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(num_threads);
  opts.SetInterOpNumThreads(1);
  impl_->session = Ort::Session(ort_env(), model_path.c_str(), opts);
  Ort::AllocatorWithDefaultOptions alloc;
  impl_->out_prob = impl_->session.GetOutputNameAllocated(0, alloc).get();
  impl_->out_state = impl_->session.GetOutputNameAllocated(1, alloc).get();
  reset();
}

SileroVad::~SileroVad() = default;

void SileroVad::reset() {
  impl_->state.fill(0.0f);
  impl_->input.fill(0.0f);  // 前 64 個樣本 = context
}

float SileroVad::operator()(const float* frame) {
  Impl& m = *impl_;
  std::memcpy(m.input.data() + kContext, frame, kFrame * sizeof(float));
  const int64_t in_shape[2] = {1, kContext + kFrame};
  const int64_t st_shape[3] = {2, 1, 128};
  Ort::Value inputs[3] = {
      Ort::Value::CreateTensor<float>(m.mem, m.input.data(), m.input.size(), in_shape, 2),
      Ort::Value::CreateTensor<float>(m.mem, m.state.data(), m.state.size(), st_shape, 3),
      Ort::Value::CreateTensor<int64_t>(m.mem, &m.sr, 1, nullptr, 0)};
  const char* in_names[3] = {"input", "state", "sr"};
  const char* out_names[2] = {m.out_prob.c_str(), m.out_state.c_str()};
  auto outputs = m.session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 2);
  const float prob = outputs[0].GetTensorData<float>()[0];
  std::memcpy(m.state.data(), outputs[1].GetTensorData<float>(), m.state.size() * sizeof(float));
  // 下一個 frame 的 context = 這個 frame 的尾端
  std::memmove(m.input.data(), m.input.data() + kFrame, kContext * sizeof(float));
  return prob;
}

// ---------- 狀態機 ----------

float VadConfig::effective_neg_threshold() const {
  return neg_threshold >= 0.0f ? neg_threshold : std::max(threshold - 0.15f, 0.01f);
}

void VadConfig::validate() const {
  if (!(threshold > 0.0f && threshold < 1.0f))
    throw std::invalid_argument("threshold 必須在 (0, 1) 內");
  if (neg_threshold >= 0.0f && !(neg_threshold > 0.0f && neg_threshold <= threshold))
    throw std::invalid_argument("neg_threshold 必須在 (0, threshold] 內");
  if (std::min({min_speech_ms, min_silence_ms, speech_pad_ms}) < 0)
    throw std::invalid_argument("時間長度參數不可為負");
  if (pause_ms != 0 && !(pause_ms > 0 && pause_ms < min_silence_ms))
    throw std::invalid_argument("pause_ms 必須在 (0, min_silence_ms) 內");
}

StreamingVad::StreamingVad(FrameClassifier* model, const VadConfig& config)
    : model_(model), config_(config) {
  config_.validate();
  min_speech_ = ms_to_samples(config_.min_speech_ms);
  min_silence_ = ms_to_samples(config_.min_silence_ms);
  pad_ = ms_to_samples(config_.speech_pad_ms);
  pause_ = config_.pause_ms > 0 ? ms_to_samples(config_.pause_ms) : -1;
  neg_threshold_ = config_.effective_neg_threshold();
  reset();
}

void StreamingVad::reset() {
  model_->reset();
  buffer_.clear();
  received_ = processed_ = 0;
  state_ = State::Silence;
  speech_start_ = last_speech_end_ = prev_end_ = 0;
  paused_ = false;
  last_prob_ = 0.0f;
}

std::vector<VadEvent> StreamingVad::accept_waveform(const float* samples, size_t n) {
  received_ += static_cast<int64_t>(n);
  buffer_.insert(buffer_.end(), samples, samples + n);
  std::vector<VadEvent> events;
  const size_t fs = static_cast<size_t>(model_->frame_size());
  const size_t frames = buffer_.size() / fs;
  for (size_t i = 0; i < frames; ++i) step(buffer_.data() + i * fs, events);
  buffer_.erase(buffer_.begin(), buffer_.begin() + frames * fs);
  return events;
}

std::vector<VadEvent> StreamingVad::flush() {
  std::vector<VadEvent> events;
  if (!buffer_.empty()) {
    std::vector<float> frame(static_cast<size_t>(model_->frame_size()), 0.0f);
    std::copy(buffer_.begin(), buffer_.end(), frame.begin());
    buffer_.clear();
    step(frame.data(), events);
  }
  if (state_ == State::Speech) {
    events.push_back(end_event(std::min(last_speech_end_ + pad_, received_), received_));
  }
  state_ = State::Silence;  // Pending：候選段不滿 min_speech，丟棄
  return events;
}

VadEvent StreamingVad::start_event(int64_t detected_at) {
  state_ = State::Speech;
  const int64_t start = std::max<int64_t>({speech_start_ - pad_, prev_end_, 0});
  return {VadEvent::Start, start, detected_at};
}

VadEvent StreamingVad::end_event(int64_t end, int64_t detected_at) {
  state_ = State::Silence;
  prev_end_ = end;
  paused_ = false;
  return {VadEvent::End, end, detected_at};
}

void StreamingVad::step(const float* frame, std::vector<VadEvent>& events) {
  const float prob = (*model_)(frame);
  last_prob_ = prob;
  const int64_t frame_start = processed_;
  processed_ += model_->frame_size();
  const int64_t frame_end = processed_;
  const int64_t detected_at = std::min(frame_end, received_);

  const bool is_speech = prob >= config_.threshold;
  const bool is_silence = prob < neg_threshold_;

  if (state_ == State::Silence) {
    if (is_speech) {
      state_ = State::Pending;
      speech_start_ = frame_start;
      last_speech_end_ = frame_end;
    }
  } else if (is_speech || !is_silence) {
    // 語音 frame，或遲滯區間內：延續目前的語音（候選）段
    if (is_speech) {
      last_speech_end_ = frame_end;
      if (paused_) {
        paused_ = false;
        events.push_back({VadEvent::Resume, frame_start, detected_at});
      }
    }
  } else {
    const int64_t silence = frame_end - last_speech_end_;
    if (state_ == State::Speech && pause_ >= 0 && !paused_ && pause_ <= silence &&
        silence < min_silence_) {
      paused_ = true;
      events.push_back({VadEvent::Pause, last_speech_end_ + pad_, detected_at});
    }
    if (silence >= min_silence_) {
      if (state_ == State::Speech) {
        events.push_back(end_event(std::min(last_speech_end_ + pad_, detected_at), detected_at));
      } else {
        state_ = State::Silence;  // 候選段太短，丟棄
      }
      return;
    }
  }

  if (state_ == State::Pending && last_speech_end_ - speech_start_ >= min_speech_) {
    events.push_back(start_event(detected_at));
  }
}

}  // namespace ozasr
