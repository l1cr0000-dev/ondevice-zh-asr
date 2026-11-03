#include "sense_voice.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common.h"
#include "ort_env.h"

namespace ozasr {

std::optional<int64_t> PartialSchedule::next_sample(int64_t n) const {
  const int64_t every = std::llround(every_s * kSampleRate);
  const int64_t first = first_s > 0 ? std::llround(first_s * kSampleRate) : every;
  const int64_t nxt = n < first ? first : first + ((n - first) / every + 1) * every;
  if (max_s > 0 && nxt > max_s * kSampleRate) return std::nullopt;
  return nxt;
}

struct SenseVoiceAsr::Impl {
  Ort::Session session{nullptr};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

namespace {

std::string metadata(const Ort::ModelMetadata& meta, const std::string& key) {
  Ort::AllocatorWithDefaultOptions alloc;
  auto value = meta.LookupCustomMetadataMapAllocated(key.c_str(), alloc);
  if (!value) throw std::runtime_error("模型 metadata 缺少欄位：" + key);
  return value.get();
}

std::vector<float> parse_floats(const std::string& csv) {
  std::vector<float> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(std::strtof(item.c_str(), nullptr));
  return out;
}

}  // namespace

SenseVoiceAsr::SenseVoiceAsr(const SenseVoiceConfig& config) : impl_(new Impl) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(config.num_threads);
  opts.SetInterOpNumThreads(1);
  // 關掉執行緒等待時的 spinning：與 Python 版相同（CPU 時間 −6~14 %，結果不變）
  if (!config.allow_spinning) opts.AddConfigEntry("session.intra_op.allow_spinning", "0");
  impl_->session = Ort::Session(ort_env(), config.model_path.c_str(), opts);

  Ort::ModelMetadata meta = impl_->session.GetModelMetadata();
  lfr_m_ = std::stoi(metadata(meta, "lfr_window_size"));
  lfr_n_ = std::stoi(metadata(meta, "lfr_window_shift"));
  neg_mean_ = parse_floats(metadata(meta, "neg_mean"));
  inv_std_ = parse_floats(metadata(meta, "inv_stddev"));
  if (neg_mean_.size() != static_cast<size_t>(80 * lfr_m_) || inv_std_.size() != neg_mean_.size())
    throw std::runtime_error("CMVN 維度與 LFR 設定不符");
  scale_ = std::stoi(metadata(meta, "normalize_samples")) ? 1.0f : 32768.0f;
  language_ = std::stoi(metadata(meta, "lang_" + config.language));
  text_norm_ = std::stoi(metadata(meta, config.use_itn ? "with_itn" : "without_itn"));

  std::ifstream in(config.tokens_path);
  if (!in) throw std::runtime_error("找不到 tokens 檔：" + config.tokens_path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t sp = line.rfind(' ');  // 「符號 id」，符號本身可能含空白
    if (sp == std::string::npos) continue;
    const size_t id = std::stoul(line.substr(sp + 1));
    if (id >= tokens_.size()) tokens_.resize(id + 1);
    tokens_[id] = line.substr(0, sp);
  }
  decoder_ = std::make_unique<GreedyDecoder>(tokens_);
}

SenseVoiceAsr::~SenseVoiceAsr() = default;

void SenseVoiceAsr::features(const float* audio, size_t n, std::vector<float>& out,
                             int& frames) const {
  std::vector<float> scaled(audio, audio + n);
  for (float& v : scaled) v *= scale_;
  std::vector<float> fb;
  fbank_.compute(scaled.data(), scaled.size(), fb);
  lfr_cmvn(fb, fbank_.num_frames(n), fbank_.num_bins(), lfr_m_, lfr_n_, neg_mean_, inv_std_, out,
           frames);
}

std::string SenseVoiceAsr::recognize(const std::vector<float>& audio) {
  ++num_decodes_;
  decoded_samples_ += static_cast<int64_t>(audio.size());
  std::vector<float> x;
  int frames = 0;
  features(audio.data(), audio.size(), x, frames);
  if (frames == 0) return "";

  const int64_t x_shape[3] = {1, frames, 80 * lfr_m_};
  const int64_t one[1] = {1};
  int32_t x_len = frames, lang = language_, norm = text_norm_;
  Ort::Value inputs[4] = {
      Ort::Value::CreateTensor<float>(impl_->mem, x.data(), x.size(), x_shape, 3),
      Ort::Value::CreateTensor<int32_t>(impl_->mem, &x_len, 1, one, 1),
      Ort::Value::CreateTensor<int32_t>(impl_->mem, &lang, 1, one, 1),
      Ort::Value::CreateTensor<int32_t>(impl_->mem, &norm, 1, one, 1)};
  const char* in_names[4] = {"x", "x_length", "language", "text_norm"};
  const char* out_names[1] = {"logits"};
  auto outputs =
      impl_->session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 4, out_names, 1);
  const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, T, vocab]
  return decoder_->decode(outputs[0].GetTensorData<float>(), static_cast<int>(shape[1]),
                          static_cast<int>(shape[2]));
}

AsrStream::AsrStream(Recognizer* recognizer, const std::optional<PartialSchedule>& schedule)
    : recognizer_(recognizer), schedule_(schedule) {
  if (schedule_) next_partial_ = schedule_->next_sample(0);
}

std::string AsrStream::decode() {
  if (audio_.empty()) return "";
  return recognizer_->recognize(audio_);
}

void AsrStream::accept_waveform(const float* chunk, size_t n) {
  if (finalized_) throw std::logic_error("stream 已 finalize，請重新建立");
  audio_.insert(audio_.end(), chunk, chunk + n);
  const int64_t have = static_cast<int64_t>(audio_.size());
  if (schedule_ && next_partial_ && have >= *next_partial_) {
    partial_ = decode();
    next_partial_ = schedule_->next_sample(have);
  }
}

std::string AsrStream::finalize() {
  if (finalized_) throw std::logic_error("stream 已 finalize");
  final_ = decode();
  finalized_ = true;
  return final_;
}

}  // namespace ozasr
