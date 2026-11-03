#include "ozasr/ozasr_c_api.h"

#include <cmath>
#include <deque>
#include <exception>
#include <memory>
#include <string>

#include "common.h"
#include "ort_env.h"
#include "pipeline.h"
#include "sense_voice.h"
#include "vad.h"

namespace {
thread_local std::string g_last_error;
}

struct ozasr_handle {
  std::unique_ptr<ozasr::SileroVad> vad;
  std::unique_ptr<ozasr::SenseVoiceAsr> asr;
  std::unique_ptr<ozasr::Pipeline> pipeline;
  std::deque<ozasr::PipelineEvent> events;
  ozasr::PipelineEvent current{};  // next_event 回傳的 text 指標指向這裡
  std::string error;
};

namespace {

template <typename F>
ozasr_status guarded(ozasr_handle* h, F&& f) {
  try {
    f();
    return OZASR_OK;
  } catch (const std::invalid_argument& e) {
    (h ? h->error : g_last_error) = e.what();
    return OZASR_ERROR_INVALID_ARGUMENT;
  } catch (const Ort::Exception& e) {
    (h ? h->error : g_last_error) = e.what();
    return OZASR_ERROR_MODEL;
  } catch (const std::exception& e) {
    (h ? h->error : g_last_error) = e.what();
    return OZASR_ERROR_RUNTIME;
  } catch (...) {
    (h ? h->error : g_last_error) = "未知錯誤";
    return OZASR_ERROR_RUNTIME;
  }
}

void push(ozasr_handle* h, std::vector<ozasr::PipelineEvent> evs) {
  for (auto& e : evs) h->events.push_back(std::move(e));
}

}  // namespace

extern "C" {

const char* ozasr_version(void) { return "0.1.0"; }

void ozasr_config_default(ozasr_config* c) {
  *c = ozasr_config{};
  c->language = "zh";
  c->asr_num_threads = 2;
  c->vad_num_threads = 2;
  c->vad_threshold = 0.5f;
  c->vad_neg_threshold = -1.0f;
  c->vad_min_speech_ms = 250;
  c->vad_min_silence_ms = 700;
  c->vad_speech_pad_ms = 30;
  c->vad_pause_ms = 150;
  c->pre_roll_ms = 100;
  c->provisional_redecode = 0;
  c->partial_first_s = 0.48f;
  c->partial_every_s = 0.96f;
  c->partial_max_s = 3.0f;
}

ozasr_status ozasr_create(const ozasr_config* c, ozasr_handle** out) {
  if (!c || !out || !c->vad_model_path || !c->asr_model_path || !c->tokens_path) {
    g_last_error = "config / out / 模型路徑不可為 NULL";
    return OZASR_ERROR_INVALID_ARGUMENT;
  }
  *out = nullptr;
  auto h = std::make_unique<ozasr_handle>();
  ozasr_status st = guarded(nullptr, [&] {
    ozasr::PipelineConfig pc;
    pc.vad = {c->vad_threshold, c->vad_neg_threshold, c->vad_min_speech_ms,
              c->vad_min_silence_ms, c->vad_speech_pad_ms, c->vad_pause_ms};
    pc.vad.validate();
    pc.pre_roll_ms = c->pre_roll_ms;
    pc.provisional = c->provisional_redecode ? ozasr::PipelineConfig::Redecode
                                             : ozasr::PipelineConfig::Finalize;
    if (c->partial_every_s > 0) {
      pc.schedule = ozasr::PartialSchedule{c->partial_every_s, c->partial_first_s, c->partial_max_s};
    } else {
      pc.schedule.reset();
    }
    ozasr::SenseVoiceConfig sc;
    sc.model_path = c->asr_model_path;
    sc.tokens_path = c->tokens_path;
    sc.language = c->language ? c->language : "zh";
    sc.num_threads = c->asr_num_threads;
    try {
      h->vad = std::make_unique<ozasr::SileroVad>(c->vad_model_path, c->vad_num_threads);
      h->asr = std::make_unique<ozasr::SenseVoiceAsr>(sc);
    } catch (const std::invalid_argument&) {
      throw;
    } catch (const Ort::Exception&) {
      throw;
    } catch (const std::exception& e) {
      throw Ort::Exception(e.what(), ORT_FAIL);
    }
    h->pipeline = std::make_unique<ozasr::Pipeline>(h->vad.get(), h->asr.get(), pc);
  });
  if (st != OZASR_OK) return st;
  *out = h.release();
  return OZASR_OK;
}

void ozasr_destroy(ozasr_handle* h) { delete h; }

ozasr_status ozasr_reset(ozasr_handle* h) {
  if (!h) return OZASR_ERROR_INVALID_ARGUMENT;
  return guarded(h, [&] {
    h->pipeline->reset();
    h->events.clear();
  });
}

ozasr_status ozasr_accept_waveform(ozasr_handle* h, const float* samples, size_t n) {
  if (!h || (!samples && n)) return OZASR_ERROR_INVALID_ARGUMENT;
  return guarded(h, [&] { push(h, h->pipeline->accept_waveform(samples, n)); });
}

ozasr_status ozasr_flush(ozasr_handle* h) {
  if (!h) return OZASR_ERROR_INVALID_ARGUMENT;
  return guarded(h, [&] { push(h, h->pipeline->flush()); });
}

int ozasr_next_event(ozasr_handle* h, ozasr_event* ev) {
  if (!h || !ev) return -1;
  if (h->events.empty()) return 0;
  h->current = std::move(h->events.front());
  h->events.pop_front();
  const auto& e = h->current;
  ev->kind = e.kind;
  ev->utterance = e.utterance;
  ev->text = e.text.c_str();
  ev->emitted_s = e.emitted_s;
  ev->audio_pos_s = e.audio_pos_s;
  ev->speech_start_s = e.speech_start_s;
  ev->audio_start_s = e.audio_start_s;
  ev->speech_end_s = std::isnan(e.speech_end_s) ? -1.0 : e.speech_end_s;
  ev->audio_end_s = std::isnan(e.audio_end_s) ? -1.0 : e.audio_end_s;
  return 1;
}

ozasr_status ozasr_get_stats(ozasr_handle* h, ozasr_stats* s) {
  if (!h || !s) return OZASR_ERROR_INVALID_ARGUMENT;
  s->num_decodes = h->asr->num_decodes();
  s->decoded_audio_s = h->asr->decoded_audio_s();
  return OZASR_OK;
}

const char* ozasr_last_error(const ozasr_handle* h) {
  return h ? h->error.c_str() : g_last_error.c_str();
}

}  // extern "C"
