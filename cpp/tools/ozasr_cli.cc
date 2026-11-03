// 命令列工具：讀 16 kHz wav，以 chunk（預設 10 ms）模擬即時送入，每個檔案輸出一行 JSON（事件與句子）。
// 多個檔案時模型只載入一次；--warmup N 先跑前 N 個檔案（不輸出、不計時），最後一行是 summary（RTF、峰值 RSS）。
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include "pipeline.h"
#include "wav.h"

using namespace ozasr;

namespace {

const char* kUsage = R"(用法：ozasr_cli [選項] <wav>...   或   ozasr_cli [選項] --list <檔案清單.txt>
  --models-dir DIR        模型根目錄（預設 models）；含 silero-vad-v6.2.3/ 與 sherpa-onnx-sense-voice-.../
  --vad-model / --asr-model / --tokens PATH   個別指定模型檔
  --list FILE             每行一個 wav 路徑
  --threads N             ASR 與 VAD 推論執行緒數（預設 2）
  --chunk-ms N            每次送入的音訊長度（預設 10）
  --lead-silence-ms N     每條音訊前補的靜音（預設 0）
  --tail-silence-ms N     每條音訊後補的靜音（預設 0）
  --warmup N              前 N 個檔案暖機：不輸出、不計入 summary（預設 0）
  --threshold F --min-speech-ms N --min-silence-ms N --speech-pad-ms N --pause-ms N --pre-roll-ms N
  --provisional finalize|redecode
  --partial-first-s F --partial-every-s F --partial-max-s F   （every <= 0 關閉 partial）
  --language zh|auto|...  （預設 zh）
輸出：每個檔案一行 JSON {"type":"file",...}；最後多一行 {"type":"summary",...}。
)";

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); }
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += static_cast<char>(c);
  }
  return o + "\"";
}

std::string num(double v) {
  if (std::isnan(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.6f", v);
  return b;
}

const char* kind_name(PipelineEvent::Kind k) {
  switch (k) {
    case PipelineEvent::Start: return "start";
    case PipelineEvent::Partial: return "partial";
    case PipelineEvent::Provisional: return "provisional";
    default: return "final";
  }
}

double peak_rss_mb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return ru.ru_maxrss / 1048576.0;  // bytes
#else
  return ru.ru_maxrss / 1024.0;  // KB
#endif
}

}  // namespace

int main(int argc, char** argv) {
  std::string models_dir = "models", vad_path, asr_path, tokens_path, language = "zh", list_file;
  int threads = 2, chunk_ms = 10, lead_ms = 0, tail_ms = 0, warmup = 0;
  PipelineConfig cfg;
  std::vector<std::string> wavs;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s 缺少參數\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
    else if (a == "--models-dir") models_dir = val();
    else if (a == "--vad-model") vad_path = val();
    else if (a == "--asr-model") asr_path = val();
    else if (a == "--tokens") tokens_path = val();
    else if (a == "--list") list_file = val();
    else if (a == "--threads") threads = std::atoi(val());
    else if (a == "--chunk-ms") chunk_ms = std::atoi(val());
    else if (a == "--lead-silence-ms") lead_ms = std::atoi(val());
    else if (a == "--tail-silence-ms") tail_ms = std::atoi(val());
    else if (a == "--warmup") warmup = std::atoi(val());
    else if (a == "--threshold") cfg.vad.threshold = static_cast<float>(std::atof(val()));
    else if (a == "--min-speech-ms") cfg.vad.min_speech_ms = std::atoi(val());
    else if (a == "--min-silence-ms") cfg.vad.min_silence_ms = std::atoi(val());
    else if (a == "--speech-pad-ms") cfg.vad.speech_pad_ms = std::atoi(val());
    else if (a == "--pause-ms") cfg.vad.pause_ms = std::atoi(val());
    else if (a == "--pre-roll-ms") cfg.pre_roll_ms = std::atoi(val());
    else if (a == "--provisional")
      cfg.provisional = std::string(val()) == "redecode" ? PipelineConfig::Redecode : PipelineConfig::Finalize;
    else if (a == "--partial-first-s" && cfg.schedule) cfg.schedule->first_s = std::atof(val());
    else if (a == "--partial-every-s") {
      const double v = std::atof(val());
      if (v <= 0) cfg.schedule.reset();
      else if (cfg.schedule) cfg.schedule->every_s = v;
    }
    else if (a == "--partial-max-s" && cfg.schedule) cfg.schedule->max_s = std::atof(val());
    else if (a == "--language") language = val();
    else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "未知選項 %s\n%s", a.c_str(), kUsage); return 2; }
    else wavs.push_back(a);
  }
  if (!list_file.empty()) {
    std::ifstream in(list_file);
    std::string line;
    while (std::getline(in, line))
      if (!line.empty()) wavs.push_back(line);
  }
  if (wavs.empty()) { std::fputs(kUsage, stderr); return 2; }
  const std::string sv = models_dir + "/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";
  if (vad_path.empty()) vad_path = models_dir + "/silero-vad-v6.2.3/silero_vad.onnx";
  if (asr_path.empty()) asr_path = sv + "/model.int8.onnx";
  if (tokens_path.empty()) tokens_path = sv + "/tokens.txt";

  try {
    cfg.vad.validate();
    const auto t_load = std::chrono::steady_clock::now();
    SileroVad vad(vad_path, threads);
    SenseVoiceConfig sc;
    sc.model_path = asr_path;
    sc.tokens_path = tokens_path;
    sc.language = language;
    sc.num_threads = threads;
    SenseVoiceAsr asr(sc);
    Pipeline pipeline(&vad, &asr, cfg);
    const double load_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
    const double rss_loaded = peak_rss_mb();

    const size_t chunk = static_cast<size_t>(ms_to_samples(chunk_ms));
    double total_audio = 0, total_compute = 0;
    int counted = 0;
    for (size_t f = 0; f < wavs.size(); ++f) {
      std::vector<float> audio = load_wav(wavs[f]);
      audio.insert(audio.begin(), static_cast<size_t>(ms_to_samples(lead_ms)), 0.0f);
      audio.insert(audio.end(), static_cast<size_t>(ms_to_samples(tail_ms)), 0.0f);
      pipeline.reset();
      std::vector<PipelineEvent> events;
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i < audio.size(); i += chunk) {
        auto evs = pipeline.accept_waveform(audio.data() + i, std::min(chunk, audio.size() - i));
        for (auto& e : evs) events.push_back(std::move(e));
      }
      for (auto& e : pipeline.flush()) events.push_back(std::move(e));
      const double compute =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (static_cast<int>(f) < warmup) continue;
      const double dur = static_cast<double>(audio.size()) / kSampleRate;
      total_audio += dur;
      total_compute += compute;
      ++counted;

      std::string o = "{\"type\":\"file\",\"wav\":" + json_str(wavs[f]) + ",\"duration_s\":" + num(dur) +
                      ",\"compute_s\":" + num(compute) + ",\"events\":[";
      for (size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];
        o += (i ? "," : "");
        o += std::string("{\"type\":\"") + kind_name(e.kind) + "\",\"utterance\":" +
             std::to_string(e.utterance) + ",\"text\":" + json_str(e.text) +
             ",\"emitted_s\":" + num(e.emitted_s) + ",\"audio_pos_s\":" + num(e.audio_pos_s) +
             ",\"speech_start_s\":" + num(e.speech_start_s) +
             ",\"audio_start_s\":" + num(e.audio_start_s) +
             ",\"speech_end_s\":" + num(e.speech_end_s) + ",\"audio_end_s\":" + num(e.audio_end_s) + "}";
      }
      o += "],\"utterances\":[";
      const auto& us = pipeline.utterances();
      for (size_t i = 0; i < us.size(); ++i) {
        const auto& u = us[i];
        o += (i ? "," : "");
        o += "{\"index\":" + std::to_string(u.index) + ",\"text\":" + json_str(u.text) +
             ",\"speech_start_s\":" + num(u.speech_start_s) + ",\"speech_end_s\":" + num(u.speech_end_s) +
             ",\"audio_start_s\":" + num(u.audio_start_s) + ",\"first_token_s\":" + num(u.first_token_s) +
             ",\"end_emitted_s\":" + num(u.end_emitted_s) + ",\"final_s\":" + num(u.final_s) + "}";
      }
      o += "]}";
      std::puts(o.c_str());
    }
    {
      std::printf(
          "{\"type\":\"summary\",\"files\":%d,\"audio_s\":%s,\"compute_s\":%s,\"rtf\":%s,"
          "\"load_s\":%s,\"rss_after_load_mb\":%s,\"peak_rss_mb\":%s,\"num_decodes\":%lld,"
          "\"decoded_audio_s\":%s,\"threads\":%d,\"chunk_ms\":%d}\n",
          counted, num(total_audio).c_str(), num(total_compute).c_str(),
          num(total_audio > 0 ? total_compute / total_audio : kNone).c_str(), num(load_s).c_str(),
          num(rss_loaded).c_str(), num(peak_rss_mb()).c_str(),
          static_cast<long long>(asr.num_decodes()), num(asr.decoded_audio_s()).c_str(), threads,
          chunk_ms);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "錯誤：%s\n", e.what());
    return 1;
  }
  return 0;
}
