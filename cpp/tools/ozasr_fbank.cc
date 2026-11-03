// 對齊驗證用：wav → log-mel（frames × 80 float32 小端，不含檔頭）；給 --asr-model 時另輸出 LFR + CMVN 特徵。
// 用法：ozasr_fbank in.wav fbank.f32 [--asr-model model.int8.onnx --tokens tokens.txt --features feats.f32]
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "sense_voice.h"
#include "wav.h"

using namespace ozasr;

static void write_floats(const std::string& path, const std::vector<float>& v) {
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(v.data()),
                                              static_cast<std::streamsize>(v.size() * 4));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fputs("用法：ozasr_fbank in.wav fbank.f32 [--asr-model M --tokens T --features feats.f32]\n", stderr);
    return 2;
  }
  std::string asr, tokens, feats;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--asr-model")) asr = argv[i + 1];
    else if (!std::strcmp(argv[i], "--tokens")) tokens = argv[i + 1];
    else if (!std::strcmp(argv[i], "--features")) feats = argv[i + 1];
  }
  try {
    std::vector<float> audio = load_wav(argv[1]);
    std::vector<float> scaled(audio);
    for (float& v : scaled) v *= 32768.0f;
    Fbank fbank;
    std::vector<float> out;
    fbank.compute(scaled.data(), scaled.size(), out);
    write_floats(argv[2], out);
    if (!asr.empty() && !feats.empty()) {
      SenseVoiceConfig sc;
      sc.model_path = asr;
      sc.tokens_path = tokens;
      SenseVoiceAsr model(sc);
      std::vector<float> f;
      int frames = 0;
      model.features(audio.data(), audio.size(), f, frames);
      write_floats(feats, f);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "錯誤：%s\n", e.what());
    return 1;
  }
  return 0;
}
