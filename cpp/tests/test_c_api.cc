#include <cstring>

#include "check.h"
#include "ozasr/ozasr_c_api.h"

int main() {
  ozasr_config cfg;
  ozasr_config_default(&cfg);
  CHECK_EQ(cfg.vad_min_silence_ms, 700);
  CHECK_EQ(cfg.vad_pause_ms, 150);
  CHECK_NEAR(cfg.partial_first_s, 0.48, 1e-6);
  CHECK(std::strlen(ozasr_version()) > 0);

  // 參數檢查
  ozasr_handle* h = nullptr;
  CHECK_EQ(ozasr_create(nullptr, &h), OZASR_ERROR_INVALID_ARGUMENT);
  CHECK_EQ(ozasr_create(&cfg, &h), OZASR_ERROR_INVALID_ARGUMENT);  // 沒給模型路徑
  CHECK(h == nullptr);

  // 模型不存在 → OZASR_ERROR_MODEL 與訊息，不崩潰
  cfg.vad_model_path = "/nonexistent/silero_vad.onnx";
  cfg.asr_model_path = "/nonexistent/model.onnx";
  cfg.tokens_path = "/nonexistent/tokens.txt";
  CHECK_EQ(ozasr_create(&cfg, &h), OZASR_ERROR_MODEL);
  CHECK(h == nullptr);
  CHECK(std::strlen(ozasr_last_error(nullptr)) > 0);

  // 不合法的 VAD 參數
  cfg.vad_pause_ms = 800;
  CHECK_EQ(ozasr_create(&cfg, &h), OZASR_ERROR_INVALID_ARGUMENT);

  CHECK_EQ(ozasr_finalize_utterance(nullptr), OZASR_ERROR_INVALID_ARGUMENT);

  // NULL handle 的呼叫安全
  ozasr_destroy(nullptr);
  CHECK_EQ(ozasr_accept_waveform(nullptr, nullptr, 0), OZASR_ERROR_INVALID_ARGUMENT);
  ozasr_event ev;
  CHECK(ozasr_next_event(nullptr, &ev) < 0);
  TEST_MAIN_END();
}
