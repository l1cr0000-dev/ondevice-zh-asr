#include <vector>

#include "check.h"
#include "scripted_vad.h"

using namespace ozasr;

static std::vector<VadEvent> run(const std::vector<float>& probs, const VadConfig& cfg,
                                 size_t chunk, size_t total_samples = 0) {
  ScriptedVad model(probs);
  StreamingVad vad(&model, cfg);
  const size_t total = total_samples ? total_samples : probs.size() * 512;
  std::vector<float> audio(total, 0.0f);
  std::vector<VadEvent> events;
  for (size_t i = 0; i < total; i += chunk) {
    auto e = vad.accept_waveform(audio.data() + i, std::min(chunk, total - i));
    events.insert(events.end(), e.begin(), e.end());
  }
  auto e = vad.flush();
  events.insert(events.end(), e.begin(), e.end());
  return events;
}

int main() {
  VadConfig cfg{0.5f, -1.0f, 250, 700, 30, 150};
  const auto events = run(two_part_script(), cfg, 160);
  // 開始 / 停頓 / 恢復 / 再停頓 / 結束
  CHECK_EQ(events.size(), static_cast<size_t>(5));
  CHECK(events[0].kind == VadEvent::Start);
  CHECK_EQ(events[0].sample, 2560 - 480);  // 語音起點 5 frame − pad 30 ms
  CHECK_EQ(events[0].detected_at, 13 * 512);  // 語音滿 250 ms（8 frame）後才確認
  CHECK(events[1].kind == VadEvent::Pause);
  CHECK_EQ(events[1].sample, 25 * 512 + 480);
  CHECK_EQ(events[1].detected_at, 30 * 512);  // 靜音 5 frame = 160 ms ≥ 150 ms
  CHECK(events[2].kind == VadEvent::Resume);
  CHECK_EQ(events[2].sample, 31 * 512);
  CHECK(events[3].kind == VadEvent::Pause);
  CHECK(events[4].kind == VadEvent::End);
  CHECK_EQ(events[4].sample, 36 * 512 + 480);
  CHECK_EQ(events[4].detected_at, 58 * 512);  // 靜音 22 frame = 704 ms ≥ 700 ms

  // 結果與 chunk 大小無關
  for (size_t chunk : {1u, 7u, 160u, 512u, 5000u}) {
    const auto e2 = run(two_part_script(), cfg, chunk);
    CHECK_EQ(e2.size(), events.size());
    for (size_t i = 0; i < std::min(e2.size(), events.size()); ++i) {
      CHECK(e2[i].kind == events[i].kind);
      CHECK_EQ(e2[i].sample, events[i].sample);
      CHECK_EQ(e2[i].detected_at, events[i].detected_at);
    }
  }

  // 沒設 pause：只有開始與結束；語音段中的短靜音不切段
  VadConfig no_pause{0.5f, -1.0f, 250, 700, 30, 0};
  const auto e3 = run(two_part_script(), no_pause, 160);
  CHECK_EQ(e3.size(), static_cast<size_t>(2));
  CHECK(e3[0].kind == VadEvent::Start && e3[1].kind == VadEvent::End);

  // 遲滯：threshold 0.5、neg 0.35，0.4 不算靜音也不算語音，語音段不會被切開
  std::vector<float> hyst;
  hyst.insert(hyst.end(), 2, 0.0f);
  hyst.insert(hyst.end(), 10, 0.9f);
  hyst.insert(hyst.end(), 40, 0.4f);
  hyst.insert(hyst.end(), 5, 0.9f);
  hyst.insert(hyst.end(), 30, 0.0f);
  const auto e4 = run(hyst, no_pause, 512);
  CHECK_EQ(e4.size(), static_cast<size_t>(2));
  CHECK_EQ(e4[1].sample, 57 * 512 + 480);

  // 太短的語音（< min_speech）直接丟棄
  std::vector<float> shortp(3, 0.0f);
  shortp.insert(shortp.end(), 6, 0.9f);
  shortp.insert(shortp.end(), 30, 0.0f);
  CHECK(run(shortp, no_pause, 512).empty());

  // flush：語音進行中輸入結束，在已送入的樣本數處結束（pad 不超出音訊）
  std::vector<float> open(5, 0.0f);
  open.insert(open.end(), 20, 0.9f);
  const auto e5 = run(open, no_pause, 512);
  CHECK_EQ(e5.size(), static_cast<size_t>(2));
  CHECK(e5[1].kind == VadEvent::End);
  CHECK_EQ(e5[1].sample, 25 * 512);
  CHECK_EQ(e5[1].detected_at, 25 * 512);

  // 參數檢查
  bool threw = false;
  try { VadConfig{0.5f, -1.0f, 250, 100, 30, 150}.validate(); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  CHECK_NEAR(VadConfig().effective_neg_threshold(), 0.35, 1e-6);
  TEST_MAIN_END();
}
