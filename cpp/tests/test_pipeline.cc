#include <string>
#include <vector>

#include "check.h"
#include "pipeline.h"
#include "scripted_vad.h"

using namespace ozasr;

// 回傳「n<樣本數>」，可由文字得知每次辨識看到多少音訊
struct LengthRecognizer : Recognizer {
  std::vector<size_t> calls;
  std::string recognize(const std::vector<float>& a) override {
    calls.push_back(a.size());
    return "n" + std::to_string(a.size());
  }
};

static std::vector<PipelineEvent> run(Pipeline& p, size_t total, size_t chunk) {
  std::vector<float> audio(total, 0.0f);
  std::vector<PipelineEvent> events;
  for (size_t i = 0; i < total; i += chunk) {
    auto e = p.accept_waveform(audio.data() + i, std::min(chunk, total - i));
    events.insert(events.end(), e.begin(), e.end());
  }
  auto e = p.flush();
  events.insert(events.end(), e.begin(), e.end());
  return events;
}

int main() {
  PipelineConfig cfg;
  cfg.schedule = PartialSchedule{0.2, 0.1, 0.0};  // 0.1 s、0.3 s、0.5 s…各辨識一次當 partial
  double fake_time = 0;
  ScriptedVad vad(two_part_script());
  LengthRecognizer asr;
  Pipeline p(&vad, &asr, cfg, [&] { return fake_time; });
  const auto events = run(p, 66 * 512, 160);

  // 事件順序：start → partial… → provisional → partial（恢復後重新解碼）… → final
  CHECK(events.front().kind == PipelineEvent::Start);
  CHECK(events.back().kind == PipelineEvent::Final);
  int provisionals = 0, finals = 0;
  for (const auto& e : events) {
    provisionals += e.kind == PipelineEvent::Provisional;
    finals += e.kind == PipelineEvent::Final;
  }
  // 停頓兩次：第一次後恢復說話（暫定結果被修正），第二次後靜音滿 700 ms 定稿
  CHECK_EQ(provisionals, 2);
  CHECK_EQ(finals, 1);

  // Start：VAD 起點 2080（含 pad）、pre-roll 100 ms 往前 1600 但不早於 0 → 從 480 送入
  CHECK_NEAR(events.front().speech_start_s, 2080.0 / 16000, 1e-9);
  CHECK_NEAR(events.front().audio_start_s, 480.0 / 16000, 1e-9);

  // 第一個暫定結果：finalize 時看到 [480, 15360) = 14880 個樣本
  const PipelineEvent* first_prov = nullptr;
  for (const auto& e : events)
    if (e.kind == PipelineEvent::Provisional) { first_prov = &e; break; }
  CHECK(first_prov != nullptr);
  CHECK_EQ(first_prov->text, std::string("n14880"));
  CHECK_NEAR(first_prov->audio_end_s, 15360.0 / 16000, 1e-9);

  // 定稿：第二次停頓（frame 40，音訊到 20992）後 stream 已 finalize 並不再收音，沒有恢復說話，
  // 以暫定結果定稿 → [480, 20992) = 20512（"finalize" 做法的語意：最終文字 = 最後一次暫定結果）
  const PipelineEvent& fin = events.back();
  CHECK_EQ(fin.text, std::string("n20512"));
  CHECK_EQ(events[events.size() - 2].text, std::string("n20512"));
  CHECK_NEAR(fin.speech_end_s, (36.0 * 512 + 480) / 16000, 1e-9);
  CHECK_EQ(p.utterances().size(), static_cast<size_t>(1));
  CHECK_EQ(p.utterances()[0].text, std::string("n20512"));
  CHECK_EQ(p.utterances()[0].provisionals.size(), static_cast<size_t>(2));
  CHECK_EQ(p.utterances()[0].resume_redecode_s.size(), static_cast<size_t>(1));

  // 每次送入的 chunk 大小不同，音訊時間軸上的結果相同
  ScriptedVad vad2(two_part_script());
  LengthRecognizer asr2;
  Pipeline p2(&vad2, &asr2, cfg, [&] { return fake_time; });
  const auto events2 = run(p2, 66 * 512, 100);
  CHECK_EQ(events2.back().text, fin.text);
  CHECK_EQ(p2.utterances()[0].provisionals.size(), static_cast<size_t>(2));

  // 模擬即時時刻：chunk 的最後一個樣本到達後才處理（時鐘不動 → emitted_s = 音訊位置）
  CHECK_NEAR(events.front().emitted_s, events.front().audio_pos_s, 1e-9);

  // redecode 做法：停頓時另開 stream，原 stream 不中斷，結果相同
  PipelineConfig rc = cfg;
  rc.provisional = PipelineConfig::Redecode;
  ScriptedVad vad3(two_part_script());
  LengthRecognizer asr3;
  Pipeline p3(&vad3, &asr3, rc, [&] { return fake_time; });
  const auto events3 = run(p3, 66 * 512, 160);
  // 原 stream 不中斷，一路收音到 End 事件發出時（58 frame）→ [480, 29696) = 29216
  CHECK_EQ(events3.back().text, std::string("n29216"));
  CHECK_EQ(p3.utterances()[0].provisionals.size(), static_cast<size_t>(2));
  CHECK_EQ(p3.utterances()[0].provisionals[0].text, std::string("n14880"));

  // reset 後可重用
  p.reset();
  CHECK(p.utterances().empty());
  CHECK(!p.in_speech());
  // finalize_utterance：句子進行中放開 → 立刻定稿，且之後可繼續送入（不需 reset）
  {
    double t = 0;
    ScriptedVad v(two_part_script());
    LengthRecognizer a;
    Pipeline q(&v, &a, cfg, [&] { return t; });
    std::vector<float> audio(20 * 512, 0.0f);
    std::vector<PipelineEvent> ev;
    for (size_t i = 0; i < audio.size(); i += 160) {
      auto e = q.accept_waveform(audio.data() + i, std::min<size_t>(160, audio.size() - i));
      ev.insert(ev.end(), e.begin(), e.end());
    }
    CHECK(q.in_speech());
    const auto fin_ev = q.finalize_utterance();
    CHECK_EQ(fin_ev.size(), static_cast<size_t>(1));
    CHECK(fin_ev[0].kind == PipelineEvent::Final);
    CHECK(fin_ev[0].text.size() > 1);  // "n<樣本數>"
    CHECK_NEAR(fin_ev[0].audio_pos_s, 20.0 * 512 / 16000, 1e-9);  // 在放開當下就定稿，沒有等 700 ms
    CHECK(!q.in_speech());
    CHECK_EQ(q.utterances().size(), static_cast<size_t>(1));
    CHECK(q.finalize_utterance().empty());
    // 之後可繼續送入
    auto more = q.accept_waveform(audio.data(), 160);
    CHECK(more.empty());
  }

  // 句中停頓（已出暫定結果）時放開：直接以暫定結果定稿，不再解碼
  {
    double t = 0;
    ScriptedVad v(two_part_script());
    LengthRecognizer a;
    Pipeline q(&v, &a, cfg, [&] { return t; });
    std::vector<float> audio(31 * 512, 0.0f);
    std::vector<PipelineEvent> ev;
    for (size_t i = 0; i < audio.size(); i += 160) {
      auto e = q.accept_waveform(audio.data() + i, std::min<size_t>(160, audio.size() - i));
      ev.insert(ev.end(), e.begin(), e.end());
    }
    CHECK(ev.back().kind == PipelineEvent::Provisional);
    const size_t decodes = a.calls.size();
    const auto fin_ev = q.finalize_utterance();
    CHECK_EQ(fin_ev.size(), static_cast<size_t>(1));
    CHECK_EQ(fin_ev[0].text, ev.back().text);
    CHECK_EQ(a.calls.size(), decodes);
  }

  // 偽串流排程：first 0.48、every 0.96、max 3.0
  PartialSchedule s{0.96, 0.48, 3.0};
  CHECK_EQ(*s.next_sample(0), 7680);
  CHECK_EQ(*s.next_sample(7680), 7680 + 15360);
  CHECK_EQ(*s.next_sample(23039), 23040);
  CHECK_EQ(*s.next_sample(38399), 38400);
  CHECK(!s.next_sample(38400).has_value());  // 下一次 53760 > 3 s
  TEST_MAIN_END();
}
