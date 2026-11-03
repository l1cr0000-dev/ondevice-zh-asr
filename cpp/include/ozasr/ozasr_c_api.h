/* 離線低延遲中文 ASR 辨識核心的純 C API（Swift / Kotlin JNI 用）。
 *
 * 用法：
 *   ozasr_config cfg; ozasr_config_default(&cfg);
 *   cfg.vad_model_path = "...silero_vad.onnx"; cfg.asr_model_path = "...model.int8.onnx"; cfg.tokens_path = "...tokens.txt";
 *   ozasr_handle* h = NULL;
 *   if (ozasr_create(&cfg, &h) != OZASR_OK) { ... }
 *   // 收音迴圈：16 kHz、單聲道、float32、範圍約 [-1, 1]，chunk 長度任意（建議 10 ms = 160 樣本）
 *   ozasr_accept_waveform(h, samples, n);
 *   ozasr_event ev;
 *   while (ozasr_next_event(h, &ev) == 1) { ... }
 *   ozasr_flush(h);   // 輸入結束，結束進行中的句子；之後要再用請 ozasr_reset
 *   ozasr_destroy(h);
 *
 * 執行緒：同一個 handle 不可同時從多個執行緒呼叫；不同 handle 互相獨立。
 * 事件的 text 指標只在下一次呼叫 ozasr_next_event / ozasr_accept_waveform / ozasr_flush / ozasr_reset 之前有效。
 * 時間欄位（秒，以 ozasr_create / ozasr_reset 後第一個樣本為 0）：speech_*、audio_*、audio_pos_s 是音訊時間軸上的位置；
 * emitted_s 是「模擬即時」的發出時刻（音訊依實際時長節奏到達 + 實測計算時間），用於離線評測，即時使用時請改用自己的牆鐘。
 */
#ifndef OZASR_C_API_H_
#define OZASR_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define OZASR_API __declspec(dllexport)
#else
#define OZASR_API __attribute__((visibility("default")))
#endif

typedef struct ozasr_handle ozasr_handle;

typedef enum {
  OZASR_OK = 0,
  OZASR_ERROR_INVALID_ARGUMENT = 1,
  OZASR_ERROR_MODEL = 2,    /* 模型檔缺失或載入失敗 */
  OZASR_ERROR_RUNTIME = 3   /* 推論或內部錯誤；訊息見 ozasr_last_error */
} ozasr_status;

typedef enum {
  OZASR_EVENT_START = 0,        /* VAD 判定語音開始，建立辨識 */
  OZASR_EVENT_PARTIAL = 1,      /* 說話中的 partial 結果（整段文字，會被後續結果取代） */
  OZASR_EVENT_PROVISIONAL = 2,  /* 句中停頓後的暫定結果；之後又開口則會被修正 */
  OZASR_EVENT_FINAL = 3         /* 定稿 */
} ozasr_event_kind;

typedef struct ozasr_config {
  const char* vad_model_path;  /* silero_vad.onnx */
  const char* asr_model_path;  /* SenseVoice model.int8.onnx */
  const char* tokens_path;     /* tokens.txt */
  const char* language;        /* "zh"（預設）、"auto"、"en"…，對應模型 metadata 的 lang_<language> */
  int32_t asr_num_threads;     /* 預設 2 */
  int32_t vad_num_threads;     /* 預設 2 */
  float vad_threshold;         /* 預設 0.5 */
  float vad_neg_threshold;     /* < 0 = max(threshold - 0.15, 0.01) */
  int32_t vad_min_speech_ms;   /* 預設 250 */
  int32_t vad_min_silence_ms;  /* 定稿靜音，預設 700 */
  int32_t vad_speech_pad_ms;   /* 預設 30 */
  int32_t vad_pause_ms;        /* 暫定結果的停頓長度，預設 150；0 = 關閉 */
  int32_t pre_roll_ms;         /* 預設 100 */
  int32_t provisional_redecode;/* 0 = finalize（預設）；1 = redecode */
  float partial_first_s;       /* 第一個 partial 的累積音訊長度，預設 0.48 */
  float partial_every_s;       /* 之後的間隔，預設 0.96；<= 0 = 不產生 partial */
  float partial_max_s;         /* 累積超過此長度後不再出 partial，預設 3.0；<= 0 = 不限 */
} ozasr_config;

typedef struct ozasr_event {
  int32_t kind;            /* ozasr_event_kind */
  int32_t utterance;       /* 句子編號，從 0 起 */
  const char* text;        /* UTF-8；START 為空字串 */
  double emitted_s;        /* 模擬即時的發出時刻 */
  double audio_pos_s;      /* 事件發出時已送入的音訊長度 */
  double speech_start_s;   /* VAD 起點（含 pad） */
  double audio_start_s;    /* 實際送入 ASR 的第一個樣本（含 pre-roll） */
  double speech_end_s;     /* 只有 FINAL，其餘為 -1 */
  double audio_end_s;      /* 只有 PROVISIONAL：重新解碼涵蓋的音訊終點，其餘為 -1 */
} ozasr_event;

typedef struct ozasr_stats {
  int64_t num_decodes;       /* ASR 整段推論的次數 */
  double decoded_audio_s;    /* 推論的音訊總長（重複辨識會重複計） */
} ozasr_stats;

OZASR_API const char* ozasr_version(void);
OZASR_API void ozasr_config_default(ozasr_config* config);
OZASR_API ozasr_status ozasr_create(const ozasr_config* config, ozasr_handle** out);
OZASR_API void ozasr_destroy(ozasr_handle* handle);
OZASR_API ozasr_status ozasr_reset(ozasr_handle* handle);
OZASR_API ozasr_status ozasr_accept_waveform(ozasr_handle* handle, const float* samples, size_t num_samples);
OZASR_API ozasr_status ozasr_flush(ozasr_handle* handle);
/* 取出一個事件：回傳 1 = 有事件（寫入 *event）、0 = 沒有、< 0 = 錯誤 */
OZASR_API int ozasr_next_event(ozasr_handle* handle, ozasr_event* event);
OZASR_API ozasr_status ozasr_get_stats(ozasr_handle* handle, ozasr_stats* stats);
/* 最近一次錯誤訊息（create 失敗時 handle 為 NULL，回傳全域的最後一次錯誤） */
OZASR_API const char* ozasr_last_error(const ozasr_handle* handle);

#ifdef __cplusplus
}
#endif

#endif /* OZASR_C_API_H_ */
