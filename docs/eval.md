# 評測說明

全鏈路（VAD + 串流 ASR）的評測方法、指標定義與文字正規化規則。程式在 `eval/`，資料準備在 `scripts/prepare_eval_data.py`。

```bash
uv run scripts/prepare_eval_data.py      # 下載並抽樣評測集到 data/（一次即可，可重複執行）
uv run eval/run_eval.py                  # 預設設定跑一次
uv run eval/run_eval.py --model x-asr-480ms-int8 zipformer-zh-14m \
    --min-silence-ms 100 300 500 800 --pre-roll-ms 0 100 200 300 --tag sweep   # 參數掃描
```

多個值的參數（`--model`、`--min-silence-ms`、`--pre-roll-ms`、`--chunk-ms`、`--threads`）取笛卡兒積。
`--limit N` 只跑前 N 句（除錯用）。`--help` 列出全部參數與預設值。

## 評測資料

| 項目 | 值 |
|---|---|
| 母體 | AISHELL-1 test set，20 位說話人、7176 句（腳本會檢查句數） |
| 抽樣 | `sorted(random.Random(20261030).sample(sorted(全部 ID), 200))` |
| 規模 | 200 句、17.2 分鐘、20 位說話人全部涵蓋 |
| 音訊來源 | Hugging Face `shenyunhang/AISHELL-1`，固定 revision `2724409d53`，逐檔以 LFS SHA-256 校驗 |
| 標註來源 | 同 repo 的 `aishell_transcript_v0.8.txt`，SHA-256 `b5f33b9e…`（寫死在腳本中） |
| 輸出 | `data/aishell1_test_200/{wav/, manifest.jsonl, meta.json}`，不進 git |

選這個來源的原因：官方 `AISHELL/AISHELL-1` 的 Hugging Face repo 只上傳了 100 位訓練集說話人，沒有 test set；OpenSLR SLR33 只提供 15 GB 的完整壓縮檔。`shenyunhang/AISHELL-1` 保留官方的 `data_aishell/wav/test/<說話人>/<句子>.wav` 結構，可以只下載抽中的 200 個檔案（約 33 MB）。它是第三方鏡像，不是官方發布；句數與標註檔大小都與官方一致，但音訊內容無法對官方 checksum 驗證。

## 執行方式

- 每個設定在**獨立子程序**中載入模型、跑完全部句子、量峰值記憶體；父程序只做 CER 與彙整。
- 每條 wav 用 `src.pipeline.transcribe` 以 `chunk_ms`（預設 10 ms）切塊送入，**模擬即時**：一個 chunk 要等最後一個樣本「到達」（依音訊實際時長）才能送入，送入後的計算時間實際量測並累加，計算比音訊慢時會自然積壓。不真的 sleep。
- 一條 wav 被 VAD 切成多段時，各段文字以空白連接後當作這條的假設文字（正規化會去掉空白）。
- 單執行緒（VAD 與 ASR 的 ONNX Runtime intra-op 都是 1）、CPU。

## 文字正規化（CER 前）

參考與假設都經過**同一個**函式 `eval.text_norm.normalize`，依序：

1. **NFKC**：全形英數與標點轉半形（`ＡＢＣ１２３` → `ABC123`、`，` → `,`），相容字元轉標準形。
2. **英文轉小寫**。
3. **繁轉簡**：OpenCC `t2s`（AISHELL 標註是簡體）。
4. **去標點與符號**：Unicode 類別 `P*`（標點）與 `S*`（符號，如 `%`、`~`）的字元換成空白。換成空白而非刪除，避免 `hello,world` 黏成一個英文詞。
5. **切 token**：連續的 ASCII 英數為一個 token（一個英文詞或一串數字），其他非空白字元各一個 token；空白（含 AISHELL 標註中的分詞空白、全形空白）只用來分隔、不計入。

中文因此是**按字**計算；英文按詞計算，避免 `CLOUD` 錯一個字母就算 5 個錯。AISHELL-1 標註沒有英文與阿拉伯數字，這條規則只影響模型輸出英文的情況。

**不做**：中文數字與阿拉伯數字互轉（ITN）、同音字 / 異體字合併、語氣詞刪除。規則的測試在 `tests/test_eval.py`。

## 指標定義

### CER

`CER = (S + D + I) / N`，以上述 token 為單位、用 jiwer 計算編輯距離。報告**語料級** CER：全部句子的錯誤數總和 / 參考 token 總數，不是逐句 CER 的平均。VAD 漏掉整句時，假設為空，全部算刪除。

### RTF

`RTF = Σ 每句處理時間 / Σ 每句音訊時長`。每句處理時間 = `transcribe()` 的牆鐘時間，涵蓋 VAD、ASR 解碼、finalize（含補零）；**不含**讀 wav 與模型載入。正式計時前先跑前 N 句暖機（預設 N = 3，結果 JSON 記錄在 `config.warmup`），暖機結果丟棄、這 N 句之後仍正常計入。

注意 10 ms chunk 的 Python 呼叫開銷會計入 RTF；chunk 越大 RTF 越低，與延遲需一起看。

### 首字延遲

- **起點**：VAD 判定的語音起點，即該句段在音訊時間軸上的起點 `Utterance.speech_start_s`。這是 VAD 狀態機判定為語音的第一個 frame 再往前延伸 `speech_pad_ms`（30 ms）的位置，**不是**強制對齊的真實起點，也**不是** VAD 發出「開始」事件的時刻。
- **終點**：第一個非空 partial 結果輸出的模擬即時時刻（若 partial 一直為空、直到 finalize 才出字，取 finalize 完成時刻）。
- 因此首字延遲包含：VAD 確認語音所需的等待（約 `min_speech_ms`，結果 JSON 另列 `vad_start_delay_ms`）、ASR 等待第一個 chunk 收滿、以及計算時間。
- **每條音訊取一個值**：該條第一個有輸出文字的句段。這樣不同 `min_silence_ms` 下樣本數固定為 200，可以直接比較；沒有任何輸出的句子不計入，數量另列 `no_text_utterances`。逐段數值保留在結果 JSON 的 `utterances[].segments`。
- 報告平均與 P90（另存 P50 / P99），單位 ms。分位數用 numpy 線性內插。
- `pre_roll_ms` 只改變送入 ASR 的起點，不改變上述起點，因此不同 pre-roll 的首字延遲可直接比較。

### 尾字延遲（輔助）

每條音訊最後一個句段：從 VAD 語音終點（`speech_end_s`，含 padding）到最終結果輸出。包含 VAD 等待 `min_silence_ms` 確認結束，以及 finalize 的補零解碼。

### 峰值記憶體

子程序結束前的 `resource.getrusage(RUSAGE_SELF).ru_maxrss`（macOS 單位是 bytes，Linux 是 KB，程式已分別換算），單位 MB。包含 Python 直譯器與 numpy / onnxruntime 本身；另記錄模型載入前與載入後的 RSS，兩者之差約等於模型佔用。

### 模型大小

推論所需全部檔案的磁碟大小：Silero VAD onnx + ASR encoder / decoder / joiner + `tokens.txt`。結果 JSON 的 `model_files` 逐檔列出路徑、大小與精度（依檔名：`.int8.` 為 INT8，其餘 onnx 為 FP32）。

| 模型 | 精度 | 參數量 |
|---|---|---|
| zipformer-zh-14m | FP32 | 約 14M（模型名稱） |
| x-asr-480ms-int8 | encoder / joiner INT8，decoder FP32 | 未公開，尚未統計 |
| silero-vad-v6.2.3 | FP32 | 約 2M（上游說明） |

### 每條切段數

每條 wav 被 VAD 切成幾段。AISHELL-1 每條都是一句話，理想值是 1；平均值與 `3+` 段的比例上升代表句中停頓被切開（「切碎」），會讓 ASR 丟失上下文、切點附近容易出錯。報告平均、最大值與分佈（0 / 1 / 2 / 3+ 段）。0 段代表 VAD 整句漏掉。

## 結果檔

每個設定一個檔案：`eval/results/<日期>_<設定名>.json`，設定名為 `<模型>_sil<min_silence_ms>_pre<pre_roll_ms>_chunk<chunk_ms>_t<執行緒數>`；掃描另存總表 `<日期>_summary_<tag>.json`。`eval/results/` 不進 git。

| 欄位 | 內容 |
|---|---|
| `config` | 模型（含 sherpa-onnx 檔名與解碼方式）、VAD 模型與全部參數、pre-roll、chunk、執行緒數、暖機句數、manifest 路徑 |
| `environment` | commit（與是否有未提交改動）、機型、晶片、macOS 版本、Python 與套件版本 |
| `metrics` | 上述全部指標與 CER 的 S / D / I / N |
| `model_files` | 每個模型檔的路徑、大小、精度 |
| `utterances` | 每句的參考、假設、正規化後文字、處理時間、每個句段的時間戳與延遲 |

## 已知限制

- 語音起點用 VAD，不是強制對齊，VAD 本身的起點誤差（約 ±50 ms）會進入首字延遲。
- 延遲是模擬即時，假設音訊準時到達、沒有其他程序搶 CPU；真實麥克風輸入另有音訊驅動的緩衝延遲。
- 200 句只佔 test set 的 2.8%。單一設定的 CER bootstrap 95% 信賴區間約 ±1.1 個百分點（X-ASR，CER 約 5%）/ ±2 個百分點（zh-14M，約 15%）。比較兩個設定時用同一批句子做**成對** bootstrap，靈敏得多（例如 0.4 個百分點的差距已接近顯著），數字見 progress.md。
- 尾字延遲受句尾靜音長度限制：AISHELL 句尾靜音中位數約 410 ms、全部短於 800 ms，`min_silence_ms` 大於句尾靜音時句子由音訊結束（flush）收尾，量到的尾字延遲會低估即時輸入的實際值。
- 同一天、同一設定名的結果會覆寫。
