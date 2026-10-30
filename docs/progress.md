# 開發紀錄

按時間順序追加，新的紀錄放在最下面。

---

## 2026-10-30　VAD 模組（第一階段）

環境：Apple M5 / macOS 27.0.1 / Python 3.11 / onnxruntime 1.31.1，單執行緒，CPU。

### 做了什麼

- `scripts/download_models.sh`：下載模型到 `models/`，固定版本並校驗 SHA-256，可重複執行（校驗通過即跳過）。
  - Silero VAD **v6.2.3**（tag commit `5cd7945676`），MIT 授權
  - 來源：`https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx`
  - 路徑：`models/silero-vad-v6.2.3/silero_vad.onnx`，2.3 MB，FP32
  - SHA-256：`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- `src/vad/`
  - `silero.py`：`SileroVadModel`，以 ONNX Runtime 直接呼叫模型，逐 frame（512 樣本 / 32 ms）回傳語音機率。
  - `detector.py`：`VadConfig`（參數）、`StreamingVad`（串流：`accept_waveform(chunk)` / `flush()` → 開始 / 結束事件）、`detect_segments`（離線：整段音訊 → 語音段）。
  - `load_wav`：讀 16 kHz wav，多聲道取平均；取樣率不符直接報錯，不偷偷重取樣。
- `tests/test_vad.py`（19 個，連同既有冒煙測試共 23 個）：純靜音 / -60 dB 底噪無輸出；合成語音偵測到正確段數且起止誤差 < 100 ms；4 種 chunk 大小的串流結果與離線一致；用假分類器逐一驗證每個參數的行為。
- `tests/fixtures/vad_two_utterances.{wav,json}`：macOS `say`（Tingting）合成兩句中文，修掉頭尾靜音後放在已知位置，真實起止精確已知。產生腳本 `make_vad_fixture.py` 一併提交。
- `scripts/demo_vad.py`：輸入 wav 印出語音段；`--stream` 以 10 ms chunk 模擬即時送入，印出事件與延遲。

### 關鍵設計決策

1. **直接用 onnxruntime，不用 sherpa-onnx 的 VAD 介面。**
   - 評測需要中間量：首字延遲要用「VAD 觸發點」當語音起點，也要量 VAD 本身的演算法延遲。直接呼叫模型能拿到每 frame 機率與事件發出時刻；sherpa-onnx 只回傳切好的段落。
   - 第二、三階段要轉 Core ML、寫 Swift，模型 I/O（state、64 樣本 context、sr）與判斷規則遲早要自己掌握，現在用 Python 寫清楚，之後照搬。
   - 成本低：模型只有三個輸入，包裝約 40 行，context 處理已對照上游 `utils_vad.py`。sherpa-onnx 留給 ASR。
2. **離線與串流共用同一個狀態機。** 離線 = 整段送入串流再 `flush()`，兩者結果必然一致（有測試保證），避免兩套邏輯各自漂移。
3. **後端與判斷邏輯分離。** 狀態機只依賴 `FrameClassifier` 協定（`frame_size` / `__call__` / `reset`），換 Core ML 時只換模型包裝；測試也因此能用假分類器精確驗證規則。
4. **事件同時記錄位置與發出時刻**（`sample` / `detected_at`），兩者之差就是 VAD 的演算法延遲，供之後首字延遲的拆解使用。
5. **判斷規則與上游一致**：遲滯雙閾值；語音段內短於 `min_silence_ms` 的靜音不切段；含內部短靜音的總長不足 `min_speech_ms` 則丟棄；前後 padding 不跨越前一段終點、不超出音訊範圍。

### 目前的參數設定

與上游 Silero `get_speech_timestamps` 預設值相同，**尚未依評測調整**：

| 參數 | 值 | 說明 |
|---|---|---|
| `threshold` | 0.5 | 機率 ≥ 此值為語音 frame |
| `neg_threshold` | 0.35（= threshold − 0.15） | 機率 < 此值為靜音 frame，中間維持原狀態 |
| `min_speech_ms` | 250 | 語音段最短長度，也是串流「開始」事件的最小延遲 |
| `min_silence_ms` | 100 | 切段所需最短靜音，也是「結束」事件的最小延遲 |
| `speech_pad_ms` | 30 | 語音段前後延伸 |
| 執行緒數 | 1 | ONNX Runtime intra-op |

實測（合成測試音訊 6.2 s）：串流「開始」事件延遲 286 ms、「結束」98 ms；RTF 約 0.002–0.004（單次量測，未排除暖機，僅供量級參考）。

### 遇到的問題

- **自然口語下，逗號停頓與斷句停頓的長度重疊。** 用兩段自錄音訊驗收：
  - 錄音 1（自由說 3 句）：其中兩句之間的實際靜音只有約 290 ms，預設參數正確切成 3 段。
  - 錄音 2（照稿念 4 句，句間停約 1 秒）：偵測到 5 段。第 1 句逗號處停了 320 ms 被切開；其餘逗號停頓 64–96 ms 未被切開；句間停頓 1.5–3 s 皆正確切開；短句「對。」語音 288 ms，僅略高於 `min_speech_ms`，勉強保留。
  - 結論：句間停頓（290 ms）可能比逗號停頓（320 ms）還短，**單靠靜音長度不存在能同時切對兩段錄音的 `min_silence_ms`**，這是純聲學 VAD 的固有限制。
  - 對 ASR 而言，在句中誤切（丟失上下文、切點附近易錯）比誤合併（串流 ASR 可處理長段）代價高，因此 pipeline 傾向用較大的 `min_silence_ms`（候選 400–500 ms）與較小的 `min_speech_ms`（候選 150–200 ms）。**依驗證紀律，預設值暫不更動，待 pipeline 階段以 CER 對比後再定。**
- **在專案外層目錄執行 `uv run` 會用到錯的 Python。** 外層沒有 `pyproject.toml`，uv 退回 anaconda 的直譯器，出現 `ModuleNotFoundError: sounddevice`。所有 `uv run` 指令都要在 `ondevice-zh-asr/` 內執行。
- **測試本身要能區分對錯。** 遲滯測試的第一版，有無遲滯結果相同、等於沒測；改成機率落在雙閾值之間、會影響切段的序列，並加上 `neg_threshold = threshold` 的對照組。
- `scripts/` 下的腳本無法直接 `import src`（專案未安裝成套件），`demo_vad.py` 以插入 repo 根目錄到 `sys.path` 處理。

### 待辦

- pipeline 階段：以 CER 比較數組 `min_silence_ms` / `min_speech_ms`，決定 ASR 前端用的參數。
- 補一組有背景噪音（電視、風扇）的自錄樣本，檢查誤觸發。
- （可選）`demo_vad.py` 加 `--show-probs` 印出逐 frame 機率時間軸，方便調參。

---

## 2026-10-30　串流 ASR 模組（第一階段）

環境：Apple M5 / macOS 27.0.1 / Python 3.11 / sherpa-onnx 2.0.1，單執行緒，CPU。

### 選型

在 sherpa-onnx release `asr-models` 中比較串流中文模型（大小為推論所需檔案，非壓縮檔）：

| 模型 | 架構 | FP32 / INT8 (MB) | 訓練資料 | 備註 |
|---|---|---|---|---|
| **zipformer-zh-14M-2023-02-23**（選用） | transducer | 53 / 24 | WenetSpeech | 最小；同一壓縮檔含 FP32 與 INT8 |
| zipformer-small-ctc-zh-2025-04-01 | CR-CTC | 87 / 25 | 未註明 | 單一 onnx，轉 Core ML 較單純 |
| zipformer-multi-zh-hans-2023-12-13 | transducer | 約 2 倍 / 73 | 14k 小時（含 AISHELL-1 train） | AISHELL test 屬同領域 |
| zipformer-small-bilingual-zh-en-2023-02-16 | transducer | 110 / 47 | 中英 | 只在需要中英夾雜時考慮 |

更大的（zh-2025-06-30、x-asr 2026 系列，INT8 皆 120 MB 以上）先不考慮。選 14M 當第一個基準，換模型要等評測框架完成後以 CER 等指標對比。
注意：14M 只用 WenetSpeech 訓練，在 AISHELL-1 上是跨領域，CER 預期偏高。

### 做了什麼

- `scripts/download_models.sh`：新增 `fetch_archive`，下載 tar.bz2 → 校驗 SHA-256 → 解壓 → 寫入 `.sha256` 標記 → 刪除壓縮檔；標記吻合即跳過。
- `src/asr/base.py`：抽象介面 `StreamingAsr.create_stream()` → `AsrStream`，`accept_waveform(chunk)` / `get_partial_result()` / `finalize()`。基底類別統一處理 finalize 之後的狀態（再送入或重複 finalize 會報錯）。
- `src/asr/sherpa.py`：`SherpaTransducerAsr`。`accept_waveform` 內就把能解碼的音訊解碼完，讓計時包含計算；`finalize` 補 0.66 s 靜音（與官方範例一致）再 `input_finished()`。關閉 sherpa 的 endpoint detection，斷句交給 VAD。
- `src/asr/__init__.py`：`load_asr(name, num_threads)` 依名稱載入；新增後端只要實作兩個類別並登記名稱。
- `tests/test_asr.py`（10 個）：假後端驗證介面約定；用模型自帶 test_wavs 驗證 0.wav 與官方文件輸出完全一致、兩句對人工標註 CER ≤ 0.15、4 種 chunk 大小結果一致、partial 確實是串流（送一半就有、是最終結果的前綴、長度不倒退）、多個 stream 互不影響。
- `scripts/demo_asr.py`：依實際時長節奏逐 chunk 送入（`--fast` 不等待），印出 partial 變化與最終結果。

### 實測（單次量測，僅供量級參考）

| wav | 時長 | 結果 | CER（對人工標註） |
|---|---|---|---|
| 0.wav | 5.61 s | 对我做了介绍那么我想说的是大家如果对我的研究感兴趣 | 3/28（省略「啊、呢、呢」） |
| 1.wav | 5.15 s | 重点呢想谈三个问题首先就是这一轮全球金融动能的表现 | 2/26（省略「呢」，「动荡」→「动能」） |

10 ms chunk 時 RTF 約 0.06，100 ms chunk 約 0.014（chunk 越小呼叫開銷越大），finalize 8–27 ms。

### 待辦

- 評測框架完成後，在 AISHELL-1 test 上量 14M 基準，再決定是否換 small-ctc 或 multi-zh-hans。

---

## 2026-10-30　接入 X-ASR 480 ms 串流模型（INT8）

zh-14M 在自錄音訊上錯誤明顯（「下午」→「笑我」、「笔电」→「鼻垫」），改試 X-ASR。

### 選型

- X-ASR 在 sherpa-onnx release 有離線版與 160 / 480 / 960 / 1920 ms 四種 chunk 的串流版，各有 FP32、INT8、帶標點版本。
- 選 **480 ms 串流、不帶標點、官方 INT8 版**：
  - 串流：首字延遲是核心指標，離線版沒有 partial。
  - 480 ms：160 ms 準確度通常最差，1920 ms 已不算即時。其餘 chunk 與離線版等評測框架完成後再對比。
  - 不帶標點：第一階段不做標點，CER 計算前也會去標點。
  - INT8：FP32 版 encoder 單檔 566 MB，不適合端側。**偏離 CLAUDE.md「第一階段以 FP32 為基準」**，這是官方提供的量化版、非自行量化，報告時要註明精度。
- 檔案：encoder.int8 148 MB、joiner.int8 2.5 MB、decoder 10.8 MB（官方 INT8 版的 decoder 仍為 FP32），含 tokens 合計約 161 MB。原始模型 https://github.com/Gilgamesh-J/X-ASR，訓練資料與授權尚未確認。
- 註冊名稱 `x-asr-480ms-int8`；`load_asr` 預設仍是 `zipformer-zh-14m`，待評測後再決定預設。

### 句尾補零（tail padding）要依模型設定

encoder metadata：X-ASR 每個 chunk 48 frame（480 ms）、輸入 T = 61 frame（右側 context 13 frame）；zh-14M 為 32 / 39 frame。
X-ASR 用原本的 0.66 s 補零會漏掉句尾的字（例如 1.wav 少了「表现」）。在 6 個測試音訊上逐一嘗試：

| 補零 (s) | 結果 | finalize 耗時 (ms) |
|---|---|---|
| 0.66 | 4/6 句漏字 | – |
| 1.2 | 2/6 句漏字 | 27 |
| 1.4 / 1.6 | 與 3.0 s 完全相同 | 41 |
| 2.0 | 與 3.0 s 完全相同 | 55 |
| 3.0 | （基準） | 81 |

取 **1.6 s**（與 1.4 s 計算量相同、留餘量），在 registry 設定。zh-14M 補 0 / 0.66 / 3.0 s 結果都相同，維持 0.66 s。
補零不需等待真實時間，代價只有 finalize 的計算量，會計入尾字延遲。

### 對比（單次量測，僅供量級參考，正式數字待評測框架）

| | zh-14M（FP32） | X-ASR 480 ms（INT8） |
|---|---|---|
| 模型大小 | 53 MB | 161 MB |
| 載入時間 | 0.25 s | 1.0 s |
| 0.wav（CER） | 3/28，省略語氣詞 | 0/28 |
| 1.wav（CER） | 2/26，「动荡」→「动能」 | 0/26 |
| my_voice.wav | 你好克劳的你是谁 | 你好 CLOUD 你是谁 |
| my_voice_script.wav | …对**笑我**的会议…带**鼻垫**… | …对**下午**的会议…带**笔电**…（全對） |
| RTF（`--fast`，100 ms chunk） | 0.013–0.014 | 0.030–0.035 |
| 第一個 partial（0.wav，即時節奏） | 送入 0.73 s 時 | 送入 1.11 s 時 |
| finalize | 8 ms | 40 ms |

- X-ASR 準確度明顯較好，代價是體積約 3 倍、RTF 約 2.5 倍、第一個 partial 晚約 0.4–0.7 s（與 chunk 480 ms vs 320 ms 一致）。
- X-ASR 英文輸出為大寫並以空格分隔（如 `CLOUD`），CER 正規化需處理（轉小寫、去空白已在規則內）。

### 測試

`tests/test_asr.py` 的真實模型測試改為對每個已登記模型各跑一遍（共 39 個測試）。X-ASR 的 CER 上限設 0.05，同時用來抓補零不足（漏「表现」時 CER ≈ 0.077）。X-ASR 自帶的 test_wavs 沒有標註，未納入測試。

預設模型改為 `x-asr-480ms-int8`（`src.asr.DEFAULT_MODEL`，`load_asr()` 與 `demo_asr.py` 皆使用）。依據是上表的單次對比，AISHELL-1 上的正式數字待評測框架完成後補上；zh-14M 保留登記，供對比。

### 首字延遲粗量（單次，非正式評測）

8 句（兩模型自帶 test_wavs + 2 段自錄），10 ms chunk、模擬即時節奏；語音起點用 Silero VAD 第一段起點（含 30 ms padding）。
RTF << 1 不會積壓，因此 partial 出現時刻以「該 chunk 收齊的音訊時刻 + 該次計算時間」換算，未實際 sleep。

| 模型 | P50 | P90 | 最小 / 最大 |
|---|---|---|---|
| zh-14M | 572 ms | 687 ms | 188 / 732 ms |
| X-ASR 480 ms | 913 ms | 1035 ms | 513 / 1057 ms |

- X-ASR 的 partial 只在送入 0.63、1.11、1.59、2.07 s…時更新（首個 chunk 需 61 frame ≈ 0.61 s，之後每 48 frame = 0.48 s）；zh-14M 每 0.32 s 一次。
- 每次計算約 13 ms，延遲幾乎全是演算法延遲（等 chunk 收滿），換更快的後端（Core ML / ANE）幫助不大，要降延遲只能換更小的 chunk（如 X-ASR 160 ms 版）。
- **決定：暫時維持 480 ms 版**，160 ms 版等評測框架完成後再以 CER / 首字延遲對比。

---

## 2026-10-30　pipeline：VAD + 串流 ASR（第一階段）

### 做了什麼

- `src/pipeline/`：`StreamingPipeline(vad_model, asr, config)`，`accept_waveform(chunk)` / `flush()` → 事件（`start` / `partial` / `final`）。VAD「開始」→ 建立 ASR stream；語音中持續送入；VAD「結束」→ `finalize()`，整句結果存進 `pipeline.utterances`。離線便利函式 `transcribe(audio, ...)`。
- 每句（`Utterance`）記錄：語音起點 / 終點（VAD，含 padding）、實際送入 ASR 的起點（含 pre-roll）、VAD 開始 / 結束事件發出時刻、首字時刻、最終結果時刻、最終文字、partial 歷程；衍生首字延遲與尾字延遲。
- `scripts/run_pipeline.py`：輸入 wav，輸出 JSON（設定、機器、RTF、每句結果與時間戳；`--partials` 附 partial 歷程）。
- `tests/test_pipeline.py`（9 個）：假 VAD + 記錄輸入的假 ASR 精確驗證 pre-roll 起點、句間不重疊、時間順序、計算時間計入、與 chunk 大小無關；真實模型跑 fixture 兩句，檢查 CER、起止誤差、首字延遲範圍、chunk 大小無關。

### 關鍵設計決策

1. **pre-roll 用環形緩衝實作。** VAD「開始」事件比事件位置晚 min_speech_ms + padding（約 280 ms），所以 pipeline 保留最近約 min_speech + pad + pre_roll + 100 ms 的音訊，建立 stream 時從 `VAD 起點 − pre_roll` 補送。起點不早於上一句已送入 ASR 的終點，同一段音訊不會被辨識兩次。
2. **依 VAD 事件的樣本位置切分送入 ASR 的音訊，不依 chunk 邊界。** 一個大 chunk 內同時有結束與下一句開始時也能正確分配；送入 ASR 的音訊與 chunk 大小無關（有測試保證）。句子結束時送到 `detected_at`（含 min_silence_ms 的靜音尾巴）。
3. **時間戳用「模擬即時」虛擬時鐘，不真的 sleep。** chunk 最後一個樣本到達的音訊時刻 + 實際量測的計算時間；計算比音訊慢時自然積壓。與先前首字延遲粗量的方法一致，跑長音訊也不用等。時鐘可注入，測試因此可決定。

### pre-roll 小規模對比（單次，非正式評測）

7 句（zh-14M 自帶 0.wav / 1.wav 經 VAD 切成 5 句 + fixture 2 句），10 ms chunk，VAD 預設參數：

| pre-roll | X-ASR 480 ms | zh-14M |
|---|---|---|
| 0 ms | 文字與其他設定完全相同 | 「那么」→「什么」、「重点呢」漏「呢」 |
| 100 ms | 同上；首字延遲多數句比 0 ms 少 100 ms | 「感兴趣」→「感谢兴趣」，其餘最好 |
| 300 / 500 ms | 同上 | 「金融动荡」→「金融进入共党」等，明顯變差 |

- pre-roll 會改變 ASR chunk 與語音的對齊，首字延遲隨之非單調變化（同一句可差 ±480 ms，即一個 X-ASR chunk）。
- 預設取 **100 ms**（加上 VAD 30 ms padding，共比 VAD 起點早 130 ms 送入）。樣本太少，評測集完成後以 CER 重新決定。
- 注意：VAD 預設 `min_silence_ms = 100` 在這組音訊上把 0.wav、1.wav 各切成多句；pipeline 參數調整仍待評測（見 VAD 待辦）。

### 待辦

- 評測框架用 `transcribe` 產生首字 / 尾字延遲；在 AISHELL-1 上比較 pre-roll 與 `min_silence_ms`。
- 麥克風輸入（`scripts/mic_demo.py`）直接接 `StreamingPipeline`。
