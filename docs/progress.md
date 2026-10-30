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

---

## 2026-10-30　評測框架與 VAD / pre-roll 參數掃描（第一階段）

環境：Apple M5（Mac17,3）/ macOS 27.0.1 / onnxruntime 1.31.1 / sherpa-onnx 2.0.1，單執行緒，CPU。
評測集：AISHELL-1 test 隨機 200 句（seed 20261030，17.2 分鐘，20 位說話人），10 ms chunk 模擬即時。
掃描在 commit `e9482b8` + 未提交的評測程式上執行（評測程式即 `f576d57` 提交的內容）；預設設定在 `f576d57` 上重跑，CER 與延遲逐位相同。
方法與指標定義見 [eval.md](eval.md)。

### 做了什麼

- `scripts/prepare_eval_data.py`：固定 seed 從 AISHELL-1 test（7176 句）抽 200 句，逐檔下載並以 SHA-256 校驗，寫出 `manifest.jsonl` / `meta.json`。
  - 官方 `AISHELL/AISHELL-1`（Hugging Face）只上傳了 100 位訓練集說話人；OpenSLR 只有 15 GB 完整包。改用保留原始目錄結構的鏡像 `shenyunhang/AISHELL-1`（固定 revision），只需下載約 33 MB。這是第三方鏡像，句數與標註檔大小與官方一致，音訊無法對官方 checksum 驗證。
- `eval/text_norm.py`：NFKC（全形轉半形）→ 小寫 → OpenCC 繁轉簡 → 去標點與符號 → 中文按字、英文按詞切 token。
- `eval/metrics.py`：jiwer 語料級 CER、延遲分位數。
- `eval/run_eval.py`：CER、RTF、首字 / 尾字延遲、峰值 RSS（獨立子程序）、模型大小、每條切段數；多值參數取笛卡兒積做掃描。每次結果記錄完整設定與環境，存成 `eval/results/<日期>_<設定名>.json`。
- `tests/test_eval.py`（16 個）：正規化規則、語料級 CER 不等於逐句平均、彙整邏輯、fixture 端到端。

### 掃描結果

`min_silence_ms` ∈ {100, 300, 500, 800} × `pre_roll_ms` ∈ {0, 100, 200, 300}，兩個模型共 32 組。其他 VAD 參數：threshold 0.5、min_speech 250 ms、pad 30 ms。

**CER (%)**（列：min_silence_ms，欄：pre_roll_ms）

| | X-ASR 0 | 100 | 200 | 300 | zh-14M 0 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| 100 | 7.67 | 6.90 | 6.67 | 6.40 | 22.71 | 17.91 | 17.11 | 15.84 |
| 300 | 6.17 | 5.47 | 5.40 | 5.30 | 21.01 | 17.01 | 16.37 | 15.01 |
| **500** | 5.87 | **5.07** | 5.04 | 5.00 | 19.14 | 15.27 | 14.80 | 13.47 |
| 800 | 5.64 | 4.83 | 4.83 | 4.90 | 18.74 | 15.31 | 14.87 | 13.47 |

**首字延遲 mean / P90 (ms)**

| | X-ASR 0 | 100 | 200 | 300 | zh-14M 0 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|---|---|
| 100 | 1028 / 1130 | 981 / 1029 | 943 / 943 | 901 / 1305 | 809 / 1060 | 674 / 956 | 652 / 860 | 658 / 760 |
| 300 | 1043 / 1129 | 992 / 1029 | 954 / 1054 | 908 / 1305 | 822 / 1372 | 678 / 956 | 655 / 860 | 664 / 760 |
| **500** | 1046 / 1129 | **990 / 1029** | 953 / 976 | 907 / 1305 | 830 / 1372 | 679 / 956 | 656 / 860 | 664 / 760 |
| 800 | 1051 / 1129 | 994 / 1029 | 959 / 1400 | 909 / 1305 | 831 / 1372 | 679 / 958 | 658 / 860 | 664 / 760 |

**VAD 切段與尾字延遲**（只與 min_silence_ms 有關；pre-roll 不影響切段，尾字延遲各 pre-roll 相差 < 2 ms）

| min_silence_ms | 每條平均段數（max） | 1 / 2 / 3+ 段的條數 | 尾字延遲 mean，X-ASR / zh-14M (ms) |
|---|---|---|---|
| 100 | 1.74 (6) | 101 / 67 / 32 | 147 / 110 |
| 300 | 1.42 (5) | 136 / 51 / 13 | 335 / 300 |
| **500** | 1.19 (3) | 168 / 27 / 5 | 447 / 410 |
| 800 | 1.07 (3) | 186 / 13 / 1 | 452 / 416 |

RTF：X-ASR 0.034–0.039、zh-14M 0.014–0.015，各設定之間差距在量測雜訊內（同設定重跑差約 5%）。峰值 RSS：X-ASR 531–538 MB、zh-14M 245–248 MB，與參數無關。沒有任何一句 VAD 整句漏掉。

**成對 bootstrap**（同樣 200 句重抽 5000 次，CER 差的 95% 信賴區間，單位：百分點）

| 比較（另一參數固定） | X-ASR | zh-14M |
|---|---|---|
| min_silence 100 → 500（pre 100） | −1.83 [−2.65, −1.09] | −2.63 [−3.58, −1.75] |
| min_silence 300 → 500 | −0.40 [−0.81, 0.00] | −1.73 [−2.54, −0.98] |
| min_silence 500 → 800 | −0.23 [−0.55, +0.07] | +0.03 [−0.27, +0.41] |
| pre-roll 0 → 100（sil 500） | −0.80 [−1.41, −0.20] | −3.87 [−5.22, −2.54] |
| pre-roll 100 → 200 | −0.03 [−0.66, +0.68] | −0.47 [−1.61, +0.62] |
| pre-roll 200 → 300 | −0.03 [−0.66, +0.54] | −1.33 [−2.40, −0.26] |

單一設定的 CER 本身 95% 信賴區間約 ±1.1 個百分點（X-ASR）/ ±2 個百分點（zh-14M），但同一批句子上的成對比較靈敏得多。

### 分析

1. **min_silence_ms 是 CER 的主要因素，問題正是切碎。** AISHELL 每條都是一句話，但 100 ms 時一半的音訊被切成 2 段以上（最多 6 段），句中停頓處被切開後 ASR 失去上下文。500 ms 時 84% 的音訊是 1 段。
2. **500 → 800 的收益不顯著，代價是尾字延遲。** 表中 500 / 800 的尾字延遲幾乎相同，這是**評測集造成的假象**：200 條音訊句尾靜音全部短於 800 ms（中位數 412 ms、89% 短於 500 ms），句子多半由音訊結束（flush）而非靜音計時結束，量到的尾字延遲被截斷。100 / 300 ms 時尾字延遲 ≈ min_silence + finalize（X-ASR 約 40 ms），即時輸入時 800 ms 應約為 840 ms，比 500 ms 多 300 ms 才出最終結果。
3. **pre-roll 0 → 100 ms 顯著改善 CER**（首字起音被 VAD 切掉）；X-ASR 在 100 ms 以上已無差異，zh-14M 則一路改善到 300 ms。
4. **首字延遲被 ASR chunk 量化。** X-ASR 第一個 chunk 需 61 frame（約 0.61 s）、之後每 0.48 s 更新一次。pre-roll 越長，第一個 chunk 越早收滿，所以多數句子的首字延遲剛好少 100 ms（P50：0 → 300 ms 為 1125 / 1025 / 926 / 827 ms）；但 chunk 中真正的語音也越少，比多數句子多等一個 chunk（+480 ms）的句子變多：pre-roll 0 / 100 / 200 / 300 時分別有 2 / 7 / 20 / 36 條（sil 500）。P90 因此在 1029 和 1305 之間跳動，pre-roll 200 的 P90 在 min_silence 500 時是 976、800 時是 1400，是因為那 20 條（10%）剛好落在 P90 邊界上。zh-14M 的 chunk 是 320 ms，量化較細，pre-roll 越長平均與 P90 都越好。
5. **VAD 確認語音的 291 ms（≈ min_speech 250 ms + frame 對齊）只要小於第一個 chunk 的等待時間就不會出現在首字延遲裡**，因為 pre-roll 的音訊在建立 stream 時一次補送。

### 決定

- **pipeline 預設 `min_silence_ms` 100 → 500**（`PipelineConfig` 的預設 VAD 設定；VAD 模組本身仍是上游預設值 100，`demo_vad.py` 不變）。100 → 500 兩個模型都顯著改善；500 → 800 不顯著，但每句最終結果晚 300 ms。代價：自然口語中句間停頓短於 500 ms 時兩句會合併（見 VAD 模組紀錄：實測句間停頓可短至 290 ms），但串流 ASR 能處理長段，partial 仍持續輸出，影響的只是「最終結果」的時機。
- **`pre_roll_ms` 維持 100**。X-ASR 上 100 / 200 / 300 CER 無顯著差異；200 的平均首字延遲少 37 ms，但需多等一個 chunk 的句子從 3.5% 增為 10%，P90 不穩定（976–1400 ms）；100 的 P90 在四種 min_silence 下都是 1029 ms。zh-14M 偏好 300 ms（CER 比 100 ms 低 1.8 個百分點、P90 低 196 ms），若之後改用 zh-14M 或其他 chunk 大小的模型，需要重新掃描。
- 預設設定（X-ASR 480 ms、sil 500、pre 100）：**CER 5.07%、RTF 0.036、首字延遲 990 / 1029 ms（mean / P90）、尾字延遲 449 ms（受上述截斷影響，即時約 540 ms）、峰值 RSS 533 MB、模型 163.6 MB**。

### 待辦

以下與先前各段的待辦已統一整理到 [backlog.md](backlog.md)（含優先級），之後以該檔為準。

- 首字延遲約 1 s，主要是 X-ASR 480 ms chunk 的演算法延遲（計算只佔約 13 ms / chunk）。下一步在同一評測集上比較 X-ASR 160 ms 版。
- 尾字延遲要在句尾留足靜音的音訊上量（例如在每條 wav 後補 1 s 靜音），才能反映 min_silence 的真實代價。
- 剩下 16% 的音訊仍被切成多段：可試 `threshold` / `neg_threshold` 或 ASR 端的 endpoint 規則。
- 多執行緒 RTF、X-ASR 參數量尚未量測。

---

## 2026-10-30　第一階段整理（不改演算法）

### 做了什麼

- 從乾淨 clone 照 README 快速開始走一遍，補上缺漏：要在 `ondevice-zh-asr/` 內執行、各步驟下載量與時間、pytest 預期結果（沒有模型時 24 個測試會被跳過而不是失敗）、麥克風 demo 的命令。
- 錯誤處理：`load_wav` 統一丟 `FileNotFoundError`（不存在）/ `ValueError`（無法解碼、取樣率不符），先前不存在或非音訊的檔案會讓 `demo_vad` / `demo_asr` / `run_pipeline` 印出 `LibsndfileError` traceback。`demo_asr` 改成先載入模型再讀 wav（預設 wav 在模型目錄內）。`demo_mic` 開啟裝置失敗、`prepare_eval_data` 網路錯誤、`run_eval` 空 manifest 都改成錯誤訊息 + 非零狀態碼。`--threads` / `--chunk-ms` 等參數在 argparse 階段檢查正負。
- 去重複：`src/audio.py`（`SAMPLE_RATE`、`load_wav`）、`src/paths.py`（`MODELS_DIR`）、`VAD_MODEL_NAME` 常數、`src/cli.py`（pipeline 共用參數與 `PipelineConfig` 建立，`run_pipeline` / `demo_mic` / `run_eval` 共用；顯示寬度函式）。原有的匯入路徑（`src.vad.load_wav`、`src.asr.SAMPLE_RATE`）保留。
- `ruff format` 全部通過；新增 `tests/test_audio.py`、`tests/test_cli.py`（共 84 個測試）。

### 驗證

重構後在同一評測集重跑預設設定（兩個模型）：200 句辨識結果逐句相同，CER、首字 / 尾字延遲、切段數、模型大小逐位相同；RTF 0.0364 → 0.0370（X-ASR）、0.0151 → 0.0157（zh-14M），在同設定重跑的雜訊範圍（約 5%）內。


---

## 2026-10-30　模型來源查證、3000 句評測集、記憶體探查（B6、B8、B11）

環境：Apple M5（Mac17,3）/ macOS 27.0.1 / onnxruntime 1.31.1 / sherpa-onnx 2.0.1，單執行緒，CPU。
評測在 commit `f6b11a8` + 未提交的程式上執行（結果 JSON 記錄為 `f6b11a8 (dirty)`），程式即 `982ca41` 提交的內容（之後只改了文件）。

### 模型來源與授權（B6）

完整查證與來源連結見 [models.md](models.md)，摘要：

- **X-ASR**：Apache-2.0（repo LICENSE、Hugging Face 模型卡）。訓練資料只有「約 100 萬小時開源及收集語音」一句，**沒有資料集清單；是否含 AISHELL-1 train / test 未知**。技術報告標示 Coming Soon。官方評測沒有 AISHELL-1。
- **zh-14M**：Apache-2.0。sherpa-onnx / sherpa-ncnn 文件寫明只用 WenetSpeech 訓練（子集未寫），依文件不含 AISHELL-1。
- **結論**：X-ASR 在 AISHELL-1 上的 CER 不能當作跨領域泛化能力，README 已註明；要有保證跨領域的數字，需要自錄的評測集（backlog B6a）。
- **公開 GitHub 的授權評估**：可公開（模型與資料都是寬鬆授權，repo 不散布權重與音訊）。唯一阻礙是 `tests/fixtures/vad_two_utterances.wav` 由 macOS 系統語音合成，macOS 授權排除公開散布系統語音（backlog B13）。

### 評測集改為 3000 句，CER 報告信賴區間（B8）

- `prepare_eval_data.py` 預設改為從 AISHELL-1 test 以同一 seed（20261030）隨機抽 3000 句（251.2 分鐘、20 位說話人、約 470 MB）；`-n 200` 仍產生原本的冒煙子集，`-n 7176` 為完整集。
  - 原本打算跑完整 7176 句（約 1.1 GB、約 10 小時音訊），下載速度約每分鐘 100 句；改用 3000 句以縮短下載與評測時間。代價是不能與論文的完整 test set 數字逐位對照。
  - 3000 句不是 200 句的超集（各自獨立抽樣）。中斷的完整集下載不能拿前段直接用：下載順序依句子 ID 排序，前 2700 句只涵蓋部分說話人。
- `run_eval.py` 預設改用 3000 句，`--smoke` 用 200 句。結果檔名加上評測集名稱（`<日期>_aishell1_test_3000_<設定>.json`），不同集的結果不再互相覆寫。
- CER 新增 95% 百分位 bootstrap 信賴區間（10000 次、seed 0），兩種重抽單位：句子、說話人（cluster bootstrap）。定義見 [eval.md](eval.md)。
- `prepare_eval_data.py` 修正：連線中斷造成的截斷回應（`http.client.IncompleteRead`）不是 `OSError`，原本不會重試，下載 7176 句時在第 2286 句中斷。現在一併重試（最多 5 次）。

**3000 句基準**（預設設定：sil 500、pre-roll 100、10 ms chunk）

| 模型 | CER (%) | 95% CI 以句 | 95% CI 以說話人 | S / D / I / N | RTF | 首字 mean / P90 (ms) | 尾字 mean (ms) | 每條切段數 | 峰值 RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| X-ASR 480 ms | **4.15** | [3.89, 4.43] | [3.59, 4.75] | 1614 / 179 / 27 / 43813 | 0.0358 | 961 / 1029 | 453 | 1.18 | 618 |
| zh-14M | 13.49 | [13.00, 13.99] | [11.68, 15.31] | 4524 / 1131 / 255 / 43813 | 0.0155 | 665 / 958 | 417 | 1.18 | 333 |

**與 200 句子集對照**（同一份程式在 200 句上重跑：辨識結果、CER、首字延遲、切段數與 `f576d57` 相同；尾字延遲含實測計算時間，449 → 447 ms）

| 模型 | 200 句 CER [95% CI 以句] | 3000 句 CER [95% CI 以句] |
|---|---|---|
| X-ASR | 5.07 [4.04, 6.18] | 4.15 [3.89, 4.43] |
| zh-14M | 15.27 [13.34, 17.35] | 13.49 [13.00, 13.99] |

- 兩個模型在 3000 句上都比 200 句低約 1–2 個百分點，都在 200 句的信賴區間內：200 句剛好抽到比平均難的句子，不是程式改變（200 句重跑的辨識結果相同）。
- **以說話人重抽的區間寬 2.2 倍（X-ASR）/ 3.7 倍（zh-14M）**：錯誤率在說話人之間差異大，20 位說話人是不確定性的主要來源，多抽句子無法再縮小這部分。zh-14M 的說話人差異比 X-ASR 更明顯。
- 每條切段數 1.18（2548 / 365 / 87 條為 1 / 2 / 3+ 段），與 200 句的 1.19 一致；首字延遲 P90 都是 1029 ms（X-ASR）。
- RTF 與 200 句時相同（X-ASR 0.0354–0.0358）。

### 記憶體探查（B11）

新增 `scripts/probe_memory.py`：每個變體在獨立子程序中載入模型、整句送入解碼 N 句，記錄各階段 RSS。ORT 的 session 設定透過 sherpa-onnx provider 設定檔傳入（`provider="cpu:<設定檔>"`，sherpa-onnx 2.0.1 `csrc/session.cc` 支援 `EnableCpuMemArena`、`GraphOptimizationLevel`、`SessionConfig.*`；Python API 本身沒有這些參數）。

**RSS 基線與 session 設定**（X-ASR，解碼 30 句，MB）

| 變體 | 直譯器啟動 | import 後 | 載入後 | 解碼後 | peak | 載入增加 |
|---|---|---|---|---|---|---|
| default | 21 | 38 | 473 | 483 | 483 | 435 |
| `EnableCpuMemArena=0` | 21 | 38 | 472 | 483 | 483 | 434 |
| `session.disable_prepacking=1` | 20 | 38 | 466 | 473 | 473 | 428 |
| `GraphOptimizationLevel=0` | 21 | 38 | 616 | 622 | 622 | 577 |

zh-14M（30 句）：default 與 no-arena 都是載入後 190、peak 193 MB（載入增加 152 MB）。

**解碼句數的影響**（X-ASR，peak MB）：100 句 488 / 482（arena 開 / 關），1000 句 491 / 482。

**直接用 onnxruntime 載入同樣三個檔案**（每檔載入增加的 RSS，arena 開 / 關）：encoder.int8（148.1 MB）+217 / +222、decoder（10.8 MB）+22 / +22、joiner.int8（2.5 MB）+5 / +5，合計約 245 MB。

分析：

1. **arena 不是原因。** 關掉 CPU memory arena 後載入與峰值的變化都在 ±1 MB 內（1000 句時 peak 491 → 482，約 9 MB），因為 435 MB 的增加幾乎全部發生在**載入階段**，推論期間只再增加約 10–20 MB。arena 只影響推論時的暫存張量配置。
2. 關 prepacking 只少 7 MB；關圖最佳化反而多 144 MB（未融合的節點各自保留中間張量與權重格式）。
3. **經 sherpa-onnx 載入比直接用 onnxruntime 多約 190 MB**（435 vs 245）。兩邊的 ORT 函式庫不同（sherpa-onnx 自帶 `libonnxruntime.dylib`，Python 套件是 onnxruntime 1.31.1），差額來源未確認。下一步用 sherpa-onnx C API 或讀其載入程式碼確認（backlog B11）。
4. **評測的峰值 RSS 隨句數增加**：X-ASR 200 句 531 → 3000 句 618 MB、zh-14M 252 → 333 MB；但上面單獨解碼 100 → 1000 句只多 3 MB，所以增加的部分在評測 / pipeline 端（例如逐句結果累積或 partial 歷程），原因未查。在修正之前，不同評測集的峰值 RSS 不能直接比較（README 已註明）。
5. 決定：不改 `src/asr` 的 session 設定（沒有可量到的收益）。

### 待辦

見 [backlog.md](backlog.md)：B11（sherpa-onnx 載入多出的 190 MB、評測端 RSS 增長）、B13（替換 fixture 後才能公開 repo）、B6a（不與訓練資料重疊的評測集）。VAD / pre-roll 掃描之後要在 3000 句集上重做。
