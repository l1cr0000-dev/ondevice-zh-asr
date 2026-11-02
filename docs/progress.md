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
  - 同一 seed 下 `random.sample` 前段的抽選幾乎相同，200 句中有 199 句也在 3000 句集裡（事後檢查發現，原先以為是各自獨立的抽樣）。中斷的完整集下載不能拿前段直接用：下載順序依句子 ID 排序，前 2700 句只涵蓋部分說話人。
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
3. **經 sherpa-onnx 載入比直接用 onnxruntime 多約 190 MB**（435 vs 245）。兩邊的 ORT 函式庫不同（sherpa-onnx 自帶 `libonnxruntime.dylib`，Python 套件是 onnxruntime 1.31.1），差額來源見下一段的追查。
4. **評測的峰值 RSS 隨句數增加**：X-ASR 200 句 531 → 3000 句 618 MB、zh-14M 252 → 333 MB；但上面單獨解碼 100 → 1000 句只多 3 MB，所以增加的部分在評測 / pipeline 端（例如逐句結果累積或 partial 歷程），原因未查。在修正之前，不同評測集的峰值 RSS 不能直接比較（README 已註明）。
5. 決定：不改 `src/asr` 的 session 設定（沒有可量到的收益）。

### 待辦

見 [backlog.md](backlog.md)：B11（ORT 1.28 的載入碎片、評測端 RSS 增長）、B13（替換 fixture 後才能公開 repo）、B6a（不與訓練資料重疊的評測集）。VAD / pre-roll 掃描之後要在 3000 句集上重做。

### 追查：sherpa-onnx 載入多出的約 190 MB

問題：經 sherpa-onnx 載入 X-ASR 比直接用 onnxruntime 載入多約 190 MB RSS，是不是權重存了兩份？

方法：載入完成後在同一程序內執行 `vmmap --summary <pid>`，比較存活配置量（malloc zone 的 BYTES ALLOCATED）、已釋放未歸還（FRAG SIZE）與 physical footprint。直接載入用 `uv run --no-project --with onnxruntime==<版本>` 的臨時環境，不動專案依賴。

| 載入方式 | ORT 版本 | 存活配置 (MB) | 碎片 (MB) | 碎片比例 | physical footprint (MB) |
|---|---|---|---|---|---|
| sherpa-onnx 2.0.1 | 1.28.2（內附） | 219.0 | 182.5 | 46% | 259（載入時峰值 415） |
| onnxruntime | 1.28.0 | 222.1 | 191.7 | 47% | 443 |
| onnxruntime | 1.29.0 | 222.5 | 187.7 | 46% | 440 |
| onnxruntime | 1.30.0 | 222.2 | 33.7 | 14% | 286 |
| onnxruntime | 1.31.1（專案依賴） | 205.5 | 25.8–35.4 | 12–15% | 261–270 |

結論：

1. **沒有重複的權重。** 各版本的存活配置量都在 205–222 MB，比三個檔案合計（161 MB）多約 45–60 MB（這部分的組成未拆解）；sherpa-onnx 沒有比直接載入多存一份。
2. **多出來的是 ORT 1.28–1.29 載入時的暫存配置**：載入過程配置、用完釋放，但 malloc 沒有把這些頁面還給系統，仍計入 RSS。這和 sherpa-onnx 無關，同版本的 onnxruntime 直接載入也一樣；ORT 1.30.0 起降到約 34 MB。
3. 先前在 backlog 寫的推測「sherpa-onnx 先把檔案讀進記憶體再建 session」是錯的：讀 sherpa-onnx 2.0.1 原始碼（`csrc/online-zipformer2-transducer-model.cc`），桌面版是以檔案路徑建立 `Ort::Session`；先讀進緩衝的路徑只用於 Android / HarmonyOS 的 asset manager。
4. 這些碎片是同一程序之後可以重複使用的記憶體，所以推論期間 RSS 不會再大幅增加（解碼 1000 句只多 3 MB）。但評測報的 peak RSS（`ru_maxrss`）包含它。iOS 的 jetsam 依據的是 physical footprint；sherpa-onnx 這一行的 footprint（259 MB）比 RSS 低，但載入時的峰值仍到 415 MB。
5. sherpa-onnx 2.0.1 是 PyPI 最新版，目前無法只靠升版解決。可行做法是等上游升級 ORT，或自行以 ORT ≥ 1.30 編譯 sherpa-onnx（backlog B11）。第三階段改用 Core ML 時這個問題不存在，但要重新量。

磁碟上的評測資料（同一天檢查）：中斷的完整集下載 `data/aishell1_test_full/` 有 2936 個檔案，其中 1249 個以 hardlink 與 3000 句集共用（不佔額外空間），1687 個（約 260 MB）只在完整集裡、目前沒有用到。

---

## 2026-10-31　端到端首字延遲（B5 部分）、X-ASR 160 ms 對比（B1）

環境：Apple M5（Mac17,3）/ macOS 27.0.1 / onnxruntime 1.31.1 / sherpa-onnx 2.0.1，單執行緒，CPU。
評測在 commit `e72d97e` + 未提交的程式上執行（結果 JSON 記錄為 `e72d97e (dirty)`），程式即本次提交的內容。
評測期間同時有其他單執行緒工作在跑（模型下載、200 句掃描）；X-ASR 480 ms 的 RTF 換算成相同分母後與上次一致（見下），沒有看到干擾。

### 端到端首字延遲：不再用 VAD 本身當起點

- `run_eval.py` 新增 `--lead-silence-ms`（預設 1000）：每條音訊前補 1 秒數位靜音再送入 pipeline。
- 新增 `e2e_first_token_latency_ms` = 第一個非空 partial 的模擬即時時刻 − 補靜音結束點。與 VAD 無關，但含原始音訊開頭未知長度的靜音，**只適合設定間相對比較**。原本以 VAD 起點的首字延遲保留，兩者都報告。定義與限制見 [eval.md](eval.md)。
- RTF 的分母改為實際送入的音訊長度（含補靜音）。結果 JSON 的位置（`speech_start_s` 等）換回原始音訊時間軸。結果檔名加 `_lead1000`，不覆寫舊結果；`--lead-silence-ms 0` 重現舊設定。
- 開發時用 `--smoke --limit 20`（約 5 秒 / 模型）先確認能跑通，再跑 200 / 3000 句。

**補靜音前後**（3000 句，sil 500、pre-roll 100）

| 模型 | 補靜音 | CER (%) | RTF | 首字（VAD 起點）mean / P90 (ms) | 端到端首字 mean / P50 / P90 / P99 (ms) | 尾字 mean (ms) |
|---|---|---|---|---|---|---|
| X-ASR 480 ms | 0（上次，`f6b11a8`） | 4.15 | 0.0358 | 961 / 1029 | – | 453 |
| X-ASR 480 ms | 1000 ms | 4.11 | 0.0301 | 972 / 1029 | 1487 / 1533 / 1623 / 2073 | 451 |
| zh-14M | 0（上次） | 13.49 | 0.0155 | 665 / 958 | – | 417 |
| zh-14M | 1000 ms | 12.92 | 0.0134 | 666 / 958 | 1179 / 1144 / 1464 / 1844 | 416 |

- CER 小幅變化：1000 ms 不是 VAD frame（32 ms）的整數倍，補靜音讓 VAD frame 相對語音的切點位移。兩個模型都在上次的 95% CI 內。
- RTF 下降主要來自分母變大：0.0358 × 251.2 / 301.2 ≈ 0.0299，與 0.0301 一致。
- 端到端減 VAD 起點約 515 ms（X-ASR）/ 513 ms（zh-14M），兩個模型幾乎相同，符合「差值 = 原始音訊開頭靜音 + VAD 起點位置，與模型無關」的預期。

### X-ASR 160 ms INT8（B1）

- 查過的候選（sherpa-onnx release `asr-models` 中的中文串流模型）：X-ASR 160 / 960 ms INT8；`streaming-zipformer-zh-int8-2025-06-30`（multi-zh-hans large，約 160M）與 `multi-zh-hans-int8-2023-12-13`，兩者依 icefall `egs/multi_zh-hans/ASR/README.md` 訓練資料含 AISHELL-1（170 h），在 AISHELL-1 test 上是同領域、不能公平比較；`streaming-zipformer-small-ctc-zh-int8-2025-04-01`（需新增 CTC 後端，訓練資料未知）。**只下載 160 ms INT8。**
- 檔案：`sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05`（release `asr-models`，SHA-256 `bdf2c0f2…`，為 GitHub release 記錄的 digest；同一來源對 480 ms 版的 digest 與先前固定的值一致）。來源、授權、訓練資料與 480 ms 版相同（同一系列），見 [models.md](models.md)。
- encoder metadata：`decode_chunk_len` 16（160 ms）、`T` 29（右側 context 13 frame）；encoder.int8 148.1 MB，與 480 ms 版大小相同。
- 參數量：X-ASR README 的比較表列 0.16B（標為 offline 模式；README 說是 offline-streaming 統一模型）。
- 註冊名稱 `x-asr-160ms-int8`，預設模型不變。

**句尾補零**：整句送入（不經 VAD），與補 3.0 s 的結果逐句比對（200 句 AISHELL + 4 個官方 test_wavs）。

| 補零 (s) | 與 3.0 s 不同的句數 | finalize mean (ms) |
|---|---|---|
| 0.64 | 15 / 54（前 50 句） | 40 |
| 1.0 | 4 / 54 | 62 |
| 1.6（480 ms 版的值） | 3 / 204 | 97 |
| **2.0** | 0 / 204 | 121 |
| 2.4 / 4.0 | 0 / 204 | 146 / 245 |

取 2.0 s。小 chunk 版需要的補零反而比 480 ms 版（1.6 s）長，原因未查；finalize 計算量也較大（每秒音訊要跑 6.25 次 encoder，480 ms 版約 2 次）。

**pre-roll 掃描**（200 句子集）：

| 模型 | pre-roll | CER (%) [95% CI] | 端到端首字 mean / P90 (ms) | 首字（VAD 起點）mean / P90 (ms) |
|---|---|---|---|---|
| 160 ms | 0 | 6.20 [4.98, 7.49] | 1404 / 1599 | 890 / 1119 |
| 160 ms | **100** | 5.54 [4.41, 6.79] | 1364 / 1559 | 850 / 1023 |
| 160 ms | 200 | 5.40 [4.29, 6.60] | 1365 / 1559 | 851 / 1077 |
| 160 ms | 300 | 5.60 [4.54, 6.72] | 1361 / 1549 | 847 / 985 |
| 480 ms | 0 / 100 / 200 / 300 | 5.40 / 5.14 / 4.80 / 4.77 | 1561 / 1515 / 1480 / 1451（mean） | – |

160 ms 的 100 / 200 / 300 差異都在 CI 內，維持 100 ms，與 480 ms 只差模型。

**3000 句對比**（sil 500、pre-roll 100、補靜音 1000 ms）

| 模型 | chunk | CER (%) | 95% CI 以句 | RTF | 端到端首字 mean / P50 / P90 / P99 (ms) | 首字（VAD 起點）mean / P90 | 尾字 mean / P90 (ms) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|
| X-ASR | 160 ms | 5.05 | [4.76, 5.35] | 0.0670 | 1359 / 1369 / 1559 / 1819 | 845 / 1023 | 523 / 601 | 626 |
| X-ASR | 480 ms | **4.11** | [3.85, 4.38] | 0.0301 | 1487 / 1533 / 1623 / 2073 | 972 / 1029 | 451 / 528 | 622 |
| zh-14M | 320 ms | 12.92 | [12.44, 13.41] | 0.0134 | **1179** / 1144 / 1464 / 1844 | 666 / 958 | 416 / 494 | 335 |

取捨圖：[latency_tradeoff.png](latency_tradeoff.png)（`uv run scripts/plot_latency_tradeoff.py` 產生）。

![chunk 大小 vs CER vs 端到端首字延遲](latency_tradeoff.png)

成對比較（同一批 3000 句，160 − 480）：

- **CER +0.94 個百分點，95% 成對 bootstrap CI [0.77, 1.11]**（以句重抽 10000 次、seed 0），顯著變差（相對 +23%）。X-ASR README 自己的 benchmark 上 160 ms 也比 480 ms 差約 0.5–1.2 個百分點，方向一致。
- **端到端首字延遲逐句差：mean −128 ms、P50 −164 ms、P10 −324 ms、P90 +156 ms**；88% 的句子變快，12% 反而變慢。
- 尾字延遲 +72 ms（finalize 補零較長、encoder 次數較多）。RTF 2.2 倍。

**延遲拆解：VAD 確認不是瓶頸**（每條第一個有字的句段）

| 模型 | VAD 確認（語音起點 → start 事件）mean | start 事件 → 首字 mean / P10 / P50 / P90 | start 事件後 100 ms 內出字的比例 |
|---|---|---|---|
| 160 ms | 291 ms | 554 / 409 / 569 / 729 ms | 0.1% |
| 480 ms | 291 ms | 681 / 253 / 733 / 743 ms | 0% |

- 上一版 backlog 的推測是「換小 chunk 後 VAD 確認延遲會露出來」，**數據不支持**：VAD 發出 start 事件時，pipeline 已把 pre-roll 起的約 390 ms 音訊（100 + 291 ms）一次送入，160 ms 模型（需 290 ms 特徵）此時已能解碼，但 P10 仍要再等 409 ms 才出第一個字，幾乎沒有句子在 start 事件當下就出字。
- 也就是說，160 ms 模型的首字是由**模型本身多晚輸出第一個字**決定的，VAD 的 291 ms 被蓋住了。chunk 從 480 縮到 160 ms 只省約 130 ms，遠小於 chunk 差（320 ms）。
- 模型為何晚出字（transducer 的輸出延遲、需要更多右側音訊、或 AISHELL 的首字起音比 VAD 起點晚）目前無法區分，需要有字級時間對齊的評測集（B4 / B5）。
- 對 B2（VAD 預先啟動）的影響：在這兩個模型上，提早建立 stream 預期幫助有限，因為首字不是卡在 VAD。做 B2 前要先確認瓶頸。

### 決定

- **預設模型維持 X-ASR 480 ms。** 160 ms 換到平均 128 ms 的首字延遲，代價是 CER 相對 +23%（顯著）、RTF 2.2 倍、尾字 +72 ms，不划算。
- zh-14M 首字延遲最低（端到端比 X-ASR 480 ms 少約 300 ms），但 CER 高 3 倍，不考慮。
- 首字延遲要再降，方向從「換更小 chunk」改為先弄清楚模型的輸出延遲（B4 / B5 的對齊資料）。

### 測試

- `tests/test_eval.py`：彙整的端到端首字延遲；端到端測試檢查 `lead_silence_ms`、檔名 `_lead1000`，以及端到端延遲大於 VAD 起點延遲。
- `tests/test_asr.py`：`x-asr-160ms-int8` 加入真實模型測試（0.wav 1/28、1.wav 0/26）。
- `uv run pytest` 94 passed、mypy strict、ruff 通過。

### 追加：VAD 預先啟動的上限實驗（B2）

問題：如果 VAD 一看到語音就建立 ASR stream（「預先啟動」），首字延遲最多能省多少？

方法：`--min-speech-ms 0`，VAD 第一個語音 frame 就發出 start 事件。這相當於預先啟動且從不丟棄，是預先啟動在延遲上的上限，不需要改程式。200 句子集前 100 句，其他設定同上（sil 500、pre-roll 100、補靜音 1000 ms）。

| 模型 | min_speech_ms | VAD 確認 mean (ms) | CER (%) | 首字（VAD 起點）mean / P90 (ms) | 端到端首字 mean / P90 (ms) | 每條切段數 | RTF |
|---|---|---|---|---|---|---|---|
| 160 ms | 250 | 291 | 4.59 | 848 / 1024 | 1368 / 1533 | 1.14 | 0.0695 |
| 160 ms | 0 | 65 | 4.59 | 848 / 1024 | 1368 / 1533 | 1.14 | 0.0690 |
| 480 ms | 250 | 291 | 4.59 | 1014 / 1029 | 1534 / 1623 | 1.14 | 0.0308 |
| 480 ms | 0 | 65 | 4.59 | 1014 / 1029 | 1534 / 1623 | 1.14 | 0.0307 |

- stream 平均提早 226 ms 建立，但兩個模型的首字延遲、CER、切段數**完全相同**（到 ms）。VAD 起點位置（含 pre-roll 的送入起點）不變，送進模型的音訊相同；第一個字輸出的時刻取決於音訊何時到達，與 stream 何時建立無關。
- 結論：在 X-ASR 160 / 480 ms 上，VAD 預先啟動**不會降低首字延遲**，只會在誤觸發時多出計算量。AISHELL 是安靜環境的單句，本實驗量不到誤觸發；就算量到，也沒有延遲收益可以拿來抵。
- 決定：不實作 B2。只有在換成首字確實卡在 VAD 的模型（start 事件當下即出字的比例明顯 > 0）時再重新評估。

---

## 2026-10-31　首字延遲拆解：字級對齊（B5 小規模版）

完整的推論過程與結論見 [latency-analysis.md](latency-analysis.md)，這裡記錄做了什麼與數字。
環境同上；評測在 commit `b7a1ac6` + 未提交的程式上執行，程式即本次提交的內容。

### 量測檢查

- partial 在每次送入音訊後讀一次（平常每 10 ms）。VAD 開始事件時會一次補送約 390 ms，補送完才讀。改成補送也每 10 ms 讀一次做對照（100 句、兩個模型）：首字時刻差 mean 0.04 / 0.06 ms、max 1.75 ms，沒有句子差超過 5 ms，首字也從未在補送當下出現。**不需修正。**
- 首字時刻解析度：音訊到達以 chunk（10 ms）為單位，計算時間用 `perf_counter` 實測。

### 做了什麼

- `AsrStream.get_token_timestamps()`（預設 None）；sherpa-onnx 後端讀結果 JSON 的 `timestamps`（40 ms 解析度，以 stream 第一個樣本為 0）。pipeline 在第一個非空 partial 出現時記錄 `Utterance.first_token_audio_s`，評測結果每個句段新增 `first_token_audio_s`（原始音訊時間軸）。
- `scripts/prepare_alignment.py`：從 3000 句集以 seed 20261031 抽 300 句（20 位說話人），以字為單位跑 MFA，輸出 `data/aishell1_align_300/`（manifest、`alignments.jsonl`、10 句抽查檔）。
  - MFA 3.4.3（conda-forge，獨立 conda 環境 `mfa`，不進專案依賴）；Mandarin MFA acoustic v3.0.0（GMM-HMM，MFCC + pitch，frame shift 10 ms，fMLLR 說話人調適）、mandarin_china_mfa 詞典與 G2P，皆 CC BY 4.0。聲學模型訓練資料：Common Voice zh-CN / zh-TW v16.1、aidatatang_200zh、AISHELL-3、THCHS-30、GlobalPhone，**不含 AISHELL-1**（MFA models 文件）。
  - 上游沒有公布 checksum，腳本固定 2026-10-31 下載時的 SHA-256。`mfa model download` 在不穩的網路下沒有重試，改用 curl 下載到 `models/mfa/`。
  - `.lab` 逐字以空白分隔並加 `--no_tokenization`（否則 MFA 要另裝中文分詞套件）。300 句的 1051 個字都在詞典裡。
  - 300 / 300 句對齊成功（全形「ｋ」被 MFA 轉成「k」，比對時兩邊做 NFKC + 小寫）。耗時 34 秒（4 個 job）。首字起點中位數 0.49 s，首字長度中位數 230 ms。
- `scripts/latency_breakdown.py`：三段拆解、`docs/latency_breakdown.png`、數字存 `eval/results/<日期>_aishell1_align_300_breakdown.json`。

### 300 句評測（含 token 時間戳）

| 模型 | CER (%) [95% CI] | 首字（VAD 起點）mean / P90 | 端到端首字 mean / P90 | RTF |
|---|---|---|---|---|
| 480 ms | 3.72 [2.91, 4.58] | 973 / 1029 | 1485 / 1623 | 0.0302 |
| 160 ms | 4.53 [3.66, 5.47] | 842 / 1023 | 1355 / 1589 | 0.0678 |

與 3000 句的值一致（首字延遲差 < 10 ms）。

### 拆解結果（mean / P50 / P90，ms）

| 模型 | 句數 | ① VAD → 真實首字起點 | ② 真實起點 → 模型時間戳 | 首字結束 → 模型時間戳 | ③ 時間戳 → 首字出現 | 合計 |
|---|---|---|---|---|---|---|
| 480 ms | 283 | −26 / −18 / 2 | 432 / 418 / 563 | 190 / 188 / 292 | 561 / 606 / 648 | 966 / 1025 / 1029 |
| 160 ms | 280 | −26 / −18 / 0 | 550 / 553 / 740 | 308 / 301 / 470 | 310 / 320 / 325 | 835 / 861 / 1023 |

- 首字辨識錯（假設首字 ≠ 參考首字）的 17 / 20 句不計入。
- ③ 對照架構推算：「時間戳所在 chunk 結尾 + 130 ms 右側 context − 時間戳」（送入起點以 VAD 起點 − 100 ms 近似），③ 只多出 31–34 ms（P10–P90 27–39 ms）。時間戳在 chunk 內的平均位置 84 ms（480）/ 12 ms（160）：模型幾乎總在下一個 chunk 開頭才輸出首字。

### 結論

- **主因是模型的輸出延遲（②）**：模型在首字唸完 190 / 308 ms 後才輸出它，chunk 越小越晚。VAD 切點準確（①）；③ 是 chunk + 右側 context 的架構成本，可完全解釋。
- 換 160 ms：③ −251 ms、② +118 ms，淨 −131 ms，與 3000 句的 −128 ms 一致。
- 推論端唯一直接針對 ② 的做法是 sherpa-onnx 的 `blank_penalty`（`from_transducer` 有此參數），下一步掃描它對 ② 與 CER 的影響。延遲懲罰訓練、FastEmit 等需重新訓練，不在第一階段範圍。

### 待確認

- **MFA 對齊的人工抽查**：`data/aishell1_align_300/check/` 的 10 句（`.TextGrid` 給 Praat，`.txt` 是 Audacity 標籤）。抽查前 MFA 邊界的誤差範圍未知。

### 測試

- `tests/test_pipeline.py`：假後端的 token 時間戳換算到音訊時間軸；真實模型的首字時間戳落在送入範圍內且不晚於首字出現時刻。
- `uv run pytest` 95 passed、mypy strict、ruff 通過。

---

## 2026-10-31　blank_penalty 掃描（推論端降低輸出延遲）

依字級拆解的結論，主因是模型晚輸出首字。推論端唯一直接針對它的是 sherpa-onnx 的 `blank_penalty`：解碼時從 blank 的 log 機率減去固定值，讓模型較早輸出非 blank token。
環境同上；評測在 commit `1c3d5f3` + 未提交的程式上執行，程式即本次提交的內容。

### 做了什麼

- `SherpaTransducerConfig.blank_penalty`（預設 0）、`load_asr(..., blank_penalty=)`、`run_eval.py --blank-penalty`（可多值掃描，非 0 時結果名稱加 `_bp<值>`）。預設不變。
- `scripts/plot_blank_penalty.py` → [blank_penalty_tradeoff.png](blank_penalty_tradeoff.png)；`latency_breakdown.py --config ..._bp<值>` 拆解各 penalty，結果檔名含設定。
- 先用 20 句試跑（penalty 0 / 1 / 2，160 ms：首字 −15 / −24 ms），再在 300 句對齊集上掃 0 / 1 / 2 / 4 / 6 / 8，兩個模型共 12 組，約 17 分鐘。

### 結果（300 句對齊集）

| 模型 | penalty | CER (%) [95% CI] | 端到端首字 mean / P90 (ms) | ② 真實起點 → 時間戳 mean | 首字結束 → 時間戳 mean | ③ mean |
|---|---|---|---|---|---|---|
| 480 ms | 0 | 3.72 [2.91, 4.58] | 1485 / 1623 | 432 | 190 | 561 |
| 480 ms | 1 | 3.70 [2.91, 4.52] | 1471 / 1623 | 419 | 176 | 560 |
| 480 ms | 2 | 3.84 [3.05, 4.67] | 1445 / 1623 | 397 | 154 | 555 |
| 480 ms | 4 | 4.58 [3.68, 5.56] | 1339 / 1593 | 334 | 92 | 508 |
| 480 ms | 6 | 6.84 [5.70, 8.07] | 1138 / 1533 | 222 | −21 | 431 |
| 480 ms | 8 | 23.58 [20.93, 26.50] | 1062 / 1142 | 36 | −206 | 537 |
| 160 ms | 0 | 4.53 [3.66, 5.47] | 1355 / 1589 | 550 | 308 | 310 |
| 160 ms | 1 | 4.46 [3.59, 5.38] | 1339 / 1560 | 535 | 293 | 311 |
| 160 ms | 2 | 4.42 [3.54, 5.35] | 1308 / 1529 | 504 | 262 | 308 |
| 160 ms | 4 | 5.42 [4.45, 6.44] | 1174 / 1370 | 406 | 163 | 279 |
| 160 ms | 6 | 9.56 [8.14, 11.11] | 1005 / 1180 | 266 | 22 | 251 |
| 160 ms | 8 | 39.41 [35.34, 43.86] | 901 / 1019 | 145 | −94 | 277 |

拆解只計首字辨識正確的句子；首字辨識錯的句數隨 penalty 增加（480：17 → 36，160：20 → 96），penalty ≥ 6 的拆解樣本有偏差。

**成對比較（同一批 300 句，相對 penalty 0；以句重抽 10000 次）**

| 模型 | penalty | CER 差（百分點）[95% CI] | 端到端首字差 (ms) [95% CI] |
|---|---|---|---|
| 480 ms | 1 | −0.02 [−0.35, +0.30] | −15 [−24, −6] |
| 480 ms | 2 | +0.12 [−0.28, +0.50] | −40 [−55, −26] |
| 480 ms | 4 | **+0.86 [+0.31, +1.45]** | −146 [−171, −122] |
| 480 ms | 6 | **+3.12 [+2.25, +4.06]** | −348 [−374, −320] |
| 160 ms | 1 | −0.07 [−0.35, +0.21] | −16 [−21, −10] |
| 160 ms | 2 | −0.12 [−0.50, +0.28] | −47 [−56, −38] |
| 160 ms | 4 | **+0.88 [+0.32, +1.48]** | −181 [−195, −167] |
| 160 ms | 6 | **+5.02 [+3.92, +6.24]** | −350 [−365, −334] |

### 分析

- **penalty 確實壓低了輸出延遲（②）**，與拆解的結論一致：penalty 4 時，模型把首字放在首字結束後 92 ms（480）/ 163 ms（160），penalty 0 時是 190 / 308 ms。
- **penalty ≤ 2：首字延遲少 40–47 ms，300 句上 CER 差異不顯著**；但信賴區間約 ±0.4 個百分點，不能排除小幅變差。
- **penalty 4：再多省約 100–140 ms，CER 顯著變差約 0.9 個百分點**（相對 +23%）。
- **penalty ≥ 6：模型在首字還沒唸完就輸出（首字結束 → 時間戳 ≤ 22 ms，甚至為負），等於在猜**，CER 失控（penalty 8 為 24% / 39%）。
- penalty 也讓 ③ 變小一些（480：561 → 508 ms @ 4），因為字較早輸出，落在 chunk 內較前面的位置。
- 兩條路徑會到達相近的點：480 ms + penalty 4（CER 4.58%、1339 ms）與 160 ms + penalty 0–1（4.46–4.53%、1339–1355 ms）幾乎重疊；160 ms + penalty 2（4.42%、1308 ms）略好，但 RTF 是 480 ms 的 2.3 倍。

### 決定

- **預設維持 X-ASR 480 ms、penalty 0**（2026-10-31 決定，依 300 句結果，不再等 3000 句）。不採用 penalty 2：只省約 40 ms（約首字延遲的 4%，感受不到），300 句的 CER 信賴區間仍容許最多約 +0.5 個百分點；參數保留為選項（`--blank-penalty`）。3000 句驗證列為有空再做。
- 想再降 100 ms 以上的首字延遲，推論端只剩用 CER 換（約每 100 ms 換 +0.6–0.9 個百分點）；不付這個代價就需要重新訓練（延遲懲罰訓練等，第一階段範圍外）。

### 測試

- `tests/test_asr.py`：`blank_penalty` 傳到後端並反映在模型名稱上。`tests/test_eval.py`：非預設設定反映在結果名稱上。
- `uv run pytest` 97 passed、mypy strict、ruff 通過。

---

## 2026-10-31　斷句與尾字延遲診斷（B3、B9 第一步：只量測，不改演算法）

何時判定一句話結束。環境同上；評測在 commit `43282fe` + 未提交的程式上執行，程式即本次提交的內容。數字由 `uv run scripts/endpoint_diagnosis.py` 重現，存 `eval/results/<日期>_endpoint_diagnosis.json`。

### 做了什麼

- **修正尾字延遲量測（B3）**：`run_eval.py --tail-silence-ms`（預設 1500），每條音訊後補數位靜音，句子改由 VAD 靜音計時收尾。3000 句預設設定：CER 4.11% → 4.11%（D 156 → 153）、切段分佈不變；**尾字延遲 451 → 531 ms**（P90 528 → 542），與「min_silence + 約 30 ms」一致，舊值是被截斷的。
- **新指標「最終文字就緒時間」**：`Utterance.final_text_ready_s`，最終結果的完整文字第一次出現在 partial 中的時刻；評測結果逐句段記錄 `end_emitted_s`、`final_text_ready_s`、`final_s`。
- **`run_eval.py --jobs N`**：並行子程序（見 eval.md「並行」）。profile 顯示 85% 時間在 ASR 推論、9% 在 VAD，Python 開銷 < 2%；`--jobs 2` 延遲偏差 ≤ 5 ms、牆鐘時間約減半，`--jobs 4` 會讓 RTF 翻倍、尾字延遲 +49 ms，不適合量延遲。
- `scripts/endpoint_diagnosis.py`：下列四項分析。

### 1. 句尾時間線（300 句對齊集，相對 MFA 真實末字結束點，ms：mean / P10 / P50 / P90）

| | X-ASR 480 ms | X-ASR 160 ms |
|---|---|---|
| VAD 終點（含 30 ms pad） | 50 / −12 / 52 / 124 | 57 / −12 / 53 / 124 |
| VAD 結束事件 | 536 / 470 / 540 / 610 | 543 / 470 / 540 / 610 |
| 最終文字就緒 | 574 / 509 / 580 / 651 | 644 / 577 / 651 / 722 |
| 最終結果（= 真實尾字延遲） | 580 / 518 / 581 / 651 | 659 / 593 / 657 / 728 |
| 結束事件前 partial 已是完整句子 | 8 / 300 | 17 / 300 |

3000 句上同樣只有 41 / 2997 句（1.4%）在結束事件前就緒。**partial 在 VAD 判定結束時幾乎都還不完整**；最終文字是 finalize（補 0.66 s 零 + input_finished）解出的，finalize 只花約 40 ms 計算。

### 2. 不 finalize、持續送靜音（300 句，相對真實末字結束，ms：mean / P10 / P50 / P90）

整條音訊 + 2 s 靜音送入同一個 stream，不經 VAD、不 finalize；時刻為音訊到達時間（不含計算）。

| | X-ASR 480 ms | X-ASR 160 ms |
|---|---|---|
| partial 變成完整句子 | 1081 / 708 / 1100 / 1422 | 942 / 627 / 950 / 1233 |
| 末字後 500 ms，partial 少幾個字 | mean 1.65（完整 5 / 300） | mean 1.26（完整 14 / 300） |
| 末字後 1000 ms | mean 0.64（完整 111 / 300） | mean 0.42（完整 173 / 300） |

「partial 完整」與「partial 最後一次變化」幾乎同時（差 < 10 ms）：partial 一停止變化就是完整句子，但要約 1.1 s，比目前的 VAD 500 ms + finalize（約 580 ms）慢一倍。原因同首字延遲：模型在字唸完後約 190 ms 才輸出它，再加上 chunk 與右側 context 的等待；finalize 的補零等於瞬間送完這段等待。

### 3. 停頓分佈與切段的代價

**句中停頓**（3000 句，VAD 最細設定切出的段間靜音，AISHELL 每條都是單句）：≥ 100 ms 的停頓共 2207 個，mean 415 / P10 160 / P50 320 / P90 768 ms。含至少一個 ≥ T 停頓的句子比例（≈ min_silence = T 時被切開的比例）：

| T (ms) | 200 | 300 | 400 | 500 | 600 | 800 | 1000 | 1500 |
|---|---|---|---|---|---|---|---|---|
| 句子比例 | 36.6% | 27.9% | 21.1% | 15.5% | 11.3% | 6.3% | 3.3% | 1.1% |

與實際評測吻合（500 ms 時 457 / 3000 = 15.2% 被切開）。被切開句子的切點靜音（相鄰句段間距 + 2 × pad）：mean 832 / P10 544 / P50 704 / P90 1226 ms。

**min_silence 掃描**（3000 句，後補 1500 ms；800 / 1000 / 1500 用 `--jobs 2`，尾字延遲約 +5 ms）。CER 差為相對 500 ms 的成對差（百分點，以句重抽 10000 次），「被切開」= 500 ms 時被切成多段的 457 句：

| min_silence | CER | 被切開比例 | 尾字延遲 mean / P90 | 全部 | 被切開的 457 句 | 其餘 2543 句 |
|---|---|---|---|---|---|---|
| 300 | 4.56% | 27.6% | 340 / 351 | +0.45 [+0.34, +0.56] | +0.62 [+0.22, +1.02] | +0.41 [+0.32, +0.51] |
| 500 | 4.11% | 15.2% | 531 / 542 | — | — | — |
| 800 | 3.78% | 6.3% | 839 / 854 | −0.33 [−0.42, −0.25] | −1.85 [−2.34, −1.40] | −0.01 [−0.01, +0.00] |
| 1000 | 3.70% | 3.2% | 1064 / 1079 | −0.41 [−0.50, −0.31] | −2.28 [−2.81, −1.79] | −0.01 [−0.01, +0.00] |
| 1500 | 3.64% | 1.1% | 1525 / 1536 | −0.47 [−0.57, −0.37] | −2.62 [−3.13, −2.10] | −0.01 [−0.02, −0.00] |

- **切段的代價全部在被切開的句子上**：這 457 句的 CER 在 500 ms 時 8.41%，1500 ms 時 5.79%；其餘句子不受影響。
- 2026-10-30 在 200 句、未補句尾靜音時「500 → 800 不顯著」：當時句子多半由音訊結束收尾，800 ms 實際上很少發生；3000 句、補靜音後是顯著的 −0.33 個百分點。
- 尾字延遲 ≈ min_silence + 30–40 ms，每多 100 ms 的門檻就晚 100 ms 出最終結果。

**句間停頓**：AISHELL 沒有句間停頓，只有自錄音訊的 5 個樣本：自由說話（my_voice.wav，3 句）288、448 ms；照稿念（my_voice_script.wav）1536、1856、2976 ms，同檔中句內逗號停頓 320 ms。自由說話的句間停頓落在 AISHELL 句中停頓的中位數附近（320 ms），與句中停頓大幅重疊；但樣本只有 5 個，無法量化重疊程度，需要 B4 的連續語音集。

### 結論

1. **VAD 判定結束前，partial 幾乎不含完整句子**（1.4–3%）。只靠持續送靜音，partial 要約 1.1 s 才完整，比現行做法慢；「partial 不再變化就提早結束」無法直接降低尾字延遲。finalize 的補零是讓句尾快速出字的關鍵。
2. **真實尾字延遲（MFA 末字結束 → 最終結果）**：480 ms 模型 580 ms（P90 651），160 ms 模型 659 ms（P90 728），其中約 490 ms 是 VAD 的 min_silence 等待。
3. **min_silence 是 CER 與尾字延遲之間的直接取捨**：500 → 1000 ms，CER −0.41 個百分點（相對 −10%），尾字延遲 +530 ms。
4. **AISHELL 只能量「切錯」的代價，量不到「合併」的代價**：拉長門檻在這個評測集上看起來幾乎沒有 CER 成本，但句間停頓短於門檻時兩句會合併，最終結果延後。任何結束判定方案都需要 B4 的連續多句評測集才能完整評估。

---

## 2026-10-31　連續多句評測集（B4 第一版）與單一門檻的基準

環境同上；評測在 commit `43282fe` + 未提交的程式上執行，程式即本次提交的內容。各設定的完整數字在 [scoreboard.md](scoreboard.md)。

### 做了什麼

- `scripts/prepare_continuous_data.py`：由 300 句 MFA 對齊集合成 `data/aishell1_continuous_300/`，同一說話人 4 句一條，共 77 條、300 句、223 個句間停頓、23.2 分鐘。每句保留真實起訖點外 50 ms 原音，句間停頓（真實末字結束 → 真實首字起點）0.2 / 0.3 / 0.5 / 0.8 / 1.2 / 2.0 s 各 37–38 個。細節與限制見 eval.md。
- `eval/segmentation.py`：句段對應到參考句子，算句間邊界切開率、句中誤切、漏句、每句真實尾字延遲（依之後的停頓分組）、每句真實首字延遲。`run_eval.py` 遇到含 `sentences` 的 manifest 自動計算，結果在 `metrics.segmentation`。
- 先用 `--limit 5` 試跑確認對應邏輯，再跑全部 77 條 × 5 種 min_silence（`--jobs 1`，每組約 1 分鐘）。
- README 基準表改為後補 1.5 s 靜音的正式值（三個模型都重跑）：尾字延遲 X-ASR 480 / 160 / zh-14M 為 531 / 604 / 494 ms（舊值 451 / 523 / 416 被截斷）；RTF 分母多了後補靜音，不能與舊值比較。CER、首字延遲不變。

### 結果

- **句間切開是門檻的階梯函數**：比門檻長一檔以上的停頓全部切開，比門檻短的幾乎全部合併（300 ms 門檻下 0.2 s 停頓切開 8%），停頓剛好等於門檻時切開 70–76%（300 / 500 / 800 ms 門檻）。這裡的停頓是真實語音間距，VAD 看到的靜音還要扣掉發音收尾的弱能量。
- **合併的代價很大**：被併入下一句的句子，真實尾字延遲從約 0.4–0.6 s 變成平均 2–10 s（要等下一句甚至下下句說完）。500 ms 門檻下，0.2 / 0.3 s 停頓後的句子平均 4.9 / 5.8 s 才有最終結果。
- **切碎的代價**：句中誤切 82 / 49 / 16 / 8 / 1 句（300 / 500 / 800 / 1000 / 1500 ms），與單句集一致；CER 4.00 / 3.53 / 3.26 / 3.26 / 3.23%（300 句的信賴區間約 ±0.8 個百分點，只看趨勢）。
- **真實首字延遲**（與 VAD 無關，B5）：約 990–1000 ms（P90 1094–1156），與對齊集上的「VAD 起點首字延遲 973 ms + VAD 早 26 ms」一致。
- 單一門檻下沒有兩全的設定：門檻短則句中誤切多、CER 高；門檻長則短停頓的句子被合併，最終結果延後數秒，且正確切開的句子也要多等。

### 待確認

- 句間停頓長度是人為分檔，真實口語中各長度的比例未知，不能把各分檔加權成單一數字。
- MFA 邊界仍未人工抽查。沒有噪音（B4 的噪音部分未做）。

---

## 2026-10-31　結束判定方案 A：暫定結果 + 長門檻定稿（B9）

環境同上；評測在 commit `7037637` + 未提交的程式上執行，程式即本次提交的內容。數字在 [scoreboard.md](scoreboard.md)。

### 做了什麼

- `VadConfig.pause_ms`：語音段中靜音滿 pause_ms 時發出「停頓」事件（每次停頓一次、不結束語音段）。離線切段忽略它。
- pipeline：收到停頓事件時，另開一個 ASR stream，把這句目前為止送入的全部音訊重新解碼並 finalize，輸出暫定結果（`PipelineEvent("provisional")`、`Utterance.provisionals`）；原 stream 不中斷。之後若又開始說話，最終結果取代暫定結果；靜音滿 min_silence 照常定稿。另開 stream 是因為 sherpa-onnx 的 stream 無法複製。
- `run_eval.py --pause-ms`（可掃描）；`metrics.provisional`（暫定結果數、被取代數、最後一個與最終結果相同的比例、重新解碼耗時）；連續集多了每句「結果延遲」（第一個涵蓋整句的暫定或最終結果）。
- 先用連續集 `--limit 5` 試跑，再跑全部 77 條（pause 200 / 300 × min_silence 800 / 1000）與 300 句對齊集（pause 200 / min_silence 1000）。預設不變（pause 關閉）。

### 結果

- **最終文字不受影響**：連續集 4 組設定都與同 min_silence 的單一門檻逐句相同（77 / 77 條，句段邊界也相同）。因此 3000 句的 CER 直接等於 min_silence 1000 的 3.70%（預設 500 為 4.11%），不需重跑。
- **結果延遲**（pause 200 / min_silence 1000）：
  - 單句（對齊集，真實末字結束起算）：409 ms（P90 498），預設 580 ms（P90 651）。
  - 連續集：停頓 ≥ 0.3 s 的句子 441–538 ms；預設下 0.3 s / 0.5 s 停頓後的句子因合併要 5.8 / 2.3 s，0.8 s 以上約 575 ms。
  - 停頓 0.2 s 仍有 1.5 s（P90 4.0 s）：VAD 看到的靜音常短於 200 ms，停頓事件不一定發出。
- **代價 1：暫定結果會被修正**。單句集 35% 的句子先出現至少一個半句的暫定結果（1 個 22%、2 個以上 13%）；連續集 462 個暫定結果中 303 個被取代。最後一個暫定結果與最終結果相同 98.7–99%。
- **代價 2：重新解碼**。耗時與句段長度成正比，單句 mean 148 / P90 210 ms、連續集 239 / 432 ms，直接加在結果延遲上（pause 200 + 重新解碼 ≈ 結果延遲的大部分）。RTF 單句 0.027 → 0.057、連續集 0.032 → 0.101（每次停頓都從頭解碼一次）。
- pause 300 比 200 少 18% 的暫定結果，但 0.3 s 停頓抓不到（2.0 s）、其餘慢約 90 ms。

### 下一步候選

- **方案 A2：停頓時 finalize 原 stream，恢復說話時才重新解碼**。暫定結果只需 finalize（約 40 ms），結果延遲預期降到約 pause + 40 ms；重新解碼移到恢復說話時（只有 35% 的句子），代價變成恢復後的 partial 晚一次重新解碼的時間。最終文字仍與長門檻相同。

---

## 2026-11-01　結束判定方案 A2：停頓時 finalize、恢復說話時才重新解碼（B9）

環境同上；評測在 commit `b311a49` + 未提交的程式上執行，程式即本次提交的內容。數字在 [scoreboard.md](scoreboard.md)。

### 做了什麼

- VAD 新增「恢復」事件：停頓事件之後再出現語音 frame 時發出（結束事件時重置，下一段開頭不會誤發）。
- `PipelineConfig.provisional = "finalize"`：停頓時直接 finalize 原 stream 當暫定結果，之後的音訊只緩衝；恢復事件時另開 stream 重新解碼這句的全部音訊並接手（`Utterance.resume_redecode_s` 記錄耗時）；沒有恢復就以暫定結果定稿。`run_eval.py --provisional finalize`（名稱加 `fin`）。
- 先 `--limit 5` 試跑，再跑連續集 77 條（pause 200 / 300 × min_silence 800 / 1000）、300 句對齊集，3000 句用 `--jobs 4` 只確認 CER。

### 結果（pause 200 / min_silence 1000）

| | 預設（500 ms） | 方案 A | 方案 A2 |
|---|---|---|---|
| CER，3000 句 | 4.11% | 3.70% | 3.75% |
| 單句：真實末字結束 → 第一個完整結果（mean / P50 / P90 ms） | 580 / 581 / 651 | 409 / 409 / 498 | **299 / 295 / 366** |
| 連續集：停頓 ≥ 0.3 s 後的句子（mean ms） | 0.3 / 0.5 s 停頓要 5.8 / 2.3 s，0.8 s 以上約 575 | 441–538 | **276–296** |
| 產生暫定結果 | — | 148 ms（單句）/ 239 ms（連續） | **49 ms** |
| 恢復時重新解碼 | — | — | 單句 83 / P90 143 ms，連續 179 / P90 364 ms |
| RTF（單句 / 連續） | 0.027 / 0.032 | 0.057 / 0.101 | **0.033 / 0.071** |

- **A2 的延遲與計算量都明顯優於 A**：暫定結果只要補零解碼，結果延遲約等於 pause 200 ms + VAD frame 對齊 + 49 ms。
- **代價：最終文字與 A 不完全相同**。3000 句 CER 相對 A +0.05 個百分點（成對 95% CI [+0.02, +0.08]，顯著），38 / 3000 句不同（多 27、少 6 個錯）；相對預設仍 −0.36 [−0.46, −0.26]。連續集 2 / 77 條不同，都是停頓後句尾少一個字（如「怎料」→「怎」）。
- **原因**：停頓後很快 finalize，句尾後面只有 0.2–0.3 s 真實靜音、其餘是數位零。單獨驗證（300 句，句尾後接 x 秒真實音訊再補 y 秒零，與長靜音解碼比較）：0.2 s 真實靜音時補零 0.66 / 1.0 / 1.5 s 分別有 231 / 90 / 2 句不同；1.5 s 以上已不再改善（`TRANSDUCER_SPECS` 已用 1.6 s），剩下約 1/300 是數位零與真實靜音的差異，A2 無法完全避免。方案 A 的「最後一個暫定結果與最終結果相同 98.7–99%」也是同一現象，只是 A 的最終結果另由原 stream 在長靜音後產生。
- 送入粒度不影響解碼：整段一次送入、10 ms 逐塊送入、前半一次 + 後半逐塊，40 條連續集的結果完全相同。所以 A2 恢復時重新解碼與原 stream 等價。
- 修正率同 A（單句 35% 的句子先出現半句暫定結果）；被修正的句子恢復後的 partial 晚一次重新解碼（mean 83–179 ms）。

### 可選的折衷：A2 + 定稿時重新解碼

停頓後沒有恢復、到 min_silence 定稿時，再以完整緩衝（含長靜音）重新解碼一次當最終結果，最終文字即與 A 相同（CER 3.70%）。暫定結果延遲不變（約 300 ms），定稿晚約一次重新解碼的時間，計算量約回到 A 的水準。未實作。

---

## 2026-11-01　非串流模型對照：SenseVoice Small（小樣本）

環境同上；評測在 commit `8e9b618` + 未提交的程式上執行，程式即本次提交的內容。數字在 [scoreboard.md](scoreboard.md)「模型對照」一節。

### 做了什麼

- `scripts/download_models.sh` 加入 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17`（SHA-256 校驗）；來源、訓練資料（未公開，是否含 AISHELL-1 未知）與授權（FunASR Model License v1.1，未明文提到商業用途）寫進 [models.md](models.md)。
- `src/asr/sense_voice.py`：非串流後端，`accept_waveform` 只累積音訊、`finalize` 整段辨識；選用 `partial_every_s` 定期整段重新辨識當偽 partial。登記 `sensevoice-small-int8`、`sensevoice-small-int8-p480`（0.48 s）。language auto、不做 ITN。
- 依要求先用小樣本：對齊集前 100 句、連續集前 25 條（100 句），X-ASR 與兩種 SenseVoice 各跑 500 ms 與 A2。

### 結果

- **CER**：SenseVoice 單句 2.16% vs X-ASR 2.93%，成對差 −0.77 個百分點 [−1.89, +0.28]，100 句上**不顯著**；連續集 2.15 vs 2.36%（A2）。
- **首字延遲**：沒有偽 partial 時說話中沒有文字，首字約 4.4 s（≈ 整句長度 + 靜音門檻）。加偽 partial 後 **443 ms**，比 X-ASR 的 1061 ms 快約 620 ms：沒有 chunk 與右側 context 的等待，也沒有串流模型的輸出延遲。
- **說完 → 完整結果**（A2）：SenseVoice 368 ms vs X-ASR 296 ms，整句辨識（112 ms）比 X-ASR 的補零 finalize（49 ms）慢。
- **計算量**：偽 partial 每次重新辨識整段，單句 RTF 0.107、連續集（合併成長句段）0.309，X-ASR A2 為 0.029 / 0.057。
- **記憶體**：SenseVoice 峰值 RSS 940–1013 MB，X-ASR 約 535 MB；模型 230.7 vs 163.6 MB。

### 指標缺口

首字延遲只計 partial。X-ASR 一定先有 partial，沒有影響；非串流模型在 A2 下的第一個可見文字是暫定結果（單句 3528 ms），現行指標顯示 5035 ms。應改為「第一次看到文字 = partial 或暫定結果，取早者」。

### 待確認

- 100 句不足以判定 CER 差異，需要 3000 句（只看 CER 可用 `--jobs 4`）。
- 偽 partial 的計算量隨句長平方成長，長句或連續說話時的最壞情況未量。

---

## 2026-11-01　SenseVoice 記憶體追查（B11）

環境同上；commit `2824385` + 未提交的程式（`scripts/probe_footprint.py`，即本次提交）。全部在專案環境（Python 3.11.15）執行，重複 3 次數字相差 < 5 MB。

問題：評測時 SenseVoice 峰值 RSS 約 940–1010 MB，X-ASR 約 535 MB。多出來的是模型必需，還是可以省掉？

方法：`uv run scripts/probe_footprint.py`，在同一程序內以 `vmmap --summary` 分階段記錄 RSS、physical footprint（iOS jetsam 的依據）、malloc 存活配置與碎片（已釋放未歸還）；`--ort-direct` 用指定版本的 onnxruntime 直接載入 `model.int8.onnx`（228 MB）。

**經專案後端載入（sherpa-onnx 2.0.1，內附 ORT 1.28.2），MB**

| 階段 | SenseVoice footprint | malloc 存活 / 碎片 | X-ASR footprint | malloc 存活 / 碎片 |
|---|---|---|---|---|
| import 後 | 24 | 7 / 2 | 23 | 7 / 2 |
| 載入後 | **850** | 497 / **314** | 270（載入時峰值 429） | 223 / 186 |
| 解碼 3 s 後 | 859 | 519 / 313 | 282 | 255 / 176 |
| 解碼 10 s 後 | 886 | 570 / 288 | 282 | 256 / 176 |
| 解碼 30 s 後 | 941 | 641 / 272 | 283 | 259 / 178 |

**直接用 onnxruntime 載入 SenseVoice（只載入，不推論），相對載入前的增量，MB**

| ORT | 載入方式 | Δfootprint | footprint 峰值 | Δmalloc 存活 | Δ碎片 |
|---|---|---|---|---|---|
| 1.28.0 | 檔案路徑 | 817 | 853 | 483 | 309 |
| 1.31.1（專案依賴） | 檔案路徑 | **308** | **336** | 255 | 24 |
| 1.28.0 | 先讀進記憶體 | 1304 | 1331 | 709 | 311 |
| 1.31.1 | 先讀進記憶體 | 760 | 788 | 483 | 24 |

「先讀進記憶體」是 Python 端以 bytes 建 session：onnxruntime 的 Python 包裝會保留這份 bytes，多了一份模型大小的存活配置，不能直接代表 sherpa-onnx（C++ 讀檔建 session 後釋放緩衝，`csrc/offline-sense-voice-model.cc`）。

### 結論

1. **SenseVoice 載入後的 850 MB 中，約 500 MB 是 ORT 1.28 的載入開銷，不是模型必需。** 同一個檔案用 ORT 1.28.0 載入 footprint +817 MB，與 sherpa-onnx（內附 1.28.2）的 826 MB 幾乎相同；用 ORT 1.31.1 只要 +308 MB（峰值 336），碎片 309 → 24 MB、存活配置 483 → 255 MB。和 X-ASR 的情況（B11，2026-10-30）同一個原因，只是 SenseVoice 是單一 228 MB 的大檔，放大得更明顯。
2. **用 ORT ≥ 1.30 時，SenseVoice 的載入代價估計約 330 MB footprint**（import 24 + 308），X-ASR 用 ORT 1.31.1 直接載入約 261–270 MB（B11）。兩者差距從約 580 MB 縮小到約 60 MB。這是只載入、未推論的數字，SenseVoice 推論時另外隨音訊長度增加（經 sherpa-onnx 實測約 3 MB / 秒音訊，30 s 音訊約 +90 MB；車載 1–3 s 的短句約 +10 MB）。
3. 用 Python 3.13 環境量到的數字（Δfootprint 150–225 MB 等）與 3.11 不同且不穩定，不採用；記憶體數字要在同一個 Python 版本下比較。

### 要把這個代價真的拿掉

sherpa-onnx 2.0.1（PyPI 最新）內附 ORT 1.28.2，只升級 Python 套件無法解決。可行做法：(1) 以 ORT ≥ 1.30 自行編譯 sherpa-onnx；(2) SenseVoice 改成直接用 onnxruntime 推論（需自行實作 80 維 fbank、LFR、CMVN 前處理與 CTC 解碼）；(3) 第三階段轉 Core ML 時重新量。不論哪種，都要再量推論時的 footprint，才能確認上面的估計。

---

## 2026-11-01　SenseVoice 改用 onnxruntime 1.31.1 直接推論

環境同上；評測在 commit `16342bb` + 未提交的程式上執行，程式即本次提交的內容。

### 做了什麼

- `SenseVoiceOrtAsr`（`src/asr/sense_voice.py`）：照 sherpa-onnx 2.0.1 的實作重寫前處理與解碼，用專案依賴的 onnxruntime 以檔案路徑載入模型。fbank 用 kaldi-native-fbank 1.22.5（k2-fsa，Apache-2.0，sherpa-onnx 用的同一個函式庫；新增為專案依賴），80 維、hamming、dither 0、樣本 ×32768；LFR 7 / 6（`csrc/lfr.cc`：ceil(T/6) 個輸出，邊界重複首 / 末 frame）；CMVN 取自模型 metadata；CTC greedy（blank 0）後丟掉前 4 個 token。登記 `sensevoice-small-int8-ort`、`sensevoice-small-int8-ort-p480`。
- `SenseVoiceStream` 改為接收「整段音訊 → 文字」的函式，兩個後端共用累積與偽 partial 的邏輯。

### 驗證

- **與 sherpa-onnx 版在 300 句對齊集上逐句完全相同**（300 / 300）；測試另比對兩個官方測試音訊，並檢查 LFR 的 frame 數。
- 同樣的 100 句子集重跑：CER（2.16 / 2.15–2.29%）、首字（443 ms）、結果延遲（A2 385 / P90 454 ms）、RTF 都與 sherpa-onnx 版相同。

### 記憶體

| | footprint 載入後 | 解碼 3–5 s 後 | 解碼 30 s 後 | 評測峰值 RSS（單句 / 連續） |
|---|---|---|---|---|
| SenseVoice（sherpa-onnx，ORT 1.28.2） | 850 | 859–868 | 940 | 939 / 1006 |
| **SenseVoice（onnxruntime 1.31.1）** | **346** | **358–366** | 459 | **437 / 532** |
| X-ASR（sherpa-onnx，ORT 1.28.2） | 271（載入峰值 427） | 279 | 279 | 536 / 538 |

- 先前估計的「改用新版 ORT 後約 330 MB」得到確認：載入後 346 MB，車載長度（3–5 s）的句子推論後 358–366 MB。
- **SenseVoice（ORT 1.31.1）的峰值已低於目前的 X-ASR**：X-ASR 仍經 sherpa-onnx（ORT 1.28.2）載入，載入時峰值 427 MB、評測峰值 RSS 約 536 MB。兩者在同一個 ORT 版本下的公平比較，要等 X-ASR 也不經 sherpa-onnx 載入（B11 估計約 261–270 MB）。
- SenseVoice 推論時 footprint 約隨音訊長度 +3.8 MB / 秒（attention 看整段），X-ASR 不隨長度增加；長語音時差距會拉大。

### 結論

記憶體翻倍是 sherpa-onnx 內附舊版 ORT 造成的，不是 SenseVoice 本身的代價。改用新版 ORT 後，SenseVoice 在車載短句上的記憶體與 X-ASR 相當，剩下的代價主要是偽 partial 的計算量（RTF 0.107 / 0.32）與授權。

---

## 2026-11-01　X-ASR 480 ms 改用 onnxruntime 1.31.1 直接推論（B11）

環境同上；評測在 commit `eaf4ff1` + 未提交的程式上執行，程式即本次提交的內容。

### 做了什麼

- `ZipformerOrtAsr`（`src/asr/zipformer_ort.py`）：不經 sherpa-onnx，以專案依賴的 onnxruntime 1.31.1 用檔案路徑載入 encoder / decoder / joiner，照 sherpa-onnx 2.0.1 的串流 transducer 實作重寫：
  - fbank：kaldi-native-fbank 串流 `OnlineFbank`，80 維、povey、dither 0、snip_edges=False、low_freq 20、high_freq −400，樣本維持 [−1, 1]。用 knf 預設的 high_freq 0（模型自帶 `test_onnx.py` 的設定）會讓 9 句測試中 2 句出現不同的字，所以照 sherpa-onnx 的 −400。
  - encoder：每次 T = 61 個 frame、前進 48 個，「已處理 + 61 < 可用 frame」才送；116 個狀態張量（19 層 × 6 + embed_states、processed_lens）初始全零，以 `new_<名稱>` 輸出取代。
  - greedy search：decoder 初始輸入 [−1, blank]；每個 encoder frame 一次 joiner，blank_penalty > 0 時減在 blank 的 logit 上，不是 blank / `<unk>` 就輸出並更新 decoder；token 時間 = 輸出 frame × 40 ms。
  - 文字：token 開頭的「▁」視為空白，但前後都是中文時省略（「昨天是 MONDAY TODAY IS 礼拜二」）。sherpa-onnx 的 Python 結果就是這個格式，逐字比對才能相同。
  - 句尾補零沿用 `TRANSDUCER_SPECS` 的 1.6 s。登記 `x-asr-480ms-int8-ort`；預設模型仍是 sherpa-onnx 版。

### 驗證

- **200 句冒煙集與 sherpa-onnx 版逐句相同**：200 / 200 條 hyp 相同；238 / 238 段的文字、VAD 起訖點、首字在音訊中的位置、partial 次數全部相同（只有牆鐘時間不同）。
- 測試：X-ASR 自帶的 4 個中英混合音訊 + 2 個 WenetSpeech 音訊，blank_penalty 0 與 1.5 下文字與 token 時間都與 sherpa-onnx 相同；另跑既有的 chunk 大小無關、partial 串流、stream 互相獨立等測試。

| 200 句冒煙集（500 ms，jobs 1） | CER | RTF | 首字 mean / P90 | 尾字 mean | 峰值 RSS（載入後） |
|---|---|---|---|---|---|
| X-ASR（sherpa-onnx，ORT 1.28.2） | 5.14% | 0.0282 | 1002 / 1031 | 535 | 535（—） |
| **X-ASR（onnxruntime 1.31.1）** | 5.14% | 0.0291 | 1003 / 1034 | 537 | **348**（325） |

- RTF 多約 3%、尾字多 2 ms：greedy search 改在 Python 逐 frame 呼叫 joiner（每個 480 ms chunk 12 次 joiner + 每個字 1 次 decoder），是 Python 與 ORT 呼叫開銷。延遲上可以忽略。

### 記憶體（`scripts/probe_footprint.py`，MB）

| 階段 | sherpa-onnx footprint（峰值） | onnxruntime 1.31.1 footprint（峰值） | RSS sherpa / ort | malloc 碎片 sherpa / ort |
|---|---|---|---|---|
| 載入後 | 274（**426**） | 274（**274**） | 476 / 303 | 185 / 28 |
| 解碼 3 s 後 | 279（426） | 289（289） | 483 / 319 | 177 / 10 |
| 解碼 30 s 後 | 280（426） | 290（290） | 484 / 320 | 178 / 10 |

- **載入時的 footprint 峰值 426 → 274 MB、RSS 476 → 303 MB**：ORT 1.28 載入時的暫時配置與碎片消失。
- 載入後的 footprint 兩邊相同（274），穩定狀態的 footprint 反而多約 10 MB（289 vs 279）：B11 估計的「約 261–270 MB」只算載入。RSS 差約 170 MB 而載入後 footprint 相同，推測 ORT 1.28 留下的碎片頁多已標記為可回收、不計入 footprint（未驗證），所以 footprint 的差別只出現在峰值。多出的 10 MB 原因未查（可能是 Python 端的 knf 特徵與 numpy 狀態複本）。
- 與 SenseVoice（ORT 1.31.1）在同一個 ORT 版本下比較：X-ASR 載入後 274 / 推論後 289–290，SenseVoice 346 / 358–366（30 s 459）。

### 結論

X-ASR 不經 sherpa-onnx 推論，辨識結果不變，評測峰值 RSS 535 → 348 MB，iOS 更在意的 footprint 峰值 426 → 274 MB；代價是 RTF +3%。

### 補記：160 ms 版

`x-asr-160ms-int8-ort` 以同一個 `ZipformerOrtAsr` 登記（同為 zipformer2，T / chunk 由 metadata 讀取，句尾補零沿用 `TRANSDUCER_SPECS` 的 2.0 s）。pytest（132 passed）：6 個測試音訊在 blank_penalty 0 / 1.5 下文字與 token 時間都與 sherpa-onnx 版相同，通用測試（CER 上限、chunk 大小無關、partial 串流、stream 獨立）也通過。footprint 尚未量。

冒煙集逐句比對（合併時補做，`--jobs 2`，RTF / RSS 僅供參考）：200 / 200 條 hyp 相同；238 / 238 段的文字、VAD 起訖點、首字位置、partial 次數相同。CER 5.54% 兩邊相同，評測峰值 RSS 530 → 338 MB。

---

## 2026-11-01　決定：結束判定預設改為方案 A2

環境同上；commit `eaf4ff1` + 未提交的程式，程式即本次提交的內容。

### 決定

`PipelineConfig()` 的預設由單一門檻（min_silence 500 ms）改為方案 A2：`VadConfig(min_silence_ms=1000, pause_ms=200)` + `provisional="finalize"`。停頓 200 ms 先 finalize 出暫定結果，靜音滿 1000 ms 定稿，恢復說話時才重新解碼。`src/cli.py` 與 `eval/run_eval.py` 的 `--pause-ms` / `--provisional` 預設取自 `PipelineConfig()`，所以 `run_pipeline.py`、`run_eval.py`、`demo_mic.py` 預設都是 A2。退回舊設定用 `--min-silence-ms 500 --pause-ms 0`。

- **理由**：說完 → 完整結果 580 → 299 ms（300 句對齊集，真實末字結束起算，P90 651 → 366）；CER 4.11 → 3.75%（3000 句，成對差 −0.36 個百分點 [−0.46, −0.26]）；連續多句集上 0.3 / 0.5 s 的句間停頓不再因合併而要等數秒（約 290 ms）。相對方案 A，延遲與計算量都更低。
- **代價**：35% 的句子會先出半句暫定結果、之後被修正；RTF 0.027 → 0.033（對齊集；連續集 0.032 → 0.071）；早 finalize 偶爾漏句尾的字，CER 比方案 A 高 0.05 個百分點（顯著）。0.2 s 的句間停頓仍會被合併。

### 程式變更

- `run_eval.py`：`--pause-ms 0` 時 `--provisional` 沒有作用，統一記成 redecode 並去重（否則預設 finalize 會讓 pause 0 的組合被略過）。注意掃描 `--min-silence-ms` 時要一起給 `--pause-ms 0`，否則每個門檻都開暫定結果，且不大於 200 ms 的門檻會被略過。
- `run_pipeline.py` 的 JSON 多 `provisionals` 與 `settings.provisional`。
- 測試：預設設定的單元測試、命令列退回單一門檻的測試。內附 fixture 兩句間隔 1.0 s，預設 1000 ms 不切開，兩句切段的整合測試改用單一門檻 500 ms，另加預設下「一段定稿、第一句先有暫定結果」的整合測試。

### 驗證（依要求只跑小集合，未重跑 3000 句）

改為預設後不加任何參數重跑，與先前以參數指定 A2 的結果（`b311a49`+）逐句比較：

| 評測集 | 對照 | 最終文字相同 | 暫定結果文字相同 | CER (%) 本次 / 對照 | RTF 本次 / 對照 |
|---|---|---|---|---|---|
| 200 句冒煙集 | 3000 句 A2 中重疊的 199 句 | 199 / 199 | 199 / 199 | 4.80（200 句） | 0.039 / —（對照為 `--jobs 4`） |
| 300 句對齊集 | 對齊集 A2 | 300 / 300 | 300 / 300 | 3.32 / 3.32 | 0.033 / 0.033 |
| 連續多句集 77 條 | 連續集 A2 | 77 / 77 | 77 / 77 | 3.30 / 3.30 | 0.085 / 0.071 |

- 辨識結果逐句相同，3000 句 CER 3.75% 沿用先前的數字（3000 句中重疊的 199 句逐句相同）。
- 連續集的 RTF 與結果延遲（各停頓 +7–11 ms）略高，可能是重跑時同機有其他工作在跑（未確認）；文字相同，延遲差只來自計算時間。

---

## 2026-11-01　合併：X-ASR onnxruntime 後端 + A2 預設

`feat/x-asr-ort` 與 `feat/a2-default` 合併進 `feat/latency-b1`（merge commit `d614e74`）。pytest 135 passed、mypy、ruff 通過。依決定只跑 200 句冒煙集驗證，3000 句未重跑。

**200 句冒煙集，合併後預設（A2：pause 200 / min_silence 1000），jobs 1**

| 後端 | CER (%) [95% CI] | RTF | 首字 mean / P90 (ms) | 端到端首字 mean / P90 (ms) | 峰值 RSS (MB) |
|---|---|---|---|---|---|
| X-ASR（sherpa-onnx） | 4.80 [3.76, 5.93] | 0.0335 | 1003 / 1030 | 1512 / 1624 | 541 |
| X-ASR（onnxruntime 1.31.1） | 4.80 [3.76, 5.93] | 0.0318 | 1002 / 1029 | 1511 / 1623 | **353** |

- 兩個後端 200 / 200 條 hyp 相同，205 / 205 段的文字、VAD 起訖點、首字位置、暫定結果相同。
- sherpa 版的結果與 `feat/a2-default` 分支上的同設定冒煙結果 200 / 200 條相同，合併沒有改變行為。
- **決定（2026-11-01）：預設模型改為 `x-asr-480ms-int8-ort`**（`DEFAULT_MODEL`）。理由：辨識結果逐句相同，峰值 RSS 541 → 353 MB，RTF 不變（0.0335 → 0.0323，雜訊範圍內）。不給 `--model` 時的冒煙集重跑：CER 4.80%、峰值 RSS 353 MB，200 / 200 條 hyp 與 sherpa 版相同。sherpa-onnx 版保留為 `x-asr-480ms-int8`，`--model x-asr-480ms-int8` 可退回。

---

## 2026-11-01　B9 短停頓：縮短 pause 門檻讓 0.2 s 句間停頓也出暫定結果

環境：Apple M5 / macOS 27.0.1，單執行緒，X-ASR 480 ms INT8（onnxruntime 1.31.1 後端），min_silence 1000、`--provisional finalize`。評測在 commit `d016396` + 未提交的程式上執行，程式即本次提交的內容。依要求只用 300 句連續多句集與 300 句對齊集，未跑 3000 句。數字在 [scoreboard.md](scoreboard.md)「B9 短停頓」。

**負載**：評測時同機有其他工作階段在跑評測（load average 4–5）。辨識結果、暫定結果、切段是確定性的（兩批重跑逐句相同），但 RTF 與延遲的絕對值偏高、彼此不穩（A2 對照的對齊集 RTF 兩批 0.047 / 0.061，先前閒置時 0.033）。比較只看同批，計算量另以與負載無關的「解碼音訊 / 音訊」衡量。

### 問題在哪

A2 下 0.2 s 停頓後的句子平均 1.3 s（P90 3.7 s）才有完整結果。VAD 的靜音是「最後一個語音 frame 結束」到現在的長度，以 32 ms frame 計，所以 **pause 200 實際要 7 個 frame = 224 ms**。`scripts/pause_analysis.py screen` 逐 frame 量連續集：0.2 s 句間停頓中機率 < 0.35 的最長連續段，61% 滿 224 ms、84% 滿 192 ms、100% 滿 160 ms。不滿的那 39% 不發停頓事件，與下一句合併，要等下一句說完。

### 做了什麼

- `scripts/pause_analysis.py`（計算在 `eval/pause.py`，有測試）：`screen` 不跑 ASR，估計各 pause / 機率門檻下句間停頓抓到的比例與句中出現停頓的比例；`cost` 讀 run_eval 結果，算對齊集的結果延遲（MFA 真實末字起算）與半句暫定結果比例、連續集的句間 / 句中暫定結果數、解碼音訊量。
- `--neg-threshold`（`src/cli.py`、`run_eval.py`；非預設的 threshold / neg_threshold 會加進結果檔名 `_th` / `_neg`），用來試調 VAD 遲滯。
- 篩選後跑全鏈路（兩個集合）：pause 200（對照）/ 190 / 150 / 120，以及 pause 200 + neg_threshold 0.5。pause 200 / 150 / 120 另外同批連續重跑一次比延遲。

### 結果

| | A2（pause 200，實際 224 ms） | pause 150（160 ms） | pause 120（128 ms） |
|---|---|---|---|
| 連續集：0.2 s 停頓後的句子，結果延遲 mean / P90 | 1314 / 3688 ms | **232 / 285** | **186 / 246** |
| 連續集：其他停頓 | 296–315 | 225–247 | 184–203 |
| 對齊集：結果延遲 mean / P90 | 339 / 405 | 283 / 351 | 226 / 295 |
| 對齊集：先出現半句暫定結果的句子 | 36% | 40% | 41% |
| 連續集：句中暫定結果（之後被修正） | 174 | 204 | 218 |
| 解碼音訊 / 音訊（對齊 / 連續） | 1.324 / 2.386 | 1.370 / 2.585 | 1.392 / 2.612 |
| CER 對齊 / 連續 | 3.32 / 3.30% | 3.28 / 3.30% | 3.30 / 3.30% |

（延遲為同批第 2 批，含負載造成的計算時間；對照的閒置值為 299 ms。）

- **pause 150 / 120 都消除了 0.2 s 停頓的合併**（暫定結果層面；最終斷句仍由 min_silence 1000 決定，句間切開率與句中誤切不變），其餘句子也因停頓事件早 64 / 96 ms 而快 60–110 ms。
- **代價**：半句暫定結果的句子 +4 / +5 個百分點，連續集句中暫定結果 +17% / +25%；每次被修正的暫定結果在恢復說話時要重新解碼一次，解碼量單句 +3.5% / +5%、連續 +8% / +9%。CER 不變：連續集最終文字逐條相同；對齊集 4 / 1 句不同（多為句尾的字變對，錯字 143 → 141 / 142，300 句看不出差異）。
- **RTF 沒有可信的差**：同批 pause 200 / 150 / 120 為 0.061 / 0.070 / 0.050（對齊）、0.105 / 0.101 / 0.081（連續），排序與解碼量不一致，被負載蓋過。依解碼量估計（上限），閒置時 RTF 約增加 3.5–8%（pause 150：約 0.033 → 0.034、0.071 → 0.077），未實測，待閒置時重量確認。
- **pause 190**（192 ms）只抓到 84% 的 0.2 s 停頓，平均 626 ms（P90 2245）。
- **調 VAD 閾值幫助很小**：neg_threshold 0.5（等於 threshold、取消遲滯）時 0.2 s 停頓仍有 37% 抓不到，平均 1235 ms；篩選表顯示即使只為停頓另用 < 0.8 的機率門檻，在 224 ms 下也只抓到 87%，而要抓滿就得縮短長度，句中誤觸發還比單純縮短 pause 多（< 0.8 / 192 ms 句中 44%，< 0.35 / 160 ms 句中 40%）。所以沒有實作「停頓專用機率門檻」。

### 沒有試的方向

- **能量訊號**：Silero 的機率已經是逐 frame 的細訊號，篩選顯示瓶頸是長度門檻而不是訊號粒度；AISHELL 是安靜環境，能量與機率在句間停頓上不會有差異，噪音下能量更不可靠。
- **結合 ASR 的 endpoint 規則**：目的是減少句中的半句暫定結果（不是降低延遲）。transducer 沒有句尾符號，sherpa-onnx 的 endpoint 規則也是「已解碼內容 + 尾端靜音長度」，靜音部分與 VAD 相同；要判斷語意上是否說完需要標點 / 語言模型，第一階段不做。

### 結論與待決定

- **pause 150 明顯優於 A2**：0.2 s 停頓的結果延遲 1.3 s → 0.23 s，所有句子快約 60 ms；代價是半句暫定結果的句子 36% → 40%、解碼量 +3.5–8%，CER 不變。pause 120 再快約 50 ms，但修正率與解碼量再多一點。
- 已可用 `--pause-ms 150`（`run_pipeline.py`、`run_eval.py`、`demo_mic.py`）或 `VadConfig(min_silence_ms=1000, pause_ms=150)` 使用。**是否改預設由使用者決定**；改之前建議在閒置時重量 RTF。

---

## 2026-11-01　決定：A2 的 pause 預設 200 → 150（B9）

合併 `feat/b9-short-pause`（fast-forward 到 `d57e283`）後，把 `PipelineConfig()` 的 `pause_ms` 由 200 改為 150；`run_pipeline`、`run_eval`、`demo_mic` 都跟著改。`--pause-ms 200` 可退回。

- **理由**（上一段的 300 句連續集與對齊集）：0.2 s 句間停頓後的句子，結果延遲 1314 → 232 ms（P90 3688 → 285），其餘句子約快 55–70 ms；CER 與切段不變。
- **代價**：先出半句暫定結果的句子 36% → 40%，解碼量 +3.5–8%。
- pause 120 數字更好（186 ms），但只比 150 多 1 個百分點的修正率，且 150 的分析較完整；先採 150。

**200 句冒煙集驗證**（`x-asr-480ms-int8-ort`，jobs 1）

| pause | CER (%) | RTF | 首字 mean / P90 (ms) | 峰值 RSS (MB) | 暫定結果總數 / 有 2 個以上暫定結果的段 |
|---|---|---|---|---|---|
| 200 | 4.80 | 0.0318 | 1002 / 1029 | 353 | 316 / 78 |
| **150** | 4.80 | 0.0324 | 1006 / 1029 | 353 | 334 / 87 |

- 198 / 200 句最終文字相同。不同的 2 句：一句句尾「起抵」→「启迪」（錯字 +1），一句補回漏掉的「脸」（錯字 −1），CER 不變。
- 依「優化階段只跑小集合」，3000 句未重跑。

---

## 2026-11-01　X-ASR onnxruntime 後端調校：執行緒數、session 選項、批次 joiner（B8）

環境：Apple M5（4 效能核 + 6 節能核）、macOS 27.0.1、onnxruntime 1.31.1，`x-asr-480ms-int8-ort` + Silero v6.2.3，A2（pause 150 / min_silence 1000）。評測在 `feat/latency-b1`（`4a2d1d1`）+ 本次提交的程式上執行，全部 `--jobs 1`。

**量測條件的限制**：量測期間有其他工作階段的評測一直佔滿 1 個核心（load average 3–9），單執行緒 RTF 比閒置時的 0.032 高約 40–60%，絕對值不能當基準。比較時同一批交錯重跑（`bench_asr_compute.py --repeat 3` 取中位數），只看相對差異；同一設定重跑的差異約 ±5–7%。多執行緒受負載的影響比單執行緒大（要搶更多核心），因此**多執行緒的結論待閒置時重量**。

### 做了什麼

- `scripts/bench_asr_compute.py`：不經 pipeline，只量計算。每句以 10 ms chunk 送入再 finalize，記錄每次 encoder 推論、每個 chunk 的 greedy search（joiner + decoder）、finalize（= A2 停頓時產生暫定結果的計算）、一次送入整句的重新解碼（= A2 恢復說話時的計算）、Silero 每 frame 的耗時；footprint 沿用 `probe_footprint.py` 的 vmmap 量測。token 序列的雜湊用來比對結果是否改變。每個設定一個子程序，`--threads` × `--preset`（session 選項預設組）取笛卡兒積。
- `ZipformerOrtAsr` / `SileroVadModel` 多一個 `sess_options` 參數（只給量測用，預設行為不變）。
- **greedy search 批次送入 joiner**（`ZipformerOrtStream._greedy_search`），見下方。

### 計算時間拆解（單執行緒，60 句）

profile（cProfile）：時間 97% 在 `session.run`，Python 端（fbank 送入、stack、argmax）< 2%。以總時間計，encoder 約 94%（每個 480 ms chunk 約 12 ms，負載較輕時），joiner 約 2.8%（每個 chunk 12 次、每次約 0.03 ms），decoder 0.4%。encoder 圖有 12,261 個節點，其中 4073 個 Constant、數千個 Shape / Gather / Slice / Unsqueeze 等小運算，MatMulInteger 只有 344 個：**每個 chunk 只有 61 frame，時間多花在大量小運算上**，這也是多執行緒幫不上忙的原因。

### 1. 執行緒數 1 / 2 / 4

**全鏈路（`run_eval.py --threads 1 2 4`，負載下）**

| 集合 | threads | CER | RTF | 首字 mean / P90 (ms) | 說完 → 完整結果 mean / P90 (ms) | finalize mean / P90 (ms) | 恢復時重新解碼 mean / P90 (ms) | 峰值 RSS (MB) | 結果與 1 執行緒不同 |
|---|---|---|---|---|---|---|---|---|---|
| 300 句對齊集 | **1** | 3.28% | **0.0437** | 982 / 1033 | **248** / 318 | **63** / 76 | **104** / 179 | 358 | — |
| | 2 | 3.28% | 0.0529 | 985 / 1037 | 260 / 332 | 75 / 90 | 122 / 210 | 363 | 1 / 300 句 |
| | 4 | 3.35% | 0.0676 | 989 / 1044 | 275 / 344 | 90 / 106 | 146 / 260 | 362 | 4 / 300 句 |
| 200 句冒煙集 | **1** | 4.80% | **0.0510** | 1013 / 1037 | — | **72** / 89 | **128** / 210 | 364 | — |
| | 2 | 4.57% | 0.0579 | 1018 / 1039 | — | 81 / 96 | 139 / 222 | 354 | 8 / 200 句 |
| | 4 | 4.37% | 0.0696 | 1022 / 1046 | — | 92 / 108 | 156 / 255 | 363 | 11 / 200 句 |

（「說完 → 完整結果」以 MFA 真實末字結束點起算，`pause_analysis.py cost`，只有對齊集有。finalize = 暫定結果時刻 − 停頓事件時已送入的音訊終點。冒煙集這三列是批次 joiner 之前的程式，對齊集是之後的程式，同一批內可比。）

**純計算（`bench_asr_compute.py --threads 1 2 4 --preset default no-spin --repeat 3`，60 句）**

| threads | spinning | ASR RTF | encoder mean / P90 (ms) | finalize mean (ms) | 重新解碼 (ms / 音訊 s) | VAD (ms / frame) | 結果不同（/ 60 句） |
|---|---|---|---|---|---|---|---|
| 1 | 開（預設） | 0.0485 | 17.7 / 18.3 | 63 | 34.9 | 0.091 | — |
| 1 | 關 | 0.0450 | 16.4 / 16.9 | 59 | 32.8 | 0.093 | 0 |
| 2 | 開 | 0.0538 | 19.5 / 20.6 | 71 | 39.4 | 0.100 | 3 |
| 2 | 關 | 0.0460 | 16.7 / 17.4 | 60 | 33.2 | 0.097 | 3 |
| 4 | 開 | 0.0596 | 21.4 / 22.5 | 78 | 44.1 | 0.114 | 5 |
| 4 | 關 | 0.0499 | 18.1 / 18.7 | 65 | 35.9 | 0.106 | 5 |

- **多執行緒沒有加速，在負載下反而更慢**：2 / 4 執行緒的 RTF +21% / +55%（全鏈路），停頓時的 finalize +12 / +27 ms、恢復時的重新解碼 +18 / +42 ms，都直接加在結果延遲上（說完 → 完整結果 +12 / +27 ms）。關掉 spinning 後 2 執行緒與 1 執行緒相同（差異在雜訊內），4 執行緒仍慢約 10%。負載較輕時的初步量測（同 60 句、未重跑）：2 執行緒 encoder −7%、4 執行緒 +17%。原因見上方拆解：圖由大量小運算組成，執行緒同步的開銷蓋過平行化的收益。
- **多執行緒會改變辨識結果**：純 ASR 上 2 / 4 執行緒有 3 / 5（/ 60）句的 token 不同，全鏈路 200 句 8 / 11 句、300 句 1 / 4 句的最終文字不同。CER 的變化在 200 句信賴區間（約 ±1 個百分點）內，不能當成多執行緒變準（300 句 3.28 / 3.28 / 3.35%）。原因推測是運算拆給多個執行緒後浮點累加順序改變（未深究）。所以**單執行緒的結果才是可重現的基準**。
- 峰值 RSS 與 footprint 不隨執行緒數變化（±5 MB）。首字延遲幾乎不變（+3–9 ms）：首字主要卡在模型輸出延遲（B1），計算只佔一小部分。
- **決定：預設維持單執行緒**。VAD 每 frame 約 0.09–0.11 ms（只佔 RTF 約 0.3%），執行緒數一樣沒有幫助。

### 2. ORT session 選項（`bench_asr_compute.py --threads 1 --preset ... --repeat 3`，60 句）

| 選項（ASR 三個 session 與 VAD 都套用） | ASR RTF | encoder mean (ms) | search mean (ms / chunk) | finalize mean (ms) | footprint 載入後 / 峰值 (MB) | 峰值 RSS (MB) | 結果不同 |
|---|---|---|---|---|---|---|---|
| 預設（ORT_ENABLE_ALL、arena 開、prepacking 開） | 0.0457 | 16.7 | 0.29 | 60 | 289 / 329 | 366 | — |
| graph_optimization_level = EXTENDED | 0.0483 | 17.6 | 0.32 | 63 | 284 / 320 | 352 | 0 |
| graph_optimization_level = BASIC | 0.0609（+33%） | 21.7 | 0.98 | 80 | 281 / 317 | 355 | **2** |
| enable_cpu_mem_arena = False | 0.0460 | 16.8 | 0.29 | 60 | 288 / 324 | 361 | 0 |
| enable_mem_pattern = False | 0.0471 | 17.2 | 0.29 | 62 | 284 / 320 | 358 | 0 |
| session.disable_prepacking = 1 | 0.0951（+108%） | 33.5 | 2.30 | 126 | 283 / 321 | 359 | **2** |
| session.set_denormal_as_zero = 1 | 0.0464 | 16.9 | 0.29 | 61 | 290 / 328 | 365 | 0 |

- **沒有任何選項同時帶來明顯的速度或記憶體收益**：EXTENDED / 關 arena / 關 mem pattern / denormal 的 RTF 在 ±5% 雜訊內，footprint 差 ≤ 9 MB、RSS 差 ≤ 14 MB（單次量測，未重複確認）。B11 經 sherpa-onnx 時關 arena 也無效，這裡結論相同。
- **BASIC 與關 prepacking 明顯變慢且改變結果**：兩者都讓 joiner / encoder 的 `DynamicQuantizeLinear + MatMulInteger` 不被融合成 `DynamicQuantizeMatMul`（prepacking 是該融合 kernel 的前提），量化方式改變（下一節），結果不同。
- 決定：**session 選項維持預設**，只記錄。

### 3. greedy search 批次送入 joiner（採用）

原本每個 encoder frame 呼叫一次 joiner（每 chunk 12 次）。改為：把 chunk 中尚未解碼的 frame 一次送入 joiner（decoder_out 相同），取第一個輸出 token 的 frame，更新 decoder 後從下一個 frame 再批次送入。決策順序與逐 frame 完全相同，呼叫次數降為「1 + 這個 chunk 輸出的 token 數」（平均約 2.2 次）。

**前提是 joiner 逐列獨立**。joiner 是 `Add → Tanh → DynamicQuantizeLinear → MatMulInteger`，依 ONNX 定義，DynamicQuantizeLinear 對整個 tensor 只算一個 scale，批次送入時各列的量化會互相影響。實測：
- 關閉圖最佳化時，批次與逐列的 logit 最多差 0.156（tanh 輸出各列的範圍不同，例如 max 0.979–0.994）。
- ORT_ENABLE_ALL / EXTENDED（預設）下，ORT 把它融合成 `DynamicQuantizeMatMul`，arm64 上實際是逐列量化：200 句全部 32,028 個 frame（以初始 decoder_out 批次送入每個 chunk）的批次 logit 與逐列**逐位元相同**。
- 因此 `ZipformerOrtAsr` 載入時以 8 列不同尺度的隨機輸入自我檢查（`_joiner_is_row_independent`），不成立（例如別的平台或 ORT 版本）就自動退回逐 frame。

**驗證**：200 句冒煙集與 300 句對齊集全鏈路，與改動前逐句相同：hyp 200 / 200、300 / 300 相同；句段 205 / 205、307 / 307 的文字、VAD 起訖、首字在音訊中的位置、partial 次數全部相同；暫定結果的文字也全部相同。新增測試 `test_x_asr_ort_batched_joiner_matches_per_frame`（6 個測試音訊、blank_penalty 1.5，文字與 token 時間相同）。

| 改動前 → 後（負載下，同批） | search (ms / chunk) | ASR RTF（純計算） | 全鏈路 RTF 200 / 300 句 | finalize mean (ms) 300 句 | 恢復時重新解碼 mean (ms) 300 句 | 說完 → 完整結果 mean / P90 (ms) |
|---|---|---|---|---|---|---|
| 逐 frame | 0.67 | 0.0504 | 0.0510 / 0.0520 | 75 | 122 | 260 / 325 |
| **批次** | **0.29** | **0.0457** | **0.0493 / 0.0487** | **70** | **114** | **255 / 321** |

- greedy search 的時間減少 57%，整體約 −3–6%（逐 frame 呼叫的開銷原本就只佔約 3%，量到的整體差異有一部分是雜訊）。這大致抵銷了 ort 後端相對 sherpa-onnx 多出的約 3% RTF。
- 結果延遲只快約 5 ms：finalize 與重新解碼中 encoder 仍佔 90% 以上。

### 結論

- 這個模型在 CPU 上已經沒有可調的 session 參數：時間幾乎全在 encoder 大量的小運算上，多執行緒與各種 session 選項都沒有收益，而多執行緒、BASIC、關 prepacking 還會改變結果。**預設維持：單執行緒、ORT_ENABLE_ALL、其餘選項為 ORT 預設**。
- 採用批次 joiner：結果逐句不變，greedy search 時間減半，RTF 與延遲小幅改善。
- 想再降 encoder 的計算，要從模型本身著手（Core ML / ANE、或把小運算在匯出時常數摺疊），屬於第二階段。
- 待辦：閒置時重量執行緒 1 / 2 / 4 的 RTF（本次在負載下量測）；依「優化階段只跑小集合」，3000 句未重跑。

---

## 2026-11-01　未合併分支的實驗紀錄（保留索引）

以下實驗做完但沒有合併進 `feat/latency-b1`（只合併有顯著提升或確定採用的）。完整紀錄（progress / scoreboard / models.md 的段落與程式）留在各分支上，**分支不要刪**；原始結果 JSON 已集中複製到主目錄（`eval/results/` 不進版控，只存在本機）：

- 結果：`eval/results/archive/<分支名，/ 換成 ->/`（a2-default、b4-noise、b9-short-pause、light-ctc、ort-tuning、pseudo-stream；b7、b11 沒有輸出檔）。主目錄 `eval/results/` 本身是合併後各次評測的結果。
- 產生的資料集（同樣只在本機）：`data/aishell1_continuous_300_tcar_snr0`、`data/aishell1_continuous_300_tcar_snr-5`（B4）、`data/aishell1_align_300_short3`（偽串流短句子集，MFA 語音長度 ≤ 3 s 的 100 句）。

| 分支 / commit | 內容 | 結論（小集合，200 / 300 句） |
|---|---|---|
| `feat/light-ctc` / 3740c3a | 串流 zipformer small CTC INT8（27 MB）的 onnxruntime 後端 `small-ctc-zh-int8-ort` | CER 約 2 倍（200 句 9.34 vs 4.80%、300 句 7.00 vs 3.28%，皆顯著）；首字快約 300 ms（P50 706 vs 1036 ms）、峰值 RSS 約 183 vs 359–448 MB。不當預設，輕量備選 |
| `feat/pseudo-stream` / 161140b、b6583c5、d331ad9（**2026-11-02 已合併**，並設為預設） | SenseVoice 偽串流降計算量（`PartialSchedule`）、≤ 3 s 短句子集、授權查證 | 建議設定 `sensevoice-small-int8-ort-p480-960-max3`：可見首字 429–449 vs X-ASR 970–1086 ms，RTF 同量級，CER 持平；說完 → 完整結果慢 18–128 ms、RSS 多約 60–170 MB；商用依據只有 issue #287 的維護者回覆。保留為選項，預設不變（100 句 ×2、25 條，負載下量測） |
| `feat/b4-noise` / d4f00da | DEMAND TCAR 車內噪音連續集（20 / 10 / 5 / 0 / −5 dB）與 VAD 誤觸發指標 | 誤觸發全為 0、CER 3.30–3.53% 與乾淨集無顯著差；TCAR 能量 98.7% 在 100 Hz 以下，寬頻 SNR 高估干擾，SNR 定義待決定 |
| `feat/b11-eval-rss` / 21048e7 | 評測峰值 RSS 隨句數上升：改成以最長 20 句暖機 | 增長是推論暖機（malloc 碎片、程式碼分頁），不是洩漏；修正評測口徑，不影響 pipeline |
| `feat/b7-no-fp32` / 0763a19、be2df9c、0d135e2 | 取消 FP32 參照、第二階段改為官方 INT8 + Core ML、X-ASR 參數量 | 純文件；X-ASR 153.4M 參數（97% INT8）、zh-14M 13.8M |

---

## 2026-11-01　SenseVoice 熱詞：CTC prefix beam search + 熱詞加分（只改解碼）

環境：Apple M5 / macOS 27.0.1 / Python 3.11 / onnxruntime 1.31.1，單執行緒，CPU。分支 `feat/sv-hotword`，程式 `51f45f6`、`2de7e6e`（結果在此 commit 上產生）。同機可能有其他工作，RTF / 解碼耗時只看同批相對差。依「優化階段只跑小集合」，只用 300 句對齊集與 200 句冒煙集。

### 做了什麼

- `src/asr/ctc_decode.py`：
  - `ctc_greedy`（原本 `_recognize_audio` 內的迴圈搬出來，程式相同）。
  - `ctc_prefix_beam_search(log_probs, beam, graph, top_k=10, blank_skip=0.999)`：標準 CTC prefix beam search（每個前綴記以 blank / 非 blank 結尾的機率）。每個 frame 只展開機率前 `top_k` 的非 blank token；blank 機率 ≥ 0.999 的 frame 只做「接 blank / 重複最後一個 token」。逐 frame 做 log-softmax，不整段先算（整段算要多兩份 (T, 25055) 暫存，長句峰值 RSS 多約 35 MB；逐 frame 後與 greedy 相同）。
  - `HotwordGraph`：熱詞的 Aho-Corasick 自動機。前綴每多拼對熱詞的一個 token 加 `score`（暫時加分），拼到一半失敗時扣回，拼完時轉為永久（共 `長度 × score`），結束時扣掉沒拼完的部分。排序用「CTC log 機率 + 熱詞加分」。
  - `tokenize_hotword`：熱詞依 `tokens.txt` 最長匹配切 token（中文一字一 token；英文詞前加 `▁`）。對不上的詞略過並列在 `decoder.skipped_hotwords`。
- `src/asr/sense_voice.py`：`SenseVoiceConfig` 加 `beam` / `hotwords` / `hotword_score`；解碼抽成 `SenseVoiceDecoder`（`SenseVoiceOrtAsr.decoder`、`make_decoder()`）。**beam 1 且沒有熱詞時走原本的 greedy**（丟前 4 個 token）；否則略過前 4 個 frame（語言 / 情緒 / 事件 / ITN 的查詢位置；300 句對齊集上這 4 個 frame 的 argmax 一律是這 4 個特殊 token，0 / 300 例外）做 beam search。偽串流變體（`-p480`）用同一個解碼器。sherpa-onnx 版 SenseVoice 與 transducer 模型給 beam / 熱詞時報錯。
- `src/asr/__init__.py`：`DecodeOptions(blank_penalty, beam, hotwords, hotword_score)`，`load_asr(..., decode=DecodeOptions(...))`；原本的 `blank_penalty=` 參數保留。
- CLI：`run_eval.py` / `run_pipeline.py` 加 `--hotwords FILE --hotword-score X --beam N`（`run_eval` 的 `--hotword-score` / `--beam` 可給多個值掃描；結果名稱加 `_beam<N>` / `_hw-<檔名><分數>`）。熱詞檔一行一個詞，`#` 後為註解。
- `scripts/hotword_eval.py`：每句整段跑一次模型（不經 VAD），同一份 logits 依序用各設定解碼，量召回率、熱詞句 / 非熱詞句 CER（bootstrap CI + 對 greedy 的成對差）、誤觸發、每句解碼耗時。`eval.metrics.paired_bootstrap_cer_diff` 從 `feat/pseudo-stream` 原樣複製（兩邊合併時相同，不會衝突出不同版本）。
- 測試 `tests/test_ctc_decode.py`（12 個測試函式）：加分帳（拼完 / 拼一半 / 失敗 / 長短熱詞巢狀 / fail 轉移）、近乎平手時熱詞翻盤且加分 0 不翻、拼完後不刪後面的字、blank 跳過、未正規化 logits、峰值機率下 beam search = greedy；模型測試：預設解碼與改動前的 greedy 程式逐字相同（舊程式照抄在測試裡）、`DecodeOptions` 傳到 `-p480` 變體、其他後端拒絕。pytest 156 個、mypy、ruff 通過。

### 熱詞清單（`eval/hotwords/`）

- `aishell1_align_300_targets.txt`：目標熱詞 57 個，逐句讀 300 句對齊集的參考文本人工挑的人名 26、地名 12、機構 / 品牌 / 媒體 19，不挑本身就是一般詞的（「苹果」「北京」）。出現 64 次、49 句（熱詞句）。**greedy 認錯 21 個詞**（朱圣祎、李双江、马塞洛特、马震、佟健、蔡兴桥、洪永城、郑嘉颖、蓝莹、莫利托、付敬懿、吴显亭、韩榆、贵港、明孝陵、法政大学、海信科龙、威灵电器、美芝、万龙滑雪场，中新网 6 次錯 1 次）。
- `distractors.txt`：干擾熱詞 50 個（車艙指令「打开空调」「座椅加热」、人名、地名、車廠 / 品牌），腳本檢查不出現在兩個集合的任何參考文本中（原本的「乌鲁木齐」出現在冒煙集，換成「拉萨」）。
- 解碼時的熱詞清單 = 目標 + 干擾，共 107 個。

**這是樂觀的設定**：目標熱詞是從同一批句子挑出來的，等於事先知道會出現哪些專有名詞（車艙場景的通訊錄、常用地點清單才接近這個情況）。召回率與熱詞句 CER 的改善只說明「詞在清單裡時能救回多少」；清單對無關句子的副作用看非熱詞句與冒煙集。

### 結果：300 句對齊集（`uv run scripts/hotword_eval.py --score 0 1 1.5 2 3 4`）

熱詞句 49 句 / 753 字、非熱詞句 251 句 / 3548 字。Δ 是相對 greedy（beam 1、加分 0）的成對差（百分點，95% CI）。誤觸發 = hyp 中某熱詞的次數超過 ref 的部分（目標 / 干擾）。模型（特徵 + encoder）RTF 0.0269。

| beam | score | 召回 % | 熱詞句 CER % [CI] | Δ [CI] | 非熱詞句 CER % | Δ [CI] | 非熱詞句改變 | 誤觸發 | 解碼 ms mean / P90 | 解碼 RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0（greedy） | 67.2 | 6.24 [3.94, 8.95] | — | 1.97 | — | — | 0 / 0 | 0.7 / 1.0 | 0.0001 |
| 1 | 1.5 | 89.1 | 3.98 [1.80, 6.70] | −2.26 [−3.45, −1.23] | 2.03 | +0.06 [−0.06, +0.17] | 5 | 0 / 0 | 5.0 / 7.2 | 0.0010 |
| 1 | 3 | 92.2 | 3.45 [1.69, 5.45] | −2.79 [−4.35, −1.36] | 2.23 | **+0.25 [+0.08, +0.45]** | 14 | 0 / 0 | 4.9 / 7.2 | 0.0010 |
| 1 | 4 | 92.2 | 3.85 [2.01, 5.91] | −2.39 [−4.02, −0.93] | 2.71 | **+0.73 [+0.39, +1.13]** | 23 | 0 / 0 | 4.9 / 7.2 | 0.0010 |
| 4 | 0 | 67.2 | 6.24 | +0.00 | 1.97 | +0.00 | 0 | 0 / 0 | 5.5 / 8.1 | 0.0011 |
| 4 | 1 | 85.9 | 4.52 [2.33, 7.18] | −1.73 [−2.74, −0.88] | 1.94 | −0.03 [−0.09, +0.00] | 1 | 0 / 0 | 5.6 / 8.1 | 0.0011 |
| 4 | 1.5 / 2 | 90.6 | 3.98 [1.85, 6.68] | **−2.26 [−3.36, −1.29]** | 1.94 | −0.03 [−0.09, +0.00] | 1 | 0 / 0 | 5.5 / 8.1 | 0.0011 |
| 4 | 3 | 92.2 | 3.59 [1.83, 5.57] | −2.66 [−3.97, −1.53] | 1.97 | +0.00 [−0.08, +0.08] | 2 | 0 / 1 | 5.5 / 8.1 | 0.0011 |
| 4 | 4 | 93.8 | 3.32 [1.61, 5.29] | −2.92 [−4.29, −1.73] | 2.00 | +0.03 [−0.06, +0.14] | 3 | 0 / 1 | 5.5 / 8.1 | 0.0011 |
| 8 | 0 | 67.2 | 6.24 | +0.00 | 1.97 | +0.00 | 0 | 0 / 0 | 6.2 / 9.1 | 0.0013 |
| 8 | 1 | 85.9 | 4.52 [2.33, 7.18] | −1.73 [−2.74, −0.88] | 1.97 | +0.00 | 0 | 0 / 0 | 6.4 / 9.3 | 0.0013 |
| 8 | 1.5 / 2 | 90.6 | 3.98 [1.85, 6.68] | **−2.26 [−3.36, −1.29]** | 1.97 | **+0.00（0 句改變）** | 0 | 0 / 0 | 6.3 / 9.3 | 0.0013 |
| 8 | 3 | 92.2 | 3.59 [1.83, 5.57] | −2.66 [−3.97, −1.53] | 2.00 | +0.03 [+0.00, +0.09] | 1 | 0 / 1 | 6.3 / 9.3 | 0.0013 |
| 8 | 4 | 93.8 | 3.32 [1.61, 5.29] | −2.92 [−4.29, −1.73] | 2.00 | +0.03 [+0.00, +0.09] | 1 | 0 / 1 | 6.3 / 9.3 | 0.0013 |

（beam 1 的 score 1 / 2 略；beam 4 / 8 的 score 1.5 與 2 結果相同，合併成一列。完整表與逐句結果在 `eval/results/2026-11-01_aishell1_align_300_hotword.json`，不進版控。）

- 不帶熱詞的 beam search（score 0）與 greedy **逐句相同**（beam 4 / 8 都是 0 / 300 句不同）。
- beam 8、score 1.5–2：21 個認錯的詞救回 15 個（剩朱圣祎、佟健、付敬懿、贵港、明孝陵、美芝；多半是正確字不在該 frame 的前 10 名，例如「佟」→「董」），召回 67.2 → 90.6%，熱詞句 CER 6.24 → 3.98%（−2.26 個百分點，CI 不含 0，**顯著**），非熱詞句 0 句改變、0 誤觸發。
- score ≥ 3 召回再多 1–2 個詞，但開始出現誤觸發（干擾詞「清华大学」把「清华法学院」改成「清华大学院」）。
- **beam 1 加熱詞不可用**：拼到熱詞前一兩個字就有暫時加分，寬度 1 時這一步就被選走，之後扣回也救不了（「港媒」→「港美」被「美芝」的「美」拉走、「最好」→「最后」被「后视镜折叠」拉走），非熱詞句 CER 隨加分顯著變差。

### 結果：200 句冒煙集（同一份熱詞清單，`--manifest data/aishell1_test_200/manifest.jsonl`）

冒煙集與對齊集有 11 句重疊，其中 8 句含目標熱詞（出現 8 次，greedy 漏 3 次：佟健 ×2、朴泰桓），其餘 192 句對這份清單而言是無關句子，用來看副作用。greedy 非熱詞句 CER 3.50%。

| beam | score | 召回 % | 非熱詞句 CER % | Δ [CI] | 非熱詞句改變 | 誤觸發 |
|---|---|---|---|---|---|---|
| 1 | 1.5 / 2 / 4 | 62.5 | 3.68 / 3.71 / 4.17 | +0.18 / +0.21 / **+0.67 [+0.35, +1.02]** | 6 / 7 / 29 | 0 |
| 4 | 1.5–3 | 62.5 | 3.54 | +0.04 [+0.00, +0.11] | 1（「兴安」→「新安」，被「新京报」拉走） | 0 |
| 4 | 4 | 62.5 | 3.61 | +0.11 [+0.00, +0.24] | 4 | 0 |
| **8** | **1–4** | 62.5 | **3.50** | **+0.00** | **0** | 0 |

召回沒有提升：佟健、朴泰桓的正確字不在前 10 名（beam search 選不到）。

### 結果：全鏈路（`run_eval.py`，A2 預設斷句，300 句對齊集，`--jobs 3`）

熱詞檔 = `cat eval/hotwords/aishell1_align_300_targets.txt eval/hotwords/distractors.txt`。

| 設定 | CER % [CI] | 相對 greedy [CI] | 熱詞句 / 非熱詞句 CER % | 改變句數（非熱詞句） | RTF | 峰值 RSS (MB) |
|---|---|---|---|---|---|---|
| `sensevoice-small-int8-ort`（greedy） | 2.56 [2.01, 3.15] | — | 6.24 / 1.78 | — | 0.0283 | 441 |
| beam 4、score 2 | 2.16 [1.68, 2.68] | **−0.40 [−0.60, −0.21]** | 3.98 / 1.78 | 16（0） | 0.0317 | 445 |
| beam 8、score 2 | 2.16 [1.68, 2.68] | **−0.40 [−0.60, −0.21]** | 3.98 / 1.78 | 16（0） | 0.0292 | 446 |

首字 / 尾字延遲三者相同（非串流，首字等 finalize；beam search 每句多約 5 ms，在 10 ms chunk 的量測格內）。RTF 是 3 個子程序並行、負載下的值，差異以上表的解碼 RTF（+0.0012）為準。

### 結論

- **建議參數：beam 8、hotword_score 1.5–2**（預設 2 以下）。在這兩個小集合上：熱詞句 CER −2.26 個百分點（顯著），全鏈路 300 句整體 CER 2.56 → 2.16%（顯著），非熱詞句與無關句子 0 句改變、0 誤觸發。beam 4 幾乎一樣好，但冒煙集上有 1 句被熱詞的首字拉錯。
- **解碼開銷小**：每句 +5.6 ms（greedy 0.7 → 6.3 ms，P90 9.3 ms），解碼 RTF +0.0012，約是模型 RTF（0.027）的 4.5%；峰值 RSS +4 MB。偽串流（`-p480`）每次 partial 都重新解碼，開銷同比例增加。
- 預設不變：`SenseVoiceConfig` 預設 beam 1、無熱詞 = greedy，結果與改動前逐句相同；預設模型仍是 `x-asr-480ms-int8-ort`。

### 限制

- 目標熱詞從評測句子裡挑出（見上），召回率是「清單裡有」時的上限，不代表一般情況；只有 64 + 8 個熱詞出現，CI 寬。
- 救不回正確字不在該 frame 前 10 名的詞（佟健、朴泰桓、付敬懿等）：只改解碼的上限。調大 `top_k` 未掃描。
- 熱詞以字為單位、無發音資訊：同音異字（「洪永成」→「洪永城」）能救，聲學差很多的救不了；同一字串在無關語境也會被加分（「清华大学院」），所以 score 不宜高。
- 較短的熱詞恰好落在較長熱詞中間時不另計分（見 `ctc_decode.py` 說明）。
- 未量 3000 句；未在車艙噪音集上量；未做 sherpa-onnx 版 SenseVoice 的熱詞（sherpa-onnx 的離線 CTC 也沒有熱詞）。

### 與 icefall / sherpa-onnx ContextGraph 的差異（實作中發現）

第一版照 icefall strict mode：拼完熱詞時另外加上完整加分，離開節點時才扣回暫時加分。實測拼完到離開之間該前綴多了一倍加分，接任何字都要先被扣分，不出字的前綴暫時領先，**熱詞後面的字會被刪掉**（「安丰公司的业务」→「安丰公司业务」、「云投集团等发债企业」→「云投集团」，beam 1 / 4 時；beam 8 也偶發）。改成拼完即把暫時加分轉為永久、暫時部分歸零，這類刪字消失（有測試 `test_hotword_does_not_delete_following_token`）。

### 移植到 X-ASR（transducer）的初步評估（未實作）

- 做法：greedy search 換成 modified beam search（每個 encoder frame 每個假設最多出 1 個 token，假設數 = beam），每個假設帶同一個 `HotwordGraph` 的狀態與加分；`HotwordGraph` / `tokenize_hotword` 可直接重用。decoder 是無狀態的 2-token context，假設擴展時對新 context 批次呼叫 decoder 即可；joiner 已有批次送入（B8），beam 4 時每 frame 一次送 4 列。
- 詞表：X-ASR 是 5000 個 BPE（中文為「▁字」），**目標熱詞中 2 / 57 個含詞表外的字**（祎、邳），這些字根本輸出不了，熱詞救不回；SenseVoice 詞表 25055 沒有這個問題。英文熱詞要用 `bpe.model` 切分，需要 sentencepiece（目前不在依賴中，要下載）。
- 串流特有：partial 會隨熱詞前綴的暫時加分跳動（拼一半時先顯示熱詞、之後被扣回而改掉），需要決定 partial 用「目前最佳」還是「扣掉暫時加分後的最佳」。
- 開銷估計：目前 greedy search 0.29 ms / chunk、佔 ASR RTF 不到 3%（B8）；beam 4 的 joiner 列數 ×4、多了 decoder 呼叫與 Python 端假設管理，粗估 search 變 4–8 倍、整體 RTF +5–15%。finalize / 恢復時重新解碼同比例變慢，會直接加在結果延遲上（目前 finalize 約 63 ms，估計 +5–20 ms）。
- 工作量估計約 150–250 行程式 + 測試；建議先在 200 / 300 句上照本節的方法驗證。sherpa-onnx 的 transducer 原生支援 hotwords（modified_beam_search），可先用 sherpa-onnx 後端快速量一次效果再決定是否移植到 ORT 後端。

## 2026-11-02　B0：麥克風路徑的真實延遲 vs 模擬即時

環境：Apple M5（Mac17,3，10 核）/ macOS 27.0.1 / onnxruntime 1.31.1，單執行緒。模型 `x-asr-480ms-int8-ort`，A2（pause 150 / min_silence 1000，provisional finalize），pre-roll 100 ms。量測程式在 `feat/b0-mic-latency`（基於 `46fe454`，量測時腳本尚未提交）。**負載**：量測期間另有 2–3 個其他工作階段的 python 程序各佔滿約一顆核心（Core ML、穩定度、偽串流實驗），load average 3.3–5.7。

### 做了什麼

- `scripts/mic_latency.py`：同一段參考音訊跑三條路徑，逐句成對比較。
  - **sim**：模擬即時（與 `eval/run_eval.py` 相同），依 block 切塊，不 sleep。
  - **thread**：不出聲、不碰麥克風。另一條執行緒照真實時鐘每個 block 時長送一次 block 到 queue，主執行緒的「queue 取出 → `accept_waveform` → 處理事件」與 `demo_mic.py` 相同；事件時刻用 `perf_counter()`。
  - **portaudio**：真的開啟輸入裝置（sounddevice / PortAudio，`latency="low"`），callback 節奏、緩衝與時間戳都是真的，但**丟棄麥克風收到的聲音**（不儲存），改送檔案音訊。延遲起點用 PortAudio 的 `inputBufferAdcTime`（block 第一個樣本的擷取時刻）推得的 t0。**需使用者同意才執行**：預設 `--modes sim thread`，要跑 portaudio 必須明確給 `--modes ... portaudio --allow-mic`。
- `eval/realtime.py`：時間換算與逐句指標（長串流排列、ADC 時刻 → t0、PortAudio 時鐘 → perf_counter 偏移、demo_mic 的 t0 估計、句段歸屬、四個延遲指標），`tests/test_realtime.py` 8 個測試。
- 參考音訊：MFA 對齊集（`data/aishell1_align_300`）前 **30 句**，每句前補 1000 ms、後補 1500 ms 靜音，接成一條 209 s 的長串流。指標：首字（VAD 起點 / MFA 真實起點）、說完 → 完整結果（MFA 末字結束 → 涵蓋整句的暫定或最終結果，同 `pause_analysis.py cost`）、說完 → 定稿。
- block 大小 160 / 320 / 512 / 1024 樣本（10 / 20 / 32 / 64 ms；160 = demo_mic 預設）。

```bash
uv run scripts/mic_latency.py --n 30                         # sim + thread
uv run scripts/mic_latency.py --n 30 --modes sim thread portaudio --allow-mic   # 需使用者同意
```

結果：`eval/results/2026-11-02_mic-latency_n30.json`、`..._n30_rep.json`（block 160 重跑一次）。portaudio 路徑的 4 種 block 與一次重跑都在使用者要求停止使用麥克風**之前**已跑完；之後不再開啟任何輸入裝置。

### 結果

延遲 mean / P50 / P90（ms），30 句：

| 路徑 | block | 首字（VAD 起點） | 首字（真實起點） | 說完 → 完整結果 | 說完 → 定稿 |
|---|---|---|---|---|---|
| sim | 160 | 1011 / 1026 / 1031 | 1036 / 1044 / 1107 | 229 / 226 / 272 | 1042 / 1037 / 1093 |
| thread | 160 | 1017 / 1032 / 1041 | 1042 / 1049 / 1112 | 244 / 241 / 296 | 1044 / 1039 / 1095 |
| portaudio | 160 | 1020 / 1035 / 1046 | 1046 / 1056 / 1124 | 250 / 252 / 290 | 1046 / 1041 / 1098 |
| sim（重跑） | 160 | 1015 / 1032 / 1035 | 1041 / 1048 / 1111 | 244 / 239 / 290 | 1042 / 1037 / 1093 |
| thread（重跑） | 160 | 1016 / 1031 / 1038 | 1041 / 1047 / 1111 | 242 / 238 / 294 | 1044 / 1040 / 1096 |
| portaudio（重跑） | 160 | 1015 / 1030 / 1037 | 1040 / 1048 / 1114 | 230 / 227 / 277 | 1046 / 1041 / 1097 |
| sim | 320 | 1021 / 1037 / 1045 | 1046 / 1056 / 1117 | 248 / 245 / 294 | 1047 / 1042 / 1095 |
| thread | 320 | 1022 / 1038 / 1045 | 1047 / 1056 / 1118 | 244 / 242 / 291 | 1051 / 1047 / 1099 |
| portaudio | 320 | 1025 / 1042 / 1048 | 1051 / 1061 / 1118 | 244 / 241 / 289 | 1055 / 1051 / 1101 |
| sim | 512 | 1024 / 1040 / 1042 | 1049 / 1058 / 1120 | 235 / 232 / 278 | 1038 / 1035 / 1087 |
| thread | 512 | 1029 / 1045 / 1050 | 1055 / 1064 / 1124 | 244 / 244 / 288 | 1043 / 1040 / 1090 |
| portaudio | 512 | 1037 / 1052 / 1056 | 1062 / 1068 / 1130 | 252 / 248 / 299 | 1050 / 1047 / 1097 |
| sim | 1024 | 1040 / 1045 / 1076 | 1065 / 1081 / 1123 | 267 / 271 / 315 | 1053 / 1056 / 1100 |
| thread | 1024 | 1042 / 1048 / 1081 | 1067 / 1085 / 1120 | 265 / 266 / 313 | 1057 / 1059 / 1105 |
| portaudio | 1024 | 1057 / 1063 / 1095 | 1083 / 1099 / 1141 | 274 / 278 / 316 | 1076 / 1078 / 1123 |

所有路徑的辨識結果逐句相同（0 句不同）。

差距拆解（真實路徑，ms，mean / P50 / P90）。「交付延遲」= block 最後一個樣本擷取完 → 放進 queue；「牆鐘 − 內部模擬」= 同一次執行中事件的牆鐘時刻 − pipeline 自己的模擬時刻（只含交付 + queue 的落後）；「內部模擬 − sim」= 同一份計算在即時節奏下與 sim 的差（計算時間波動）：

| 路徑 | block | 交付延遲 | queue 等待 | 首字 牆鐘 − 內部模擬 | 首字 內部模擬 − sim | 完整結果 內部模擬 − sim | 首字 真實 − sim（成對） |
|---|---|---|---|---|---|---|---|
| thread | 160 | 2 / 2 / 2 | 0 / 0 / 0 | 2 / 2 / 2 | 4 / 3 / 9 | 13 / 11 / 31 | 6 / 5 / 11 |
| portaudio | 160 | 3 / 4 / 5 | 1 / 0 / 0 | 4 / 4 / 5 | 6 / 5 / 13 | 18 / 18 / 36 | 9 / 8 / 18 |
| thread（重跑） | 160 | 2 / 2 / 2 | 0 / 0 / 0 | 2 / 2 / 2 | −2 / −2 / 1 | −4 / −6 / 13 | 0 / 0 / 3 |
| portaudio（重跑） | 160 | 3 / 4 / 5 | 0 / 0 / 0 | 3 / 4 / 5 | −4 / −5 / −2 | −17 / −16 / −12 | −1 / −1 / 1 |
| thread | 320 | 3 / 4 / 5 | 0 / 0 / 0 | 3 / 4 / 5 | −2 / −2 / −1 | −7 / −7 / −5 | 1 / 1 / 3 |
| portaudio | 320 | 7 / 7 / 9 | 0 / 0 / 0 | 7 / 7 / 9 | −3 / −3 / −2 | −11 / −10 / −9 | 4 / 4 / 5 |
| thread | 512 | 4 / 5 / 5 | 0 / 0 / 0 | 4 / 5 / 5 | 1 / 1 / 4 | 5 / 7 / 16 | 5 / 5 / 8 |
| portaudio | 512 | 11 / 11 / 13 | 0 / 0 / 0 | 12 / 12 / 13 | 1 / 2 / 5 | 5 / 6 / 15 | 13 / 12 / 15 |
| thread | 1024 | 4 / 5 / 5 | 0 / 0 / 0 | 4 / 5 / 5 | −2 / −2 / 0 | −5 / −3 / 0 | 2 / 2 / 5 |
| portaudio | 1024 | 22 / 22 / 24 | 0 / 0 / 0 | 22 / 22 / 24 | −5 / −5 / −4 | −15 / −16 / −11 | 17 / 18 / 19 |

PortAudio（MacBook Air 內建麥克風，16 kHz）：

| block | callback − ADC 時間戳 | 其中超過 block 時長的部分 | PortAudio 回報的輸入延遲 | demo_mic 的 t0 偏差 |
|---|---|---|---|---|
| 160 | 14 ms | 4 ms | 263 ms | −2.6 / −8.3 ms（兩次） |
| 320 | 27 ms | 7 ms | 273 ms | +5.6 ms |
| 512 | 43 ms | 11 ms | 285 ms | +9.7 ms |
| 1024 | 86 ms | 22 ms | 445 ms | +20.5 ms |

PortAudio 的串流時鐘（`currentTime` / `inputBufferAdcTime`）在 macOS 上與 `time.perf_counter()` 是同一個時鐘（callback 內兩者差 19–64 µs，std 約 5 µs）。callback 間隔非常穩定（std 約 25 µs），4 種 block 都沒有輸入溢位。

### 差距來源

1. **驅動 / PortAudio 的交付延遲（系統性）**：callback 比 block 最後一個樣本的 ADC 時間戳晚約 block 時長的 1/3（4 / 7 / 11 / 22 ms），與 block 大小成正比、幾乎沒有抖動。這是模擬路徑完全沒有的部分，也是唯一穩定的差距。block 160 時只有 4 ms。
2. **block 大小本身**（模擬也看得到）：block 越大，等待湊滿 block 的時間越長。sim 160 → 1024 首字 +29 ms、完整結果 +38 ms（含 run 間波動）；真實路徑再加上 1 的 22 ms。
3. **執行緒排程 + queue**：可忽略。thread 路徑的回放執行緒 sleep 喚醒誤差 2–5 ms；主迴圈處理每 block 平均 < 2 ms，queue 幾乎從不積壓（等待 P90 0 ms）。
4. **計算時間的 run 間波動（非系統性，最大的一項）**：「內部模擬 − sim」在 −17 到 +18 ms 間跳動、兩個方向都有，block 160 重跑時正負號互換。這是機器負載（其他工作階段佔滿 2–3 顆核心）與 CPU 頻率 / 核心排程造成的計算時間差，主要落在說完 → 完整結果（停頓時的 finalize 約 50–60 ms 的計算全部在這條路徑上）。沒有看到「即時節奏下閒置導致降頻、計算一致變慢」的現象。
5. **demo_mic 顯示的延遲偏小**：`demo_mic.py` 以 `min(callback 時刻 − 已收樣本時長)` 估 t0，假設 callback 在樣本擷取的瞬間被呼叫，因此少算交付延遲（block 320 / 512 / 1024 少算 6 / 10 / 21 ms）。block 160 時反而多算 3–8 ms：串流剛啟動時的前幾個 callback 時間戳與之後不一致，`min` 會挑到它們。差距都在 25 ms 以內，未修改 demo_mic。

### 結論

- **在 demo_mic 的預設 block（160 = 10 ms）下，真實麥克風軟體路徑與模擬即時的差距約 4 ms 的系統性交付延遲 + ±15 ms 的計算波動**：首字成對差 mean −1 ~ +9 ms、完整結果 −13 ~ +21 ms（兩次）。相對首字約 1 s、完整結果約 230–250 ms，**模擬即時的正式數字可以代表真實麥克風路徑（ADC 時間戳之後的部分）**，不需要修正。
- 大 block 會實際增加延遲（1024 樣本：首字 +46 ms、定稿 +34 ms，相對 sim 160），維持 10 ms block。
- 結果與 sim 逐句相同：真實時鐘、callback 切塊不影響辨識。

### 限制

- **不含聲學路徑與 ADC 時間戳之前的硬體延遲**：portaudio 路徑以 Core Audio 的 `inputBufferAdcTime` 為起點。PortAudio 自己回報的輸入延遲（263–445 ms）遠大於時間戳推得的 14–86 ms，兩者哪個反映真實聲音 → callback 的延遲，只能用聲學回送（喇叭 → 內建麥克風，用 `outputBufferDacTime` 或 click 對齊）或虛擬音訊裝置（BlackHole）驗證。這兩種都需使用者同意，未做；使用者已要求不再開啟麥克風。
- 只有 30 句、每種設定 1 次（block 160 兩次），在其他工作佔用 CPU 的負載下量測；計算波動的大小與方向會隨負載改變。
- thread / portaudio 路徑沒有終端輸出（demo_mic 會刷新 partial），終端寫入的耗時未計入（每次 < 1 ms 量級，未量）。
- 參考音訊是 AISHELL 單句加數位靜音，不是車艙的連續對話。

---

## 2026-11-02　partial 穩定度：量化畫面上的文字跳動（X-ASR vs SenseVoice 偽串流）

### 動機

之前只量 CER 與延遲，沒有量「使用者看到的字被改了多少次、改了多少」。兩件事要靠它判斷：A2 的暫定結果在說話恢復後會被取代（約 36–40% 的句子先出半句）；SenseVoice 偽串流每次 partial 都整段重新辨識，文字可能整句跳，這是它能否當預設的前提之一。

### 指標（`eval/stability.py`，定義見 [eval.md](eval.md)「partial 穩定度」）

- 結果 JSON 每個句段新增 `partials`（每次 partial 變化的時刻與文字）；`metrics.stability` 由父程序從 JSON 計算。
- 畫面文字序列 = partial、暫定結果、最終結果依時間合成（與偽串流比較的「可見首字」口徑一致），經 CER 的同一套正規化。
- **修正**：前一次顯示不是這次的前綴（已顯示的字被改或刪）；只追加不算。**修正率** = 修正次數 / 顯示更新次數。
- **被改掉的字數**：共同前綴口徑（共同前綴後的全部字，畫面上要重畫的字，上界）與 LCS 對齊口徑（真的被換掉或刪掉的字，下界），報告相對最終字數的比例。
- **前綴穩定延遲**：每個最終字從「第一次出現在該位置」到「之後不再變」的時間。
- 依被取代的是 partial 還是暫定結果分開統計。
- 測試 `tests/test_stability.py`：純追加 = 0 修正、改最後一字、整句重寫、刪字、標點 / 繁簡不算、暫定結果被取代、彙總。

### 環境

Apple M5（Mac17,3）、macOS 27.0.1、Python 3.11.15、單執行緒、`--jobs 2`（每批兩個設定並行）。X-ASR 與 `sensevoice-small-int8-ort-p480` 在 `285bb32`（本分支）上跑；`sensevoice-small-int8-ort-p480-960-max3` 在 `feat/pseudo-stream`（`d331ad9`，未合併）加上本分支的穩定度改動（暫時 cherry-pick、未提交，只在暫存 worktree）上跑，同一批另跑 p480 作對照：200 句結果與主分支逐項一致（修正率、字數、穩定延遲相同，延遲差 ≤ 10 ms）。LCS 對齊口徑在跑完後加入，由結果 JSON 重算（不需重跑模型）。機器上同時有其他工作，RTF 偏高，只看同批相對差。全部 A2 預設斷句（pause 150 / min_silence 1000 / finalize）。

### 結果

200 句冒煙集（`--smoke`）/ 300 句連續集（`data/aishell1_continuous_300`，77 條）。「每條」= 每個 wav（連續集一條 4 句）。

| 模型 | 集 | CER % | 每條顯示次數 | 修正率 | 有修正的條 | 每條被改的字 mean / P90 | 被改字 / 最終字（前綴 / 對齊） | 前綴穩定延遲 mean / P90 (ms) | 一出現就穩定的字 |
|---|---|---|---|---|---|---|---|---|---|
| X-ASR（預設） | 200 | 4.80 | 8.5 | **5.2%** | 31.5% | 0.43 / 1 | **2.9% / 2.9%** | **19 / 0** | **97.1%** |
| SenseVoice p480 | 200 | 3.40 | 8.3 | 29.5% | 84.5% | 6.83 / 20 | 45.5% / 17.6% | 315 / 989 | 74.1% |
| SenseVoice p480-960-max3 | 200 | 3.40 | 4.4 | 33.9% | 73.5% | 3.22 / 7 | 21.5% / 10.0% | 249 / 997 | 85.1% |
| X-ASR（預設） | 連續 | 3.30 | 34.0 | **8.5%** | 92.2% | 3.23 / 6 | **5.8% / 5.8%** | **41 / 0** | **94.2%** |
| SenseVoice p480 | 連續 | 2.60 | 30.4 | 20.9% | 98.7% | 25.51 / 67 | 45.7% / 12.8% | 521 / 1001 | 78.5% |
| SenseVoice p480-960-max3 | 連續 | 2.60 | 11.9 | 21.4% | 79.2% | 8.81 / 21 | 15.8% / 4.4% | 275 / 0 | 90.4% |

依被取代的文字來源（被取代次數 / 其中修正 / 被改的字）：

| 模型 | 集 | partial 被取代 | 暫定結果被取代 |
|---|---|---|---|
| X-ASR | 200 / 連續 | 1374 / **0** / 0 ・ 2129 / **0** / 0 | 123 / 78 / 87 ・ 333 / 209 / 249 |
| SenseVoice p480 | 200 / 連續 | 1435 / 422 / 1322 ・ 2159 / 443 / 1821 | 14 / 5 / 44 ・ 21 / 12 / 143 |
| SenseVoice max3 | 200 / 連續 | 583 / 208 / 510 ・ 467 / 127 / 301 | 92 / 21 / 134 ・ 291 / 35 / 377 |

**X-ASR 的修正全部來自 A2 暫定結果，而且幾乎都是「縮回再補回」**：串流 transducer 的 partial 本身只會追加（0 次修正）。暫定結果是 finalize（補零）的輸出，會把 encoder lookahead 裡的最後 1–2 個字先吐出來；說話恢復、重新解碼後的串流 partial 還沒走到那裡，畫面先少掉那 1–2 個字，下一個 chunk 再補回。修正的暫定結果中，下一次顯示是它的真前綴（縮回）的有 78 / 78（200 句）、208 / 209（連續），之後最終文字以原暫定結果開頭（補回）的有 76 / 205。平均每次修正約 1.1 字。

**SenseVoice 偽串流的跳動是本質性的**：每次整段重新辨識，句中的字也會換（「背后意义静止」→「背后意义竟只是」），共同前綴口徑下 45% 的最終字被重畫過、LCS 口徑 13–18% 的字真的被換過；字的穩定延遲 P90 約 1 s（約兩次 partial 間隔）。另外第一個 0.48 s 的 partial 常是英文 / 數字碎片（「low」「jo」）：200 句 65 / 205 句段（32%）、連續集 23 / 159（14%）。max3（每 0.96 s、累積 > 3 s 停止）的 partial 次數少一半，被改的字也約少一半，但這部分是「3 s 後畫面凍結、等暫定結果」換來的，不是 partial 變穩。

### 顯示層的反事實（只改畫面顯示規則，由結果 JSON 模擬，不重跑模型）

`可見首字` = 每條第一個有字的句段第一次顯示任何字的時刻（原始音訊時間軸，含開頭靜音，只看相對差）。

| 模型 | 集 | 顯示規則 | 修正率 | 有修正的條 | 被改字 / 最終字（前綴 / 對齊） | 穩定延遲 mean / P90 (ms) | 可見首字 mean (ms) |
|---|---|---|---|---|---|---|---|
| X-ASR | 200 | 原樣 | 5.2% | 31.5% | 2.9% / 2.9% | 19 / 0 | 1508 |
| X-ASR | 200 | **hold** | **0.1%** | **1.0%** | **0.1% / 0.1%** | 0 / 0 | 1508 |
| X-ASR | 連續 | 原樣 | 8.5% | 92.2% | 5.8% / 5.8% | 41 / 0 | 1027 |
| X-ASR | 連續 | **hold** | **0.2%** | **5.2%** | **0.1% / 0.1%** | 1 / 0 | 1027 |
| SenseVoice p480 | 200 | 原樣 | 29.5% | 84.5% | 45.5% / 17.6% | 315 / 989 | 918 |
| SenseVoice p480 | 200 | la2 | 8.7% | 29.0% | 16.5% / 13.9% | 193 / 306 | 1662（+744） |
| SenseVoice p480 | 200 | 去 ASCII | 26.0% | 74.5% | 42.7% / 14.8% | 300 / 987 | 1076（+158） |
| SenseVoice p480 | 連續 | 原樣 | 20.9% | 98.7% | 45.7% / 12.8% | 521 / 1001 | 493 |
| SenseVoice p480 | 連續 | la2 | 5.2% | 53.2% | 20.0% / 17.6% | 423 / 0 | 1162（+669） |
| SenseVoice p480 | 連續 | 去 ASCII | 20.1% | 97.4% | 45.2% / 12.3% | 519 / 1001 | 570（+77） |

- **hold**：暫定結果之後，partial 若是暫定結果的真前綴（恢復後還沒追上）就不顯示、維持暫定結果。X-ASR 的修正幾乎全部消失，首字不變。
- **la2**（local agreement）：partial 只顯示最近兩次 partial 的共同前綴。偽串流的修正率降到 5–9%，但可見首字晚 670–740 ms，**比 X-ASR 原樣還晚**，首字優勢全部吐回去；被換掉的字（對齊口徑）也沒有減少，因為最終結果仍會改掉已確認的前綴。max3 搭 la2 更糟（partial 太少）。
- **去 ASCII**：丟掉全是英文 / 數字的 partial，修正略減，首字晚 80–160 ms。

### 結論

- **跳動程度**：在同一套口徑下，SenseVoice 偽串流（p480）被改掉的字是 X-ASR 的 2–6 倍（對齊口徑 12.8–17.6% vs 2.9–5.8%；前綴口徑 45% vs 3–6%），修正率 3–6 倍，字的穩定延遲 P90 約 1 s vs 0；單句有 85% 的句子畫面上出現過被改掉的字（X-ASR 31%）。X-ASR 的跳動只是 A2 暫定結果「縮回再補回」1–2 個字，用 hold 顯示規則可以幾乎歸零（0.1%），不影響延遲。
- **偽串流的跳動會是當預設的阻礙，至少不能只看首字**：它比 X-ASR 快約 0.5 s 的首字，有一部分是「先顯示不可靠的字」換來的——第一個 partial 有 14–32% 是英文碎片、字的穩定時間 P90 約 1 s。用 local agreement 把跳動壓下來，首字就比 X-ASR 還晚。對車艙的「只看最終結果執行指令」不受影響（CER 仍是偽串流較好、最終結果不受 partial 影響）；對「邊說邊顯示」的畫面，偽串流的跳動明顯比 X-ASR 差。要讓偽串流當預設，需要先決定畫面要不要顯示 partial，或找到不犧牲首字的穩定方法（例如只顯示與前一次一致的前綴 + 其餘淡色顯示、第一個 partial 濾掉 ASCII / 過短的結果）。
- max3 排程讓 partial 次數與被改的字減半，但代價是 3 s 後畫面不再更新，穩定度數字不能單獨看。
- 預設不變（量測只加欄位；顯示規則的反事實沒有改程式）。

### 限制

- 只在 200 / 300 句小集合上；沒有信賴區間（字數級的指標彼此相關，bootstrap 未做）。
- 「畫面」是事件驅動的模擬：partial 只在文字改變時記錄；暫定結果後的第一個 partial 若與停頓前最後一個 partial 相同，不會產生事件，畫面維持暫定結果直到下一次變化（與真實 UI 依事件更新的行為一致）。
- 共同前綴口徑把「句中改一字」之後的字都算成被改，是上界；對齊口徑是下界，兩者一起看。
- 沒有使用者研究：多少跳動是可接受的、淡色顯示未穩定的字是否有效，都沒有量。
- max3 在未合併分支上量；合併 `feat/pseudo-stream` 後應在主分支重跑同一組命令確認。
- hold / la2 / 去 ASCII 只是用結果 JSON 模擬，沒有實作進 pipeline 或 demo。

重現：

```bash
uv run eval/run_eval.py --smoke --model x-asr-480ms-int8-ort sensevoice-small-int8-ort-p480 --jobs 2 --tag stab
uv run eval/run_eval.py --manifest data/aishell1_continuous_300/manifest.jsonl \
    --model x-asr-480ms-int8-ort sensevoice-small-int8-ort-p480 --jobs 2 --tag stab
```

---

## 2026-11-02　第二階段前哨：onnxruntime CoreML EP 跑 X-ASR 480 ms 官方 INT8（不加速）

環境：Apple M5（4 效能核 + 6 節能核）、macOS 27.0.1、onnxruntime 1.31.1（PyPI 版已內建 `CoreMLExecutionProvider`，不需另裝）、單執行緒。程式在 `feat/latency-b1`（`46fe454`）+ 本次提交上執行。量測期間機器上有其他工作階段（load average 2–4），計算時間只看同批交錯的相對差。第二階段的定義照使用者的決定：以官方 INT8 模型為準，不自行量化、不建 FP32 參照。

### 做了什麼

- `ZipformerOrtAsr` 多一個 `coreml: CoreMLOptions | None` 參數（預設 None = 只用 CPU，預設行為不變）：`compute_units`（CPUOnly / CPUAndGPU / CPUAndNeuralEngine / ALL）、`model_format`（MLProgram / NeuralNetwork）、`parts`（哪幾個 session 用 CoreML，預設只有 encoder）、`fix_batch_dim`（見下）、`extra`（其他 provider 選項，例如 `ModelCacheDirectory`）。
- 登記 4 個量測用名稱 `x-asr-480ms-int8-ort-coreml-{cpu,gpu,ane,all}`（encoder 用 CoreML EP、MLProgram），可直接給 `run_eval.py --model`。
- `scripts/bench_asr_compute.py` 加 `--ep cpu coreml:<units>[:<格式>[:<parts>]]`，並記錄載入時間。
- 測試：`sess_options` 與 `coreml` 不能同時給；CoreML encoder（CPUOnly）在 4 個官方測試音訊上與 CPU 版文字接近（沒有 CoreML EP 時 skip）。

### 節點分配（verbose log 的 GetCapability / VerifyEachNodeIsAssignedToAnEp）

| 模型 | 設定 | 圖中節點（ORT 基本最佳化後） | CoreML 接下的節點 | partition 數 | 主要掉回 CPU 的運算 |
|---|---|---|---|---|---|
| encoder | MLProgram，N 為符號維度 | 7431 | 1880 | 525 | 形狀推不出（3854 次「has no shape」）+ INT8 運算 |
| encoder | **MLProgram，N 固定為 1** | 4828 | **3915** | **415** | MatMulInteger 325、DynamicQuantizeLinear 306、Log 99、Equal 95、Expand 43、GatherElements 19 |
| encoder | NeuralNetwork，N = 1 | 4828 | 3008 | 629 | 同上，另有更多運算不支援 |
| encoder | MLProgram，N = 1，`RequireStaticInputShapes` / `EnableOnSubgraphs` | 4828 | 3914 / 3915 | 415 | 沒有差別 |
| joiner | MLProgram（N 不能固定，批次 joiner 會變動） | 8 | 6（Add、Tanh 等） | 2 | DynamicQuantizeLinear + MatMulInteger |
| decoder | MLProgram，N = 1 | 12 | 1（embedding Gather） | 1 | int64 的 Clip、GreaterOrEqual，之後的張量沒有形狀 |
| Silero VAD | ALL（含 / 不含 `EnableOnSubgraphs`） | 2（頂層是 sr 的 If） | 0 / 子圖中 10 | 0 / 少量 | 頂層 If 不支援；子圖只接得下零星運算 |

- encoder 的輸入、狀態都有符號批次維度 `N`，ORT 推不出大部分中間張量的形狀，CoreML EP 只接得下 1/4。以 `SessionOptions.add_free_dimension_override_by_name("N", 1)` 固定後（單一 stream 一定是 1）常數摺疊更多、圖從 7431 縮到 4828 個節點，CoreML 接下 81%。`CoreMLOptions.fix_batch_dim` 預設開，只套在 encoder / decoder。
- **剩下的 19% 正好是每一層的量化矩陣乘法**：CoreML EP 不支援 `DynamicQuantizeLinear`、`MatMulInteger`（ONNX 的動態 INT8 量化格式在 Core ML 沒有對應運算），於是每個線性層都把圖切斷，encoder 被切成 415 段，每個 chunk 要在 CPU 與 Core ML 之間來回 415 次。
- 而且 partition 先於 ORT 的融合：CPU 版會把 `DynamicQuantizeLinear + MatMulInteger` 融合成 `DynamicQuantizeMatMul`（B8），切給 CoreML 之後這兩個運算各自孤立、不再融合，CPU 端的 INT8 矩陣乘法反而變慢。

**ORT profiling（encoder 10 個 chunk，負載下、profiling 本身也有開銷，只看比例）**

| EP | 每 chunk (ms) | CoreML 段 | CPU 段 | CPU 段最大項 |
|---|---|---|---|---|
| CPU | 29.4 | — | 23.5 ms / 3127 個 kernel | Conv 5.6、DynamicQuantizeMatMul 5.0（融合後） |
| CoreML CPUOnly | 55.3 | 40.9 ms / 415 次呼叫 | 14.0 ms / 913 個 kernel | MatMulInteger 10.8（未融合）、DynamicQuantizeLinear 1.3 |
| CoreML CPUAndNeuralEngine | 48.9 | 37.9 ms / 415 次 | 11.8 ms | MatMulInteger 9.2 |
| CoreML ALL | 61.4 | 47.8 ms / 415 次 | 14.6 ms | MatMulInteger 11.3 |

每次 CoreML 呼叫平均約 0.1 ms，幾乎都是固定開銷（輸入輸出複製、排程），415 次就約 40 ms，比整個 CPU encoder 還久。

### 純計算（`bench_asr_compute.py --threads 1 --ep ... --repeat 2`，冒煙集前 60 句）

| EP（encoder） | 載入 (s) | ASR RTF | 相對 CPU | encoder mean / P90 (ms) | finalize mean / P90 (ms) | 重新解碼 (ms / 音訊 s) | footprint 載入後 / 峰值 (MB) | 峰值 RSS (MB) | token 不同（/ 60 句） |
|---|---|---|---|---|---|---|---|---|---|
| **CPU（基準）** | 0.8 | **0.0444** | 1.00 | **16.2 / 17.0** | **58.5 / 69.6** | **33.6** | 296 / 332 | 370 | — |
| CoreML CPUOnly，MLProgram | 17.3 | 0.0993 | 2.24 倍慢 | 37.1 / 38.2 | 130.4 / 152.8 | 78.6 | 315 / 384 | 464 | 1 |
| CoreML CPUAndGPU，MLProgram | 19.4 | 0.1210 | 2.73 倍慢 | 45.2 / 47.1 | 159.7 / 189.3 | 89.7 | 324 / 388 | 468 | 1 |
| CoreML CPUAndNeuralEngine，MLProgram | 21.2 | 0.1213 | 2.73 倍慢 | 45.4 / 50.9 | 159.2 / 202.0 | 82.6 | 319 / 398 | 480 | 2 |
| CoreML ALL，MLProgram | 20.4 | 0.1125 | 2.53 倍慢 | 42.1 / 43.7 | 148.1 / 173.7 | 81.7 | 311 / 392 | 474 | 2 |
| CoreML CPUOnly，NeuralNetwork | 5.9 | 0.1021 | 2.30 倍慢 | 38.2 / 39.3 | 134.1 / 156.8 | 74.5 | 394 / 465 | 516 | 4 |
| CoreML CPUAndGPU，NeuralNetwork | 6.3 | 0.1494 | 3.36 倍慢 | 56.2 / 58.5 | 196.4 / 228.3 | 107.2 | 468 / 571 | 576 | 2 |
| CoreML CPUAndNeuralEngine，NeuralNetwork | 9.4 | 0.1273 | 2.87 倍慢 | 47.8 / 52.6 | 167.3 / 198.5 | 91.0 | 388 / 472 | 524 | 3 |
| CoreML ALL，NeuralNetwork | 9.1 | 0.1331 | 3.00 倍慢 | 50.0 / 55.1 | 175.4 / 221.0 | 120.9 | 388 / 473 | 525 | 3 |

- **沒有任何組合比 CPU 快**：encoder 每 chunk 慢 2.3–3.5 倍，finalize（停頓時產生暫定結果）多 72–138 ms，這些都會直接加在結果延遲上。CPUOnly 最不慢，表示 GPU / ANE 本身沒帶來收益，時間都花在切段的往返上。
- 每個設定兩輪的 token 都相同（可重現）；token 雜湊顯示 CPUOnly 與 CPUAndGPU 結果相同、CPUAndNeuralEngine 與 ALL 相同（ALL 實際把能上 ANE 的段放在 ANE）。
- **載入要編譯 Core ML 模型**：MLProgram 17–21 s、NeuralNetwork 6–9 s（CPU 0.8 s）。給 `ModelCacheDirectory` 後第二次載入降到 3.7 s（CPUAndNeuralEngine，快取 21 MB），仍是 CPU 的 4–5 倍。
- 記憶體：footprint 峰值多 50–240 MB、峰值 RSS 多 94–206 MB（Core ML 模型與每段的輸入輸出緩衝）。

### 全鏈路（`run_eval.py --smoke`，200 句，`--jobs 1`，A2 預設設定）

| 模型 | CER | 95% CI（以句） | RTF | 首字 mean / P90 (ms) | 暫定結果產生 / 恢復時重新解碼 mean (ms) | 峰值 RSS (MB) | 最終文字與 CPU 不同 | 這些句子的錯字數變化 |
|---|---|---|---|---|---|---|---|---|
| `x-asr-480ms-int8-ort`（CPU） | 4.80% | [3.76, 5.93] | **0.0369** | 1008 / 1034 | **53 / 88** | 354 | — | — |
| `…-coreml-cpu` | 4.67% | [3.63, 5.80] | 0.0727（1.97 倍） | 1023 / 1048 | 107 / 185 | 449 | 9 句 | 5 句變好、2 句變差，合計 −4 |
| `…-coreml-gpu` | 4.67% | [3.63, 5.80] | 0.0856（2.32 倍） | 1029 / 1058 | 126 / 220 | 457 | 9 句（與 cpu 相同） | 同上 |
| `…-coreml-ane` | 4.73% | [3.70, 5.87] | 0.0802（2.17 倍） | 1024 / 1051 | 118 / 208 | 412 | 7 句 | 3 好、1 差，合計 −2 |
| `…-coreml-all` | 4.73% | [3.70, 5.87] | 0.0769（2.08 倍） | 1023 / 1049 | 113 / 199 | 464 | 7 句（與 ane 相同） | 同上 |

- CER 差異（−0.07 至 −0.13 個百分點）遠在信賴區間（約 ±1 個百分點）內，**不能解讀為變準**；結果改變的原因是非量化部分改在 Core ML 以不同的浮點精度 / 累加順序計算。差異集中在人名、同音字（例：周云路 → 周云露、首饰 → 熟食）。
- 首字延遲只多 15–21 ms：首字主要卡在模型輸出延遲（B1），計算只佔一小部分；但 RTF 約 2 倍，A2 停頓時產生暫定結果的計算 53 → 107–126 ms、恢復說話時的重新解碼 88 → 185–220 ms，都直接加在結果延遲上。

### 附帶發現：CPU 上固定 N = 1

只在 CPU 上給 `add_free_dimension_override_by_name("N", 1)`（隨機輸入 40 個 chunk × 6 輪交錯）：encoder 每 chunk 19.8 → 18.5 ms（約 −7%，常數摺疊後節點變少），但輸出**不逐位元相同**（40 個 chunk 串接狀態後 encoder_out 最多差 0.23），推測是融合 / 量化路徑改變。沒有採用；要用的話需在 200 / 300 句上確認 CER 與逐句差異。

### 結論

- **onnxruntime CoreML EP 不能加速官方 INT8 的 X-ASR**：任何 compute units × 格式都比 CPU 慢 2–3.4 倍（純計算）、全鏈路 RTF 約 2 倍，載入多 3–20 s、記憶體多約 100 MB。CER 沒有顯著變化。預設維持 CPU。
- 主要障礙是**模型的量化格式**：官方 INT8 是 ONNX 的動態量化（`DynamicQuantizeLinear` + `MatMulInteger`），Core ML 沒有對應運算，每個線性層都被切回 CPU，encoder 被切成 415 段；每段往返約 0.1 ms 的固定開銷，總開銷超過整個 CPU encoder。其次是符號批次維度讓形狀推不出（已用固定 N 解決）。
- 也就是說「官方 INT8 + Core ML」這條路，用 ORT CoreML EP 走不通；要讓 encoder 整段（或少數幾段）落在 Core ML，矩陣乘法必須以 Core ML 支援的形式表達。

### 下一步選項（只評估，未下載）

官方 Hugging Face repo（GilgameshWind/X-ASR-zh-en，只看了檔案清單）另有 `deployment/models/chunk-480ms-model/encoder-480ms.onnx`（FP32，593 MB）、joiner 10 MB、decoder 11 MB，以及 `streaming_exp/pretrained.pt`（2.56 GB，PyTorch checkpoint）。本機目前沒有 `onnx`、`coremltools`、`torch`。

| 選項 | 做法 | 需要下載 | 預期 / 風險 |
|---|---|---|---|
| A. 改寫官方 INT8 ONNX（建議先做） | 用 `onnx` 把每組 `DynamicQuantizeLinear + MatMulInteger (+ Mul/Add)` 換成 `MatMul`，權重以 INT8 × scale 反量化成 FP16 / FP32 常數（只用官方 INT8 權重，不自行量化），再接 ORT CoreML EP（N 固定為 1） | `onnx` 套件（wheel 約 15–20 MB） | 量化運算消失後只剩 Log / Equal / Expand 等少量不支援（約 280 個節點，可能多半能常數摺疊或改寫），partition 可望降到個位數；權重變 FP16 約 300 MB（Core ML 內），比 INT8 的 155 MB 大。數值上是「INT8 權重 + 浮點啟動值」，與官方 INT8（動態量化啟動值）不同，CER 要重量 |
| B. coremltools 直接轉 | coremltools 7 起已移除 ONNX 前端，只能從 PyTorch（`torch.jit.trace` / `torch.export`）轉；需 icefall 的 X-ASR zipformer2 程式碼、串流狀態的輸入輸出自己包；轉出後可用 coremltools 的權重量化（W8 / palettization）在 ANE 上跑 | `pretrained.pt` 2.56 GB + torch（約 200–300 MB）+ coremltools（約 10 MB）+ X-ASR repo 程式碼 | 最接近第三階段（iOS + Core ML）的最終形態，也最能用到 ANE；工作量最大（zipformer2 串流版轉 Core ML 社群少有現成腳本），且 2.56 GB checkpoint 可能含優化器狀態 |
| C. 用 FP32 ONNX 接 CoreML EP | 下載 `encoder-480ms.onnx` 直接跑 | 593 MB + 21 MB | 與 A 的效果類似但權重不是官方 INT8；屬於使用者已排除的「FP32 參照」範疇，不建議 |

建議：先做 A（小下載、可在現有評測框架內 1–2 天驗證 partition 數與加速比）；若 A 證明整段 encoder 在 ANE 上能明顯快於 CPU，再投入 B 為 iOS 做正式的 Core ML 模型。

---

## 2026-11-01　偽串流（SenseVoice）降計算量與短句表現

環境同上；評測在 commit `b6583c5` + 未提交的程式上執行，程式即 `feat/pseudo-stream` 的提交。數字在 [scoreboard.md](scoreboard.md)「偽串流降計算量」一節，比較表由 `uv run scripts/pseudo_stream_compare.py` 產生。

**範圍**：原本要在 200 / 300 句上重跑偽串流 vs X-ASR 的完整比較，中途依使用者決定（「跳過，只做新的」）停掉。完整比較沿用上方「非串流模型對照」的 100 句結果；本次只做降計算量、短句子集、授權。完整比較停掉前跑完的 200 句冒煙集兩列（X-ASR、SenseVoice 無偽 partial）保留在 scoreboard。

### 做了什麼

- `PartialSchedule`（`src/asr/sense_voice.py`）：偽 partial 的時機以已累積的音訊長度計，`first_s`（第一次）、`every_s`（之後的間隔）、`max_s`（超過就不再重新辨識，只等暫定結果 / 定稿）。新登記 `sensevoice-small-int8-ort-p960`、`-p480-960`、`-p480-max3`、`-p480-960-max3`；`-p480` 行為不變（逐樣本相同的時機）。**預設模型不變**（X-ASR）。
- 「只在停頓時重新辨識」不需要新程式：`sensevoice-small-int8-ort`（無偽 partial）在 A2 下只在停頓 150 ms 時 finalize 出暫定結果。
- 計算量指標：`StreamingAsr.compute_stats()`，SenseVoice 記錄整段辨識的次數與音訊總長；`run_eval` 存成 `metrics.asr_compute`（含 `decoded_audio_ratio`）。評測時同機另一個工作階段在跑多執行緒 benchmark（load average 2.5–6.6），RTF 與延遲都偏高且不穩，第一輪的數字（p480 單句 RTF 0.244、結果延遲 562 ms）比第二輪（0.163、414 ms）差很多，所以加了這個與負載無關的指標，表中用第二輪。
- 首字延遲改報「可見」口徑：partial、暫定結果、最終結果取最早（上方 SenseVoice 段落的「指標缺口」），由比較腳本從結果 JSON 計算，`run_eval` 的 `first_token_latency_ms` 口徑不變。
- 短句子集 `scripts/prepare_short_subset.py`：對齊集中 MFA 語音長度 ≤ 3 s 的 100 句（1.43–2.98 s）。AISHELL 句子平均 4–5 s，車艙指令多是 1–3 s。
- `eval.metrics.paired_bootstrap_cer_diff`：同一批句子的成對 bootstrap CER 差。
- 授權查證寫進 [models.md](models.md)「商用限制查證」。

### 結果（短句 / 單句 / 連續）

| | X-ASR | SenseVoice p480 | **SenseVoice p480-960-max3** | SenseVoice 只在停頓時 |
|---|---|---|---|---|
| 首字（可見，真實起點，ms） | 970 / 1059 / 1086 | 440 / 455 / 444 | **429 / 449 / 433** | 2588 / 3443 / 3364 |
| 說完 → 完整結果 mean（ms） | **251 / 252 / 246** | 323 / 414 / 835 | 269 / 303 / 374 | 302 / 371 / 498 |
| 重新辨識音訊比 | — | 1.88 / 3.63 / 11.61 | 1.19 / 1.45 / 3.37 | 0.50 / 0.72 / 1.86 |
| RTF | 0.033 / 0.042 / 0.097 | 0.090 / 0.163 / 0.480 | 0.039 / 0.045 / 0.095 | 0.025 / 0.036 / 0.085 |

1. **第一次 partial 決定首字，之後的更新可以稀疏。** p480-960（0.48 s 後每 0.96 s）首字與 p480 相同（438 vs 440 ms），短句計算量 1.88 → 1.19（−37%）、單句 3.63 → 2.30。單純把間隔加倍（p960）計算量差不多，首字卻變成 934 ms，與 X-ASR 差不多，失去偽串流的意義。
2. **限制重新辨識長度（max 3 s）解決長句段的最壞情況。** 連續集（A2 下多句合併成長句段）p480 每次都重新辨識十幾秒的音訊，音訊比 11.6、RTF 0.48，計算積壓讓「說完 → 完整結果」變成 835 / P90 1396 ms；加 max 3 s 後音訊比 3.86、結果 453 ms。短句（≤ 3 s）不受影響。代價：句段超過 3 s 後 partial 不再更新，畫面停在前 3 s 的文字，直到停頓（150 ms）出暫定結果。
3. **兩者合併（p480-960-max3）是唯一在所有集合上都划算的設定**：首字 429–449 ms（X-ASR 970–1086，快約 540–650 ms）；RTF 0.039 / 0.045 / 0.095，與 X-ASR 0.033 / 0.042 / 0.097 同一量級（兩邊在不同時間量，負載不同，只能說同量級）；說完 → 完整結果短句 269 vs 251 ms，連續 374 vs 246 ms，仍比 X-ASR 慢。
4. **只在停頓時辨識**計算量最低（音訊比 0.5–0.7，比 X-ASR 還省），但說話中完全沒有文字，第一次看到字要 2.6–3.4 s，不適合要即時回饋的介面；適合只需要整句結果的指令辨識（結果延遲 302 ms，短句）。
5. **CER**：三個小集合上 SenseVoice 與 X-ASR 都不顯著（短句 +0.19 [−0.49, +0.89]，單句 −0.84 [−1.99, +0.27]，連續 −0.21 [−1.13, +0.65]）；200 句冒煙集 −1.40 [−2.26, −0.57] 顯著，SenseVoice 較好。各偽串流設定定稿都是整段辨識，最終文字逐句相同，降計算量不影響 CER。

### 結論：偽串流在哪裡贏、代價是什麼

- **贏**：首字延遲（約 430–450 ms vs X-ASR 約 1 s，快 0.5–0.6 s，短句也一樣），這是 X-ASR 推論端降不下來的瓶頸；CER 持平或略好（200 句上顯著）。
- **代價**：
  - 計算量：沒調整的 p480 在短句是 X-ASR 的約 2.7 倍 RTF，長句段約 5 倍；用 p480-960-max3 才壓到同量級。
  - 結果延遲：說完 → 完整結果比 X-ASR 慢 18–128 ms（短句 / 單句 / 連續）。整段辨識比 X-ASR 的補零 finalize 慢；推測停頓時偶爾要先等正在進行的偽 partial 算完（未驗證）。
  - partial 品質：每次都是整段重新辨識，文字可能整句跳動（未量修正率）；max 3 s 後長句的 partial 停止更新。
  - 記憶體與模型：峰值 RSS 410–536 vs 351–370 MB，模型 230.7 vs 163.6 MB；SenseVoice 推論記憶體隨音訊長度增加（約 3.8 MB / s）。
  - 授權：FunASR Model License 文字沒有商用條款，商用依據只有維護者在 issue 的回覆（models.md），比 X-ASR 的 Apache-2.0 弱。
- **決定**：偽串流保留為選項，**預設不變**。若要用，建議 `sensevoice-small-int8-ort-p480-960-max3`。真要切換預設前需要：(1) 在閒置機器上重量 RTF 與延遲；(2) 量 partial 的跳動 / 修正率；(3) 取得 SenseVoice 商用的書面確認。

### 限制

- 只跑小量（100 句 ×2、25 條），CER 的信賴區間約 ±1 個百分點。
- RTF 與延遲在同機有其他 benchmark 時量，設定之間只能用重新辨識音訊比比較計算量。
- 沒有跑車內噪音。

---

## 2026-11-02　決定：預設改為 SenseVoice 偽串流（`sensevoice-small-int8-ort-p480-960-max3`）

使用者決定（「改成預設」）。先合併 `feat/pseudo-stream`（偽串流降計算量的 `PartialSchedule`、短句子集、授權查證；與已合併的熱詞解碼共用 `SenseVoiceConfig` / `SenseVoiceDecoder`），再把 `src/asr/__init__.py` 的 `DEFAULT_MODEL` 從 `x-asr-480ms-int8-ort` 改為 `sensevoice-small-int8-ort-p480-960-max3`。X-ASR 保留為選項；熱詞預設不開（beam 1 = greedy，與合併前逐句相同）。

**200 句冒煙集**（`run_eval.py --smoke --model sensevoice-small-int8-ort-p480-960-max3 x-asr-480ms-int8-ort`，A2 預設斷句，同一批依序跑；load average 約 2.6–3.4，主要是 Claude 桌面程式，RTF 只看相對差）：

| | SenseVoice 偽串流（新預設） | X-ASR 480 ms（舊預設） |
|---|---|---|
| CER [句 95% CI] | **3.40** [2.50, 4.37] | 4.80 [3.76, 5.93] |
| 首字延遲（VAD 起點）mean / P90 (ms) | **405 / 409** | 1006 / 1029 |
| 端到端首字延遲 mean / P90 (ms) | **914 / 971** | 1515 / 1623 |
| 尾字延遲 mean (ms) | 998 | 998 |
| RTF | 0.0507 | **0.0324** |
| 峰值 RSS (MB) | 444 | **358** |
| 模型大小 (MB) | 230.7 | **163.6** |

X-ASR 這一列與合併前的冒煙集結果相同（CER 4.80%）。RTF 比為 1.56 倍，比偽串流降計算量一節「同量級」的說法高；那一節是不同時間、不同負載下量的，這裡是同批。

**採用的理由**：首字快約 0.6 s、CER 持平或略好（200 句顯著，300 句各子集不顯著）、已有熱詞（beam 8 + 熱詞，熱詞句 CER 顯著下降）。

**已知代價**（各節見上）：

- partial 穩定度：被改掉的字是 X-ASR 的 2–6 倍，字穩定延遲 P90 約 1 s，第一個 partial 14–32% 是英文 / 數字碎片；累積超過 3 s 後 partial 不再更新。若畫面要即時顯示 partial，這是主要缺點；只依最終結果執行指令則不受影響。
- 說完 → 完整結果慢約 20–130 ms（短句 / 單句 / 連續）。
- RTF 約 1.5 倍、峰值 RSS 多約 90 MB（隨音訊長度增加，約 3.8 MB / s）、模型大 67 MB。
- 授權：FunASR Model License 文字沒有商用條款，商用依據只有 issue #287 的維護者回覆；正式商用前需書面確認（[models.md](models.md)）。
- 沒有 token 時間戳（`first_token_audio_s` 為 None），依賴它的分析（字級延遲拆解）只適用 X-ASR。

**測試**：`tests/test_pipeline.py::test_end_to_end_two_utterances` 用預設模型，原本檢查的「partial 只追加、有 token 時間戳」是真串流的性質，改為偽串流時只檢查 CER、切段與首字延遲範圍。pytest 176 個、mypy、ruff 通過。

**待辦**：閒置機器上重量 RTF 與延遲；SenseVoice 商用書面確認（使用者處理）；若畫面要顯示 partial，需要不犧牲首字的穩定化顯示規則（B14）。依「優化階段只跑小集合」，3000 句未跑，README 的 3000 句模型比較表沒有 SenseVoice。

---

## 2026-11-02　SenseVoice 偽串流的執行緒數（1 / 2 / 4）

**問題**：X-ASR 多執行緒沒有收益（B8：每個 chunk 只有 61 frame，同步開銷蓋過平行化，且改變 1–5% 句子的結果）。SenseVoice 每次對已累積的音訊整段推論，矩陣大得多，多執行緒可能有效。重點是 A2 停頓時的暫定結果（整段重新辨識）與「說完 → 完整結果」——新預設在這項比 X-ASR 慢 18–128 ms，主要就是這段計算。

**環境**：commit `f3134d4`（分支 `feat/sv-threads`）、Apple M5（4 效能核 + 6 節能核）、macOS 27.0.1、onnxruntime 1.31.1、`sensevoice-small-int8-ort-p480-960-max3`、A2 預設（pause 150 fin / min_silence 1000）、chunk 10 ms。`run_eval.py --threads` 同時設定 ASR 與 Silero VAD 的 `intra_op_num_threads`（VAD 計算量可忽略）。**同機另一個工作階段在跑 SenseVoice 評測**，load average 1.8–8.0（多在 3–6）。

**做法**：
- 全鏈路：200 句冒煙集、短句子集（`aishell1_align_300_short3`，100 句）、300 句對齊集，各 1 / 2 / 4 執行緒，**3 輪交錯**（每輪依序跑完 3 集合 × 3 執行緒），表中取 3 輪中位數。「說完 → 完整結果」= 真實末字結束（MFA 對齊）→ 第一個涵蓋整句的暫定或最終結果（與 `pause_analysis.py cost` 相同算法）。
- 純計算：新腳本 `scripts/bench_sv_compute.py`，從 200 句集截取前 1 / 2 / 3 / 5 s 各 40 段，計時每次整段辨識（fbank + 推論 + CTC greedy），另記程序 CPU 時間（所有執行緒合計）。執行緒 1 / 2 / 4 × session 選項預設 / 關 spinning（`session.intra_op.allow_spinning=0`），3 輪交錯取中位數。為此 `SenseVoiceOrtAsr` 加 `sess_options` 參數（與 X-ASR、Silero 相同）。

### 結果：全鏈路（3 輪中位數）

| 集合 | 執行緒 | CER % | RTF | 首字 mean (ms) | 暫定結果計算 mean / P90 (ms) | 恢復時重新解碼 mean (ms) | 說完 → 完整結果 mean / P90 (ms) | 峰值 RSS (MB) | 與 1 執行緒不同的句數 |
|---|---|---|---|---|---|---|---|---|---|
| 200 句 | 1 | 3.40 | 0.0537 | 407 | 118 / 177 | 100 | — | 444 | — |
| | 2 | 3.40 | 0.0427 | 405 | 90 / 132 | 76 | — | 443 | 0 / 0 / 0 |
| | 4 | 3.40 | 0.0418 | 405 | 84 / 120 | 71 | — | 446 | 0 / 0 / 0 |
| 短句 100 | 1 | 2.49 | 0.0543 | 413 | 116 / 136 | 79 | 302 / 374 | 414 | — |
| | 2 | 2.49 | 0.0430 | 410 | 86 / 98 | 60 | **271 / 342** | 414 | 0 / 0 / 0 |
| | 4 | 2.49 | 0.0373 | 407 | 72 / 82 | 50 | **257 / 324** | 416 | 0 / 0 / 0 |
| 對齊 300 | 1 | 2.56 | 0.0760 | 422 | 161 / 236 | 137 | 363 / 464 | 446 | — |
| | 2 | 2.56 | 0.0563 | 417 | 115 / 168 | 97 | **311 / 397** | 447 | 0 / 0 / 0 |
| | 4 | 2.56 | 0.0415 | 411 | 81 / 115 | 68 | **273 / 351** | 453 | 0 / 0 / 0 |

各輪的「說完 → 完整結果」：短句 t1 302 / 319 / 274、t2 271 / 281 / 253、t4 257 / 261 / 250；對齊 300 t1 363 / 372 / 332、t2 311 / 311 / 278、t4 288 / 273 / 273。**每一輪都是 1 > 2 > 4**，負載變動（輪 3 較閒）影響絕對值但不影響排序。

### 結果：純計算（整段辨識一次，3 輪中位數，load 1.6–3.3）

| 執行緒 | session 選項 | 1 s | 2 s | 3 s | 5 s | 5 s 的 CPU 時間 | 結果與 1 執行緒不同 |
|---|---|---|---|---|---|---|---|
| 1 | 預設 | 36.8 | 61.1 | 85.2 | 136.3 | 136 | — |
| 1 | 關 spinning | 36.8 | 61.1 | 85.3 | 136.2 | 136 | 0 |
| 2 | 預設 | 24.6 | 39.6 | 53.6 | 83.7（−39%） | 167（+23%） | 0 |
| 2 | 關 spinning | 25.9 | 40.7 | 55.4 | 85.8 | 155（+14%） | 0 |
| 4 | 預設 | 20.9 | 33.2 | 43.7 | 66.9（−51%） | 267（+96%） | 0 |
| 4 | 關 spinning | 22.6 | 34.4 | 46.0 | 69.7 | 202（+49%） | 0 |

（mean，ms；P90 與 mean 差 < 1 ms。fbank 等前處理約 3 ms，其餘是模型推論。）前一次同樣的量測在負載較高時（load 2–4），4 執行緒 5 s 為 77.7 ms、2 執行緒 84.2 ms——**4 執行緒的額外收益對負載很敏感**，2 執行緒穩定。

1. **多執行緒對 SenseVoice 有效，與 X-ASR 相反。** 單次整段辨識 2 執行緒 −33~39%、4 執行緒 −43~51%（音訊越長收益越大）。全鏈路上暫定結果的計算 −24~29%（2）/ −29~50%（4），說完 → 完整結果短句 −31 / −45 ms、300 句 −52 / −90 ms。首字幾乎不變（−2~11 ms）：首字取決於 0.48 s 音訊的第一次辨識，只要約 30 ms。
2. **辨識結果完全不變**：三個集合、每一輪，2 / 4 執行緒的最終文字與 1 執行緒逐句相同（CER 成對差 0）；純計算 160 段的文字也全部相同。X-ASR 會改變 1–5% 句子，SenseVoice 不會（推論是一次整段，沒有跨 chunk 的狀態累積誤差）。
3. **代價是 CPU 時間**：spinning 開時 2 執行緒 CPU +23%、4 執行緒 +96%（等待中的執行緒在空轉）。關 spinning 讓 CPU 增加降到 +14% / +49%，牆鐘只慢 2–4%。峰值 RSS 不變（±7 MB）。
4. **與 X-ASR 的差距**：偽串流一節的「說完 → 完整結果」X-ASR 短句約 251 ms。SenseVoice 2 執行緒短句 271 ms、4 執行緒 257 ms，差距由約 50 ms 縮到 20 / 6 ms（不同時間、不同負載量的，只能看量級）。

### 結論與建議

- **值得改**：2 執行緒是好的取捨——結果不變、說完 → 完整結果約 −30~50 ms、RTF −20~26%，只多約 1/4 的 CPU 時間，在有負載時也穩定。4 執行緒再快 15–40 ms，但 CPU 時間幾乎加倍、收益受負載影響大；車機上與其他程序共用 CPU 時不建議。
- **本分支未改預設**（任務規定）。若要改：建議 SenseVoice 的預設執行緒數改為 2（X-ASR 維持 1，B8 已證明它沒有收益且會改結果）；評測報告仍照 CLAUDE.md「預設單執行緒，另報多執行緒」，或在報告中註明。關 spinning 可另行考慮（省 CPU、慢 2–4%）。

### 限制

- 同機另一個工作階段的 SenseVoice 評測一直在跑（load 1.8–8.0），**多執行緒受負載影響比單執行緒大**；表中的改善是同批交錯的相對差，閒置機器上的絕對值可能不同（多半是 4 執行緒收益更大）。
- 只跑小集合（200 / 100 / 300 句）；連續多句集（長句段、max 3 s 之後的暫定結果）沒跑。
- 只在 M5（4 效能核）量；效能核較少的裝置（iPhone）4 執行緒可能排到節能核。
