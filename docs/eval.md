# 評測說明

全鏈路（VAD + 串流 ASR）的評測方法、指標定義與文字正規化規則。程式在 `eval/`，資料準備在 `scripts/prepare_eval_data.py`。

```bash
uv run scripts/prepare_eval_data.py      # 下載正式評測集（AISHELL-1 test 隨機 3000 句）到 data/（一次即可，可重複執行）
uv run scripts/prepare_eval_data.py -n 200   # 200 句冒煙子集
uv run eval/run_eval.py                  # 預設設定，3000 句正式評測集
uv run eval/run_eval.py --smoke          # 200 句子集，快速確認沒有改壞（約 1 分鐘 / 模型）
uv run eval/run_eval.py --model x-asr-480ms-int8 zipformer-zh-14m \
    --min-silence-ms 100 300 500 800 --pause-ms 0 --pre-roll-ms 0 100 200 300 --tag sweep   # 參數掃描
```

多個值的參數（`--model`、`--min-silence-ms`、`--pre-roll-ms`、`--chunk-ms`、`--threads`）取笛卡兒積。
`--jobs N` 以 N 個子程序並行（見下方「並行」）。`--limit N` 只跑前 N 句（除錯用；例如 `--smoke --limit 20` 約 5 秒 / 模型，改程式時先用它確認能跑通）。`--lead-silence-ms` 設定每條音訊前補的靜音（預設 1000，見首字延遲）。`--help` 列出全部參數與預設值。
正式報告的數字一律用 3000 句評測集；`--smoke` 只用來在改動後快速確認結果沒變（200 句的 CER 信賴區間約 ±1 個百分點，不適合下結論）。
`--manifest data/aishell1_test_full/manifest.jsonl` 可跑完整 7176 句（需先 `prepare_eval_data.py -n 7176`，約 1.1 GB）。

## 評測資料

| 項目 | 正式評測集 | 冒煙子集 |
|---|---|---|
| 母體 | AISHELL-1 test set，20 位說話人、7176 句（腳本會檢查句數） | 同左 |
| 抽樣 | `sorted(random.Random(20261030).sample(sorted(全部 ID), 3000))` | 同左，200 句。同一 seed 下 `random.sample` 前段的抽選幾乎相同，200 句中有 199 句也在 3000 句集裡 |
| 規模 | 3000 句、251.2 分鐘（4.19 小時）、20 位說話人 | 200 句、17.2 分鐘、20 位說話人全部涵蓋 |
| 指令 | `prepare_eval_data.py`（預設） | `prepare_eval_data.py -n 200` |
| 輸出 | `data/aishell1_test_3000/` | `data/aishell1_test_200/` |
| 音訊來源 | Hugging Face `shenyunhang/AISHELL-1`，固定 revision `2724409d53`，逐檔以 LFS SHA-256 校驗 | 同左 |
| 標註來源 | 同 repo 的 `aishell_transcript_v0.8.txt`，SHA-256 `b5f33b9e…`（寫死在腳本中） | 同左 |

輸出目錄內含 `wav/`、`manifest.jsonl`、`meta.json`，不進 git。模型訓練資料是否含 AISHELL-1 見 [models.md](models.md)。

選這個來源的原因：官方 `AISHELL/AISHELL-1` 的 Hugging Face repo 只上傳了 100 位訓練集說話人，沒有 test set；OpenSLR SLR33 只提供 15 GB 的完整壓縮檔。`shenyunhang/AISHELL-1` 保留官方的 `data_aishell/wav/test/<說話人>/<句子>.wav` 結構，可以只下載抽中的檔案（200 句約 32 MB、3000 句約 470 MB）。它是第三方鏡像，不是官方發布；句數與標註檔大小都與官方一致，但音訊內容無法對官方 checksum 驗證。

### 連續多句集（B4）

`uv run scripts/prepare_continuous_data.py` → `data/aishell1_continuous_300/`。由 300 句 MFA 字級對齊集（`prepare_alignment.py`）合成：

- 同一說話人的句子（seed 20261101 打亂）每 4 句串成一條，共 77 條、300 句、23.2 分鐘。
- 每句只保留真實語音起訖點（MFA 首字起點 / 末字結束）外 50 ms 的原始音訊；句間停頓（前一句末字結束 → 下一句首字起點）依 0.2 / 0.3 / 0.5 / 0.8 / 1.2 / 2.0 s 循環分配，各 37–38 個，扣掉兩邊 50 ms 的部分填數位靜音。
- manifest 每條多了 `sentences`（每句的 `id`、`text`、真實 `start_s` / `end_s`）與 `pauses_s`。
- 每句的真實起訖點已知，不依賴 VAD；起訖點的精度取決於 MFA（約 10 ms frame，尚未人工抽查）。
- 限制：margin 內是原始錄音底噪、停頓中間是數位零，底噪不連續；句間停頓長度是人為分檔，不是真實口語的分佈，所以結果要**按停頓長度分開看**，整體平均值取決於分檔比例、沒有意義。沒有噪音。

`run_eval.py --manifest data/aishell1_continuous_300/manifest.jsonl` 時，結果多一個 `metrics.segmentation`（見下方「斷句指標」）。

## 執行方式

- 每個設定在**獨立子程序**中載入模型、跑完全部句子、量峰值記憶體；父程序只做 CER 與彙整。
- 每條 wav 用 `src.pipeline.transcribe` 以 `chunk_ms`（預設 10 ms）切塊送入，**模擬即時**：一個 chunk 要等最後一個樣本「到達」（依音訊實際時長）才能送入，送入後的計算時間實際量測並累加，計算比音訊慢時會自然積壓。不真的 sleep。
- 一條 wav 被 VAD 切成多段時，各段文字以空白連接後當作這條的假設文字（正規化會去掉空白）。
- 每條 wav 前面先補 `lead_silence_ms`（預設 1000 ms）的數位靜音（全零樣本）再送入，作為端到端首字延遲的起點。結果 JSON 中的位置（`speech_start_s` 等）都換回原始音訊的時間軸（補靜音結束 = 0）。設 `--lead-silence-ms 0` 可重現 2026-10-30 以前的設定。
- 每條 wav 後面再補 `tail_silence_ms`（預設 1500 ms）的數位靜音，讓句子以 VAD 的靜音計時收尾，而不是由音訊結束（flush）收尾，尾字延遲才不會被截斷（見尾字延遲）。設 `--tail-silence-ms 0` 可重現 2026-10-31 以前的設定；結果名稱非 0 時加 `_tail<值>`。
- 單執行緒（VAD 與 ASR 的 ONNX Runtime intra-op 都是 1）、CPU。`--threads 2 / 4` 沒有加速且會改變少數句子的結果（progress.md 2026-11-01「X-ASR onnxruntime 後端調校」）；計算時間的細部拆解用 `scripts/bench_asr_compute.py`。

### 並行（`--jobs`）

時間幾乎全花在模型推論（profile：ASR `decode_stream` 85%、VAD 9%，Python 開銷 < 2%），單程序沒有可省的空間。`--jobs N` 同時跑 N 個子程序；設定數少於 N 時把同一設定的句子連續切成多份，結果依 manifest 順序合併（峰值 RSS、載入時間取各份最大值）。結果 JSON 的 `config.parallel_jobs` / `config.num_shards` 記錄並行方式。

並行會互搶 CPU，計算時間變長，而計算時間會計入模擬即時的延遲。200 句冒煙集實測（2026-10-31，Apple M5，4 效能核 + 6 節能核）：

| jobs | 牆鐘時間 | RTF | 首字延遲 mean (ms) | 尾字延遲 mean (ms) | CER |
|---|---|---|---|---|---|
| 1 | 44 s | 0.0268 | 1002 | 533 | 5.14% |
| 2 | 25 s | 0.0297（+11%） | 1003 | 538 | 5.14% |
| 3 | 22 s | 0.0366（+37%） | 1007 | 550 | 5.14% |
| 4 | 25 s | 0.0552（+106%） | 1016 | 582 | 5.14% |

- **CER 與切段不受影響**（逐句結果相同），只看 CER 的掃描可用 `--jobs 4`。
- **延遲比較用 `--jobs 2`**（偏差 ≤ 5 ms），且比較的兩邊要用同樣的 jobs。
- **正式的 RTF、峰值 RSS 與基準延遲一律 `--jobs 1`**（預設）。

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

**95% 信賴區間**（`metrics.cer_ci95`）：百分位 bootstrap，重抽 10000 次、seed 0，每次重新計算語料級 CER（Σ錯誤 / Σ字數），取 2.5% 與 97.5% 分位數。報告兩種：

- **以句重抽**（`utterance`，主要報告值）：把句子當成獨立樣本。回答「同樣這 20 位說話人、換一批句子，CER 會落在哪裡」。
- **以說話人重抽**（`speaker`）：cluster bootstrap，每次抽 20 位說話人（可重複），帶入各自全部句子。同一說話人的句子錯誤率彼此相關，這個區間較寬，比較接近「換一批說話人」時的不確定性。說話人只有 20 位，區間本身也不太穩定，作為輔助參考。說話人由句子 ID 中的 `S\d{4}` 解析，解析不到時為 `null`。

比較兩個設定時，信賴區間重疊不代表差異不顯著；要用同一批句子的成對 bootstrap（見 progress.md）。

### RTF

`RTF = Σ 每句處理時間 / Σ 每句送入的音訊時長`（含前補靜音，`processed_s`）。每句處理時間 = `transcribe()` 的牆鐘時間，涵蓋 VAD、ASR 解碼、finalize（含補零）；**不含**讀 wav 與模型載入。正式計時前先跑前 N 句暖機（預設 N = 3，結果 JSON 記錄在 `config.warmup`），暖機結果丟棄、這 N 句之後仍正常計入。

注意 10 ms chunk 的 Python 呼叫開銷會計入 RTF；chunk 越大 RTF 越低，與延遲需一起看。

### 首字延遲

報告兩種，終點相同（第一個非空 partial 輸出的模擬即時時刻），起點不同：

| | 端到端首字延遲（`e2e_first_token_latency_ms`，主要） | VAD 起點首字延遲（`first_token_latency_ms`，輔助） |
|---|---|---|
| 起點 | 前補靜音的結束點 = 原始音訊第 0 個樣本 | VAD 判定的語音起點（含 `speech_pad_ms`） |
| 與 VAD 的關係 | 無關：起點由評測框架決定 | 起點由被測系統的一部分決定 |
| 包含 | 原始音訊開頭的靜音 + VAD 確認延遲 + ASR 等待 chunk + 計算 | ASR 等待 chunk + 計算 + VAD 確認延遲（扣掉 VAD 切點與真實起音的誤差） |
| 適合 | 比較不同 VAD / pipeline / 模型設定（相對比較） | 拆解延遲來源 |

**端到端首字延遲的限制**：AISHELL 原始音訊開頭本來就有一段長度不一的靜音（未標註），這段時間被算進延遲，所以絕對值**高估**使用者感受到的延遲，不能直接當作「說話後多久出字」。但同一條音訊在所有設定下都一樣，因此適合做設定間的相對比較（差值有意義）。要得到絕對值需要已知語音起點的評測集（backlog B4 / B5）。

**VAD 起點首字延遲的限制**：VAD 切得晚時延遲反而「看起來比較好」，等於用被測系統的一部分當尺；改變 VAD 參數或做 VAD 預先啟動（B2）時，這個數字不能用來比較。

**為何要補靜音**：原始音訊沒有與 VAD 無關的語音起點標註；補靜音後，「靜音結束」是評測框架自己決定、精確已知的時間點，可以當作不依賴 VAD 的起點。注意 1000 ms 不是 VAD frame（32 ms）的整數倍，補靜音會讓 VAD frame 相對語音的切點位移，結果與不補時會有少量差異（20 句小集合上 CER 差 1 個字）。

VAD 起點首字延遲的細節：

- **起點**：VAD 判定的語音起點，即該句段在音訊時間軸上的起點 `Utterance.speech_start_s`。這是 VAD 狀態機判定為語音的第一個 frame 再往前延伸 `speech_pad_ms`（30 ms）的位置，**不是**強制對齊的真實起點，也**不是** VAD 發出「開始」事件的時刻。
- **終點**：第一個非空 partial 結果輸出的模擬即時時刻（若 partial 一直為空、直到 finalize 才出字，取 finalize 完成時刻）。
- 因此首字延遲包含：VAD 確認語音所需的等待（約 `min_speech_ms`，結果 JSON 另列 `vad_start_delay_ms`）、ASR 等待第一個 chunk 收滿、以及計算時間。
- **每條音訊取一個值**：該條第一個有輸出文字的句段。這樣不同 `min_silence_ms` 下樣本數固定為 200，可以直接比較；沒有任何輸出的句子不計入，數量另列 `no_text_utterances`。逐段數值保留在結果 JSON 的 `utterances[].segments`。
- 報告平均與 P90（另存 P50 / P99），單位 ms。分位數用 numpy 線性內插。
- `pre_roll_ms` 只改變送入 ASR 的起點，不改變上述起點，因此不同 pre-roll 的首字延遲可直接比較。
- 端到端首字延遲取**同一個句段**（每條第一個有字的句段），樣本數相同。
- 結果 JSON 每個句段另記 `first_token_audio_s`：ASR 後端回報的第一個 token 時間戳（sherpa-onnx 為 40 ms 解析度）換算到原始音訊時間軸，用於延遲拆解（[latency-analysis.md](latency-analysis.md)）。

### 尾字延遲（輔助）

每條音訊最後一個句段：從 VAD 語音終點（`speech_end_s`，含 padding）到最終結果輸出。包含 VAD 等待 `min_silence_ms` 確認結束，以及 finalize 的補零解碼。

**最終文字提早就緒**（`final_text_ready_lead_ms`，診斷用）：同一句段，「最終結果的完整文字第一次出現在 partial 中」的時刻（`final_text_ready_s`）到最終結果的時間差；partial 從未等於最終結果時為 0。`final_text_ready_before_end` 是最終文字在 VAD 結束事件之前就已出現的句數。逐句段另記 `end_emitted_s`、`final_text_ready_s`、`final_s`（原始音訊時間軸）。以 MFA 真實末字結束點為起點的拆解見 `scripts/endpoint_diagnosis.py`。

### 斷句指標（連續多句集）

`eval/segmentation.py`。句段與句子「重疊」指句段的 VAD 區間與句子的真實語音區間交集超過 50 ms。

| 欄位 | 定義 |
|---|---|
| `boundary_detected_by_pause` | 每種停頓長度下，句間邊界被正確切開的比例。有一個句段同時與前後兩句重疊 = 合併 |
| `sentences_split_inside` | 與 2 個以上句段重疊的句子數（句中誤切） |
| `sentences_missed` | 沒有任何句段重疊的句子數 |
| `exact_files` | 句段與句子一一對應的條數 |
| `tail_latency_ms_by_pause` | 每句的**真實尾字延遲**：真實末字結束 → 與它重疊的最後一個句段的最終結果。依這句之後的停頓分組，`end` 是每條最後一句。兩句被合併時前一句要等合併的句段結束，延遲包含下一句的長度，這就是合併的代價 |
| `first_token_latency_true_onset_ms` | 每句的**真實首字延遲**：句子是句段的第一句時（沒有被併入前一句），句段首字出現時刻 → 真實首字起點。與 VAD 無關（B5） |

### 暫定結果（`--pause-ms`）

**2026-11-01 起預設為方案 A2**（`--min-silence-ms 1000 --pause-ms 200 --provisional finalize`，結果名稱 `_sil1000_…_pause200fin`）；退回舊的單一門檻 500 ms 用 `--min-silence-ms 500 --pause-ms 0`（`--pause-ms 0` 時 `--provisional` 沒有作用）。掃描 `--min-silence-ms` 時記得一起給 `--pause-ms 0`，否則每個門檻都會開暫定結果（pause 不小於 min_silence 的組合略過）。

`--provisional finalize` 為方案 A2（停頓時 finalize 原 stream、恢復說話時才重新解碼，結果名稱加 `fin`），此時 `redecode_ms` 是補零 finalize 的耗時，`resume_redecode_ms` 是每次恢復時重新解碼的耗時。`--pause-ms N`（N < min_silence）開啟方案 A：靜音滿 N ms 先輸出暫定結果（重新解碼整句），滿 min_silence 才定稿（pipeline 模組說明）。結果 JSON 每個句段多 `provisionals`（`emitted_s`、`audio_end_s`、`text`），`metrics.provisional`：

| 欄位 | 定義 |
|---|---|
| `provisionals` / `superseded` | 暫定結果總數 / 之後同一句段又開始說話、被最終結果取代的數量（文字通常是半句） |
| `last_matches_final` | 句段最後一個暫定結果與最終結果文字相同的比例 |
| `redecode_ms` | 暫定結果時刻 − 停頓事件時已送入的音訊終點，≈ 重新解碼的計算時間 |

連續多句集另有 `segmentation.result_latency_ms_by_pause`：每句真實末字結束 → 第一個涵蓋整句的結果（暫定結果的 `audio_end_s` 不早於末字結束，或最終結果，取早者）。最終文字與同 min_silence 不開暫定結果時相同，CER 與斷句指標也相同。

### partial 穩定度（`metrics.stability`）

量化畫面上的文字跳動，程式在 `eval/stability.py`。結果 JSON 每個句段記錄每次 partial 變化的 `partials`（`emitted_s`、`text`），2026-11-02 以前的結果沒有這個欄位，`stability` 為 `null`。

**顯示文字的口徑**：每個句段依時間合成「畫面上的文字序列」——partial、暫定結果、最終結果每次發出都取代畫面上這句的文字（同一時刻依 partial → 暫定結果 → 最終結果），與偽串流比較的「可見首字」口徑一致。文字先經 CER 的同一套正規化（`tokenize`：去標點、繁轉簡、中文按字），標點的增減不算跳動；正規化後與上一次相同的事件略過。已定稿的句段不再變，所以逐句段計算、按條（wav）彙總。

| 欄位 | 定義 |
|---|---|
| `unstable_update_rate` | 修正次數 / 顯示更新次數。相鄰兩次顯示 A → B，A 不是 B 的前綴（A 已顯示的字有被改掉或刪掉）= 修正；只往後追加不算。第一次出現（空 → 第一個字）不算更新 |
| `revised_tokens_per_utterance` | 每條被改掉或刪掉的字數（每次修正 = len(A) − A 與 B 的共同前綴長度，加總），mean / P90 |
| `revised_token_ratio` | 全部被改掉的字數 / 全部最終字數（語料級）。「每顯示 100 個最終字，畫面上有幾個字被改過」 |
| `revised_token_ratio_aligned` | 同上，但每次修正改以最長共同子序列對齊：len(A) − LCS(A, B)，只算真的被換掉或刪掉的字（句中改一字，後面的字不算）。共同前綴口徑是上界、對齊口徑是下界 |
| `utterances_with_revision` | 至少有一次修正的條數比例 |
| `displays_per_utterance` | 每條畫面文字改變的次數（含第一次出現） |
| `stable_delay_ms` | **前綴穩定延遲**：最終文字的第 i 個字，穩定時刻 = 從這次顯示起、之後每次顯示的前 i+1 個字都等於最終文字前 i+1 個字的最早時刻；延遲 = 穩定時刻 − 畫面上第一次有第 i 個位置（任何字）的時刻。全部字一起算分位數 |
| `stable_on_first_ratio` | 前綴穩定延遲 = 0 的字的比例（一出現就是對的、之後沒被動過） |
| `by_replaced_kind` | 依被取代的文字是 partial 還是暫定結果，分別列被取代次數、其中修正的次數與比例、被改掉的字數。A2 暫定結果在說話恢復後被改了多少看 `provisional` 這一列 |

說明：

- 修正以**共同前綴**定義（對應串流 ASR 文獻的 unstable partial rate / unstable token rate，例如 Shangguan et al., Interspeech 2020「Analyzing the Quality and Stability of a Streaming End-to-End On-Device Speech Recognizer」；Google Cloud Speech 串流結果的 `stability` 是模型自估的「不會再變」機率，不是事後量測），因為使用者感受到的是「已出現的字被改」；中間改一個字，後面的字也算被改（畫面上它們會重畫），所以被改掉的字數是上界。
- 前綴穩定延遲的起點是「第一次在這個位置出字」，不是語音時刻；一個字出現時就對、但後來被改掉又改回來，延遲從第一次出現算起。
- 偽串流（SenseVoice 每隔一段整段重新辨識）的 partial 次數與間隔由排程決定，不同排程的「修正率」分母不同，比較時同時看被改掉的字數比例。

### 重新辨識的計算量（`asr_compute`，非串流後端才有）

偽串流（SenseVoice 定期整段重新辨識）的計算量隨句長成長，RTF 又會受同機其他工作影響。後端實作 `StreamingAsr.compute_stats()` 時，`metrics.asr_compute` 記錄暖機之後的整段辨識次數 `num_decodes`、辨識的音訊總長 `decoded_audio_s`，以及 `decoded_audio_ratio = decoded_audio_s / Σ processed_s`（與 RTF 同分母）。這個比例與機器負載無關，用來比較偽串流各設定的計算量；X-ASR 等串流後端為 `null`。

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

每個設定一個檔案：`eval/results/<日期>_<資料集>_<設定名>.json`，資料集是 manifest 所在目錄名（`aishell1_test_3000` / `aishell1_test_200` / `aishell1_test_full`），設定名為 `<模型>_sil<min_silence_ms>_pre<pre_roll_ms>_chunk<chunk_ms>_t<執行緒數>`；掃描另存總表 `<日期>_<資料集>_summary_<tag>.json`。`eval/results/` 不進 git。（2026-10-30 以前的結果檔名沒有資料集前綴，都是 200 句子集。）

| 欄位 | 內容 |
|---|---|
| `config` | 模型（含 sherpa-onnx 檔名與解碼方式）、VAD 模型與全部參數、pre-roll、chunk、執行緒數、暖機句數、manifest 路徑 |
| `environment` | commit（與是否有未提交改動）、機型、晶片、macOS 版本、Python 與套件版本 |
| `metrics` | 上述全部指標、CER 的 S / D / I / N 與 95% 信賴區間 |
| `model_files` | 每個模型檔的路徑、大小、精度 |
| `utterances` | 每句的參考、假設、正規化後文字、處理時間、每個句段的時間戳與延遲 |

## 已知限制

- VAD 起點首字延遲的起點用 VAD，不是強制對齊，VAD 本身的起點誤差（約 ±50 ms）會進入該數字；端到端首字延遲含原始音訊開頭未知長度的靜音，只適合相對比較（見上）。
- 延遲是模擬即時，假設音訊準時到達、沒有其他程序搶 CPU；真實麥克風輸入另有音訊驅動的緩衝延遲。
- 評測集只有 20 位說話人。以句重抽的信賴區間沒有反映說話人之間的差異，換一批說話人時的不確定性見以說話人重抽的區間。
- 模型訓練資料是否含 AISHELL-1 會影響 CER 的解讀（同領域 / 跨領域 / 資料洩漏），見 [models.md](models.md)。
- 3000 句佔 test set 的 42%。論文的 AISHELL-1 CER 通常是完整 7176 句，數字不能逐位對照。
- 2026-10-30 的 VAD / pre-roll 參數掃描是在 200 句子集上做的；3000 句集只跑了預設設定。
- 尾字延遲受句尾靜音長度限制：AISHELL 句尾靜音中位數約 410 ms、全部短於 800 ms，`min_silence_ms` 大於句尾靜音時句子由音訊結束（flush）收尾，量到的尾字延遲會低估即時輸入的實際值。2026-10-31 起預設後補 1500 ms 靜音修正（補的靜音加上句尾原有靜音，`min_silence_ms` 不超過約 1500 ms 時都由 VAD 收尾）；之前的結果仍有此問題。
- 同一天、同一設定名的結果會覆寫。
