# 問題與待辦

持續更新的清單：已知問題、要做的優化、優先順序。做完的項目移到最下方「已完成」，並在 [progress.md](progress.md) 寫實驗紀錄。數字出自 2026-10-30 的 AISHELL-1 掃描（commit `f576d57`），細節見 progress.md。

**目前狀態（2026-10-31）**：斷句 / 尾字延遲的診斷完成（B3 已解決，B9 有數據，見 progress.md 與 [scoreboard.md](scoreboard.md)），下一步建 B4 連續多句評測集，再在其上比較結束判定方案。
**2026-10-30**：第一階段全鏈路已完成（tag `v1`，commit `f6b11a8`），開始逐項處理下列問題。麥克風路徑直接使用 `StreamingPipeline`，對 pipeline 的改動（VAD 預先啟動、斷句）會自動套用到麥克風。

優先級：**P0** 影響核心目標或結論的可信度｜**P1** 影響品質，該做｜**P2** 有空再做

## 建議處理順序

1. **B13** 公開前替換 macOS 系統語音產生的測試 fixture（要公開 repo 時才必要）
2. **B4 + B5** 評測可信度：合成連續多句音訊集、與 VAD 無關的首字起點（之後所有斷句優化都靠它驗證；B3 已於 2026-10-31 以句尾補靜音解決）
3. **B1** 首字延遲：160 ms 對比與字級拆解已完成（2026-10-31，主因是模型輸出延遲）；blank_penalty 掃描完成，預設維持 penalty 0（B2 不實作）。推論端已無明顯可降空間，延遲線在第一階段告一段落
4. **B9** 結束判定方案（2026-10-31 診斷完成，在 B4 上比較方案，結果記在 scoreboard.md）
5. **B0** 麥克風真實端到端延遲
6. 第二階段前：**B7** FP32 參照、**B11** 記憶體與模型體積
7. 有空再做：B8、B10、B12

---

## 延遲

### B0　麥克風路徑的真實端到端延遲（P1）

- **現況**：`scripts/demo_mic.py` 已完成（v1），能連續說多句並即時顯示 partial 與每句首字延遲。但所有正式延遲數字都是「模擬即時」，不含音訊驅動的緩衝延遲，也假設沒有其他程序搶 CPU；麥克風的牆鐘延遲沒有與模擬值正式對比過。
- **做法**：用固定音訊經喇叭播放或虛擬音訊裝置（例如 BlackHole）回送到輸入，同一段音訊同時跑模擬與真實路徑，比較首字延遲分佈。
- **完成標準**：模擬 vs 真實的首字延遲對比表，並說明差距來源（PortAudio 緩衝、排程抖動）。

### B1　首字延遲約 1 秒（P0）

- **現況**：X-ASR 480 ms，首字延遲（VAD 起點）mean 972 / P90 1029 ms，端到端 mean 1487 ms。
- **2026-10-31 已做 X-ASR 160 ms 對比**（progress.md）：端到端首字延遲平均只快 128 ms，CER +0.94 個百分點（顯著），RTF 2.2 倍，預設維持 480 ms。拆解顯示 VAD start 事件後模型平均還要 554 ms 才出第一個字（160 ms 版），瓶頸是模型輸出第一個字的時機，不是 chunk 等待或 VAD。
- **2026-10-31 字級對齊拆解**（300 句，MFA；[latency-analysis.md](latency-analysis.md)）：主因是模型的輸出延遲，模型在首字唸完 190 ms（480）/ 308 ms（160）後才輸出它；VAD 切點準確（比真實起點早 26 ms）；其餘是 chunk + 130 ms 右側 context。
- **2026-10-31 blank_penalty 掃描**（300 句）：penalty 2 首字 −40 / −47 ms（480 / 160），CER 差異不顯著；penalty 4 −146 / −181 ms，CER +0.9 個百分點（顯著）；≥ 6 失控。
- **決定（2026-10-31）**：預設維持 480 ms、penalty 0。不採用 penalty 2：只省約 40 ms（約首字延遲的 4%，感受不到），300 句的 CER 信賴區間仍容許最多約 +0.5 個百分點；參數保留為選項（`--blank-penalty`）。
- **之後**：再往下降需要用 CER 換，或重新訓練（第一階段不做）。penalty 2 的 3000 句驗證有空再做（P2）。

### B2　VAD 預先啟動（P2，原 P1；依 2026-10-31 結果降級）

- **現況**：VAD 要等語音持續 min_speech_ms（250 ms）才發出「開始」，平均 291 ms。原本推測換小 chunk 後會成為瓶頸，**數據不支持**：160 ms 與 480 ms 模型在 start 事件後 100 ms 內出字的比例都是 0–0.1%，VAD 的等待被模型的出字延遲蓋住。
- **做法**：「預先啟動」：VAD 一看到語音 frame（PENDING 狀態）就建立 ASR stream 開始解碼，partial 先不輸出或標記為暫定；若最後判定為雜訊就丟棄。
- **2026-10-31 上限實驗**（progress.md）：`--min-speech-ms 0`（VAD 確認 291 → 65 ms）下，X-ASR 160 / 480 ms 的首字延遲、CER 完全不變。**不實作**，除非換模型後首字確實卡在 VAD。
- **完成標準**：先確認在目標模型上首字確實卡在 VAD（start 事件當下即出字的比例明顯 > 0），再做；量首字延遲降低的量，以及誤觸發時是否會輸出錯字（B4 的噪音集）。

### B3　尾字延遲量不準（已解決，2026-10-31）

- 每條評測音訊後補 1.5 s 數位靜音（`run_eval.py --tail-silence-ms`，預設 1500），句子改由 VAD 靜音計時收尾。3000 句預設設定尾字延遲 451 → 531 ms；以 MFA 真實末字結束點起算為 580 ms（P90 651）。見 progress.md。

---

## 評測可信度

### B4　評測集太乾淨，測不到 VAD 的真實難處（P0）

- **現況**：AISHELL 是安靜環境的單句朗讀。沒有長靜音、背景噪音、連續多句，VAD 的誤觸發與句間切分完全沒測到。
- **做法**：由 AISHELL 句子合成長音訊：多句以不同長度的靜音（例如 0.2–2 s）串接、句尾留足靜音，再以不同 SNR 混入噪音（需選一個授權清楚的噪音集）。句子位置由合成過程精確已知，同時解決 B3 與 B5。
- **完成標準**：`prepare_eval_data.py` 可產生長音訊集，`run_eval.py` 在上面報告句間切分正確率、誤觸發數、真實尾字延遲。
- **部分完成（2026-10-31）**：`scripts/prepare_continuous_data.py` 由 300 句對齊集合成連續多句集（77 條，句間停頓 0.2–2 s 分檔），`run_eval.py` 報告句間切開率、句中誤切、每句真實尾字 / 首字延遲（scoreboard.md）。**未做**：噪音混入（需選授權清楚的噪音集）、誤觸發數、更大規模（目前只有 300 句，受 MFA 對齊集限制）。

### B5　首字延遲的起點來自 VAD 本身（P0）

- **現況**：起點是 VAD 判定的語音起點（含 30 ms padding），VAD 切得晚時首字延遲反而「看起來比較好」，等於用被測系統的一部分當尺。
- **做法**：改用與 VAD 無關的參考起點。B4 合成音訊中句子的位置精確已知；單句集可用能量法或強制對齊估計起音。保留 VAD 起點當作輔助欄位。
- **部分完成（2026-10-31）**：端到端首字延遲（前補 1 秒靜音）與 VAD 無關但只適合相對比較；300 句 MFA 字級對齊集（`scripts/prepare_alignment.py`）提供真實首字起點，VAD 起點平均早 26 ms（P90 0–2 ms）。對齊誤差待人工抽查；完整版仍待 B4 的合成長音訊集。
- **完成標準**：docs/eval.md 的定義更新，並報告兩種起點的差距分佈。

### B6a　評測集與任何模型訓練資料都不重疊（P1，原 B6 的後續）

- **現況**：B6 查證結果（[models.md](models.md)）：X-ASR 是否用過 AISHELL-1 train / test **未知**，官方只說「約 100 萬小時開源及收集語音」。AISHELL-1 上的 X-ASR CER 無法當作泛化能力。
- **做法**：B4 的合成長音訊集改用自錄句子（或其他確定不在訓練資料中的語料）為來源，作為「保證跨領域」的評測集。X-ASR 技術報告（README 標示 Coming Soon）公布後再查一次訓練資料清單。

### B7　第一階段缺少 FP32 基準（P1）

- **現況**：CLAUDE.md 規定第一階段以 FP32 為基準，但 X-ASR 用的是官方 INT8 版（FP32 encoder 566 MB）。第二階段做量化對比時缺少 FP32 參照。
- **做法**：下載 FP32 版跑一次評測，只用來當參照，不必當預設。

### B8　其他評測缺口（P2）

- ~~多執行緒 RTF（CLAUDE.md 要求另報）~~ **2026-11-01 已量**（progress.md「X-ASR onnxruntime 後端調校」，200 / 300 句）：`x-asr-480ms-int8-ort` 2 / 4 執行緒沒有加速（負載下 RTF +21% / +55%，關 spinning 後 2 執行緒持平），停頓時的 finalize 與恢復時的重新解碼變慢，且會改變 1–5% 句子的結果；預設維持單執行緒。ORT session 選項（圖最佳化等級、arena、mem pattern、prepacking、denormal）都沒有收益。**未做**：量測時有其他程序佔用 CPU，閒置時重量執行緒 1 / 2 / 4 的 RTF。
- X-ASR 參數量（可用 `onnx` 套件統計 initializer）。
- 正式評測集已改為 3000 句（CER 信賴區間以句重抽約 ±0.27 個百分點，X-ASR）。VAD / pre-roll 掃描仍是 200 句子集上的結果，之後的參數選擇要在 3000 句集上重做。
- 評測音訊來自第三方鏡像（`shenyunhang/AISHELL-1`），無法對官方 checksum 驗證。
- 只在一台 Apple M5 Mac、CPU 上驗證過；其他 Apple Silicon 機型的數字未知。

---

## VAD 與斷句

### B9　單一靜音閾值無法同時切對句中停頓與句間停頓（P1）

- **現況**：min_silence 500 ms 時，仍有 16% 的單句音訊被切成多段；而自錄音訊中句間停頓可短至 290 ms，會被合併。純聲學 VAD 的固有限制。
- **2026-10-31 診斷**（progress.md、[scoreboard.md](scoreboard.md)）：
  - 切段的代價全在被切開的句子上：500 ms 時被切開的 15% 句子 CER 8.4%，其餘 3.2%；門檻 500 → 1000 ms，CER 4.11 → 3.70%（−0.41 個百分點，顯著），尾字延遲 +530 ms。門檻每 +100 ms，最終結果晚 100 ms。
  - VAD 結束事件時 partial 只有 1.4–3% 的句子已完整（平均少 1.65 字）；不 finalize、只送靜音要約 1.1 s 才完整。「partial 不再變化就提早結束」無法直接降低尾字延遲。
  - 句中停頓中位數 320 ms，自由說話的句間停頓（288、448 ms，僅 5 個樣本）落在同一範圍。AISHELL 量不到合併的代價，方案比較需要 B4。
- **2026-10-31 方案 A（暫定結果 + 長門檻定稿，`--pause-ms`，預設關閉）**：pause 200 / min_silence 1000 時 CER 4.11 → 3.70%（最終文字與 min_silence 1000 相同），單句真實尾字的結果延遲 580 → 409 ms，連續集 0.3 / 0.5 s 停頓不再要等數秒。代價：35% 的句子先出現半句的暫定結果再被修正；重新解碼 148–239 ms、RTF 約 2–3 倍。**2026-11-01 方案 A2**（`--provisional finalize`）：單句結果延遲 299 ms（P90 366）、RTF 約為 A 的一半，但最終文字偶爾漏句尾的字，CER 3.75%（相對 A +0.05 個百分點，顯著；相對單一門檻 500 ms −0.36）。**2026-11-01 改為預設**（`PipelineConfig()` 與各腳本；`--min-silence-ms 500 --pause-ms 0` 退回舊設定），見 progress.md。
- **2026-11-01 短停頓**（progress.md、scoreboard.md「B9 短停頓」）：A2 的 pause 200 實際要 224 ms（32 ms frame 進位），0.2 s 句間停頓有 39% 抓不到。pause 150（160 ms）讓連續集 0.2 s 停頓後的結果延遲 1.3 s → 0.23 s、其餘句子快約 60 ms；代價是半句暫定結果的句子 36% → 40%、解碼量 +3.5–8%，CER 不變。調 neg_threshold 幾乎沒幫助。`--pause-ms 150` 可用，是否改預設待決定（建議先在閒置時重量 RTF）。
- **原做法**：把「VAD 決定送不送音訊」與「何時輸出最終結果」分開：VAD 只過濾長靜音，斷句改由 ASR 端的 endpoint 規則判斷（結合已解碼內容與尾端靜音，sherpa-onnx 有內建規則）。也可先試 `threshold` / `neg_threshold`。
- **完成標準**：在 B4 的長音訊集上，切段正確率與 CER 都不低於目前設定。

### B10　VAD 小工具（P2）

- `demo_vad.py` 加 `--show-probs`，印出逐 frame 機率時間軸，方便調參。
- 補一組有背景噪音（電視、風扇）的自錄樣本，檢查誤觸發（可併入 B4）。

---

## 資源佔用（第二階段處理）

### B11　記憶體與模型體積偏重（P1）

- **現況**：模型 163.6 MB 對 iOS app 也偏大，對 iPhone 端側部署是明顯風險。2026-10-30 用 `scripts/probe_memory.py` 查過（細節見 progress.md）：
  - 直譯器啟動 21 MB、import numpy / sherpa-onnx 後 38 MB；X-ASR 載入後 472 MB（**載入增加約 435 MB**），解碼 1000 句後 491 MB。增加幾乎全在載入階段。
  - `EnableCpuMemArena=0`（經 sherpa-onnx provider 設定檔）：載入與峰值都在 ±1 MB 內，**無效**。關 prepacking 只少約 7 MB；關圖最佳化反而多 144 MB。
  - 同樣三個檔案直接用 onnxruntime 1.31.1 載入只增加約 245 MB。**已查明**（vmmap，見 progress.md）：多出的約 190 MB 不是重複的權重，存活配置量兩邊相同（219 vs 205 MB），差別是 sherpa-onnx 2.0.1 內附的 ORT 1.28.2 在載入時留下約 183 MB 已釋放但未歸還的 malloc 碎片。onnxruntime 1.28.0 / 1.29.0 直接載入可重現（碎片 192 / 188 MB），1.30.0 起降到約 34 MB。sherpa-onnx 2.0.1 已是 PyPI 最新版。
  - 評測時峰值 RSS 隨句數增加（X-ASR 200 句 531 MB → 3000 句 618 MB；zh-14M 252 → 333 MB），但單獨解碼 100 → 1000 句只增加 3 MB，增加的部分在評測 / pipeline 端，原因未查。這讓「峰值 RSS」指標混入了評測框架本身，違背 CLAUDE.md 的量測原則。
- **2026-11-01 SenseVoice**（`scripts/probe_footprint.py`，progress.md）：經 sherpa-onnx 載入後 footprint 850 MB（X-ASR 270），其中約 500 MB 是 ORT 1.28 的載入開銷；同一檔案用 ORT 1.31.1 直接載入只要 +308 MB。改用 ORT ≥ 1.30 後 SenseVoice 與 X-ASR 的差距估計從約 580 MB 縮到約 60 MB（未含推論，推論時約 +3 MB / 秒音訊）。
- **做法**：(1) 等 sherpa-onnx 改用 ORT ≥ 1.30，或自行以較新的 ORT 編譯 sherpa-onnx，再用 `probe_memory.py` 對比；iOS 端以 phys_footprint（jetsam 依據）而非 RSS 為準；(2) 在 worker 中每 N 句取樣 RSS，找出評測端的增長來源並修正，讓峰值 RSS 與句數無關；(3) 第二階段量化與 Core ML 轉換時一併對比記憶體與體積。

---

## 工程

### B12　雜項（P2）

- 沒有 CI 自動跑 pytest / mypy / ruff。
- 專案沒有安裝成套件，`scripts/` 與 `eval/run_eval.py` 都靠 `sys.path.insert` 才能 import `src`。根治要改成可安裝套件（頂層名稱 `src` 需改名），改動面大。
- 沒有 git remote，README 的 `<repo-url>` 待補；本專案自己的 License 未定（第三方授權已查清，見 [models.md](models.md)）。

### B13　公開前替換測試 fixture（P1，公開 repo 前必做）

- **現況**：`tests/fixtures/vad_two_utterances.wav` 由 macOS `say`（Tingting）合成，自 `ea86c17` 起已在 git 歷史中。macOS 軟體授權第 2 條 F 款排除在公開分享情境中發布或散布系統語音，公開 GitHub repo 屬於此情境。
- **做法**：改用自錄音訊或 AISHELL-1 句子（Apache-2.0，附來源）重做 fixture 與 JSON 標註，更新 `make_vad_fixture.py`；公開前改寫歷史或另開乾淨的公開 repo。同時新增 LICENSE、補上 `<repo-url>`。
- **完成標準**：git 歷史中沒有系統語音產生的音訊，`uv run pytest` 全部通過。

---

## 已完成

- 2026-10-30　VAD `min_silence_ms` 100 → 500、pre-roll 維持 100（AISHELL-1 掃描，見 progress.md）。
- 2026-10-30　麥克風輸入 `scripts/demo_mic.py`（原 B0 的 demo 部分，真實延遲量測留在 B0）。
- 2026-10-30　第一階段整理：腳本錯誤處理、共用 CLI 參數與常數、`ruff format` 全部通過（見 progress.md）。
- 2026-10-30　**B6** X-ASR / zh-14M 訓練資料與授權查證，寫進 [models.md](models.md) 與 `download_models.sh` 註解；公開散布的授權評估（可公開，唯一阻礙見 B13）。
- 2026-10-30　正式評測集改為 AISHELL-1 test 隨機 3000 句，CER 報告 95% bootstrap 信賴區間（以句 / 以說話人）；200 句子集保留為 `--smoke`（原 B8 的一項）。
- 2026-10-31　**B3** 尾字延遲量測修正：評測音訊後補 1.5 s 靜音（見 progress.md）。
