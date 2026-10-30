# 問題與待辦

持續更新的清單：已知問題、要做的優化、優先順序。做完的項目移到最下方「已完成」，並在 [progress.md](progress.md) 寫實驗紀錄。數字出自 2026-10-30 的 AISHELL-1 掃描（commit `f576d57`），細節見 progress.md。

**目前狀態（2026-10-30）**：第一階段全鏈路已完成（tag `v1`，commit `f6b11a8`），開始逐項處理下列問題。麥克風路徑直接使用 `StreamingPipeline`，對 pipeline 的改動（VAD 預先啟動、斷句）會自動套用到麥克風。

優先級：**P0** 影響核心目標或結論的可信度｜**P1** 影響品質，該做｜**P2** 有空再做

## 建議處理順序

1. **B13** 公開前替換 macOS 系統語音產生的測試 fixture（要公開 repo 時才必要）
2. **B5 + B4 + B3** 評測可信度：合成長音訊集、與 VAD 無關的首字起點、真實尾字延遲（之後所有優化都靠它驗證）
3. **B1** 首字延遲：X-ASR 160 ms 對比；接著 **B2** VAD 預先啟動
4. **B9** 斷句改由 ASR endpoint 判斷（在 B4 的長音訊集上驗證）
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

- **現況**：X-ASR 480 ms，首字延遲 mean 990 / P90 1029 ms。計算每個 chunk 約 13 ms，延遲幾乎全是在等 chunk 收滿（第一個 chunk 約 0.61 s，之後每 0.48 s 一次）。換 Core ML / ANE 對此無幫助。
- **做法**：在同一評測集上比較 X-ASR 160 ms 版（有 INT8 版）。要重新試句尾補零長度，並重新掃描 pre-roll（最佳值跟 chunk 大小有關）。
- **完成標準**：160 ms 與 480 ms 的 CER / 首字延遲對比表，依結果決定預設模型。

### B2　VAD 確認延遲將成為瓶頸（P1，依賴 B1）

- **現況**：VAD 要等語音持續 min_speech_ms（250 ms）才發出「開始」，平均 291 ms。目前被 480 ms 的 chunk 蓋住，換小 chunk 後會露出來。
- **做法**：「預先啟動」：VAD 一看到語音 frame（PENDING 狀態）就建立 ASR stream 開始解碼，partial 先不輸出或標記為暫定；若最後判定為雜訊就丟棄。
- **完成標準**：首字延遲降低的量，以及誤觸發時是否會輸出錯字（要在 B4 的噪音集上驗證）。

### B3　尾字延遲量不準（P1）

- **現況**：AISHELL 句尾靜音中位數約 410 ms、全部短於 800 ms，min_silence ≥ 500 ms 時句子多半由音訊結束收尾，量到的尾字延遲被截斷（500 / 800 ms 都量到約 450 ms）。即時輸入時應約為 min_silence + 40 ms。
- **做法**：併入 B4 的長音訊評測集。

---

## 評測可信度

### B4　評測集太乾淨，測不到 VAD 的真實難處（P0）

- **現況**：AISHELL 是安靜環境的單句朗讀。沒有長靜音、背景噪音、連續多句，VAD 的誤觸發與句間切分完全沒測到。
- **做法**：由 AISHELL 句子合成長音訊：多句以不同長度的靜音（例如 0.2–2 s）串接、句尾留足靜音，再以不同 SNR 混入噪音（需選一個授權清楚的噪音集）。句子位置由合成過程精確已知，同時解決 B3 與 B5。
- **完成標準**：`prepare_eval_data.py` 可產生長音訊集，`run_eval.py` 在上面報告句間切分正確率、誤觸發數、真實尾字延遲。

### B5　首字延遲的起點來自 VAD 本身（P0）

- **現況**：起點是 VAD 判定的語音起點（含 30 ms padding），VAD 切得晚時首字延遲反而「看起來比較好」，等於用被測系統的一部分當尺。
- **做法**：改用與 VAD 無關的參考起點。B4 合成音訊中句子的位置精確已知；單句集可用能量法或強制對齊估計起音。保留 VAD 起點當作輔助欄位。
- **完成標準**：docs/eval.md 的定義更新，並報告兩種起點的差距分佈。

### B6a　評測集與任何模型訓練資料都不重疊（P1，原 B6 的後續）

- **現況**：B6 查證結果（[models.md](models.md)）：X-ASR 是否用過 AISHELL-1 train / test **未知**，官方只說「約 100 萬小時開源及收集語音」。AISHELL-1 上的 X-ASR CER 無法當作泛化能力。
- **做法**：B4 的合成長音訊集改用自錄句子（或其他確定不在訓練資料中的語料）為來源，作為「保證跨領域」的評測集。X-ASR 技術報告（README 標示 Coming Soon）公布後再查一次訓練資料清單。

### B7　第一階段缺少 FP32 基準（P1）

- **現況**：CLAUDE.md 規定第一階段以 FP32 為基準，但 X-ASR 用的是官方 INT8 版（FP32 encoder 566 MB）。第二階段做量化對比時缺少 FP32 參照。
- **做法**：下載 FP32 版跑一次評測，只用來當參照，不必當預設。

### B8　其他評測缺口（P2）

- 多執行緒 RTF（CLAUDE.md 要求另報）。
- X-ASR 參數量（可用 `onnx` 套件統計 initializer）。
- 正式評測集已改為 3000 句（CER 信賴區間以句重抽約 ±0.27 個百分點，X-ASR）。VAD / pre-roll 掃描仍是 200 句子集上的結果，之後的參數選擇要在 3000 句集上重做。
- 評測音訊來自第三方鏡像（`shenyunhang/AISHELL-1`），無法對官方 checksum 驗證。
- 只在一台 Apple M5 Mac、CPU 上驗證過；其他 Apple Silicon 機型的數字未知。

---

## VAD 與斷句

### B9　單一靜音閾值無法同時切對句中停頓與句間停頓（P1）

- **現況**：min_silence 500 ms 時，仍有 16% 的單句音訊被切成多段；而自錄音訊中句間停頓可短至 290 ms，會被合併。純聲學 VAD 的固有限制。
- **做法**：把「VAD 決定送不送音訊」與「何時輸出最終結果」分開：VAD 只過濾長靜音，斷句改由 ASR 端的 endpoint 規則判斷（結合已解碼內容與尾端靜音，sherpa-onnx 有內建規則）。也可先試 `threshold` / `neg_threshold`。
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
  - 同樣三個檔案直接用 onnxruntime Python 載入，只增加約 245 MB（encoder 148 MB 檔案 → +217–222 MB）。經 sherpa-onnx 載入多出的約 190 MB 是下一個要查的對象（推測：sherpa-onnx 先把檔案讀進記憶體再建 session，緩衝未釋放；**未驗證**）。
  - 評測時峰值 RSS 隨句數增加（X-ASR 200 句 531 MB → 3000 句 618 MB；zh-14M 252 → 333 MB），但單獨解碼 100 → 1000 句只增加 3 MB，增加的部分在評測 / pipeline 端，原因未查。這讓「峰值 RSS」指標混入了評測框架本身，違背 CLAUDE.md 的量測原則。
- **做法**：(1) 以 C++ 或 sherpa-onnx C API 對照載入，確認多出的 190 MB 來源；(2) 在 worker 中每 N 句取樣 RSS，找出評測端的增長來源並修正，讓峰值 RSS 與句數無關；(3) 第二階段量化與 Core ML 轉換時一併對比記憶體與體積。

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
