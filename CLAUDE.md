# CLAUDE.md

## 專案目標

在 Apple 設備上實現**離線、低延遲的端側中文串流語音辨識全鏈路**：

```
麥克風 / wav → VAD → 串流 ASR → 文字輸出 → 評測
```

用途是求職端側語音算法崗位的作品集。重點是**可複現、有量化指標**，而不是功能多。

## 範圍與階段

| 階段 | 內容 | 完成標準 |
|---|---|---|
| 1（目前） | VAD + 串流 ASR + 評測框架 | 一條命令跑完評測集，輸出 CER / RTF / 首字延遲 / 峰值記憶體 / 模型大小 |
| 2 | 量化（INT8 / FP16）、Core ML 轉換 | 與階段 1 基準逐項對比，記錄精度損失與加速比 |
| 3 | iOS demo（Swift + Core ML） | 真機上即時辨識，並量測裝置端指標 |

第一階段**不做**：量化、Core ML、iOS、自訓練模型、說話人分離、標點恢復。需要時另開討論，不要順手加。

## 硬體限制

- **只有 Apple 設備**：Apple Silicon Mac、iPhone/iPad。沒有 Android、沒有開發板、沒有 NVIDIA GPU。
- 推論後端只考慮：CPU（ONNX Runtime / sherpa-onnx）、Metal（MPS）、Core ML（CPU / GPU / ANE）。
- 不要引入需要 CUDA 的依賴，也不要建議需要非 Apple 硬體驗證的方案。

## 候選技術棧（待驗證，以評測結果為準）

- VAD：Silero VAD（ONNX）
- 串流 ASR：sherpa-onnx 串流 Zipformer / 串流 Paraformer（中文）
- 評測集：AISHELL-1 test（主），可另加自錄的噪音 / 遠場樣本
- 環境：`uv` 管理，Python 3.11

選型變更要附上對比的評測數據。

## 核心指標定義

所有指標由 `eval/` 的腳本產生，報告時必須同時記錄：**commit hash、機型與晶片、macOS 版本、執行緒數、模型與設定**。

### CER（字錯誤率）

- `CER = (S + D + I) / N`，以**字**為單位做編輯距離。S/D/I 為替換/刪除/插入數，N 為參考文本字數。
- 計算前，參考與假設都要經過**同一個**正規化函式：去標點與空白、全形轉半形、英文轉小寫、繁簡統一（統一成簡體，因為 AISHELL 標註是簡體）。正規化規則寫在程式裡並有測試。
- 報告**語料級** CER（全部錯誤數總和 / 全部字數總和），不是逐句 CER 的平均。

### RTF（實時率）

- `RTF = 處理總時間 / 音訊總時長`，小於 1 表示快於即時。
- 計時範圍：從音訊樣本送入 VAD 起，到最終結果輸出為止。**不含**模型載入；暖機（前 N 句）另外排除並註明 N。
- 要註明執行緒數（預設單執行緒，另報多執行緒）。

### 首字延遲（First Token Latency）

- 以**模擬即時**方式餵音訊：按固定 chunk 大小（例如 10 ms）、依音訊實際時長節奏送入，不能一次全部灌入。
- 定義：從**語音開始時刻**（參考時間軸上的語音起點，優先用強制對齊或 VAD 觸發點，用哪個要註明）到**第一個非空 partial 結果輸出**的牆鐘時間。
- 報告 P50 / P90 / P99，單位 ms。
- 另外報告**尾字延遲**（語音結束到最終結果輸出），作為輔助指標。

### 峰值記憶體

- 推論程序的 peak RSS，在獨立子程序中量測，避免評測框架本身的記憶體干擾。
- macOS 的 `resource.getrusage(...).ru_maxrss` 單位是 **bytes**（Linux 是 KB），換算時注意。
- 報告單位 MB。

### 模型大小

- 推論所需全部模型檔案（VAD + ASR，含 tokens/詞表）在磁碟上的總大小，單位 MB。
- 同時記錄參數量與數值精度（FP32 / FP16 / INT8）。

## 程式規範

- Python 3.11，用 `uv` 管理依賴：`uv sync` 安裝，`uv run ...` 執行。
- 全面型別標註，`uv run mypy` 要通過（strict）。
- 每個模組（`src/vad`、`src/asr`、`src/pipeline`、`eval`）至少有一個最小測試，放在 `tests/`。測試用的小音訊片段放 `tests/fixtures/`（這是唯一允許提交 wav 的地方，檔案要小）。
- `scripts/` 下的腳本都要能**一條命令**執行，參數有合理預設值，`--help` 說明清楚。
- 模型放 `models/`、資料放 `data/`，兩者都不進 git。用 `scripts/` 的下載腳本取得，並校驗 checksum。
- 串流介面統一為「逐 chunk 送入 → 回傳 partial / final 結果」，VAD 和 ASR 都遵循這個形狀，方便之後替換成 Core ML 後端。

## 常用命令

```bash
uv sync                 # 安裝依賴
uv run scripts/check_env.py  # 環境檢查（套件版本、ORT providers、麥克風）
uv run pytest           # 跑測試
uv run mypy             # 型別檢查
uv run ruff check .     # lint
uv run scripts/prepare_eval_data.py  # 下載評測集（AISHELL-1 test 隨機 3000 句）到 data/；-n 200 為冒煙子集
uv run eval/run_eval.py      # 全鏈路評測（3000 句）；--smoke 跑 200 句子集；多值參數做掃描，見 docs/eval.md
```

## 驗證紀律

- **每次改動都要能用 eval 腳本驗證。** 不能憑感覺說「變好了」或「應該更快」。
- 宣稱改進時，要給出改動前後同一設定下的指標對比表。
- 指標變差也要如實記錄，不要只挑好看的數字。
- 評測結果寫進 `eval/results/`（不進 git），重要的基準數字整理到 README 的指標表與 `docs/`。
- 提交前至少跑過 `uv run pytest`。
