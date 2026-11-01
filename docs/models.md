# 模型與資料來源

本專案用到的模型與評測資料：來源、版本、訓練資料、授權，以及這些資訊對評測結論與公開散布的影響（backlog B6）。

查證日期 2026-10-30。原則：只寫來源明文寫出的內容，附連結；來源沒寫的一律標「**未知**」，不從其他線索推測。

## 總表

| 項目 | 授權 | 訓練資料 | 含 AISHELL-1 train | 含 AISHELL-1 test |
|---|---|---|---|---|
| X-ASR-zh-en（480 / 160 ms 串流，INT8） | Apache-2.0 | 約 100 萬小時「開源及收集」語音，資料集清單未公開 | **未知** | **未知** |
| zipformer-zh-14M-2023-02-23 | Apache-2.0 | WenetSpeech（文件明寫「只用 WenetSpeech」，子集與時數未寫） | 否（依文件） | 否（依文件） |
| SenseVoice Small（非串流，INT8；2026-11-01 加入，評測對照用） | FunASR Model Open Source License v1.1（見下） | SenseVoice 整體工作「超過 40 萬小時」，Small 的資料集清單未公開 | **未知** | **未知** |
| Silero VAD v6.2.3 | MIT | 不在本次調查範圍 | – | – |
| 評測資料 AISHELL-1 test | Apache-2.0 | – | – | – |

## X-ASR-zh-en

本專案使用的檔案：sherpa-onnx release `asr-models` 中的
`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2`（SHA-256 `14919f03…`）與 2026-10-31 加入的 `sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05.tar.bz2`（SHA-256 `bdf2c0f2…`），見 `scripts/download_models.sh`。兩者是同一系列、只有 chunk 大小不同，以下來源、訓練資料與授權同時適用。參數量：README 比較表列 X-ASR-zh-en（offline）0.16B。壓縮檔內的 README 只有一句「Files in this directory are from https://github.com/Gilgamesh-J/X-ASR/tree/main」，**沒有附 LICENSE 檔**，也沒有訓練資料說明。

### 來源

| 來源 | 連結 | 查閱版本 |
|---|---|---|
| GitHub repo（README、README_zh、LICENSE、訓練 recipe） | https://github.com/Gilgamesh-J/X-ASR | commit `838297cd47`（2026-07-29） |
| Hugging Face 模型卡 | https://huggingface.co/GilgameshWind/X-ASR-zh-en | revision `689ff18c58` |
| ModelScope | https://www.modelscope.ai/Gilgamesh-J/X-ASR-zh-en | 未查閱 |
| sherpa-onnx 模型發布頁 | https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models | sherpa-onnx 官方文件沒有 X-ASR 的說明頁 |
| 論文 / 技術報告 | 無。README 標示「Coming Soon」，預計包含訓練方案與評測協議 | – |

### 訓練資料

- README 與模型卡的原文只有：「trained on approximately **1 million hours** of open-source and collected speech data」（中文版：「基于约 100 万小时开源及收集语音数据训练」）。
- **資料集清單：未知。** 沒有列出任何資料集名稱或各自時數。
- **是否含 AISHELL-1 train：未知。** **是否含 AISHELL-1 test（或 dev）：未知。** README、README_zh、模型卡都沒有出現「AISHELL」字樣。
- 「收集」資料的來源與授權：未知。
- repo 的 `X-ASR-zh-en/zipformer/multi_dataset.py` / `train.py` 有 `--use-aishell2`、`--use-tal-csasr`、`--use-librispeech` 三個開關（預設都是關，README 的訓練指令也沒有帶這些參數）。這是 icefall 多資料集 recipe 的程式碼（檔頭版權 Xiaomi 2023），**不能據此判斷實際訓練資料**；而且它指的是 AISHELL-2，不是 AISHELL-1。

### 已公布的評測

README 的評測表涵蓋 LibriSpeech、GigaSpeech、WenetSpeech（net / meeting）與 GigaSpeechBench 各垂直領域，**沒有 AISHELL-1**。因此本專案的 AISHELL-1 數字無法與官方數字對照。

### 授權

- repo 根目錄 `LICENSE` 是 Apache License 2.0 全文；README「License」一節：「This project is released under the Apache-2.0 License」。
- Hugging Face 模型卡 metadata：`license: apache-2.0`。
- 權重是否另有授權：README 與模型卡都沒有另外規定，權重與程式碼同在一個 repo、同一個 LICENSE 之下。repo 沒有 NOTICE 檔。
- 訓練資料各自的授權條款如何延伸到權重：未說明（未知）。

## zipformer-zh-14M-2023-02-23

本專案使用的檔案：sherpa-onnx release `asr-models` 中的 `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2`（SHA-256 `2cbd71b6…`）。壓縮檔內 README 的 front matter 為 `license: apache-2.0`，並寫明由 Hugging Face `csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` 以內附的 `export-onnx-zh-14M.sh` 轉出；該腳本註明原始 checkpoint 來自 `marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23`。

### 來源

| 來源 | 連結 |
|---|---|
| sherpa-onnx 文件 | https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html （`csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 (Chinese)` 一節） |
| sherpa-ncnn 文件（原始模型） | https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/zipformer-transucer-models.html （`marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23 (Chinese)` 一節） |
| Hugging Face（ONNX） | https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 （revision `204ad334e2`） |
| Hugging Face（原始 checkpoint） | https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23 （revision `caf018c3b4`） |
| 訓練程式 | https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming （sherpa-ncnn 文件所附） |
| 論文 | 沒有針對此模型的論文。架構為 icefall 的 streaming Zipformer（pruned transducer stateless7）；此模型自己的評測結果未公布 |

### 訓練資料

- sherpa-onnx 文件原文：「supports only Chinese as it is trained on the WenetSpeech corpus」；sherpa-ncnn 文件：「It is trained on the WenetSpeech corpus so it supports only Chinese」，參數量約 14M。
- 用的是 WenetSpeech 哪個子集（S / M / L）、訓練幾個 epoch：未知（兩個 Hugging Face repo 的模型卡只有 license，沒有說明文字）。
- 是否含 AISHELL-1 train / test：依上述文件「只用 WenetSpeech」，**否**。這是根據文件陳述，沒有可驗證的訓練 manifest。

### 授權

兩個 Hugging Face repo 的模型卡 metadata 都是 `license: apache-2.0`；sherpa-onnx 壓縮檔內 README 同樣是 `license: apache-2.0`。WenetSpeech 資料集本身在 OpenSLR 上標示為 CC BY 4.0（https://www.openslr.org/121/），下載需填申請表取得密碼。

## SenseVoice Small

查證日期 2026-11-01。本專案使用的檔案：sherpa-onnx release `asr-models` 中的 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`（163 MB，SHA-256 `7d1efa21…`，取自 GitHub release 頁面的 digest），解壓後 `model.int8.onnx` 228 MB、`tokens.txt`。同一 release 另有 `…-int8-2025-09-09` 版，未採用（與原始 SenseVoiceSmall 的差異未查）。壓縮檔內 README 只寫「converted from https://github.com/FunAudioLLM/SenseVoice」；LICENSE 只有一行「Ref to https://github.com/modelscope/FunASR?tab=readme-ov-file#license」。

| 來源 | 連結 |
|---|---|
| 原始模型 | https://github.com/FunAudioLLM/SenseVoice （程式碼 MIT；權重依各模型卡，官方 SenseVoiceSmall 模型卡連到 FunASR MODEL_LICENSE） |
| 模型授權 | https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE （FunASR Model Open Source License Agreement v1.1） |

### 訓練資料

- SenseVoice repo 只寫整體工作「training on more than 400,000 hours」，沒有對應到釋出的 Small checkpoint，也沒有資料集清單。
- AISHELL-1 / AISHELL-2 只出現在評測比較中（結果是圖片），沒有說是否用於訓練。是否含 AISHELL-1 train / test：**未知**。與 X-ASR 相同，AISHELL-1 上的 CER 不能當作泛化能力。

### 授權

FunASR Model Open Source License v1.1：可使用、複製、修改、分享（含微調與衍生模型），須標明來源與作者並保留模型名稱；對 Alibaba 或軟體的「無根據的詆毀或惡意攻擊」會自動失去授權；軟體「僅供參考與學習」、不負責損失；協議可修訂、公布即生效；準據法欄位是未填的範本（「[Country/Region]」）。**未明文提到商業用途**。本專案只用於評測對照，模型檔不進 git；若之後要預設採用或在 iOS app 內散布，需再確認這份授權。

#### 商用限制查證（2026-11-01，偽串流比較時補查）

逐項重讀第一手來源：

| 來源 | 內容 |
|---|---|
| `MODEL_LICENSE`（FunASR main 分支，v1.1） | 全文**沒有出現「commercial / 商業」**，既不允許也不禁止。2.1 授權「use, copy, modify, and share」；2.2 須標明來源與作者、保留模型名稱；3 稱軟體「for reference and learning purposes only」、風險自負；4.2 詆毀 / 惡意攻擊即失去授權；5 違反任何條款自動終止；6 Alibaba 可單方修訂、公布即生效、繼續使用視為接受；7 準據法與版權年份（「[2023-2028]」）是未填的範本 |
| Hugging Face `FunAudioLLM/SenseVoiceSmall` 模型卡 | metadata `license: other`、`license_name: model-license`（連到上面的 MODEL_LICENSE），內文沒有提商用 |
| FunASR README「License」 | 工具程式碼 MIT；預訓練權重另依各模型卡，模型卡連到 MODEL_LICENSE 時適用該協議 |
| SenseVoice GitHub issue #287（2026-05-11 提問、2026-05-20 關閉） | 有人問「權重是 MIT 還是 FunASR Model License、能否商用、需否標示」。維護者帳號 LauraGPT 回覆：SenseVoice 權重適用 FunASR Model License，「Commercial use is permitted」，建議在文件或 about 頁標示 FunASR / SenseVoice。https://github.com/FunAudioLLM/SenseVoice/issues/287 |

**結論：**

- 商用目前**可以做，但依據只是一則 issue 回覆**，不是授權文字本身。協議文字沒有商用條款，又寫「僅供參考與學習」，且 Alibaba 可單方修訂、立即生效，準據法未填。對比 X-ASR / zh-14M / Silero 的 Apache-2.0 / MIT，這是**明顯較弱的法律基礎**。
- 實際條件：散布（例如打包進 iOS app）時要標明 FunASR / SenseVoice 來源與作者、保留模型名稱 SenseVoiceSmall；不可有 4.2 所述行為；要追蹤 MODEL_LICENSE 的修訂。
- 正式商用前應取得書面確認（郵件或正式授權），並保存 issue #287 回覆與當時 MODEL_LICENSE 的快照。
- 訓練資料未公開，資料授權如何延伸到權重同樣未知。
- 對本專案的影響：偽串流（SenseVoice）只**保留為選項，預設維持 X-ASR**（Apache-2.0）。授權是不改預設的理由之一（另見 progress.md 2026-11-01「偽串流 vs 真串流」）。

## Silero VAD v6.2.3

授權 MIT（https://github.com/snakers4/silero-vad ，tag `v6.2.3`）。VAD 不影響 CER 的領域判斷，訓練資料不在本次調查範圍。

## 評測資料：AISHELL-1 test

- 官方：OpenSLR SLR33（https://www.openslr.org/33/ ），標示「License: Apache License v.2.0」。
- 本專案實際下載自鏡像 Hugging Face `shenyunhang/AISHELL-1`（revision `2724409d53`），模型卡 metadata `license: apache-2.0`。這是第三方鏡像，見 [eval.md](eval.md)。
- `data/` 不進 git，本專案不散布 AISHELL 音訊。

## 對評測結論的影響

- **X-ASR 在 AISHELL-1 test 的 CER 不能當作泛化能力的證據。** 訓練資料是否含 AISHELL-1 train（同領域）或 test（資料洩漏）都未知，因此無法判斷這個數字屬於同領域、跨領域，還是被洩漏的資料拉低。報告時要寫成「AISHELL-1 test 上的 CER，訓練資料是否含 AISHELL-1 未知」，不寫「跨領域」，也不和只用其他資料訓練的模型直接比泛化能力。
- **zh-14M 依文件只用 WenetSpeech 訓練**，AISHELL-1 對它是跨領域。兩個模型的 CER 差距因此混了「模型容量 / 訓練量」與「可能的同領域優勢」兩個因素，無法拆開。
- 要得到不受此問題影響的泛化數字，需要確定沒有進入任何模型訓練資料的評測集，例如自錄音訊（B4 的合成長音訊集若以自錄句子為來源即可滿足）。

## 公開到 GitHub 當作品集的授權評估

範圍：本 repo 的程式碼、文件、`scripts/download_models.sh`、`scripts/demo_*.py`，以及 git 中已提交的檔案。以下是依各來源條款文字做的整理，不是法律意見。

| 內容 | 結論 | 依據 |
|---|---|---|
| 自己寫的程式碼與文件 | 可以，需自選授權 | 目前 README 寫「License: TBD」，repo 沒有 LICENSE 檔 |
| `download_models.sh` | 可以 | 腳本只含上游 URL 與 SHA-256，執行時才從上游下載，本 repo 不散布權重。即使將來直接散布權重（例如打包進 iOS app），Apache-2.0 / MIT 也允許，條件是附上授權全文並保留版權聲明 |
| demo 腳本 | 可以 | 在使用者本機下載並執行模型，不涉及散布 |
| 評測資料 | 可以（未散布） | `data/`、`eval/results/` 都在 `.gitignore`；AISHELL-1 本身也是 Apache-2.0 |
| 文件中引用的 AISHELL 句子 / 辨識結果片段 | 可以 | Apache-2.0 允許重製，附來源即可 |
| **`tests/fixtures/vad_two_utterances.wav`** | **不建議公開** | 見下 |

**阻礙：測試 fixture 是 macOS 系統語音合成的音訊。** `vad_two_utterances.wav` 由 `tests/fixtures/make_vad_fixture.py` 呼叫 macOS `say`（語音 Tingting）產生，已提交進 git（commit `ea86c17` 起）。本機 macOS 軟體授權（`/Library/Documentation/License.lpdf`，第 2 條 F 款「Voices; Live Captions」）規定系統語音只能用於個人、非商業用途，並明文排除在「public sharing」情境下「publishing or redistribution」系統語音。公開 GitHub repo 屬於這種情境。處理方式（擇一）：

1. 改用自己錄的兩句話（授權自己決定），或從 AISHELL-1 取兩句拼接（Apache-2.0，附來源）。fixture 的 JSON 記錄真實起止時間，換成真人錄音後要重新標註或改用「修掉頭尾靜音後放在已知位置」的同一套做法。
2. 已在 git 歷史中：只刪檔不夠，公開前要改寫歷史（例如 `git filter-repo`）或以乾淨的歷史另開公開 repo。

**其他公開前要補的事項：**

- 新增 LICENSE（自選，例如與上游一致的 Apache-2.0 或 MIT），並把 README 的「License: TBD」改成實際授權與第三方授權清單（可連到本文件）。
- 公開的作品集只描述作者自己的工作；模型名稱、來源與授權如實標示，不要寫成自己訓練的模型。

**結論：** 授權面允許公開本專案（含下載腳本與 demo）；三個模型都是寬鬆授權，且 repo 不散布權重與資料。唯一需要先處理的是 macOS 系統語音產生的測試 fixture。X-ASR 訓練資料未公開，影響的是**評測結論的解讀**，不影響散布。若將來商業使用，X-ASR 「收集」資料的來源不明，是需要另外確認的風險。
