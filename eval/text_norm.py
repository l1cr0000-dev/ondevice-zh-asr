"""CER 計算前的文字正規化。參考與假設都必須經過同一個 `normalize`。規則說明見 docs/eval.md。

順序：
1. NFKC：全形英數與標點轉半形（Ａ→A、１→1、，→,），相容字元轉標準形。
2. 英文轉小寫。
3. OpenCC t2s：繁體轉簡體（AISHELL 標註是簡體）。
4. Unicode 類別為標點（P*）或符號（S*）的字元換成空白。換成空白而非直接刪除，避免
   `hello,world` 黏成一個英文詞。
5. 切成 token：連續的 ASCII 英數為一個 token（一個英文詞 / 一串數字），其他非空白字元各自為
   一個 token（中文按字）。空白只用來分隔，不計入。

不做：數字與中文數字互轉（ITN）、同音字 / 異體字合併、語氣詞刪除。
"""

import re
import unicodedata

import opencc

_T2S = opencc.OpenCC("t2s")
_TOKEN = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def _strip_punct(text: str) -> str:
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)


def tokenize(text: str) -> list[str]:
    """正規化並切成計算編輯距離用的 token。"""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _strip_punct(_T2S.convert(text))
    return _TOKEN.findall(text)


def normalize(text: str) -> str:
    """正規化後以單一空白連接 token，供 jiwer 以「詞」為單位計算（= 中文按字）。"""
    return " ".join(tokenize(text))
