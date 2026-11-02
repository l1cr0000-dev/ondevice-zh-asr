"""partial 穩定度：量化畫面上的文字跳動（使用者看到的字被改了多少次、改了多少）。

口徑：每個句段依時間合成「畫面上的文字序列」——partial、暫定結果、最終結果每次發出時都取代畫面上這句的文字（與偽串流比較的「可見首字」口徑一致：三者都算看得到）。已定稿的句段不再變，
所以逐句段計算、按條（wav）彙總。文字先經 `eval.text_norm.tokenize`（與 CER 同一套正規化，去標點、
繁轉簡），標點或空白的變化不算跳動；正規化後與上一次相同的事件略過。

- 修正（unstable update）：相鄰兩次顯示 A → B，A 不是 B 的前綴，也就是 A 已顯示的字有被改掉或刪掉。
  只往後追加不算修正。被改掉的字數 = len(A) − 共同前綴長度（畫面上要重畫的字，上界）；另報以最長共同
  子序列對齊的字數 len(A) − LCS(A, B)（真的被換掉或刪掉的字，下界）。
- 前綴穩定時間：最終文字的第 i 個字，「穩定」= 從某次顯示起，之後每次顯示的前 i+1 個字都等於最終文字的
  前 i+1 個字；延遲 = 穩定時刻 − 畫面上第一次有第 i 個位置（任何字）的時刻。0 = 一出現就是對的、之後不變。
- 依被取代的文字是 partial 或暫定結果分開統計，暫定結果（A2）被說話恢復後的文字取代時改了多少就在這裡。

參考：串流 ASR 文獻的 unstable partial rate / unstable token rate（Shangguan et al., Interspeech 2020），
這裡以共同前綴定義，與「畫面上已出現的字被改掉」直接對應。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eval.metrics import summarize
from eval.text_norm import tokenize

KINDS = ("partial", "provisional", "final")


@dataclass(frozen=True)
class Display:
    """畫面上這句的一次顯示：發出時刻（原始音訊時間軸，秒）、來源、正規化後的 token。"""

    emitted_s: float
    kind: str
    tokens: tuple[str, ...]


def common_prefix(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """最長共同子序列長度（句子短，O(len(a)·len(b)) 即可）。"""
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = row[j + 1]
            row[j + 1] = prev + 1 if x == y else max(row[j + 1], row[j])
            prev = cur
    return row[-1]


def display_sequence(seg: dict[str, Any]) -> list[Display]:
    """一個句段的顯示序列（依時間排序；同一時刻依 partial → 暫定結果 → 最終結果）。

    `seg` 需有 `partials`（`emitted_s`、`text`）；`provisionals` 與最終結果（`final_s`、`text`）可有可無。
    """
    events: list[tuple[float, int, str, str]] = [
        (p["emitted_s"], 0, "partial", p["text"]) for p in seg.get("partials", [])
    ]
    events += [(p["emitted_s"], 1, "provisional", p["text"]) for p in seg.get("provisionals", [])]
    if seg.get("final_s") is not None:
        events.append((seg["final_s"], 2, "final", seg["text"]))
    out: list[Display] = []
    last: tuple[str, ...] = ()
    for t, _, kind, text in sorted(events, key=lambda e: (e[0], e[1])):
        tokens = tuple(tokenize(text))
        if tokens == last:
            continue
        out.append(Display(t, kind, tokens))
        last = tokens
    return out


@dataclass
class SegmentStability:
    updates: int  # 顯示文字改變的次數（含第一次出現）
    unstable: int  # 其中前一次的字被改掉或刪掉的次數
    revised_tokens: int  # 被改掉或刪掉的字數總和（共同前綴之後的全部字，上界）
    revised_tokens_aligned: int  # 以 LCS 對齊時前一次的字沒保留下來的字數總和（下界）
    final_tokens: int
    # 依被取代的文字的來源：{kind: [被取代次數, 修正次數, 被改掉的字數]}
    by_kind: dict[str, list[int]]
    stable_delay_s: list[float]  # 最終文字每個字：穩定時刻 − 第一次出現在該位置的時刻


def segment_stability(seq: Sequence[Display]) -> SegmentStability:
    by_kind = {k: [0, 0, 0] for k in KINDS}
    unstable = revised = aligned = 0
    prev: tuple[str, ...] = ()
    prev_kind = None
    for d in seq:
        r = len(prev) - common_prefix(prev, d.tokens)
        if prev_kind is not None:
            stat = by_kind[prev_kind]
            stat[0] += 1
            stat[1] += r > 0
            stat[2] += r
        unstable += r > 0
        revised += r
        aligned += len(prev) - lcs(prev, d.tokens) if r else 0
        prev, prev_kind = d.tokens, d.kind

    final = seq[-1].tokens if seq else ()
    # suffix_min[k] = min(第 k 次之後每次顯示與最終文字的共同前綴)；隨 k 遞增不減
    suffix_min = [0] * len(seq)
    m = len(final)
    for k in range(len(seq) - 1, -1, -1):
        m = min(m, common_prefix(seq[k].tokens, final))
        suffix_min[k] = m
    delays: list[float] = []
    for i in range(len(final)):
        first = next(d.emitted_s for d in seq if len(d.tokens) > i)
        stable = next(seq[k].emitted_s for k in range(len(seq)) if suffix_min[k] > i)
        delays.append(stable - first)
    return SegmentStability(len(seq), unstable, revised, aligned, len(final), by_kind, delays)


def summarize_stability(utterances: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """全部句子的穩定度統計；結果 JSON 沒有逐次 partial 紀錄（舊結果）時回傳 None。

    - `unstable_update_rate`：修正次數 / 顯示更新次數（第一次出現不算更新，分母不含）。
    - `revised_tokens_per_utterance`：每條被改掉的字數（mean / P90…），`revised_token_ratio` =
      全部被改掉的字數 / 全部最終字數（語料級）；`*_aligned` 改以 LCS 對齊計算（只算真的被換掉或刪掉的字，
      句中改一字時後面的字不算）。`utterances_with_revision`：至少被改一次的條數比例。
    - `stable_delay_ms`：每個最終字的前綴穩定延遲（全部字一起算分位數），`stable_on_first_ratio` = 延遲 0 的比例。
    - `by_replaced_kind`：被取代的是 partial / 暫定結果時，各自的次數、修正比例、被改掉的字數。
    """
    if not utterances or any(
        "partials" not in seg for u in utterances for seg in u.get("segments", [])
    ):
        return None
    updates = unstable = revised = aligned = final_tokens = 0
    per_utt_revised: list[float] = []
    per_utt_updates: list[float] = []
    delays_ms: list[float] = []
    by_kind = {k: [0, 0, 0] for k in KINDS}
    with_revision = 0
    for u in utterances:
        utt_revised = utt_updates = 0
        for seg in u["segments"]:
            s = segment_stability(display_sequence(seg))
            # 第一次出現（空白 → 第一個字）不是「更新」
            updates += max(s.updates - 1, 0)
            utt_updates += s.updates
            unstable += s.unstable
            revised += s.revised_tokens
            aligned += s.revised_tokens_aligned
            utt_revised += s.revised_tokens
            final_tokens += s.final_tokens
            delays_ms.extend(d * 1000 for d in s.stable_delay_s)
            for k, v in s.by_kind.items():
                by_kind[k] = [a + b for a, b in zip(by_kind[k], v, strict=True)]
        per_utt_revised.append(utt_revised)
        per_utt_updates.append(utt_updates)
        with_revision += utt_revised > 0
    return {
        "updates": updates,
        "unstable_updates": unstable,
        "unstable_update_rate": unstable / updates if updates else 0.0,
        "displays_per_utterance": summarize(per_utt_updates),
        "revised_tokens": revised,
        "revised_tokens_per_utterance": summarize(per_utt_revised),
        "revised_token_ratio": revised / final_tokens if final_tokens else 0.0,
        "revised_tokens_aligned": aligned,
        "revised_token_ratio_aligned": aligned / final_tokens if final_tokens else 0.0,
        "utterances_with_revision": with_revision / len(utterances),
        "stable_delay_ms": summarize(delays_ms),
        "stable_on_first_ratio": (
            sum(d == 0 for d in delays_ms) / len(delays_ms) if delays_ms else 0.0
        ),
        "by_replaced_kind": {
            k: {
                "replaced": v[0],
                "unstable": v[1],
                "unstable_rate": v[1] / v[0] if v[0] else 0.0,
                "revised_tokens": v[2],
            }
            for k, v in by_kind.items()
            if k != "final"  # 最終結果不會被取代
        },
    }


# ---------- 第一個 partial 的顯示規則（只改畫面要不要顯示，不改辨識結果） ----------


@dataclass(frozen=True)
class FirstPartialRule:
    """句段在畫面上還沒有任何字時，partial 要滿足這個規則才顯示；一旦顯示過，之後的 partial 照常顯示。

    偽串流第一次 partial 只看了零點幾秒的音訊，常是英文 / 數字碎片（「low」「jo」）或單一語氣詞
    （「嗯」），顯示出來下一次就被整個換掉。暫定結果與最終結果不受規則影響，一律顯示。

    - `min_cjk`：至少要有幾個中文（非 ASCII）token；0 = 不限。
    - `drop_ascii`：正規化後全是 ASCII（英文詞 / 數字）就不顯示。
    """

    min_cjk: int = 0
    drop_ascii: bool = False

    def shows(self, text: str) -> bool:
        tokens = tokenize(text)
        if not tokens:
            return False
        if self.drop_ascii and all(t.isascii() for t in tokens):
            return False
        return sum(not t.isascii() for t in tokens) >= self.min_cjk

    def describe(self) -> str:
        parts = (["去 ASCII"] if self.drop_ascii else []) + (
            [f"≥ {self.min_cjk} 中文字"] if self.min_cjk else []
        )
        return "、".join(parts) or "原樣"


def apply_first_partial_rule(seg: dict[str, Any], rule: FirstPartialRule) -> dict[str, Any]:
    """回傳套用顯示規則後的句段（淺複製）：丟掉第一個符合規則的 partial 之前的 partial。

    `e2e_first_token_latency_s`（第一個非空 partial 的時刻）同步改成第一個顯示的 partial；
    沒有任何 partial 顯示時為 None（畫面要等暫定結果 / 最終結果才有字）。
    """
    partials = seg.get("partials", [])
    k = next((i for i, p in enumerate(partials) if rule.shows(p["text"])), len(partials))
    out = dict(seg)
    out["partials"] = partials[k:]
    if "e2e_first_token_latency_s" in seg:
        out["e2e_first_token_latency_s"] = partials[k]["emitted_s"] if k < len(partials) else None
    return out
