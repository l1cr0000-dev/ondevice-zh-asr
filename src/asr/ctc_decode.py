"""CTC 解碼：greedy 與帶熱詞加分（context biasing）的 prefix beam search。只改解碼，不動模型。

熱詞加分用 Aho-Corasick 自動機，概念同 icefall / sherpa-onnx 的 `ContextGraph`：
- 候選前綴每多拼對熱詞的一個 token 加 `score`（暫時加分）；拼到一半失敗（自動機沿 fail 退回較短的
  後綴或根）時，把退回部分的暫時加分扣掉，因此「拼到一半」不會白白占便宜。
- 拼完一個熱詞時，暫時加分轉為永久（共 `長度 × score`），暫時部分歸零；狀態仍留在節點上，
  可繼續拼更長的熱詞（例如「北京」→「北京大学」）。
- 解碼結束時扣掉最後一個狀態的暫時加分（沒拼完的熱詞不算）。
最後排序用「CTC 分數 + 熱詞加分」，因此加分只影響候選排名，不改模型的機率。

與 icefall strict mode 的差異：icefall 在拼完時「另外」加上完整加分、離開節點時才扣回暫時加分，
拼完到離開之間該前綴暫時多了一倍加分。實測（300 句對齊集）這會讓熱詞後面的字被刪掉
（「安丰公司的业务」→「安丰公司业务」：接任何字都要先扣回加分，不出字的前綴暫時領先），
beam 1 時尤其嚴重，因此這裡改成拼完即轉永久、不重複計分。
較短的熱詞恰好是較長熱詞中間的一段時（例如「新网」之於「中新网」以外的「X新网Y」），不另計分。

prefix beam search 為了效率：
- 每個 frame 只展開機率最高的 `top_k` 個非 blank token。
- blank 機率 ≥ `blank_skip` 的 frame 只做「接 blank / 重複最後一個 token」兩種轉移（不產生新 token）。
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

BLANK = 0
NEG_INF = float("-inf")


def ctc_greedy(log_probs: npt.NDArray[np.floating]) -> list[int]:
    """逐 frame argmax、合併重複、去掉 blank。"""
    tokens: list[int] = []
    prev = -1
    for y in log_probs.argmax(axis=-1).tolist():
        if y != BLANK and y != prev:
            tokens.append(y)
        prev = y
    return tokens


def tokenize_hotword(word: str, vocab: dict[str, int]) -> list[int] | None:
    """熱詞 → token id（從左到右最長匹配）。英文單字前加「▁」對應 SentencePiece 的詞首標記。

    中文大多一字一 token。有 token 對不上時回傳 None（呼叫端略過並提示）。
    """
    pieces = [("▁" + w if w.isascii() else w) for w in word.split()] if word.strip() else []
    ids: list[int] = []
    longest = max(map(len, vocab), default=0)
    for piece in pieces:
        i = 0
        while i < len(piece):
            for j in range(min(len(piece), i + longest), i, -1):
                if piece[i:j] in vocab:
                    ids.append(vocab[piece[i:j]])
                    i = j
                    break
            else:
                return None
    return ids or None


@dataclass(eq=False)
class _Node:
    depth: int
    is_end: bool = False  # 有熱詞在此結束
    children: dict[int, "_Node"] = field(default_factory=dict)
    fail: "_Node | None" = None


# 自動機狀態：（節點, 尚未轉為永久的暫時加分）
HotwordState = tuple[_Node, float]


class HotwordGraph:
    """熱詞的 Aho-Corasick 自動機。`step` 回傳（加分變化, 新狀態）。"""

    def __init__(self, hotwords: Iterable[Sequence[int]], score: float) -> None:
        self.score = score
        self.root = _Node(depth=0)
        self.num_words = 0
        for word in hotwords:
            if not word:
                continue
            node = self.root
            for t in word:
                if t not in node.children:
                    node.children[t] = _Node(depth=node.depth + 1)
                node = node.children[t]
            node.is_end = True
            self.num_words += 1
        # BFS 建 fail：最長的、同時也是某熱詞字首的真後綴
        self.root.fail = self.root
        queue = list(self.root.children.values())
        for child in queue:
            child.fail = self.root
        while queue:
            node = queue.pop(0)
            for t, child in node.children.items():
                f = node.fail
                assert f is not None
                while t not in f.children and f is not self.root:
                    assert f.fail is not None
                    f = f.fail
                child.fail = f.children.get(t, self.root)
                queue.append(child)

    @property
    def start(self) -> HotwordState:
        return self.root, 0.0

    def step(self, state: HotwordState, token: int) -> tuple[float, HotwordState]:
        node, pending = state
        if token in node.children:
            nxt = node.children[token]
            new_pending = pending + self.score
        else:
            while token not in node.children and node is not self.root:
                assert node.fail is not None
                node = node.fail
            nxt = node.children.get(token, self.root)
            # 退到較短的後綴：暫時加分重算為該後綴的長度（之前已轉永久的部分不受影響）
            new_pending = self.score * nxt.depth
        delta = new_pending - pending
        if nxt.is_end:
            new_pending = 0.0  # 拼完：暫時加分轉永久（已計入 delta），不再扣回
        return delta, (nxt, new_pending)

    def finalize(self, state: HotwordState) -> float:
        """結束時扣回沒拼完的暫時加分。"""
        return -state[1]


@dataclass
class _Prefix:
    pb: float = NEG_INF  # 以 blank 結尾的 log 機率
    pnb: float = NEG_INF  # 以最後一個 token 結尾的 log 機率
    context: float = 0.0  # 熱詞加分累積
    state: HotwordState | None = None

    @property
    def total(self) -> float:
        return float(np.logaddexp(self.pb, self.pnb))


def _child(
    beams: dict[tuple[int, ...], _Prefix],
    prefix: tuple[int, ...],
    src: _Prefix,
    token: int | None,
    graph: HotwordGraph | None,
) -> _Prefix:
    """下一個 frame 的候選 `prefix`（不存在就由 `src` 接上 `token` 建立；`token=None` 表示前綴不變）。

    同一個 token 序列的熱詞狀態是確定的，不論從哪個前綴走到，加分都相同。
    """
    p = beams.get(prefix)
    if p is None:
        if token is None or graph is None:
            p = _Prefix(context=src.context, state=src.state)
        else:
            assert src.state is not None
            delta, state = graph.step(src.state, token)
            p = _Prefix(context=src.context + delta, state=state)
        beams[prefix] = p
    return p


def ctc_prefix_beam_search(
    log_probs: npt.NDArray[np.floating],
    beam: int,
    graph: HotwordGraph | None = None,
    top_k: int = 10,
    blank_skip: float = 0.999,
) -> list[int]:
    """CTC prefix beam search；`graph` 給定時以「CTC 分數 + 熱詞加分」排序候選。回傳最佳 token 序列。

    `log_probs` 是 (T, vocab) 的 log 機率，也可以直接給未正規化的 logits（逐 frame 做 log-softmax）。
    """
    start = None if graph is None else graph.start
    beams: dict[tuple[int, ...], _Prefix] = {(): _Prefix(pb=0.0, state=start)}
    skip_lp = float(np.log(blank_skip))
    for row in np.asarray(log_probs, dtype=np.float32):
        # 逐 frame 做 log-softmax（已是 log 機率時不變）；不整段先算，避免多一份 (T, vocab) 的暫存
        m = row.max()
        frame = row - (m + np.log(np.exp(row - m).sum()))
        lp_blank = float(frame[BLANK])
        if lp_blank >= skip_lp:
            for prefix, p in beams.items():
                pnb = p.pnb + float(frame[prefix[-1]]) if prefix else NEG_INF
                p.pb, p.pnb = p.total + lp_blank, pnb
            continue
        k = min(top_k, len(frame) - 1)
        top = np.argpartition(-frame, k)[: k + 1]
        cand = [int(t) for t in top[np.argsort(-frame[top])] if t != BLANK][:k]
        nxt: dict[tuple[int, ...], _Prefix] = {}
        for prefix, p in beams.items():
            total = p.total
            # 接 blank：前綴不變
            same = _child(nxt, prefix, p, None, graph)
            same.pb = float(np.logaddexp(same.pb, total + lp_blank))
            last = prefix[-1] if prefix else None
            for t in cand:
                lp = float(frame[t])
                if t == last:
                    # 重複最後一個 token 且中間沒有 blank：合併成同一個前綴
                    same.pnb = float(np.logaddexp(same.pnb, p.pnb + lp))
                    # 中間隔著 blank：才是新的一個 token
                    ext = _child(nxt, prefix + (t,), p, t, graph)
                    ext.pnb = float(np.logaddexp(ext.pnb, p.pb + lp))
                else:
                    ext = _child(nxt, prefix + (t,), p, t, graph)
                    ext.pnb = float(np.logaddexp(ext.pnb, total + lp))
            if last is not None and last not in cand:
                same.pnb = float(np.logaddexp(same.pnb, p.pnb + float(frame[last])))
        ranked = sorted(nxt.items(), key=lambda kv: kv[1].total + kv[1].context, reverse=True)
        beams = dict(ranked[:beam])

    def final(p: _Prefix) -> float:
        bonus = p.context
        if graph is not None and p.state is not None:
            bonus += graph.finalize(p.state)
        return p.total + bonus

    best = max(beams.items(), key=lambda kv: final(kv[1]))
    return list(best[0])
