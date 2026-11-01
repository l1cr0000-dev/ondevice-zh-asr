"""評測指標：語料級 CER 與其 bootstrap 信賴區間、延遲分位數。"""

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer
import numpy as np

from eval.text_norm import normalize


@dataclass(frozen=True)
class CerResult:
    substitutions: int
    deletions: int
    insertions: int
    ref_tokens: int  # N：參考 token 數（中文 = 字數）

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        return self.errors / self.ref_tokens if self.ref_tokens else 0.0


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> CerResult:
    """語料級 CER：全部錯誤數總和 / 全部參考字數總和。正規化後參考為空的句子會報錯。"""
    if len(references) != len(hypotheses):
        raise ValueError("參考與假設句數不同")
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    empty = [i for i, r in enumerate(refs) if not r]
    if empty:
        raise ValueError(f"正規化後參考為空：第 {empty} 句")
    out = jiwer.process_words(refs, hyps)
    return CerResult(
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        ref_tokens=out.substitutions + out.deletions + out.hits,
    )


def utterance_counts(
    references: Sequence[str], hypotheses: Sequence[str]
) -> tuple[list[int], list[int]]:
    """逐句的（錯誤數, 參考 token 數），正規化與 `corpus_cer` 相同；兩者總和等於語料級的 errors / N。"""
    if len(references) != len(hypotheses):
        raise ValueError("參考與假設句數不同")
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    out = jiwer.process_words(refs, hyps)
    errors: list[int] = []
    for chunks in out.alignments:
        n = 0
        for c in chunks:
            if c.type in ("substitute", "delete"):
                n += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                n += c.hyp_end_idx - c.hyp_start_idx
        errors.append(n)
    return errors, [len(r.split()) for r in refs]


def bootstrap_cer_ci(
    errors: Sequence[int],
    ref_tokens: Sequence[int],
    groups: Sequence[str] | None = None,
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """語料級 CER 的百分位 bootstrap 信賴區間。

    `groups=None` 時以句為單位重抽；給 `groups`（例如說話人）時以組為單位重抽（cluster bootstrap），
    每次抽中一組就帶入該組全部句子，反映同一說話人的句子彼此相關。
    """
    err = np.asarray(errors, dtype=np.int64)
    ref = np.asarray(ref_tokens, dtype=np.int64)
    if groups is not None:
        keys = sorted(set(groups))
        index = {k: i for i, k in enumerate(keys)}
        gid = np.array([index[g] for g in groups])
        err = np.bincount(gid, weights=err, minlength=len(keys))
        ref = np.bincount(gid, weights=ref, minlength=len(keys))
    rng = np.random.default_rng(seed)
    cers = np.empty(n_resamples)
    batch = max(1, 2_000_000 // len(err))  # 每批的索引矩陣約 2M 個元素，全量 7176 句也不吃記憶體
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        idx = rng.integers(0, len(err), size=(stop - start, len(err)))
        cers[start:stop] = err[idx].sum(axis=1) / ref[idx].sum(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(cers, [alpha, 1 - alpha])
    return float(lo), float(hi)


def paired_bootstrap_cer_diff(
    errors_a: Sequence[int],
    errors_b: Sequence[int],
    ref_tokens: Sequence[int],
    n_resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """同一批句子上兩個設定的語料級 CER 差（B − A）與成對百分位 bootstrap 信賴區間。

    每次以句為單位重抽，A 與 B 用同一組索引，消除句子難易度的共同變異。回傳 (差, 下界, 上界)。
    """
    ea = np.asarray(errors_a, dtype=np.int64)
    eb = np.asarray(errors_b, dtype=np.int64)
    ref = np.asarray(ref_tokens, dtype=np.int64)
    if not len(ea) == len(eb) == len(ref):
        raise ValueError("兩個設定的句數不同")
    diff = eb - ea
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples)
    batch = max(1, 2_000_000 // len(ref))
    for start in range(0, n_resamples, batch):
        stop = min(start + batch, n_resamples)
        idx = rng.integers(0, len(ref), size=(stop - start, len(ref)))
        out[start:stop] = diff[idx].sum(axis=1) / ref[idx].sum(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(out, [alpha, 1 - alpha])
    return float(diff.sum() / ref.sum()), float(lo), float(hi)


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """平均與 P50 / P90 / P99（numpy 線性內插）。空序列回傳 None。"""
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
