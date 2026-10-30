"""評測指標：語料級 CER、延遲分位數。"""

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
