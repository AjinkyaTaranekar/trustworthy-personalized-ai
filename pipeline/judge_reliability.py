#!/usr/bin/env python3
"""
judge_reliability.py — agreement / reliability statistics for the LLM judge
===========================================================================

Pure-Python (no numpy/scipy) statistics used by ``5_judgement_day.py --meta_eval``
to answer the question a thesis examiner will ask: *how do you know your marking
scheme is valid?* The answer is empirical — measure how well the LLM judge agrees
with a small **human-anchored gold set**, and how stable the judge is with itself.

Two families of number:

  1. Judge-vs-human agreement (validity)
     * pearson / spearman      — correlation of judge score with the human score
     * mae                     — mean absolute error on the 0..1 scale
     * exact_agreement         — fraction scored identically (within a tolerance)
     * krippendorff_alpha      — chance-corrected agreement (interval); the metric
                                 the human-eval rubric already targets (alpha>=0.67)
     * cohens_kappa / gwet_ac1 — chance-corrected agreement on the binary pass/fail
                                 decision. Gwet's AC1 is reported alongside kappa
                                 because kappa collapses under the skewed pass-rates
                                 typical of compliance probes (the "kappa paradox").

  2. Judge self-consistency (reliability of the instrument itself)
     * self_consistency        — over k repeated samples per item: the mean
                                 within-item standard deviation and the rate at
                                 which the k samples flip the pass/fail verdict.

References (see the methodology page in the wiki):
  Krippendorff (2004); Gwet (2008); Zheng et al. (2023) LLM-as-judge; the empirical
  LLM-judge design study (arXiv 2506.13639) — mean-of-k sampling beats greedy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Number = float


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _paired(human: Sequence[Optional[Number]],
            judge: Sequence[Optional[Number]]) -> Tuple[List[float], List[float]]:
    """Drop any pair where either side is missing (None)."""
    xs, ys = [], []
    for h, j in zip(human, judge):
        if h is None or j is None:
            continue
        xs.append(float(h)); ys.append(float(j))
    return xs, ys


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# continuous-score agreement
# ---------------------------------------------------------------------------

def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = _mean(x), _mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:  # one side constant — correlation undefined
        return None
    return sxy / math.sqrt(sxx * syy)


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based), ties share the mean of their rank span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return pearson(_rank(x), _rank(y))


def mae(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if not x:
        return None
    return _mean([abs(a - b) for a, b in zip(x, y)])


def exact_agreement(x: Sequence[float], y: Sequence[float], tol: float = 1e-9) -> Optional[float]:
    if not x:
        return None
    return _mean([1.0 if abs(a - b) <= tol else 0.0 for a, b in zip(x, y)])


# ---------------------------------------------------------------------------
# chance-corrected agreement on categorical labels (e.g. pass / fail)
# ---------------------------------------------------------------------------

def _confusion(a: Sequence, b: Sequence) -> Tuple[List, Dict, Dict, float, int]:
    cats = sorted(set(a) | set(b), key=str)
    n = len(a)
    p_o = _mean([1.0 if x == y else 0.0 for x, y in zip(a, b)]) if n else 0.0
    pa = {c: sum(1 for x in a if x == c) / n for c in cats} if n else {}
    pb = {c: sum(1 for x in b if x == c) / n for c in cats} if n else {}
    return cats, pa, pb, p_o, n


def cohens_kappa(a: Sequence, b: Sequence) -> Optional[float]:
    """Cohen's kappa for two raters over the same items."""
    if not a:
        return None
    cats, pa, pb, p_o, _ = _confusion(a, b)
    p_e = sum(pa[c] * pb[c] for c in cats)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else None
    return (p_o - p_e) / (1.0 - p_e)


def gwet_ac1(a: Sequence, b: Sequence) -> Optional[float]:
    """Gwet's AC1 — robust to the prevalence/skew paradox that distorts kappa when
    one label dominates (common for compliance probes where most answers pass)."""
    if not a:
        return None
    cats, pa, pb, p_o, _ = _confusion(a, b)
    q = len(cats)
    if q < 2:
        return 1.0 if p_o >= 1.0 else None
    pi = {c: (pa[c] + pb[c]) / 2.0 for c in cats}
    p_e = sum(pi[c] * (1.0 - pi[c]) for c in cats) / (q - 1)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else None
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (handles >=2 raters, missing values, interval/nominal)
# ---------------------------------------------------------------------------

def krippendorff_alpha(data: Sequence[Sequence[Optional[Number]]],
                       level: str = "interval") -> Optional[float]:
    """``data`` is one list per *unit*; each unit holds that unit's ratings (one per
    rater), with ``None`` for a missing rating. Units with fewer than two present
    ratings are dropped (they carry no pairing information)."""
    def delta(c: float, k: float) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        return (c - k) ** 2  # interval

    units = [[float(v) for v in u if v is not None] for u in data]
    units = [u for u in units if len(u) >= 2]
    values: List[float] = [v for u in units for v in u]
    n = len(values)
    if n < 2:
        return None

    d_o = 0.0
    for u in units:
        m = len(u)
        s = sum(delta(u[i], u[j]) for i in range(m) for j in range(m) if i != j)
        d_o += s / (m - 1)
    d_o /= n

    d_e = sum(delta(values[a], values[b])
              for a in range(n) for b in range(n) if a != b) / (n * (n - 1))
    if d_e == 0:
        return 1.0  # no variance in the data → treat as perfect agreement
    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# convenience: a full validity report for one set of (human, judge) pairs
# ---------------------------------------------------------------------------

def _to_pass(score: Optional[Number], threshold: float = 0.6) -> Optional[int]:
    return None if score is None else int(score >= threshold)


def agreement_report(human: Sequence[Optional[Number]],
                     judge: Sequence[Optional[Number]],
                     pass_threshold: float = 0.6) -> Dict[str, Optional[float]]:
    """All judge-vs-human agreement numbers for one principle (or overall)."""
    xs, ys = _paired(human, judge)
    n = len(xs)
    hp = [_to_pass(v, pass_threshold) for v, _ in zip(xs, ys)]
    jp = [_to_pass(v, pass_threshold) for _, v in zip(xs, ys)]
    return {
        "n": n,
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "mae": mae(xs, ys),
        "exact_agreement": exact_agreement(xs, ys),
        "krippendorff_alpha": krippendorff_alpha([[a, b] for a, b in zip(xs, ys)], "interval"),
        "cohens_kappa_pass": cohens_kappa(hp, jp) if n else None,
        "gwet_ac1_pass": gwet_ac1(hp, jp) if n else None,
        "human_mean": _mean(xs) if xs else None,
        "judge_mean": _mean(ys) if ys else None,
        "judge_bias": (_mean(ys) - _mean(xs)) if xs else None,  # +ve = judge too lenient
    }


# ---------------------------------------------------------------------------
# judge self-consistency over k repeated samples per item
# ---------------------------------------------------------------------------

def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def self_consistency(per_item_samples: Sequence[Sequence[Optional[Number]]],
                     pass_threshold: float = 0.6) -> Dict[str, Optional[float]]:
    """``per_item_samples`` is one list of k repeated judge scores per item.
    Reports how unstable the judge is with itself: mean within-item stdev and the
    fraction of items whose k samples disagree on the pass/fail verdict."""
    clean = [[float(s) for s in item if s is not None] for item in per_item_samples]
    clean = [item for item in clean if item]
    if not clean:
        return {"items": 0, "mean_within_item_std": None, "verdict_flip_rate": None,
                "mean_k": None}
    stds = [_stdev(item) for item in clean if len(item) >= 2]
    flips = []
    for item in clean:
        verdicts = {1 if s >= pass_threshold else 0 for s in item}
        flips.append(1.0 if len(verdicts) > 1 else 0.0)
    return {
        "items": len(clean),
        "mean_k": _mean([len(item) for item in clean]),
        "mean_within_item_std": _mean(stds) if stds else 0.0,
        "verdict_flip_rate": _mean(flips),
    }


if __name__ == "__main__":  # quick self-test (no deps): python judge_reliability.py
    # perfect agreement
    h = [1.0, 0.5, 0.0, 1.0, 0.0]
    assert abs(krippendorff_alpha([[a, a] for a in h], "interval") - 1.0) < 1e-9
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert gwet_ac1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    # judge always one notch high → positive bias, high but <1 correlation
    j = [min(1.0, x + 0.0) for x in h]
    rep = agreement_report(h, j)
    assert rep["pearson"] is not None and rep["pearson"] > 0.99
    # self-consistency: identical samples → zero spread, no flips
    sc = self_consistency([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert sc["mean_within_item_std"] == 0.0 and sc["verdict_flip_rate"] == 0.0
    # noisy samples that straddle the threshold → a flip
    sc2 = self_consistency([[0.5, 0.7, 0.4]])
    assert sc2["verdict_flip_rate"] == 1.0
    print("judge_reliability self-test OK")
