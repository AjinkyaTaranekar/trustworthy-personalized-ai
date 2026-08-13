#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for the head-to-head scores.

Reads a saved comparison_rank_*.json and resamples the ranked questions with
replacement to put a 95% interval on each condition's H2H score and on the
difference between any two conditions.

Runs entirely from saved judge verdicts: no GPU, no model calls, no re-judging.
The per-question beat-credit is the same tie-aware formula compare_report.py
uses to build the leaderboard, so the point estimates reproduce
tab_rank_overall.tex exactly. The script asserts that reproduction before
reporting any interval, and exits non-zero if it fails.

Usage:
    python rank_bootstrap.py                                  # newest run in results/
    python rank_bootstrap.py --rank_json results/comparison_rank_20260704_131401.json
    python rank_bootstrap.py --b 20000 --seed 7 --tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
from typing import Any, Dict, List, Tuple

DEFAULT_PAIRS = [
    ("sft_constitution", "sft_template"),
    ("sft_constitution", "thinker_executor"),
    ("sft_constitution", "vanilla_base"),
    ("vanilla_tools", "vanilla_base"),
    ("sft_constitution", "vanilla_tools"),
]


def beats_of(v: Dict[str, Any]) -> Dict[str, float]:
    """Tie-aware per-question H2H contribution, 0..1.

    Mirrors compare_report.py: labels sharing a rank form a tie group, and the
    beat-credit of that group is averaged across its members, so a tied rival
    counts as half a beat.
    """
    ranking = v.get("ranking") or []
    if v.get("error") or not ranking:
        return {}
    n = len(ranking)
    ranks = v.get("ranks") or {lbl: i + 1 for i, lbl in enumerate(ranking)}
    groups: Dict[int, List[str]] = {}
    for lbl in ranking:
        groups.setdefault(ranks.get(lbl, n), []).append(lbl)
    beat: Dict[str, float] = {}
    pos = 0
    for r in sorted(groups):
        grp = groups[r]
        avg = sum(n - 1 - (pos + j) for j in range(len(grp))) / len(grp)
        for lbl in grp:
            beat[lbl] = avg / (n - 1) if n > 1 else 1.0
        pos += len(grp)
    return beat


def percentile_ci(vals: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    v = sorted(vals)
    n = len(v)
    return v[int(alpha / 2 * n)], v[min(int((1 - alpha / 2) * n), n - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank_json", default=None,
                    help="comparison_rank_*.json (default: newest under results/)")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--b", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 gives a 95%% interval")
    ap.add_argument("--tex", action="store_true",
                    help="also emit a LaTeX table of the pairwise differences")
    args = ap.parse_args()

    path = args.rank_json or max(
        glob.glob(os.path.join(args.results_dir, "comparison_rank_*.json")),
        key=os.path.getmtime, default=None)
    if not path:
        raise SystemExit(f"no comparison_rank_*.json under {args.results_dir}")
    data = json.load(open(path, encoding="utf-8"))
    conds = list(data["aggregates"].keys())
    print(f"rank verdicts: {path}")
    print(f"judge: {data.get('judge_model')}")

    per_item: List[Tuple[str, Dict[str, float]]] = []
    for it in data["items"]:
        b = beats_of(it)
        if b:
            per_item.append((it.get("suite"), b))
    if not per_item:
        raise SystemExit("no ranked items in this report")
    print(f"ranked questions: {len(per_item)}")

    def mean_h2h(sample, cond):
        vals = [s[cond] for _, s in sample if cond in s]
        return statistics.fmean(vals) if vals else float("nan")

    point = {c: mean_h2h(per_item, c) for c in conds}

    # Guard: the point estimates must match the aggregates already in the report,
    # otherwise the scoring rule here has drifted from compare_report.py and the
    # intervals would describe a different quantity from the published table.
    drift = []
    for c in conds:
        published = data["aggregates"][c].get("h2h_score")
        if published is not None and abs(point[c] - published) > 1e-3:
            drift.append((c, point[c], published))
    if drift:
        for c, got, want in drift:
            print(f"  MISMATCH {c}: recomputed {got:.4f} vs report {want:.4f}")
        raise SystemExit("scoring rule does not reproduce the saved aggregates; "
                         "refusing to report intervals")
    print("point estimates reproduce the saved aggregates exactly")

    suites: Dict[str, int] = {}
    for su, _ in per_item:
        suites[su] = suites.get(su, 0) + 1
    print("\nquestions per suite (one question moves that suite score by 1/n):")
    for k, v in sorted(suites.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} n={v:3d}   1/n = {1.0/v:.3f}")

    rng = random.Random(args.seed)
    n = len(per_item)
    boot = {c: [] for c in conds}
    pairs = [p for p in DEFAULT_PAIRS if p[0] in conds and p[1] in conds]
    diffs = {p: [] for p in pairs}
    for _ in range(args.b):
        samp = [per_item[rng.randrange(n)] for _ in range(n)]
        m = {c: mean_h2h(samp, c) for c in conds}
        for c in conds:
            boot[c].append(m[c])
        for a, b in pairs:
            diffs[(a, b)].append(m[a] - m[b])

    print(f"\n=== H2H score, 95% percentile bootstrap "
          f"(B={args.b:,}, paired on questions, seed={args.seed}) ===")
    for c in sorted(conds, key=lambda c: -point[c]):
        lo, hi = percentile_ci(boot[c], args.alpha)
        print(f"  {c:20s} {point[c]:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n=== pairwise differences ===")
    rows = []
    for a, b in pairs:
        lo, hi = percentile_ci(diffs[(a, b)], args.alpha)
        obs = point[a] - point[b]
        sep = not (lo <= 0 <= hi)
        share = sum(1 for x in diffs[(a, b)] if x <= 0) / args.b
        print(f"  {a:18s} - {b:18s} {obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"{'separated' if sep else 'NOT separated (interval contains zero)'}"
              f"  [share of resamples <=0: {share:.3f}]")
        rows.append((a, b, obs, lo, hi, sep))

    if args.tex:
        print("\n% --- LaTeX: pairwise differences ---")
        print("\\begin{tabular}{@{}llrlc@{}}")
        print("  \\toprule")
        print("  Condition A & Condition B & A $-$ B & 95\\% CI & Separated? \\\\")
        print("  \\midrule")
        for a, b, obs, lo, hi, sep in sorted(rows, key=lambda r: -r[2]):
            ae, be = a.replace("_", "\\_"), b.replace("_", "\\_")
            print(f"  {ae} & {be} & ${obs:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$ & "
                  f"{'yes' if sep else 'no'} \\\\")
        print("  \\bottomrule")
        print("\\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
