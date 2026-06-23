#!/usr/bin/env python3
"""
experiment_figures.py — dissertation figures for the ablation ladder (offline, no GPU)
=====================================================================================

Renders the four judge-free figure families from `experiment_metrics.py` to vector
PDFs in reports/dissertation_assets/. Names are prefixed `fig_ladder_*` so they never
clobber export_assets.py's 2-condition `fig_per_family.pdf` / `fig_think_collapse.pdf`.

Families (metrics_and_figures_plan.md):
  1. Compliance ladder + deltas   — fig_ladder_compliance / _per_family / _deltas
  2. Reasoning depth & relocation — fig_think_distribution / _reasoning_location / _depth_vs_cost
  3. Tool behaviour & failures    — fig_tool_usage
  4. Drift over turns + category  — fig_drift_curve / _category_heatmap

Usage:
    python experiment_figures.py --labels vanilla_base vanilla_tools sft_template \
        sft_constitution thinker_executor
(or call render_all() from analyze_experiments.py --figures)
"""

from __future__ import annotations

import argparse
import os
import random
import statistics as st
from typing import Dict, List, Optional

import experiment_metrics as em
import principle_families as pf

# Okabe-Ito colourblind-safe palette; first colour ~ TCD blue.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
SHORT = {"vanilla_base": "C0 base", "vanilla_tools": "C1 +tools", "sft_template": "C2 tmpl",
         "sft_constitution": "C3 const", "thinker_executor": "C4 T–E"}


def _short(lbl: str) -> str:
    return SHORT.get(lbl, lbl)


def _try_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.autolayout": True})
        return plt
    except Exception as e:  # pragma: no cover
        print(f"  [figures skipped: matplotlib unavailable: {e}]")
        return None


def _save(fig, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {os.path.relpath(path)}")


def _records(reports_dir: str, label: str):
    p = em.find_report(reports_dir, "constitution", label)
    return em.response_records(em.load(p)) if p else []


# ---------------------------------------------------------------------------
# Family 1 — Compliance ladder + deltas
# ---------------------------------------------------------------------------

def fig_ladder_compliance(plt, M: Dict[str, dict], labels, outdir):
    metrics = [("const", lambda m: (m.get("constitution") or {}).get("overall_rule")),
               ("category", lambda m: (m.get("categories") or {}).get("overall")),
               ("adversarial", lambda m: (m.get("adversarial") or {}).get("overall")),
               ("drift", lambda m: (m.get("drift") or {}).get("overall"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n, w = len(labels), 0.8 / max(1, len(metrics))
    for j, (name, getter) in enumerate(metrics):
        xs = [i + j * w for i in range(n)]
        ax.bar(xs, [getter(M[l]) or 0 for l in labels], width=w, label=name, color=PALETTE[j])
    ax.set_xticks([i + w * (len(metrics) - 1) / 2 for i in range(n)])
    ax.set_xticklabels([_short(l) for l in labels])
    ax.set_ylabel("score (rule-based)"); ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    _save(fig, outdir, "fig_ladder_compliance.pdf"); plt.close(fig)


def fig_ladder_per_family(plt, M, labels, outdir):
    fams = pf.FAMILIES
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n, w = len(labels), 0.8 / len(fams)
    for j, fam in enumerate(fams):
        vals = [((M[l].get("constitution") or {}).get("by_family") or {}).get(fam, 0) for l in labels]
        ax.bar([i + j * w for i in range(n)], vals, width=w, label=fam, color=PALETTE[j])
    ax.set_xticks([i + w * (len(fams) - 1) / 2 for i in range(n)])
    ax.set_xticklabels([_short(l) for l in labels])
    ax.set_ylabel("compliance (rule)"); ax.set_ylim(0, 1)
    ax.legend(ncol=5, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    _save(fig, outdir, "fig_ladder_per_family.pdf"); plt.close(fig)


def _bootstrap_delta(a: Dict[str, float], b: Dict[str, float], n=2000, seed=42):
    keys = [k for k in a if k in b]
    if not keys:
        return None
    av, bv = [a[k] for k in keys], [b[k] for k in keys]
    d = st.fmean(bv) - st.fmean(av)
    rng = random.Random(seed); m = len(keys); ds = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        ds.append(st.fmean([bv[i] for i in idx]) - st.fmean([av[i] for i in idx]))
    ds.sort()
    return d, ds[int(0.025 * n)], ds[int(0.975 * n)]


def fig_ladder_deltas(plt, reports_dir, labels, outdir):
    """Forest plot: adjacent-rung deltas in constitution rule-pass, with bootstrap 95% CI."""
    items = {}
    for l in labels:
        items[l] = {x["key"]: (1.0 if x["rule_passed"] else 0.0) for x in _records(reports_dir, l)}
    rows = []
    for i in range(1, len(labels)):
        ci = _bootstrap_delta(items[labels[i - 1]], items[labels[i]])
        if ci:
            rows.append((f"{_short(labels[i-1])} → {_short(labels[i])}", *ci))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(rows) + 1.2))
    ys = range(len(rows))
    for y, (_lab, d, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="#555", lw=1.5)
        ax.plot(d, y, "o", color=PALETTE[0] if d >= 0 else PALETTE[3], ms=7)
    ax.axvline(0, color="#aaa", ls="--", lw=1)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Δ constitution rule-pass (bootstrap 95% CI)")
    ax.invert_yaxis()
    _save(fig, outdir, "fig_ladder_deltas.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Family 2 — Reasoning depth & relocation
# ---------------------------------------------------------------------------

def fig_think_distribution(plt, reports_dir, M, labels, outdir):
    data = [[x["think_len"] for x in _records(reports_dir, l)] for l in labels]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels([_short(l) for l in labels])
    ax.set_ylabel("<think> length (chars)")
    for i, l in enumerate(labels, 1):
        emp = (M[l].get("depth") or {}).get("think_empty_rate")
        if emp is not None:
            ax.text(i, ax.get_ylim()[1] * 0.95, f"{emp*100:.0f}% empty", ha="center", fontsize=7.5, color="#444")
    _save(fig, outdir, "fig_think_distribution.pdf"); plt.close(fig)


def fig_reasoning_location(plt, M, labels, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    in_think = [(M[l].get("depth") or {}).get("ext_ratio_mean") or 0 for l in labels]
    in_answer = [1 - v for v in in_think]
    x = range(len(labels))
    ax.bar(x, in_think, color=PALETTE[0], label="reasoning in <think>")
    ax.bar(x, in_answer, bottom=in_think, color=PALETTE[1], label="text in answer body")
    ax.set_xticks(list(x)); ax.set_xticklabels([_short(l) for l in labels])
    ax.set_ylabel("share of generated text"); ax.set_ylim(0, 1)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    _save(fig, outdir, "fig_reasoning_location.pdf"); plt.close(fig)


def fig_depth_vs_cost(plt, reports_dir, labels, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, l in enumerate(labels):
        xs, ys = [], []
        for x in _records(reports_dir, l):
            lat = x["latency_s"]
            if lat is None:
                continue
            xs.append(x["think_len"] + x["answer_chars"]); ys.append(lat)
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.5, color=PALETTE[i % len(PALETTE)], label=_short(l))
    ax.set_xlabel("generated text length (think + answer, chars)")
    ax.set_ylabel("latency (s)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, outdir, "fig_depth_vs_cost.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Family 3 — Tool behaviour
# ---------------------------------------------------------------------------

def fig_tool_usage(plt, M, labels, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = range(len(labels))
    a1.bar(x, [(M[l].get("tool") or {}).get("tool_calls_per_resp") or 0 for l in labels], color=PALETTE[0])
    a1.set_xticks(list(x)); a1.set_xticklabels([_short(l) for l in labels], rotation=30, ha="right")
    a1.set_ylabel("tool calls / response"); a1.set_title("Tool intensity", fontsize=9)
    w = 0.38
    a2.bar([i - w/2 for i in x], [(M[l].get("tool") or {}).get("tool_fail_rate") or 0 for l in labels],
           width=w, color=PALETTE[3], label="failure rate")
    a2.bar([i + w/2 for i in x], [(M[l].get("tool") or {}).get("decoy_bait_rate") or 0 for l in labels],
           width=w, color=PALETTE[4], label="decoy-bait rate")
    a2.set_xticks(list(x)); a2.set_xticklabels([_short(l) for l in labels], rotation=30, ha="right")
    a2.set_ylabel("rate"); a2.set_title("Tool discipline", fontsize=9)
    a2.legend(fontsize=7.5, frameon=False)
    _save(fig, outdir, "fig_tool_usage.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Family 4 — Drift over turns + category coverage
# ---------------------------------------------------------------------------

def fig_drift_curve(plt, M, labels, outdir):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, l in enumerate(labels):
        dm = M[l].get("drift") or {}
        curve = dm.get("curve") or []
        if not curve:
            continue
        ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, color=PALETTE[i % len(PALETTE)],
                label=_short(l))
        fd = dm.get("first_drift_at")
        if fd:
            ax.axvline(fd, color=PALETTE[i % len(PALETTE)], ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("conversation turn"); ax.set_ylabel("adherence (rule)"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _save(fig, outdir, "fig_drift_curve.pdf"); plt.close(fig)


def fig_category_heatmap(plt, M, labels, outdir):
    cats = []
    for l in labels:
        cats = list(((M[l].get("categories") or {}).get("by_category") or {}).keys()) or cats
    if not cats:
        return
    import numpy as np
    grid = np.array([[((M[l].get("categories") or {}).get("by_category") or {}).get(c, np.nan)
                      for l in labels] for c in cats], dtype=float)
    fig, ax = plt.subplots(figsize=(6.6, max(3.5, 0.32 * len(cats))))
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([_short(l) for l in labels], rotation=30, ha="right")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="score")
    _save(fig, outdir, "fig_category_heatmap.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render_all(labels: List[str], reports_dir: str = "reports",
               outdir: Optional[str] = None) -> None:
    plt = _try_mpl()
    if plt is None:
        return
    outdir = outdir or os.path.join(reports_dir, "dissertation_assets")
    M = {l: em.condition_metrics(reports_dir, l) for l in labels}
    fig_ladder_compliance(plt, M, labels, outdir)
    fig_ladder_per_family(plt, M, labels, outdir)
    fig_ladder_deltas(plt, reports_dir, labels, outdir)
    fig_think_distribution(plt, reports_dir, M, labels, outdir)
    fig_reasoning_location(plt, M, labels, outdir)
    fig_depth_vs_cost(plt, reports_dir, labels, outdir)
    fig_tool_usage(plt, M, labels, outdir)
    fig_drift_curve(plt, M, labels, outdir)
    fig_category_heatmap(plt, M, labels, outdir)
    print(f"\n  figures → {os.path.relpath(outdir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render ablation-ladder dissertation figures.")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    render_all(args.labels, args.reports_dir, args.outdir)
