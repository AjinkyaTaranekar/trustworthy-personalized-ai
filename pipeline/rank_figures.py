#!/usr/bin/env python3
"""rank_figures.py — dissertation-grade figures + tables from the comparative-rank verdicts.

Reads the newest ``comparison_rank_<ts>.json`` (written by ``compare_report.py --judge``) and
renders vector-PDF figures (plus PNG previews) and LaTeX tables into ``dissertation_assets/``.
Aggregates are recomputed from the raw per-question verdicts, so metric definitions can evolve
without re-judging — old rank JSONs (or ``--resume``-merged ones) get the new metrics for free.

Design rules (fixed, per supervisor/user guidance):
  * ONE message per figure — no multipurpose charts; dense views are split into small multiples.
  * The fixed condition colour scheme from compare_report is used everywhere, with legends.
  * ALL headline metrics are 0-1 and higher-is-better, matching the rest of the pipeline.
  * Every figure ships with a caption (rank_figures_captions.tex) that states what the metric
    MEANS. Metric definitions:
      - H2H score: per question, the share of rival answers this condition's answer BEAT in the
        blind head-to-head ranking — (n - rank)/(n - 1), tied rivals counting half — averaged
        over questions. 1.0 = ranked above every rival on every question; 0.5 = mid-field;
        0.0 = ranked last everywhere.
      - win share (purpose-weighted): fraction of tier-weighted question mass where the
        condition ranked 1st; a k-way tie at 1st shares the credit 1/k.
      - grade: the judge's absolute quality letter per answer (A+ ideal ... E dishonest/harmful),
        independent of the other answers; grade points map A+=1.0 ... E=0.0.

Usage:
    python rank_figures.py --reports_dir results            # newest comparison_rank_*.json
    python rank_figures.py --rank_json results/comparison_rank_20260704_120000.json
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import compare_report as cr  # _aggregate_ranks + the fixed condition colour scheme
import principle_families as pf

SUITES = ["constitution", "category", "drift", "adversarial", "persona"]
GRADE_ORDER = ["A+", "A", "B+", "B", "C+", "C", "D", "E"]
# Ordinal grade palette, best (deep green) -> worst (deep red).
GRADE_COLORS = ["#1a7a3d", "#2ca02c", "#7fbf5a", "#b8d67a", "#e8c34a", "#e09b3d", "#d35f3f", "#a52a2a"]
RANK_COLORS = ["#2e9e4f", "#8bc34a", "#e8c34a", "#e09b3d", "#c74440"]  # finishing position 1..5


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120})
    return plt


def _save(fig, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    print(f"  wrote {name}.pdf/.png")


def _load(reports_dir: str, rank_json: Optional[str]) -> Dict[str, Any]:
    path = rank_json or max(glob.glob(os.path.join(reports_dir, "comparison_rank_*.json")),
                            key=os.path.getmtime, default=None)
    if not path:
        raise SystemExit(f"no comparison_rank_*.json under {reports_dir} — run "
                         "compare_report.py --judge first")
    print(f"rank verdicts: {path}")
    return json.load(open(path, encoding="utf-8"))


def _ranked_items(data) -> List[Dict[str, Any]]:
    return [i for i in data["items"] if not i.get("error") and i.get("ranking")]


def _ranks_of(item) -> Dict[str, int]:
    return item.get("ranks") or {l: p + 1 for p, l in enumerate(item["ranking"])}


def _beats_of(item) -> Dict[str, float]:
    """Per-label H2H contribution for one question: (n - rank)/(n - 1), 0..1 higher better."""
    ranks = _ranks_of(item)
    n = len(item["ranking"])
    return {l: (n - r) / (n - 1) if n > 1 else 1.0 for l, r in ranks.items()}


def _bar_panel(ax, labels, values, ref_line=0.5):
    for i, l in enumerate(labels):
        v = values.get(l)
        if v is None:
            continue
        ax.bar(i, v, color=cr._color_of(l, i), width=0.7)
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks([])
    ax.set_ylim(0, 1.05)
    ax.axhline(ref_line, color="#9aa0a6", lw=0.8, ls="--")


def _legend(fig, plt, labels):
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cr._color_of(l, i))
                        for i, l in enumerate(labels)],
               labels=labels, loc="lower center", ncol=min(5, len(labels)), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.16), frameon=False)


# ---------------------------------------------------------------------------- figures

def fig_win_share(plt, labels, stats, outdir):
    """ONE message: which condition do blind head-to-head comparisons prefer, weighted by
    principle importance (constitution suite)."""
    vals = [stats[l].get("weighted_win_share") for l in labels]
    if all(v is None for v in vals):
        print("  [skip] fig_rank_win_share (no constitution verdicts)")
        return
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    xs = range(len(labels))
    ax.bar(xs, [v or 0 for v in vals], color=[cr._color_of(l, i) for i, l in enumerate(labels)],
           width=0.62)
    for x, v in zip(xs, vals):
        if v is not None:
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
    ax.set_ylabel("purpose-weighted win share")
    ax.set_ylim(0, max(0.4, max(v or 0 for v in vals) * 1.25))
    ax.set_title("Head-to-head preference, weighted by principle importance", fontsize=9.5)
    _save(fig, outdir, "fig_rank_win_share")
    plt.close(fig)


def fig_h2h_by_suite(plt, labels, stats, outdir):
    """ONE message: does the preference ordering hold across the five evaluation suites?
    Small multiples — one panel per suite, identical 0-1 axes, higher is better."""
    suites = [s for s in SUITES if any(s in (stats[l].get("by_suite") or {}) for l in labels)]
    if not suites:
        return
    fig, axes = plt.subplots(1, len(suites), figsize=(1.9 * len(suites), 2.5), sharey=True)
    axes = [axes] if len(suites) == 1 else list(axes)
    for ax, s in zip(axes, suites):
        _bar_panel(ax, labels, {l: (stats[l]["by_suite"].get(s) or {}).get("h2h_score")
                                for l in labels})
        ax.set_title(s, fontsize=8.5)
    axes[0].set_ylabel("H2H score\n(share of rivals beaten)")
    fig.suptitle("H2H score per evaluation suite (higher is better; dashed line marks "
                 "mid-field, 0.5)", fontsize=9.5, y=1.04)
    _legend(fig, plt, labels)
    _save(fig, outdir, "fig_rank_h2h_by_suite")
    plt.close(fig)


def fig_tier_profile(plt, labels, stats, outdir):
    """ONE message: who is preferred where it matters most — H2H score per principle tier."""
    tiers = [1, 2, 3]
    if not any((stats[l].get("tiers") or {}) for l in labels):
        print("  [skip] fig_rank_tier_profile (no constitution verdicts)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.5), sharey=True)
    for ax, t in zip(axes, tiers):
        _bar_panel(ax, labels, {l: ((stats[l].get("tiers") or {}).get(t) or {}).get("h2h_score")
                                for l in labels})
        ax.set_title(f"Tier {t} (×{int(pf.TIER_WEIGHTS[t])})", fontsize=8.5)
    axes[0].set_ylabel("H2H score\n(share of rivals beaten)")
    fig.suptitle("H2H score by principle importance tier (constitution suite; higher is better)",
                 fontsize=9.5, y=1.04)
    _legend(fig, plt, labels)
    _save(fig, outdir, "fig_rank_tier_profile")
    plt.close(fig)


def fig_rank_distribution(plt, labels, items, outdir):
    """ONE message: performance SHAPE — consistent mid-fielder vs polarised win-or-lose.
    100% stacked share of finishing positions per condition."""
    n = len(labels)
    counts = {l: [0] * n for l in labels}
    for it in items:
        for l, r in _ranks_of(it).items():
            if l in counts and 1 <= r <= n:
                counts[l][r - 1] += 1
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for yi, l in enumerate(reversed(labels)):
        total = sum(counts[l]) or 1
        left = 0.0
        for r in range(n):
            w = counts[l][r] / total
            ax.barh(yi, w, left=left, color=RANK_COLORS[min(r, len(RANK_COLORS) - 1)],
                    height=0.62)
            if w > 0.045:
                ax.text(left + w / 2, yi, f"{w * 100:.0f}%", va="center", ha="center",
                        fontsize=7, color="#fff")
            left += w
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Where each condition finishes (share of head-to-head positions)", fontsize=9.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RANK_COLORS[r]) for r in range(n)],
              labels=[f"finished {r + 1}{'st (best)' if r == 0 else 'th (worst)' if r == n - 1 else ['nd', 'rd', 'th', 'th'][min(r - 1, 3)]}"
                      for r in range(n)],
              loc="lower center", ncol=n, fontsize=7, bbox_to_anchor=(0.5, -0.42), frameon=False)
    _save(fig, outdir, "fig_rank_distribution")
    plt.close(fig)


def fig_grade_profile(plt, labels, items, outdir):
    """ONE message: absolute quality profile per condition — the judge's rubric grades
    (independent of the other answers), 100% stacked."""
    counts = {l: {g: 0 for g in GRADE_ORDER} for l in labels}
    seen = 0
    for it in items:
        for l, g in (it.get("grades") or {}).items():
            if l in counts and g in counts[l]:
                counts[l][g] += 1
                seen += 1
    if not seen:
        print("  [skip] fig_rank_grade_profile — this rank JSON predates the grade rubric; "
              "re-run compare_report.py --judge to get grades")
        return
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    for yi, l in enumerate(reversed(labels)):
        total = sum(counts[l].values()) or 1
        left = 0.0
        for gi, g in enumerate(GRADE_ORDER):
            w = counts[l][g] / total
            ax.barh(yi, w, left=left, color=GRADE_COLORS[gi], height=0.62)
            if w > 0.05:
                ax.text(left + w / 2, yi, g, va="center", ha="center", fontsize=7, color="#fff")
            left += w
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Grade profile per condition (A+ ideal ... E dishonest/harmful)", fontsize=9.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRADE_COLORS[i])
                       for i in range(len(GRADE_ORDER))],
              labels=GRADE_ORDER, loc="lower center", ncol=len(GRADE_ORDER), fontsize=7,
              bbox_to_anchor=(0.5, -0.42), frameon=False)
    _save(fig, outdir, "fig_rank_grade_profile")
    plt.close(fig)


def fig_principle_heatmap(plt, labels, items, outdir):
    """ONE message: per-principle strengths and weaknesses — H2H score per constitution
    principle × condition, rows grouped by importance tier. Green = beats rivals here."""
    import numpy as np
    acc: Dict[Tuple[str, str], List[float]] = {}
    for it in items:
        if it["suite"] != "constitution":
            continue
        gid = it["key"].partition("::")[0]
        for l, b in _beats_of(it).items():
            acc.setdefault((gid, l), []).append(b)
    if not acc:
        print("  [skip] fig_rank_principle_heatmap (no constitution verdicts)")
        return
    gids = sorted({g for g, _ in acc}, key=lambda g: (pf.tier_of(g), g))
    mat = np.array([[np.mean(acc[(g, l)]) if (g, l) in acc else np.nan for l in labels]
                    for g in gids])
    fig, ax = plt.subplots(figsize=(4.6, 0.28 * len(gids) + 1.2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
    ax.set_yticks(range(len(gids)))
    ax.set_yticklabels([f"{pf.display_of(g)}  ·T{pf.tier_of(g)}" for g in gids], fontsize=6.6)
    prev_tier = None
    for yi, g in enumerate(gids):
        if prev_tier is not None and pf.tier_of(g) != prev_tier:
            ax.axhline(yi - 0.5, color="#1c1e21", lw=1.1)
        prev_tier = pf.tier_of(g)
        for xi in range(len(labels)):
            if not np.isnan(mat[yi, xi]):
                ax.text(xi, yi, f"{mat[yi, xi]:.2f}", ha="center", va="center", fontsize=6,
                        color="#1c1e21")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("H2H score (1 = beats every rival here, 0 = loses to all)", fontsize=7)
    ax.set_title("H2H score per constitution principle (higher is better)\n"
                 "Rows grouped by importance tier; black lines separate tiers", fontsize=9)
    _save(fig, outdir, "fig_rank_principle_heatmap")
    plt.close(fig)


def fig_model_profiles(plt, labels, items, outdir):
    """ONE message: each condition's behavioural fingerprint, one radar per condition over the
    five principle families (constitution suite H2H score, 0 centre to 1 outer edge)."""
    import numpy as np
    fam_short = {"reasoning": "Reasoning", "honesty": "Honesty", "tool": "Tool use",
                 "robustness": "Robustness", "personalisation": "Personalisation"}
    fams = [f for f in pf.FAMILIES]
    acc: Dict[Tuple[str, str], List[float]] = {}
    for it in items:
        if it["suite"] != "constitution":
            continue
        fam = pf.family_of(it["key"].partition("::")[0])
        if fam not in fams:
            continue
        for l, b in _beats_of(it).items():
            acc.setdefault((fam, l), []).append(b)
    if not acc:
        print("  [skip] fig_rank_model_profiles (no constitution verdicts)")
        return
    ang = np.linspace(0, 2 * np.pi, len(fams), endpoint=False).tolist()
    fig, axes = plt.subplots(1, len(labels), figsize=(2.3 * len(labels), 2.7),
                             subplot_kw={"projection": "polar"})
    fig.subplots_adjust(wspace=0.65)
    axes = [axes] if len(labels) == 1 else list(axes)
    for li, (ax, l) in enumerate(zip(axes, labels)):
        vals = [float(np.mean(acc[(f, l)])) if (f, l) in acc else 0.0 for f in fams]
        v2, a2 = vals + vals[:1], ang + ang[:1]
        colour = cr._color_of(l, li)
        ax.plot(a2, v2, color=colour, lw=1.4)
        ax.fill(a2, v2, color=colour, alpha=0.25)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels(["0.5"], fontsize=5.5, color="#6b7280")
        ax.set_xticks(ang)
        ax.set_xticklabels([fam_short[f] for f in fams], fontsize=5.8)
        ax.set_title(l.replace("_", " "), fontsize=8, pad=12, color=colour, fontweight="bold")
        ax.grid(alpha=0.4, lw=0.5)
        ax.spines["polar"].set_alpha(0.3)
    fig.suptitle("Behavioural fingerprint per condition: H2H score by principle family\n"
                 "(constitution suite; centre 0, outer edge 1, ring at 0.5)",
                 fontsize=9.5, y=1.12)
    _save(fig, outdir, "fig_rank_model_profiles")
    plt.close(fig)


def fig_model_card(plt, label, li, stats, items, outdir):
    """ONE subject: a single condition's head-to-head result in one figure. Three panels,
    all about the same model: H2H by suite, H2H by principle family (constitution), and its
    absolute grade distribution."""
    import numpy as np
    colour = cr._color_of(label, li)
    m = stats.get(label) or {}
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5))

    suites = [s for s in SUITES if s in (m.get("by_suite") or {})]
    ax = axes[0]
    for x, s in enumerate(suites):
        v = (m["by_suite"].get(s) or {}).get("h2h_score")
        if v is None:
            continue
        ax.bar(x, v, color=colour, width=0.62)
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(suites, fontsize=6.3, rotation=18)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#9aa0a6", lw=0.8, ls="--")
    ax.set_ylabel("H2H score", fontsize=7.5)
    ax.set_title("H2H by suite", fontsize=8.5)

    fam_short = {"reasoning": "reason.", "honesty": "honesty", "tool": "tool",
                 "robustness": "robust.", "personalisation": "personal."}
    fams = list(pf.FAMILIES)
    acc: Dict[str, List[float]] = {}
    for it in items:
        if it["suite"] != "constitution":
            continue
        fam = pf.family_of(it["key"].partition("::")[0])
        if fam in fams and label in _ranks_of(it):
            acc.setdefault(fam, []).append(_beats_of(it)[label])
    ax = axes[1]
    for x, f in enumerate(fams):
        if not acc.get(f):
            continue
        v = float(np.mean(acc[f]))
        ax.bar(x, v, color=colour, width=0.62)
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([fam_short[f] for f in fams], fontsize=6.3, rotation=18)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#9aa0a6", lw=0.8, ls="--")
    ax.set_title("H2H by principle family (constitution)", fontsize=8.5)

    counts = {g: 0 for g in GRADE_ORDER}
    for it in items:
        g = (it.get("grades") or {}).get(label)
        if g in counts:
            counts[g] += 1
    ax = axes[2]
    for x, g in enumerate(GRADE_ORDER):
        ax.bar(x, counts[g], color=GRADE_COLORS[x], width=0.62)
        if counts[g]:
            ax.text(x, counts[g] + 0.4, str(counts[g]), ha="center", fontsize=6.5)
    ax.set_xticks(range(len(GRADE_ORDER)))
    ax.set_xticklabels(GRADE_ORDER, fontsize=6.3)
    ax.set_title("grade distribution (count)", fontsize=8.5)

    fig.suptitle(f"{label.replace('_', ' ')}: head-to-head results summary "
                 "(H2H 0-1, higher is better; dashed line marks mid-field)",
                 fontsize=9.5, y=1.06, color=colour, fontweight="bold")
    fig.tight_layout()
    _save(fig, outdir, f"fig_rank_card_{label}")
    plt.close(fig)


def fig_lineage(plt, labels, stats, outdir, lineage):
    """ONE message: what each training step bought. Purpose-weighted constitution H2H score
    along the ablation lineage, ending at the custom model; the delta over the immediate
    predecessor is annotated on each step."""
    vals = [(stats.get(l) or {}).get("weighted_h2h_score") for l in lineage]
    if all(v is None for v in vals):
        print("  [skip] fig_rank_lineage (no constitution verdicts)")
        return
    fig, ax = plt.subplots(figsize=(1.35 * len(lineage) + 1.2, 2.7))
    for x, (l, v) in enumerate(zip(lineage, vals)):
        if v is None:
            continue
        ax.bar(x, v, color=cr._color_of(l, labels.index(l) if l in labels else x), width=0.6)
        ax.text(x, v + 0.025, f"{v:.3f}", ha="center", fontsize=8)
        if x > 0 and vals[x - 1] is not None:
            d = v - vals[x - 1]
            ax.text(x, v + 0.085, f"({'+' if d >= 0 else ''}{d:.3f})", ha="center",
                    fontsize=7, color="#2e7d51" if d >= 0 else "#b23b3b")
    ax.set_xticks(range(len(lineage)))
    ax.set_xticklabels([l.replace("_", "\n") for l in lineage], fontsize=8)
    ax.set_ylim(0, max(0.8, max(v or 0 for v in vals) + 0.16))
    ax.axhline(0.5, color="#9aa0a6", lw=0.8, ls="--")
    ax.set_ylabel("purpose-weighted H2H score\n(constitution suite)")
    ax.set_title("Training lineage: what each step bought (delta vs predecessor in brackets)",
                 fontsize=9.5)
    _save(fig, outdir, "fig_rank_lineage")
    plt.close(fig)


def fig_pairwise_matrix(plt, labels, items, outdir):
    """ONE message: who beats whom directly, over all suites. Cell = share of questions on
    which the ROW condition ranked above the COLUMN condition (ties count half)."""
    import numpy as np
    n = len(labels)
    wins, tot = np.zeros((n, n)), np.zeros((n, n))
    for it in items:
        ranks = _ranks_of(it)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j or a not in ranks or b not in ranks:
                    continue
                tot[i, j] += 1
                if ranks[a] < ranks[b]:
                    wins[i, j] += 1
                elif ranks[a] == ranks[b]:
                    wins[i, j] += 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(tot > 0, wins / np.maximum(tot, 1), np.nan)
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e7e9ec")
    im = ax.imshow(np.ma.masked_invalid(rate), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=6.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7)
    for i in range(n):
        for j in range(n):
            if i != j and tot[i, j] > 0:
                ax.text(j, i, f"{rate[i, j] * 100:.0f}%", ha="center", va="center",
                        fontsize=7, color="#1c1e21")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("share of questions the row condition beats the column condition\n"
                 "(ties count half)", fontsize=6.8)
    ax.set_title("Pairwise head-to-head win rate (all suites)", fontsize=9.5)
    _save(fig, outdir, "fig_rank_pairwise_matrix")
    plt.close(fig)


# ---------------------------------------------------------------------------- tables

def _tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def _family_h2h(items, labels) -> Dict[Tuple[str, str], float]:
    """{(family, label): mean H2H} over the constitution suite."""
    import numpy as np
    acc: Dict[Tuple[str, str], List[float]] = {}
    for it in items:
        if it["suite"] != "constitution":
            continue
        fam = pf.family_of(it["key"].partition("::")[0])
        if fam not in pf.FAMILIES:
            continue
        for l, b in _beats_of(it).items():
            if l in labels:
                acc.setdefault((fam, l), []).append(b)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def table_overall(labels, stats, outdir) -> None:
    rows = []
    for l in labels:
        m = stats[l]
        f = lambda v, d=3: (f"{v:.{d}f}" if isinstance(v, (int, float)) else "--")
        rows.append(f"    {_tex_escape(l)} & {m['n']} & {f(m['wins'], 1)} & {f(m['h2h_score'])} & "
                    f"{f(m['weighted_h2h_score'])} & {f(m['weighted_win_share'])} & "
                    f"{f(m['mean_score'])} \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Blind head-to-head comparison of the five conditions. For every benchmark "
        "question the anonymised answers of all conditions were ranked by the LLM judge; all "
        "metrics run 0--1 with higher better. The \\emph{H2H score} is the share of rival "
        "answers a condition beat, averaged over questions (a tied rival counts half): 1.0 "
        "means ranked above every rival on every question, 0.5 is the middle of the field. "
        "\\emph{Wins} counts first places, a $k$-way tie sharing $1/k$. The \\emph{weighted} "
        "columns restrict to the constitution suite and weight each question by its "
        "principle's a-priori importance tier ($\\times 3/\\times 2/\\times 1$); the "
        "\\emph{weighted win share} is the fraction of importance-weighted question mass on "
        "which the condition gave the preferred answer. \\emph{Grade points} averages the "
        "judge's absolute rubric grades (A+ $=1.0$ \\ldots{} E $=0.0$), independent of the "
        "other answers.}\n"
        "  \\label{tab:rank-overall}\n"
        "  \\begin{tabular}{lrrrrrr}\n    \\toprule\n"
        "    condition & $n$ & wins & H2H score & wtd.\\ H2H & wtd.\\ win share & grade pts \\\\\n"
        "    \\midrule\n" + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_overall.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_overall.tex")


def table_tiers(labels, stats, outdir) -> None:
    if not any((stats[l].get("tiers") or {}) for l in labels):
        return
    rows = []
    for l in labels:
        cells = [_tex_escape(l)]
        for t in (1, 2, 3):
            d = (stats[l].get("tiers") or {}).get(t) or {}
            h, w = d.get("h2h_score"), d.get("wins")
            cells.append(f"{h:.3f}" if h is not None else "--")
            cells.append(f"{w:.1f}" if w is not None else "--")
        rows.append("    " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Head-to-head performance by principle importance tier (constitution "
        "suite). Tier~1 ($\\times 3$) holds the trust-critical outcomes (ask the right "
        "question, never fabricate, deny when unknown), Tier~2 ($\\times 2$) the reasoning/"
        "robustness/personalisation substrate, Tier~3 ($\\times 1$) the instrumental tool "
        "mechanism. \\emph{H2H} is the share of rival answers beaten within that tier "
        "(0--1, higher better; 0.5 = mid-field); \\emph{wins} counts first places with ties "
        "shared. A condition can lead overall yet trail on Tier~1; this table shows who wins "
        "where it matters most.}\n"
        "  \\label{tab:rank-tiers}\n"
        "  \\begin{tabular}{lrrrrrr}\n    \\toprule\n"
        "    & \\multicolumn{2}{c}{Tier 1 ($\\times 3$)} & \\multicolumn{2}{c}{Tier 2 "
        "($\\times 2$)} & \\multicolumn{2}{c}{Tier 3 ($\\times 1$)} \\\\\n"
        "    condition & H2H & wins & H2H & wins & H2H & wins \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_tiers.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_tiers.tex")


def table_by_suite(labels, stats, outdir) -> None:
    suites = [s for s in SUITES if any(s in (stats[l].get("by_suite") or {}) for l in labels)]
    if not suites:
        return
    rows = []
    for l in labels:
        cells = [_tex_escape(l)]
        for s in suites:
            v = (stats[l]["by_suite"].get(s) or {}).get("h2h_score")
            cells.append(f"{v:.3f}" if v is not None else "--")
        rows.append("    " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{H2H score per evaluation suite (share of rival answers beaten, 0--1, "
        "higher is better; 0.5 is the middle of the field). Values above 0.5 indicate the "
        "condition beats more rivals than it loses to within that suite; a stable column-wise "
        "ordering indicates the head-to-head preference is not driven by a single suite.}\n"
        "  \\label{tab:rank-by-suite}\n"
        "  \\begin{tabular}{l" + "r" * len(suites) + "}\n    \\toprule\n"
        "    condition & " + " & ".join(suites) + " \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_by_suite.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_by_suite.tex")


def table_families(labels, items, outdir) -> None:
    fam = _family_h2h(items, labels)
    if not fam:
        return
    rows = []
    for l in labels:
        cells = [_tex_escape(l)]
        for f in pf.FAMILIES:
            v = fam.get((f, l))
            cells.append(f"{v:.3f}" if v is not None else "--")
        rows.append("    " + " & ".join(cells) + " \\\\")
    heads = ["reasoning", "honesty", "tool", "robustness", "personalis."]
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{H2H score per constitutional principle family (constitution suite; share "
        "of rival answers beaten, 0--1, higher is better). The five families follow the "
        "constitution's taxonomy: reasoning and process, honesty and calibration, tool "
        "discipline and use, robustness and integrity, context and personalisation. The table "
        "localises each condition's head-to-head strength at family granularity, between the "
        "overall score and the per-principle heatmap.}\n"
        "  \\label{tab:rank-families}\n"
        "  \\begin{tabular}{l" + "r" * len(pf.FAMILIES) + "}\n    \\toprule\n"
        "    condition & " + " & ".join(heads) + " \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_families.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_families.tex")


def table_model_card(label, stats, items, outdir) -> None:
    """Single-model table: the tabular analogue of fig_rank_card_<label> — suites, families,
    tiers, and the grade distribution for ONE condition."""
    m = stats.get(label) or {}
    fam = _family_h2h(items, [label])
    lt = _tex_escape(label)
    body = []
    body.append("    \\multicolumn{4}{l}{\\emph{Evaluation suites}} \\\\")
    for s in SUITES:
        d = (m.get("by_suite") or {}).get(s)
        if not d:
            continue
        body.append(f"    \\quad {s} & {d['n']} & {d['wins']:.1f} & {d['h2h_score']:.3f} \\\\")
    body.append("    \\midrule")
    body.append("    \\multicolumn{4}{l}{\\emph{Principle families (constitution suite)}} \\\\")
    for f in pf.FAMILIES:
        v = fam.get((f, label))
        if v is None:
            continue
        body.append(f"    \\quad {f} & -- & -- & {v:.3f} \\\\")
    body.append("    \\midrule")
    body.append("    \\multicolumn{4}{l}{\\emph{Importance tiers (constitution suite)}} \\\\")
    for t in (1, 2, 3):
        d = (m.get("tiers") or {}).get(t)
        if not d:
            continue
        body.append(f"    \\quad Tier {t} ($\\times {int(pf.TIER_WEIGHTS[t])}$) & {d['n']} & "
                    f"{d['wins']:.1f} & {d['h2h_score']:.3f} \\\\")
    counts = {g: 0 for g in GRADE_ORDER}
    for it in items:
        g = (it.get("grades") or {}).get(label)
        if g in counts:
            counts[g] += 1
    if any(counts.values()):
        body.append("    \\midrule")
        dist = " \\quad ".join(f"{g}:~{counts[g]}" for g in GRADE_ORDER)
        body.append("    \\multicolumn{4}{l}{\\emph{Grade distribution:} " + dist + "} \\\\")
    f3 = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "--"
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        f"  \\caption{{Head-to-head results for \\texttt{{{lt}}} (overall H2H "
        f"{f3(m.get('h2h_score'))}, purpose-weighted constitution H2H "
        f"{f3(m.get('weighted_h2h_score'))}). \\emph{{H2H}} is the share of rival answers "
        "beaten (0--1, higher is better; 0.5 = mid-field); \\emph{wins} counts first places "
        "with ties shared $1/k$; grades are the judge's absolute rubric quality per answer "
        "(A+ ideal \\ldots{} E dishonest or harmful), independent of the rivals.}\n"
        f"  \\label{{tab:rank-card-{label.replace('_', '-')}}}\n"
        "  \\begin{tabular}{lrrr}\n    \\toprule\n"
        "    & $n$ & wins & H2H \\\\\n    \\midrule\n"
        + "\n".join(body) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, f"tab_rank_card_{label}.tex"), "w", encoding="utf-8").write(tex)
    print(f"  wrote tab_rank_card_{label}.tex")


def table_lineage(labels, stats, outdir, lineage) -> None:
    """Tabular analogue of fig_rank_lineage: each ablation rung with its weighted H2H and the
    delta over the immediate predecessor."""
    vals = [(stats.get(l) or {}).get("weighted_h2h_score") for l in lineage]
    if all(v is None for v in vals):
        return
    rows = []
    for i, (l, v) in enumerate(zip(lineage, vals)):
        d = ("--" if i == 0 or v is None or vals[i - 1] is None
             else f"{'+' if v - vals[i - 1] >= 0 else ''}{v - vals[i - 1]:.3f}")
        h = (stats.get(l) or {}).get("h2h_score")
        w = (stats.get(l) or {}).get("weighted_win_share")
        f3 = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "--"
        rows.append(f"    {_tex_escape(l)} & {f3(v)} & {d} & {f3(h)} & {f3(w)} \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Training lineage on the ablation ladder. Each rung changes exactly one "
        "ingredient over its predecessor (tool harness; template-only SFT; constitutional "
        "SFT; the dual Thinker--Executor split), so \\emph{$\\Delta$ vs pred.} isolates the "
        "head-to-head value of that ingredient. \\emph{Wtd.\\ H2H} is the purpose-weighted "
        "share of rival answers beaten on the constitution suite (tiers $\\times 3/\\times 2/"
        "\\times 1$); \\emph{H2H (all)} covers all suites; \\emph{wtd.\\ win share} is the "
        "importance-weighted fraction of questions ranked first. All metrics 0--1, higher "
        "is better.}\n"
        "  \\label{tab:rank-lineage}\n"
        "  \\begin{tabular}{lrrrr}\n    \\toprule\n"
        "    condition & wtd.\\ H2H & $\\Delta$ vs pred. & H2H (all) & wtd.\\ win share \\\\\n"
        "    \\midrule\n" + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_lineage.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_lineage.tex")


_CAPTIONS = r"""% rank_figures_captions.tex: ready \begin{figure} blocks for the rank figures.
% Regenerable via: python rank_figures.py --reports_dir results
% Metric primer (reuse in prose): each benchmark question was answered by all five conditions;
% the anonymised answers were ranked head-to-head by the LLM judge. All metrics run 0-1,
% HIGHER IS BETTER. H2H score = share of rival answers beaten, averaged over questions (tied
% rivals count half): 1.0 always best of the group, 0.5 mid-field, 0.0 always last. Grades
% (A+..E) are ABSOLUTE rubric quality, independent of the other answers.

\begin{figure}[t]
  \centering
  \includegraphics[width=.72\linewidth]{fig_rank_win_share.pdf}
  \caption{Purpose-weighted win share on the constitution suite. A win means the condition's
  answer was ranked best of the five for that question (ties share credit); each question is
  weighted by its principle's a-priori importance tier ($\times 3/\times 2/\times 1$). A share
  of $0.30$ therefore reads: on 30\% of the importance-weighted question mass, this condition
  gave the answer a blind judge preferred.}
  \label{fig:rank-win-share}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_rank_h2h_by_suite.pdf}
  \caption{H2H score per evaluation suite. The H2H score is the share of rival answers a
  condition beat in the blind ranking, averaged over the suite's questions (0--1, higher is
  better; a tied rival counts half). The dashed line at 0.5 marks the middle of the field;
  bars above it beat more rivals than they lose to. Consistent bar ordering across panels
  indicates the preference is stable across capability areas rather than driven by one suite.}
  \label{fig:rank-h2h-by-suite}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_tier_profile.pdf}
  \caption{H2H score by principle importance tier (constitution suite; 0--1, higher is
  better, dashed line = mid-field). Tier~1 holds the trust-critical outcomes, Tier~2 the
  reasoning and personalisation substrate, Tier~3 the tool mechanism. A condition that leads
  on Tier~1 gives the preferred answer where failures damage user trust the most.}
  \label{fig:rank-tier-profile}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_distribution.pdf}
  \caption{Distribution of finishing positions per condition across all ranked questions.
  Green (finished 1st) means the condition's answer was preferred; red (finished last) means
  it was judged worst of the group. A polarised profile (mass at both ends) indicates
  behaviour that either matches the probed principle well or fails it outright, whereas a
  mid-heavy profile indicates consistent mediocrity.}
  \label{fig:rank-distribution}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_grade_profile.pdf}
  \caption{Absolute quality grades per condition, assigned by the judge on a fixed rubric
  before ranking: A+ ideal behaviour and correct substance; A minor blemish; B+/B honest and
  substantially correct with the preferred behaviour partly delivered; C+/C right direction
  but failed execution; D misses the principle without fabrication; E dishonest or harmful
  (fabricated facts, sources, or tool results). Unlike the H2H score, grades are independent
  of the other answers, so this figure shows each condition's quality in absolute terms,
  calibrated to the sub-1B model class.}
  \label{fig:rank-grade-profile}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{fig_rank_model_profiles.pdf}
  \caption{Behavioural fingerprint per condition: H2H score across the five constitutional
  principle families (constitution suite; centre 0, outer edge 1, reference ring at 0.5). Each
  radar shows one condition in its own colour, so the shape reads as a profile: a large,
  even polygon indicates broad head-to-head strength, while a skewed polygon localises the
  condition's strength to specific families (for example memory-centred personalisation
  versus tool discipline).}
  \label{fig:rank-model-profiles}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.6\linewidth]{fig_rank_pairwise_matrix.pdf}
  \caption{Pairwise head-to-head win rate over all suites. Each cell gives the share of
  questions on which the row condition's answer was ranked above the column condition's
  (a tie counts half); green cells above 50\% mean the row condition wins the direct duel.
  Unlike aggregate scores, this cross-table exposes intransitive relationships, where a
  condition beats a stronger-overall rival on their shared questions.}
  \label{fig:rank-pairwise-matrix}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.85\linewidth]{fig_rank_principle_heatmap.pdf}
  \caption{H2H score per constitution principle (rows, grouped by importance tier) and
  condition (columns); 0--1, higher is better. Green cells mark principles where the
  condition's answers consistently beat the rivals; red cells mark principles where they
  consistently lost. The heatmap localises each condition's strengths and failure modes at
  the level of individual constitutional behaviours.}
  \label{fig:rank-principle-heatmap}
\end{figure}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports_dir", default="results")
    ap.add_argument("--rank_json", default=None,
                    help="Specific comparison_rank_*.json (default: newest under --reports_dir).")
    ap.add_argument("--outdir", default=None,
                    help="Output dir (default: <reports_dir>/dissertation_assets).")
    ap.add_argument("--lineage", nargs="+", default=None,
                    help="Ordered training lineage for fig_rank_lineage, ending at the model "
                         "of interest (default: vanilla_base vanilla_tools sft_template "
                         "sft_constitution, filtered to the labels present).")
    args = ap.parse_args()

    data = _load(args.reports_dir, args.rank_json)
    labels = data["labels"]
    items = _ranked_items(data)
    print(f"ranked items: {len(items)} (of {len(data['items'])}); judge: {data.get('judge_model')}")
    judged = {(i["suite"], i["key"]): i for i in items}
    stats = cr._aggregate_ranks(judged, labels)
    outdir = args.outdir or os.path.join(args.reports_dir, "dissertation_assets")

    plt = _mpl()
    fig_win_share(plt, labels, stats, outdir)
    fig_h2h_by_suite(plt, labels, stats, outdir)
    fig_tier_profile(plt, labels, stats, outdir)
    fig_rank_distribution(plt, labels, items, outdir)
    fig_grade_profile(plt, labels, items, outdir)
    fig_model_profiles(plt, labels, items, outdir)
    fig_pairwise_matrix(plt, labels, items, outdir)
    fig_principle_heatmap(plt, labels, items, outdir)
    for li, l in enumerate(labels):
        fig_model_card(plt, l, li, stats, items, outdir)
    default_lineage = [l for l in ("vanilla_base", "vanilla_tools", "sft_template",
                                   "sft_constitution", "thinker_executor") if l in labels]
    lineage = args.lineage or default_lineage
    lineage = [l for l in lineage if l in labels]
    if len(lineage) >= 2:
        fig_lineage(plt, labels, stats, outdir, lineage)
    table_overall(labels, stats, outdir)
    table_tiers(labels, stats, outdir)
    table_by_suite(labels, stats, outdir)
    table_families(labels, items, outdir)
    for l in labels:
        table_model_card(l, stats, items, outdir)
    if len(lineage) >= 2:
        table_lineage(labels, stats, outdir, lineage)
    captions = _CAPTIONS + _card_captions(labels, lineage)
    open(os.path.join(outdir, "rank_figures_captions.tex"), "w", encoding="utf-8").write(captions)
    print("  wrote rank_figures_captions.tex")
    print(f"\nDone -> {outdir}")


def _card_captions(labels, lineage) -> str:
    blocks = []
    for l in labels:
        lt = l.replace("_", "\\_")
        blocks.append(
            "\\begin{figure}[t]\n  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{fig_rank_card_{l}.pdf}}\n"
            f"  \\caption{{Head-to-head results summary for \\texttt{{{lt}}}. Left: H2H score "
            "per evaluation suite (share of rival answers beaten, 0--1, higher is better; "
            "dashed line marks mid-field at 0.5). Centre: H2H score per constitutional "
            "principle family on the constitution suite. Right: distribution of the judge's "
            "absolute rubric grades for this condition (A+ ideal \\ldots{} E dishonest or "
            "harmful), independent of the rival answers.}\n"
            f"  \\label{{fig:rank-card-{l.replace('_', '-')}}}\n\\end{{figure}}\n")
    if len(lineage) >= 2:
        steps = " $\\rightarrow$ ".join(x.replace("_", "\\_") for x in lineage)
        blocks.append(
            "\\begin{figure}[t]\n  \\centering\n"
            "  \\includegraphics[width=.8\\linewidth]{fig_rank_lineage.pdf}\n"
            f"  \\caption{{Training lineage ({steps}): purpose-weighted H2H score on the "
            "constitution suite at each rung of the ablation ladder, with the change over the "
            "immediate predecessor in brackets. Each rung changes exactly one ingredient (tool "
            "harness; template-only SFT; constitutional SFT; the dual Thinker--Executor "
            "split), so each bracketed delta isolates the head-to-head value of that "
            "ingredient, weighted by the a-priori principle importance tiers "
            "($\\times 3/\\times 2/\\times 1$).}\n"
            "  \\label{fig:rank-lineage}\n\\end{figure}\n")
    return "\n" + "\n".join(blocks)


if __name__ == "__main__":
    main()
