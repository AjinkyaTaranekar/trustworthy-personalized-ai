#!/usr/bin/env python3
"""rank_figures.py — dissertation-grade figures + tables from the comparative-rank verdicts.

Reads the newest ``comparison_rank_<ts>.json`` (written by ``compare_report.py --judge``) and
renders vector-PDF figures (plus PNG previews) and LaTeX tables into ``dissertation_assets/``.

Design rules (fixed, per supervisor/user guidance):
  * ONE message per figure — no multipurpose charts; dense views are split into small multiples.
  * The fixed condition colour scheme from compare_report is used everywhere, with legends.
  * Every figure ships with a caption (rank_figures_captions.tex) that states what the metric
    MEANS — mean rank, win share, Borda, grades — so a reader never has to guess what a number
    denotes. Metric definitions:
      - rank: the blind judge ordered all conditions' answers to the SAME question, 1 = ranked
        best of the group, 5 = worst; equal grades share a (competition) rank.
      - mean rank: average of those positions over all ranked questions (lower = better).
      - win share (purpose-weighted): fraction of tier-weighted question mass where the
        condition ranked 1st; a k-way tie at 1st shares the credit 1/k.
      - grade: the judge's absolute quality letter per answer (A+ ideal ... E dishonest/harmful),
        independent of the other answers.

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
RANK_COLORS = ["#2e9e4f", "#8bc34a", "#e8c34a", "#e09b3d", "#c74440"]  # rank 1..5


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


def fig_mean_rank_by_suite(plt, labels, stats, outdir):
    """ONE message: does the preference ordering hold across the five evaluation suites?
    Small multiples — one panel per suite, identical axes."""
    suites = [s for s in SUITES if any(s in (stats[l].get("by_suite") or {}) for l in labels)]
    if not suites:
        return
    fig, axes = plt.subplots(1, len(suites), figsize=(1.9 * len(suites), 2.5), sharey=True)
    axes = [axes] if len(suites) == 1 else list(axes)
    n = len(labels)
    for ax, s in zip(axes, suites):
        for i, l in enumerate(labels):
            v = (stats[l]["by_suite"].get(s) or {}).get("mean_rank")
            if v is None:
                continue
            ax.bar(i, v - 1, bottom=1, color=cr._color_of(l, i), width=0.7)
            ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_title(s, fontsize=8.5)
        ax.set_xticks([])
        ax.set_ylim(1, n + 0.4)
        ax.axhline((n + 1) / 2, color="#9aa0a6", lw=0.8, ls="--")
    axes[0].set_ylabel("mean rank\n(1 = best of group)")
    fig.suptitle("Mean head-to-head rank per evaluation suite (dashed line = middle of the field)",
                 fontsize=9.5, y=1.04)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cr._color_of(l, i))
                        for i, l in enumerate(labels)],
               labels=labels, loc="lower center", ncol=min(5, n), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.16), frameon=False)
    _save(fig, outdir, "fig_rank_mean_by_suite")
    plt.close(fig)


def fig_tier_profile(plt, labels, stats, outdir):
    """ONE message: who is preferred where it matters most — mean rank per principle tier."""
    tiers = [1, 2, 3]
    if not any((stats[l].get("tiers") or {}) for l in labels):
        print("  [skip] fig_rank_tier_profile (no constitution verdicts)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.5), sharey=True)
    n = len(labels)
    for ax, t in zip(axes, tiers):
        for i, l in enumerate(labels):
            v = ((stats[l].get("tiers") or {}).get(t) or {}).get("mean_rank")
            if v is None:
                continue
            ax.bar(i, v - 1, bottom=1, color=cr._color_of(l, i), width=0.7)
            ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_title(f"Tier {t} (×{int(pf.TIER_WEIGHTS[t])})", fontsize=8.5)
        ax.set_xticks([])
        ax.set_ylim(1, n + 0.4)
        ax.axhline((n + 1) / 2, color="#9aa0a6", lw=0.8, ls="--")
    axes[0].set_ylabel("mean rank\n(1 = best of group)")
    fig.suptitle("Mean rank by principle importance tier — constitution suite", fontsize=9.5, y=1.04)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cr._color_of(l, i))
                        for i, l in enumerate(labels)],
               labels=labels, loc="lower center", ncol=min(5, n), fontsize=7.5,
               bbox_to_anchor=(0.5, -0.16), frameon=False)
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
              labels=[f"rank {r + 1}" + (" (best)" if r == 0 else " (worst)" if r == n - 1 else "")
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
    """ONE message: per-principle strengths and weaknesses — mean rank per constitution
    principle × condition, rows grouped by importance tier."""
    import numpy as np
    acc: Dict[Tuple[str, str], List[int]] = {}
    for it in items:
        if it["suite"] != "constitution":
            continue
        gid = it["key"].partition("::")[0]
        for l, r in _ranks_of(it).items():
            acc.setdefault((gid, l), []).append(r)
    if not acc:
        print("  [skip] fig_rank_principle_heatmap (no constitution verdicts)")
        return
    gids = sorted({g for g, _ in acc}, key=lambda g: (pf.tier_of(g), g))
    mat = np.array([[np.mean(acc[(g, l)]) if (g, l) in acc else np.nan for l in labels]
                    for g in gids])
    fig, ax = plt.subplots(figsize=(4.6, 0.28 * len(gids) + 1.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=1, vmax=len(labels), aspect="auto")
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
                ax.text(xi, yi, f"{mat[yi, xi]:.1f}", ha="center", va="center", fontsize=6,
                        color="#1c1e21")
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("mean rank (1 = ranked best of the group, "
                 f"{len(labels)} = worst)", fontsize=7)
    ax.set_title("Mean head-to-head rank per constitution principle\n"
                 "(rows grouped by importance tier; black lines separate tiers)", fontsize=9)
    _save(fig, outdir, "fig_rank_principle_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------- tables

def _tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def table_overall(labels, stats, outdir) -> None:
    rows = []
    for l in labels:
        m = stats[l]
        f = lambda v, d=2: (f"{v:.{d}f}" if isinstance(v, (int, float)) else "--")
        rows.append(f"    {_tex_escape(l)} & {m['n']} & {f(m['wins'], 1)} & {f(m['mean_rank'])} & "
                    f"{f(m['borda'], 0)} & {f(m['weighted_mean_rank'])} & "
                    f"{f(m['weighted_win_share'])} \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Blind head-to-head comparison of the five conditions. For every benchmark "
        "question the anonymised answers of all conditions were ranked by the LLM judge "
        "(1 = best of the group; equal rubric grades share a rank). \\emph{Wins} counts "
        "first-place finishes, a $k$-way tie sharing $1/k$; \\emph{mean rank} averages the "
        "finishing position over all ranked questions (lower is better; 3.0 is the middle of a "
        "five-condition field); \\emph{Borda} awards $4,3,2,1,0$ points from best to worst "
        "(tie groups share the mean), rewarding consistently high placement; the "
        "\\emph{weighted} columns restrict to the constitution suite and weight each question "
        "by its principle's a-priori importance tier ($\\times 3/\\times 2/\\times 1$), so the "
        "\\emph{weighted win share} is the fraction of importance-weighted question mass on "
        "which the condition gave the preferred answer.}\n"
        "  \\label{tab:rank-overall}\n"
        "  \\begin{tabular}{lrrrrrr}\n    \\toprule\n"
        "    condition & $n$ & wins & mean rank & Borda & wtd.\\ mean rank & wtd.\\ win share \\\\\n"
        "    \\midrule\n" + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    path = os.path.join(outdir, "tab_rank_overall.tex")
    open(path, "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_overall.tex")


def table_tiers(labels, stats, outdir) -> None:
    if not any((stats[l].get("tiers") or {}) for l in labels):
        return
    rows = []
    for l in labels:
        cells = [_tex_escape(l)]
        for t in (1, 2, 3):
            d = (stats[l].get("tiers") or {}).get(t) or {}
            mr, w = d.get("mean_rank"), d.get("wins")
            cells.append(f"{mr:.2f}" if mr is not None else "--")
            cells.append(f"{w:.1f}" if w is not None else "--")
        rows.append("    " + " & ".join(cells) + " \\\\")
    tex = (
        "\\begin{table}[t]\n  \\centering\n"
        "  \\caption{Head-to-head performance by principle importance tier (constitution "
        "suite). Tier~1 ($\\times 3$) holds the trust-critical outcomes (ask the right "
        "question, never fabricate, deny when unknown), Tier~2 ($\\times 2$) the reasoning/"
        "robustness/personalisation substrate, Tier~3 ($\\times 1$) the instrumental tool "
        "mechanism. \\emph{rank} is the mean finishing position among the five conditions "
        "(1 = preferred answer, lower is better); \\emph{wins} counts first places with ties "
        "shared. A condition can lead overall yet trail on Tier~1 — this table shows who wins "
        "where it matters most.}\n"
        "  \\label{tab:rank-tiers}\n"
        "  \\begin{tabular}{lrrrrrr}\n    \\toprule\n"
        "    & \\multicolumn{2}{c}{Tier 1 ($\\times 3$)} & \\multicolumn{2}{c}{Tier 2 "
        "($\\times 2$)} & \\multicolumn{2}{c}{Tier 3 ($\\times 1$)} \\\\\n"
        "    condition & rank & wins & rank & wins & rank & wins \\\\\n    \\midrule\n"
        + "\n".join(rows) + "\n    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    open(os.path.join(outdir, "tab_rank_tiers.tex"), "w", encoding="utf-8").write(tex)
    print("  wrote tab_rank_tiers.tex")


_CAPTIONS = r"""% rank_figures_captions.tex — ready \begin{figure} blocks for the rank figures.
% Regenerable via: python rank_figures.py --reports_dir results
% Metric primer (reuse in prose): each benchmark question was answered by all five conditions;
% the anonymised answers were ranked head-to-head by the LLM judge. rank 1 = preferred answer
% of the group; grades (A+..E) are ABSOLUTE rubric quality, independent of the other answers.

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
  \includegraphics[width=\linewidth]{fig_rank_mean_by_suite.pdf}
  \caption{Mean head-to-head rank per evaluation suite (1 = ranked best of the five, 5 =
  worst; dashed line marks the middle of the field at 3). Bars below the dashed line are
  better than the average condition. Consistent bar ordering across panels indicates the
  preference is stable across capability areas rather than driven by one suite.}
  \label{fig:rank-mean-by-suite}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_tier_profile.pdf}
  \caption{Mean head-to-head rank by principle importance tier (constitution suite). Tier~1
  holds the trust-critical outcomes, Tier~2 the reasoning and personalisation substrate,
  Tier~3 the tool mechanism. Lower is better; the dashed line is the middle of the field. A
  condition that leads on Tier~1 gives the preferred answer where failures damage user trust
  the most.}
  \label{fig:rank-tier-profile}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_distribution.pdf}
  \caption{Distribution of finishing positions per condition across all ranked questions.
  Green (rank 1) means the condition's answer was preferred; red (rank 5) means it was judged
  worst of the group. A polarised profile (mass at both ends) indicates behaviour that
  either matches the probed principle well or fails it outright, whereas a mid-heavy profile
  indicates consistent mediocrity.}
  \label{fig:rank-distribution}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.9\linewidth]{fig_rank_grade_profile.pdf}
  \caption{Absolute quality grades per condition, assigned by the judge on a fixed rubric
  before ranking: A+ ideal behaviour and correct substance; A minor blemish; B+/B honest and
  substantially correct with the preferred behaviour partly delivered; C+/C right direction
  but failed execution; D misses the principle without fabrication; E dishonest or harmful
  (fabricated facts, sources, or tool results). Unlike ranks, grades are independent of the
  other answers, so this figure shows each condition's quality in absolute terms, calibrated
  to the sub-1B model class.}
  \label{fig:rank-grade-profile}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=.85\linewidth]{fig_rank_principle_heatmap.pdf}
  \caption{Mean head-to-head rank per constitution principle (rows, grouped by importance
  tier) and condition (columns). Green cells mark principles where the condition's answers
  were consistently preferred; red cells mark principles where they were consistently judged
  worst. The heatmap localises each condition's strengths and failure modes at the level of
  individual constitutional behaviours.}
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
    fig_mean_rank_by_suite(plt, labels, stats, outdir)
    fig_tier_profile(plt, labels, stats, outdir)
    fig_rank_distribution(plt, labels, items, outdir)
    fig_grade_profile(plt, labels, items, outdir)
    fig_principle_heatmap(plt, labels, items, outdir)
    table_overall(labels, stats, outdir)
    table_tiers(labels, stats, outdir)
    open(os.path.join(outdir, "rank_figures_captions.tex"), "w", encoding="utf-8").write(_CAPTIONS)
    print("  wrote rank_figures_captions.tex")
    print(f"\nDone -> {outdir}")


if __name__ == "__main__":
    main()
