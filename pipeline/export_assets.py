"""Export dissertation-ready assets (LaTeX tables, summary JSON, figures) from the
benchmark reports in pipeline/reports/.

WHY THIS EXISTS: analysis.ipynb produces interactive Plotly figures only — nothing
lands in a form a LaTeX dissertation can \\input or \\includegraphics. This script
turns the existing report JSON into:
  * LaTeX tables  -> reports/dissertation_assets/tab_*.tex   (\\input-able)
  * a summary     -> reports/dissertation_assets/summary.json (every aggregate)
  * figures (PDF) -> reports/dissertation_assets/fig_*.pdf    (if matplotlib present)

It also computes the statistics the dissertation's Statistical Analysis Plan promises
but the pipeline did not previously calculate: paired McNemar p-values, Wilson 95% CIs,
and Cohen's h effect sizes. No third-party stats dependency is required.

Usage:
    python export_assets.py                # auto-pick latest vanilla + sft probe pair
    python export_assets.py --vanilla X.json --sft Y.json
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import principle_families as pf

REPORTS = os.path.join(os.path.dirname(__file__), "reports")
OUTDIR = os.path.join(REPORTS, "dissertation_assets")


# ----------------------------------------------------------------------------- stats
def wilson_ci(passes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def cohens_h(p1, p2):
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return phi(p1) - phi(p2)


# ----------------------------------------------------------------------------- io
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest(patterns, must_not=()):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(REPORTS, pat))
    files = [f for f in files if not any(m in os.path.basename(f) for m in must_not)]
    return sorted(files)[-1] if files else None


def _label(path):
    try:
        return (load(path).get("run_metadata", {}).get("model_label") or "").lower()
    except Exception:
        return ""


def _is_sft_label(lbl):
    return any(t in lbl for t in ("sft", "trustworthy-ai", "checkpoint_sft", "thinker", "executor"))


def _is_base_label(lbl):
    return ("qwen3-0.6b" in lbl or "qwen3-0.6B".lower() in lbl) and not _is_sft_label(lbl)


def pick_by_label(kind):
    """Select a (vanilla, sft) pair by run_metadata.model_label, NOT filename.

    Filenames are unreliable: several base-model re-runs are NOT marked 'vanilla',
    so a filename heuristic silently pairs vanilla-vs-vanilla. We select by the
    label the server recorded.
    """
    files = sorted(glob.glob(os.path.join(REPORTS, f"{kind}_*.json")))
    base = [f for f in files if _is_base_label(_label(f))]
    sftf = [f for f in files if _is_sft_label(_label(f))]
    van = base[-1] if base else None
    sft = sftf[-1] if sftf else None
    return van, sft


def pick_pair(kind):
    return pick_by_label(kind)


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(name, body, caption, label):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  wrote {os.path.relpath(path)}  ({caption})")


# ----------------------------------------------------------------------------- think collapse
def think_stats(report):
    lengths, empties = [], []
    for grp in report.get("probe_results", []):
        for q in grp.get("question_results", []):
            if "think_length" in q:
                lengths.append(q["think_length"])
                empties.append(1 if q.get("think_empty") else 0)
    if not lengths:
        return None
    return {
        "n": len(lengths),
        "mean_think_chars": round(sum(lengths) / len(lengths), 1),
        "pct_empty": round(100 * sum(empties) / len(empties), 1),
    }


def paired_outcomes(van, sft):
    """Pair per (principle, question_idx) -> (vanilla_pass, sft_pass) booleans."""
    def index(rep):
        out = {}
        for grp in rep.get("probe_results", []):
            pid = grp.get("id")
            for q in grp.get("question_results", []):
                out[(pid, q.get("question_idx"))] = bool(q.get("rule_passed"))
        return out
    vi, si = index(van), index(sft)
    keys = sorted(set(vi) & set(si))
    return [(vi[k], si[k]) for k in keys]


# ----------------------------------------------------------------------------- builders
def build_per_principle(summary, van, sft):
    vs, ss = van["scores_by_principle"], sft["scores_by_principle"]
    rows = []
    for pid in pf.PRINCIPLES:
        if pid not in vs or pid not in ss:
            continue
        v, s = vs[pid], ss[pid]
        rows.append((pid, pf.display_of(pid), pf.family_of(pid), pf.framing_of(pid),
                     v, s, round(s - v, 4), cohens_h(s, v)))
    summary["per_principle"] = [
        {"id": r[0], "family": r[2], "framing": r[3],
         "vanilla": r[4], "sft": r[5], "delta": r[6], "cohens_h": round(r[7], 3)}
        for r in rows
    ]
    # LaTeX
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Principle & Family & Vanilla & SFT & $\Delta$ & Cohen's $h$ \\", r"\midrule"]
    for _id, disp, fam, _fr, v, s, d, h in rows:
        arrow = r"\,$\uparrow$" if d > 0 else (r"\,$\downarrow$" if d < 0 else "")
        lines.append(f"{tex_escape(disp)} & {fam} & {v:.2f} & {s:.2f} & "
                     f"{d:+.2f}{arrow} & {h:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_h1_comparison.tex", "\n".join(lines),
                "per-principle vanilla vs SFT", "tab:h1-comparison")


def build_per_family(summary, van, sft):
    vf = pf.aggregate_by_family(van["scores_by_principle"])
    sf = pf.aggregate_by_family(sft["scores_by_principle"])
    summary["per_family"] = {
        fam: {"vanilla": round(vf.get(fam, (float("nan"),))[0], 4),
              "sft": round(sf.get(fam, (float("nan"),))[0], 4),
              "n": vf.get(fam, (0, 0))[1]}
        for fam in pf.FAMILIES if fam in vf
    }
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Family & $n$ & Vanilla & SFT & $\Delta$ \\", r"\midrule"]
    for fam in pf.FAMILIES:
        if fam not in vf:
            continue
        v, n = vf[fam]
        s = sf.get(fam, (float("nan"), n))[0]
        lines.append(f"{pf.FAMILY_LABELS[fam]} & {n} & {v:.2f} & {s:.2f} & {s - v:+.2f} \\\\")
    # framing split (C3AI prediction check)
    vfr, sfr = pf.aggregate_by_framing(van["scores_by_principle"]), pf.aggregate_by_framing(sft["scores_by_principle"])
    lines.append(r"\midrule")
    for fr in ("positive", "negative", "calibration"):
        if fr in vfr:
            v, n = vfr[fr]
            s = sfr.get(fr, (float("nan"), n))[0]
            lines.append(f"\\emph{{{fr}-framed}} & {n} & {v:.2f} & {s:.2f} & {s - v:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    summary["per_framing"] = {fr: {"vanilla": round(vfr[fr][0], 4),
                                   "sft": round(sfr.get(fr, (float('nan'),))[0], 4)}
                              for fr in vfr}
    write_table("tab_per_family.tex", "\n".join(lines),
                "per-family + framing split", "tab:per-family")


def build_significance(summary, van, sft):
    pairs = paired_outcomes(van, sft)
    b = sum(1 for v, s in pairs if v and not s)   # vanilla pass, sft fail
    c = sum(1 for v, s in pairs if not v and s)   # vanilla fail, sft pass
    n = len(pairs)
    vpass = sum(1 for v, _ in pairs if v)
    spass = sum(1 for _, s in pairs if s)
    summary["significance"] = {
        "n_paired_probes": n,
        "vanilla_pass": vpass, "sft_pass": spass,
        "vanilla_rate": round(vpass / n, 4) if n else None,
        "sft_rate": round(spass / n, 4) if n else None,
        "vanilla_wilson_ci": [round(x, 4) for x in wilson_ci(vpass, n)],
        "sft_wilson_ci": [round(x, 4) for x in wilson_ci(spass, n)],
        "mcnemar_b_van_only": b, "mcnemar_c_sft_only": c,
        "mcnemar_p_exact": round(mcnemar_exact_p(b, c), 5),
        "cohens_h_overall": round(cohens_h(spass / n, vpass / n), 3) if n else None,
    }


def build_think_collapse(summary, van, sft):
    tv, ts = think_stats(van), think_stats(sft)
    summary["think_collapse"] = {"vanilla": tv, "sft": ts}
    if not (tv and ts):
        return
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Condition & Mean \texttt{<think>} chars & \% empty \\", r"\midrule",
             f"Vanilla & {tv['mean_think_chars']:.0f} & {tv['pct_empty']:.0f}\\% \\\\",
             f"SFT & {ts['mean_think_chars']:.0f} & {ts['pct_empty']:.0f}\\% \\\\",
             r"\bottomrule", r"\end{tabular}"]
    write_table("tab_think_collapse.tex", "\n".join(lines),
                "reasoning-trace collapse", "tab:think-collapse")


def sibling(kind, probe_path):
    """Given a constitution_probe_<suffix>.json, return <kind>_<suffix>.json if it
    exists. adversarial/category/drift reports lack model_label, but share the
    timestamp suffix of the constitution run from the SAME benchmark batch — so
    batch-matching by suffix is the reliable way to pair them."""
    base = os.path.basename(probe_path)
    suffix = base.replace("constitution_probe", "", 1)  # e.g. _vanilla_20260525_170847.json
    cand = os.path.join(REPORTS, f"{kind}{suffix}")
    return cand if os.path.exists(cand) else None


def build_adversarial(summary, vpath, spath):
    van = sibling("adversarial", vpath)
    sft = sibling("adversarial", spath)
    if not sft:
        return
    rep_s = load(sft)
    rep_v = load(van) if van else None
    summary["adversarial"] = {
        "sft_score": rep_s.get("adversarial_score"),
        "vanilla_score": rep_v.get("adversarial_score") if rep_v else None,
        "sft_by_type": {k: v["score"] for k, v in rep_s.get("score_by_attack_type", {}).items()},
    }
    types = sorted(rep_s.get("score_by_attack_type", {}))
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Attack type & Vanilla & SFT \\", r"\midrule"]
    for t in types:
        sv = rep_s["score_by_attack_type"][t]["score"]
        vv = (rep_v or {}).get("score_by_attack_type", {}).get(t, {}).get("score") if rep_v else None
        vtxt = f"{vv:.2f}" if isinstance(vv, (int, float)) else "--"
        lines.append(f"{tex_escape(t)} & {vtxt} & {sv:.2f} \\\\")
    lines.append(r"\midrule")
    vtot = f"{rep_v.get('adversarial_score'):.2f}" if rep_v else "--"
    lines.append(f"\\textbf{{Overall}} & {vtot} & {rep_s.get('adversarial_score'):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_adversarial.tex", "\n".join(lines),
                "adversarial by attack type", "tab:adversarial")


def build_category(summary, vpath, spath):
    van = sibling("category_probes", vpath)
    sft = sibling("category_probes", spath)
    if not sft:
        return
    rep_s = load(sft)
    rep_v = load(van) if van else None
    sc_s = rep_s.get("scores_by_category", {})
    sc_v = (rep_v or {}).get("scores_by_category", {})
    summary["category"] = {"sft_score": rep_s.get("category_score"),
                           "vanilla_score": (rep_v or {}).get("category_score"),
                           "sft_by_category": sc_s}
    cats = sorted(set(sc_s) | set(sc_v))
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Category & Vanilla & SFT \\", r"\midrule"]
    for c in cats:
        vv = sc_v.get(c)
        sv = sc_s.get(c)
        vt = f"{vv:.2f}" if isinstance(vv, (int, float)) else "--"
        st = f"{sv:.2f}" if isinstance(sv, (int, float)) else "--"
        lines.append(f"{tex_escape(c)} & {vt} & {st} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_category.tex", "\n".join(lines),
                "category coverage (Suite B)", "tab:category")


def build_retries(summary, sft):
    # per-principle avg retries from the sft probe report
    per = {}
    for grp in sft.get("probe_results", []):
        rs = [q.get("harness_retries", 0) for q in grp.get("question_results", [])]
        if rs:
            per[grp["id"]] = round(sum(rs) / len(rs), 2)
    hm_path = os.path.join(REPORTS, "harness_metrics.json")
    hm = load(hm_path) if os.path.exists(hm_path) else {}
    summary["retries"] = {"avg_by_principle": per,
                          "harness_metrics": {k: hm.get(k) for k in
                                              ("request_count", "total_retries", "retry_success_rate")}}


def build_runs(summary):
    rows = []
    for pat, kind in (("constitution_probe_*.json", "constitution"),
                      ("adversarial_*.json", "adversarial"),
                      ("category_probes_*.json", "category"),
                      ("context_drift_*.json", "drift")):
        for p in sorted(glob.glob(os.path.join(REPORTS, pat))):
            r = load(p)
            meta = r.get("run_metadata", {})
            score = (r.get("constitution_score") or r.get("adversarial_score")
                     or r.get("category_score") or r.get("drift_score"))
            rows.append((os.path.basename(p), kind,
                         meta.get("model_label", "?"), r.get("timestamp", "?"),
                         score))
    summary["runs"] = [{"file": a, "kind": b, "model": c, "timestamp": d, "score": e}
                       for a, b, c, d, e in rows]
    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Kind & Model & Timestamp & Score \\", r"\midrule"]
    for _f, kind, model, ts, score in rows:
        st = f"{score:.3f}" if isinstance(score, (int, float)) else "--"
        lines.append(f"{kind} & {tex_escape(model)[:28]} & {tex_escape(ts)} & {st} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab_runs.tex", "\n".join(lines), "all benchmark runs", "tab:runs")


# ----------------------------------------------------------------------------- figures
def build_figures(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [figures skipped: matplotlib unavailable: {e}]")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    # think collapse
    tc = summary.get("think_collapse", {})
    if tc.get("vanilla") and tc.get("sft"):
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].bar(["Vanilla", "SFT"], [tc["vanilla"]["mean_think_chars"], tc["sft"]["mean_think_chars"]],
                  color=["#4c72b0", "#c44e52"])
        ax[0].set_title("Mean <think> length (chars)")
        ax[1].bar(["Vanilla", "SFT"], [tc["vanilla"]["pct_empty"], tc["sft"]["pct_empty"]],
                  color=["#4c72b0", "#c44e52"])
        ax[1].set_title("% empty <think>")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "fig_think_collapse.pdf"))
        plt.close(fig)
        print("  wrote fig_think_collapse.pdf")
    # per-family bars
    pfam = summary.get("per_family", {})
    if pfam:
        fams = list(pfam)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        x = range(len(fams))
        ax.bar([i - 0.2 for i in x], [pfam[f]["vanilla"] for f in fams], width=0.4, label="Vanilla")
        ax.bar([i + 0.2 for i in x], [pfam[f]["sft"] for f in fams], width=0.4, label="SFT")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f[:8] for f in fams], rotation=20)
        ax.set_ylabel("compliance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "fig_per_family.pdf"))
        plt.close(fig)
        print("  wrote fig_per_family.pdf")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla")
    ap.add_argument("--sft")
    args = ap.parse_args()

    auto_v, auto_s = pick_by_label("constitution_probe")
    vpath = args.vanilla or auto_v
    spath = args.sft or auto_s
    if not (vpath and spath):
        print("ERROR: need a vanilla (base-labelled) and an SFT-labelled constitution_probe report.")
        print(f"  vanilla={vpath}  sft={spath}")
        return
    van, sft = load(vpath), load(spath)
    vlabel = van.get("run_metadata", {}).get("model_label", "?")
    slabel = sft.get("run_metadata", {}).get("model_label", "?")
    print(f"vanilla = {os.path.basename(vpath)}   [model_label: {vlabel}]")
    print(f"sft     = {os.path.basename(spath)}   [model_label: {slabel}]")
    # Hygiene warnings — surface the exact problems found during reconciliation.
    if _is_base_label(slabel.lower()):
        print("  !! WARNING: the 'sft' run is labelled as the BASE model — not a fine-tuned"
              " checkpoint. Pass --sft explicitly with a trustworthy-ai-sft run.")
    nv = sum(len(g.get("question_results", [])) for g in van.get("probe_results", []))
    ns = sum(len(g.get("question_results", [])) for g in sft.get("probe_results", []))
    if nv != ns:
        print(f"  !! WARNING: probe counts differ (vanilla n={nv}, sft n={ns}). The paired"
              " comparison uses only matched (principle, question_idx) pairs; rerun both"
              " conditions with the SAME probe config for a clean 66-probe comparison.")

    summary = {"sources": {"vanilla": os.path.basename(vpath), "sft": os.path.basename(spath)},
               "principle_count_note": pf.PRINCIPLE_COUNT_NOTE,
               "unprobed_principles": pf.UNPROBED_PRINCIPLES}

    os.makedirs(OUTDIR, exist_ok=True)
    for fn in (lambda: build_per_principle(summary, van, sft),
               lambda: build_per_family(summary, van, sft),
               lambda: build_significance(summary, van, sft),
               lambda: build_think_collapse(summary, van, sft),
               lambda: build_adversarial(summary, vpath, spath),
               lambda: build_category(summary, vpath, spath),
               lambda: build_retries(summary, sft),
               lambda: build_runs(summary)):
        try:
            fn()
        except Exception as e:
            print(f"  [skip asset: {type(e).__name__}: {e}]")

    build_figures(summary)

    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary.json written to {os.path.relpath(OUTDIR)}")
    sig = summary.get("significance", {})
    tc = summary.get("think_collapse", {})
    print(f"  overall: vanilla {sig.get('vanilla_rate')} vs sft {sig.get('sft_rate')} "
          f"(McNemar p={sig.get('mcnemar_p_exact')}, h={sig.get('cohens_h_overall')})")
    if tc.get("vanilla") and tc.get("sft"):
        print(f"  think:   {tc['vanilla']['mean_think_chars']}->{tc['sft']['mean_think_chars']} chars, "
              f"empty {tc['vanilla']['pct_empty']}%->{tc['sft']['pct_empty']}%")


if __name__ == "__main__":
    main()
