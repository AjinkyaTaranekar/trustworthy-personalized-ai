#!/usr/bin/env python3
"""
analyze_experiments.py — Cross-condition benchmark consolidator (zero GPU, offline)
===================================================================================

Consolidates the per-condition `4_benchmark.py` report JSONs for the five-rung
ablation ladder into one comparison, with the *isolating deltas* that are the
dissertation's argument:

    C0 vanilla_base      (base model, tools effectively off)
    C1 vanilla_tools     (base model, native tool calling)   Δ = value of tool access
    C2 sft_template      (Exp 1: template-based SFT)          Δ = value of SFT format scaffolding
    C3 sft_constitution  (Exp 2: constitutional SFT)          Δ = value of constitutional content   ← H1
    C4 thinker_executor  (Exp 3: dual-adapter framework)      Δ = value of the architecture          ← H2

All five conditions are benchmarked through the SAME `4_benchmark.py` suites (the
dual model is served by `3_infererence.py --thinker --executor` on the same
endpoint), so every condition's reports share one schema and pair cleanly here.

Scoring discipline (supervisor flagged LLM-graded-LLM circularity):
  * Constitutional suite is reported as RULE-based (deterministic, primary) AND
    combined (rule+judge, secondary) in separate rows — never blended silently.
  * Bootstrap CIs accompany each adjacent delta because n is deliberately small.

Usage
-----
    # Auto-discover the latest report of each suite per label, in ladder order:
    python analyze_experiments.py \
        --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor

    # Restrict to specific suites, custom dirs:
    python analyze_experiments.py --labels base sft dual \
        --suites constitution persona --reports_dir reports --output_dir reports

Outputs (under --output_dir):
    experiment_ladder_<ts>.csv    — machine-readable ladder (one row per metric)
    experiment_ladder_<ts>.tex    — LaTeX tabular for the dissertation
    experiment_h3_failures_<ts>.csv — probes the top rung still fails / regresses on
"""

import argparse
import csv
import glob
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import experiment_metrics as em  # judge-free depth/tool metrics (single source of truth)
import principle_families as pf  # a-priori family + purpose-tier weighting

# Suite -> report filename prefix written by 4_benchmark.py (`<prefix>_<label>_<ts>.json`).
SUITE_PREFIX: Dict[str, str] = {
    "constitution": "constitution_probe",
    "categories":   "category_probes",
    "drift":        "context_drift",
    "adversarial":  "adversarial",
    "persona":      "persona_conversations",
}
ALL_SUITES = list(SUITE_PREFIX)

PERSONA_DIMENSIONS = [
    "personalisation", "memory_consistency", "empathy",
    "trustworthiness", "coherence", "goal_completion",
]


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

def find_report(reports_dir: Path, suite: str, label: str) -> Optional[Path]:
    """Latest report for this suite/label. Supports two layouts:
      A. per-condition subdir:  <reports_dir>/<label>/<prefix>_*.json   (what the runbook uses —
         each single-server benchmark run writes to `--output_dir reports/<label>`)
      B. label-in-filename:     <reports_dir>/<prefix>_<label>_*.json    (the `--models` hot-swap mode)
    Subdir layout wins when present. Derived artefacts (*_rescored, *_alignment) are excluded."""
    prefix = SUITE_PREFIX[suite]

    def _latest(paths):
        paths = [p for p in sorted(paths) if not p.stem.endswith(("_rescored", "_alignment"))]
        return paths[-1] if paths else None

    sub = reports_dir / label
    if sub.is_dir():
        hit = _latest(sub.glob(f"{prefix}_*.json"))
        if hit:
            return hit
    return _latest(reports_dir.glob(f"{prefix}_{label}_*.json"))


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-suite extraction: headline metric(s) + item-level pass/fail for bootstrap
# ---------------------------------------------------------------------------

def _const_rule_items(report: Dict[str, Any]) -> Dict[str, float]:
    """{group_id::q_idx -> 1.0/0.0} from rule_passed — the deterministic signal."""
    items: Dict[str, float] = {}
    for gr in report.get("probe_results", []):
        gid = gr.get("id", "?")
        for qr in gr.get("question_results", []):
            key = f"{gid}::{qr.get('question_idx')}"
            items[key] = 1.0 if qr.get("rule_passed") else 0.0
    return items


def _const_combined_items(report: Dict[str, Any]) -> Dict[str, float]:
    items: Dict[str, float] = {}
    for gr in report.get("probe_results", []):
        gid = gr.get("id", "?")
        for qr in gr.get("question_results", []):
            cs = qr.get("combined_score")
            if cs is not None:
                items[f"{gid}::{qr.get('question_idx')}"] = float(cs)
    return items


def _adversarial_items(report: Dict[str, Any]) -> Dict[str, float]:
    return {p.get("id", str(i)): (1.0 if p.get("passed") else 0.0)
            for i, p in enumerate(report.get("probe_results", []))}


def _persona_items(report: Dict[str, Any]) -> Dict[str, float]:
    items: Dict[str, float] = {}
    for r in report.get("persona_results", []):
        ov = (r.get("judge") or {}).get("overall")
        if ov is not None:
            items[r.get("persona_id", "?")] = float(ov)
    return items


# (headline_label, getter(report) -> Optional[float], item_getter(report) -> {key: val})
SUITE_METRICS: Dict[str, List[Tuple[str, Any, Any]]] = {
    "constitution": [
        ("constitution (rule)",
         lambda r: _mean_or_none(_const_rule_items(r).values()),
         _const_rule_items),
        ("constitution (combined)",
         lambda r: r.get("constitution_score"),
         _const_combined_items),
        # --- purpose-weighted "trustworthiness" score (a-priori tiers; principle_families) ---
        # Headline weighting requested by the supervisor; reported ALONGSIDE the unweighted
        # rows above and the per-tier breakdown below (never instead of them).
        ("constitution (rule, purpose-weighted)",
         lambda r: em.constitution_metrics(r).get("overall_weighted"), lambda r: {}),
        ("constitution (combined, purpose-weighted)",
         lambda r: pf.aggregate_weighted(r.get("scores_by_principle") or {}), lambda r: {}),
        ("  tier-1 trust-critical (rule)",
         lambda r: (em.constitution_metrics(r).get("by_tier") or {}).get(1), lambda r: {}),
        ("  tier-2 pressure+personalise (rule)",
         lambda r: (em.constitution_metrics(r).get("by_tier") or {}).get(2), lambda r: {}),
        ("  tier-3 tool/reasoning mech (rule)",
         lambda r: (em.constitution_metrics(r).get("by_tier") or {}).get(3), lambda r: {}),
        # --- judge-free depth & tool-behaviour rows (experiment_metrics.py) ---
        # Structural proxies computed from the constitution report's per-response
        # signals; bootstrap deltas apply where item-level vectors exist.
        ("reasoning: <think> length (chars)",
         lambda r: em.depth_metrics(r).get("think_len_mean"), em.think_len_items),
        ("reasoning: <think>-empty rate",
         lambda r: em.depth_metrics(r).get("think_empty_rate"), lambda r: {}),
        ("reasoning: externalisation ratio",
         lambda r: em.depth_metrics(r).get("ext_ratio_mean"), em.ext_ratio_items),
        ("reasoning: clarification rate",
         lambda r: em.depth_metrics(r).get("clarification_rate"), em.clarification_items),
        ("validity: hollow-pass rate",
         lambda r: em.depth_metrics(r).get("hollow_pass_rate"), em.hollow_pass_items),
        ("tool: calls / response",
         lambda r: em.tool_metrics(r).get("tool_calls_per_resp"), lambda r: {}),
        ("tool: failure rate",
         lambda r: em.tool_metrics(r).get("tool_fail_rate"), lambda r: {}),
        ("tool: decoy-bait rate",
         lambda r: em.tool_metrics(r).get("decoy_bait_rate"), lambda r: {}),
    ],
    "categories": [
        ("category coverage", lambda r: r.get("category_score"), lambda r: {}),
    ],
    "drift": [
        ("context drift (adherence)", lambda r: r.get("drift_score"),
         lambda r: {str(i): v for i, v in enumerate(r.get("adherence_curve") or [])}),
    ],
    "adversarial": [
        ("adversarial resistance", lambda r: r.get("adversarial_score"), _adversarial_items),
    ],
    "persona": [
        ("persona conversation", lambda r: r.get("persona_score"), _persona_items),
    ],
}


def _mean_or_none(vals) -> Optional[float]:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Bootstrap paired delta CI
# ---------------------------------------------------------------------------

def bootstrap_delta(items_a: Dict[str, float], items_b: Dict[str, float],
                    n: int = 2000, alpha: float = 0.05,
                    seed: int = 42) -> Optional[Tuple[float, float, float]]:
    """Paired bootstrap of mean(b) - mean(a) over the shared item keys.

    Returns (delta, lo, hi) percentile CI, or None if there is no shared paired data."""
    keys = [k for k in items_a if k in items_b]
    if not keys:
        return None
    a = [items_a[k] for k in keys]
    b = [items_b[k] for k in keys]
    delta = statistics.fmean(b) - statistics.fmean(a)
    rng = random.Random(seed)
    m = len(keys)
    deltas: List[float] = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        deltas.append(statistics.fmean([b[i] for i in idx]) - statistics.fmean([a[i] for i in idx]))
    deltas.sort()
    lo = deltas[int((alpha / 2) * n)]
    hi = deltas[min(n - 1, int((1 - alpha / 2) * n))]
    return (delta, lo, hi)


# ---------------------------------------------------------------------------
# Build the ladder
# ---------------------------------------------------------------------------

def build_ladder(labels: List[str], suites: List[str], reports_dir: Path,
                 bootstrap_n: int = 2000) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Path]]]:
    """Returns (rows, found_paths). Each row = one metric across all conditions,
    plus adjacent isolating deltas (with bootstrap CI where item data exists)."""
    found: Dict[str, Dict[str, Path]] = {lbl: {} for lbl in labels}
    reports: Dict[str, Dict[str, Dict[str, Any]]] = {lbl: {} for lbl in labels}
    for lbl in labels:
        for suite in suites:
            p = find_report(reports_dir, suite, lbl)
            if p:
                found[lbl][suite] = p
                reports[lbl][suite] = load_json(p)

    rows: List[Dict[str, Any]] = []
    for suite in suites:
        for metric_label, getter, item_getter in SUITE_METRICS[suite]:
            scores: Dict[str, Optional[float]] = {}
            items: Dict[str, Dict[str, float]] = {}
            for lbl in labels:
                rep = reports[lbl].get(suite)
                scores[lbl] = getter(rep) if rep else None
                items[lbl] = item_getter(rep) if rep else {}

            deltas: List[Dict[str, Any]] = []
            for i in range(1, len(labels)):
                prev, cur = labels[i - 1], labels[i]
                pv, cv = scores[prev], scores[cur]
                point = (cv - pv) if (pv is not None and cv is not None) else None
                ci = bootstrap_delta(items[prev], items[cur], n=bootstrap_n) if (
                    items[prev] and items[cur]) else None
                deltas.append({"from": prev, "to": cur, "delta": point,
                               "ci": (ci[1], ci[2]) if ci else None})
            rows.append({"suite": suite, "metric": metric_label,
                         "scores": scores, "deltas": deltas})
    return rows, found


def persona_dimension_matrix(labels: List[str], reports_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """{dimension -> {label -> mean}} from each condition's persona report dimension_means."""
    matrix: Dict[str, Dict[str, Optional[float]]] = {d: {} for d in PERSONA_DIMENSIONS}
    for lbl in labels:
        p = find_report(reports_dir, "persona", lbl)
        means = (load_json(p).get("dimension_means") or {}) if p else {}
        for d in PERSONA_DIMENSIONS:
            matrix[d][lbl] = means.get(d)
    return matrix


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson r over paired samples; None if <3 points or zero variance in either."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def persona_dimension_correlation(labels: List[str], reports_dir: Path
                                  ) -> Tuple[Dict[str, Dict[str, Optional[float]]], int]:
    """Pearson correlation between the six persona dimensions, pooled across every
    (condition, persona) judged sample. Answers the supervisor's "overlapping metrics
    compound" concern: dimensions that correlate near 1.0 are redundant. Returns
    (matrix[d1][d2] -> r, n_samples)."""
    # Collect a per-dimension vector of per-persona scores, pooled across all conditions.
    vectors: Dict[str, List[float]] = {d: [] for d in PERSONA_DIMENSIONS}
    n = 0
    for lbl in labels:
        p = find_report(reports_dir, "persona", lbl)
        if not p:
            continue
        for rec in load_json(p).get("persona_results", []):
            dims = (rec.get("judge") or {}).get("dimensions") or {}
            if not all(isinstance(dims.get(d), dict) and dims[d].get("score") is not None
                       for d in PERSONA_DIMENSIONS):
                continue  # only fully-judged personas contribute a complete row
            n += 1
            for d in PERSONA_DIMENSIONS:
                vectors[d].append(float(dims[d]["score"]))
    matrix = {d1: {d2: (1.0 if d1 == d2 else _pearson(vectors[d1], vectors[d2]))
                   for d2 in PERSONA_DIMENSIONS}
              for d1 in PERSONA_DIMENSIONS}
    return matrix, n


def h3_failure_inventory(labels: List[str], reports_dir: Path) -> List[Dict[str, Any]]:
    """Probes the top rung (last label) still FAILS on, or REGRESSES on vs the prior rung.
    This is the H3 'quantified limits' evidence. Uses rule-based per-principle scores."""
    if len(labels) < 2:
        return []
    top, prev = labels[-1], labels[-2]
    p_top = find_report(reports_dir, "constitution", top)
    p_prev = find_report(reports_dir, "constitution", prev)
    if not p_top:
        return []
    top_scores = load_json(p_top).get("scores_by_principle", {}) or {}
    prev_scores = (load_json(p_prev).get("scores_by_principle", {}) or {}) if p_prev else {}
    inventory: List[Dict[str, Any]] = []
    for pid, s_top in sorted(top_scores.items()):
        s_prev = prev_scores.get(pid)
        fails = s_top < 0.6
        regresses = s_prev is not None and (s_top - s_prev) < -0.05
        if fails or regresses:
            inventory.append({
                "principle": pid,
                f"{prev}": round(s_prev, 4) if s_prev is not None else None,
                f"{top}": round(s_top, 4),
                "delta": round(s_top - s_prev, 4) if s_prev is not None else None,
                "flag": "FAIL+REGRESS" if (fails and regresses) else ("FAIL" if fails else "REGRESS"),
            })
    return inventory


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


def _fmt_delta(d: Dict[str, Any]) -> str:
    if d["delta"] is None:
        return "—"
    s = f"{d['delta']:+.3f}"
    if d["ci"]:
        s += f" [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
    return s


def print_ladder(rows: List[Dict[str, Any]], labels: List[str]) -> None:
    print(f"\n{'='*100}")
    print("  EXPERIMENT LADDER -- five-rung ablation (scores per condition)")
    print(f"{'='*100}")
    header = f"  {'Metric':<26}" + "".join(f"{lbl[:14]:>15}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        line = f"  {r['metric']:<26}" + "".join(_fmt(r["scores"][lbl]).rjust(15) for lbl in labels)
        print(line)
    print("\n  ISOLATING DELTAS (adjacent rungs; [bootstrap 95% CI] where item-level data exists)")
    print("  " + "-" * 96)
    for r in rows:
        print(f"  {r['metric']}:")
        for d in r["deltas"]:
            print(f"      {d['from'][:18]:>18} -> {d['to'][:18]:<18}  {_fmt_delta(d)}")


def write_csv(rows: List[Dict[str, Any]], labels: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (["suite", "metric"] + labels
                  + [f"delta_{labels[i-1]}__{labels[i]}" for i in range(1, len(labels))]
                  + [f"ci_{labels[i-1]}__{labels[i]}" for i in range(1, len(labels))])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row: Dict[str, Any] = {"suite": r["suite"], "metric": r["metric"]}
            for lbl in labels:
                row[lbl] = r["scores"][lbl]
            for d in r["deltas"]:
                row[f"delta_{d['from']}__{d['to']}"] = d["delta"]
                row[f"ci_{d['from']}__{d['to']}"] = (
                    f"[{d['ci'][0]:.4f},{d['ci'][1]:.4f}]" if d["ci"] else "")
            w.writerow(row)
    print(f"\n  CSV   -> {path}")


def write_latex(rows: List[Dict[str, Any]], labels: List[str], path: Path) -> None:
    safe = [lbl.replace("_", r"\_") for lbl in labels]
    lines = [
        r"% Auto-generated by analyze_experiments.py — five-rung ablation ladder.",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Benchmark scores across the five-rung ablation ladder.}",
        r"\label{tab:experiment-ladder}",
        r"\begin{tabular}{l" + "r" * len(labels) + "}",
        r"\toprule",
        "Metric & " + " & ".join(safe) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = " & ".join(_fmt(r["scores"][lbl]).strip() for lbl in labels)
        lines.append(f"{r['metric'].replace('_', chr(92)+'_')} & {cells} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX -> {path}")


def write_h3_csv(inventory: List[Dict[str, Any]], path: Path) -> None:
    if not inventory:
        print("  H3 inventory: top rung fails/regresses on no principles (or no constitution report).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
        w.writeheader()
        w.writerows(inventory)
    print(f"  H3 failure inventory -> {path}  ({len(inventory)} principles)")


def print_persona_matrix(matrix: Dict[str, Dict[str, Optional[float]]], labels: List[str]) -> None:
    if not any(any(v is not None for v in matrix[d].values()) for d in PERSONA_DIMENSIONS):
        return
    print(f"\n  PERSONA DIMENSION MEANS (Suite E)")
    print(f"  {'Dimension':<20}" + "".join(f"{lbl[:14]:>15}" for lbl in labels))
    print("  " + "-" * (20 + 15 * len(labels)))
    for d in PERSONA_DIMENSIONS:
        print(f"  {d:<20}" + "".join(_fmt(matrix[d][lbl]).rjust(15) for lbl in labels))


_DIM_ABBR = {d: d[:5] for d in PERSONA_DIMENSIONS}


def write_dimension_correlation(matrix: Dict[str, Dict[str, Optional[float]]], n: int,
                                path: Path) -> None:
    """Print + CSV the 6x6 persona-dimension correlation (the 'are the metrics redundant?' check).
    |r| ≥ 0.9 between distinct dimensions is flagged — candidates to merge or re-define."""
    if n < 3:
        print(f"\n  PERSONA DIMENSION CORRELATION: skipped (only {n} fully-judged personas; need ≥3).")
        return
    print(f"\n  PERSONA DIMENSION CORRELATION  (Pearson r, pooled over {n} judged personas)")
    hdr = "  " + " " * 20 + "".join(f"{_DIM_ABBR[d]:>8}" for d in PERSONA_DIMENSIONS)
    print(hdr)
    redundant = []
    for d1 in PERSONA_DIMENSIONS:
        cells = ""
        for d2 in PERSONA_DIMENSIONS:
            r = matrix[d1][d2]
            cells += f"{r:>8.2f}" if r is not None else f"{'-':>8}"
            if d1 < d2 and r is not None and abs(r) >= 0.9:
                redundant.append((d1, d2, r))
        print(f"  {d1:<20}{cells}")
    if redundant:
        print("  [!] near-redundant dimension pairs (|r|>=0.9) — consider merging / re-defining:")
        for a, b, r in redundant:
            print(f"        {a} ~ {b}:  r={r:+.2f}")
    else:
        print("  No distinct-dimension pair exceeds |r|=0.9 — dimensions look non-redundant.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension"] + PERSONA_DIMENSIONS)
        for d1 in PERSONA_DIMENSIONS:
            w.writerow([d1] + [("" if matrix[d1][d2] is None else round(matrix[d1][d2], 4))
                               for d2 in PERSONA_DIMENSIONS])
    print(f"  correlation CSV -> {path}  (n={n})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Consolidate per-condition benchmark reports into the ablation ladder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Condition labels in LADDER ORDER (deltas are computed between adjacent rungs).")
    ap.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES,
                    help=f"Suites to include (default: all — {ALL_SUITES}).")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--output_dir", default="reports")
    ap.add_argument("--bootstrap_n", type=int, default=2000)
    ap.add_argument("--figures", action="store_true",
                    help="Also render dissertation figures (experiment_figures.py) to "
                         "<output_dir>/dissertation_assets/. Needs matplotlib.")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows, found = build_ladder(args.labels, args.suites, reports_dir, args.bootstrap_n)

    print(f"\n  Reports discovered per condition (dir={reports_dir}):")
    for lbl in args.labels:
        have = ", ".join(s for s in args.suites if s in found[lbl]) or "NONE"
        missing = [s for s in args.suites if s not in found[lbl]]
        flag = "" if not missing else f"   [!] missing: {', '.join(missing)}"
        print(f"    {lbl:<22} {have}{flag}")

    print_ladder(rows, args.labels)
    print_persona_matrix(persona_dimension_matrix(args.labels, reports_dir), args.labels)
    corr, n_corr = persona_dimension_correlation(args.labels, reports_dir)
    write_dimension_correlation(corr, n_corr, output_dir / f"persona_dimension_correlation_{ts}.csv")

    write_csv(rows, args.labels, output_dir / f"experiment_ladder_{ts}.csv")
    write_latex(rows, args.labels, output_dir / f"experiment_ladder_{ts}.tex")
    write_h3_csv(h3_failure_inventory(args.labels, reports_dir),
                 output_dir / f"experiment_h3_failures_{ts}.csv")

    if args.figures:
        try:
            import experiment_figures as ef
            ef.render_all(args.labels, str(reports_dir), str(output_dir / "dissertation_assets"))
        except Exception as e:
            print(f"  [figures skipped: {e}]")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
