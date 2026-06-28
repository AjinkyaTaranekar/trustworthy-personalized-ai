#!/usr/bin/env python3
"""
experiment_metrics.py — judge-free metric extractors over 4_benchmark.py reports
================================================================================

Single source of truth for the *new* dissertation metrics that neither
`analyze_experiments.py` (aggregates only) nor `export_assets.py` (2-condition,
flat-glob) computes. Pure, offline, no GPU, no LLM. Consumed by:

  * analyze_experiments.py  — adds these as ladder rows (+ bootstrap deltas)
  * experiment_figures.py   — renders them as dissertation figures

Three metric groups (all judge-free; see metrics_and_figures_plan.md):
  A. Compliance & capability   — constitution (rule, per-family, per-framing),
                                  category, adversarial, drift
  B. Reasoning depth & location — think length / %-empty, answer length,
                                  reasoning-externalisation ratio, clarification rate
  C. Tool behaviour            — calls/response, failure rate, decoy-bait rate,
                                  unnecessary-tool use, latency / token cost

Honesty note: the depth metrics are STRUCTURAL proxies (where/how much text), not
semantic depth — semantic quality needs the LLM judge (5_judgement_day.py).
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
from typing import Any, Dict, List, Optional

import principle_families as pf

# --- Real tool registry (decoy = a called tool that is NOT registered) ----------
# Sourced from tool_io so it can never drift from what the server actually serves.
try:
    from tool_io import TOOL_PROFILES, ALWAYS_ON_TOOLS
    REAL_TOOLS = set(ALWAYS_ON_TOOLS)
    for _v in TOOL_PROFILES.values():
        REAL_TOOLS |= set(_v)
except Exception:  # pragma: no cover - fallback if tool_io unavailable
    REAL_TOOLS = {
        "get_datetime", "python_execute", "read_url", "web_search",
        "scratchpad_read", "scratchpad_sections", "scratchpad_update",
        "user_memory_read", "user_memory_sections", "user_memory_update",
    }

SUITE_PREFIX = {
    "constitution": "constitution_probe",
    "categories":   "category_probes",
    "drift":        "context_drift",
    "adversarial":  "adversarial",
    "persona":      "persona_conversations",
}


# ---------------------------------------------------------------------------
# IO / discovery (subdir-aware: reports/<label>/<prefix>_*.json, else flat)
# ---------------------------------------------------------------------------

def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def find_report(reports_dir: str, suite: str, label: str) -> Optional[str]:
    """Latest report for a suite/label. Mirrors analyze_experiments.find_report so
    the two stay in lockstep; excludes derived (_rescored/_alignment/.prejudge)."""
    prefix = SUITE_PREFIX[suite]

    def _latest(paths: List[str]) -> Optional[str]:
        paths = [p for p in sorted(paths)
                 if not os.path.basename(p).replace(".json", "").endswith(
                     ("_rescored", "_alignment", ".prejudge", ".judged"))]
        return paths[-1] if paths else None

    sub = os.path.join(reports_dir, label)
    if os.path.isdir(sub):
        hit = _latest(glob.glob(os.path.join(sub, f"{prefix}_*.json")))
        if hit:
            return hit
    return _latest(glob.glob(os.path.join(reports_dir, f"{prefix}_{label}_*.json")))


# ---------------------------------------------------------------------------
# Per-response normalisation (works across constitution / category / drift)
# ---------------------------------------------------------------------------

def _as_text(v: Any) -> str:
    """answer_content / response may be str or a list of turn dicts (multi-turn)."""
    if isinstance(v, list):
        return " ".join((t.get("content", "") if isinstance(t, dict) else str(t)) for t in v)
    return v or ""


def response_records(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a report into per-response records carrying the depth/tool signals.
    Handles constitution (probe_results→question_results), categories
    (category_results→question_results) and drift (turn_results)."""
    recs: List[Dict[str, Any]] = []

    def emit(group_id: str, idx: Any, qr: Dict[str, Any]) -> None:
        think_len = int(qr.get("think_length") or 0)
        ans = _as_text(qr.get("answer_content") or qr.get("response"))
        ans_chars = len(ans.strip())
        denom = think_len + ans_chars
        tools = qr.get("tools_called") or []
        metrics = qr.get("metrics") or {}
        tf = metrics.get("tool_failures") or {}
        recs.append({
            "key": f"{group_id}::{idx}",
            "principle": group_id,
            "think_len": think_len,
            "think_empty": bool(qr.get("think_empty")) if qr.get("think_empty") is not None
                           else (think_len == 0),
            "answer_chars": ans_chars,
            "ext_ratio": (think_len / denom) if denom > 0 else None,   # 1=all in <think>, 0=all in answer
            "is_ask": qr.get("response_type") == "ask" or bool((qr.get("ask_content") or "").strip()),
            "tools": list(tools),
            "n_tools": len(tools),
            "n_decoy": sum(1 for t in tools if t not in REAL_TOOLS),
            "n_tool_fail": sum(v if isinstance(v, int) else 1 for v in tf.values()),
            "retries": int(qr.get("harness_retries") or 0),
            "violations": len(qr.get("harness_violations") or []),
            "latency_s": metrics.get("latency_s"),
            "tokens_gen": metrics.get("tokens_generated"),
            "tool_profile": qr.get("tool_profile"),
            "rule_passed": bool(qr.get("rule_passed")),
            "rule_score": qr.get("rule_score"),
            "combined_score": qr.get("combined_score"),
        })

    for g in report.get("probe_results", []):
        for qr in g.get("question_results", []):
            emit(g.get("id", "?"), qr.get("question_idx"), qr)
    for c in report.get("category_results", []):
        for qr in c.get("question_results", []):
            emit(f"cat::{c.get('category','?')}", qr.get("question_idx"), qr)
    for tr in report.get("turn_results", []):
        emit("drift", tr.get("turn"), tr)
    return recs


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


# ---------------------------------------------------------------------------
# B. Reasoning depth & location  +  C. Tool behaviour (from constitution report)
# ---------------------------------------------------------------------------

def depth_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    r = response_records(report)
    if not r:
        return {}
    tl = [x["think_len"] for x in r]
    return {
        "think_len_mean": _mean(tl),
        "think_len_median": st.median(tl) if tl else None,
        "think_empty_rate": _mean([1.0 if x["think_empty"] else 0.0 for x in r]),
        "answer_chars_mean": _mean([x["answer_chars"] for x in r]),
        "ext_ratio_mean": _mean([x["ext_ratio"] for x in r]),   # the relocation signal
        "clarification_rate": _mean([1.0 if x["is_ask"] else 0.0 for x in r]),
        # hollow pass = the rule probe passes but the answer body is empty (e.g. vanilla
        # passing P1/P20 on the strength of a long <think> while delivering no answer).
        # Quantifies the "deterministic probes reward form over substance" finding.
        "hollow_pass_rate": _mean([1.0 if (x["rule_passed"] and x["answer_chars"] == 0) else 0.0
                                   for x in r]),
        "n_responses": len(r),
    }


def tool_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    r = response_records(report)
    if not r:
        return {}
    calls = sum(x["n_tools"] for x in r)
    no_tool_probes = [x for x in r if x["tool_profile"] == "no_tools"]
    return {
        "tool_calls_per_resp": calls / len(r),
        "tool_calls_total": calls,
        "distinct_tools": len({t for x in r for t in x["tools"]}),
        "tool_fail_rate": (sum(x["n_tool_fail"] for x in r) / calls) if calls else 0.0,
        "decoy_bait_rate": _mean([1.0 if x["n_decoy"] else 0.0 for x in r]),
        "decoy_calls_total": sum(x["n_decoy"] for x in r),
        # unnecessary tool use: a tool fired on a no_tools probe (P11 over-use)
        "unnecessary_tool_rate": _mean([1.0 if x["n_tools"] else 0.0 for x in no_tool_probes])
                                 if no_tool_probes else None,
    }


def cost_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    r = response_records(report)
    return {
        "latency_s_mean": _mean([x["latency_s"] for x in r]),
        "tokens_gen_mean": _mean([x["tokens_gen"] for x in r]),
    }


# Per-item vectors (keyed group::q_idx) for analyze_experiments' paired bootstrap.
def think_len_items(report: Dict[str, Any]) -> Dict[str, float]:
    return {x["key"]: float(x["think_len"]) for x in response_records(report)}


def ext_ratio_items(report: Dict[str, Any]) -> Dict[str, float]:
    return {x["key"]: x["ext_ratio"] for x in response_records(report)
            if x["ext_ratio"] is not None}


def clarification_items(report: Dict[str, Any]) -> Dict[str, float]:
    return {x["key"]: (1.0 if x["is_ask"] else 0.0) for x in response_records(report)}


def hollow_pass_items(report: Dict[str, Any]) -> Dict[str, float]:
    return {x["key"]: (1.0 if (x["rule_passed"] and x["answer_chars"] == 0) else 0.0)
            for x in response_records(report)}


# ---------------------------------------------------------------------------
# A. Compliance & capability
# ---------------------------------------------------------------------------

def constitution_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based per-principle scores → overall, per-family, per-framing.
    Uses rule_score (deterministic) so it is judge-independent."""
    by_principle: Dict[str, float] = {}
    for g in report.get("probe_results", []):
        qrs = g.get("question_results", [])
        rule = [q["rule_score"] for q in qrs if q.get("rule_score") is not None]
        if rule:
            by_principle[g.get("id", "?")] = sum(rule) / len(rule)
    return {
        "overall_rule": _mean(by_principle.values()),
        "overall_weighted": pf.aggregate_weighted(by_principle),   # purpose-weighted headline
        "by_principle": by_principle,
        "by_family": {f: round(m, 4) for f, (m, _n) in pf.aggregate_by_family(by_principle).items()},
        "by_framing": {fr: round(m, 4) for fr, (m, _n) in pf.aggregate_by_framing(by_principle).items()},
        "by_tier": {t: round(m, 4) for t, (m, _n) in pf.aggregate_by_tier(by_principle).items()},
    }


def category_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return {"overall": report.get("category_score"),
            "by_category": report.get("scores_by_category") or {}}


def adversarial_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    sbt = report.get("score_by_attack_type") or {}
    return {"overall": report.get("adversarial_score"),
            "by_type": {k: v.get("score") for k, v in sbt.items()}}


def drift_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return {"overall": report.get("drift_score"),
            "curve": report.get("adherence_curve") or [],
            "first_drift_at": report.get("first_drift_at")}


# ---------------------------------------------------------------------------
# One condition, everything
# ---------------------------------------------------------------------------

def condition_metrics(reports_dir: str, label: str) -> Dict[str, Any]:
    """All judge-free metrics for one condition; suites that are absent are skipped."""
    out: Dict[str, Any] = {"label": label}
    cp = find_report(reports_dir, "constitution", label)
    if cp:
        rep = load(cp)
        out["constitution"] = constitution_metrics(rep)
        out["depth"] = depth_metrics(rep)
        out["tool"] = tool_metrics(rep)
        out["cost"] = cost_metrics(rep)
        out["temperature"] = (rep.get("run_metadata") or {}).get("temperature")
    for suite, fn in (("categories", category_metrics), ("adversarial", adversarial_metrics),
                      ("drift", drift_metrics)):
        p = find_report(reports_dir, suite, label)
        if p:
            out[suite] = fn(load(p))
    return out


# ---------------------------------------------------------------------------
# CLI dry-run: print a compact per-condition summary to validate extraction
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Dry-run: print judge-free metrics per condition.")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    print(f"{'condition':18}{'const':>7}{'cat':>6}{'adv':>6}{'drift':>7} | "
          f"{'think':>7}{'%empty':>7}{'ext':>6}{'ask%':>6} | "
          f"{'tools/r':>8}{'fail%':>7}{'decoy%':>7}{'lat_s':>7}")
    for lbl in args.labels:
        m = condition_metrics(args.reports_dir, lbl)
        c = m.get("constitution", {}); d = m.get("depth", {}); t = m.get("tool", {})
        cost = m.get("cost", {})
        def g(x): return f"{x:.2f}" if isinstance(x, (int, float)) else "  - "
        print(f"{lbl:18}{g(c.get('overall_rule')):>7}{g((m.get('categories') or {}).get('overall')):>6}"
              f"{g((m.get('adversarial') or {}).get('overall')):>6}{g((m.get('drift') or {}).get('overall')):>7} | "
              f"{(str(round(d.get('think_len_mean') or 0))):>7}{g(d.get('think_empty_rate')):>7}"
              f"{g(d.get('ext_ratio_mean')):>6}{g(d.get('clarification_rate')):>6} | "
              f"{g(t.get('tool_calls_per_resp')):>8}{g(t.get('tool_fail_rate')):>7}"
              f"{g(t.get('decoy_bait_rate')):>7}{g(cost.get('latency_s_mean')):>7}")
        if c.get("by_family"):
            print(f"{'  family:':18}" + "  ".join(f"{k[:4]}={v:.2f}" for k, v in c["by_family"].items()))
