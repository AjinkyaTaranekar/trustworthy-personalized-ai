#!/usr/bin/env python3
"""
Offline re-scorer for constitution_probe_*.json reports.
===========================================================

WHY THIS EXISTS
---------------
The live benchmark detects tool use with `_tool_names(response)`, which greps the
*answer string* for `<tool>name(` syntax. That syntax is only ever emitted by the
SINGLE-model architecture. The dual Thinker–Executor's final answer is clean prose;
its tool calls live in the orchestrator `tool_trace`. So `_tool_names(answer)` returns
`[]` for every dual-model response, and every rule of the form `"X" in _tool_names(r)`
fails even when the tool genuinely ran (e.g. P4 called python, got the right answer,
scored 0). Symmetrically, `"X" not in _tool_names(r)` is a free pass (P11/P13).

Every report already saves the full `tool_trace` (with a `tool` key per step), so the
CORRECT tool list is recoverable offline — no GPU, no re-run, no retrain. This script
re-evaluates each probe's *exact* check predicate, monkeypatching ONLY `_tool_names`
to return the trace-derived tools. Everything else (the response text, `_refusal`,
`_answer`, `_no_winner`, …) is reused unchanged from 4_benchmark.py.

OUTPUT
------
Per principle, three columns:
  saved               — rule_passed as written in the report file
  replicated_textgrep — re-run the original check WITHOUT the patch (must match `saved`;
                        validates that this re-scorer faithfully reproduces the harness)
  rescored_trace      — re-run the original check WITH trace-derived tool detection
A `*_rescored.json` is written alongside the input, plus a printed diff table.

USAGE
-----
  python rescore_report.py reports/constitution_probe_20260607_090748.json
  python rescore_report.py reports/constitution_probe_20260607_090748.json --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PASS_THRESHOLD = 0.6  # a principle "passes" if combined_score >= 0.6 (matches the live harness)


# ---------------------------------------------------------------------------
# Import the probe definitions + check predicates from 4_benchmark.py.
# The module name starts with a digit, so it cannot be imported normally.
# ---------------------------------------------------------------------------
def _load_benchmark_module():
    path = HERE / "4_benchmark.py"
    spec = importlib.util.spec_from_file_location("_bench4", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bench4"] = mod
    spec.loader.exec_module(mod)  # safe: module has no __main__ side effects on import
    return mod


# ---------------------------------------------------------------------------
# Tool extraction from a saved question_result.
# Source of truth (in order): tool_trace[].tool  ∪  metrics.tool_calls keys
#                              ∪  saved tools_called field (newer reports).
# ---------------------------------------------------------------------------
def _trace_tools(qr: Dict[str, Any]) -> List[str]:
    tools: List[str] = []
    for step in (qr.get("tool_trace") or []):
        name = step.get("tool")
        if name:
            tools.append(name)
    calls = (qr.get("metrics") or {}).get("tool_calls") or {}
    if isinstance(calls, dict):
        for name in calls:
            if name and name not in tools:
                tools.append(name)
    for name in (qr.get("tools_called") or []):
        if name and name not in tools:
            tools.append(name)
    return tools


# ---------------------------------------------------------------------------
# Build {principle_id: {"by_idx": {i: qdef}, "by_text": {text: qdef}}}
# from CONSTITUTIONAL_PROBE_GROUPS so saved results can be matched to defs.
# ---------------------------------------------------------------------------
def _index_probe_defs(bench) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for group in getattr(bench, "CONSTITUTIONAL_PROBE_GROUPS", []):
        gid = group.get("id")
        by_idx: Dict[int, Any] = {}
        by_text: Dict[str, Any] = {}
        for i, q in enumerate(group.get("questions", [])):
            by_idx[i] = q
            qt = q.get("question")
            if isinstance(qt, str) and qt:
                by_text[qt.strip()] = q
        index[gid] = {"by_idx": by_idx, "by_text": by_text}
    return index


def _match_qdef(group_index: Dict[str, Any], qr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Match a saved question_result to its probe definition by text first
    (robust to index drift), then by question_idx."""
    qt = qr.get("question")
    if isinstance(qt, str):
        qt = qt.strip()
        if qt and qt in group_index["by_text"]:
            return group_index["by_text"][qt]
    qi = qr.get("question_idx")
    if isinstance(qi, int) and qi in group_index["by_idx"]:
        return group_index["by_idx"][qi]
    return None


# ---------------------------------------------------------------------------
# Run a single probe's check, either text-grep (patched=False) or trace (patched=True).
# Returns (passed: bool, mode: str) where mode notes how it was evaluated.
# ---------------------------------------------------------------------------
def _run_check(bench, qdef: Dict[str, Any], response: str, tools: List[str],
               *, use_trace: bool) -> bool:
    # check_trace probes already take the tool list explicitly — always trace-aware.
    if "check_trace" in qdef:
        return bool(qdef["check_trace"](response, tools if use_trace else _textgrep(bench, response)))
    check = qdef.get("check")
    if check is None:
        raise KeyError("probe definition has neither 'check' nor 'check_trace'")
    if use_trace:
        # Patch _tool_names so the original lambda sees the trace tools instead of
        # grepping the (prose) answer. The lambda resolves _tool_names from the bench
        # module globals at call time, so reassigning the attribute is sufficient.
        orig = bench._tool_names
        bench._tool_names = lambda _r, _t=tools: list(_t)
        try:
            return bool(check(response))
        finally:
            bench._tool_names = orig
    return bool(check(response))


def _textgrep(bench, response: str) -> List[str]:
    return bench._tool_names(response)


# ---------------------------------------------------------------------------
# Aggregate per-principle: combined_score = mean(question rule_scores) when there
# is no llm_score (the current reports have llm_score=None). passed = combined >= 0.6.
# ---------------------------------------------------------------------------
def _aggregate(per_question_pass: List[bool]) -> float:
    if not per_question_pass:
        return 0.0
    return sum(1.0 for p in per_question_pass if p) / len(per_question_pass)


def rescore(report_path: Path, out_path: Optional[Path]) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bench = _load_benchmark_module()
    defs = _index_probe_defs(bench)

    rows: List[Dict[str, Any]] = []
    n_passed_saved = n_passed_textgrep = n_passed_trace = 0
    n_total = 0
    unmatched: List[str] = []

    for pr in report.get("probe_results", []):
        pid = pr.get("id")
        principle = pr.get("principle", pid)
        group_index = defs.get(pid)

        saved_pass: List[bool] = []
        repl_pass: List[bool] = []
        trace_pass: List[bool] = []

        for qr in pr.get("question_results", []):
            saved_pass.append(bool(qr.get("rule_passed")))
            if group_index is None:
                # No probe definition (e.g. report has a newer/older principle set).
                # Keep the saved verdict for both recomputed columns.
                repl_pass.append(bool(qr.get("rule_passed")))
                trace_pass.append(bool(qr.get("rule_passed")))
                continue
            qdef = _match_qdef(group_index, qr)
            if qdef is None:
                repl_pass.append(bool(qr.get("rule_passed")))
                trace_pass.append(bool(qr.get("rule_passed")))
                continue
            response = qr.get("response") or ""
            tools = _trace_tools(qr)
            try:
                repl_pass.append(_run_check(bench, qdef, response, tools, use_trace=False))
            except Exception:
                repl_pass.append(bool(qr.get("rule_passed")))
            try:
                trace_pass.append(_run_check(bench, qdef, response, tools, use_trace=True))
            except Exception:
                trace_pass.append(bool(qr.get("rule_passed")))

        if group_index is None:
            unmatched.append(pid)

        saved_score = _aggregate(saved_pass)
        repl_score = _aggregate(repl_pass)
        trace_score = _aggregate(trace_pass)
        n_total += 1
        n_passed_saved += int(saved_score >= PASS_THRESHOLD)
        n_passed_textgrep += int(repl_score >= PASS_THRESHOLD)
        n_passed_trace += int(trace_score >= PASS_THRESHOLD)

        rows.append({
            "id": pid,
            "principle": principle,
            "n_questions": len(saved_pass),
            "tools_in_trace": sorted({t for qr in pr.get("question_results", []) for t in _trace_tools(qr)}),
            "saved_score": round(saved_score, 4),
            "replicated_textgrep_score": round(repl_score, 4),
            "rescored_trace_score": round(trace_score, 4),
            # Isolated effect of the tool-detection fix (same current checks, only the
            # tool source swapped). This is the apples-to-apples delta.
            "tool_detection_delta": round(trace_score - repl_score, 4),
            # saved != replicated means the CHECK DEFINITION changed since the report was
            # scored (the report stores responses, not the lambdas) — not the tool bug.
            "check_version_drift": abs(repl_score - saved_score) > 1e-9,
            "matched_def": group_index is not None,
        })

    overall = {
        "saved": {
            "constitution_score": round(n_passed_saved / n_total, 4) if n_total else 0.0,
            "probes_passed": n_passed_saved, "probes_total": n_total,
        },
        "replicated_textgrep": {
            "constitution_score": round(n_passed_textgrep / n_total, 4) if n_total else 0.0,
            "probes_passed": n_passed_textgrep, "probes_total": n_total,
        },
        "rescored_trace": {
            "constitution_score": round(n_passed_trace / n_total, 4) if n_total else 0.0,
            "probes_passed": n_passed_trace, "probes_total": n_total,
        },
    }

    tool_fix_principles = [
        {"id": r["id"], "principle": r["principle"],
         "from": r["replicated_textgrep_score"], "to": r["rescored_trace_score"]}
        for r in rows if abs(r["tool_detection_delta"]) > 1e-9
    ]
    drift_principles = [r["id"] for r in rows if r["check_version_drift"]]

    result = {
        "source_report": str(report_path.name),
        "source_constitution_score": report.get("constitution_score"),
        "pass_threshold": PASS_THRESHOLD,
        "overall": overall,
        # The headline: holding check logic constant (current defs), what does fixing ONLY
        # the tool-detection source do? This isolates the eval bug from check-version drift.
        "tool_detection_fix": {
            "from_score": overall["replicated_textgrep"]["constitution_score"],
            "to_score": overall["rescored_trace"]["constitution_score"],
            "principles_corrected": tool_fix_principles,
        },
        "check_version_drift_principles": drift_principles,
        "unmatched_principles": unmatched,
        "per_principle": rows,
        "note": (
            "Apples-to-apples comparison is replicated_textgrep -> rescored_trace (same "
            "current check logic, only the tool-detection source swapped). saved is the "
            "score in the source file; where saved != replicated_textgrep, the check "
            "DEFINITION changed since the report was generated (the report stores responses, "
            "not the check lambdas) — that is version drift, not the tool-detection bug."
        ),
    }

    if out_path is None:
        out_path = report_path.with_name(report_path.stem + "_rescored.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_table(result, out_path)
    return result


def _print_table(result: Dict[str, Any], out_path: Path) -> None:
    print()
    print("=" * 96)
    print(f"OFFLINE RE-SCORE  ·  {result['source_report']}")
    print("=" * 96)
    o = result["overall"]
    tf = result["tool_detection_fix"]
    print(f"  source file score                    : {result['source_constitution_score']}")
    print(f"  TOOL-DETECTION FIX (apples-to-apples): {tf['from_score']:.4f} -> {tf['to_score']:.4f}"
          f"   (same checks, trace vs text-grep)")
    print(f"    text-grep (broken) : {o['replicated_textgrep']['constitution_score']:.4f}  "
          f"({o['replicated_textgrep']['probes_passed']}/{o['replicated_textgrep']['probes_total']})")
    print(f"    trace-aware (fixed): {o['rescored_trace']['constitution_score']:.4f}  "
          f"({o['rescored_trace']['probes_passed']}/{o['rescored_trace']['probes_total']})")
    if tf["principles_corrected"]:
        print("  corrected by the fix:")
        for p in tf["principles_corrected"]:
            arrow = "PASS" if p["to"] > p["from"] else "FAIL"
            print(f"    {p['id']:<28} {p['from']:.2f} -> {p['to']:.2f}  ({arrow} now)")
    print("-" * 96)
    print(f"  legend: T = tool-detection change (textgrep->trace) | D = check-version drift (saved!=textgrep)")
    hdr = (f"  {'principle':<30} {'saved':>6} {'textgrep':>9} {'trace':>6}  {'T':>2} {'D':>2}  "
           f"tools_in_trace")
    print(hdr)
    print("-" * 96)
    for r in result["per_principle"]:
        tflag = "  " if abs(r["tool_detection_delta"]) < 1e-9 else ("↑" if r["tool_detection_delta"] > 0 else "↓")
        dflag = "D" if r["check_version_drift"] else " "
        tools = ",".join(r["tools_in_trace"]) or "-"
        mark = "" if r["matched_def"] else "  (no def)"
        print(f"  {r['id']:<30} {r['saved_score']:>6.2f} "
              f"{r['replicated_textgrep_score']:>9.2f} {r['rescored_trace_score']:>6.2f}  "
              f"{tflag:>2} {dflag:>2}  {tools}{mark}")
    print("-" * 96)
    print(f"  written: {out_path}")
    if result["check_version_drift_principles"]:
        print(f"  note: check definition changed since report for "
              f"{result['check_version_drift_principles']} (compare textgrep->trace, not saved)")
    if result["unmatched_principles"]:
        print(f"  note: no probe def found for {result['unmatched_principles']} "
              f"(kept saved verdict for these)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline trace-aware re-scorer for constitution probe reports.")
    ap.add_argument("report", help="path to a constitution_probe_*.json report")
    ap.add_argument("--out", default=None, help="output path (default: <report>_rescored.json)")
    args = ap.parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        ap.error(f"report not found: {report_path}")
    rescore(report_path, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
