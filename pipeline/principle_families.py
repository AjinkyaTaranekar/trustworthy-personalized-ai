"""Canonical principle -> family map for the constitutional probe suite.

SINGLE SOURCE OF TRUTH for the dissertation's principle stratification
(background Ch.2 taxonomy table, Ch.4 per-family metric, Ch.5 per-family results).
Grounded in pipeline/constitution.md.

IMPORTANT CONSISTENCY NOTES (surfaced for the dissertation — see export_assets.py
and the .tex DRIFT flags):
  * constitution.md's prose header says "19 principles" but the file actually
    DEFINES 25 (P1-P25).
  * The benchmark (4_benchmark.py) scores 21 ITEMS: P1-P21 with P2 and P3 merged
    into a single "P2P3_tool_discipline" probe, plus the harness-only
    "H2_memory_persistence" probe. So "21" in the dissertation = these 21 scored
    items, NOT constitution principles 1-21.
  * Constitution principles P22-P25 (CONSEQUENCE_CHECK, INTERLEAVED TOOL CHAINING,
    SCRATCHPAD-FIRST, PARTIAL CAPABILITY DECLARATION) are defined but NOT probed.
  The dissertation must state this precisely; do not claim "21 principles" without
  this clarification.

Framing follows the C3AI distinction \cite{kyrychenko2025c3aicraftingevaluatingconstitutions}:
  positive    = behaviour-based "do X"
  negative    = prohibitive "never do X"
  calibration = conditional honesty ("use the tool if available, else admit the gap")
"""

# probe_id -> (display, constitution_part, family, framing)
PRINCIPLES = {
    "P1_decompose_first":    ("P1 Decompose First",        "I",   "reasoning",       "positive"),
    "P2P3_tool_discipline":  ("P2/P3 Tool Discipline",     "I/II", "tool",           "negative"),
    "P4_math_code":          ("P4 Math = Code",            "I",   "tool",            "positive"),
    "P5_realtime_honesty":   ("P5 Real-Time Honesty",      "I",   "honesty",         "calibration"),
    "P6_context_gate":       ("P6 User Context Gate",      "I",   "personalisation", "positive"),
    "P7_uncertainty":        ("P7 Uncertainty Quant.",     "I",   "honesty",         "calibration"),
    "P8_impossibility":      ("P8 Impossibility Ack.",     "I",   "honesty",         "negative"),
    "P9_no_winner":          ("P9 Tradeoff Presentation",  "I",   "robustness",      "negative"),
    "P10_correct_tool_use":  ("P10 Correct Tool Use",      "II",  "tool",            "positive"),
    "P11_tool_avoidance":    ("P11 Tool Avoidance",        "II",  "tool",            "negative"),
    "P12_tool_failure":      ("P12 Tool Failure Handling", "II",  "tool",            "calibration"),
    "P13_no_tool_faking":    ("P13 No Tool Faking",        "II",  "tool",            "negative"),
    "P14_hold_pressure":     ("P14 Hold Under Pressure",   "III", "robustness",      "negative"),
    "P15_self_correction":   ("P15 Explicit Self-Correction", "III", "reasoning",    "positive"),
    "P16_cutoff_awareness":  ("P16 Knowledge Cutoff",      "III", "honesty",         "calibration"),
    "P17_single_question":   ("P17 Multi-Step Clarification", "III", "personalisation", "positive"),
    "P18_explicit_dont_know":("P18 Explicit I Don't Know", "III", "honesty",         "negative"),
    "P19_search_entity_facts":("P19 Search Entity Facts",  "III", "tool",            "calibration"),
    "P20_first_principles":  ("P20 First Principles",      "III", "reasoning",       "positive"),
    "P21_greedy_followup":   ("P21 5W+H Questioning",      "III", "reasoning",       "positive"),
    "H2_memory_persistence": ("H2 Memory Persistence",     "harness", "personalisation", "positive"),
}

# Display order for the five families.
FAMILIES = ["reasoning", "honesty", "tool", "robustness", "personalisation"]

FAMILY_LABELS = {
    "reasoning":       "Reasoning \\& Process",
    "honesty":         "Honesty \\& Calibration",
    "tool":            "Tool Discipline \\& Use",
    "robustness":      "Robustness \\& Integrity",
    "personalisation": "Context \\& Personalisation",
}

# Constitution principles defined but NOT scored by the probe suite.
UNPROBED_PRINCIPLES = {
    "P22": "CONSEQUENCE_CHECK",
    "P23": "INTERLEAVED TOOL CHAINING",
    "P24": "SCRATCHPAD-FIRST",
    "P25": "PARTIAL CAPABILITY DECLARATION",
}

PRINCIPLE_COUNT_NOTE = (
    "Probe suite scores 21 items (P1-P21, with P2+P3 merged, plus H2_memory_persistence). "
    "constitution.md defines 25 principles; P22-P25 are unprobed."
)


def family_of(probe_id):
    rec = PRINCIPLES.get(probe_id)
    return rec[2] if rec else "unknown"


def framing_of(probe_id):
    rec = PRINCIPLES.get(probe_id)
    return rec[3] if rec else "unknown"


def display_of(probe_id):
    rec = PRINCIPLES.get(probe_id)
    return rec[0] if rec else probe_id


def aggregate_by_family(scores_by_principle):
    """scores_by_principle: {probe_id: float} -> {family: (mean, n)}."""
    buckets = {f: [] for f in FAMILIES}
    for pid, score in scores_by_principle.items():
        fam = family_of(pid)
        if fam in buckets and score is not None:
            buckets[fam].append(score)
    return {f: (sum(v) / len(v), len(v)) for f, v in buckets.items() if v}


def aggregate_by_framing(scores_by_principle):
    """{probe_id: float} -> {framing: (mean, n)}  (the C3AI positive/negative split)."""
    buckets = {}
    for pid, score in scores_by_principle.items():
        fr = framing_of(pid)
        if score is None:
            continue
        buckets.setdefault(fr, []).append(score)
    return {fr: (sum(v) / len(v), len(v)) for fr, v in buckets.items()}


if __name__ == "__main__":
    # Quick self-report.
    from collections import Counter
    fam_counts = Counter(v[2] for v in PRINCIPLES.values())
    fr_counts = Counter(v[3] for v in PRINCIPLES.values())
    print("Scored items:", len(PRINCIPLES))
    print("By family:", dict(fam_counts))
    print("By framing:", dict(fr_counts))
    print("Unprobed:", UNPROBED_PRINCIPLES)
    print(PRINCIPLE_COUNT_NOTE)
