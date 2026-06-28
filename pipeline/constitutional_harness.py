"""
Constitutional Harness
======================
Inference-time validation and correction loop for constitutional principle compliance.

Checks P1 (First Principles decomposition), P3 (TOOL DISCIPLINE), P4 (MATH=CODE), P18 (ANSWER PRESENT)
on every model response. On violation, injects a corrective prompt and retries generation.
Tracks per-principle failure rates and adaptively reinforces weak principles.

Usage (from 3_infererence.py):
    from constitutional_harness import ConstitutionalHarness
    _HARNESS = ConstitutionalHarness()
    response, violations, retries = _HARNESS.check_and_steer(
        response, conv, question, tool_profile_label, generate_fn, max_retries=2
    )
"""

import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime",
    "scratchpad_read", "scratchpad_update",
})

_TOOL_PROFILE_MAP: Dict[str, frozenset] = {
    "all_tools":          frozenset({"python_execute", "web_search", "read_url", "get_datetime"}),
    "compute_only":       frozenset({"python_execute"}),
    "compute_and_search": frozenset({"python_execute", "web_search", "read_url"}),
    "no_tools":           frozenset(),
}

_MATH_SIGNAL_RE = re.compile(
    r"(?:"
    r"\d+\.?\d*\s*(?:each|per\s+\w+|times|divided|\*|×|%)"
    r"|(?:calculat|comput|what\s+is\s+\d|how\s+much|total\s+cost|percentage\s+of|average\s+of)"
    r")",
    re.IGNORECASE,
)

_CHECKED_PRINCIPLES = ("P1", "P3", "P4", "P18", "P20", "P21")
_ADAPTATION_THRESHOLD = 0.30
_RETRY_TEMP_SCALES = (1.3, 1.6)

# Signals indicating First Principles decomposition in <think> (P20).
# Looks for language that breaks a question to its core before answering.
_FIRST_PRINCIPLES_RE = re.compile(
    r"\bfirst principles?\b"
    r"|\bfundamentally\b"
    r"|\birreducible\b"
    r"|\bat its core\b"
    r"|\bwhat is (fundamentally|actually|really) being asked\b"
    r"|\b5w\+?h\b"
    r"|\b(who|what|when|where|why|how)\s+(is|are|do|does|did|would|am|have)\b",
    re.IGNORECASE,
)

# Two-regex approach — mirrors the split in 2_model_trainer._WPLUS_H_UPPERCASE/CONTEXT.
# Without splitting, re.IGNORECASE makes \b(WHO|WHAT|...)\b match plain "What",
# so "What do you think?" falsely registers as a valid 5W+H question.

# Case-sensitive: matches only when the model explicitly labels a 5W+H axis in ALLCAPS.
_GREEDY_FOLLOWUP_CAPS = re.compile(r"\b(WHO|WHAT|WHEN|WHERE|WHY|HOW)\b")

# Case-insensitive: context-specific phrases that target the USER's situation.
_GREEDY_FOLLOWUP_CTX = re.compile(
    r"your\s+\b(why|who|what|when|where|how|situation|context|background|goal"
    r"|motivation|reason|timeline|role|setup)\b"
    r"|\bwhy\s+(are\s+you|do\s+you|did\s+you|would\s+you|is\s+this)\b"
    r"|\bwho\s+(are\s+you|is\s+this|is\s+the)\b"
    r"|\bwhat\s+(is\s+your|are\s+you|does\s+your|specifically|exactly)\b"
    r"|\bwhen\s+(do\s+you|are\s+you|is\s+this|is\s+your)\b"
    r"|\bwhere\s+(are\s+you|do\s+you|is\s+this|does\s+this)\b"
    r"|\bhow\s+(do\s+you|are\s+you|does\s+your|would\s+you|much)\b"
    r"|tell\s+me\s+(more\s+)?(about\s+)?(your|who|why|what|how|when|where)\b"
    r"|to\s+(give|help)\s+(you\s+)?(more|better|a\s+sharper|sharper)\b"
    r"|\b5w\+?h\b",
    re.IGNORECASE,
)

_BEHAVIOURAL_CATEGORIES = frozenset({
    "first_principles_questioning", "user_context_behavioral",
    "ambiguous_underspecified", "multi_step_clarification",
    "appraisal_empathy", "subjective_tradeoffs",
})


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

def run_checks(
    response: str,
    question: str,
    tool_profile_label: str,
    scratchpad_store: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    """Deterministic constitutional checks for P1, P3, P4, P18, P24a, P24b.

    scratchpad_store and session_id are optional — when absent, P24 checks are skipped.
    Returns a list of violation strings. Empty list = compliant.
    """
    violations: List[str] = []
    active_tools = _TOOL_PROFILE_MAP.get(tool_profile_label, frozenset())

    # ── P1: <think> block present and substantive ────────────────────────────
    # v3 pipeline: CAPABILITY_CHECK label removed. The requirement is now that
    # <think> exists and contains meaningful reasoning (First Principles + 5W+H).
    has_think = bool(re.search(r"<think\b", response, re.IGNORECASE))

    if not has_think:
        violations.append(
            "PRINCIPLE_1: <think> block is entirely absent. "
            "Open every response with <think> containing First Principles decomposition "
            "and a 5W+H scan, then close with </think> before any tool calls or <answer>."
        )

    # ── P3: TOOL DISCIPLINE ──────────────────────────────────────────────────
    xml_calls    = set(re.findall(r"<tool>(\w+)\(", response))
    native_calls = set(re.findall(r'"name"\s*:\s*"(\w+)"', response))
    called_tools = xml_calls | native_calls

    hallucinated = called_tools - _ALL_TOOL_NAMES

    # A tool is flagged as "unavailable" only when the model's own CAPABILITY_CHECK
    # explicitly acknowledges the tool is not available yet the model calls it anyway.
    # This distinguishes deliberate misuse from profile misconfiguration.
    think_match = re.search(r"<think\b.*?</think>", response, re.IGNORECASE | re.DOTALL)
    think_text  = think_match.group(0) if think_match else ""
    unavailable: set = set()
    for tool in (called_tools & _ALL_TOOL_NAMES) - active_tools:
        # Flag only if the think block explicitly marks this tool as not available
        if re.search(rf"{re.escape(tool)}\s+not\s+available", think_text, re.IGNORECASE):
            unavailable.add(tool)

    if hallucinated:
        violations.append(
            f"PRINCIPLE_3: Hallucinated tool(s) that do not exist: {sorted(hallucinated)}. "
            "Never invent tools; only call tools listed in the session profile."
        )
    if unavailable:
        violations.append(
            f"PRINCIPLE_3: Called tool(s) not available in this session: {sorted(unavailable)}. "
            f"Active tools are: {sorted(active_tools) if active_tools else ['none']}."
        )

    # ── P4: MATH = CODE ─────────────────────────────────────────────────────
    if "python_execute" in active_tools:
        needs_math     = bool(_MATH_SIGNAL_RE.search(question))
        has_code_call  = bool(re.search(r'<tool>\s*python_execute|"name"\s*:\s*"python_execute"', response))
        numeric_answer = bool(re.search(r"<answer>.*\d[\d,.]+.*</answer>", response, re.DOTALL))
        if needs_math and not has_code_call and numeric_answer:
            violations.append(
                "PRINCIPLE_4: Numeric answer given without python_execute despite the tool being "
                "available. MATH = CODE: delegate all precision arithmetic to python_execute."
            )

    # ── P18: <answer> block present ─────────────────────────────────────────
    if not re.search(r"<answer\b", response, re.IGNORECASE):
        violations.append(
            "PRINCIPLE_18: <answer> block is absent. "
            "Every response must end with <answer>...</answer>."
        )

    # ── P20: First Principles decomposition in <think> ───────────────────────
    # Check whenever <think> is present and non-trivial (>= 20 chars).
    # For 0.6B, missing First Principles is the most common failure — the model
    # jumps straight to answer without decomposing the question first.
    if has_think and think_text and len(think_text) >= 20:
        if not _FIRST_PRINCIPLES_RE.search(think_text):
            violations.append(
                "PRINCIPLE_20: <think> block lacks First Principles decomposition. "
                "Before answering, explicitly identify: what is this fundamentally asking? "
                "What are the non-negotiable constraints? Then run the 5W+H scan: "
                "WHO/WHAT/WHEN/WHERE/WHY/HOW — name which dimension is the critical unknown."
            )

    # ── P21: Greedy 5W+H follow-up in <answer> ──────────────────────────────
    # Only checked for behavioural categories — not math or impossible tasks.
    # Extract category from question context if possible; default to checking always.
    answer_m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if answer_m:
        answer_text = answer_m.group(1).strip()
        # Split into sentences; check if last sentence ends with ?
        last_sentence = ""
        for part in re.split(r"(?<=[.!?])\s+", answer_text):
            if part.strip():
                last_sentence = part.strip()
        ends_with_question = last_sentence.endswith("?")
        has_5wh_in_question = bool(
            _GREEDY_FOLLOWUP_CAPS.search(last_sentence)
            or _GREEDY_FOLLOWUP_CTX.search(last_sentence)
        )

        if not ends_with_question:
            violations.append(
                "PRINCIPLE_21: <answer> does not end with a follow-up question. "
                "Every response must close with ONE targeted 5W+H question — "
                "the single most important user dimension still missing. "
                "Example: 'To understand your WHY better: what is driving this for you?'"
            )
        elif ends_with_question and not has_5wh_in_question:
            violations.append(
                "PRINCIPLE_21: <answer> ends with a question but it does not target a "
                "specific 5W+H dimension. Ask about WHO/WHAT/WHEN/WHERE/WHY/HOW — "
                "name the dimension explicitly so the user knows what context you need."
            )

    # ── P24a: scratchpad-first ───────────────────────────────────────────────
    if scratchpad_store is not None:
        non_pad_calls = len(re.findall(r"<tool>(?!scratchpad_)", response))
        pad_read_present = bool(re.search(r"<tool>\s*scratchpad_read", response))
        if non_pad_calls >= 3 and not pad_read_present:
            violations.append(
                "PRINCIPLE_24a: Response uses 3+ tool calls but scratchpad_read() was never called. "
                "Complex queries (3+ requirements or 2+ tools) require scratchpad-first: "
                "read the pad, write context and tasks, re-check constitution, then execute."
            )

    # ── P24b: task completion ────────────────────────────────────────────────
    if scratchpad_store is not None and session_id is not None:
        tasks_text = scratchpad_store.get_section(session_id, "tasks")
        incomplete = re.findall(r"\[YES(?:-NEXT)?\]", tasks_text)
        answer_present = bool(re.search(r"<answer\b", response, re.IGNORECASE))
        if incomplete and answer_present:
            violations.append(
                f"PRINCIPLE_24b: {len(incomplete)} task(s) marked [YES] or [YES-NEXT] in the "
                f"scratchpad were not completed before <answer>. The scratchpad is a contract — "
                f"do not close it with planned work undone. Complete all [YES] tasks first."
            )

    return violations


# ---------------------------------------------------------------------------
# build_corrective_prompt
# ---------------------------------------------------------------------------

def build_corrective_prompt(violations: List[str]) -> str:
    """Format violations into a targeted corrective user turn for retry injection."""
    if not violations:
        return (
            "[HARNESS] Please review your previous response and ensure it follows the "
            "required format: <think> with First Principles + 5W+H decomposition, then "
            "tool calls, then <answer> ending with a targeted 5W+H follow-up question."
        )
    bullet_list = "\n".join(f"  - {v}" for v in violations)
    return (
        "[HARNESS] Your previous response had violations that must be corrected:\n"
        f"{bullet_list}\n\n"
        "Rewrite your response in full, fixing each violation. "
        "Required structure:\n"
        "  1. <think> — First Principles decomposition + 5W+H scan (name critical unknown)\n"
        "  2. Tool calls as needed (user_memory_read first)\n"
        "  3. <answer> — full response with stated assumptions\n"
        "  4. Last line of <answer> — ONE targeted 5W+H follow-up question ending with ?"
    )


# ---------------------------------------------------------------------------
# HarnessMetrics
# ---------------------------------------------------------------------------

class HarnessMetrics:
    """Rolling per-principle pass/fail tracker with adaptive suffix generation."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size    = window_size
        self.request_count  = 0
        self.total_retries  = 0
        self.retry_successes = 0
        self._windows: Dict[str, deque] = {p: deque(maxlen=window_size) for p in _CHECKED_PRINCIPLES}

    def record(self, violations: List[str], retried: bool = False, retry_cleared: bool = False) -> None:
        """Record one request's violations. Call once per final response."""
        self.request_count += 1
        if retried:
            self.total_retries += 1
        if retry_cleared:
            self.retry_successes += 1
        # Normalise "PRINCIPLE_N" → "PN" so keys match _CHECKED_PRINCIPLES format
        _principle_re = re.compile(r"PRINCIPLE_(\d+)", re.IGNORECASE)
        violated_principles: set = set()
        for v in violations:
            key = v.split(":")[0].strip()
            m = _principle_re.match(key)
            if m:
                violated_principles.add(f"P{m.group(1)}")
            else:
                violated_principles.add(key)
        for p in _CHECKED_PRINCIPLES:
            self._windows[p].append(p in violated_principles)

    def fail_rate(self, principle: str) -> float:
        """Fraction of recent requests that violated this principle (0.0–1.0)."""
        w = self._windows.get(principle, deque())
        return sum(w) / len(w) if w else 0.0

    def get_adaptation_suffix(self) -> str:
        """Return a bolded reminder block for any principle exceeding the failure threshold."""
        weak = [p for p in _CHECKED_PRINCIPLES if self.fail_rate(p) >= _ADAPTATION_THRESHOLD]
        if not weak:
            return ""
        reminders = {
            "P1":  "**CRITICAL: Every response MUST open with <think> containing First Principles decomposition + 5W+H scan, then close </think> before any tool calls or <answer>. This is non-negotiable.**",
            "P3":  "**CRITICAL: Only call tools explicitly listed as ✓ in the session profile. Never invent tool names.**",
            "P4":  "**CRITICAL: Any question requiring numeric computation MUST use python_execute. MATH = CODE.**",
            "P18": "**CRITICAL: Every response MUST end with <answer>...</answer>. Never omit the answer block.**",
        }
        lines = ["", "⚠ HARNESS ADAPTATION (principles currently failing):"]
        for p in weak:
            lines.append(reminders.get(p, f"**{p}: Review the constitution.**"))
        return "\n".join(lines)

    def snapshot(self) -> Dict:
        """Return full metrics dict."""
        retry_success_rate = (
            round(self.retry_successes / self.total_retries, 3)
            if self.total_retries > 0 else None
        )
        return {
            "window_size":        self.window_size,
            "request_count":      self.request_count,
            "total_retries":      self.total_retries,
            "retry_success_rate": retry_success_rate,
            "adaptation_active":  [p for p in _CHECKED_PRINCIPLES if self.fail_rate(p) >= _ADAPTATION_THRESHOLD],
            "principles": {
                p: {
                    "checks":    len(self._windows[p]),
                    "failures":  int(sum(self._windows[p])),
                    "fail_rate": round(self.fail_rate(p), 4),
                }
                for p in _CHECKED_PRINCIPLES
            },
        }

    def save(self, path: str) -> None:
        """Persist metrics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)

    def load(self, path: str) -> None:
        """Restore counters from a previously saved metrics file."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        self.request_count   = data.get("request_count", 0)
        self.total_retries   = data.get("total_retries", 0)
        rsr = data.get("retry_success_rate")
        self.retry_successes = round(rsr * self.total_retries) if rsr and self.total_retries else 0
        for p, stats in data.get("principles", {}).items():
            if p in self._windows:
                failures = stats.get("failures", 0)
                checks   = stats.get("checks", 0)
                passes   = checks - failures
                for _ in range(failures):
                    self._windows[p].append(True)
                for _ in range(passes):
                    self._windows[p].append(False)


# ---------------------------------------------------------------------------
# ConstitutionalHarness
# ---------------------------------------------------------------------------

class ConstitutionalHarness:
    """Inference-time constitutional validation and correction loop."""

    def __init__(
        self,
        metrics_path: str = "reports/harness_metrics.json",
        ssd_log_path: Optional[str] = None,
        scratchpad_store: Optional[Any] = None,
    ) -> None:
        self.metrics_path = metrics_path
        self.ssd_log_path = ssd_log_path
        self.scratchpad_store = scratchpad_store
        self.metrics = HarnessMetrics()
        self.metrics.load(metrics_path)

    def _log_ssd_candidate(self, question: str, response: str, retried: bool) -> None:
        """Append a constitutionally-passing response to the SSD training log."""
        if self.ssd_log_path is None:
            return
        entry = {
            "question": question,
            "response": response,
            "retried": retried,
            "principles_checked": list(_CHECKED_PRINCIPLES),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        Path(self.ssd_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.ssd_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def check_and_steer(
        self,
        response: str,
        conv: List[Dict],
        question: str,
        tool_profile_label: str,
        generate_fn: Callable[[List[Dict], float], str],
        max_retries: int = 2,
        session_id: Optional[str] = None,
    ) -> Tuple[str, List[str], int]:
        """Validate response; retry with corrective prompt and escalating temperature on failure.

        generate_fn(conv, temp_scale) — temp_scale=1.0 on normal generation; >1.0 on retries.
        session_id — used for P24b scratchpad task-completion check.
        Returns: (final_response, final_violations, retries_used)
        """
        print(f"[HARNESS] Checking response for constitutional violations...")

        violations = run_checks(
            response, question, tool_profile_label,
            scratchpad_store=self.scratchpad_store,
            session_id=session_id,
        )

        if not violations:
            print(f"[HARNESS] ✓ No violations — response passed (P1, P3, P4, P18)")
            self.metrics.record([], retried=False)
            self.metrics.save(self.metrics_path)
            self._log_ssd_candidate(question, response, retried=False)
            return response, [], 0

        print(f"[HARNESS] ✗ Violations found ({len(violations)}):")
        for v in violations:
            print(f"[HARNESS]   · {v}")

        retries_used       = 0
        current_response   = response
        current_violations = violations

        for attempt in range(1, max_retries + 1):
            temp_scale = (
                _RETRY_TEMP_SCALES[attempt - 1]
                if attempt - 1 < len(_RETRY_TEMP_SCALES)
                else _RETRY_TEMP_SCALES[-1]
            )
            corrective = build_corrective_prompt(current_violations)
            retry_conv = conv + [{"role": "user", "content": corrective}]
            print(f"[HARNESS] Injecting corrective prompt → retry {attempt}/{max_retries} (temp_scale={temp_scale})...")
            current_response   = generate_fn(retry_conv, temp_scale)
            retries_used       = attempt
            current_violations = run_checks(
                current_response, question, tool_profile_label,
                scratchpad_store=self.scratchpad_store,
                session_id=session_id,
            )

            if not current_violations:
                print(f"[HARNESS] ✓ Retry {attempt} passed — violations cleared")
                self.metrics.record([], retried=True, retry_cleared=True)
                self.metrics.save(self.metrics_path)
                self._log_ssd_candidate(question, current_response, retried=True)
                return current_response, [], retries_used

            print(f"[HARNESS] ✗ Retry {attempt} still violated ({', '.join(v.split(':')[0] for v in current_violations)})")

        print(f"[HARNESS] ✗ Exhausted {max_retries} retries — returning best response with violation flags")
        self.metrics.record(current_violations, retried=True, retry_cleared=False)
        self.metrics.save(self.metrics_path)
        return current_response, current_violations, retries_used

    def log_adaptation(self) -> None:
        """Print adaptation status to server terminal."""
        active = self.metrics.snapshot()["adaptation_active"]
        if active:
            for p in active:
                rate = self.metrics.fail_rate(p)
                print(f"[HARNESS] Adaptation active: {p} fail_rate={rate:.2f} — reinforcing in system prompt")
