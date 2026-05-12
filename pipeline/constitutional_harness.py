"""
Constitutional Harness
======================
Inference-time validation and correction loop for constitutional principle compliance.

Checks P1 (CAPABILITY_CHECK), P3 (TOOL DISCIPLINE), P4 (MATH=CODE), P18 (ANSWER PRESENT)
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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime",
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

_CHECKED_PRINCIPLES = ("P1", "P3", "P4", "P18")
_ADAPTATION_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

def run_checks(
    response: str,
    question: str,
    tool_profile_label: str,
) -> List[str]:
    """Deterministic constitutional checks for P1, P3, P4, P18.

    Returns a list of violation strings in 'PRINCIPLE_N: description' format.
    Empty list means the response is compliant.
    """
    violations: List[str] = []
    active_tools = _TOOL_PROFILE_MAP.get(tool_profile_label, frozenset())

    # ── P1: <think> + CAPABILITY_CHECK ──────────────────────────────────────
    has_think    = bool(re.search(r"<think\b", response, re.IGNORECASE))
    has_capcheck = "CAPABILITY_CHECK" in response

    if not has_think:
        violations.append(
            "PRINCIPLE_1: <think> block is entirely absent. "
            "Every response must open with <think>CAPABILITY_CHECK...</think>."
        )
    elif not has_capcheck:
        violations.append(
            "PRINCIPLE_1: <think> block present but CAPABILITY_CHECK label is missing. "
            "The capability check must be explicitly labelled so it can be audited."
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

    return violations


# ---------------------------------------------------------------------------
# build_corrective_prompt
# ---------------------------------------------------------------------------

def build_corrective_prompt(violations: List[str]) -> str:
    """Format violations into a targeted corrective user turn for retry injection."""
    if not violations:
        return (
            "[HARNESS] Please review your previous response and ensure it follows "
            "the constitutional format: <think>CAPABILITY_CHECK...</think><answer>...</answer>."
        )
    bullet_list = "\n".join(f"  - {v}" for v in violations)
    return (
        "[HARNESS] Your previous response had constitutional violations that must be "
        "corrected before the response can be shown to the user:\n"
        f"{bullet_list}\n\n"
        "Please rewrite your response in full, fixing each violation listed above. "
        "Remember: every response requires <think>CAPABILITY_CHECK...</think> and <answer>...</answer>."
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
            "P1":  "**CRITICAL: Every response MUST open with <think>CAPABILITY_CHECK...</think>. This is non-negotiable.**",
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

    def __init__(self, metrics_path: str = "reports/harness_metrics.json") -> None:
        self.metrics_path = metrics_path
        self.metrics = HarnessMetrics()
        self.metrics.load(metrics_path)

    def check_and_steer(
        self,
        response: str,
        conv: List[Dict],
        question: str,
        tool_profile_label: str,
        generate_fn: Callable[[List[Dict]], str],
        max_retries: int = 2,
    ) -> Tuple[str, List[str], int]:
        """Validate response; retry with corrective prompt on failure.

        Returns: (final_response, final_violations, retries_used)
        """
        print(f"[HARNESS] Checking response for constitutional violations...")

        violations = run_checks(response, question, tool_profile_label)

        if not violations:
            print(f"[HARNESS] ✓ No violations — response passed (P1, P3, P4, P18)")
            self.metrics.record([], retried=False)
            self.metrics.save(self.metrics_path)
            return response, [], 0

        print(f"[HARNESS] ✗ Violations found ({len(violations)}):")
        for v in violations:
            print(f"[HARNESS]   · {v}")

        retries_used      = 0
        current_response  = response
        current_violations = violations

        for attempt in range(1, max_retries + 1):
            corrective = build_corrective_prompt(current_violations)
            retry_conv = conv + [{"role": "user", "content": corrective}]
            print(f"[HARNESS] Injecting corrective prompt → retry {attempt}/{max_retries}...")
            current_response   = generate_fn(retry_conv)
            retries_used       = attempt
            current_violations = run_checks(current_response, question, tool_profile_label)

            if not current_violations:
                print(f"[HARNESS] ✓ Retry {attempt} passed — violations cleared")
                self.metrics.record([], retried=True, retry_cleared=True)
                self.metrics.save(self.metrics_path)
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
