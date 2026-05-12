# Constitutional Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an inference-time constitutional validation and correction loop that intercepts every model response, checks P1/P3/P4/P18 compliance, retries with targeted corrections on failure, tracks per-principle failure rates for adaptive system-prompt reinforcement, and enables side-by-side with/without harness benchmarking.

**Architecture:** New module `constitutional_harness.py` owns all check logic and metrics tracking. `3_infererence.py` plugs it in after the tool loop via a `check_and_steer()` call and exposes two new endpoints. `4_benchmark.py` gains a `--with_harness` flag that runs the full probe suite twice and saves a comparison report.

**Tech Stack:** Python stdlib only for the harness module (re, json, collections, pathlib). FastAPI (already in use) for endpoints. requests (already in use) for benchmark HTTP calls.

**Spec:** `docs/superpowers/specs/2026-05-12-constitutional-harness-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `pipeline/constitutional_harness.py` | All check logic, corrective prompts, metrics tracking, adaptive suffix |
| Create | `pipeline/tests/test_constitutional_harness.py` | Unit tests for harness module |
| Modify | `pipeline/config.py` | Add `ENABLE_HARNESS` flag |
| Modify | `pipeline/3_infererence.py` | Tool prints, harness startup + hook, new endpoints, adaptation suffix in system prompt, `harness_enabled` request field |
| Modify | `pipeline/4_benchmark.py` | `harness_enabled` param on `_complete()`, `run_probe()`, `run_constitution_probes()`, `--with_harness` CLI flag, comparison report |
| Modify | `README.md` | New flag, new endpoints, `--with_harness` usage |

---

## Task 1 — Add `ENABLE_HARNESS` to config

**Files:**
- Modify: `pipeline/config.py`

- [ ] **Add the flag** after `ENABLE_ONTOLOGY_VERIF`:

```python
# ENABLE_HARNESS: Inference-time constitutional validation and correction loop.
# After each model response, checks P1/P3/P4/P18 compliance. On violation,
# injects a corrective prompt and retries (up to 2 times). Tracks per-principle
# failure rates and adaptively reinforces weak principles in the system prompt.
ENABLE_HARNESS: bool = False
```

- [ ] **Verify it reads from env** by running:

```bash
cd pipeline
python -c "import os; os.environ['PIPELINE_ENABLE_HARNESS']='true'; from config import PipelineConfig; c=PipelineConfig.from_env(); print(c.ENABLE_HARNESS)"
```
Expected: `True`

- [ ] **Commit:**

```bash
git add pipeline/config.py
git commit -m "feat: add ENABLE_HARNESS flag to pipeline config"
```

---

## Task 2 — Create `constitutional_harness.py` with tests

**Files:**
- Create: `pipeline/constitutional_harness.py`
- Create: `pipeline/tests/__init__.py`
- Create: `pipeline/tests/test_constitutional_harness.py`

### Step 2a — Write the failing tests first

- [ ] **Create `pipeline/tests/__init__.py`** (empty file)

- [ ] **Create `pipeline/tests/test_constitutional_harness.py`:**

```python
"""Unit tests for constitutional_harness.py — run with: pytest pipeline/tests/test_constitutional_harness.py -v"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from constitutional_harness import run_checks, build_corrective_prompt, HarnessMetrics, ConstitutionalHarness


# ── Fixtures ────────────────────────────────────────────────────────────────

PROFILE_COMPUTE = "compute_only"    # python_execute ✓ only
PROFILE_NO_TOOLS = "no_tools"       # no tools

GOOD_RESPONSE = (
    "<think>CAPABILITY_CHECK: I have python_execute available.\n"
    "The user wants 7+3. I will use python_execute.\n</think>"
    "<tool>python_execute(code='print(7+3)')</tool>"
    "<answer>The answer is 10.</answer>"
)

BAD_NO_THINK = "The answer is 10."

BAD_NO_CAPCHECK = "<think>I will compute this.</think><answer>10.</answer>"

BAD_NO_ANSWER = "<think>CAPABILITY_CHECK: ok.</think>I think it is 10."

BAD_HALLUCINATED_TOOL = (
    "<think>CAPABILITY_CHECK: ok.</think>"
    "<tool>fly_to_moon(destination='Mars')</tool>"
    "<answer>Done.</answer>"
)

BAD_UNAVAILABLE_TOOL = (
    "<think>CAPABILITY_CHECK: web_search not available.</think>"
    "<tool>web_search(query='weather')</tool>"
    "<answer>Here is the weather.</answer>"
)

BAD_MATH_NO_CODE = (
    "<think>CAPABILITY_CHECK: python_execute available.</think>"
    "<answer>7823 multiplied by 349 minus 1024 is 2,730,603.</answer>"
)

MATH_QUESTION = "What is 7823 multiplied by 349, then subtract 1024?"
PLAIN_QUESTION = "What is the capital of France?"


# ── run_checks ───────────────────────────────────────────────────────────────

def test_run_checks_clean_response():
    violations = run_checks(GOOD_RESPONSE, MATH_QUESTION, PROFILE_COMPUTE)
    assert violations == [], f"Expected no violations, got: {violations}"


def test_run_checks_missing_think_block():
    violations = run_checks(BAD_NO_THINK, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_1" in v for v in violations)
    assert any("think" in v.lower() for v in violations)


def test_run_checks_missing_capability_check():
    violations = run_checks(BAD_NO_CAPCHECK, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_1" in v for v in violations)
    assert any("CAPABILITY_CHECK" in v for v in violations)


def test_run_checks_missing_answer_block():
    violations = run_checks(BAD_NO_ANSWER, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_18" in v for v in violations)


def test_run_checks_hallucinated_tool():
    violations = run_checks(BAD_HALLUCINATED_TOOL, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_3" in v for v in violations)
    assert any("fly_to_moon" in v for v in violations)


def test_run_checks_unavailable_tool():
    violations = run_checks(BAD_UNAVAILABLE_TOOL, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_3" in v for v in violations)
    assert any("web_search" in v for v in violations)


def test_run_checks_math_no_code():
    violations = run_checks(BAD_MATH_NO_CODE, MATH_QUESTION, PROFILE_COMPUTE)
    assert any("PRINCIPLE_4" in v for v in violations)


def test_run_checks_math_no_code_skipped_when_no_tool():
    # P4 only fires when python_execute is available
    violations = run_checks(BAD_MATH_NO_CODE, MATH_QUESTION, PROFILE_NO_TOOLS)
    assert not any("PRINCIPLE_4" in v for v in violations)


def test_run_checks_multiple_violations():
    violations = run_checks(BAD_NO_THINK, MATH_QUESTION, PROFILE_COMPUTE)
    # Missing think → P1; missing answer → P18
    assert len(violations) >= 2


# ── build_corrective_prompt ──────────────────────────────────────────────────

def test_build_corrective_prompt_contains_violations():
    violations = ["PRINCIPLE_1: missing CAPABILITY_CHECK", "PRINCIPLE_18: no answer block"]
    prompt = build_corrective_prompt(violations)
    assert "PRINCIPLE_1" in prompt
    assert "PRINCIPLE_18" in prompt
    assert "[HARNESS]" in prompt


def test_build_corrective_prompt_empty():
    prompt = build_corrective_prompt([])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# ── HarnessMetrics ───────────────────────────────────────────────────────────

def test_harness_metrics_record_violations():
    m = HarnessMetrics()
    m.record(["PRINCIPLE_1: missing capcheck"])
    snap = m.snapshot()
    assert snap["principles"]["P1"]["failures"] == 1
    assert snap["principles"]["P1"]["checks"] == 1


def test_harness_metrics_record_pass():
    m = HarnessMetrics()
    m.record([])
    snap = m.snapshot()
    for p in snap["principles"].values():
        assert p["failures"] == 0
        assert p["checks"] == 1


def test_harness_metrics_adaptation_suffix_empty_when_healthy():
    m = HarnessMetrics()
    for _ in range(10):
        m.record([])
    assert m.get_adaptation_suffix() == ""


def test_harness_metrics_adaptation_suffix_present_when_failing():
    m = HarnessMetrics(window_size=10)
    for _ in range(4):  # 4/10 = 40% > 30% threshold
        m.record(["PRINCIPLE_4: math without code"])
    suffix = m.get_adaptation_suffix()
    assert "P4" in suffix or "MATH" in suffix


def test_harness_metrics_save_load(tmp_path):
    m = HarnessMetrics()
    m.record(["PRINCIPLE_1: missing capcheck"])
    path = tmp_path / "harness_metrics.json"
    m.save(str(path))
    m2 = HarnessMetrics()
    m2.load(str(path))
    snap = m2.snapshot()
    assert snap["principles"]["P1"]["failures"] == 1


# ── ConstitutionalHarness.check_and_steer ───────────────────────────────────

def test_harness_passes_clean_response():
    harness = ConstitutionalHarness()
    calls = []

    def fake_generate(conv):
        calls.append(conv)
        return GOOD_RESPONSE

    result, violations, retries = harness.check_and_steer(
        response=GOOD_RESPONSE,
        conv=[{"role": "user", "content": MATH_QUESTION}],
        question=MATH_QUESTION,
        tool_profile_label=PROFILE_COMPUTE,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert violations == []
    assert retries == 0
    assert len(calls) == 0  # no retry needed


def test_harness_retries_on_violation():
    harness = ConstitutionalHarness()
    calls = []

    def fake_generate(conv):
        calls.append(conv)
        return GOOD_RESPONSE  # clean on retry

    result, violations, retries = harness.check_and_steer(
        response=BAD_NO_ANSWER,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert retries == 1
    assert violations == []
    assert len(calls) == 1


def test_harness_exhausts_retries_returns_best():
    harness = ConstitutionalHarness()

    def always_bad(conv):
        return BAD_NO_THINK  # always violates

    result, violations, retries = harness.check_and_steer(
        response=BAD_NO_THINK,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=always_bad,
        max_retries=2,
    )
    assert retries == 2
    assert len(violations) > 0  # still reports violations
```

- [ ] **Run tests to confirm they all fail** (module doesn't exist yet):

```bash
cd pipeline
python -m pytest tests/test_constitutional_harness.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'constitutional_harness'`

### Step 2b — Implement `constitutional_harness.py`

- [ ] **Create `pipeline/constitutional_harness.py`:**

```python
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
# Constants — kept in sync with sft_gold_response_generator.py
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime",
})

# Tool profiles — maps label → set of active tool names
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

# Principles checked at inference time (P14 excluded — needs multi-turn turn tags)
_CHECKED_PRINCIPLES = ("P1", "P3", "P4", "P18")

# Failure rate above this threshold triggers adaptive suffix for that principle
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
    # Check both XML (<tool>name(...)</tool>) and native (<tool_call>{"name":...}</tool_call>) formats
    xml_calls    = set(re.findall(r"<tool>(\w+)\(", response))
    native_calls = set(re.findall(r'"name"\s*:\s*"(\w+)"', response))
    called_tools = xml_calls | native_calls

    hallucinated = called_tools - _ALL_TOOL_NAMES
    unavailable  = (called_tools & _ALL_TOOL_NAMES) - active_tools

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
        needs_math      = bool(_MATH_SIGNAL_RE.search(question))
        has_code_call   = bool(re.search(r"<tool>\s*python_execute|\"name\"\s*:\s*\"python_execute\"", response))
        numeric_answer  = bool(re.search(r"<answer>.*\d[\d,.]+.*</answer>", response, re.DOTALL))
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
        self.window_size = window_size
        self.request_count = 0
        self.total_retries = 0
        self.retry_successes = 0
        # Per-principle rolling windows: deque of bools (True=fail)
        self._windows: Dict[str, deque] = {p: deque(maxlen=window_size) for p in _CHECKED_PRINCIPLES}

    def record(self, violations: List[str], retried: bool = False, retry_cleared: bool = False) -> None:
        """Record one request's violations. Call once per final response."""
        self.request_count += 1
        if retried:
            self.total_retries += 1
        if retry_cleared:
            self.retry_successes += 1
        violated_principles = {v.split(":")[0].strip() for v in violations}
        for p in _CHECKED_PRINCIPLES:
            self._windows[p].append(p in violated_principles)

    def fail_rate(self, principle: str) -> float:
        """Fraction of recent requests that violated this principle (0.0–1.0)."""
        w = self._windows.get(principle, deque())
        return sum(w) / len(w) if w else 0.0

    def get_adaptation_suffix(self) -> str:
        """Return a bolded reminder block for any principle exceeding the failure threshold.

        Empty string when all principles are healthy (fail_rate < threshold).
        Injected into the live system prompt so the model gets stronger guidance
        on the principles it is currently weakest on.
        """
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
        """Return full metrics dict — used by /harness/metrics endpoint and save()."""
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
        """Persist metrics to JSON so they survive server restarts."""
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
        self.request_count  = data.get("request_count", 0)
        self.total_retries  = data.get("total_retries", 0)
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

        Args:
            response:           The model's initial response text.
            conv:               Full conversation list (role/content dicts) up to and
                                including the assistant's response.
            question:           The user's latest question (for P4 math signal check).
            tool_profile_label: Active tool profile name (e.g. 'compute_only').
            generate_fn:        Callable that takes a conversation list and returns
                                a new response string. Wraps _generate() in inference server.
            max_retries:        Maximum number of corrective retry attempts (default 2).

        Returns:
            (final_response, final_violations, retries_used)
            final_violations is [] if the response eventually passed.
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

        retries_used = 0
        current_response = response
        current_violations = violations

        for attempt in range(1, max_retries + 1):
            corrective = build_corrective_prompt(current_violations)
            retry_conv = conv + [{"role": "user", "content": corrective}]
            print(f"[HARNESS] Injecting corrective prompt → retry {attempt}/{max_retries}...")
            current_response = generate_fn(retry_conv)
            retries_used = attempt
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
```

- [ ] **Run tests — all should pass:**

```bash
cd pipeline
python -m pytest tests/test_constitutional_harness.py -v
```
Expected: All tests pass.

- [ ] **Commit:**

```bash
git add pipeline/constitutional_harness.py pipeline/tests/__init__.py pipeline/tests/test_constitutional_harness.py
git commit -m "feat: add constitutional_harness module with unit tests"
```

---

## Task 3 — Add tool prints to `3_infererence.py`

**Files:**
- Modify: `pipeline/3_infererence.py` (lines ~889–906, the tool execution block)

- [ ] **Replace the tool execution block** (find the section starting `tc = _parse_native_tool_call...`):

Replace:
```python
            # Parse tool call — XML path for trained tools, JSON path for native/new tools
            tc = _parse_native_tool_call(response) if use_native else _parse_tool_call(response)
            if tc:
                fn_name = tc["function"]
                if fn_name not in _REGISTRY:
                    raw_result = f"Error: tool '{fn_name}' is not registered on this server."
                elif not use_native and fn_name not in active_tools:
                    raw_result = f"Error: tool '{fn_name}' is not available in profile '{req.tool_profile}'."
                else:
                    try:
                        raw_result = _REGISTRY[fn_name].fn(**tc["kwargs"])
                    except Exception as e:
                        raw_result = f"Tool execution error: {e}"
                tools_used[fn_name] = tools_used.get(fn_name, 0) + 1
                result = _sanitise_tool_output(fn_name, str(raw_result))
                conv.append({"role": "tool", "content": result})
            else:
                break
```

With:
```python
            # Parse tool call — XML path for trained tools, JSON path for native/new tools
            tc = _parse_native_tool_call(response) if use_native else _parse_tool_call(response)
            if tc:
                fn_name = tc["function"]
                kwargs_preview = str(tc["kwargs"])[:120]
                print(f"[TOOL] Calling: {fn_name}({kwargs_preview})")
                if fn_name not in _REGISTRY:
                    raw_result = f"Error: tool '{fn_name}' is not registered on this server."
                    print(f"[TOOL] Error: tool '{fn_name}' is not registered on this server")
                elif not use_native and fn_name not in active_tools:
                    raw_result = f"Error: tool '{fn_name}' is not available in profile '{req.tool_profile}'."
                    print(f"[TOOL] Error: tool '{fn_name}' not available in profile '{req.tool_profile}'")
                else:
                    try:
                        raw_result = _REGISTRY[fn_name].fn(**tc["kwargs"])
                        result_preview = str(raw_result)[:80].replace("\n", "\\n")
                        print(f"[TOOL] Result ({len(str(raw_result))} chars): {result_preview}")
                    except Exception as e:
                        raw_result = f"Tool execution error: {e}"
                        print(f"[TOOL] Execution error in {fn_name}: {e}")
                tools_used[fn_name] = tools_used.get(fn_name, 0) + 1
                result = _sanitise_tool_output(fn_name, str(raw_result))
                conv.append({"role": "tool", "content": result})
            else:
                break
```

- [ ] **Verify syntax:**

```bash
cd pipeline
python -c "import ast; ast.parse(open('3_infererence.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Commit:**

```bash
git add pipeline/3_infererence.py
git commit -m "feat: add [TOOL] print statements to inference server tool loop"
```

---

## Task 4 — Wire harness into `3_infererence.py`

**Files:**
- Modify: `pipeline/3_infererence.py`

Four changes in this task: (A) guarded import + global, (B) `harness_enabled` field on `CompletionRequest`, (C) harness hook in `chat_completions`, (D) new endpoints + adaptation suffix in `_build_system_prompt`.

### 4A — Guarded import and global

- [ ] **Add after the other guarded imports** (after the `_ontology_available` block):

```python
try:
    from constitutional_harness import ConstitutionalHarness
    _harness_available = True
except ImportError as _e:
    _harness_available = False
    ConstitutionalHarness = None
    print(f"[INFO] constitutional_harness not importable ({_e}) — ENABLE_HARNESS disabled")
```

- [ ] **Add to globals section** (near `_ONTO_GRAPH`, `_DEPENDENCY_MONITOR` etc):

```python
_HARNESS: Optional[Any] = None   # ConstitutionalHarness — set at startup when ENABLE_HARNESS=True
```

- [ ] **Add harness instantiation to server startup** (find the `@app.on_event("startup")` block or the model-loading section; add after model is loaded):

```python
    # ── Constitutional Harness ────────────────────────────────────────────
    global _HARNESS
    if cfg.ENABLE_HARNESS:
        if _harness_available:
            _HARNESS = ConstitutionalHarness(metrics_path="reports/harness_metrics.json")
            print(f"[HARNESS] Constitutional harness enabled (max_retries=2, window=50)")
            _HARNESS.log_adaptation()
        else:
            print("[HARNESS] ENABLE_HARNESS=true but constitutional_harness module not found — skipping")
    else:
        print("[HARNESS] Disabled (set PIPELINE_ENABLE_HARNESS=true to enable)")
```

### 4B — `harness_enabled` field on `CompletionRequest`

- [ ] **Add to `CompletionRequest`** after `tool_mode`:

```python
    harness_enabled: Optional[bool] = None
    # Override cfg.ENABLE_HARNESS for this single request.
    # None → use server default. True → always run harness. False → skip harness.
    # Used by 4_benchmark.py --with_harness to toggle per probe without server restart.
```

### 4C — Harness hook in `chat_completions`

- [ ] **Add harness block** after the `dep_disclosure` block and before the `onto_score` block:

```python
        # ── Constitutional Harness: validate + steer ──────────────────────
        harness_violations: List[str] = []
        harness_retries: int = 0
        effective_harness = (
            req.harness_enabled if req.harness_enabled is not None else cfg.ENABLE_HARNESS
        )
        if effective_harness and _HARNESS is not None:
            adaptation_needed = bool(_HARNESS.metrics.get_adaptation_suffix())
            final, harness_violations, harness_retries = _HARNESS.check_and_steer(
                response=final,
                conv=conv,
                question=user_turn,
                tool_profile_label=req.tool_profile,
                generate_fn=lambda c: _generate(
                    c, req.max_new_tokens, req.temperature, req.greedy
                )[0],
                max_retries=2,
            )
            if adaptation_needed != bool(_HARNESS.metrics.get_adaptation_suffix()):
                _HARNESS.log_adaptation()
```

- [ ] **Add `harness_violations` and `harness_retries` to the return dict** after `"ontology_score"`:

```python
            "harness_violations": harness_violations,
            "harness_retries":    harness_retries,
```

### 4D — New endpoints and adaptation suffix

- [ ] **Add `/harness/metrics` and `/harness/reset` endpoints** after the `/metrics/reset` endpoint:

```python
@app.get("/harness/metrics")
def harness_metrics() -> Dict[str, Any]:
    """Per-principle failure rates, retry stats, and current adaptation state."""
    if _HARNESS is None:
        raise HTTPException(status_code=404, detail="Harness not enabled (set PIPELINE_ENABLE_HARNESS=true)")
    return _HARNESS.metrics.snapshot()


@app.post("/harness/reset")
def harness_reset() -> Dict[str, Any]:
    """Reset rolling harness metrics counters."""
    if _HARNESS is None:
        raise HTTPException(status_code=404, detail="Harness not enabled")
    from constitutional_harness import HarnessMetrics
    _HARNESS.metrics = HarnessMetrics()
    print("[HARNESS] Metrics reset")
    return {"status": "reset"}
```

- [ ] **Inject adaptation suffix in `_build_system_prompt`** — append it as a final part:

```python
def _build_system_prompt(
    base: str,
    user_ctx: Optional[Any] = None,
    appraisal_ctx: Optional[Any] = None,
) -> str:
    parts = []
    if cfg.ENABLE_EMPATHY:
        parts.append(APPRAISAL_SYSTEM_PREFIX)
    parts.append(base)
    if cfg.ENABLE_PERSONALISATION and user_ctx is not None and not user_ctx.is_empty():
        parts.append("\n" + user_ctx.to_prompt_block())
    # Harness meta-adaptation: reinforce principles the model is currently failing
    if cfg.ENABLE_HARNESS and _HARNESS is not None:
        suffix = _HARNESS.metrics.get_adaptation_suffix()
        if suffix:
            parts.append(suffix)
    return "".join(parts)
```

- [ ] **Verify syntax:**

```bash
cd pipeline
python -c "import ast; ast.parse(open('3_infererence.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Commit:**

```bash
git add pipeline/3_infererence.py
git commit -m "feat: wire constitutional harness into inference server with endpoints and adaptation"
```

---

## Task 5 — Harness comparison in `4_benchmark.py`

**Files:**
- Modify: `pipeline/4_benchmark.py`

Three changes: (A) `harness_enabled` on `_complete()`, (B) `run_probe()` + `run_constitution_probes()` accept it, (C) `--with_harness` flag and comparison function.

### 5A — Add `harness_enabled` to `_complete()`

- [ ] **Replace `_complete()`:**

```python
def _complete(server_url: str, messages: List[Dict], tool_profile: str,
              system_override: Optional[str] = None,
              max_new_tokens: int = 1024, temperature: float = 0.7,
              harness_enabled: Optional[bool] = None) -> Dict[str, Any]:
    body = {
        "messages":       messages,
        "tool_profile":   tool_profile,
        "system_override": system_override,
        "max_new_tokens": max_new_tokens,
        "temperature":    temperature,
    }
    if harness_enabled is not None:
        body["harness_enabled"] = harness_enabled
    return _http(server_url, "/v1/chat/completions", "POST", body)
```

### 5B — Thread `harness_enabled` through `run_probe()` and `run_constitution_probes()`

- [ ] **Update `run_probe()` signature** (add `harness_enabled` parameter):

```python
def run_probe(
    server_url: str,
    probe: Dict[str, Any],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    harness_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    tool_profile = probe.get("tool_profile", "no_tools")
    system       = probe.get("system", _SYS_NONE)
    questions    = probe["question"] if isinstance(probe["question"], list) else [probe["question"]]
    history: List[Dict] = []
    final_response = ""

    for q in questions:
        history.append({"role": "user", "content": q})
        result = _complete(
            server_url, history, tool_profile, system, max_new_tokens, temperature,
            harness_enabled=harness_enabled,
        )
        final_response = result["response"]
        history.append({"role": "assistant", "content": final_response})

    try:
        passed = bool(probe["check"](final_response))
    except Exception:
        passed = False

    return {
        "id":          probe["id"],
        "principle":   probe["principle"],
        "description": probe["description"],
        "question":    probe["question"],
        "response":    final_response,
        "passed":      passed,
        "score":       1.0 if passed else 0.0,
        "harness_retries":    result.get("harness_retries", 0),
        "harness_violations": result.get("harness_violations", []),
    }
```

- [ ] **Update `run_constitution_probes()` signature** (add `harness_enabled`):

```python
def run_constitution_probes(
    server_url: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    baseline_path: Optional[Path] = None,
    harness_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    total = len(CONSTITUTIONAL_PROBES)
    harness_tag = "" if harness_enabled is None else (" [HARNESS ON]" if harness_enabled else " [HARNESS OFF]")
    print(f"\n{'='*60}")
    print(f"  CONSTITUTIONAL DRIFT PROBE SUITE{harness_tag}")
    print(f"{'='*60}")
    print(f"  Server       : {server_url}")
    print(f"  Probes       : {total}")
    print(f"  Max tokens   : {max_new_tokens}  |  Temperature: {temperature}")
    if baseline_path:
        bl = str(baseline_path) if baseline_path.exists() else f"{baseline_path} (missing)"
        print(f"  Baseline     : {bl}")

    results = []
    for idx, probe in enumerate(CONSTITUTIONAL_PROBES, 1):
        q_short = (probe["question"][0] if isinstance(probe["question"], list) else probe["question"])[:60]
        print(f"\n  [{idx}/{total}] [{probe['id']}] {q_short}...")
        result = run_probe(server_url, probe, max_new_tokens, temperature, harness_enabled=harness_enabled)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        retries_note = f"  retries={result['harness_retries']}" if result.get("harness_retries") else ""
        print(f"  → {status}  ({probe['description']}){retries_note}")

    scores = [r["score"] for r in results]
    constitution_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Score: {constitution_score:.3f}  ({int(sum(scores))}/{len(scores)} passed)")

    drift: Optional[float] = None
    drift_warning = False
    if baseline_path and baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        b_score = baseline.get("constitution_score")
        if b_score is not None:
            drift = constitution_score - b_score
            drift_warning = drift < -DRIFT_THRESHOLD
            tag = "*** DRIFT WARNING ***" if drift_warning else "OK"
            print(f"  Drift from baseline ({b_score:.3f}): {drift:+.3f}  [{tag}]")

    return {
        "constitution_score":  round(constitution_score, 4),
        "probes_passed":       int(sum(scores)),
        "probes_total":        len(scores),
        "scores_by_principle": {r["id"]: r["score"] for r in results},
        "drift_from_baseline": round(drift, 4) if drift is not None else None,
        "drift_warning":       drift_warning,
        "probe_results":       results,
    }
```

### 5C — Comparison function and `--with_harness` flag

- [ ] **Add `run_harness_comparison()` function** before `main()`:

```python
def run_harness_comparison(
    server_url: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    output_dir: Path = Path("reports"),
    timestamp: str = "",
) -> Dict[str, Any]:
    """Run constitutional probes twice — without then with harness — and produce a diff report."""
    print(f"\n{'='*60}")
    print("  HARNESS COMPARISON — WITH vs WITHOUT")
    print(f"{'='*60}")

    print(f"\n  [HARNESS COMPARISON] Running probes without harness...")
    without = run_constitution_probes(server_url, max_new_tokens, temperature, harness_enabled=False)
    score_without = without["constitution_score"]
    print(f"  [HARNESS COMPARISON] Without harness: constitution_score={score_without:.4f} "
          f"({without['probes_passed']}/{without['probes_total']} passed)")

    print(f"\n  [HARNESS COMPARISON] Running probes with harness...")
    with_h = run_constitution_probes(server_url, max_new_tokens, temperature, harness_enabled=True)
    score_with = with_h["constitution_score"]
    delta = score_with - score_without
    print(f"  [HARNESS COMPARISON] With harness:    constitution_score={score_with:.4f} "
          f"({with_h['probes_passed']}/{with_h['probes_total']} passed)  [{delta:+.4f}]")

    # Per-principle delta
    delta_by_principle = {}
    print(f"  [HARNESS COMPARISON] Per-principle delta:")
    for probe_id in without["scores_by_principle"]:
        s_without = without["scores_by_principle"][probe_id]
        s_with    = with_h["scores_by_principle"].get(probe_id, 0.0)
        d = s_with - s_without
        delta_by_principle[probe_id] = round(d, 4)
        was = "PASS" if s_without == 1.0 else "FAIL"
        now = "PASS" if s_with == 1.0 else "FAIL"
        marker = "(+1)" if d > 0 else "(-1)" if d < 0 else "( 0)"
        print(f"    {probe_id:<30}: {was} → {now}  {marker}")

    # Harness retry stats from with_harness run
    total_retries   = sum(r.get("harness_retries", 0) for r in with_h["probe_results"])
    retried_probes  = [r for r in with_h["probe_results"] if r.get("harness_retries", 0) > 0]
    retry_successes = sum(1 for r in retried_probes if r["passed"])
    retry_rate      = round(retry_successes / len(retried_probes), 3) if retried_probes else None

    print(f"  [HARNESS COMPARISON] Retries triggered: {total_retries}  |  "
          f"Retry success rate: {retry_rate*100:.1f}%" if retry_rate is not None
          else f"  [HARNESS COMPARISON] Retries triggered: {total_retries}")

    report = {
        "timestamp":       timestamp,
        "server_url":      server_url,
        "without_harness": without,
        "with_harness":    with_h,
        "delta": {
            "constitution_score":  round(delta, 4),
            "scores_by_principle": delta_by_principle,
        },
        "harness_stats": {
            "total_retries":      total_retries,
            "retry_success_rate": retry_rate,
            "principles_triggered": list({
                v.split(":")[0].strip()
                for r in retried_probes
                for v in r.get("harness_violations", [])
            }),
        },
    }
    save_report(report, output_dir, f"constitution_probe_harness_comparison_{timestamp}.json")
    return report
```

- [ ] **Add `--with_harness` flag to the arg parser** (find the existing `ap.add_argument` block):

```python
ap.add_argument("--with_harness", action="store_true",
                help="Run constitutional probes with AND without harness; save comparison report")
```

- [ ] **Add harness comparison call in `main()`** (inside the `if args.probe_only or not args.adversarial_only` block, after the existing `run_constitution_probes` call):

```python
        if args.with_harness:
            run_harness_comparison(
                args.server_url,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                output_dir=output_dir,
                timestamp=timestamp,
            )
```

- [ ] **Verify syntax:**

```bash
cd pipeline
python -c "import ast; ast.parse(open('4_benchmark.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Commit:**

```bash
git add pipeline/4_benchmark.py
git commit -m "feat: add --with_harness flag to benchmark for with/without harness comparison"
```

---

## Task 6 — Update `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Add `ENABLE_HARNESS` to the feature flags table** (after `ENABLE_ONTOLOGY_VERIF`):

```markdown
| `PIPELINE_ENABLE_HARNESS` | `false` | Inference-time constitutional validation loop — checks P1/P3/P4/P18 on every response, retries with corrective prompt on failure, adapts system prompt to reinforce weak principles |
```

- [ ] **Add to the endpoints table** (after `/metrics/reset`):

```markdown
GET  /harness/metrics                   per-principle failure rates, retry stats, adaptation state
POST /harness/reset                     reset rolling harness counters
```

- [ ] **Add harness benchmarking usage** (after the constitutional probes section):

```markdown
### Harness comparison benchmark

Run the constitutional probe suite with and without the harness to quantify its contribution:

```bash
# Server must be running with PIPELINE_ENABLE_HARNESS=true
PIPELINE_ENABLE_HARNESS=true python pipeline/3_infererence.py --model_dir models/checkpoint_sft

# In a second terminal:
python pipeline/4_benchmark.py --probe_only --with_harness
# Saves: reports/constitution_probe_harness_comparison_{timestamp}.json
# Prints per-principle FAIL→PASS deltas and retry success rate
```
```

- [ ] **Commit:**

```bash
git add README.md
git commit -m "docs: document constitutional harness flag, endpoints, and benchmark usage"
```

---

## Task 7 — End-to-end smoke test

- [ ] **Run all unit tests:**

```bash
cd pipeline
python -m pytest tests/test_constitutional_harness.py -v
```
Expected: All pass.

- [ ] **Verify config flag works:**

```bash
cd pipeline
python -c "import os; os.environ['PIPELINE_ENABLE_HARNESS']='true'; from config import PipelineConfig; c=PipelineConfig.from_env(); print('ENABLE_HARNESS:', c.ENABLE_HARNESS)"
```
Expected: `ENABLE_HARNESS: True`

- [ ] **Verify standalone harness logic:**

```bash
cd pipeline
python -c "
from constitutional_harness import run_checks, build_corrective_prompt
bad = 'The answer is 42.'
v = run_checks(bad, 'What is 6 times 7?', 'no_tools')
print('Violations:', v)
print('Corrective:', build_corrective_prompt(v)[:120])
"
```
Expected: violations list contains `PRINCIPLE_1` and `PRINCIPLE_18`, corrective prompt contains `[HARNESS]`.

- [ ] **Verify syntax on all modified Python files:**

```bash
cd pipeline
python -c "
import ast
for f in ['constitutional_harness.py', '3_infererence.py', '4_benchmark.py', 'config.py']:
    ast.parse(open(f).read())
    print(f, 'OK')
"
```
Expected: All four print `OK`.

- [ ] **Final commit if anything was adjusted:**

```bash
git add -A
git commit -m "feat: constitutional harness — inference-time constitutional validation with meta-adaptation"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `run_checks()` for P1/P3/P4/P18 | Task 2 |
| `build_corrective_prompt()` | Task 2 |
| `HarnessMetrics` with rolling window, adaptation suffix, save/load | Task 2 |
| `ConstitutionalHarness.check_and_steer()` | Task 2 |
| `ENABLE_HARNESS` config flag | Task 1 |
| Guarded import in `3_infererence.py` | Task 4A |
| `harness_enabled` on `CompletionRequest` | Task 4B |
| Harness hook after tool loop | Task 4C |
| `harness_violations` + `harness_retries` in response | Task 4C |
| `/harness/metrics` + `/harness/reset` endpoints | Task 4D |
| Adaptation suffix injected into system prompt | Task 4D |
| `[TOOL]` print statements | Task 3 |
| `[HARNESS]` print statements | Task 2 (in `check_and_steer`) |
| `4_benchmark.py --with_harness` flag | Task 5C |
| Comparison report with per-principle delta | Task 5C |
| `[HARNESS COMPARISON]` benchmark prints | Task 5C |
| README documentation | Task 6 |

**All spec requirements covered. No gaps.**

**Placeholder scan:** No TBD/TODO/placeholder language found. All code blocks are complete.

**Type consistency:** `check_and_steer()` returns `Tuple[str, List[str], int]` — used correctly in Task 4C as `final, harness_violations, harness_retries`. `run_probe()` returns dict with `harness_retries` and `harness_violations` keys — consumed correctly in `run_harness_comparison()`. `HarnessMetrics.record()` signature used consistently across Tasks 2 and 4C.
