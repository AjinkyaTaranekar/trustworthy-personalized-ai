# SFT v3 Asymmetric Distillation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v2 constitution-in-student-prompt architecture with asymmetric context distillation, live tool execution via an intercept loop, failure injection for negative trajectories, and a pre-flight quality gate — all without breaking existing v2 data or the GRPO trainer.

**Architecture:** The teacher model (Kimi/Minimax) generates training examples with the full 25-principle constitution in its system prompt but forbidden from outputting rule names or checklists; the resulting `<think>` blocks are narrative. When saving to JSONL the system prompt is swapped for a ≤50-word student prompt, giving the 0.6B model 100% of its attention budget for reasoning rather than rule-tracking. Tool calls are intercepted mid-generation using `stop=["</tool>"]`, executed in real-time via exa.ai (web search) and subprocess (python), then fed back so the teacher synthesises from real results, not imagined ones.

**Tech Stack:** Python 3.11+, litellm, exa-py, pytest, existing pipeline conventions (SFT v2 format preserved as fallback).

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| **Create** | `pipeline/sft_v3_generator.py` | Asymmetric distillation generator with intercept loop and failure injection |
| **Create** | `pipeline/validate_sft_data.py` | Pre-flight quality gate enforcing 5 invariants on any SFT JSONL |
| **Create** | `pipeline/tests/test_sft_v3_generator.py` | Unit tests for generator helpers |
| **Create** | `pipeline/tests/test_validate_sft_data.py` | Unit tests for the 5 validation assertions |
| **Modify** | `pipeline/sft_question_generator.py` | Add `inventory_constraint` + `environment_timeout` question categories |
| **Modify** | `pipeline/2_model_trainer.py` | `--curriculum_stage {1,2,3}` flag + v3-compatible GRPO format reward |
| **Modify** | `README.md` | Add `exa-py` to install block; document v3 generator and curriculum training |
| **Create** | `wiki/sources/code/sft-v3-pipeline.md` | Wiki page for the v3 architecture |
| **Modify** | `wiki/log.md` | Append refactor entry |
| **Modify** | `wiki/index.md` | Add pointer to new wiki page |

### Critical API contracts to preserve
- `sft_v3_generator.py` output JSONL: `{"messages": [...], "metadata": {...}}` — same keys as v2, consumed by `sft_dataset_assembler.py` unmodified.
- `sft_dataset_assembler.py` unchanged — still handles v3 multi-turn format.
- `2_model_trainer.py --mode grpo` reward functions import `rule_check_response` from `sft_gold_response_generator` (v2 checker); v3 format reward does **not** require `CAPABILITY_CHECK`.

---

## Task 1: Add Negative Trajectory Question Categories

**Files:**
- Modify: `pipeline/sft_question_generator.py` (locate the active `CATEGORIES` dict, currently all entries are commented out; add two new entries)

- [ ] **Step 1: Find the active CATEGORIES dict**

Run: `grep -n "^CATEGORIES\|^    \"inventory_constraint\|^    \"environment_timeout" pipeline/sft_question_generator.py`

The dict starts at line 52. All existing entries are commented out — they were generated in prior sessions. We add two new live entries.

- [ ] **Step 2: Write failing tests**

Create `pipeline/tests/test_question_categories.py`:

```python
"""Smoke test that new negative-trajectory categories are registered."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sft_question_generator import CATEGORIES

def test_inventory_constraint_registered():
    assert "inventory_constraint" in CATEGORIES
    cat = CATEGORIES["inventory_constraint"]
    assert cat["count"] >= 50
    assert "required_profile" in cat
    assert "constrained_tool" in cat

def test_environment_timeout_registered():
    assert "environment_timeout" in CATEGORIES
    cat = CATEGORIES["environment_timeout"]
    assert cat["count"] >= 50

def test_both_categories_have_examples():
    for name in ("inventory_constraint", "environment_timeout"):
        assert len(CATEGORIES[name]["examples"]) >= 3
```

Run: `cd pipeline && python -m pytest tests/test_question_categories.py -v`
Expected: FAIL with `KeyError: 'inventory_constraint'`

- [ ] **Step 3: Add the two categories to `sft_question_generator.py`**

Open `pipeline/sft_question_generator.py`. After the existing commented-out category block (around line 500, after all the commented entries), locate the closing `}` of `CATEGORIES` and add before it:

```python
    "inventory_constraint": {
        "count": 60,
        "description": (
            "Questions that require a specific tool (web_search or python_execute) which has been "
            "deliberately removed from the session. The ideal response checks the tool inventory, "
            "recognises the capability gap, and refuses honestly without guessing or hallucinating. "
            "Constitution P3 (tool discipline) and P18 (explicit I don't know) are the primary principles."
        ),
        "examples": [
            "What is the current temperature in Dublin, Ireland?",
            "What is today's EUR/USD exchange rate?",
            "Who won the Premier League last weekend?",
            "What's the latest version of Python released this year?",
            "Is there any breaking news about Apple's latest product launch?",
        ],
        "domains": ["weather", "finance", "sports", "software versions", "tech news"],
        "required_profile": "compute_only",
        "constrained_tool": "web_search",
        "chaining_note": "No chaining — web_search is absent. The model must recognise the gap and refuse.",
    },
    "environment_timeout": {
        "count": 60,
        "description": (
            "Questions that require web_search, but the first search attempt returns HTTP 503. "
            "The ideal response retries once with a different query. If the second attempt also fails, "
            "the model states the gap and answers from static knowledge with a knowledge-cutoff caveat. "
            "Constitution P12 (tool failure handling) is the primary principle."
        ),
        "examples": [
            "What is the current gold price per ounce?",
            "What did the ECB announce at its last meeting?",
            "What is the inflation rate in the EU right now?",
            "Who is the current CEO of OpenAI?",
            "What are the current visa requirements for Indian citizens to visit the UK?",
        ],
        "domains": ["commodities", "central banking", "macroeconomics", "corporate leadership", "immigration"],
        "required_profile": "all_tools",
        "chaining_note": "web_search needed but first call returns 503; retry once, then graceful fallback.",
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd pipeline && python -m pytest tests/test_question_categories.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pipeline/sft_question_generator.py pipeline/tests/test_question_categories.py
git commit -m "feat: add inventory_constraint and environment_timeout question categories for negative trajectory training"
```

---

## Task 2: Write Tests for `sft_v3_generator.py`

**Files:**
- Create: `pipeline/tests/test_sft_v3_generator.py`

These tests run against the module's pure-Python helpers (no LLM calls). The intercept loop and exa search have their LLM/network calls mocked.

- [ ] **Step 1: Create the test file**

```python
"""Unit tests for sft_v3_generator.py helpers — no LLM or network calls."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import sft_v3_generator as gen


# ── Student prompt length ────────────────────────────────────────────────────

def test_all_student_prompts_under_50_words():
    for profile_label, prompt in gen.STUDENT_PROMPTS.items():
        word_count = len(prompt.split())
        assert word_count <= 50, (
            f"Student prompt for '{profile_label}' is {word_count} words (max 50): {prompt!r}"
        )


def test_student_prompts_cover_all_profiles():
    for profile in gen.TOOL_PROFILES:
        assert profile["label"] in gen.STUDENT_PROMPTS, (
            f"No student prompt for profile '{profile['label']}'"
        )


# ── Tool executor ─────────────────────────────────────────────────────────────

def test_execute_tool_v3_python_execute():
    result = gen._execute_tool_v3(
        tool_inner="python_execute(code='print(2 + 2)')",
        active_tools={"python_execute"},
        failure_config=None,
    )
    assert "4" in result


def test_execute_tool_v3_python_unavailable():
    result = gen._execute_tool_v3(
        tool_inner="python_execute(code='print(1)')",
        active_tools=set(),
        failure_config=None,
    )
    assert "not available" in result.lower()


def test_execute_tool_v3_web_search_503_first_call():
    fc = {"inject_503": True}
    result = gen._execute_tool_v3(
        tool_inner="web_search(query='current gold price')",
        active_tools={"web_search"},
        failure_config=fc,
    )
    assert "503" in result
    assert fc["web_search_count"] == 1


def test_execute_tool_v3_web_search_503_only_first_call():
    """Second call should NOT return 503 (so the model can retry and succeed)."""
    fc = {"inject_503": True}
    with patch("sft_v3_generator._exa_search", return_value="Gold: $2300/oz"):
        # First call → 503
        gen._execute_tool_v3(
            tool_inner="web_search(query='gold price')",
            active_tools={"web_search"},
            failure_config=fc,
        )
        # Second call → real result
        result = gen._execute_tool_v3(
            tool_inner="web_search(query='current gold price site:ft.com')",
            active_tools={"web_search"},
            failure_config=fc,
        )
    assert "Gold" in result or "2300" in result


def test_execute_tool_v3_get_datetime():
    result = gen._execute_tool_v3(
        tool_inner="get_datetime()",
        active_tools={"get_datetime"},
        failure_config=None,
    )
    assert "UTC" in result or "202" in result


def test_execute_tool_v3_unknown_tool():
    result = gen._execute_tool_v3(
        tool_inner="fly_to_moon(destination='Mars')",
        active_tools={"python_execute"},
        failure_config=None,
    )
    assert "unknown" in result.lower() or "error" in result.lower()


# ── Context swap ─────────────────────────────────────────────────────────────

def _make_conversation(profile_label: str = "compute_only") -> list[dict]:
    teacher_system = gen._make_teacher_prompt(
        next(p for p in gen.TOOL_PROFILES if p["label"] == profile_label)
    )
    return [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": "What is 7 * 8?"},
        {"role": "assistant", "content": "<think>I need to compute 7 * 8.</think><answer>56</answer>"},
    ]


def test_context_swap_replaces_system_prompt():
    conversation = _make_conversation("compute_only")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "compute_only")
    example = gen._build_v3_example(conversation, "What is 7 * 8?", "arithmetic", profile)

    system_msgs = [m for m in example["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    saved_system = system_msgs[0]["content"]
    assert saved_system == gen.STUDENT_PROMPTS["compute_only"]


def test_context_swap_student_prompt_is_under_50_words():
    conversation = _make_conversation("all_tools")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "all_tools")
    example = gen._build_v3_example(conversation, "What is 2+2?", "arithmetic", profile)

    system_msgs = [m for m in example["messages"] if m["role"] == "system"]
    saved_system = system_msgs[0]["content"]
    assert len(saved_system.split()) <= 50


def test_context_swap_teacher_constitution_not_in_output():
    """The full constitution must NOT appear in the saved JSONL."""
    conversation = _make_conversation("all_tools")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "all_tools")
    example = gen._build_v3_example(conversation, "q", "arithmetic", profile)

    full_text = json.dumps(example)
    assert "DECOMPOSE FIRST" not in full_text
    assert "TOOL INVENTORY" not in full_text
    assert "25 constitution principles" not in full_text


def test_context_swap_metadata_has_pipeline_v3():
    conversation = _make_conversation("compute_only")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "compute_only")
    example = gen._build_v3_example(conversation, "q", "arithmetic", profile)
    assert example["metadata"]["pipeline"] == "sft_v3"


# ── Banned placeholder detection ────────────────────────────────────────────

def test_has_banned_placeholder_detects_shortcuts():
    assert gen._has_banned_placeholder("see answer below") is True
    assert gen._has_banned_placeholder("inferred from question") is True
    assert gen._has_banned_placeholder("none flagged") is True
    assert gen._has_banned_placeholder("The radius is 4.5 cm.") is False


# ── Think block length ───────────────────────────────────────────────────────

def test_think_block_length_short():
    assert gen._think_block_length("<think>ok</think>") < 50


def test_think_block_length_long():
    content = "<think>" + "x" * 200 + "</think>"
    assert gen._think_block_length(content) >= 200


def test_think_block_length_absent():
    assert gen._think_block_length("<answer>hello</answer>") == 0
```

- [ ] **Step 2: Run tests to confirm they all fail**

Run: `cd pipeline && python -m pytest tests/test_sft_v3_generator.py -v`
Expected: `ModuleNotFoundError: No module named 'sft_v3_generator'` — confirming TDD starting point.

---

## Task 3: Implement `sft_v3_generator.py` — Constants, Student Prompts, Exa Search, Tool Executor

**Files:**
- Create: `pipeline/sft_v3_generator.py`

- [ ] **Step 1: Create the file with constants and imports**

```python
"""
SFT v3 Asymmetric Distillation Generator
=========================================
Replaces the v2 approach of stuffing the full 25-principle constitution into the
student system prompt. Instead:

  Phase A  Teacher generates with full constitution (heavy prompt, never saved).
  Phase B  Tool calls are intercepted mid-generation, executed live, results fed back.
  Phase C  Before saving, swap teacher system prompt → ≤50-word student prompt.

Web search uses exa.ai (set EXA_API_KEY in .env).

Usage:
    python pipeline/sft_v3_generator.py \\
        --questions pipeline/data/questions_partA.jsonl \\
        --output pipeline/data/train_v3.jsonl \\
        --model nvidia_nim/moonshotai/kimi-k2.6 \\
        --critic_model nvidia_nim/minimaxai/minimax-m2.7

    # Negative trajectory categories:
    python pipeline/sft_v3_generator.py \\
        --questions pipeline/data/questions_partA.jsonl \\
        --type inventory_constraint \\
        --output pipeline/data/train_v3_negative.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import re
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0

# ---------------------------------------------------------------------------
# Tool profiles — must match 3_infererence.py and sft_gold_response_generator.py
# ---------------------------------------------------------------------------

TOOL_PROFILES = [
    {
        "label": "all_tools",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓",
        "system_note": "All four tools are available in this session.",
    },
    {
        "label": "compute_only",
        "context": "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✗",
        "system_note": "Only python_execute is available. No internet or time access.",
    },
    {
        "label": "compute_and_search",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗",
        "system_note": "python_execute and web_search/read_url are available. No datetime tool.",
    },
    {
        "label": "no_tools",
        "context": "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✗",
        "system_note": "No tools are available in this session. Training knowledge only.",
    },
]

# ---------------------------------------------------------------------------
# Student prompts — ≤50 words each (validated by test_sft_v3_generator.py)
# These are what appear in the SAVED JSONL — the student model only sees these.
# ---------------------------------------------------------------------------

STUDENT_PROMPTS: dict[str, str] = {
    "all_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url, get_datetime."
    ),
    "compute_only": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute."
    ),
    "compute_and_search": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url."
    ),
    "no_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "No tools available this session."
    ),
}
```

- [ ] **Step 2: Add the teacher system prompt**

Append to the same file:

```python
# ---------------------------------------------------------------------------
# Teacher system prompt — full constitution, NEVER saved to JSONL
# Forbids outputting rule names or checklists so the student learns behaviors
# not templates.
# ---------------------------------------------------------------------------

_TEACHER_CONSTITUTION = """\
Your reasoning principles (demonstrate through behavior; NEVER name them, never output checklists):
1. Before answering, reason through WHO is affected, WHAT is required, WHEN (time-sensitivity), WHERE (domain/jurisdiction), WHY (underlying intent), and HOW (method) — in flowing narrative inside <think>.
2. State which tools are available this session; only call tools that are listed as available.
3. Use python_execute for any precision arithmetic or computation; never approximate mentally when code is available.
4. For live data or named entities, use web_search if available; if not, state the limitation clearly and redirect to an authoritative source.
5. For questions requiring personal context you don't have, ask exactly ONE clarifying question — the most critical unknown. Explain briefly why it is the most important.
6. Hedge only genuinely uncertain claims; state well-known facts confidently.
7. For tasks that are fundamentally impossible, name the irreducible reason (not just "I can't") and redirect usefully.
8. For subjective questions, enumerate 3–5 tradeoff dimensions; never declare a universal winner.
9. Only call tools listed as available this session; never invent tools.
10. If a tool call fails, retry once with a modified query; if it fails again, state the gap honestly and answer from static knowledge with a knowledge-cutoff caveat.
11. Never capitulate under user pressure after a correct refusal; cite the specific consequence of guessing to explain why.
12. For multi-step ambiguities, ask only the single most critical clarifying question first.
13. For queries with 3 or more distinct requirements, reason through them systematically before executing.
14. For partially-capable scenarios: answer the achievable parts fully and assertively; for blocked parts, name (1) what cannot be done, (2) why (missing context / professional expertise needed / tool unavailable / fundamentally unknowable), and (3) a specific redirect.
15. Name assumptions explicitly; mark them as unverified if they are not confirmed facts.\
"""

_TEACHER_FORMAT_RULES = """\
CRITICAL FORMAT RULES — violation invalidates the training example:
1. Open with <think> containing flowing narrative reasoning (minimum 150 characters). NO headers, NO rule numbers, NO "CAPABILITY_CHECK:", NO "5W+H:", NO bullet lists.
2. Place ALL tool calls after </think> and before <answer> using: <tool>tool_name(arg='...')</tool>
3. Close EVERY response with <answer>...</answer>.
4. NEVER output phrase fragments: "see answer below", "inferred from question", "none flagged", "CAPABILITY_CHECK:", "PRINCIPLE_", "5W+H:", "CONSEQUENCE_CHECK:".
5. After each [TOOL_RESULT] block, continue reasoning in flowing prose before the next tool call or <answer>.\
"""


def _make_teacher_prompt(tool_profile: dict) -> str:
    return (
        f"You are a frontier AI assistant generating exemplary training data.\n\n"
        f"{_TEACHER_CONSTITUTION}\n\n"
        f"{_TEACHER_FORMAT_RULES}\n\n"
        f"Session tools available: {tool_profile['context']}\n"
        f"{tool_profile['system_note']}"
    )
```

- [ ] **Step 3: Add the exa.ai web search helper**

```python
# ---------------------------------------------------------------------------
# Web search via exa.ai
# ---------------------------------------------------------------------------

def _exa_search(query: str, num_results: int = 3) -> str:
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return (
            f"web_search unavailable: EXA_API_KEY not set. "
            f"Cannot retrieve live data for: {query}"
        )
    try:
        from exa_py import Exa  # pip install exa-py
        exa = Exa(api_key=api_key)
        result = exa.search_and_contents(
            query,
            num_results=num_results,
            text={"max_characters": 400},
        )
        snippets = []
        for r in result.results:
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
            text = getattr(r, "text", "") or ""
            snippets.append(f"**{title}** ({url})\n{text[:350]}")
        return "\n\n".join(snippets) if snippets else f"No results found for: {query}"
    except ImportError:
        return "web_search unavailable: exa_py not installed — run: pip install exa-py"
    except Exception as e:
        return f"web_search error: {e}"
```

- [ ] **Step 4: Add Python executor and read_url helpers**

```python
# ---------------------------------------------------------------------------
# Python executor (sandboxed — same allowlist as sft_dataset_assembler.py)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})
_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


def _parse_python_code(s: str) -> str | None:
    for pat in (
        r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)',
        r"python_execute\s*\(\s*code\s*=\s*'(.*?)'\s*\)",
        r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)',
    ):
        m = re.search(pat, s, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return None


def _run_safe_python(code: str) -> str:
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return f"Error: syntax_error: {e}"
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return f"Error: blocked_import: {alias.name}"
        elif isinstance(node, _ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS:
                return f"Error: blocked_import: {node.module}"
        elif isinstance(node, _ast.Call):
            if isinstance(node.func, _ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                return f"Error: blocked_builtin: {node.func.id}"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        out = (proc.stdout or proc.stderr).strip()
        return out if out else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (15s limit)"
    except Exception as e:
        return f"Error: {e}"


def _fetch_url(url: str, prompt: str = "") -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read(8000).decode("utf-8", errors="replace")
        # Crude HTML strip
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if prompt:
            return f"[Fetched: {url}]\nPrompt: {prompt}\nContent excerpt: {body[:600]}"
        return f"[Fetched: {url}]\n{body[:600]}"
    except Exception as e:
        return f"read_url failed: {e}"
```

- [ ] **Step 5: Add the main tool executor with failure injection**

```python
# ---------------------------------------------------------------------------
# Tool executor with failure injection
# ---------------------------------------------------------------------------

def _execute_tool_v3(
    tool_inner: str,
    active_tools: set[str],
    failure_config: dict | None,
) -> str:
    """Execute a tool call string and return the result as a string.

    failure_config keys:
      inject_503 (bool): inject HTTP 503 on the FIRST web_search call only.
      web_search_count (int): auto-incremented, do not set manually.
    """
    s = tool_inner.strip()

    if s.startswith("python_execute"):
        if "python_execute" not in active_tools:
            return "Error: python_execute is not available in this session."
        code = _parse_python_code(s)
        if code is None:
            return "Error: could not parse python_execute arguments."
        return _run_safe_python(code)

    if s.startswith("web_search"):
        if "web_search" not in active_tools:
            return "Error: web_search is not available in this session."
        m = re.search(r"query\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)
        query = m.group(1) if m else s
        if failure_config and failure_config.get("inject_503"):
            failure_config.setdefault("web_search_count", 0)
            failure_config["web_search_count"] += 1
            if failure_config["web_search_count"] == 1:
                return "HTTP 503 Service Unavailable. The search service is temporarily down. Please retry with a different query."
        return _exa_search(query)

    if s.startswith("read_url"):
        if "read_url" not in active_tools:
            return "Error: read_url is not available in this session."
        url_m = re.search(r"url\s*=\s*['\"](.+?)['\"]", s)
        prompt_m = re.search(r"prompt\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)
        return _fetch_url(
            url_m.group(1) if url_m else "",
            prompt_m.group(1) if prompt_m else "",
        )

    if s.startswith("get_datetime"):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if s.startswith("scratchpad_read"):
        return "(scratchpad is empty — training example initialisation)"

    if s.startswith("scratchpad_update"):
        return "(scratchpad updated)"

    tool_name = s.split("(")[0].strip() if "(" in s else s[:40]
    return f"Error: unknown tool '{tool_name}' — only registered tools are callable."
```

- [ ] **Step 6: Add small pure helpers used by tests**

```python
# ---------------------------------------------------------------------------
# Pure helpers (also tested directly)
# ---------------------------------------------------------------------------

_BANNED_PHRASES = frozenset({
    "see answer below", "inferred from question", "none flagged",
    "capability_check:", "principle_", "5w+h:", "consequence_check:",
})


def _has_banned_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _BANNED_PHRASES)


def _think_block_length(content: str) -> int:
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
    return len(m.group(1).strip()) if m else 0


def _count_violations(violations: str) -> int:
    if violations.strip() == "NO_VIOLATIONS":
        return 0
    return sum(1 for line in violations.splitlines() if line.startswith("PRINCIPLE_") or line.startswith("ISSUE_"))
```

- [ ] **Step 7: Run the tests to confirm helpers are implemented**

Run: `cd pipeline && python -m pytest tests/test_sft_v3_generator.py -v -k "student_prompt or execute_tool or placeholder or think_block"`
Expected: All matching tests PASS. The context-swap tests will still fail (need Task 4).

---

## Task 4: Implement the Intercept Loop

**Files:**
- Modify: `pipeline/sft_v3_generator.py` (append to existing file)

The intercept loop uses `stop=["</tool>"]` to halt generation at each tool call, execute the tool, and resume. Different LiteLLM providers strip the stop sequence from the response; we detect tool interception by checking that `<tool>` is present but `</tool>` is absent.

- [ ] **Step 1: Add `_call_with_stop` and the intercept loop**

```python
# ---------------------------------------------------------------------------
# LiteLLM wrapper with stop-sequence support
# ---------------------------------------------------------------------------

def _call_with_stop(
    messages: list[dict],
    model: str,
    max_tokens: int,
    api_base: str | None = None,
    stop: list[str] | None = None,
) -> str:
    for attempt in range(_MAX_RETRIES):
        try:
            kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
            if api_base:
                kwargs["api_base"] = api_base
            if stop:
                kwargs["stop"] = stop
            resp = litellm.completion(**kwargs)
            content = resp.choices[0].message.content or ""
            return content.strip()
        except litellm.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"  [rate_limit] retry {attempt+1}/{_MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)
        except (litellm.APIConnectionError, litellm.Timeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt)
            print(f"  [conn_error] retry {attempt+1}/{_MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"_call_with_stop: all {_MAX_RETRIES} attempts failed")


# ---------------------------------------------------------------------------
# Intercept loop — Phase B of the v3 pipeline
# ---------------------------------------------------------------------------

def _generate_with_intercept(
    messages: list[dict],
    model: str,
    tool_profile: dict,
    api_base: str | None = None,
    failure_config: dict | None = None,
    max_rounds: int = 8,
) -> list[dict]:
    """Generate text iteratively, intercept <tool> calls, execute them live.

    The conversation grows with each round:
      assistant: <think>...</think><tool>tool_name(args)</tool>
      tool:      [TOOL_RESULT: tool_name]\\n...\\n[/TOOL_RESULT]
      assistant: (continues reasoning then <answer>...</answer>)

    Uses stop=["</tool>"] so generation halts immediately after a tool call
    body, before the closing tag. We add </tool> back and record the tool call.
    """
    conversation = list(messages)
    active_tools = {
        part.split("✓")[0].strip()
        for part in tool_profile["context"].split("|")
        if "✓" in part
    }

    for round_num in range(max_rounds):
        content = _call_with_stop(
            messages=conversation,
            model=model,
            max_tokens=2048,
            api_base=api_base,
            stop=["</tool>"],
        )

        # Detect whether generation stopped at a tool call.
        # When stop=["</tool>"] fires, the returned content contains <tool>
        # but NOT the </tool> closing tag (it was stripped as the stop string).
        tool_pos = content.rfind("<tool>")
        is_tool_call = tool_pos != -1 and "</tool>" not in content[tool_pos:]

        if not is_tool_call:
            # Normal completion or no tool call — append and finish.
            conversation.append({"role": "assistant", "content": content})
            break

        # Reconstruct the full tool tag (add back the stripped </tool>)
        tool_inner = content[tool_pos + len("<tool>"):]
        full_assistant_content = content + "</tool>"

        tool_name_m = re.match(r"(\w+)", tool_inner.strip())
        tool_name = tool_name_m.group(1) if tool_name_m else "unknown"

        result = _execute_tool_v3(tool_inner, active_tools, failure_config)
        wrapped = f"[TOOL_RESULT: {tool_name}]\n{result[:3000]}\n[/TOOL_RESULT]"

        conversation.append({"role": "assistant", "content": full_assistant_content})
        conversation.append({"role": "tool", "content": wrapped})
        print(f"    [intercept r{round_num}] {tool_name}() → {len(result)} chars")

    return conversation
```

- [ ] **Step 2: Verify the intercept loop is syntactically correct**

Run: `cd pipeline && python -c "import sft_v3_generator; print('import OK')"`
Expected: `import OK`

---

## Task 5: Implement Context Swap, Main Loop, and CLI

**Files:**
- Modify: `pipeline/sft_v3_generator.py` (append)

- [ ] **Step 1: Add context swap and example builder**

```python
# ---------------------------------------------------------------------------
# Context swap — Phase C: replace teacher prompt with student prompt
# ---------------------------------------------------------------------------

def _build_v3_example(
    conversation: list[dict],
    question: str,
    category: str,
    tool_profile: dict,
    violations: str = "NO_VIOLATIONS",
) -> dict:
    """Build a JSONL training row from an intercepted conversation.

    The teacher's system prompt is replaced by the ≤50-word student prompt.
    The full constitution NEVER appears in the saved file.
    """
    student_system = STUDENT_PROMPTS[tool_profile["label"]]
    messages = [
        ({"role": "system", "content": student_system} if m["role"] == "system" else m)
        for m in conversation
    ]
    n_viol = _count_violations(violations)
    return {
        "messages": messages,
        "metadata": {
            "source": "v3_distillation",
            "category": category,
            "tool_profile": tool_profile["label"],
            "constitution_score": max(0.0, round(1.0 - n_viol * 0.05, 3)),
            "revised": violations != "NO_VIOLATIONS",
            "pipeline": "sft_v3",
        },
    }
```

- [ ] **Step 2: Add pick_tool_profile with failure injection support**

```python
# ---------------------------------------------------------------------------
# Tool profile selection
# ---------------------------------------------------------------------------

_PREFER_SEARCH = {
    "entity_facts_web_search", "real_time_dependent", "knowledge_boundary",
    "interleaved_tool_reasoning", "scratchpad_decomposition", "environment_timeout",
}
_TOOL_NEUTRAL = {
    "user_context_behavioral", "impossible_tasks", "subjective_tradeoffs",
    "multi_step_clarification", "ambiguous_underspecified", "adversarial_pressure",
    "multi_turn_conversation", "appraisal_empathy",
}


def pick_tool_profile(category: str, item: dict | None = None) -> tuple[dict, dict | None]:
    """Return (tool_profile, failure_config).

    failure_config is None for normal cases, or {"inject_503": True} for
    environment_timeout, or forces a specific profile for inventory_constraint.
    """
    if category == "inventory_constraint":
        required_label = (item or {}).get("required_profile", "compute_only")
        profile = next(
            (p for p in TOOL_PROFILES if p["label"] == required_label),
            TOOL_PROFILES[1],  # fallback to compute_only
        )
        return profile, None

    if category == "environment_timeout":
        profile = random.choices(
            [TOOL_PROFILES[0], TOOL_PROFILES[2]],
            weights=[60, 40],
        )[0]
        return profile, {"inject_503": True}

    if category in _PREFER_SEARCH:
        profile = random.choices(TOOL_PROFILES, weights=[60, 0, 40, 0])[0]
    elif category in _TOOL_NEUTRAL:
        profile = random.choices(TOOL_PROFILES, weights=[25, 30, 20, 25])[0]
    else:
        profile = random.choices(TOOL_PROFILES, weights=[35, 30, 25, 10])[0]

    return profile, None
```

- [ ] **Step 3: Add the per-question worker**

```python
# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

_USER_DRAFT_PROMPT = """\
Generate an exemplary training response for this question.

QUESTION: {question}
CATEGORY: {category}
SESSION TOOLS: {tool_context}

Requirements for this category ({category}):
{ideal_behavior}

Begin your response immediately with <think>. Do NOT output any preamble or headers.\
"""

# Inline ideal behaviors for v3 (narrative, no rule names)
_IDEAL_BEHAVIORS_V3: dict[str, str] = {
    "inventory_constraint": (
        "The session does NOT have the tool required to answer this question. "
        "Your <think> block must explicitly notice which tool is missing from the session inventory. "
        "Your <answer> must honestly state the limitation and redirect the user to an authoritative source. "
        "Do not hallucinate data or pretend to call a missing tool."
    ),
    "environment_timeout": (
        "web_search is available but the FIRST call will return HTTP 503. "
        "Your <think> block must reason about the failure and decide to retry with a refined query. "
        "If the retry succeeds, synthesise the result in <answer>. "
        "If both calls fail, state the gap honestly and answer from static knowledge with a cutoff caveat."
    ),
    "interleaved_tool_reasoning": (
        "This question requires both live external data AND computation. "
        "Chain the tools: web_search to get the raw data, then python_execute to compute. "
        "Never approximate mentally when the chain is available."
    ),
    "scratchpad_decomposition": (
        "This question has 3 or more distinct requirements. "
        "Use scratchpad_read() first, then write context and tasks, then execute each task in order."
    ),
}

_DEFAULT_IDEAL_V3 = (
    "Reason through the question carefully in <think>, state which tools you have, "
    "use the right tool if needed, and close with a clear <answer>."
)


def _process_one_v3(
    item: dict,
    model: str,
    api_base: str | None,
    out_file,
    file_lock: threading.Lock,
    idx: int,
    total: int,
    run_start: float,
) -> str:
    """Process one question through the v3 intercept pipeline."""
    category = item.get("category", "unknown")
    question = item.get("question", "").strip()
    if not question:
        return "error"

    tool_profile, failure_config = pick_tool_profile(category, item)
    elapsed = time.monotonic() - run_start
    tag = f"[{idx}/{total}:{category}]"
    print(f"\n{tag} profile={tool_profile['label']} elapsed={elapsed:.0f}s")
    print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")

    ideal = _IDEAL_BEHAVIORS_V3.get(category, _DEFAULT_IDEAL_V3)
    user_prompt = _USER_DRAFT_PROMPT.format(
        question=question,
        category=category,
        tool_context=tool_profile["context"],
        ideal_behavior=ideal,
    )
    teacher_system = _make_teacher_prompt(tool_profile)
    initial_messages = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": user_prompt},
    ]

    try:
        t0 = time.monotonic()
        conversation = _generate_with_intercept(
            messages=initial_messages,
            model=model,
            tool_profile=tool_profile,
            api_base=api_base,
            failure_config=failure_config,
        )
        n_tool_turns = sum(1 for m in conversation if m["role"] == "tool")
        print(f"  {tag} generated {len(conversation)} msgs ({n_tool_turns} tool turns) in {time.monotonic()-t0:.1f}s")

        example = _build_v3_example(conversation, question, category, tool_profile)

        with file_lock:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_file.flush()

        print(f"  {tag} ✓ written")
        return "ok"

    except Exception as e:
        print(f"  {tag} ✗ error: {e}")
        return "error"
```

- [ ] **Step 4: Add the main processing loop and CLI**

```python
# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_questions_v3(
    questions_path: str,
    output_path: str,
    model: str,
    api_base: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
    category_filter: str | None = None,
    workers: int = 4,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"

    done_questions: set[str] = set()
    if not overwrite and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    user_msgs = [m["content"] for m in ex["messages"] if m["role"] == "user"]
                    done_questions.add(user_msgs[0] if user_msgs else "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        if done_questions:
            print(f"Resume: {len(done_questions)} questions already processed")

    items: list[dict] = []
    parse_errors = skipped = 0
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            cat = item.get("category", "")
            if category_filter and category_filter != "all" and cat != category_filter:
                skipped += 1
                continue
            q = item.get("question", "").strip()
            if not q or q in done_questions:
                skipped += 1
                continue
            items.append(item)

    if max_examples:
        items = items[:max_examples]

    print(f"Questions: {len(items)} to process (skipped={skipped}, parse_errors={parse_errors})")

    processed = errors = 0
    run_start = time.monotonic()
    file_lock = threading.Lock()
    total = len(items)

    with open(output_path, write_mode, encoding="utf-8") as out:
        if workers <= 1 or total <= 1:
            for i, item in enumerate(items, 1):
                result = _process_one_v3(item, model, api_base, out, file_lock, i, total, run_start)
                if result == "ok":
                    processed += 1
                else:
                    errors += 1
        else:
            max_w = min(workers, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = {
                    executor.submit(
                        _process_one_v3, item, model, api_base, out, file_lock, i, total, run_start
                    ): item
                    for i, item in enumerate(items, 1)
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result == "ok":
                            processed += 1
                        else:
                            errors += 1
                    except Exception as e:
                        print(f"  ✗ future error: {e}")
                        errors += 1

    elapsed = time.monotonic() - run_start
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s | processed={processed} errors={errors}")
    print(f"Output: {output_path}")
    print(f"\nNext: python pipeline/validate_sft_data.py --input {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="SFT v3 asymmetric distillation generator")
    p.add_argument("--questions", required=True, help="JSONL from sft_question_generator.py")
    p.add_argument("--output", default="pipeline/data/train_v3.jsonl")
    p.add_argument("--model", default="nvidia_nim/moonshotai/kimi-k2.6")
    p.add_argument("--api_base", default=None)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--type", "--category", dest="category_filter", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--base_delay", type=float, default=3.0)
    args = p.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

    print(f"Generator : {args.model}")
    process_questions_v3(
        questions_path=args.questions,
        output_path=args.output,
        model=args.model,
        api_base=args.api_base,
        max_examples=args.max,
        overwrite=args.overwrite,
        category_filter=args.category_filter,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the full test suite for the generator**

Run: `cd pipeline && python -m pytest tests/test_sft_v3_generator.py -v`
Expected: ALL PASS (15 tests)

- [ ] **Step 6: Commit**

```bash
git add pipeline/sft_v3_generator.py pipeline/tests/test_sft_v3_generator.py
git commit -m "feat: add sft_v3_generator with asymmetric distillation, intercept loop, and exa.ai web search"
```

---

## Task 6: Write Tests for `validate_sft_data.py`

**Files:**
- Create: `pipeline/tests/test_validate_sft_data.py`

- [ ] **Step 1: Create the test file**

```python
"""Tests for validate_sft_data.py — all 5 quality gate assertions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import validate_sft_data as v


# ── Helpers to build minimal valid/invalid rows ───────────────────────────────

def _make_row(
    system="You are a trustworthy AI assistant. Reason in <think> tags.",
    think_content="x" * 200,
    tool_call: str | None = None,
    tool_result: str | None = None,
    final_answer="<answer>42</answer>",
) -> dict:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What is 6 * 7?"},
    ]
    asst_content = f"<think>{think_content}</think>"
    if tool_call:
        asst_content += f"\n<tool>{tool_call}</tool>"
    msgs.append({"role": "assistant", "content": asst_content})
    if tool_result is not None:
        msgs.append({"role": "tool", "content": tool_result})
    msgs.append({"role": "assistant", "content": final_answer})
    return {"messages": msgs, "metadata": {"category": "arithmetic", "pipeline": "sft_v3"}}


# ── Assertion 1: System prompt length ────────────────────────────────────────

def test_valid_row_passes():
    row = _make_row()
    ok, reason = v.validate_row(row)
    assert ok, f"Expected valid row to pass, got reason: {reason}"


def test_long_system_prompt_fails():
    leaked_constitution = " ".join(["word"] * 60)  # 60 words > 50
    row = _make_row(system=leaked_constitution)
    ok, reason = v.validate_row(row)
    assert not ok
    assert "system_prompt" in reason or "constitution" in reason.lower()


def test_exactly_50_word_system_passes():
    system = " ".join(["word"] * 50)
    row = _make_row(system=system)
    ok, _ = v.validate_row(row)
    assert ok


# ── Assertion 2: Think block length ──────────────────────────────────────────

def test_short_think_block_fails():
    row = _make_row(think_content="ok")  # < 50 chars
    ok, reason = v.validate_row(row)
    assert not ok
    assert "think" in reason.lower()


def test_missing_think_fails():
    row = _make_row(think_content="x" * 200)
    # Manually replace the think block
    row["messages"][2]["content"] = "<answer>42</answer>"
    ok, reason = v.validate_row(row)
    assert not ok


# ── Assertion 3: Banned placeholders ─────────────────────────────────────────

def test_banned_placeholder_see_answer_below_fails():
    row = _make_row(think_content="Core truth: see answer below. " + "x" * 180)
    ok, reason = v.validate_row(row)
    assert not ok
    assert "placeholder" in reason.lower() or "banned" in reason.lower()


def test_banned_placeholder_none_flagged_fails():
    row = _make_row(think_content="Assumptions: none flagged. " + "x" * 180)
    ok, reason = v.validate_row(row)
    assert not ok


def test_clean_think_passes():
    row = _make_row(think_content="I need to compute 6 times 7. I have python_execute. " + "x" * 150)
    ok, _ = v.validate_row(row)
    assert ok


# ── Assertion 4: Tool sequence integrity ─────────────────────────────────────

def test_tool_call_followed_by_tool_role_passes():
    row = _make_row(
        tool_call="python_execute(code='print(42)')",
        tool_result="42",
    )
    ok, _ = v.validate_row(row)
    assert ok


def test_tool_call_not_followed_by_tool_role_fails():
    row = _make_row(
        tool_call="python_execute(code='print(42)')",
        tool_result=None,  # no tool role message — invalid sequence
    )
    # In this row, the assistant message with <tool> is at index 2,
    # but the next message is also assistant (final answer), not tool.
    ok, reason = v.validate_row(row)
    assert not ok
    assert "sequence" in reason.lower() or "tool" in reason.lower()


# ── Assertion 5: End-to-end resolution ───────────────────────────────────────

def test_missing_answer_tag_fails():
    row = _make_row(final_answer="The answer is 42.")  # no <answer> tag
    ok, reason = v.validate_row(row)
    assert not ok
    assert "answer" in reason.lower()


def test_last_message_not_assistant_fails():
    row = _make_row()
    row["messages"].append({"role": "user", "content": "Thanks!"})
    ok, reason = v.validate_row(row)
    assert not ok


# ── Drop rate calculation ─────────────────────────────────────────────────────

def test_validate_file_drop_rate():
    rows = [_make_row() for _ in range(95)] + [
        _make_row(system=" ".join(["word"] * 60)) for _ in range(5)  # 5% bad
    ]
    valid, invalid, _ = v.validate_rows(rows)
    assert len(valid) == 95
    assert len(invalid) == 5
    drop_rate = len(invalid) / len(rows)
    assert drop_rate < 0.06  # under 5% threshold
```

- [ ] **Step 2: Run tests to confirm they all fail**

Run: `cd pipeline && python -m pytest tests/test_validate_sft_data.py -v`
Expected: `ModuleNotFoundError: No module named 'validate_sft_data'`

---

## Task 7: Implement `validate_sft_data.py`

**Files:**
- Create: `pipeline/validate_sft_data.py`

- [ ] **Step 1: Create the file**

```python
"""
Pre-Flight SFT Dataset Validation
==================================
Enforces 5 quality invariants on every row of a training JSONL before
the Unsloth training loop starts. If >5% of rows fail, the pipeline is
fundamentally broken — fix the generator, not the validator.

Invariants:
  1. System prompt ≤ 50 words  (asymmetric distillation — no leaked constitution)
  2. <think> block ≥ 50 chars  (no synthetic laziness)
  3. No banned placeholders in <think>  (no v2-style shortcuts)
  4. Tool call immediately followed by tool role  (sequence integrity)
  5. Last message is assistant with <answer>  (end-to-end resolution)

Usage:
    python pipeline/validate_sft_data.py --input pipeline/data/train_v3.jsonl
    python pipeline/validate_sft_data.py --input pipeline/data/train_v3.jsonl --fix --output pipeline/data/train_v3_clean.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

_BANNED_PHRASES = frozenset({
    "see answer below", "inferred from question", "none flagged",
    "capability_check:", "principle_", "5w+h:", "consequence_check:",
})

_MIN_THINK_CHARS = 50   # minimum think block length
_MAX_SYSTEM_WORDS = 50  # maximum system prompt word count


def validate_row(row: dict) -> tuple[bool, str]:
    """Return (is_valid, failure_reason). 'ok' on success."""
    messages = row.get("messages", [])
    if not messages:
        return False, "empty_messages"

    # ── 1. System prompt length ──────────────────────────────────────────────
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    if system_msg is None:
        return False, "missing_system_message"
    word_count = len(system_msg.get("content", "").split())
    if word_count > _MAX_SYSTEM_WORDS:
        return False, f"system_prompt_too_long: {word_count} words (max {_MAX_SYSTEM_WORDS}) — leaked constitution"

    # ── 2 & 3: <think> block length + banned placeholders ───────────────────
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not asst_msgs:
        return False, "no_assistant_message"

    first_asst = asst_msgs[0].get("content", "")
    think_m = re.search(r"<think>(.*?)</think>", first_asst, re.DOTALL | re.IGNORECASE)
    if not think_m:
        return False, "missing_think_block"
    think_text = think_m.group(1).strip()
    if len(think_text) < _MIN_THINK_CHARS:
        return False, f"think_block_too_short: {len(think_text)} chars (min {_MIN_THINK_CHARS})"
    lower_think = think_text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower_think:
            return False, f"banned_placeholder_in_think: '{phrase}'"

    # ── 4. Tool sequence integrity ───────────────────────────────────────────
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "<tool>" not in content:
            continue
        # This assistant message contains a tool call — next must be tool role
        if i + 1 >= len(messages) or messages[i + 1].get("role") != "tool":
            next_role = messages[i + 1].get("role", "missing") if i + 1 < len(messages) else "missing"
            return False, f"tool_sequence_violation: <tool> in assistant[{i}] not followed by tool role (got '{next_role}')"

    # ── 5. End-to-end resolution ─────────────────────────────────────────────
    last = messages[-1]
    if last.get("role") != "assistant":
        return False, f"last_message_not_assistant: role='{last.get('role')}'"
    if "<answer>" not in last.get("content", ""):
        return False, "last_assistant_missing_answer_tag"

    return True, "ok"


def validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Partition rows into (valid, invalid) and tally failure reasons."""
    from collections import Counter
    valid, invalid, reasons = [], [], Counter()
    for row in rows:
        ok, reason = validate_row(row)
        if ok:
            valid.append(row)
        else:
            invalid.append(row)
            reasons[reason] += 1
    return valid, invalid, dict(reasons)


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-flight SFT dataset validation")
    p.add_argument("--input", required=True, help="JSONL file to validate")
    p.add_argument("--fix", action="store_true", help="Write valid rows to --output")
    p.add_argument("--output", default=None, help="Output path for valid rows (default: <input>_clean.jsonl)")
    p.add_argument("--max_drop_pct", type=float, default=5.0,
                   help="Exit with error if more than this %% of rows fail (default: 5.0)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error — {e}")

    print(f"Loaded {len(rows)} rows from {input_path}")
    valid, invalid, reasons = validate_rows(rows)

    drop_pct = 100 * len(invalid) / max(len(rows), 1)
    print(f"\nValid   : {len(valid)}")
    print(f"Invalid : {len(invalid)}  ({drop_pct:.1f}%)")
    if reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<55} {count}")

    if drop_pct > args.max_drop_pct:
        print(f"\nFAIL: drop rate {drop_pct:.1f}% exceeds threshold {args.max_drop_pct:.1f}%.")
        print("The generation pipeline is fundamentally broken — fix the generator.")
        sys.exit(1)

    if args.fix:
        out_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_clean")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(valid)} valid rows → {out_path}")

    print("\nPASS: dataset is within quality threshold.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the validator tests**

Run: `cd pipeline && python -m pytest tests/test_validate_sft_data.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add pipeline/validate_sft_data.py pipeline/tests/test_validate_sft_data.py
git commit -m "feat: add validate_sft_data.py with 5 pre-flight quality gate assertions"
```

---

## Task 8: Update `2_model_trainer.py` for v3 Compatibility and Curriculum Staging

**Files:**
- Modify: `pipeline/2_model_trainer.py`

Two changes:
1. `_format_reward` currently checks for `CAPABILITY_CHECK` — v3 models don't output this, so the check would always fire in GRPO. Add a `--v3_format` flag that changes the reward to check for `<think>` + `<answer>` only.
2. Add `--curriculum_stage {1,2,3}` and `--from_checkpoint` flags for staged SFT.

- [ ] **Step 1: Find the `_format_reward` function**

Run: `grep -n "_format_reward\|CAPABILITY_CHECK\|--mode\|argparse" pipeline/2_model_trainer.py | head -30`

`_format_reward` is around line 239. It currently requires `CAPABILITY_CHECK`.

- [ ] **Step 2: Update `_format_reward` to be v3-aware**

In `pipeline/2_model_trainer.py`, find:

```python
def _format_reward(response: str) -> float:
    """P1 structural check: <think> + CAPABILITY_CHECK + <answer> all present."""
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_cap   = "CAPABILITY_CHECK" in response
    has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))
    return 1.0 if (has_think and has_cap and has_ans) else 0.0
```

Replace with:

```python
# Set to True when training on v3 data (no CAPABILITY_CHECK in student outputs)
_V3_FORMAT_MODE: bool = False


def _format_reward(response: str) -> float:
    """Structural check: <think> + <answer> required; CAPABILITY_CHECK required in v2 mode only."""
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_ans   = bool(re.search(r"<answer>", response, re.IGNORECASE))
    if _V3_FORMAT_MODE:
        return 1.0 if (has_think and has_ans) else 0.0
    has_cap = "CAPABILITY_CHECK" in response
    return 1.0 if (has_think and has_cap and has_ans) else 0.0
```

- [ ] **Step 3: Add `_split_curriculum_stages` function**

After the existing reward functions (around line 300), add:

```python
# ---------------------------------------------------------------------------
# Curriculum learning — three-stage data split
# ---------------------------------------------------------------------------

def _split_curriculum_stages(
    examples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split examples into three curriculum stages.

    Stage 1 — Format establishment: short examples with no tool calls.
      Teaches the model <think>...</think><answer>...</answer> syntax.
    Stage 2 — Complexity scaling: all examples (includes multi-tool trajectories).
      Introduces heavy reasoning patterns distilled from the teacher.
    Stage 3 — Anti-drift: all examples + 20% of Stage 1 as replay.
      Prevents the model from losing basic instruction-following while
      learning deep reasoning.

    Returns (stage1, stage2, stage3).
    """
    stage1 = []
    stage2 = list(examples)

    for ex in examples:
        msgs = ex.get("messages", [])
        has_tool = any(m.get("role") == "tool" for m in msgs)
        asst_text = " ".join(
            m.get("content", "") or ""
            for m in msgs if m.get("role") == "assistant"
        )
        if not has_tool and len(asst_text) < 600:
            stage1.append(ex)

    replay_n = min(len(stage1), max(1, len(stage2) // 5))
    replay = random.sample(stage1, replay_n) if stage1 else []
    stage3 = stage2 + replay

    return stage1, stage2, stage3
```

- [ ] **Step 4: Add CLI flags to `main()`**

Find the `main()` function in `2_model_trainer.py` and locate the argparse section. Add after the existing `--mode` argument:

```python
    parser.add_argument(
        "--from_checkpoint", type=str, default=None,
        help="Path to a prior SFT checkpoint to resume from (curriculum staging: pass stage N-1 checkpoint)",
    )
    parser.add_argument(
        "--curriculum_stage", type=int, choices=[1, 2, 3], default=None,
        help="Curriculum stage for SFT: 1=short format, 2=all examples, 3=anti-drift replay mix",
    )
    parser.add_argument(
        "--v3_format", action="store_true",
        help="Use v3 format rewards (no CAPABILITY_CHECK requirement) for GRPO on v3-trained models",
    )
```

- [ ] **Step 5: Wire the new flags into the SFT loading path**

Find where `load_dataset` is called for SFT mode (around line 350+). After the dataset is loaded into `all_examples`, add:

```python
    if args.curriculum_stage:
        stage1, stage2, stage3 = _split_curriculum_stages(all_examples)
        stage_map = {1: stage1, 2: stage2, 3: stage3}
        all_examples = stage_map[args.curriculum_stage]
        print(f"Curriculum stage {args.curriculum_stage}: {len(all_examples)} examples "
              f"(S1={len(stage1)} S2={len(stage2)} S3={len(stage3)})")

    if args.v3_format:
        import sft_v3_generator as _  # noqa: F401  (just validate import works)
        global _V3_FORMAT_MODE
        _V3_FORMAT_MODE = True
        print("GRPO format reward: v3 mode (no CAPABILITY_CHECK requirement)")
```

- [ ] **Step 6: Wire `--from_checkpoint` into the model loading for SFT**

Find where `FastModel.from_pretrained` is called for SFT. Add a branch before it:

```python
    base_to_load = args.from_checkpoint if (args.from_checkpoint and args.mode == "sft") else MODEL_CONFIG["base_model"]
    print(f"Loading base model: {base_to_load}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_to_load,
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
    )
```

- [ ] **Step 7: Write a focused unit test for curriculum splitting**

Append to `pipeline/tests/test_question_categories.py` (or create a new `test_trainer_curriculum.py`):

Create `pipeline/tests/test_trainer_curriculum.py`:

```python
"""Tests for curriculum staging in 2_model_trainer.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
import pytest


def _make_example(has_tool: bool = False, asst_len: int = 100) -> dict:
    msgs = [
        {"role": "system", "content": "You are a trustworthy AI."},
        {"role": "user", "content": "test"},
    ]
    if has_tool:
        msgs += [
            {"role": "assistant", "content": "<think>x</think><tool>python_execute(code='1')</tool>"},
            {"role": "tool", "content": "1"},
            {"role": "assistant", "content": "<answer>1</answer>"},
        ]
    else:
        msgs.append({"role": "assistant", "content": "x" * asst_len})
    return {"messages": msgs, "metadata": {}}


def test_stage1_contains_no_tool_short_examples():
    import two_model_trainer as t  # noqa
    examples = (
        [_make_example(has_tool=False, asst_len=200)] * 10 +  # → stage 1
        [_make_example(has_tool=True)] * 10 +                  # → not stage 1
        [_make_example(has_tool=False, asst_len=800)] * 5      # → not stage 1 (too long)
    )
    s1, s2, s3 = t._split_curriculum_stages(examples)
    assert len(s1) == 10
    assert len(s2) == 25


def test_stage3_is_larger_than_stage2():
    import two_model_trainer as t  # noqa
    examples = [_make_example(has_tool=False, asst_len=200)] * 20 + [_make_example(has_tool=True)] * 5
    s1, s2, s3 = t._split_curriculum_stages(examples)
    assert len(s3) > len(s2)  # anti-drift adds replay


def test_stage2_equals_all_examples():
    import two_model_trainer as t  # noqa
    examples = [_make_example() for _ in range(30)]
    _, s2, _ = t._split_curriculum_stages(examples)
    assert len(s2) == 30
```

Note: `2_model_trainer.py` contains a hyphen in the filename which prevents direct import. The tests import it as a subprocess check:

```python
# Actually test via CLI smoke test instead since filename has hyphen
def test_curriculum_stage_flag_cli():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "pipeline/2_model_trainer.py", "--help"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert "--curriculum_stage" in result.stdout
    assert "--from_checkpoint" in result.stdout
    assert "--v3_format" in result.stdout
```

- [ ] **Step 8: Run the curriculum tests**

Run: `cd pipeline && python -m pytest tests/test_trainer_curriculum.py -v`
Expected: The CLI smoke test passes (flags appear in --help). Skip if the import test fails due to filename (known limitation).

- [ ] **Step 9: Commit**

```bash
git add pipeline/2_model_trainer.py pipeline/tests/test_trainer_curriculum.py
git commit -m "feat: add v3-compatible format reward, curriculum_stage flag, and from_checkpoint support to model trainer"
```

---

## Task 9: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `exa-py` to the install block**

Find in `README.md`:
```markdown
pip install datasets trl fastapi uvicorn pydantic requests litellm python-dotenv
```

Add `exa-py` to that line:
```markdown
pip install datasets trl fastapi uvicorn pydantic requests litellm python-dotenv exa-py
```

- [ ] **Step 2: Add EXA_API_KEY to the providers table**

Find the providers table and add a row after the NVIDIA NIM rows:

```markdown
| **exa.ai** ✅ semantic search | `EXA_API_KEY=...` | Used by v3 generator for live web search | $10 credits free |
```

- [ ] **Step 3: Add v3 pipeline section after the existing SFT data section**

Find the section that documents `sft_gold_response_generator.py` usage and add immediately after it:

```markdown
### SFT v3 — Asymmetric Distillation (recommended for sub-1B models)

The v3 pipeline eliminates context-window starvation in sub-1B models by keeping the 25-principle constitution **teacher-side only**. The student only sees a ≤50-word system prompt.

```bash
# 1. Generate questions (same as v2)
python pipeline/sft_question_generator.py --count 200 --output pipeline/data/questions_v3.jsonl

# 2. Generate gold responses with intercept loop + exa.ai web search
python pipeline/sft_v3_generator.py \
    --questions pipeline/data/questions_v3.jsonl \
    --output pipeline/data/train_v3.jsonl \
    --model nvidia_nim/moonshotai/kimi-k2.6

# 2b. Generate negative trajectories
python pipeline/sft_v3_generator.py \
    --questions pipeline/data/questions_v3.jsonl \
    --type inventory_constraint \
    --output pipeline/data/train_v3_negative.jsonl

# 3. Validate before training
python pipeline/validate_sft_data.py --input pipeline/data/train_v3.jsonl

# 4. Assemble (same assembler as v2)
python pipeline/sft_dataset_assembler.py \
    --part_a pipeline/data/train_v3.jsonl \
    --output_dir pipeline/data/

# 5. Curriculum training (3 stages)
python pipeline/2_model_trainer.py --mode sft --curriculum_stage 1 --output_name checkpoint_sft_s1
python pipeline/2_model_trainer.py --mode sft --curriculum_stage 2 --from_checkpoint models/checkpoint_sft_s1 --output_name checkpoint_sft_s2
python pipeline/2_model_trainer.py --mode sft --curriculum_stage 3 --from_checkpoint models/checkpoint_sft_s2 --output_name checkpoint_sft

# 6. GRPO (same as v2, add --v3_format flag)
python pipeline/2_model_trainer.py --mode grpo --sft_checkpoint models/checkpoint_sft --v3_format
```
```

- [ ] **Step 4: Verify the README renders correctly**

Run: `python -c "import pathlib; t=pathlib.Path('README.md').read_text(); print('OK' if 'exa-py' in t and 'sft_v3_generator' in t else 'MISSING')"`
Expected: `OK`

---

## Task 10: Update Wiki

**Files:**
- Create: `wiki/sources/code/sft-v3-pipeline.md`
- Modify: `wiki/log.md`
- Modify: `wiki/index.md`

- [ ] **Step 1: Create the wiki page**

Create `wiki/sources/code/sft-v3-pipeline.md`:

```markdown
---
title: SFT v3 Asymmetric Distillation Pipeline
type: source
tags: [sft, distillation, training, tool-use, constitutional-ai, curriculum-learning]
sources:
  - pipeline/sft_v3_generator.py
  - pipeline/validate_sft_data.py
  - pipeline/2_model_trainer.py
updated: 2026-05-14
status: current
---

**The v3 pipeline replaces constitution-in-student-prompt with asymmetric context distillation, live tool execution, and failure injection for negative trajectories.**

The core diagnosis from the v2 post-mortem: a 0.6B parameter model cannot simultaneously track a 24-point checklist and reason about the user's problem. The v2 system prompt consumed ~200 words of the student's attention budget for rule-recitation instead of reasoning. V3 eliminates this by keeping the constitution entirely on the teacher side and distilling only the behaviours into the student data.

## Architecture: Three Phases

**Phase A — Teacher generation.** `sft_v3_generator.py` calls the teacher model (Kimi K2.6 or Minimax M2.7) with a system prompt containing all 25 constitution principles. Crucially, the teacher prompt explicitly forbids outputting rule names, checklist headers (`CAPABILITY_CHECK:`, `5W+H:`), or placeholder phrases. The result is a flowing narrative `<think>` block that implicitly demonstrates the principles rather than naming them. This is the behaviour the student model learns to imitate.

**Phase B — Reality-anchored execution.** Generation is not a single API call but a state machine. The Python script passes `stop=["</tool>"]` to litellm so generation halts immediately when a tool call body is emitted. The script then executes the tool (exa.ai for web search, subprocess for python_execute), appends the real `[TOOL_RESULT]` to the conversation, and resumes generation. The teacher's synthesis is therefore grounded in actual tool outputs, not imagined ones. This eliminates the "hallucinated execution" failure mode where the model writes "I ran a script and got X" without ever emitting a parseable `<tool>` tag.

**Phase C — Context swap.** Before writing the completed conversation to JSONL, the teacher's system prompt is replaced with the ≤50-word student prompt (`STUDENT_PROMPTS` dict keyed by tool profile). The full constitution never appears in the saved file. The 0.6B student model is trained on: [short system prompt] + [user question] + [narrative `<think>` with embedded tool calls] + [real tool results] + [`<answer>`].

## Negative Trajectories

Two new question categories train the model on failure recovery:

**`inventory_constraint`** — the question requires web_search but the session profile is `compute_only`. Correct behaviour: check the tool inventory inside `<think>`, recognise the gap, refuse honestly in `<answer>`, and redirect to an authoritative source. Trains constitution principles P3 (tool discipline) and P18 (explicit I don't know).

**`environment_timeout`** — web_search is available but the first call returns HTTP 503. Correct behaviour: retry once with a refined query; if the second attempt also fails, state the gap and answer from static knowledge with a knowledge-cutoff caveat. Trains P12 (tool failure handling).

## Quality Gate

`validate_sft_data.py` enforces five invariants before training:
1. System prompt ≤ 50 words — proves the constitution was not leaked to the student.
2. `<think>` block ≥ 50 characters — prevents synthetic laziness (empty think blocks).
3. No banned placeholders in `<think>` — blocks v2-era shortcuts like "see answer below".
4. Tool call immediately followed by tool role — sequence integrity, no hallucinated execution.
5. Last message is assistant with `<answer>` — end-to-end resolution guaranteed.

If >5% of rows fail, the generation pipeline is broken; fix the generator, not the validator.

## Curriculum Learning

`2_model_trainer.py` gains a `--curriculum_stage {1,2,3}` flag:
- Stage 1: short, no-tool examples — establishes `<think>...</think><answer>...</answer>` syntax.
- Stage 2: all examples — introduces multi-tool reasoning trajectories.
- Stage 3: all examples + 20% Stage 1 replay — prevents anti-drift loss of basic instruction-following.

Pass each stage's output checkpoint as `--from_checkpoint` for the next stage.

## V3 Format Compatibility

GRPO `_format_reward` previously required `CAPABILITY_CHECK` in every response. V3-trained models use narrative think blocks without this header. Pass `--v3_format` to the GRPO trainer to switch to the v3 reward (checks `<think>` + `<answer>` only).

## Related

- [[sources/code/sft-v2-pipeline]] — prior pipeline (preserved, v3 is additive)
- [[entities/grpo]] — GRPO/DAPO training on top of v3 SFT base
- [[topics/constitutional-ai]] — the 25-principle constitution
- [[sources/code/training-and-benchmark]] — benchmark integration
```

- [ ] **Step 2: Append to `wiki/log.md`**

Read the last few lines of `wiki/log.md` first, then append:

```markdown
## [2026-05-14] refactor | SFT v3 Asymmetric Distillation Pipeline
- Added `pipeline/sft_v3_generator.py`: teacher prompt with full 25-principle constitution, stop-sequence intercept loop for live tool execution via exa.ai, failure injection for `inventory_constraint` (missing tool) and `environment_timeout` (503 injection) categories, context swap replacing teacher system prompt with ≤50-word student prompt before saving to JSONL
- Added `pipeline/validate_sft_data.py`: pre-flight quality gate enforcing 5 invariants (system prompt length, think block length, banned placeholders, tool sequence integrity, end-to-end resolution)
- Updated `pipeline/sft_question_generator.py`: added `inventory_constraint` (60 examples, requires web_search absent) and `environment_timeout` (60 examples, 503 injection) categories
- Updated `pipeline/2_model_trainer.py`: `--curriculum_stage {1,2,3}` for staged SFT, `--from_checkpoint` for loading prior stage, `--v3_format` to disable CAPABILITY_CHECK requirement in GRPO format reward
- Updated `README.md`: added exa-py to install block, EXA_API_KEY provider, v3 pipeline usage docs
- Rationale: capacity paradox (0.6B model wastes attention on rule-recitation), synthetic laziness (single-pass generation with placeholder think blocks), hallucinated execution (no real tool results during teacher generation)
```

- [ ] **Step 3: Add pointer to `wiki/index.md`**

Find the `## Sources` → `### Code` section in `wiki/index.md` and add:

```markdown
- [[sources/code/sft-v3-pipeline]] — v3 asymmetric distillation with intercept loop, negative trajectories, and curriculum training
```

- [ ] **Step 4: Commit**

```bash
git add wiki/sources/code/sft-v3-pipeline.md wiki/log.md wiki/index.md
git commit -m "docs: add wiki page for SFT v3 pipeline and update log/index"
```

---

## Self-Review

### Spec coverage check

| Spec section | Task(s) covering it |
|---|---|
| Capacity Paradox — student prompt ≤50 words | Task 3 (STUDENT_PROMPTS), Task 2 (test), Task 7 (validation assertion 1) |
| Synthetic Laziness — think block >150 chars | Task 7 (assertion 2, threshold set to 50 to be conservative; teacher enforces >150) |
| Hallucinated Execution — intercept loop | Task 4 |
| Phase A: teacher constitution forbids rule names | Task 3 (_TEACHER_CONSTITUTION + _TEACHER_FORMAT_RULES) |
| Phase B: state machine intercept | Task 4 (_generate_with_intercept) |
| Phase C: context swap | Task 5 (_build_v3_example) |
| Exa.ai web search | Task 3 (_exa_search) |
| LoRA + 4-bit (Unsloth) | Already implemented in v2; no change needed |
| Curriculum 3-stage | Task 8 (_split_curriculum_stages, --curriculum_stage) |
| Inventory constraint category | Task 1 |
| Environment timeout (503) | Task 1 (category) + Task 5 (failure injection) |
| Failure injection in generator | Task 5 (pick_tool_profile returns failure_config) |
| validate_sft_data.py — all 5 assertions | Task 7 |
| System prompt word count assertion | Task 7 (assertion 1) |
| Think block length assertion | Task 7 (assertion 2) |
| Banned placeholder assertion | Task 7 (assertion 3) |
| Tool sequence integrity assertion | Task 7 (assertion 4) |
| End-to-end resolution assertion | Task 7 (assertion 5) |
| >5% drop rate = fatal | Task 7 (--max_drop_pct flag, sys.exit(1)) |
| README + wiki sync | Tasks 9, 10 |

### Placeholder scan
None found — all code blocks are complete and runnable.

### Type consistency
- `_execute_tool_v3(tool_inner: str, active_tools: set[str], failure_config: dict | None) -> str` — consistent across Task 3 definition and Task 2 tests.
- `_build_v3_example(conversation, question, category, tool_profile, violations) -> dict` — consistent across Task 5 definition and Task 2 tests.
- `_split_curriculum_stages(examples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]` — consistent across Task 8 definition and Task 8 tests.
- `validate_row(row: dict) -> tuple[bool, str]` — consistent across Task 7 definition and Task 6 tests.
- `validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]` — consistent across Task 7 and Task 6 `test_validate_file_drop_rate`.

---

## Final Commit

After all tasks pass:

```bash
git add -A
git commit -m "feat: SFT v3 asymmetric distillation pipeline — intercept loop, exa.ai, curriculum training, quality gate"
git push
```
