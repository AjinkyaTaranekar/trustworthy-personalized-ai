"""
SFT v3 Asymmetric Distillation Generator
=========================================
Replaces the v2 approach of including the full 25-principle constitution in the
student system prompt. Instead:

  Phase A  Teacher generates with full constitution (never saved to JSONL).
  Phase B  Tool calls intercepted mid-generation via stop=["</tool>"], executed live.
  Phase C  Before saving, swap teacher system prompt for a ≤50-word student prompt.

Web search uses exa.ai (set EXA_API_KEY in .env).

Usage:
    python sft_v3_generator.py \\
        --questions data/questions_partA.jsonl \\
        --output data/train_v3.jsonl \\
        --model nvidia_nim/minimaxai/minimax-m2.7

    python sft_v3_generator.py \\
        --questions data/questions_v3.jsonl \\
        --type inventory_constraint \\
        --output data/train_v3_negative.jsonl
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv
from pipeline_tools import ToolRegistry as _ToolRegistry
from scratchpad import ScratchpadStore as _ScratchpadStore

_TOOL_REGISTRY = _ToolRegistry()   # no scratchpad/user_memory — training context

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0

# ---------------------------------------------------------------------------
# Adaptive concurrency — scales workers down on rate limits, up after quiet period
# ---------------------------------------------------------------------------

class _AdaptiveSemaphore:
    """Semaphore whose count halves on sustained rate limits and recovers gradually."""

    def __init__(self, initial: int, min_val: int = 1, scale_up_interval: float = 300.0):
        self._min = min_val
        self._max = initial
        self._current = initial
        self._sem = threading.Semaphore(initial)
        self._lock = threading.Lock()
        self._rl_events: list[float] = []
        self._rl_window = 60.0       # look-back window for rate-limit events
        self._rl_threshold = 3       # events in window before scaling down
        self._scale_up_interval = scale_up_interval
        self._last_scale_up = time.monotonic()

    # -- public API --

    def acquire(self) -> None:
        self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    def on_rate_limit(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._rl_events.append(now)
            self._rl_events = [t for t in self._rl_events if now - t < self._rl_window]
            if len(self._rl_events) >= self._rl_threshold and self._current > self._min:
                target = max(self._min, self._current // 2)
                drop = self._current - target
                stolen = 0
                for _ in range(drop):
                    if self._sem.acquire(blocking=False):
                        stolen += 1
                    else:
                        break
                if stolen:
                    self._current -= stolen
                    self._rl_events.clear()
                    print(f"  {_tag()} [adaptive] rate limited ×{self._rl_threshold} — workers ↓ {self._current}", flush=True)

    def try_scale_up(self) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._rl_events if now - t < self._rl_window]
            if (self._current < self._max
                    and not recent
                    and now - self._last_scale_up >= self._scale_up_interval):
                self._sem.release()
                self._current += 1
                self._last_scale_up = now
                print(f"  [adaptive] quiet for {self._scale_up_interval/60:.0f}min — workers ↑ {self._current}/{self._max}", flush=True)

    @property
    def current(self) -> int:
        return self._current


_adaptive_sem: _AdaptiveSemaphore | None = None


# ---------------------------------------------------------------------------
# Token bucket — global rate limiter, paces all threads to ≤ N calls/min
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Thread-safe token bucket. Callers block in acquire() until a token is available."""

    def __init__(self, rate_per_minute: float):
        self._rate = rate_per_minute / 60.0          # tokens per second
        self._max = max(1.0, rate_per_minute / 10)   # max burst ≈ 6s worth
        self._tokens = self._max                     # start full
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._max, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def update_rate(self, rate_per_minute: float) -> None:
        with self._lock:
            self._rate = rate_per_minute / 60.0
            self._max = max(1.0, rate_per_minute / 10)


_token_bucket: _TokenBucket | None = None


# ---------------------------------------------------------------------------
# API key rotation — cycles through multiple keys when one hits rate limits
# ---------------------------------------------------------------------------

class _KeyRotator:
    """Round-robin key rotator. Rotates to the next key after N consecutive rate limits."""

    def __init__(self, keys: list[str], threshold: int = 3):
        self._keys = keys
        self._threshold = threshold
        self._idx = 0
        self._hits = 0
        self._lock = threading.Lock()

    @property
    def current_key(self) -> str | None:
        return self._keys[self._idx] if self._keys else None

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    def on_rate_limit(self) -> None:
        if len(self._keys) <= 1:
            return
        with self._lock:
            self._hits += 1
            if self._hits >= self._threshold:
                old = self._idx + 1
                self._idx = (self._idx + 1) % len(self._keys)
                self._hits = 0
                print(
                    f"  {_tag()} [key_rotator] key {old}/{len(self._keys)} exhausted "
                    f"— rotating to key {self._idx + 1}/{len(self._keys)}",
                    flush=True,
                )

    def reset_hits(self) -> None:
        with self._lock:
            self._hits = 0


_key_rotator: _KeyRotator | None = None

# Per-thread context so any helper can emit tagged log lines without passing tag everywhere.
_thread_local = threading.local()

def _tag() -> str:
    return getattr(_thread_local, "tag", "[?]")

# ---------------------------------------------------------------------------
# Tool profiles — must match 3_infererence.py and sft_gold_response_generator.py
# ---------------------------------------------------------------------------

TOOL_PROFILES = [
    {
        "label": "all_tools",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "All tools available. Scratchpad and user memory are always present regardless of profile.",
    },
    {
        "label": "compute_only",
        "context": "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "python_execute and datetime available. No internet access. Scratchpad and user memory always available.",
    },
    {
        "label": "compute_and_search",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "python_execute and web_search/read_url available. No datetime tool. Scratchpad and user memory always available.",
    },
    {
        "label": "no_tools",
        "context": "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "No compute or web tools this session. Datetime, scratchpad, and user memory are available.",
    },
]

# ---------------------------------------------------------------------------
# Student prompts — ≤50 words each (validated by tests)
# These are what appear in the SAVED JSONL — the student model only sees these.
# ---------------------------------------------------------------------------

STUDENT_PROMPTS: dict[str, str] = {
    "all_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url, get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "compute_only": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "compute_and_search": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "no_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
}

# ---------------------------------------------------------------------------
# Teacher system prompt — full constitution, NEVER saved to JSONL
# ---------------------------------------------------------------------------

_TEACHER_CONSTITUTION = """\
Your reasoning principles (demonstrate through behavior; NEVER name them, never output checklists):
1. Before answering, reason through WHO is affected, WHAT is required, WHEN (time-sensitivity), WHERE (domain/jurisdiction), WHY (underlying intent), and HOW (method) — in flowing narrative inside <think>.
2. State which tools are available this session; only call tools that are listed as available.
3. Use python_execute for any precision arithmetic or computation; never approximate mentally when code is available.
4. For live data or named entities, use web_search if available; if not, state the limitation clearly and redirect to an authoritative source.
5. For questions requiring personal context you do not have, ask exactly ONE clarifying question — the most critical unknown.
6. Hedge only genuinely uncertain claims; state well-known facts confidently.
7. For tasks that are fundamentally impossible, name the irreducible reason and redirect usefully.
8. For subjective questions, enumerate 3–5 tradeoff dimensions; never declare a universal winner.
9. Only call tools listed as available this session; never invent tools.
10. If a tool call fails, retry once with a modified query; if it fails again, state the gap honestly.
11. Never capitulate under user pressure after a correct refusal; cite the specific consequence of guessing.
12. For multi-step ambiguities, ask only the single most critical clarifying question first.
13. For queries with 3 or more distinct requirements, reason through them systematically before executing.
14. For partially-capable scenarios: answer achievable parts fully; for blocked parts name what/why/redirect.
15. Name assumptions explicitly; mark them as unverified if they are not confirmed facts.
16. Call user_memory_read at the start of every response to check for stored user context (preferences, constraints, goals, history); use what you find to personalise tone, depth, and focus.
17. Use scratchpad_update to store intermediate calculations, sub-results, or hypotheses mid-reasoning; read it back with scratchpad_read when picking up a multi-step chain.
18. Call user_memory_update before closing with <answer> whenever the conversation reveals a new, durable fact about the user (role, preference, constraint, goal) that would improve future responses.
19. For any time-sensitive query, call get_datetime immediately after user_memory_read to anchor your response in real current time before searching or computing."""

_TEACHER_FORMAT_RULES = """\
CRITICAL FORMAT RULES — violation invalidates the training example:
1. Open with <think> containing flowing narrative reasoning (minimum 150 characters). NO headers, NO rule numbers, NO "CAPABILITY_CHECK:", NO "5W+H:", NO bullet lists inside <think>.
2. Place ALL tool calls after </think> and before <answer> using: <tool>tool_name(arg='...')</tool>
3. FIRST tool calls after </think>: call user_memory_sections() to learn section keys, then user_memory_read() to fetch user context — use the result to personalise your response.
4. For multi-step problems: call scratchpad_sections() to learn section keys, then use scratchpad_update/scratchpad_read to track intermediate state.
5. Close EVERY response with <answer>...</answer>. If you learned a new durable user fact, call user_memory_update(section='<key from user_memory_sections>', content='...') immediately before <answer>.
6. NEVER output these phrases: "see answer below", "inferred from question", "none flagged", "CAPABILITY_CHECK:", "PRINCIPLE_", "5W+H:", "CONSEQUENCE_CHECK:".
7. After EVERY [TOOL_RESULT] block, open a NEW <think>...</think> block to reason about what you just learned before calling another tool or writing <answer>. This is mandatory — never skip straight from a tool result to the next tool call or to <answer> without re-thinking."""


def _make_teacher_prompt(tool_profile: dict, category: str, ideal_behavior: str) -> str:
    return (
        # FORMAT RULES come first — models anchor on the beginning of the system prompt.
        # Placing <think> requirement here ensures it is read before the constitution.
        "You are a frontier AI assistant generating exemplary training data.\n\n"
        "MANDATORY OUTPUT FORMAT — follow this exactly for every response:\n"
        "  Step 1: Open with <think> and write flowing narrative reasoning (≥150 chars).\n"
        "          No bullet points, no headers, no rule numbers inside <think>.\n"
        "  Step 2: Close reasoning with </think>.\n"
        "  Step 3: Call <tool>user_memory_sections()</tool> to see section keys, then\n"
        "          <tool>user_memory_read(prompt='what do I know about this user?')</tool>.\n"
        "          Use the result to personalise your response.\n"
        "  Step 4: For multi-step problems, call <tool>scratchpad_sections()</tool> first,\n"
        "          then use scratchpad_update/scratchpad_read to track intermediate state.\n"
        "  Step 5: Call other tools as needed. After each [TOOL_RESULT], continue in prose.\n"
        "  Step 6: If you learned a new durable user fact, call\n"
        "          <tool>user_memory_update(section='<key from sections>', content='...')</tool>.\n"
        "  Step 7: Close with <answer>...</answer>.\n"
        "  EXAMPLE SKELETON (think → tool → THINK AGAIN → tool → answer):\n"
        "    <think>The user is asking ... I need to check their memory first ...</think>\n"
        "    <tool>user_memory_sections()</tool>\n"
        "    <think>Now I know the section keys. I'll read their memory ...</think>\n"
        "    <tool>user_memory_read(prompt='user background and preferences')</tool>\n"
        "    <think>Memory shows [X]. Now I can answer, but first I need to search ...</think>\n"
        "    <tool>web_search(query='...')</tool>\n"
        "    <think>The search returned [Y]. I now have enough to answer fully ...</think>\n"
        "    <tool>user_memory_update(section='facts', content='...')</tool>\n"
        "    <think>Memory updated. Writing final answer personalised to user ...</think>\n"
        "    <answer>Based on your context, ...</answer>\n\n"
        f"Session tools available: {tool_profile['context']}\n"
        f"{tool_profile['system_note']}\n\n"
        f"CATEGORY: {category}\n"
        f"Requirements for this category:\n"
        f"{ideal_behavior}\n\n"
        f"{_TEACHER_CONSTITUTION}\n\n"
        f"{_TEACHER_FORMAT_RULES}\n"
    )




# ---------------------------------------------------------------------------
# Pure helpers (also tested directly)
# ---------------------------------------------------------------------------

def _question_id(item: dict) -> str:
    """Return a stable ID for a question item.

    Uses the 'id' field if present; otherwise derives a 12-char hex hash from
    the question text so existing JSONL files work without modification.
    """
    if item.get("id"):
        return str(item["id"])
    text = item.get("question", "")
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


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


def _wrap_missing_think(content: str) -> str:
    """If the model wrote reasoning but didn't wrap it in <think>, wrap it.

    Works for both no-tool responses (reasoning before <answer>) and
    tool-calling responses (reasoning before <tool>). Does nothing when
    <think> is already present or when there is too little pre-boundary text.
    """
    if re.search(r"<think\s*>", content, re.IGNORECASE):
        return content  # already wrapped — nothing to do

    # Find first structural boundary: <tool> or <answer>
    first_boundary = len(content)
    for pat in (r"<tool\s*>", r"<answer\s*>"):
        m = re.search(pat, content, re.IGNORECASE)
        if m and m.start() < first_boundary:
            first_boundary = m.start()

    reasoning = content[:first_boundary].strip()
    rest = content[first_boundary:]

    if len(reasoning) < 80:
        return content  # too little reasoning to wrap — quality gate will reject

    return f"<think>\n{reasoning}\n</think>\n{rest}"


def _count_violations(violations: str) -> int:
    if violations.strip() == "NO_VIOLATIONS":
        return 0
    return sum(
        1 for line in violations.splitlines()
        if line.startswith("PRINCIPLE_") or line.startswith("ISSUE_")
    )


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
        # Pace rate first, then gate concurrency — both released before any sleep.
        if _token_bucket:
            _token_bucket.acquire()
        if _adaptive_sem:
            _adaptive_sem.acquire()
        try:
            kwargs: dict = dict(model=model, messages=messages, max_tokens=max_tokens)
            if api_base:
                kwargs["api_base"] = api_base
            if stop:
                kwargs["stop"] = stop
            if _key_rotator and _key_rotator.current_key:
                kwargs["api_key"] = _key_rotator.current_key
            resp = litellm.completion(**kwargs)
            content = resp.choices[0].message.content or ""
            # Reset key hit counter on success — key is healthy
            if _key_rotator:
                _key_rotator.reset_hits()
            snippet = content[:200].replace("\n", " ")
            key_label = f" [key {_key_rotator._idx + 1}/{_key_rotator.n_keys}]" if _key_rotator and _key_rotator.n_keys > 1 else ""
            print(f"  {_tag()} api←{key_label} {snippet}{'…' if len(content) > 200 else ''}", flush=True)
            return content.strip()
        except litellm.RateLimitError as exc:
            if _key_rotator:
                _key_rotator.on_rate_limit()
            if _adaptive_sem:
                _adaptive_sem.on_rate_limit()
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            key_label = f" [key {_key_rotator._idx + 1}/{_key_rotator.n_keys}]" if _key_rotator and _key_rotator.n_keys > 1 else ""
            print(f"  {_tag()} [rate_limit]{key_label} retry {attempt+1}/{_MAX_RETRIES} in {wait:.0f}s", flush=True)
        except (litellm.APIConnectionError, litellm.Timeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt)
            print(f"  {_tag()} [conn_error] retry {attempt+1}/{_MAX_RETRIES} in {wait:.0f}s", flush=True)
        else:
            wait = None
        finally:
            if _adaptive_sem:
                _adaptive_sem.release()
        # Sleep outside the semaphore so slots are free for the scaler to steal.
        if wait:
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
    max_rounds: int = 5,
    registry: "_ToolRegistry | None" = None,
) -> list[dict]:
    """Generate text iteratively, intercept <tool> calls, execute them live.

    Uses stop=["</tool>"] so generation halts when a tool call body is emitted.
    The script then executes the tool and resumes generation with the real result.
    """
    _registry = registry or _TOOL_REGISTRY
    conversation = list(messages)
    active_tools = {
        part.split("✓")[0].strip()
        for part in tool_profile["context"].split("|")
        if "✓" in part
    }
    # Per-tool call counter — prevents the model looping on the same tool.
    # web_search and read_url are capped at 3 calls each; others at 5.
    _MAX_CALLS: dict[str, int] = {"web_search": 3, "read_url": 3}
    _DEFAULT_MAX = 5
    tool_call_counts: dict[str, int] = {}

    for round_num in range(max_rounds):
        content = _call_with_stop(
            messages=conversation,
            model=model,
            max_tokens=2048,
            api_base=api_base,
            stop=["</tool>"],
        )

        # Detect tool interception: <tool> present but </tool> absent after it.
        # Some tokenizers emit </tool and > as separate tokens; the stop sequence
        # strips the > but leaves </tool in the content, so we check for both.
        tool_pos = content.rfind("<tool>")
        stripped_end = content.rstrip()
        partial_close = stripped_end.endswith("</tool")  # tokenizer artifact
        is_tool_call = tool_pos != -1 and (
            "</tool>" not in content[tool_pos:] or partial_close
        )

        if not is_tool_call:
            conversation.append({"role": "assistant", "content": content})
            break

        # Reconstruct full tool tag, handling partial </tool artefact
        if partial_close:
            full_assistant_content = stripped_end + ">"
        else:
            full_assistant_content = content + "</tool>"

        # Extract inner text, strip any partial closing tag leftover
        tool_inner = content[tool_pos + len("<tool>"):]
        tool_inner = re.sub(r"\s*</tool\s*$", "", tool_inner).strip()

        tool_name_m = re.match(r"(\w+)", tool_inner)
        tool_name = tool_name_m.group(1) if tool_name_m else "unknown"

        # Enforce per-tool call cap to prevent search loops
        call_count = tool_call_counts.get(tool_name, 0)
        cap = _MAX_CALLS.get(tool_name, _DEFAULT_MAX)
        if call_count >= cap:
            result = f"Error: {tool_name} call limit ({cap}) reached. Synthesise what you have and write <answer>."
        else:
            result = _registry.execute(tool_inner, active_tools, failure_config)
            tool_call_counts[tool_name] = call_count + 1

        args_snippet = tool_inner[:120].replace("\n", " ")
        result_snippet = result[:200].replace("\n", " ")
        print(
            f"  {_tag()} [tool r{round_num}] call={tool_name}({args_snippet}{'…' if len(tool_inner) > 120 else ''})  "
            f"result({len(result)}c)={result_snippet}{'…' if len(result) > 200 else ''}",
            flush=True,
        )

        # Inject result and require a new <think> block — trains the model to
        # re-reason after each tool result before calling the next tool or answering.
        # The instructional suffix is stripped in _build_v3_example so it does not
        # pollute student training data.
        is_last_round = round_num >= max_rounds - 2
        followup = (
            "Open a new <think>...</think> block to reason about this result, "
            "then write your <answer> — no more tool calls."
            if is_last_round else
            "Open a new <think>...</think> block to reason about this result, "
            "then call the next tool or write your <answer>."
        )
        wrapped = f"[TOOL_RESULT: {tool_name}]\n{result}\n[/TOOL_RESULT]\n{followup}"

        conversation.append({"role": "assistant", "content": full_assistant_content})
        # Use "user" role here for the teacher's generation (NVIDIA NIM and most
        # OpenAI-compatible APIs reject role="tool" without a native tool_call_id).
        # _build_v3_example converts these to role="tool" for the student JSONL.
        conversation.append({"role": "user", "content": wrapped})

    return conversation


# ---------------------------------------------------------------------------
# Context swap — Phase C: replace teacher prompt with student prompt
# ---------------------------------------------------------------------------

def _build_v3_example(
    conversation: list[dict],
    question: str,
    category: str,
    tool_profile: dict,
    violations: str = "NO_VIOLATIONS",
    question_id: str = "",
) -> dict:
    """Build a JSONL training row. Teacher system prompt → student prompt (context swap).

    Three student-facing cleanups applied here so the JSONL is clean regardless of
    what the teacher needed internally:
      1. role="user" tool-result messages  → role="tool" with name field.
      2. Teacher scaffolding suffix stripped from tool results (kept only up to [/TOOL_RESULT]).
      3. </tool</tool> partial-tag artefact normalised to </tool>.
    """
    student_system = STUDENT_PROMPTS[tool_profile["label"]]
    messages = []
    for m in conversation:
        role = m["role"]
        content = m.get("content", "") or ""
        if role == "system":
            messages.append({"role": "system", "content": student_system})
        elif role == "user" and content.lstrip().startswith("[TOOL_RESULT:"):
            # Strip teacher-only instruction suffix — keep only up to [/TOOL_RESULT]
            end = content.find("[/TOOL_RESULT]")
            clean = (content[:end + len("[/TOOL_RESULT]")].strip()
                     if end != -1 else content.strip())
            name_m = re.match(r"\[TOOL_RESULT:\s*(\w+)\]", clean)
            tool_name = name_m.group(1) if name_m else "tool"
            messages.append({"role": "tool", "name": tool_name, "content": clean})
        elif role == "assistant":
            # Normalise </tool</tool> artefact produced by partial stop-sequence stripping
            clean = re.sub(r"</tool\s*</tool>", "</tool>", content)
            messages.append({**m, "content": clean})
        else:
            messages.append(m)

    n_viol = _count_violations(violations)
    return {
        "messages": messages,
        "metadata": {
            "source": "v3_distillation",
            "question_id": question_id,
            "category": category,
            "tool_profile": tool_profile["label"],
            "constitution_score": max(0.0, round(1.0 - n_viol * 0.05, 3)),
            "revised": violations != "NO_VIOLATIONS",
            "pipeline": "sft_v3",
        },
    }


# ---------------------------------------------------------------------------
# Tool profile selection with failure injection support
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

    failure_config is None for normal categories, {"inject_503": True} for
    environment_timeout, or a specific profile is forced for inventory_constraint.
    """
    if category == "inventory_constraint":
        required_label = (item or {}).get("required_profile", "compute_only")
        profile = next(
            (p for p in TOOL_PROFILES if p["label"] == required_label),
            TOOL_PROFILES[1],
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


# ---------------------------------------------------------------------------
# Per-category ideal behaviors for the teacher prompt
# ---------------------------------------------------------------------------

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
    )
}

_DEFAULT_IDEAL_V3 = (
    "Reason through the question step-by-step in a <think> block, demonstrating the principles. "
    "After </think>, FIRST call user_memory_read to check for stored user context and use it to personalise your response. "
    "Use other tools as needed after that, calling them with <tool> tags. "
    "For multi-step problems, use scratchpad_update to log intermediate results and scratchpad_read to retrieve them. "
    "After each tool call, continue reasoning in flowing prose before the next tool call or final answer. "
    "If the conversation reveals a new durable fact about the user, call user_memory_update before closing. "
    "Close with a clear <answer> that directly addresses the user's question, personalised using any memory found. "
    "Avoid any mention of the principles, checklists, or placeholders in your final output."
)


# ---------------------------------------------------------------------------
# Sample user profiles — injected into training so the teacher sees realistic
# memory data and learns to personalise. Eight diverse personas covering
# different roles, languages, technical levels, and cultural contexts.
# ---------------------------------------------------------------------------

_SAMPLE_USER_PROFILES: list[dict] = [
    {
        "who": "Software engineer at a fintech startup, 5 years Python/Go experience.",
        "what": "Builds data pipelines and REST APIs. Transitioning into ML engineering.",
        "where": "Dublin, Ireland. Remote-first. EU regulatory context applies.",
        "why": "Wants concise, technically rigorous answers with working code examples.",
        "how": "Reads docs carefully before asking. Prefers code over prose explanations.",
        "facts": "Strong Python. New to neural networks. Deadline-driven work style.",
        "constraints": "Limited time. No budget for expensive cloud GPU services.",
    },
    {
        "who": "MSc Computer Science student at Trinity College Dublin.",
        "what": "Writing dissertation on trustworthy AI and personalisation in LLMs.",
        "where": "University campus, Ireland. Has academic library access.",
        "why": "Needs cited, verifiable sources. Understands transformer architecture.",
        "how": "Learns by reading papers then implementing prototypes. Uses HuggingFace.",
        "facts": "Strong mathematics background. Intermediate PyTorch user. British English spelling.",
        "constraints": "Must cite sources. Thesis deadline June 2026. No local GPU.",
    },
    {
        "who": "Small business owner running an independent bakery in Madrid, Spain.",
        "what": "Managing inventory, orders, social media, and staff scheduling.",
        "where": "Madrid, Spain. Operates in Spanish. EU consumer law applies.",
        "why": "Wants simple digital tools, not complex enterprise software.",
        "how": "Non-technical but quick learner. Needs step-by-step instructions.",
        "facts": "Native Spanish speaker, basic English. Smartphone-first user.",
        "constraints": "Very limited time. Tight budget — free tools preferred. Prefers Spanish.",
    },
    {
        "who": "Registered nurse with 8 years ICU experience, Toronto, Canada.",
        "what": "Asks clinical questions, drug interactions, and protocol clarifications.",
        "where": "Ontario, Canada. Canadian healthcare regulations (PIPEDA) apply.",
        "why": "Needs quick, accurate clinical reference during 12-hour shifts.",
        "how": "Comfortable with medical terminology. Wants concise clinical summaries.",
        "facts": "Expert in critical care. Uses metric units. Prefers UpToDate-style sources.",
        "constraints": "Time-critical during shifts. Canadian dosing differs from US guidelines.",
    },
    {
        "who": "Retired secondary school teacher, 68, living in rural Brittany, France.",
        "what": "Recently got a smartphone. Learning to use the internet and online services.",
        "where": "Rural France. French-speaking. Limited broadband (4G only).",
        "why": "Wants to stay connected with grandchildren and manage paperwork online.",
        "how": "Needs jargon-free explanations with numbered steps. Patient, encouraging tone.",
        "facts": "No technical background. Fluent French only. Uses Samsung Galaxy phone.",
        "constraints": "Limited data plan. Confused by technical jargon. Needs reassurance.",
    },
    {
        "who": "Data scientist at a mid-size e-commerce company, São Paulo, Brazil.",
        "what": "Builds recommendation models and A/B testing frameworks.",
        "where": "Brazil. LGPD data privacy law applies. Uses AWS infrastructure.",
        "why": "Exploring LLM-based features: RAG and fine-tuning for recommender systems.",
        "how": "Prefers Python with benchmark numbers and tradeoff tables.",
        "facts": "Fluent English and Portuguese. Strong statistics background. Uses Jupyter daily.",
        "constraints": "No proprietary data to external APIs. Open-source models strongly preferred.",
    },
    {
        "who": "Parent of two children (ages 8 and 11), part-time librarian, Auckland, NZ.",
        "what": "Researching homework topics, family activities, household budgeting.",
        "where": "Auckland, New Zealand. NZ English spelling. NZDT timezone.",
        "why": "Wants accurate, age-appropriate information quickly.",
        "how": "Generalist. Comfortable with Google-level information literacy.",
        "facts": "Prefers NZ-specific sources and local pricing. Uses a MacBook.",
        "constraints": "Limited time (school hours). Needs child-safe content framing when relevant.",
    },
    {
        "who": "Freelance graphic designer, 29, based in Berlin, Germany.",
        "what": "Creates brand identities, social media assets, pitch decks for startups.",
        "where": "Berlin. Fluent German and English. EU GDPR applies to client data.",
        "why": "Uses AI to speed up research, copywriting, and client proposals.",
        "how": "Creative thinker. Not comfortable with code. Prefers visual or structured explanations.",
        "facts": "Uses Adobe CC and Figma. Deep design knowledge, minimal tech background.",
        "constraints": "Client NDAs — cannot share specifics. Needs output directly usable in pitches.",
    },
]


class _OneProfileMemoryStore:
    """Wraps one sampled user profile per training question. Immutable after init — thread-safe."""

    def __init__(self, profile: dict) -> None:
        self._base = dict(profile)
        self._updates: dict[str, str] = {}

    def read(self, session_id: str, prompt: str = "") -> str:
        merged = {**self._base, **self._updates}
        return "\n".join(f"[{k.upper()}] {v}" for k, v in merged.items())

    def update(self, session_id: str, section: str, content: str) -> str:
        self._updates[section] = content
        return f"(user memory updated: section='{section}')"


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------

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
    category = item.get("category", "unknown")
    question = item.get("question", "").strip()
    if not question:
        return "error"
    q_id = _question_id(item)

    tool_profile, failure_config = pick_tool_profile(category, item)
    elapsed = time.monotonic() - run_start
    tag = f"[{idx}/{total}:{category}:{q_id}]"
    _thread_local.tag = tag  # available to _call_with_stop and any helper on this thread
    print(f"\n{tag} profile={tool_profile['label']} elapsed={elapsed:.0f}s")
    print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")

    # Each question gets its own registry with a sampled user profile so the
    # teacher sees realistic memory data and learns genuine personalisation.
    sampled_profile = random.choice(_SAMPLE_USER_PROFILES)
    memory_store = _OneProfileMemoryStore(sampled_profile)
    scratchpad_store = _ScratchpadStore()
    q_registry = _ToolRegistry(user_memory_store=memory_store, scratchpad_store=scratchpad_store)
    q_registry.session_id = q_id  # required — registry gates memory/scratchpad reads on session_id being set
    print(f"  {tag} user_profile='{sampled_profile['who'][:60]}'")

    ideal = _IDEAL_BEHAVIORS_V3.get(category, _DEFAULT_IDEAL_V3)
    teacher_system = _make_teacher_prompt(tool_profile, category, ideal)

    # Pre-execute mandatory preamble tool calls and inject them as fake turns.
    # This skips 2–3 LLM round trips that every conversation wastes on boilerplate
    # tool discovery (user_memory_sections → user_memory_read → get_datetime).
    # The model starts generating from *after* the preamble, with context already loaded.
    _active_tools = {
        part.split("✓")[0].strip()
        for part in tool_profile["context"].split("|")
        if "✓" in part
    }
    _followup = (
        "Open a new <think>...</think> block to reason about this result, "
        "then call the next tool or write your <answer>."
    )
    preamble: list[dict] = []

    mem_sections = q_registry.execute("user_memory_sections()", _active_tools, None)
    mem_content = q_registry.execute(
        "user_memory_read(prompt='user background and preferences')", _active_tools, None
    )
    preamble += [
        {"role": "assistant", "content": "<think>I should check what user memory sections exist, then read the stored profile to personalise my response.</think><tool>user_memory_sections()</tool>"},
        {"role": "user", "content": f"[TOOL_RESULT: user_memory_sections]\n{mem_sections}\n[/TOOL_RESULT]\n{_followup}"},
        {"role": "assistant", "content": "<think>Good, I can see the section keys. Now I'll read the user's stored context before answering.</think><tool>user_memory_read(prompt='user background and preferences')</tool>"},
        {"role": "user", "content": f"[TOOL_RESULT: user_memory_read]\n{mem_content}\n[/TOOL_RESULT]\n{_followup}"},
    ]

    if "get_datetime" in _active_tools:
        dt_result = q_registry.execute("get_datetime()", _active_tools, None)
        preamble += [
            {"role": "assistant", "content": "<think>Let me anchor the current time before responding to any time-sensitive aspects of this question.</think><tool>get_datetime()</tool>"},
            {"role": "user", "content": f"[TOOL_RESULT: get_datetime]\n{dt_result}\n[/TOOL_RESULT]\n{_followup}"},
        ]

    print(f"  {tag} pre-seeded {len(preamble)//2} preamble tool(s): mem_sections, mem_read"
          + (", datetime" if "get_datetime" in _active_tools else ""))

    initial_messages = [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": question},
        *preamble,
    ]

    try:
        t0 = time.monotonic()
        conversation = _generate_with_intercept(
            messages=initial_messages,
            model=model,
            tool_profile=tool_profile,
            api_base=api_base,
            failure_config=failure_config,
            registry=q_registry,
        )
        # Find the first assistant turn and auto-wrap reasoning in <think> if needed.
        # Models that don't natively emit <think> tags (e.g. Minimax M2.7) write
        # flowing prose that we wrap here so training data stays format-consistent.
        first_asst_idx = next(
            (i for i, m in enumerate(conversation) if m["role"] == "assistant"), None
        )
        if first_asst_idx is not None:
            original = conversation[first_asst_idx]["content"]
            wrapped = _wrap_missing_think(original)
            if wrapped != original:
                conversation[first_asst_idx] = {**conversation[first_asst_idx], "content": wrapped}
                print(f"  {tag} auto-wrapped missing <think> block")
            first_asst = conversation[first_asst_idx]["content"]
        else:
            first_asst = ""

        think_len = _think_block_length(first_asst)
        if think_len < 150:
            print(f"  {tag} short <think> ({think_len} chars) — proceeding anyway")

        q_elapsed = time.monotonic() - t0
        n_tool_turns = sum(1 for m in conversation if m["role"] == "user" and m["content"].startswith("[TOOL_RESULT:"))
        print(f"  {tag} {len(conversation)} msgs ({n_tool_turns} tool turns) in {q_elapsed:.1f}s", flush=True)

        example = _build_v3_example(conversation, question, category, tool_profile, question_id=q_id)

        with file_lock:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_file.flush()

        print(f"  {tag} ✓ written (q_time={q_elapsed:.1f}s, workers={_adaptive_sem.current if _adaptive_sem else '?'})", flush=True)
        return "ok"

    except Exception as e:
        print(f"  {tag} error: {e}")
        return "error"


# ---------------------------------------------------------------------------
# Background watcher — commit+push output file every N new lines
# ---------------------------------------------------------------------------

def _watcher_thread(out_path: Path, threshold: int, interval: int = 30) -> None:
    """Daemon thread: commit+push output file every `threshold` new non-blank lines."""
    try:
        repo_root = Path(
            subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        )
        rel_path = out_path.relative_to(repo_root)
    except Exception as exc:
        print(f"[watcher] git repo not found — watch_commit disabled ({exc})", flush=True)
        return
    def _count() -> int:
        try:
            return sum(1 for l in out_path.open("r", encoding="utf-8") if l.strip())
        except OSError:
            return 0
    baseline = _count()
    last_committed = (baseline // threshold) * threshold
    print(f"[watcher] started — file: {out_path.name}, threshold: {threshold} lines/commit", flush=True)
    while True:
        time.sleep(interval)
        current = _count()
        checkpoint = (current // threshold) * threshold
        if checkpoint > last_committed:
            msg = f"data: auto-checkpoint — {current} lines in {out_path.name}"
            try:
                subprocess.run(["git", "add", str(rel_path)], cwd=repo_root, check=True)
                subprocess.run(["git", "commit", "-m", msg], cwd=repo_root, check=True)
                subprocess.run(["git", "push"], cwd=repo_root, check=True)
                print(f"[watcher] committed + pushed at {current} lines", flush=True)
                last_committed = checkpoint
            except subprocess.CalledProcessError as exc:
                print(f"[watcher] git error: {exc}", flush=True)


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
    min_workers: int = 1,
    api_keys: list[str] | None = None,
    skip_categories: set[str] | None = None,
    rpm_per_key: float = 38.0,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"

    done_ids: set[str] = set()
    if not overwrite and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    meta = ex.get("metadata", {})
                    qid = meta.get("question_id", "")
                    if qid:
                        done_ids.add(qid)
                    else:
                        # Fallback: hash first user message for pre-ID outputs
                        user_msgs = [m["content"] for m in ex["messages"] if m["role"] == "user"]
                        text = user_msgs[0] if user_msgs else ""
                        if text:
                            done_ids.add(hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12])
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        if done_ids:
            print(f"Resume: {len(done_ids)} questions already processed (by id)")

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
            if skip_categories and cat in skip_categories:
                skipped += 1
                continue
            q = item.get("question", "").strip()
            if not q or _question_id(item) in done_ids:
                skipped += 1
                continue
            items.append(item)

    if max_examples:
        items = items[:max_examples]

    print(f"Questions: {len(items)} to process (skipped={skipped}, parse_errors={parse_errors})")

    global _adaptive_sem, _key_rotator
    _adaptive_sem = _AdaptiveSemaphore(initial=workers, min_val=min_workers, scale_up_interval=300.0)
    print(f"Workers   : {workers} max / {min_workers} min (adaptive)", flush=True)

    # Key rotation — merge CLI keys with env var NVIDIA_NIM_API_KEYS
    _all_keys: list[str] = []
    env_keys_raw = os.environ.get("NVIDIA_NIM_API_KEYS", "")
    if env_keys_raw:
        _all_keys.extend(k.strip() for k in env_keys_raw.split(",") if k.strip())
    if api_keys:
        _all_keys.extend(k for k in api_keys if k not in _all_keys)
    if _all_keys:
        _key_rotator = _KeyRotator(_all_keys, threshold=5)
        total_rpm = rpm_per_key * len(_all_keys)
        print(f"API keys  : {len(_all_keys)} key(s) loaded — rotating after 5 rate limits per key", flush=True)
    else:
        _key_rotator = None
        total_rpm = rpm_per_key

    global _token_bucket
    _token_bucket = _TokenBucket(rate_per_minute=total_rpm)
    print(f"Rate limit: {total_rpm:.0f} RPM token bucket ({rpm_per_key:.0f} RPM × {len(_all_keys) if _all_keys else 1} key(s))", flush=True)

    _scaler_stop = threading.Event()
    def _scaler_loop():
        while not _scaler_stop.is_set():
            _scaler_stop.wait(timeout=60)
            _adaptive_sem.try_scale_up()
    threading.Thread(target=_scaler_loop, daemon=True, name="adaptive-scaler").start()

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
                        print(f"  future error: {e}")
                        errors += 1

    elapsed = time.monotonic() - run_start
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s | processed={processed} errors={errors}")
    print(f"Output: {output_path}")
    print(f"\nNext: python validate_sft_data.py --input {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="SFT v3 asymmetric distillation generator")
    p.add_argument("--questions", required=True, help="JSONL from sft_question_generator.py")
    p.add_argument("--output", default="data/train_v3.jsonl")
    p.add_argument("--model", default="nvidia_nim/moonshotai/kimi-k2.6")
    p.add_argument("--api_base", default=None)
    p.add_argument("--max", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--type", "--category", dest="category_filter", default=None)
    p.add_argument("--skip_type", dest="skip_categories", default=None, help="Comma-separated categories to skip")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min_workers", type=int, default=1)
    p.add_argument("--api_keys", default=None, help="Comma-separated API keys; rotates after 5 rate limits per key. Also reads NVIDIA_NIM_API_KEYS from .env")
    p.add_argument("--rpm", type=float, default=38.0, help="Requests per minute per key (default 38, slightly under NIM free-tier 40)")
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--base_delay", type=float, default=3.0)
    p.add_argument("--watch_commit", action="store_true",
                   help="Commit+push output file every --watch_threshold new lines in a background daemon thread")
    p.add_argument("--watch_threshold", type=int, default=50,
                   help="Lines between auto-commits when --watch_commit is active (default 50)")
    args = p.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

    if args.watch_commit:
        out_abs = Path(args.output).resolve()
        wt = threading.Thread(target=_watcher_thread, args=(out_abs, args.watch_threshold), daemon=True)
        wt.start()

    cli_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()] if args.api_keys else None
    print(f"Generator : {args.model}")
    skip_cats = {c.strip() for c in args.skip_categories.split(",")} if args.skip_categories else set()
    process_questions_v3(
        questions_path=args.questions,
        output_path=args.output,
        model=args.model,
        api_base=args.api_base,
        max_examples=args.max,
        overwrite=args.overwrite,
        category_filter=args.category_filter,
        skip_categories=skip_cats,
        workers=args.workers,
        min_workers=args.min_workers,
        api_keys=cli_keys,
        rpm_per_key=args.rpm,
    )


if __name__ == "__main__":
    main()
