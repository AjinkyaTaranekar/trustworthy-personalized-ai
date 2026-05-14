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
    python pipeline/sft_v3_generator.py \\
        --questions pipeline/data/questions_partA.jsonl \\
        --output pipeline/data/train_v3.jsonl \\
        --model nvidia_nim/moonshotai/kimi-k2.6 \\
        --critic_model nvidia_nim/minimaxai/minimax-m2.7

    python pipeline/sft_v3_generator.py \\
        --questions pipeline/data/questions_v3.jsonl \\
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
# Student prompts — ≤50 words each (validated by tests)
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
15. Name assumptions explicitly; mark them as unverified if they are not confirmed facts."""

_TEACHER_FORMAT_RULES = """\
CRITICAL FORMAT RULES — violation invalidates the training example:
1. Open with <think> containing flowing narrative reasoning (minimum 150 characters). NO headers, NO rule numbers, NO "CAPABILITY_CHECK:", NO "5W+H:", NO bullet lists inside <think>.
2. Place ALL tool calls after </think> and before <answer> using: <tool>tool_name(arg='...')</tool>
3. Close EVERY response with <answer>...</answer>.
4. NEVER output these phrases: "see answer below", "inferred from question", "none flagged", "CAPABILITY_CHECK:", "PRINCIPLE_", "5W+H:", "CONSEQUENCE_CHECK:".
5. After each [TOOL_RESULT] block, continue reasoning in flowing prose before the next tool call or <answer>."""


def _make_teacher_prompt(tool_profile: dict) -> str:
    return (
        "You are a frontier AI assistant generating exemplary training data.\n\n"
        f"{_TEACHER_CONSTITUTION}\n\n"
        f"{_TEACHER_FORMAT_RULES}\n\n"
        f"Session tools available: {tool_profile['context']}\n"
        f"{tool_profile['system_note']}"
    )


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
        from exa_py import Exa
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


# ---------------------------------------------------------------------------
# Python executor (sandboxed)
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
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if prompt:
            return f"[Fetched: {url}]\nPrompt: {prompt}\nContent: {body[:600]}"
        return f"[Fetched: {url}]\n{body[:600]}"
    except Exception as e:
        return f"read_url failed: {e}"


# ---------------------------------------------------------------------------
# Tool executor with failure injection
# ---------------------------------------------------------------------------

def _execute_tool_v3(
    tool_inner: str,
    active_tools: set[str],
    failure_config: dict | None,
) -> str:
    """Execute a tool call string and return the result.

    failure_config keys:
      inject_503 (bool): inject HTTP 503 on the FIRST web_search call only.
      web_search_count (int): auto-incremented counter.
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

    Uses stop=["</tool>"] so generation halts when a tool call body is emitted.
    The script then executes the tool and resumes generation with the real result.
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

        # Detect tool interception: <tool> present but </tool> absent after it
        tool_pos = content.rfind("<tool>")
        is_tool_call = tool_pos != -1 and "</tool>" not in content[tool_pos:]

        if not is_tool_call:
            conversation.append({"role": "assistant", "content": content})
            break

        # Reconstruct full tool tag (add back the stripped </tool> stop sequence)
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
    """Build a JSONL training row. Teacher system prompt → student prompt (context swap)."""
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

_USER_DRAFT_PROMPT = """\
Generate an exemplary training response for this question.

QUESTION: {question}
CATEGORY: {category}
SESSION TOOLS: {tool_context}

Requirements for this category ({category}):
{ideal_behavior}

Begin your response immediately with <think>. Do NOT output any preamble or headers.\
"""


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
        print(f"  {tag} {len(conversation)} msgs ({n_tool_turns} tool turns) in {time.monotonic()-t0:.1f}s")

        example = _build_v3_example(conversation, question, category, tool_profile)

        with file_lock:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_file.flush()

        print(f"  {tag} written")
        return "ok"

    except Exception as e:
        print(f"  {tag} error: {e}")
        return "error"


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
                        print(f"  future error: {e}")
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
