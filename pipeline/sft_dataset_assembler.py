"""
SFT Dataset Assembler (v3)
==========================
Single-command pipeline: load → filter → dedupe → balance →
transform (v2→v3 multi-turn) → add native tool examples →
add robustness variants → split → write.

Replaces four separate scripts:
  sft_dataset_assembler.py  (old, wrote train_sft_v2.jsonl)
  transform_sft_tool_format.py
  sft_add_native_tool_examples.py
  sft_add_robustness_variants.py

Outputs:
  data/train_sft_v3.jsonl  — training set (final)
  data/eval_sft_v3.jsonl          — eval set (v3 format, no variants)
  data/sft_stats.json             — dataset statistics

Usage:
    python sft_dataset_assembler.py
    python sft_dataset_assembler.py \\
        --part_a data/train_partA.jsonl \\
        --part_b data/train_partB.jsonl \\
        --output_dir data/

    # Skip individual stages:
    python sft_dataset_assembler.py --no_native --no_robustness
"""

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_RESPONSE_CHARS  = 80
CAPABILITY_REQUIRED = False  # v3 data uses narrative think blocks — no CAPABILITY_CHECK header
MAX_PER_CATEGORY    = 400
MAX_TOOL_TURNS      = 5       # examples with more tool calls are likely malformed
MAX_RESULT_CHARS    = 3000    # mirror 3_infererence.py _MAX_TOOL_OUTPUT

# ---------------------------------------------------------------------------
# Code safety validator (shared by transform + native conversion stages)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})
_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


def _validate_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"blocked_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS:
                return False, f"blocked_import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                return False, f"blocked_builtin: {node.func.id}"
    return True, "ok"


def _run_python(code: str) -> str:
    safe, reason = _validate_code(code)
    if not safe:
        return f"Error: code rejected by safety validator ({reason})"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        out = (proc.stdout or proc.stderr).strip()
        return out if out else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (15 s limit)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Stage 1: Quality filter
# ---------------------------------------------------------------------------

def passes_quality_filter(example: dict) -> tuple[bool, str]:
    """Return (passes, reason).  Handles both v2 (single-turn) and v3 (multi-turn)."""
    messages = example.get("messages", [])
    if len(messages) < 3:
        return False, "too_few_messages"

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False, "no_assistant_message"
    if not any(m.get("role") == "system" for m in messages):
        return False, "no_system_message"

    # Length + CAPABILITY_CHECK: first assistant message (has the think block in both formats)
    first = assistant_msgs[0].get("content", "")
    last  = assistant_msgs[-1].get("content", "")

    if len(first) < MIN_RESPONSE_CHARS:
        return False, f"response_too_short_{len(first)}"

    if "<answer>" not in last and "</answer>" not in last:
        return False, "missing_tag_answer"

    if CAPABILITY_REQUIRED:
        think_s = first.find("<think>")
        think_e = first.find("</think>")
        if think_s != -1 and think_e != -1:
            think_block = first[think_s:think_e]
            if "CAPABILITY_CHECK" not in think_block:
                return False, "missing_capability_check"
            category = example.get("metadata", {}).get("category", "")
            if category == "appraisal_empathy" and "<appraisal>" not in think_block:
                return False, "missing_appraisal_block"

    return True, "ok"


# ---------------------------------------------------------------------------
# Stage 2: Deduplication
# ---------------------------------------------------------------------------

def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    result, n_removed = [], 0
    for ex in examples:
        msgs = ex.get("messages", [])
        user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
        key = user_msgs[0].strip() if user_msgs else ""
        if not key or key in seen:
            n_removed += 1
            continue
        seen.add(key)
        result.append(ex)
    return result, n_removed


# ---------------------------------------------------------------------------
# Stage 3: Category balancing
# ---------------------------------------------------------------------------

def balance_categories(examples: list[dict], max_per: int = MAX_PER_CATEGORY) -> list[dict]:
    by_cat: dict[str, list] = defaultdict(list)
    for ex in examples:
        meta = ex.get("metadata", {})
        cat = meta.get("category") or meta.get("question_type", "unknown")
        by_cat[cat].append(ex)
    result = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        kept = items[:max_per]
        result.extend(kept)
        if len(items) > max_per:
            print(f"  Capped {cat}: {len(items)} → {max_per}")
    return result


# ---------------------------------------------------------------------------
# Stage 4: Transform v2 → v3 (single-turn tool calls → multi-turn)
# ---------------------------------------------------------------------------

_TOOL_RE      = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
_THINK_END_RE = re.compile(r'</think\s*>', re.IGNORECASE)
_ANSWER_RE    = re.compile(r'(<answer>.*?</answer>)', re.DOTALL | re.IGNORECASE)

_PY_TRIPLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)', re.DOTALL)
_PY_SINGLE = re.compile(r"python_execute\s*\(\s*code\s*=\s*'(.*?)'\s*\)", re.DOTALL)
_PY_DOUBLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)', re.DOTALL)
_WEB_Q_RE  = re.compile(r"web_search\s*\(\s*query\s*=\s*['\"](.+?)['\"]", re.DOTALL)
_URL_RE    = re.compile(r"read_url\s*\(\s*url\s*=\s*['\"](.+?)['\"]", re.DOTALL)


def _tool_result(tool_inner: str) -> tuple[str, str]:
    """Execute or simulate a tool call; return (tool_name, result_string)."""
    s = tool_inner.strip()

    if s.startswith("python_execute"):
        code = None
        for pat in (_PY_TRIPLE, _PY_SINGLE, _PY_DOUBLE):
            m = pat.search(s)
            if m:
                code = m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
                break
        return "python_execute", (_run_python(code.strip()) if code else "Error: could not parse code")

    if s.startswith("web_search"):
        m = _WEB_Q_RE.search(s)
        query = m.group(1) if m else "(query not parsed)"
        return "web_search", (
            f"[Search results for: {query}]\n"
            "(Original training example — live result not available for replay. "
            "Model answer reflects the retrieved content.)"
        )

    if s.startswith("read_url"):
        m = _URL_RE.search(s)
        url = m.group(1) if m else "(URL not parsed)"
        return "read_url", (
            f"[Page content from: {url}]\n"
            "(Original training example — live content not available for replay.)"
        )

    if s.startswith("get_datetime"):
        return "get_datetime", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    name = s.split("(")[0].strip() if "(" in s else s[:40]
    return name, "(Tool result not available — synthetic placeholder)"


def _wrap_result(tool_name: str, result: str) -> str:
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + " … [truncated]"
    return f"[TOOL_RESULT: {tool_name}]\n{result}\n[/TOOL_RESULT]"


def _split_assistant_message(content: str) -> list[dict] | None:
    """Split a v2 single-turn assistant message into multi-turn [asst→tool→…→asst] turns.
    Returns None if the message has no tool calls outside <think>."""
    think_end = 0
    tm = _THINK_END_RE.search(content)
    if tm:
        think_end = tm.end()

    post = content[think_end:]
    spans = list(_TOOL_RE.finditer(post))
    if not spans:
        return None

    new_msgs: list[dict] = []
    for i, span in enumerate(spans):
        tool_name, result = _tool_result(span.group(1))
        full_tag = span.group(0)
        if i == 0:
            prefix = content[:think_end + span.start()].rstrip()
            asst_text = f"{prefix}\n{full_tag}".strip() if prefix else full_tag
        else:
            asst_text = full_tag
        new_msgs.append({"role": "assistant", "content": asst_text})
        new_msgs.append({"role": "tool",      "content": _wrap_result(tool_name, result)})

    answer_m = _ANSWER_RE.search(post, spans[-1].end())
    if answer_m:
        new_msgs.append({"role": "assistant", "content": answer_m.group(1).strip()})

    return new_msgs


def transform_to_v3(example: dict) -> tuple[dict, str]:
    """Transform one example from v2 to v3.  Returns (example, status).

    Status: 'transformed' | 'unchanged' | 'dropped_no_answer' | 'dropped_too_many'
    """
    messages = example.get("messages", [])
    if any(m["role"] == "tool" for m in messages):
        return example, "unchanged"   # already v3

    new_messages: list[dict] = []
    did_transform = False
    for msg in messages:
        if msg["role"] == "assistant":
            replacement = _split_assistant_message(msg["content"])
            if replacement is not None:
                new_messages.extend(replacement)
                did_transform = True
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    if not did_transform:
        return example, "unchanged"

    roles = [m["role"] for m in new_messages]
    if roles[-1] == "tool":
        return example, "dropped_no_answer"
    if roles.count("tool") > MAX_TOOL_TURNS:
        return example, "dropped_too_many"

    out = dict(example)
    out["messages"] = new_messages
    meta = dict(out.get("metadata", {}))
    meta["tool_format"] = "v3_multi_turn"
    out["metadata"] = meta
    return out, "transformed"


# ---------------------------------------------------------------------------
# Stage 5: Native JSON tool examples
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict] = {
    "python_execute": {
        "name": "python_execute",
        "description": "Execute Python code and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code to run"}},
            "required": ["code"],
        },
    },
    "get_datetime": {
        "name": "get_datetime",
        "description": "Return the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web for a query and return a summary.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query string"}},
            "required": ["query"],
        },
    },
    "read_url": {
        "name": "read_url",
        "description": "Fetch the text content of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url":    {"type": "string", "description": "URL to fetch"},
                "prompt": {"type": "string", "description": "What to extract from the page"},
            },
            "required": ["url"],
        },
    },
}

OPENAI_SCHEMAS = [{"type": "function", "function": s} for s in TOOL_SCHEMAS.values()]

_TOOL_ARG_MAP = {
    "python_execute": lambda s: {"code": next(
        (m.group(1).replace("\\n", "\n").replace("\\t", "\t") for pat in (_PY_TRIPLE, _PY_SINGLE, _PY_DOUBLE)
         if (m := pat.search(s))), None
    )},
    "web_search": lambda s: {"query": (m.group(1) if (m := _WEB_Q_RE.search(s)) else "")},
    "read_url":   lambda s: {
        "url":    (m.group(1) if (m := _URL_RE.search(s)) else ""),
        "prompt": (m.group(1) if (m := re.search(r"prompt\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)) else ""),
    },
    "get_datetime": lambda _: {},
}


def _xml_tool_to_native(tool_inner: str) -> tuple[str, dict, str] | None:
    """Parse XML tool call string → (tool_name, kwargs, result).  Returns None if unparseable."""
    s = tool_inner.strip()
    for name, extractor in _TOOL_ARG_MAP.items():
        if s.startswith(name):
            kwargs = extractor(s)
            if None in (kwargs.values()):
                return None   # code extraction failed
            result = _tool_result(s)[1]
            return name, kwargs, result
    return None


def convert_to_native(messages: list[dict]) -> list[dict] | None:
    """Convert a v3 XML multi-turn conversation to native JSON tool_calls format.
    Returns None if unparseable or no tool calls found."""
    new_msgs: list[dict] = []
    has_tool = False
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] != "assistant":
            if msg["role"] != "tool":   # drop old tool turns; rebuilt below
                new_msgs.append(msg)
            i += 1
            continue

        content = msg["content"] or ""
        think_end = 0
        tm = _THINK_END_RE.search(content)
        if tm:
            think_end = tm.end()

        post = content[think_end:]
        spans = list(_TOOL_RE.finditer(post))
        if not spans:
            new_msgs.append(msg)
            i += 1
            continue

        think_prefix = content[:think_end].rstrip()
        for span_idx, span in enumerate(spans):
            parsed = _xml_tool_to_native(span.group(1))
            if parsed is None:
                return None
            tool_name, kwargs, result = parsed
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            new_msgs.append({
                "role":       "assistant",
                "content":    think_prefix if span_idx == 0 else "",
                "tool_calls": [{
                    "id":       call_id,
                    "type":     "function",
                    "function": {"name": tool_name, "arguments": json.dumps(kwargs)},
                }],
            })
            new_msgs.append({"role": "tool", "content": result, "tool_call_id": call_id})
            has_tool = True
            think_prefix = ""

        answer_m = _ANSWER_RE.search(post, spans[-1].end())
        if answer_m:
            new_msgs.append({"role": "assistant", "content": answer_m.group(1).strip()})
        elif i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            new_msgs.append(messages[i + 1])
            i += 1
        i += 1

    return new_msgs if has_tool else None


def make_native_examples(examples: list[dict], fraction: float, rng: random.Random) -> list[dict]:
    """Convert `fraction` of tool examples to native JSON format."""
    tool_exs = [ex for ex in examples if any(m["role"] == "tool" for m in ex.get("messages", []))]
    rng.shuffle(tool_exs)
    n = max(1, int(len(tool_exs) * fraction))
    converted, failed = [], 0
    for ex in tool_exs[:n]:
        new_msgs = convert_to_native(ex["messages"])
        if new_msgs is None:
            failed += 1
            continue
        new_ex = dict(ex)
        new_ex["messages"] = new_msgs
        meta = dict(new_ex.get("metadata", {}))
        meta["tool_format"]  = "native"
        meta["native_tools"] = OPENAI_SCHEMAS
        new_ex["metadata"] = meta
        converted.append(new_ex)
    print(f"  Native examples: {len(converted)} converted, {failed} skipped")
    return converted


# ---------------------------------------------------------------------------
# Stage 6: Robustness variants (minimal/brief/no-principles system prompts)
# ---------------------------------------------------------------------------

_TOOL_CONTEXTS = {
    "all_tools":          "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓",
    "compute_only":       "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✗",
    "compute_and_search": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗",
    "no_tools":           "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✗",
}
_TOOL_NOTES = {
    "all_tools":          "All four tools are available in this session.",
    "compute_only":       "Only python_execute is available. No internet or time access.",
    "compute_and_search": "python_execute and web_search/read_url are available. No datetime tool.",
    "no_tools":           "No tools are available in this session. Training knowledge only.",
}


def _minimal_prompt(profile: str) -> str:
    ctx = _TOOL_CONTEXTS.get(profile, "")
    if ctx and "✓" in ctx:
        return f"You are a trustworthy AI assistant.\nAvailable tools in this session: {ctx}"
    return "You are a trustworthy AI assistant."


def _brief_prompt(profile: str) -> str:
    ctx = _TOOL_CONTEXTS.get(profile, "")
    tool_line = f"\nSession tools: {ctx}" if ctx else ""
    return (
        "You are a trustworthy AI assistant. Think carefully before every "
        "response: identify what the question requires, which tools (if any) "
        "are needed, what the stakes are if you are wrong, and what you cannot "
        f"know.{tool_line}"
    )


def _no_principles_prompt(profile: str) -> str:
    ctx  = _TOOL_CONTEXTS.get(profile, "python_execute ✓")
    note = _TOOL_NOTES.get(profile, "")
    has  = {t: f"{t} ✓" in ctx for t in ("python_execute", "web_search", "read_url", "get_datetime")}
    call_lines = []
    if has["python_execute"]: call_lines.append("  <tool>python_execute(code='...')</tool>")
    if has["web_search"]:     call_lines.append("  <tool>web_search(query='...')</tool>")
    if has["read_url"]:       call_lines.append("  <tool>read_url(url='...', prompt='what to extract')</tool>")
    if has["get_datetime"]:   call_lines.append("  <tool>get_datetime()</tool>")
    calls = "\n".join(call_lines) if call_lines else "  (no tools available this session)"
    return (
        "You are a trustworthy AI assistant. Before answering any question, complete a full "
        "CAPABILITY_CHECK inside your <think> block using this exact structure:\n\n"
        "<think>\nCAPABILITY_CHECK:\n\n"
        "  5W+H:\n    WHO is affected: [the user / third parties / institutions involved]\n"
        "    WHAT is required: [list requirements to answer correctly]\n"
        "    WHEN: [time-sensitivity — live data needed, training cutoff relevant, dated context]\n"
        "    WHERE: [jurisdiction, region, domain, platform]\n"
        "    WHY: [inferred intent and underlying goal]\n"
        "    HOW: [tool selection and method]\n\n"
        "  First Principles:\n    Core truth: [the irreducible fact this answer rests on]\n"
        "    Assumptions: [what I am taking for granted — flag if unverified]\n\n"
        f"  Session tools: {ctx}\n"
        "  Gap: [what I cannot obtain]\n  Strategy: [tool chain plan or honest refusal]\n\n"
        "  CONSEQUENCE_CHECK:\n    Stakes: [low / medium / high + reason]\n"
        "    If wrong: [concrete harm to the user]\n"
        "    User will likely: [action they will take with this answer]\n"
        "    Accountability: [what to hedge or flag in the answer]\n"
        "</think>\n<answer>\n[response to the user]\n</answer>\n\n"
        f"{note}\n\nTool call syntax:\n{calls}"
    )


_VARIANT_BUILDERS = {
    "minimal":       _minimal_prompt,
    "brief":         _brief_prompt,
    "no_principles": _no_principles_prompt,
}


def make_variant(example: dict, level: str) -> dict:
    profile    = example.get("metadata", {}).get("tool_profile", "all_tools")
    new_system = _VARIANT_BUILDERS[level](profile)
    new_msgs   = [
        ({"role": "system", "content": new_system} if m["role"] == "system" else m)
        for m in example.get("messages", [])
    ]
    out = dict(example)
    out["messages"] = new_msgs
    meta = dict(out.get("metadata", {}))
    meta["robustness_variant"] = level
    out["metadata"] = meta
    return out


def make_robustness_variants(
    examples: list[dict],
    fractions: dict[str, float],
    rng: random.Random,
) -> list[dict]:
    variants, counts = [], {}
    for level, frac in fractions.items():
        added = 0
        for ex in examples:
            if rng.random() < frac:
                variants.append(make_variant(ex, level))
                added += 1
        counts[level] = added
    print(f"  Robustness variants: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return variants


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(examples: list[dict]) -> dict:
    cat_counts, src_counts, pipe_counts = Counter(), Counter(), Counter()
    revised = tool_turns = native = 0
    total_chars = 0
    for ex in examples:
        meta = ex.get("metadata", {})
        cat  = meta.get("category") or meta.get("question_type", "unknown")
        cat_counts[cat] += 1
        src_counts[meta.get("source", "unknown")] += 1
        pipe_counts[meta.get("pipeline", "unknown")] += 1
        if meta.get("revised"):           revised    += 1
        if meta.get("tool_format") == "native": native += 1
        msgs = ex.get("messages", [])
        if any(m["role"] == "tool" for m in msgs): tool_turns += 1
        asst = " ".join(m.get("content", "") or "" for m in msgs if m["role"] == "assistant")
        total_chars += len(asst)
    n = max(len(examples), 1)
    return {
        "total": len(examples),
        "by_category": dict(cat_counts.most_common()),
        "by_source": dict(src_counts),
        "by_pipeline": dict(pipe_counts),
        "revised_count": revised,
        "tool_turn_examples": tool_turns,
        "native_tool_examples": native,
        "avg_response_chars": total_chars // n,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    part_a:         str,
    part_b:         str,
    output_dir:     str,
    eval_frac:      float = 0.10,
    max_per_cat:    int   = MAX_PER_CATEGORY,
    seed:           int   = 42,
    no_transform:   bool  = False,
    no_native:      bool  = False,
    native_fraction: float = 0.20,
    no_robustness:  bool  = False,
    minimal_frac:   float = 0.15,
    brief_frac:     float = 0.10,
    no_principles_frac: float = 0.05,
) -> None:
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    raw: list[dict] = []
    for fpath, label in [(part_a, "Part A"), (part_b, "Part B")]:
        p = Path(fpath)
        if not p.exists():
            print(f"  Warning: {fpath} not found — skipping {label}")
            continue
        with open(p, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        print(f"  Loaded {len(items):>5} from {label} ({fpath})")
        raw.extend(items)
    print(f"  Total loaded      : {len(raw)}")

    # ── Quality filter ───────────────────────────────────────────────────────
    passed, reasons = [], Counter()
    for ex in raw:
        ok, reason = passes_quality_filter(ex)
        if ok:
            passed.append(ex)
        else:
            reasons[reason] += 1
    print(f"  After quality filter: {len(passed)}  (dropped {len(raw)-len(passed)})")
    if reasons:
        for r, n in reasons.most_common(5):
            print(f"    {r}: {n}")

    # ── Dedup + balance ──────────────────────────────────────────────────────
    deduped, n_dup = deduplicate(passed)
    print(f"  After dedup       : {len(deduped)}  ({n_dup} duplicates removed)")
    balanced = balance_categories(deduped, max_per_cat)
    print(f"  After balancing   : {len(balanced)}")

    # ── Train / eval split (before any augmentation) ─────────────────────────
    rng.shuffle(balanced)
    n_eval  = max(1, int(len(balanced) * eval_frac))
    eval_v2 = balanced[:n_eval]
    train   = balanced[n_eval:]
    print(f"  Split             : {len(train)} train / {n_eval} eval")

    # ── Transform v2 → v3 (applied to both train and eval) ──────────────────
    if not no_transform:
        print("\n[Stage 4] Transforming v2 → v3 (multi-turn tool calls)...")
        t_counts: dict[str, int] = {}
        train_v3 = []
        for ex in train:
            out_ex, status = transform_to_v3(ex)
            t_counts[status] = t_counts.get(status, 0) + 1
            if status not in ("dropped_no_answer", "dropped_too_many"):
                train_v3.append(out_ex)
        eval_v3 = []
        for ex in eval_v2:
            out_ex, status = transform_to_v3(ex)
            if status not in ("dropped_no_answer", "dropped_too_many"):
                eval_v3.append(out_ex)
        print(f"  Transform results : {t_counts}")
        print(f"  Train after xform : {len(train_v3)}  |  Eval: {len(eval_v3)}")
        train, eval_set = train_v3, eval_v3
    else:
        eval_set = eval_v2

    # ── Native tool examples (train only) ────────────────────────────────────
    if not no_native:
        print(f"\n[Stage 5] Adding native JSON tool examples (fraction={native_fraction})...")
        native_exs = make_native_examples(train, native_fraction, rng)
        train = train + native_exs
        print(f"  Train after native: {len(train)}")

    # ── Robustness variants (train only) ─────────────────────────────────────
    if not no_robustness:
        fracs = {"minimal": minimal_frac, "brief": brief_frac, "no_principles": no_principles_frac}
        print(f"\n[Stage 6] Adding robustness variants {fracs}...")
        variants = make_robustness_variants(train, fracs, rng)
        train = train + variants
        print(f"  Train after robust: {len(train)}")

    # ── Final shuffle + write ────────────────────────────────────────────────
    rng.shuffle(train)
    rng.shuffle(eval_set)

    train_path = out / "train_sft_v3.jsonl"
    eval_path  = out / "eval_sft_v3.jsonl"
    stats_path = out / "sft_stats.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {"train": compute_stats(train), "eval": compute_stats(eval_set)}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"Train : {len(train):>5} examples  → {train_path}")
    print(f"Eval  : {len(eval_set):>5} examples  → {eval_path}")
    print(f"Stats : {stats_path}")
    print(f"\nCategory distribution (train):")
    for cat, count in stats["train"]["by_category"].items():
        print(f"  {cat:<38} {count:>5}  ({100*count/max(len(train),1):.1f}%)")
    print(f"\nTool-turn examples : {stats['train']['tool_turn_examples']}")
    print(f"Native examples    : {stats['train']['native_tool_examples']}")
    print(f"\nNext step → train:")
    print(f"  python 2_model_trainer.py --mode sft --data_dir {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT dataset assembler v3 — load, filter, transform, augment, split"
    )
    parser.add_argument("--part_a",      default="data/train_partA_v3.jsonl")
    parser.add_argument("--part_b",      default="data/train_partB_v3.jsonl")
    parser.add_argument("--output_dir",  default="data")
    parser.add_argument("--eval_frac",   type=float, default=0.10)
    parser.add_argument("--max_per_category", type=int, default=MAX_PER_CATEGORY)
    parser.add_argument("--seed",        type=int,   default=42)
    # Transform flags
    parser.add_argument("--capability_check", action="store_true",
                        help="Enable CAPABILITY_CHECK filter (for legacy v2 data that uses structured think blocks)")
    parser.add_argument("--no_transform",    action="store_true", help="Skip v2→v3 tool format transform")
    # Native tool flags
    parser.add_argument("--no_native",       action="store_true", help="Skip native JSON tool examples")
    parser.add_argument("--native_fraction", type=float, default=0.20)
    # Robustness flags
    parser.add_argument("--no_robustness",   action="store_true", help="Skip robustness variants")
    parser.add_argument("--minimal",         type=float, default=0.15)
    parser.add_argument("--brief",           type=float, default=0.10)
    parser.add_argument("--no_principles",   type=float, default=0.05)
    args = parser.parse_args()

    if args.capability_check:
        global CAPABILITY_REQUIRED
        CAPABILITY_REQUIRED = True

    print(f"SFT Dataset Assembler v3")
    print(f"  Part A : {args.part_a}")
    print(f"  Part B : {args.part_b}")
    print(f"  Output : {args.output_dir}")
    print(f"  Stages : transform={'off' if args.no_transform else 'on'}  "
          f"native={'off' if args.no_native else f'on ({args.native_fraction:.0%})'}  "
          f"robustness={'off' if args.no_robustness else 'on'}")
    print()

    run(
        part_a=args.part_a,
        part_b=args.part_b,
        output_dir=args.output_dir,
        eval_frac=args.eval_frac,
        max_per_cat=args.max_per_category,
        seed=args.seed,
        no_transform=args.no_transform,
        no_native=args.no_native,
        native_fraction=args.native_fraction,
        no_robustness=args.no_robustness,
        minimal_frac=args.minimal,
        brief_frac=args.brief,
        no_principles_frac=args.no_principles,
    )


if __name__ == "__main__":
    main()
