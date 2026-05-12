"""
transform_sft_tool_format.py
============================
Transforms a single-turn SFT JSONL (v2) into the multi-turn format expected
by 3_infererence.py's tool loop (v3).

Problem:
  The v2 training data encodes tool calls and the final answer in the same
  assistant message:

    [assistant]  <think>...</think>
                 <tool>python_execute(code='...')</tool>
                 <answer>30</answer>

  The inference server expects separate turns with an actual tool result:

    [assistant]  <think>...</think><tool>python_execute(code='...')</tool>
    [tool]       [TOOL_RESULT: python_execute]\\n20\\n[/TOOL_RESULT]
    [assistant]  <answer>30</answer>

Transformation rules:
  python_execute  — code is extracted and re-executed; real stdout used as result.
  web_search      — synthetic placeholder (live results cannot be replayed).
  read_url        — synthetic placeholder.
  get_datetime    — current UTC time stamped at transformation time.
  unknown tool    — generic placeholder.
  no tool calls   — example kept unchanged.
  already v3      — example has a 'tool' role message; skipped.

Multi-tool chains produce one [assistant, tool] pair per <tool> tag.

Usage:
    python pipeline/transform_sft_tool_format.py \\
        --input  pipeline/data/train_sft_v2.jsonl \\
        --output pipeline/data/train_sft_v3.jsonl

    python pipeline/transform_sft_tool_format.py \\
        --input  pipeline/data/train_sft_v2.jsonl \\
        --output pipeline/data/train_sft_v3.jsonl \\
        --dry_run        # parse + count without writing
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# AST-based code safety validator (same allowlist as 3_infererence.py)
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


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_TOOL_RE      = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
_THINK_END_RE = re.compile(r'</think\s*>', re.IGNORECASE)
_ANSWER_RE    = re.compile(r'(<answer>.*?</answer>)', re.DOTALL | re.IGNORECASE)

# python_execute code extraction — triple-quote, then single, then double
_PY_TRIPLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)', re.DOTALL)
_PY_SINGLE = re.compile(r"python_execute\s*\(\s*code\s*=\s*'(.*?)'\s*\)", re.DOTALL)
_PY_DOUBLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)', re.DOTALL)

_WEB_SEARCH_QUERY_RE = re.compile(r"web_search\s*\(\s*query\s*=\s*['\"](.+?)['\"]", re.DOTALL)
_READ_URL_RE         = re.compile(r"read_url\s*\(\s*url\s*=\s*['\"](.+?)['\"]", re.DOTALL)

_MAX_RESULT_CHARS = 3000   # mirrors 3_infererence.py _MAX_TOOL_OUTPUT


# ---------------------------------------------------------------------------
# Code extraction + execution
# ---------------------------------------------------------------------------

def _extract_python_code(tool_inner: str) -> str | None:
    for pat in (_PY_TRIPLE, _PY_SINGLE, _PY_DOUBLE):
        m = pat.search(tool_inner)
        if m:
            code = m.group(1)
            # Unescape sequences introduced by JSON serialisation
            code = (code
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                    .replace('\\"', '"')
                    .replace("\\'", "'"))
            return code.strip()
    return None


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
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Per-tool result generation
# ---------------------------------------------------------------------------

def _tool_result(tool_inner: str) -> tuple[str, str]:
    """Return (tool_name, result_string) for any <tool> inner content."""
    s = tool_inner.strip()

    if s.startswith("python_execute"):
        code = _extract_python_code(s)
        if code is None:
            return "python_execute", "Error: could not parse code from tool call"
        return "python_execute", _run_python(code)

    if s.startswith("web_search"):
        m = _WEB_SEARCH_QUERY_RE.search(s)
        query = m.group(1) if m else "(query not parsed)"
        return "web_search", (
            f"[Search results for: {query}]\n"
            "(Original training example — live result not available for replay. "
            "Model answer reflects the retrieved content.)"
        )

    if s.startswith("read_url"):
        m = _READ_URL_RE.search(s)
        url = m.group(1) if m else "(URL not parsed)"
        return "read_url", (
            f"[Page content from: {url}]\n"
            "(Original training example — live content not available for replay. "
            "Model answer reflects the retrieved content.)"
        )

    if s.startswith("get_datetime"):
        return "get_datetime", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Unknown tool — preserve structure with a neutral placeholder
    name = s.split("(")[0].strip() if "(" in s else s[:40]
    return name, "(Tool result not available — synthetic placeholder for training)"


def _wrap_result(tool_name: str, result: str) -> str:
    if len(result) > _MAX_RESULT_CHARS:
        result = result[:_MAX_RESULT_CHARS] + " … [truncated]"
    return f"[TOOL_RESULT: {tool_name}]\n{result}\n[/TOOL_RESULT]"


# ---------------------------------------------------------------------------
# Core: split one assistant message into multi-turn messages
# ---------------------------------------------------------------------------

def _split_assistant_message(content: str) -> list[dict] | None:
    """
    Split an assistant message that contains <tool> tags (after </think>) into
    a sequence of [assistant, tool, …, assistant(<answer>)] message dicts.

    Returns None if the message contains no tool calls outside <think>.
    """
    # Find where <think> ends so we only match tool calls in the outer scope
    think_end = 0
    tm = _THINK_END_RE.search(content)
    if tm:
        think_end = tm.end()

    post_think = content[think_end:]
    tool_spans = list(_TOOL_RE.finditer(post_think))

    if not tool_spans:
        return None

    new_msgs: list[dict] = []

    for i, span in enumerate(tool_spans):
        tool_inner = span.group(1)
        full_tag   = span.group(0)          # <tool>...</tool>
        tool_name, result = _tool_result(tool_inner)

        if i == 0:
            # First assistant turn: <think> block + first <tool> tag
            prefix = content[:think_end + span.start()].rstrip()
            assistant_text = (f"{prefix}\n{full_tag}").strip() if prefix else full_tag
        else:
            # Subsequent assistant turns: just the <tool> tag
            assistant_text = full_tag

        new_msgs.append({"role": "assistant", "content": assistant_text})
        new_msgs.append({"role": "tool",      "content": _wrap_result(tool_name, result)})

    # Final assistant turn: the <answer> block (may be empty on malformed examples)
    answer_m = _ANSWER_RE.search(post_think, tool_spans[-1].end())
    answer_text = answer_m.group(1).strip() if answer_m else ""
    if answer_text:
        new_msgs.append({"role": "assistant", "content": answer_text})

    return new_msgs


# ---------------------------------------------------------------------------
# Per-example transformation
# ---------------------------------------------------------------------------

_MAX_TOOL_TURNS = 5  # examples with more tool calls than this are likely malformed


def transform_example(example: dict) -> tuple[dict, str]:
    """
    Returns (transformed_example, status).

    Status values:
      'transformed'        — at least one assistant turn was split into multi-turn
      'unchanged'          — no tool calls found; example kept as-is
      'skipped'            — example already has 'tool' role messages
      'dropped_no_answer'  — transformed result ends with a tool turn (malformed source)
      'dropped_too_many'   — transformed result has > _MAX_TOOL_TURNS tool calls (malformed source)
    """
    messages = example.get("messages", [])

    if any(m["role"] == "tool" for m in messages):
        return example, "skipped"

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

    # Sanity checks on the transformed result
    roles = [m["role"] for m in new_messages]
    if roles[-1] == "tool":
        return example, "dropped_no_answer"
    if roles.count("tool") > _MAX_TOOL_TURNS:
        return example, "dropped_too_many"

    out = dict(example)
    out["messages"] = new_messages
    meta = dict(out.get("metadata", {}))
    meta["tool_format"] = "v3_multi_turn"
    out["metadata"] = meta
    return out, "transformed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform SFT JSONL from single-turn to multi-turn tool call format (v2 → v3)"
    )
    parser.add_argument("--input",   default="pipeline/data/train_sft_v2.jsonl",
                        help="Input JSONL (default: pipeline/data/train_sft_v2.jsonl)")
    parser.add_argument("--output",  default="pipeline/data/train_sft_v3.jsonl",
                        help="Output JSONL (default: pipeline/data/train_sft_v3.jsonl)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and count transforms without writing output")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each transformed example's role sequence")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {
        "transformed": 0, "unchanged": 0, "skipped": 0,
        "dropped_no_answer": 0, "dropped_too_many": 0, "error": 0,
    }
    exec_errors = 0
    t0 = time.monotonic()

    out_file = open(out_path, "w", encoding="utf-8") if not args.dry_run else None

    try:
        with open(in_path, encoding="utf-8") as fin:
            for line_no, raw in enumerate(fin, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    example = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"  [line {line_no}] JSON error: {exc} — skipping")
                    counts["error"] += 1
                    continue

                transformed, status = transform_example(example)
                counts[status] += 1

                # Track execution errors embedded in tool results
                if status == "transformed":
                    for m in transformed["messages"]:
                        if m["role"] == "tool" and m["content"].startswith("[TOOL_RESULT: python_execute]\nError:"):
                            exec_errors += 1

                if args.verbose and status == "transformed":
                    roles = [m["role"] for m in transformed["messages"]]
                    print(f"  [line {line_no}] {' → '.join(roles)}")

                # Dropped examples are not written to output
                if out_file is not None and status not in ("dropped_no_answer", "dropped_too_many"):
                    out_file.write(json.dumps(transformed, ensure_ascii=False) + "\n")

                if line_no % 200 == 0:
                    elapsed = time.monotonic() - t0
                    print(f"  [{line_no}] transformed={counts['transformed']} "
                          f"unchanged={counts['unchanged']} "
                          f"skipped={counts['skipped']} ({elapsed:.0f}s)")
    finally:
        if out_file is not None:
            out_file.close()

    elapsed = time.monotonic() - t0
    total   = sum(counts.values())

    print(f"\n{'='*58}")
    print(f"Done in {elapsed:.1f}s  ({total} examples)")
    kept = counts["transformed"] + counts["unchanged"] + counts["skipped"]
    dropped = counts["dropped_no_answer"] + counts["dropped_too_many"]
    print(f"  Transformed (multi-turn) : {counts['transformed']}")
    print(f"    of which exec errors   : {exec_errors}  (error text used as result)")
    print(f"  Unchanged (no tools)     : {counts['unchanged']}")
    print(f"  Skipped (already v3)     : {counts['skipped']}")
    print(f"  Dropped — no answer      : {counts['dropped_no_answer']}")
    print(f"  Dropped — too many tools : {counts['dropped_too_many']}  (>{_MAX_TOOL_TURNS} calls — likely malformed)")
    print(f"  Parse errors             : {counts['error']}")
    print(f"  Net output examples      : {kept}")
    if not args.dry_run:
        print(f"  Output                   : {out_path}")
    else:
        print("  Dry run — no file written")


if __name__ == "__main__":
    main()
