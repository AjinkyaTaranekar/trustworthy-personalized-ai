"""
sft_add_native_tool_examples.py
=================================
Converts a fraction of the existing XML-format tool training examples into
Qwen3 native JSON tool-calling format, then appends them to the dataset.

WHY
---
The inference server now supports two tool-call modes:
  xml    — custom <tool>name(arg='val')</tool>  (trained, default)
  native — Qwen3 <tool_call>{"name":…,"arguments":{…}}</tool_call>

Without native-format training examples the model falls back entirely on its
pre-training for native calls.  Adding SFT examples in the native format
reinforces that capability and ensures inference/training are in sync:
both call apply_chat_template(tools=[…]) so the rendered prompt text is
byte-for-byte identical.

WHAT IT PRODUCES
----------------
For each selected XML example the script:
  1. Parses the <tool>…</tool> call to extract tool name + arguments.
  2. Re-executes python_execute calls to get the real output.
  3. Rebuilds the message list using OpenAI tool_calls / tool_call_id fields.
  4. Stores the active tool schemas in metadata["native_tools"] so that
     messages_to_text() in 2_model_trainer.py passes them to apply_chat_template.

Usage
-----
    python pipeline/sft_add_native_tool_examples.py \\
        --input  pipeline/data/train_sft_v3.jsonl \\
        --output pipeline/data/train_sft_native.jsonl \\
        --fraction 0.20

    # Then rebuild the robust dataset:
    python pipeline/sft_add_robustness_variants.py \\
        --input  pipeline/data/train_sft_v3_with_native.jsonl \\
        --output pipeline/data/train_sft_v3_robust.jsonl
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Tool schemas — OpenAI function-calling format, mirrors 3_infererence.py registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = {
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

_OPENAI_SCHEMAS = [{"type": "function", "function": s} for s in TOOL_SCHEMAS.values()]

# ---------------------------------------------------------------------------
# Argument extraction from XML tool call string
# ---------------------------------------------------------------------------

_PYTHON_TRIPLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)', re.DOTALL)
_PYTHON_SINGLE = re.compile(r"python_execute\s*\(\s*code\s*=\s*'(.*?)'\s*\)", re.DOTALL)
_PYTHON_DOUBLE = re.compile(r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)', re.DOTALL)
_WEB_QUERY     = re.compile(r"web_search\s*\(\s*query\s*=\s*['\"](.+?)['\"]", re.DOTALL)
_READ_URL_URL  = re.compile(r"read_url\s*\(\s*url\s*=\s*['\"](.+?)['\"]", re.DOTALL)
_READ_URL_PRMT = re.compile(r"prompt\s*=\s*['\"](.+?)['\"]", re.DOTALL)

_ALLOWED_IMPORTS  = frozenset({"math","statistics","decimal","fractions","cmath",
                                "random","itertools","functools","operator","collections",
                                "numbers","string","re"})
_BLOCKED_BUILTINS = frozenset({"exec","eval","compile","__import__","open","breakpoint"})


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
        return f"Error: {reason}"
    try:
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=15)
        out = (proc.stdout or proc.stderr).strip()
        return out if out else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"
    except Exception as e:
        return f"Error: {e}"


def _extract_xml_tool(tool_inner: str) -> tuple[str, dict, str] | None:
    """Return (tool_name, kwargs_dict, tool_result_str) or None if unparseable."""
    s = tool_inner.strip()

    if s.startswith("python_execute"):
        code = None
        for pat in (_PYTHON_TRIPLE, _PYTHON_SINGLE, _PYTHON_DOUBLE):
            m = pat.search(s)
            if m:
                code = m.group(1).replace("\\n", "\n").replace("\\t", "\t")
                break
        if code is None:
            return None
        result = _run_python(code)
        return "python_execute", {"code": code}, result

    if s.startswith("web_search"):
        m = _WEB_QUERY.search(s)
        query = m.group(1) if m else ""
        return "web_search", {"query": query}, (
            f"[Search results for: {query}]\n"
            "(Original training example — live result not reproducible. "
            "Model answer reflects the retrieved content.)"
        )

    if s.startswith("read_url"):
        mu = _READ_URL_URL.search(s)
        mp = _READ_URL_PRMT.search(s)
        url    = mu.group(1) if mu else ""
        prompt = mp.group(1) if mp else ""
        return "read_url", {"url": url, "prompt": prompt}, (
            f"[Page content from: {url}]\n"
            "(Original training example — live content not reproducible.)"
        )

    if s.startswith("get_datetime"):
        from datetime import datetime, timezone
        return "get_datetime", {}, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return None


# ---------------------------------------------------------------------------
# Message-level conversion
# ---------------------------------------------------------------------------

_TOOL_XML_RE  = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
_THINK_END_RE = re.compile(r'</think\s*>', re.IGNORECASE)
_ANSWER_RE    = re.compile(r'(<answer>.*?</answer>)', re.DOTALL | re.IGNORECASE)
_MAX_RESULT   = 3000


def _wrap(name: str, result: str) -> str:
    if len(result) > _MAX_RESULT:
        result = result[:_MAX_RESULT] + " … [truncated]"
    return f"[TOOL_RESULT: {name}]\n{result}\n[/TOOL_RESULT]"


def convert_to_native(messages: list[dict]) -> list[dict] | None:
    """
    Convert a message list from XML tool format to native JSON tool-call format.

    XML format (current training):
      [assistant] <think>…</think><tool>python_execute(code='…')</tool>
      [tool]      [TOOL_RESULT: python_execute]\n…\n[/TOOL_RESULT]
      [assistant] <answer>…</answer>

    Native format (target):
      [assistant] <think>…</think>          (content="", tool_calls=[{…}])
      [tool]      <tool result string>       (tool_call_id="call_xxx")
      [assistant] <answer>…</answer>

    Returns None if the example has no tool calls or cannot be parsed.
    """
    new_msgs: list[dict] = []
    has_tool_call = False

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role != "assistant":
            # Pass system / user messages through unchanged; skip old tool turns
            if role != "tool":
                new_msgs.append(msg)
            i += 1
            continue

        content = msg["content"]
        think_end = 0
        tm = _THINK_END_RE.search(content)
        if tm:
            think_end = tm.end()

        post_think = content[think_end:]
        tool_spans = list(_TOOL_XML_RE.finditer(post_think))

        if not tool_spans:
            # Regular assistant turn with no tool call — keep as-is
            new_msgs.append(msg)
            i += 1
            continue

        # Extract think prefix (may be empty if no <think> block)
        think_prefix = content[:think_end].rstrip()

        for span_idx, span in enumerate(tool_spans):
            tool_inner = span.group(1)
            parsed = _extract_xml_tool(tool_inner)
            if parsed is None:
                return None   # Unparseable — skip the whole example

            tool_name, kwargs, result = parsed
            call_id = f"call_{uuid.uuid4().hex[:8]}"

            if span_idx == 0:
                # First assistant turn carries the think prefix
                new_msgs.append({
                    "role": "assistant",
                    "content": think_prefix,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(kwargs),
                        },
                    }],
                })
            else:
                new_msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(kwargs),
                        },
                    }],
                })

            new_msgs.append({
                "role":         "tool",
                "content":      result,
                "tool_call_id": call_id,
            })
            has_tool_call = True

        # Final answer turn (next assistant message, or <answer> at end of this content)
        answer_m = _ANSWER_RE.search(post_think, tool_spans[-1].end())
        if answer_m:
            new_msgs.append({"role": "assistant", "content": answer_m.group(1).strip()})
        elif i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            new_msgs.append(messages[i + 1])
            i += 1   # skip the next message — already consumed

        i += 1

    return new_msgs if has_tool_call else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert XML tool examples to native JSON tool-calling format"
    )
    parser.add_argument("--input",    default="pipeline/data/train_sft_v3.jsonl")
    parser.add_argument("--output",   default="pipeline/data/train_sft_native.jsonl")
    parser.add_argument("--fraction", type=float, default=0.20,
                        help="Fraction of XML tool examples to convert (default: 0.20)")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    import random
    rng = random.Random(args.seed)

    in_path  = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect examples that have XML tool calls (candidates for conversion)
    candidates, non_tool = [], []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            has_tool = any(m["role"] == "tool" for m in ex.get("messages", []))
            if has_tool:
                candidates.append(ex)
            else:
                non_tool.append(ex)

    rng.shuffle(candidates)
    n_convert = max(1, int(len(candidates) * args.fraction))
    to_convert = candidates[:n_convert]

    converted, failed = [], 0
    for ex in to_convert:
        new_msgs = convert_to_native(ex["messages"])
        if new_msgs is None:
            failed += 1
            continue
        new_ex = dict(ex)
        new_ex["messages"] = new_msgs
        meta = dict(new_ex.get("metadata", {}))
        meta["tool_format"]  = "native"
        meta["native_tools"] = _OPENAI_SCHEMAS   # passed to apply_chat_template by messages_to_text
        new_ex["metadata"] = meta
        converted.append(new_ex)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in converted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"XML tool examples  : {len(candidates)}")
    print(f"Selected to convert: {n_convert}  ({args.fraction*100:.0f}%)")
    print(f"Converted OK       : {len(converted)}")
    print(f"Failed (skipped)   : {failed}")
    print(f"Output             : {out_path}")
    print()
    print("Next: merge native examples into your main dataset, then rebuild robust set:")
    print(f"  cat {args.input} {out_path} > pipeline/data/train_sft_v3_with_native.jsonl")
    print("  python pipeline/sft_add_robustness_variants.py \\")
    print("      --input pipeline/data/train_sft_v3_with_native.jsonl \\")
    print("      --output pipeline/data/train_sft_v3_robust.jsonl")


if __name__ == "__main__":
    main()
