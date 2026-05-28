"""
patch_xml_to_native.py
======================
Convert SFT training JSONL from XML tool-call format to native Hermes format.

XML format (trained):    <tool>python_execute(code="...")</tool>
Native format (Hermes):  <tool_call>{"name": "python_execute", "arguments": {"code": "..."}}</tool_call>

Why: Qwen3-0.6B already knows Hermes format from pre-training, so the model
only needs to learn WHEN to call tools — not the syntax. This frees capacity
in a 0.6B model for constitutional behaviour learning.

The trainer's messages_to_text() already handles native examples via the
metadata["native_tools"] field — no trainer changes needed.

Usage:
    python pipeline/patch_xml_to_native.py \\
        --input pipeline/data/train_sft_v3.jsonl \\
        --output pipeline/data/train_sft_v3_native.jsonl

    # Or patch multiple source files then combine:
    python pipeline/patch_xml_to_native.py \\
        --input pipeline/data/train_partA_v3.jsonl pipeline/data/train_partB_v3.jsonl \\
        --output pipeline/data/train_sft_v3_native.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# OpenAI-schema definitions — matches what 3_infererence.py registers
# Keep in sync with pipeline_tools.py tool registrations.
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = {
    "python_execute": {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Execute Python code and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to run"},
                },
                "required": ["code"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and return a summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                },
                "required": ["query"],
            },
        },
    },
    "read_url": {
        "type": "function",
        "function": {
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
    },
    "get_datetime": {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the current UTC date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "scratchpad_sections": {
        "type": "function",
        "function": {
            "name": "scratchpad_sections",
            "description": "List existing scratchpad section keys for this session.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "scratchpad_read": {
        "type": "function",
        "function": {
            "name": "scratchpad_read",
            "description": "Read the full scratchpad for this session.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "scratchpad_update": {
        "type": "function",
        "function": {
            "name": "scratchpad_update",
            "description": "Write or update a scratchpad section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "Section key"},
                    "content": {"type": "string", "description": "Section content"},
                },
                "required": ["section", "content"],
            },
        },
    },
    "user_memory_sections": {
        "type": "function",
        "function": {
            "name": "user_memory_sections",
            "description": "List existing user memory section keys.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "user_memory_read": {
        "type": "function",
        "function": {
            "name": "user_memory_read",
            "description": "Retrieve stored facts about this user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What context to retrieve"},
                },
                "required": [],
            },
        },
    },
    "user_memory_update": {
        "type": "function",
        "function": {
            "name": "user_memory_update",
            "description": "Save a fact the user explicitly shared about themselves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "Memory section key"},
                    "content": {"type": "string", "description": "Content to store"},
                },
                "required": ["section", "content"],
            },
        },
    },
}

# Profile → tools that appear in native_tools (always-on tools excluded from
# the schema list since they're described in the system prompt separately)
_PROFILE_SCHEMAS = {
    "all_tools":          ["python_execute", "web_search", "read_url", "get_datetime"],
    "compute_only":       ["python_execute", "get_datetime"],
    "compute_and_search": ["python_execute", "web_search", "read_url", "get_datetime"],
    "no_tools":           ["get_datetime"],
}


def _schemas_for_profile(profile: str):
    names = _PROFILE_SCHEMAS.get(profile, _PROFILE_SCHEMAS["all_tools"])
    return [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]


# ---------------------------------------------------------------------------
# XML tool call parser (mirrors 3_infererence.py:_parse_tool_call)
# ---------------------------------------------------------------------------

def _parse_xml_tool_call(text: str):
    """Parse <tool>name(args)</tool> → {"function": name, "kwargs": dict} or None."""
    m = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not m:
        return None
    inner = m.group(1).strip()
    fm = re.match(r"(\w+)\((.*)\)", inner, re.DOTALL)
    if not fm:
        return None
    name, args_str = fm.group(1), fm.group(2)
    kwargs = {}
    triple_args = set()
    # 1. Extract triple-quoted args first (code="""...""")
    for tm in re.finditer(r'(\w+)="""(.*?)"""', args_str, re.DOTALL):
        kwargs[tm.group(1)] = tm.group(2)
        triple_args.add(tm.group(1))
    # 2. Double-quoted args: key="..." — no backreference, safe from catastrophic backtracking
    for km in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', args_str):
        key = km.group(1)
        if key in triple_args:
            continue
        kwargs[key] = km.group(2)
    # 3. Single-quoted args: key='...'
    for km in re.finditer(r"(\w+)='((?:[^'\\]|\\.)*)'", args_str):
        key = km.group(1)
        if key in triple_args or key in kwargs:
            continue
        kwargs[key] = km.group(2)
    # 4. Unquoted args: key=value
    for km in re.finditer(r"(\w+)=([^,'\"\s)]+)", args_str):
        key = km.group(1)
        if key in triple_args or key in kwargs:
            continue
        val = km.group(2).strip()
        try:
            val = float(val) if "." in val else int(val)
        except (ValueError, TypeError):
            pass
        kwargs[key] = val
    return {"function": name, "kwargs": kwargs}


def _xml_to_native_call(tc: dict) -> str:
    """Convert parsed tool call dict to <tool_call>JSON</tool_call> string."""
    payload = {"name": tc["function"], "arguments": tc["kwargs"]}
    return f'<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>'


# ---------------------------------------------------------------------------
# System prompt: strip XML tool inventory block
# ---------------------------------------------------------------------------

def _strip_xml_tool_section(system: str) -> str:
    """Remove the 'Available tools — call them using <tool>' block.

    Uses plain string search to avoid catastrophic backtracking on long prompts.
    The block starts at "Available tools" and ends just before the next
    blank-line section ("Tool use rules:", "Security rules:", etc.).
    """
    marker = "Available tools"
    start = system.find(marker)
    if start == -1:
        return system

    end = len(system)
    for header in ("\n\nTool use rules", "\n\nSecurity rules",
                   "\n\nTool-use rules", "\n\nAlways-on tools"):
        idx = system.find(header, start)
        if idx != -1:
            end = min(end, idx)

    # Fallback: stop at the first blank line after the marker
    if end == len(system):
        blank = system.find("\n\n", start)
        if blank != -1:
            end = blank

    cleaned = system[:start] + system[end:]
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def convert_example(ex: dict) -> dict | None:
    """Convert one JSONL example from XML to native format.

    Returns the converted example or None if the example should be skipped
    (already native, or a broken partial-native example).
    """
    meta = ex.get("metadata", {})

    # Skip examples already in native format (broken partb_converted ones)
    if meta.get("tool_format") == "native":
        return None
    if meta.get("native_tools"):
        return None

    profile = meta.get("tool_profile", "all_tools")
    messages = ex.get("messages", [])
    new_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role == "system":
            new_messages.append({**msg, "content": _strip_xml_tool_section(content)})

        elif role == "assistant" and "<tool>" in content:
            tc = _parse_xml_tool_call(content)
            if tc:
                native_call = _xml_to_native_call(tc)
                # Replace the <tool>...</tool> block with <tool_call>...</tool_call>
                new_content = re.sub(
                    r"<tool>.*?</tool>", native_call, content, flags=re.DOTALL
                )
                new_messages.append({**msg, "content": new_content})
            else:
                # Couldn't parse — keep as-is (shouldn't happen)
                new_messages.append(msg)

        else:
            new_messages.append(msg)

    new_meta = {
        **meta,
        "native_tools": _schemas_for_profile(profile),
        "tool_format": "native",
    }
    return {"messages": new_messages, "metadata": new_meta}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert XML tool calls → native Hermes format")
    ap.add_argument("--input",  nargs="+", required=True,
                    help="Input JSONL file(s) — partA, partB, or combined sft_v3")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print stats and first converted example without writing")
    args = ap.parse_args()

    total = skipped = converted = no_tool = tool_converted = 0
    first_printed = False

    out_lines = []
    for fpath in args.input:
        print(f"Reading {fpath} ...", flush=True)
        for line in open(fpath, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            total += 1
            ex = json.loads(line)
            result = convert_example(ex)
            if result is None:
                skipped += 1
                continue
            converted += 1
            has_tool = any("<tool_call>" in m.get("content","") for m in result["messages"])
            if has_tool:
                tool_converted += 1
            else:
                no_tool += 1
            out_lines.append(result)

            if args.dry_run and not first_printed and has_tool:
                first_printed = True
                print("\n=== First converted tool-using example ===")
                for m in result["messages"]:
                    c = m.get("content","")
                    if m["role"] == "system":
                        print(f"[system] (len={len(c)}) {c[:300]}...")
                    elif "<tool_call>" in c or m["role"] == "tool":
                        print(f"[{m['role']}]: {c[:400]}")
                print("native_tools:", result["metadata"]["native_tools"][:1])

    print(f"\nStats:")
    print(f"  Total read    : {total}")
    print(f"  Skipped       : {skipped}  (broken native / already native)")
    print(f"  Converted     : {converted}")
    print(f"    with tool   : {tool_converted}")
    print(f"    no tool     : {no_tool}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in out_lines:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWritten → {args.output}  ({len(out_lines)} examples)")


if __name__ == "__main__":
    main()
