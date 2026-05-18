"""
Convert train_partB.jsonl → train_partB_v3.jsonl
=================================================
Transforms single-turn math entries into v3 multi-turn format:

  Before (partB):
    system + user + assistant(think+tool+answer all inline)

  After (v3):
    system + user
      → assistant: <think>reasoning</think><tool>python_execute(code=...)</tool>
      → tool:      [TOOL_RESULT: python_execute]\n{stdout}\n[/TOOL_RESULT]
      → assistant: <answer>X</answer>

Reasoning text is reconstructed from the numbered steps already present
in the code comments (which the pipeline embeds as inline narration).

Usage:
    python convert_partB_to_v3.py
    python convert_partB_to_v3.py --dry_run   # parse + execute, no write
"""

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
INPUT  = ROOT / "data" / "train_partB.jsonl"
OUTPUT = ROOT / "data" / "train_partB_v3.jsonl"

V3_SYSTEM = (
    "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
    "Available tools: python_execute. "
    "Use python_execute for all arithmetic — never approximate mentally. "
    "Give your final answer in <answer>...</answer>. "
    "Format: 2 decimal places for money/geometry, 4 for probability/fractions, integers for counts."
)

CATEGORY_MAP = {
    "word_problems": "math_word_problems",
    "algebra":       "math_algebra",
    "trigonometry":  "math_trigonometry",
    "statistics":    "math_statistics",
    "geometry":      "math_geometry",
    "arithmetic":    "math_arithmetic",
}

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_answer(content: str) -> str | None:
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_tool_and_reasoning(content: str) -> tuple[str, str] | None:
    """Return (reasoning_text, python_code) from a <tool>python_execute(...).</tool> block."""
    tool_start = content.find('<tool>')
    if tool_start == -1:
        return None
    tool_end = content.find('</tool>', tool_start)
    if tool_end == -1:
        return None

    reasoning = content[:tool_start].strip()
    inner = content[tool_start + 6 : tool_end]   # between <tool> and </tool>

    # Prefer triple double-quotes (most common in this dataset)
    m = re.search(r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)', inner, re.DOTALL)
    if m:
        return reasoning, m.group(1)

    # Triple single-quotes fallback
    m = re.search(r"python_execute\s*\(\s*code\s*=\s*'''(.*?)'''\s*\)", inner, re.DOTALL)
    if m:
        return reasoning, m.group(1)

    # Single double-quote fallback (math code rarely contains bare " chars)
    m = re.search(r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)', inner, re.DOTALL)
    if m:
        return reasoning, m.group(1)

    return None


def _extract_markdown_and_reasoning(content: str) -> tuple[str, str] | None:
    """Return (reasoning_text, code) from a ```python ... ``` block."""
    m = re.search(r'```python\n(.*?)```', content, re.DOTALL)
    if not m:
        m = re.search(r'```\n(.*?)```', content, re.DOTALL)
    if not m:
        return None
    reasoning = content[:m.start()].strip()
    return reasoning, m.group(1).strip()


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _run_code(code: str) -> str | None:
    """Execute code in a subprocess; return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None
        out = result.stdout.strip()
        return out if out else None
    except (subprocess.TimeoutExpired, Exception):
        return None


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

def _build_think_block(reasoning: str, code: str) -> str:
    """Return the content for the <think> block.

    If the LLM already produced a reasoning paragraph (CAPABILITY_CHECK etc.)
    we use it directly. If the response skipped reasoning (bare tool call),
    we reconstruct a minimal block from the code comments.
    """
    if reasoning:
        return reasoning

    # Reconstruct from code comments — pull lines starting with #
    comment_lines = [
        line.lstrip("# ").strip()
        for line in code.splitlines()
        if line.strip().startswith("#") and line.strip() != "#"
    ]
    if comment_lines:
        narration = "\n".join(f"- {c}" for c in comment_lines[:10])
        return f"Let me solve this step-by-step using python_execute.\n\n{narration}"
    return "Let me solve this step-by-step using python_execute."


# ---------------------------------------------------------------------------
# Entry converter
# ---------------------------------------------------------------------------

def convert_entry(entry: dict) -> tuple[dict, str]:
    """Return (converted_entry, extraction_method_label)."""
    msgs = entry["messages"]
    meta = entry.get("metadata", {})

    user_content  = next(m["content"] for m in msgs if m["role"] == "user")
    asst_content  = next(m["content"] for m in msgs if m["role"] == "assistant")

    answer   = _extract_answer(asst_content)
    fallback = str(meta.get("computed_answer", ""))
    if answer is None:
        answer = fallback

    # Try <tool> tag first, then markdown block
    parts = _extract_tool_and_reasoning(asst_content)
    method = "tool_tag"
    if parts is None:
        parts = _extract_markdown_and_reasoning(asst_content)
        method = "markdown_block"

    new_msgs: list[dict] = [
        {"role": "system", "content": V3_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if parts:
        reasoning, code = parts
        tool_output = _run_code(code) or fallback
        think_text  = _build_think_block(reasoning, code)

        new_msgs.append({
            "role": "assistant",
            "content": f'<think>\n{think_text}\n</think>\n<tool>python_execute(code="""{code}""")</tool>',
        })
        new_msgs.append({
            "role": "tool",
            "name": "python_execute",
            "content": f"[TOOL_RESULT: python_execute]\n{tool_output}\n[/TOOL_RESULT]",
        })
        new_msgs.append({
            "role": "assistant",
            "content": f"<answer>{answer}</answer>",
        })
    else:
        # No code found — wrap entire response in think tags
        method = "no_code"
        clean = re.sub(r'<answer>.*?</answer>', '', asst_content, flags=re.DOTALL).strip()
        new_msgs.append({
            "role": "assistant",
            "content": f"<think>\n{clean}\n</think>\n<answer>{answer}</answer>",
        })

    qt = meta.get("question_type", "")
    new_meta = {
        "source":          meta.get("source", ""),
        "question_id":     uuid.uuid4().hex[:12],
        "category":        CATEGORY_MAP.get(qt, qt),
        "tool_profile":    "compute_only",
        "constitution_score": 1.0,
        "revised":         False,
        "pipeline":        "partb_converted",
        "original_level":  meta.get("level"),
        "subject":         meta.get("subject", ""),
        "expected_answer": meta.get("expected_answer", ""),
    }

    return {"messages": new_msgs, "metadata": new_meta}, method


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert train_partB.jsonl to v3 format")
    parser.add_argument("--dry_run", action="store_true", help="Parse and execute but do not write output")
    parser.add_argument("--input",  default=str(INPUT),  help="Input JSONL path")
    parser.add_argument("--output", default=str(OUTPUT), help="Output JSONL path")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    stats = {"tool_tag": 0, "markdown_block": 0, "no_code": 0, "error": 0}
    entries: list[dict] = []

    with open(in_path, encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    print(f"Processing {len(raw_lines)} entries from {in_path.name} ...")

    for i, line in enumerate(raw_lines, 1):
        try:
            entry = json.loads(line)
            converted, method = convert_entry(entry)
            entries.append(converted)
            stats[method] += 1
            if i % 100 == 0:
                print(f"  {i}/{len(raw_lines)} done", flush=True)
        except Exception as e:
            stats["error"] += 1
            print(f"  [line {i}] ERROR: {e}", file=sys.stderr)

    print(f"\nExtraction breakdown:")
    print(f"  <tool> tag     : {stats['tool_tag']}")
    print(f"  markdown block : {stats['markdown_block']}")
    print(f"  no code found  : {stats['no_code']}")
    print(f"  errors         : {stats['error']}")
    print(f"  total written  : {len(entries)}")

    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(entries)} entries -> {out_path}")
    else:
        print("\n[dry_run] No file written.")

    # Spot-check: print one converted example
    if entries:
        print("\n--- Spot-check: entry 0 messages ---")
        for msg in entries[0]["messages"]:
            role = msg["role"]
            preview = msg["content"][:120].replace("\n", "\\n")
            name = f" [{msg['name']}]" if "name" in msg else ""
            print(f"  {role}{name}: {preview}")


if __name__ == "__main__":
    main()
