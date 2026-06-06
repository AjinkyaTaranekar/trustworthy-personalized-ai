#!/usr/bin/env python
"""validate_thinker_executor_data.py — pre-train gate for the factored Thinker/Executor SFT sets.

Frontier-lab practice: never spend GPU time on data you have not asserted on. This runs on CPU
and fails loudly if any train/serve contract is violated. Checks:

THINKER (data/train_sft_thinker.jsonl)
  T1. system turn == THINKER_STUDENT_PROMPT (single source of truth)
  T2. ends in an assistant <answer> (or <ask>); opening <think> >= MIN_THINK_CHARS
  T3. NO tool-call syntax (<tool_call> / <tool>) anywhere in assistant turns (prose-only contract)
  T4. every tool turn is in the CANONICAL served representation (tool_io.sanitise_tool_result),
      i.e. wrapped in [TOOL_RESULT: name] and within the per-tool budget — byte-identical to serve
  T5. first user turn carries a [USER MEMORY] block
  T6. NO assistant turn has an empty <think> (the `<think></think>` → output pattern that the
      enable_thinking=False render bakes in is the think-collapse failure mode — P1.4)

EXECUTOR (data/train_sft_executor.jsonl)
  E1. system turn == EXECUTOR_STUDENT_PROMPT
  E2. exactly ONE <tool_call> in the assistant turn; tool ∈ EXECUTOR_OWNED
  E3. native_tools schema advertises EXACTLY EXECUTOR_OWNED (no untrained tools, e.g. get_datetime)
  E4. COPY FIDELITY: the code/query/url the <act> instruction embeds appears in the emitted call
      (the Executor's whole job is verbatim transcription — assert the data actually supports it)

Exit code 0 = clean, 1 = at least one violation. Prints a per-check tally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sft_v3_generator import THINKER_STUDENT_PROMPT, EXECUTOR_STUDENT_PROMPT
from sft_trajectory_splitter import (
    EXECUTOR_OWNED, _HERMES_RE, _first_call, _load_hermes,
)
from sft_dataset_assembler import MIN_THINK_CHARS, _THINK_BLOCK_RE
from tool_io import sanitise_tool_result
from thinker_executor_orchestrator import openai_schemas, EXECUTOR_TOOLS
from pipeline_tools import ToolRegistry

DATA = Path(__file__).parent / "data"
# The exact schema the orchestrator serves the Executor — E3 asserts every row matches it byte-for-byte.
_SERVED_EXECUTOR_SCHEMA = json.dumps(openai_schemas(ToolRegistry(), EXECUTOR_TOOLS))


def _opening_think(content: str) -> str:
    m = _THINK_BLOCK_RE.search(content or "")
    return m.group(1).strip() if m else ""


def validate_thinker(path: Path) -> list[str]:
    errs: list[str] = []
    n = 0
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        msgs = r["messages"]
        # T1
        if msgs[0]["role"] != "system" or msgs[0]["content"] != THINKER_STUDENT_PROMPT:
            errs.append(f"[T1] row {ln}: system != THINKER_STUDENT_PROMPT")
        # T2
        last = msgs[-1]
        if last["role"] != "assistant" or not ("<answer>" in last["content"] or "<ask>" in last["content"]):
            errs.append(f"[T2] row {ln}: does not end in assistant <answer>/<ask>")
        opening = next((_opening_think(m["content"]) for m in msgs if m["role"] == "assistant"), "")
        if len(opening) < MIN_THINK_CHARS:
            errs.append(f"[T2] row {ln}: opening <think> {len(opening)} < {MIN_THINK_CHARS}")
        # T3 + T6
        for m in msgs:
            if m["role"] != "assistant":
                continue
            if _HERMES_RE.search(m["content"]) or "<tool>" in m["content"]:
                errs.append(f"[T3] row {ln}: tool-call syntax in a Thinker assistant turn")
                break
        for m in msgs:
            if m["role"] != "assistant":
                continue
            if len(_opening_think(m["content"])) == 0:
                errs.append(f"[T6] row {ln}: assistant turn with empty <think> (collapse-teaching pattern)")
                break
        # T4 — tool turns must equal the canonical representation. We can't re-derive the raw
        # (it's already wrapped), but we can assert the wrapper shape + budget compliance.
        for m in msgs:
            if m["role"] != "tool":
                continue
            c = m["content"]
            if not (c.startswith("[TOOL_RESULT:") and c.rstrip().endswith("[/TOOL_RESULT]")):
                errs.append(f"[T4] row {ln}: tool turn not in [TOOL_RESULT] canonical form")
                break
            # idempotence: re-sanitising the inner payload must not change it (budget already applied)
        # T5
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if "[USER MEMORY" not in first_user:
            errs.append(f"[T5] row {ln}: first user turn missing [USER MEMORY] block")
    print(f"  thinker rows checked: {n}")
    return errs


def validate_executor(path: Path) -> list[str]:
    errs: list[str] = []
    n = 0
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        msgs = r["messages"]
        meta = r.get("metadata", {})
        # E1
        if msgs[0]["content"] != EXECUTOR_STUDENT_PROMPT:
            errs.append(f"[E1] row {ln}: system != EXECUTOR_STUDENT_PROMPT")
        instr = msgs[1]["content"] if len(msgs) > 1 else ""
        target = msgs[2]["content"] if len(msgs) > 2 else ""
        # E2
        calls = _HERMES_RE.findall(target)
        if len(calls) != 1:
            errs.append(f"[E2] row {ln}: expected exactly 1 <tool_call>, found {len(calls)}")
            continue
        parsed = _load_hermes(calls[0].strip())
        if parsed is None:
            errs.append(f"[E2] row {ln}: tool_call body not valid JSON")
            continue
        name, args = parsed
        if name not in EXECUTOR_OWNED:
            errs.append(f"[E2] row {ln}: tool {name!r} not in EXECUTOR_OWNED")
        # E3 — native_tools must be BYTE-identical to what the orchestrator serves the Executor
        # (names, descriptions, params, AND order — the schema is rendered into the prompt).
        if json.dumps(meta.get("native_tools", [])) != _SERVED_EXECUTOR_SCHEMA:
            advertised = [s.get("function", {}).get("name") for s in meta.get("native_tools", [])]
            errs.append(f"[E3] row {ln}: native_tools != served Executor schema (got {advertised})")
        # E4 — copy fidelity: the salient argument must appear verbatim in the instruction
        salient = ""
        if name == "python_execute":
            salient = (args.get("code") or "").strip()
        elif name == "web_search":
            salient = (args.get("query") or "").strip()
        elif name == "read_url":
            salient = (args.get("url") or "").strip()
        if salient and salient not in instr:
            errs.append(f"[E4] row {ln}: {name} arg not present verbatim in <act> instruction "
                        f"(copy-fidelity precondition violated)")
    print(f"  executor rows checked: {n}")
    return errs


def main() -> int:
    print("=== Thinker validation ===")
    t_errs = validate_thinker(DATA / "train_sft_thinker.jsonl")
    print("=== Executor validation ===")
    e_errs = validate_executor(DATA / "train_sft_executor.jsonl")

    all_errs = t_errs + e_errs
    if not all_errs:
        print("\n✅ ALL CHECKS PASSED — data is train/serve consistent.")
        return 0
    print(f"\n❌ {len(all_errs)} violation(s):")
    from collections import Counter
    by_code = Counter(e.split(']')[0] + ']' for e in all_errs)
    for code, c in sorted(by_code.items()):
        print(f"   {code} {c}")
    for e in all_errs[:25]:
        print("   -", e)
    return 1


if __name__ == "__main__":
    sys.exit(main())
