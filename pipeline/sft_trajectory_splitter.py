#!/usr/bin/env python
"""sft_trajectory_splitter.py — factor v3 trajectories into Thinker + Executor SFT sets.

This is a PURE TRANSFORMATION (no GPU, no teacher calls). It reads the already-generated
v3 distillation trajectories (data/train_partA_v3.jsonl, data/train_partB_v3.jsonl) and
projects each onto two role-conditioned views, per
wiki/experiments/thinker-executor-experiment.md §7.2–7.3, §7.9–7.10:

  * Thinker view  -> data/train_sft_thinker.jsonl   (multi-turn: <think> + <ask|act|answer>)
  * Executor view -> data/train_sft_executor.jsonl  (one <act> instruction -> one tool call)

Because both views are read off the SAME trajectory, the <act> the Thinker learns to emit
and the tool call the Executor learns to produce are the same action — alignment is
structural, not hoped-for. The <act> instruction is rendered deterministically from the
observed call and its concrete arguments (no LLM, no GPU).

Tool ownership (§4.1): the Executor owns the external-world tools (python_execute,
web_search, read_url; get_datetime is advertised to it but never delegated). The Thinker's
own state tools (user_memory_*, scratchpad_*, get_datetime) are handled in the Thinker's
context by the orchestrator at inference, so their call turns are dropped from the factored
Thinker stream (their reasoning <think> is preserved) and are never emitted as <act>s.

Parsing helpers and tool schemas are reused from sft_dataset_assembler.py; the role prompts
THINKER_STUDENT_PROMPT / EXECUTOR_STUDENT_PROMPT are imported from sft_v3_generator.py — the
single sources of truth shared with the served models and the Branch B generator.

Usage:
  python sft_trajectory_splitter.py                      # factor both parts -> both outputs
  python sft_trajectory_splitter.py --limit 50           # quick subset
  python sft_trajectory_splitter.py --inspect 5          # print 5 factored rows of each, no write
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Reused parsing helpers / constants (single source of truth for tool parsing + gates).
from sft_dataset_assembler import (
    MIN_THINK_CHARS,
    MAX_RESULT_CHARS,
    _THINK_BLOCK_RE,
    _TOOL_RE,
    _unwrap_tool_result,
    _xml_tool_to_native,
)
# Role prompts — single source of truth shared with the served models.
from sft_v3_generator import THINKER_STUDENT_PROMPT, EXECUTOR_STUDENT_PROMPT

# --- Tool ownership (experiment §4.1) -------------------------------------------------
EXECUTOR_OWNED = {"python_execute", "web_search", "read_url"}   # delegated via <act>
EXECUTOR_TOOL_NAMES = {"python_execute", "web_search", "read_url", "get_datetime"}  # advertised
# get_datetime + user_memory_* + scratchpad_* stay Thinker-side (handled in context).

# Native schemas advertised to the Executor at train/inference time.
try:
    from pipeline_tools import ToolRegistry as _ToolRegistry
    EXECUTOR_SCHEMAS = _ToolRegistry().to_openai_schemas(EXECUTOR_TOOL_NAMES)
except Exception as _e:  # pragma: no cover - import fallback
    from sft_dataset_assembler import OPENAI_SCHEMAS as _ALL
    EXECUTOR_SCHEMAS = [s for s in _ALL if s.get("function", {}).get("name") in EXECUTOR_TOOL_NAMES]
    print(f"  [splitter] WARN: registry import failed ({_e}); filtered Executor schemas from assembler")

# --- Hermes <tool_call>{json}</tool_call> parsing -------------------------------------
# The v3 trajectories store calls as Qwen3 Hermes TEXT (not legacy <tool> XML, not a
# structured tool_calls field), so we parse the JSON body here. A few legacy <tool> rows
# also exist and are handled via the assembler's _xml_tool_to_native fallback.
# NB: the JSON body is delimited by the </tool_call> tag, NOT by braces — python_execute
# `code` arguments are full of { } (dicts, f-strings, sets), so a non-greedy \{.*?\} would
# truncate at the first inner brace and break json.loads. Capture to the closing tag.
_HERMES_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_INNER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _first_think(content: str) -> str:
    m = _THINK_BLOCK_RE.search(content or "")
    return m.group(1).strip() if m else ""


def _first_answer(content: str) -> str | None:
    m = _ANSWER_INNER_RE.search(content or "")
    return m.group(1).strip() if m else None


def _load_hermes(body: str) -> tuple[str, dict] | None:
    """Parse one Hermes JSON body -> (name, args). strict=False is REQUIRED: python_execute
    `code` strings carry raw newlines/tabs, which strict JSON (the default) rejects with
    JSONDecodeError. (This is the same trap that silently dropped ~780 maths calls.)"""
    try:
        obj = json.loads(body, strict=False)
    except Exception:
        return None
    name = obj.get("name")
    args = obj.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args, strict=False)
        except Exception:
            args = {}
    if not name:
        return None
    return name, (args if isinstance(args, dict) else {})


def _first_call(content: str) -> tuple[str, dict] | None:
    """First tool call in an assistant turn as (name, args). Hermes JSON first, then the
    legacy <tool>name(args)</tool> XML fallback (reusing the assembler's parser)."""
    content = content or ""
    m = _HERMES_RE.search(content)
    if m:
        parsed = _load_hermes(m.group(1))
        if parsed is not None:
            return parsed
    xm = _TOOL_RE.search(content)
    if xm:
        parsed = _xml_tool_to_native(xm.group(1))
        if parsed is not None:
            return parsed
    return None


# --- Deterministic <act> instruction rendering (§7.9 #2) ------------------------------
def act_instruction(name: str, args: dict) -> str | None:
    """Render a self-contained, plain-language <act> instruction from an Executor-owned
    call. Names the tool explicitly — free here (the tool actually used is known) and it
    makes the instruction->call mapping cleaner for the Executor (§7.2)."""
    if name == "web_search":
        q = (args.get("query") or "").strip()
        return f"Use web_search to find: {q}" if q else None
    if name == "python_execute":
        code = args.get("code") or ""
        return f"Use python_execute to run this code:\n{code}" if code.strip() else None
    if name == "read_url":
        url = (args.get("url") or "").strip()
        if not url:
            return None
        prompt = (args.get("prompt") or "").strip()
        base = f"Use read_url to fetch {url}"
        return base + (f" and extract: {prompt}" if prompt else "")
    return None


def _wrap_think(text: str) -> str:
    text = (text or "").strip()
    return f"<think>\n{text}\n</think>\n" if text else ""


def _join_think(carry: list[str], current: str) -> str:
    parts = [t for t in (carry + [current]) if t and t.strip()]
    return "\n\n".join(parts)


def _cap(result: str) -> str:
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + " … [truncated]"
    return result


# --- Thinker view (§7.2) ---------------------------------------------------------------
def factor_thinker(messages: list[dict]) -> dict | None:
    """Project one source trajectory onto the Thinker target: prose only — a <think> block
    followed by <act> (Executor-owned step) or the final <answer>. Thinker-owned tool turns
    are dropped (their <think> is carried forward). Returns a row dict, or None if it fails
    the gate (must end in <answer>; opening <think> >= MIN_THINK_CHARS)."""
    out: list[dict] = [{"role": "system", "content": THINKER_STUDENT_PROMPT}]
    carry: list[str] = []
    opening_think_len: int | None = None
    emitted_act = False
    ended_answer = False

    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
            i += 1
            continue
        if role in ("system", "tool"):
            i += 1   # original system replaced; stray tool results belong to dropped calls
            continue
        if role != "assistant":
            i += 1
            continue

        content = m.get("content") or ""
        think = _first_think(content)
        call = _first_call(content)
        answer = _first_answer(content)

        # Consume the tool result that follows a call turn (kept for <act>, dropped otherwise).
        result = None
        if call is not None and i + 1 < n and messages[i + 1].get("role") == "tool":
            result = _unwrap_tool_result(messages[i + 1].get("content") or "")
            i += 1

        if call is not None and call[0] in EXECUTOR_OWNED:
            instr = act_instruction(*call)
            if instr is None:
                if think:
                    carry.append(think)
                i += 1
                continue
            think_text = _join_think(carry, think)
            carry = []
            out.append({"role": "assistant", "content": _wrap_think(think_text) + f"<act>{instr}</act>"})
            if opening_think_len is None:
                opening_think_len = len(think_text)
            emitted_act = True
            if result is not None:
                out.append({"role": "tool", "content": _cap(result)})
        elif answer is not None:
            think_text = _join_think(carry, think)
            carry = []
            out.append({"role": "assistant", "content": _wrap_think(think_text) + f"<answer>{answer}</answer>"})
            if opening_think_len is None:
                opening_think_len = len(think_text)
            ended_answer = True
        else:
            # Thinker-owned call (user_memory_*/scratchpad_*/get_datetime) or bare think: drop
            # the turn, keep its reasoning for the next emitted Thinker turn.
            if think:
                carry.append(think)
        i += 1

    # Quality gate (§7.9 #4)
    if not ended_answer:
        return None
    if out[-1]["role"] != "assistant" or "<answer>" not in out[-1]["content"]:
        return None
    if opening_think_len is None or opening_think_len < MIN_THINK_CHARS:
        return None
    # Prose-only contract: drop rows where the teacher wrote tool-call syntax inside a
    # <think>/<answer> (a source-data artefact) — the Thinker must never emit it.
    if any(m["role"] == "assistant" and (_HERMES_RE.search(m["content"]) or "<tool>" in m["content"])
           for m in out):
        return None

    return {
        "messages": out,
        "metadata": {
            "model_role": "thinker",
            "pipeline": "thinker_executor",
            "source": "trajectory_factoring",
            "has_act": emitted_act,
            # No native_tools: the Thinker never emits tool-call syntax.
        },
    }


# --- Executor view (§7.3) --------------------------------------------------------------
def _executor_calls(content: str) -> list[tuple[str, dict, str]]:
    """All Executor-relevant calls in an assistant turn as (name, args, target_text).
    target_text is the verbatim Hermes call the Executor must learn to emit — preserved
    byte-for-byte from the source (raw newlines and all) so it matches what the model
    naturally produces and what the inference parser consumes. Multiple calls per turn are
    all yielded. Legacy <tool> XML (rare) is re-serialised since no native body exists."""
    content = content or ""
    out: list[tuple[str, dict, str]] = []
    matches = list(_HERMES_RE.finditer(content))
    if matches:
        for mm in matches:
            body = mm.group(1).strip()
            parsed = _load_hermes(body)
            if parsed is None:
                continue
            name, args = parsed
            out.append((name, args, f"<tool_call>\n{body}\n</tool_call>"))
        return out
    for xm in _TOOL_RE.finditer(content):
        parsed = _xml_tool_to_native(xm.group(1))
        if parsed is None:
            continue
        name, args = parsed
        body = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        out.append((name, args, f"<tool_call>\n{body}\n</tool_call>"))
    return out


def factor_executor(messages: list[dict], q_id: str | None) -> list[dict]:
    """Project Executor-owned calls onto one-instruction -> one-call training pairs."""
    rows: list[dict] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for name, args, target in _executor_calls(m.get("content") or ""):
            if name not in EXECUTOR_OWNED:
                continue
            instr = act_instruction(name, args)
            if instr is None:
                continue
            rows.append({
                "messages": [
                    {"role": "system", "content": EXECUTOR_STUDENT_PROMPT},
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "model_role": "executor",
                    "pipeline": "thinker_executor",
                    "source": "trajectory_factoring",
                    "tool": name,
                    "question_id": q_id,
                    "native_tools": EXECUTOR_SCHEMAS,
                },
            })
    return rows


# --- Driver ---------------------------------------------------------------------------
def _load(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def run(part_a: str, part_b: str, out_thinker: str, out_executor: str,
        limit: int | None = None, inspect: int = 0, dedup_executor: bool = True) -> None:
    sources: list[dict] = []
    for p in (part_a, part_b):
        path = Path(p)
        if not path.exists():
            print(f"  [splitter] skip missing source: {p}")
            continue
        loaded = _load(path, limit)
        print(f"  loaded {len(loaded)} trajectories from {path.name}")
        sources.extend(loaded)

    thinker_rows: list[dict] = []
    executor_rows: list[dict] = []
    n_thinker_dropped = 0
    seen_exec: set[tuple[str, str]] = set()
    tool_counts: dict[str, int] = {}

    for ex in sources:
        msgs = ex.get("messages", [])
        q_id = (ex.get("metadata") or {}).get("question_id")

        t = factor_thinker(msgs)
        if t is None:
            n_thinker_dropped += 1
        else:
            thinker_rows.append(t)

        for er in factor_executor(msgs, q_id):
            key = (er["messages"][1]["content"], er["messages"][2]["content"])
            if dedup_executor and key in seen_exec:
                continue
            seen_exec.add(key)
            executor_rows.append(er)
            tool_counts[er["metadata"]["tool"]] = tool_counts.get(er["metadata"]["tool"], 0) + 1

    print("\n=== Factoring summary ===")
    print(f"  source trajectories : {len(sources)}")
    print(f"  thinker rows kept   : {len(thinker_rows)}  (dropped {n_thinker_dropped} — no <answer> or short opening <think>)")
    n_act = sum(1 for r in thinker_rows if r['metadata']['has_act'])
    print(f"    with >=1 <act>    : {n_act}   no-tool (reason->answer): {len(thinker_rows) - n_act}")
    print(f"  executor pairs      : {len(executor_rows)}  (dedup={'on' if dedup_executor else 'off'})")
    for tool, c in sorted(tool_counts.items(), key=lambda kv: -kv[1]):
        print(f"      {tool:16s} {c}")

    if inspect:
        _print_inspect(thinker_rows, executor_rows, inspect)
        return

    Path(out_thinker).parent.mkdir(parents=True, exist_ok=True)
    with open(out_thinker, "w", encoding="utf-8") as f:
        for r in thinker_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_executor, "w", encoding="utf-8") as f:
        for r in executor_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {out_thinker}")
    print(f"  wrote {out_executor}")
    print("\nNext: train two checkpoints (no changes to 2_model_trainer.py needed):")
    print(f"  python 2_model_trainer.py --mode sft --dataset {out_thinker}  --output_name checkpoint_thinker")
    print(f"  python 2_model_trainer.py --mode sft --dataset {out_executor} --output_name checkpoint_executor")


def _print_inspect(thinker_rows: list[dict], executor_rows: list[dict], k: int) -> None:
    print(f"\n========== {k} THINKER rows ==========")
    for r in thinker_rows[:k]:
        print("-" * 70)
        for m in r["messages"]:
            tag = m["role"].upper()
            body = m["content"]
            body = body if len(body) < 600 else body[:600] + " …"
            print(f"[{tag}] {body}")
    print(f"\n========== {k} EXECUTOR rows ==========")
    for r in executor_rows[:k]:
        print("-" * 70)
        for m in r["messages"]:
            print(f"[{m['role'].upper()}] {m['content']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Factor v3 trajectories into Thinker + Executor SFT sets.")
    p.add_argument("--part_a", default="data/train_partA_v3.jsonl")
    p.add_argument("--part_b", default="data/train_partB_v3.jsonl")
    p.add_argument("--out_thinker", default="data/train_sft_thinker.jsonl")
    p.add_argument("--out_executor", default="data/train_sft_executor.jsonl")
    p.add_argument("--limit", type=int, default=None, help="Cap trajectories read per source (quick runs).")
    p.add_argument("--inspect", type=int, default=0, help="Print N factored rows of each view and exit (no write).")
    p.add_argument("--no_dedup_executor", action="store_true", help="Keep duplicate Executor instruction->call pairs.")
    args = p.parse_args()
    run(args.part_a, args.part_b, args.out_thinker, args.out_executor,
        limit=args.limit, inspect=args.inspect, dedup_executor=not args.no_dedup_executor)


if __name__ == "__main__":
    main()
