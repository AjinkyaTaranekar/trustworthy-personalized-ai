#!/usr/bin/env python
"""composed_loop_eval.py — end-to-end evaluation of the COMPOSED Thinker–Executor loop.

The trainer measures the Thinker and Executor SEPARATELY (eval_loss, ROUGE). That can pass while
the composed system fails — two checkpoints that each look fine can interact badly (e.g. the
Executor corrupting code the Thinker quoted, or the loop never reaching <answer>). Frontier labs
gate releases on end-to-end task success, not component loss. This script runs held-out QUESTIONS
through the real two-adapter loop (ThinkerExecutor) and reports:

  • completion_rate   — fraction of turns that reach a final <answer> (not <ask>, not max_steps)
  • ask_rate          — fraction that end in <ask> (clarification)
  • mean_steps        — mean Thinker↔Executor cycles per question
  • exec_parse_rate   — fraction of <act> delegations where the Executor emitted a parseable call
  • copy_fidelity     — fraction of calls whose salient arg (code/query/url) appears verbatim in the
                        <act> instruction the Thinker gave (the Executor's core job: faithful copy)
  • math_correct      — for math questions with an expected answer, fraction whose final <answer>
                        contains the correct number

Usage
  python composed_loop_eval.py --self_test                       # CPU, scripted — no model
  python composed_loop_eval.py \
      --thinker models/checkpoint_thinker --executor models/checkpoint_executor \
      --part_a data/train_partA_v3.jsonl --part_b data/train_partB_v3.jsonl \
      --n 60 --seed 42                                            # GPU

Output: a metrics table + reports/composed_loop_eval.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

from thinker_executor_orchestrator import (
    ThinkerExecutor, HFGenerator, ScriptedGenerator,
)
from pipeline_tools import ToolRegistry

_MATH_CATEGORIES = {
    "arithmetic", "algebra", "geometry", "statistics", "unit_conversion", "word_problems",
    "trigonometry", "calculus", "advanced_geometry", "gsm8k",
    "math_arithmetic", "math_algebra", "math_geometry", "math_statistics",
    "math_trigonometry", "math_word_problems", "math_calculus",
}
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_MEMORY_BLOCK_RE = re.compile(r"\[USER MEMORY.*?\[/USER MEMORY\]\s*", re.DOTALL)


def _last_number(text: str) -> Optional[str]:
    nums = _NUM_RE.findall((text or "").replace(",", ""))
    return nums[-1] if nums else None


def _answers_match(a: str, b: str, tol: float = 0.01) -> bool:
    try:
        af, bf = float(a), float(b)
        if abs(bf) < 1e-9:
            return abs(af) < 1e-6
        return abs(af - bf) / abs(bf) < tol
    except (ValueError, TypeError):
        return (a or "").strip() == (b or "").strip()


def _salient_arg(name: str, args: dict) -> str:
    if name == "python_execute":
        return (args.get("code") or "").strip()
    if name == "web_search":
        return (args.get("query") or "").strip()
    if name == "read_url":
        return (args.get("url") or "").strip()
    return ""


# --- held-out questions ----------------------------------------------------------------
def load_questions(paths: list[str], n: int, seed: int) -> list[dict]:
    rows = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [eval] skip missing: {p}")
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            meta = ex.get("metadata", {})
            q = next((m["content"] for m in ex.get("messages", []) if m.get("role") == "user"), "")
            q = _MEMORY_BLOCK_RE.sub("", q).strip()   # strip any injected memory block → raw question
            if not q:
                continue
            rows.append({
                "question":        q,
                "category":        meta.get("category") or meta.get("question_type", "unknown"),
                "expected_answer": meta.get("expected_answer"),
            })
    random.Random(seed).shuffle(rows)
    return rows[:n]


# --- scoring ---------------------------------------------------------------------------
def score_one(res: dict, category: str, expected: Optional[str]) -> dict:
    trace = res.get("tool_trace", [])
    acts = len(trace)
    parsed = sum(1 for t in trace if t.get("tool_call"))
    fid_ok = fid_total = 0
    for t in trace:
        call = t.get("tool_call")
        if not call:
            continue
        sal = _salient_arg(call.get("function", ""), call.get("kwargs", {}))
        if not sal:
            continue
        fid_total += 1
        if sal in (t.get("act") or ""):
            fid_ok += 1
    is_math = category in _MATH_CATEGORIES and expected not in (None, "")
    math_correct = None
    if is_math:
        ans_num = _last_number(res.get("answer", ""))
        math_correct = bool(ans_num and _answers_match(ans_num, str(expected)))
    return {
        "completed":  res.get("type") == "answer",
        "asked":      res.get("type") == "ask",
        "steps":      acts,
        "exec_parsed": parsed, "exec_acts": acts,
        "fid_ok": fid_ok, "fid_total": fid_total,
        "is_math": is_math, "math_correct": math_correct,
    }


def aggregate(scores: list[dict]) -> dict:
    n = len(scores) or 1
    fid_ok = sum(s["fid_ok"] for s in scores)
    fid_total = sum(s["fid_total"] for s in scores) or 1
    parsed = sum(s["exec_parsed"] for s in scores)
    acts = sum(s["exec_acts"] for s in scores) or 1
    math_rows = [s for s in scores if s["is_math"]]
    math_ok = sum(1 for s in math_rows if s["math_correct"])
    return {
        "rows":            len(scores),
        "completion_rate": round(sum(s["completed"] for s in scores) / n, 4),
        "ask_rate":        round(sum(s["asked"] for s in scores) / n, 4),
        "mean_steps":      round(sum(s["steps"] for s in scores) / n, 4),
        "exec_parse_rate": round(parsed / acts, 4),
        "copy_fidelity":   round(fid_ok / fid_total, 4),
        "math_rows":       len(math_rows),
        "math_correct":    round(math_ok / len(math_rows), 4) if math_rows else None,
    }


def _self_test() -> None:
    print("=== composed_loop_eval self-test (CPU, scripted) ===")
    thinker = [
        "<think>17×23 is deterministic arithmetic; I'll delegate it to avoid a mental-math slip and "
        "make sure the figure is exact before I commit to it in the answer.</think>\n"
        "<act>Use python_execute to run this code:\nprint(17 * 23)</act>",
        "<think>The execution returned 391, which resolves the request exactly.</think>\n"
        "<answer>17 × 23 = 391. Want me to take this further in any direction?</answer>",
    ]
    executor = ['<tool_call>\n{"name": "python_execute", "arguments": {"code": "print(17 * 23)"}}\n</tool_call>']
    orch = ThinkerExecutor(ScriptedGenerator(thinker, executor), ToolRegistry())
    res = orch.run("What is 17 times 23?", memory_text="")
    s = score_one(res, "math_arithmetic", "391")
    assert s["completed"] and s["math_correct"], s
    assert s["fid_ok"] == 1 and s["fid_total"] == 1, s   # code copied verbatim into the call
    agg = aggregate([s])
    assert agg["completion_rate"] == 1.0 and agg["copy_fidelity"] == 1.0 and agg["math_correct"] == 1.0, agg
    print(f"  {agg}")

    # copy-fidelity failure: Executor paraphrases the code the <act> specified
    bad_exec = ['<tool_call>\n{"name": "python_execute", "arguments": {"code": "print(391)"}}\n</tool_call>']
    orch2 = ThinkerExecutor(ScriptedGenerator(list(thinker), bad_exec), ToolRegistry())
    s2 = score_one(orch2.run("What is 17 times 23?", memory_text=""), "math_arithmetic", "391")
    assert s2["fid_ok"] == 0 and s2["fid_total"] == 1, s2   # "print(391)" not in the <act> instruction
    print(f"  copy-fidelity catches paraphrase: fid_ok={s2['fid_ok']}/{s2['fid_total']}")
    print("ALL SELF-TESTS PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end Thinker–Executor composed-loop eval.")
    p.add_argument("--thinker", default="models/checkpoint_thinker")
    p.add_argument("--executor", default="models/checkpoint_executor")
    p.add_argument("--base_model", default="unsloth/Qwen3-0.6B")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--part_a", default="data/train_partA_v3.jsonl")
    p.add_argument("--part_b", default="data/train_partB_v3.jsonl")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--out", default="reports/composed_loop_eval.json")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    questions = load_questions([args.part_a, args.part_b], args.n, args.seed)
    print(f"Held-out questions: {len(questions)} (seed={args.seed})")
    gen = HFGenerator(args.base_model, args.thinker, args.executor, load_in_4bit=args.load_in_4bit)
    orch = ThinkerExecutor(gen, ToolRegistry(), max_steps=args.max_steps)

    scores = []
    for i, q in enumerate(questions, 1):
        try:
            res = orch.run(q["question"], memory_text="")
            scores.append(score_one(res, q["category"], q["expected_answer"]))
        except Exception as e:
            print(f"  [eval] q{i} failed: {e}")
            scores.append({"completed": False, "asked": False, "steps": 0, "exec_parsed": 0,
                           "exec_acts": 0, "fid_ok": 0, "fid_total": 0, "is_math": False, "math_correct": None})
        if i % 10 == 0:
            print(f"  {i}/{len(questions)}")

    agg = aggregate(scores)
    print("\n=== Composed-loop eval ===")
    for k, v in agg.items():
        print(f"  {k:18s} {v}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"metrics": agg, "n": len(questions), "seed": args.seed,
                   "thinker": args.thinker, "executor": args.executor}, f, indent=2)
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
