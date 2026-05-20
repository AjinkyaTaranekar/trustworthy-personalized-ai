"""
Experiment 0 — Reasoning Paradigm Comparison
=============================================
researchplan.tex Phase 3 (Dec 2025 – Mar 2026).

Compares four reasoning strategies on Qwen3-0.6B (base model, no fine-tuning)
against the GSM8K grade-school maths benchmark and a custom logic puzzle set.
Must be run BEFORE GRPO training to establish the baseline and decide which
reasoning format the Reasoning Module uses.

Four strategies:
  baseline     Direct answer — no think tags, no explicit reasoning
  cot          Chain-of-Thought prompting (Wei et al. 2022)
  interleaved  Native CAPABILITY_CHECK format (what the model is trained on)
  tot          Tree-of-Thoughts (simulated): generate N candidates, self-rank, pick best

Usage:
    # Start the inference server first (base model or SFT checkpoint):
    python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000

    # Run full comparison (default 100 GSM8K questions):
    python experiment0_reasoning_comparison.py

    # Smoke test — 10 questions, all strategies:
    python experiment0_reasoning_comparison.py --smoke

    # Single strategy:
    python experiment0_reasoning_comparison.py --strategy interleaved --n 50

Dependencies: requests, datasets
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Strategy system prompts
# ---------------------------------------------------------------------------

_SYS_BASELINE = (
    "You are a helpful AI assistant. Answer the question directly and concisely. "
    "Give your final answer in <answer>...</answer> tags. "
    "Do NOT use thinking tags or long reasoning chains."
)

_SYS_COT = (
    "You are a helpful AI assistant. Think step by step before answering. "
    "Show your full reasoning process, then give your final answer in <answer>...</answer> tags."
)

_SYS_INTERLEAVED = (
    "You are a trustworthy AI assistant. Before answering any question, complete a "
    "CAPABILITY_CHECK inside your <think> block:\n"
    "  1. What does this question require?\n"
    "  2. Which session tools can help? (python_execute is available)\n"
    "  3. Is there a gap? If so, how will you handle it honestly?\n\n"
    "Session tools: python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✗\n\n"
    "For arithmetic/calculation: use <tool>python_execute(code='...')</tool>.\n"
    "Format: <think>CAPABILITY_CHECK ... reasoning ...</think><answer>...</answer>"
)

_SYS_TOT_GENERATE = (
    "You are a helpful AI assistant solving a problem. "
    "Generate a complete solution attempt. Show your working. "
    "End with your answer in <answer>...</answer> tags."
)

_SYS_TOT_RANK = (
    "You are evaluating solution attempts for correctness. "
    "Each attempt is labelled A, B, C. "
    "Pick the most likely correct answer. Reply with ONLY the letter (A, B, or C)."
)

STRATEGIES = {
    "baseline":    {"system": _SYS_BASELINE,    "tool_profile": "no_tools",    "tot": False},
    "cot":         {"system": _SYS_COT,          "tool_profile": "no_tools",    "tot": False},
    "interleaved": {"system": _SYS_INTERLEAVED,  "tool_profile": "compute_only","tot": False},
    "tot":         {"system": _SYS_TOT_GENERATE, "tool_profile": "no_tools",    "tot": True, "candidates": 3},
}

# ---------------------------------------------------------------------------
# Logic puzzles (no external data needed, deterministic)
# ---------------------------------------------------------------------------

LOGIC_PUZZLES = [
    {
        "id": "LP01",
        "question": "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "answer": "0.05",
        "answer_variants": ["5 cents", "$0.05", "0.05", "five cents"],
    },
    {
        "id": "LP02",
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?",
        "answer": "5",
        "answer_variants": ["5", "5 minutes", "five"],
    },
    {
        "id": "LP03",
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days does it take to cover half the lake?",
        "answer": "47",
        "answer_variants": ["47", "47 days", "forty-seven"],
    },
    {
        "id": "LP04",
        "question": "Three missionaries and three cannibals need to cross a river. The boat holds at most two people. Cannibals must never outnumber missionaries on either bank. What is the minimum number of one-way trips needed?",
        "answer": "11",
        "answer_variants": ["11", "eleven", "11 trips"],
    },
    {
        "id": "LP05",
        "question": "You have two ropes, each of which burns completely in exactly 60 minutes (but not at a uniform rate). How do you measure exactly 45 minutes?",
        "answer": "light both ends of one rope and one end of the other simultaneously",
        "answer_variants": ["light both ends", "both ends of one", "45"],
        "open_ended": True,
    },
    {
        "id": "LP06",
        "question": "If all Bloops are Razzies, and all Razzies are Lazzies, are all Bloops definitely Lazzies?",
        "answer": "yes",
        "answer_variants": ["yes", "yes, all bloops are lazzies"],
    },
    {
        "id": "LP07",
        "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "answer": "9",
        "answer_variants": ["9", "nine"],
    },
    {
        "id": "LP08",
        "question": "You are running a race and you overtake the person in second place. What place are you in now?",
        "answer": "second",
        "answer_variants": ["second", "2nd", "2"],
    },
    {
        "id": "LP09",
        "question": "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?",
        "answer": "fill 5, pour into 3, empty 3, pour remaining 2 into 3, fill 5, pour 1 into 3",
        "answer_variants": ["4 gallons", "fill the 5", "two steps"],
        "open_ended": True,
    },
    {
        "id": "LP10",
        "question": "Alice is taller than Bob. Charlie is shorter than Alice. Bob is taller than Charlie. Who is the tallest?",
        "answer": "Alice",
        "answer_variants": ["alice", "Alice"],
    },
]


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def _complete(server_url: str, messages: List[Dict], system_override: str,
              tool_profile: str, max_tokens: int = 512,
              temperature: float = 0.7) -> Dict[str, Any]:
    try:
        r = requests.post(
            server_url.rstrip("/") + "/v1/chat/completions",
            json={
                "messages": messages,
                "system_override": system_override,
                "tool_profile": tool_profile,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach server at {server_url}. Is 3_infererence.py running?")


def _extract_answer(response: str) -> str:
    """Extract the last number or final <answer> tag from a response."""
    # Try <answer> tag first
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
        # Try to extract just the number from within the answer
        num = re.search(r"[-+]?\d[\d,]*\.?\d*", text.replace(",", ""))
        return num.group().replace(",", "") if num else text

    # Fall back: last number in response
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response.replace(",", ""))
    return nums[-1].replace(",", "") if nums else ""


def _score_answer(predicted: str, expected: str,
                  variants: Optional[List[str]] = None,
                  open_ended: bool = False) -> float:
    """Return 1.0 (correct) or 0.0 (wrong)."""
    if open_ended:
        # For open-ended: check if any keyword from variants appears
        pred_lower = predicted.lower()
        for v in (variants or []):
            if v.lower() in pred_lower:
                return 1.0
        return 0.0

    # Exact numeric match with tolerance
    try:
        pf = float(predicted.replace(",", "").strip())
        ef = float(expected.replace(",", "").strip())
        if abs(ef) < 1e-9:
            return 1.0 if abs(pf) < 1e-6 else 0.0
        return 1.0 if abs(pf - ef) / abs(ef) < 0.01 else 0.0
    except (ValueError, ZeroDivisionError):
        pass

    # String match against variants
    pred_lower = predicted.lower().strip()
    candidates = [expected.lower().strip()] + [v.lower().strip() for v in (variants or [])]
    return 1.0 if pred_lower in candidates else 0.0


def _has_cap_check(response: str) -> bool:
    return bool(re.search(r"capability[_\s-]?check", response, re.IGNORECASE))


def _has_think(response: str) -> bool:
    return bool(re.search(r"<think>", response, re.IGNORECASE))


def _has_answer_tag(response: str) -> bool:
    return bool(re.search(r"<answer>", response, re.IGNORECASE))


def _used_tool(response: str) -> bool:
    return bool(re.search(r"<tool>", response, re.IGNORECASE))


# ---------------------------------------------------------------------------
# ToT runner
# ---------------------------------------------------------------------------

def _run_tot(server_url: str, question: str, n_candidates: int = 3,
             max_tokens: int = 512, temperature: float = 0.9) -> Tuple[str, List[str]]:
    """Generate n_candidates solutions then self-rank. Return (best_response, all_candidates)."""
    candidates = []
    for _ in range(n_candidates):
        res = _complete(
            server_url,
            [{"role": "user", "content": question}],
            _SYS_TOT_GENERATE,
            "no_tools",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        candidates.append(res["response"])

    if len(candidates) == 1:
        return candidates[0], candidates

    # Build ranking prompt
    rank_prompt = "Which of the following solution attempts is most likely correct?\n\n"
    for i, c in enumerate(candidates):
        label = chr(65 + i)  # A, B, C, ...
        # Show just the answer portion if available
        ans = _extract_answer(c) or c[-200:]
        rank_prompt += f"Attempt {label}: {ans}\n\n"
    rank_prompt += "Reply with ONLY the letter of the most likely correct attempt."

    try:
        rank_res = _complete(
            server_url,
            [{"role": "user", "content": rank_prompt}],
            _SYS_TOT_RANK,
            "no_tools",
            max_tokens=10,
            temperature=0.0,
        )
        chosen_letter = rank_res["response"].strip().upper()
        idx = ord(chosen_letter[0]) - 65 if chosen_letter and chosen_letter[0].isalpha() else 0
        idx = max(0, min(idx, len(candidates) - 1))
    except Exception:
        idx = 0

    return candidates[idx], candidates


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_strategy(
    server_url: str,
    strategy_name: str,
    questions: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    cfg = STRATEGIES[strategy_name]
    results = []
    total_latency = 0.0
    total_tokens = 0

    print(f"\n  [{strategy_name.upper()}]  {len(questions)} questions...")

    for i, q in enumerate(questions, 1):
        question_text = q["question"]
        expected = str(q.get("answer", ""))
        variants = q.get("answer_variants", [])
        open_ended = q.get("open_ended", False)
        t0 = time.perf_counter()

        try:
            if cfg["tot"]:
                response, all_candidates = _run_tot(
                    server_url, question_text,
                    n_candidates=cfg.get("candidates", 3),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                metrics = {}
            else:
                res = _complete(
                    server_url,
                    [{"role": "user", "content": question_text}],
                    cfg["system"],
                    cfg["tool_profile"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response = res["response"]
                metrics = res.get("metrics", {})

            latency = time.perf_counter() - t0
            total_latency += latency
            total_tokens += metrics.get("tokens_generated", 0)

            predicted = _extract_answer(response)
            score = _score_answer(predicted, expected, variants, open_ended)

            results.append({
                "id":        q.get("id", f"Q{i:03d}"),
                "question":  question_text[:100],
                "expected":  expected,
                "predicted": predicted,
                "score":     score,
                "latency_s": round(latency, 3),
                "tokens":    metrics.get("tokens_generated", 0),
                "has_think":  _has_think(response),
                "has_cap_check": _has_cap_check(response),
                "has_answer_tag": _has_answer_tag(response),
                "used_tool":  _used_tool(response),
            })

            tick = "✓" if score else "✗"
            if i % 10 == 0 or i <= 3:
                print(f"    [{i:3d}] {tick}  expected={expected[:20]}  predicted={predicted[:20]}"
                      f"  {latency:.1f}s")

        except Exception as e:
            print(f"    [{i:3d}] ERROR: {e}")
            results.append({
                "id": q.get("id", f"Q{i:03d}"),
                "question": question_text[:100],
                "expected": expected,
                "predicted": "ERROR",
                "score": 0.0,
                "latency_s": time.perf_counter() - t0,
                "error": str(e),
            })

    n = len(results)
    n_scored = sum(1 for r in results if "error" not in r)
    accuracy = sum(r["score"] for r in results) / n if n else 0.0
    cap_check_rate = sum(1 for r in results if r.get("has_cap_check")) / n_scored if n_scored else 0.0
    tool_rate = sum(1 for r in results if r.get("used_tool")) / n_scored if n_scored else 0.0
    answer_tag_rate = sum(1 for r in results if r.get("has_answer_tag")) / n_scored if n_scored else 0.0
    avg_latency = total_latency / n if n else 0.0

    summary = {
        "strategy":        strategy_name,
        "n_questions":     n,
        "accuracy":        round(accuracy, 4),
        "cap_check_rate":  round(cap_check_rate, 4),
        "tool_use_rate":   round(tool_rate, 4),
        "answer_tag_rate": round(answer_tag_rate, 4),
        "avg_latency_s":   round(avg_latency, 3),
        "total_tokens":    total_tokens,
    }

    print(f"\n    Accuracy        : {accuracy:.1%}  ({int(accuracy*n)}/{n})")
    print(f"    CAPABILITY_CHECK: {cap_check_rate:.1%}")
    print(f"    Tool use        : {tool_rate:.1%}")
    print(f"    Answer tag      : {answer_tag_rate:.1%}")
    print(f"    Avg latency     : {avg_latency:.2f}s")

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(all_results: Dict[str, Dict]) -> None:
    print(f"\n{'='*70}")
    print("  EXPERIMENT 0 — REASONING PARADIGM COMPARISON")
    print(f"{'='*70}")
    cols = list(all_results.keys())
    w = 14

    header = f"{'Metric':<28}" + "".join(f"{c.upper():>{w}}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = [
        ("Accuracy",         lambda s: f"{s['accuracy']:.1%}"),
        ("CAPABILITY_CHECK", lambda s: f"{s['cap_check_rate']:.1%}"),
        ("Tool use rate",    lambda s: f"{s['tool_use_rate']:.1%}"),
        ("Answer-tag rate",  lambda s: f"{s['answer_tag_rate']:.1%}"),
        ("Avg latency (s)",  lambda s: f"{s['avg_latency_s']:.2f}"),
        ("Total tokens",     lambda s: f"{s['total_tokens']:,}"),
    ]

    for label, fmt in rows:
        line = f"{label:<28}"
        for c in cols:
            try:
                line += f"{fmt(all_results[c]['summary']):>{w}}"
            except Exception:
                line += f"{'N/A':>{w}}"
        print(line)

    # Winner (accuracy)
    best = max(cols, key=lambda c: all_results[c]["summary"].get("accuracy", 0))
    print(f"\n  Best accuracy: {best.upper()}  "
          f"({all_results[best]['summary']['accuracy']:.1%})")
    print(f"\n  Recommendation for Reasoning Module: "
          f"use '{best}' format as the GRPO training target.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 0 — Reasoning paradigm comparison (researchplan.tex Phase 3)",
    )
    ap.add_argument("--server_url", default="http://localhost:8000",
                    help="Inference server URL (default: http://localhost:8000)")
    ap.add_argument("--strategy", default="all",
                    choices=["all", "baseline", "cot", "interleaved", "tot"],
                    help="Strategy to evaluate (default: all)")
    ap.add_argument("--benchmark", default="all",
                    choices=["all", "gsm8k", "logic"],
                    help="Benchmark to run (default: all)")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of GSM8K questions (default: 100)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 10 questions, all strategies")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output_dir", default="./reports")
    args = ap.parse_args()

    if args.smoke:
        args.n = 10
        args.strategy = "all"
        print("Smoke test mode: 10 questions, all strategies")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Verify server
    try:
        health = requests.get(args.server_url.rstrip("/") + "/health", timeout=10).json()
        print(f"Server: {args.server_url}  model={health.get('model','?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server — {e}")
        return

    # Build question set
    questions: List[Dict] = []

    if args.benchmark in ("all", "gsm8k"):
        try:
            from datasets import load_dataset
            print(f"\nLoading GSM8K (test split, {args.n} questions)...")
            gsm = load_dataset("openai/gsm8k", "main", split=f"test[:{args.n}]",
                               trust_remote_code=True)
            for i, item in enumerate(gsm):
                # GSM8K answers end with "#### <number>"
                answer_raw = item["answer"]
                match = re.search(r"####\s*([-\d,]+)", answer_raw)
                answer = match.group(1).replace(",", "") if match else answer_raw.strip()
                questions.append({
                    "id": f"GSM{i:04d}",
                    "question": item["question"],
                    "answer": answer,
                    "source": "gsm8k",
                })
        except ImportError:
            print("WARNING: 'datasets' not installed — skipping GSM8K. Install: pip install datasets")
        except Exception as e:
            print(f"WARNING: Could not load GSM8K — {e}")

    if args.benchmark in ("all", "logic"):
        print(f"Adding {len(LOGIC_PUZZLES)} logic puzzles...")
        questions.extend(LOGIC_PUZZLES)

    if not questions:
        print("ERROR: No questions available. Install 'datasets' or use --benchmark logic")
        return

    print(f"\nTotal questions: {len(questions)}")

    # Determine strategies to run
    strats = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]

    print(f"\n{'='*70}")
    print(f"  Running {len(strats)} strategy/strategies on {len(questions)} questions")
    print(f"{'='*70}")

    all_results: Dict[str, Any] = {}
    for strat in strats:
        all_results[strat] = run_strategy(
            args.server_url, strat, questions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    print_comparison_table(all_results)

    # Save report
    report = {
        "timestamp":   timestamp,
        "server_url":  args.server_url,
        "model":       health.get("model", "unknown"),
        "n_questions": len(questions),
        "strategies":  strats,
        "results":     all_results,
    }
    out_path = output_dir / f"experiment0_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
