"""
5_context_degradation.py
========================
Measures how model accuracy degrades as conversation context grows.

Thesis question: "Does SFT maintain results under context bloat?"

Runs a fixed 12-turn sequence with KNOWN correct answers against the inference
server (3_infererence.py). Uses greedy decoding (temperature=0, do_sample=False)
for deterministic, reproducible results. Records accuracy and context size at
each turn to produce the degradation curve.

Rationale for keeping separate from 4_benchmark.py:
  - Different mode: greedy (deterministic) vs sampled (benchmark)
  - Different scoring: expected correct answers per turn vs format/quality checks
  - Different thesis experiment: degradation curve vs capability snapshot
  - Output: accuracy-vs-context-tokens plot data

Usage:
    # Start the inference server first:
    python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

    # Fine-tuned model only:
    python pipeline/5_context_degradation.py --server_url http://localhost:8000

    # Base vs fine-tuned comparison (start base model server on port 8001):
    python pipeline/3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001
    python pipeline/5_context_degradation.py \\
        --server_url http://localhost:8000 \\
        --compare_url http://localhost:8001

Output:
    pipeline/reports/degradation_<timestamp>.json   (full turn-by-turn data)
    Printed table: turn | category | ctx_tokens | tool_called | correct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Turn definitions — fixed sequence with known correct answers
#
# Each entry:
#   question      : user prompt
#   expected      : substring that must appear in <answer> (None = manual)
#   requires_tool : model MUST call a tool (used to detect tool-mania/avoidance)
#   category      : failure-mode label for the thesis degradation chart
# ---------------------------------------------------------------------------

TURNS: List[Dict[str, Any]] = [
    {
        "turn": 1,
        "question": "What is 15 + 27?",
        "expected": "42",
        "requires_tool": False,
        "category": "simple_math",
        "note": "Baseline — trivial arithmetic, minimal context",
    },
    {
        "turn": 2,
        "question": "Now multiply that result by 3 and subtract 10.",
        "expected": "116",
        "requires_tool": True,
        "category": "cross_turn_reference",
        "note": "Must recall Turn 1 (42): 42×3−10=116",
    },
    {
        "turn": 3,
        "question": "Convert 500 USD to EUR using the current exchange rate.",
        "expected": None,
        "requires_tool": True,
        "category": "live_data_tool",
        "note": "Must call web_search to fetch the live USD/EUR rate, then python_execute to compute the conversion. No hardcoded rate accepted.",
    },
    {
        "turn": 4,
        "question": "What was the very first number I asked you to compute?",
        "expected": "42",
        "requires_tool": False,
        "category": "long_range_recall",
        "note": "Must recall Turn 1 from growing context — tests attention over distance",
    },
    {
        "turn": 5,
        "question": "Calculate (144 / 12) + (25 * 4) - 37.",
        "expected": "75",
        "requires_tool": True,
        "category": "fresh_computation",
        "note": "12 + 100 − 37 = 75. No prior context needed.",
    },
    {
        "turn": 6,
        "question": "Add the result of that last calculation to the EUR amount from the currency conversion.",
        "expected": "500",
        "requires_tool": True,
        "category": "multi_reference",
        "note": "Must find Turn 5 (75) AND Turn 3 (425): 75+425=500. Common failure point.",
    },
    {
        "turn": 7,
        "question": "If I invest that combined amount at 5% annual interest for 3 years (compound), what is the final value?",
        "expected": "578.81",
        "requires_tool": True,
        "category": "compound_on_prior",
        "note": "Depends on Turn 6 (500): 500×(1.05)³ = 578.8125",
    },
    {
        "turn": 8,
        "question": "What is the capital city of Ireland?",
        "expected": "dublin",
        "requires_tool": False,
        "category": "knowledge_no_tool",
        "note": "Pure knowledge — tests tool-mania (model must NOT call any tool here; answer from parametric memory only)",
    },
    {
        "turn": 9,
        "question": "What is the approximate population of that city?",
        "expected": "1.2",
        "requires_tool": False,
        "category": "coreference_knowledge",
        "note": "Must resolve 'that city' = Dublin from Turn 8",
    },
    {
        "turn": 10,
        "question": "What is 999 * 999?",
        "expected": "998001",
        "requires_tool": True,
        "category": "fresh_large_math",
        "note": "Fresh computation at high context length — tests tool-use still works deep in context",
    },
    {
        "turn": 11,
        "question": "List every numerical result we have computed so far in this conversation.",
        "expected": None,
        "requires_tool": False,
        "category": "full_recall",
        "note": "Full context recall: 42, 116, 425, 42, 75, 500, 578.81, 998001. Scored manually.",
    },
    {
        "turn": 12,
        "question": (
            "Here is a passage: '"
            + ("The quick brown fox jumps over the lazy dog. " * 40)
            + "' My secret number is 7391. "
            "What is the secret number I just gave you?"
        ),
        "expected": "7391",
        "requires_tool": False,
        "category": "needle_in_haystack",
        "note": "280-word distractor + hidden number. Tests attention at peak context load.",
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http(server_url: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
    url = server_url.rstrip("/") + path
    try:
        if body is not None:
            r = requests.post(url, json=body, timeout=180)
        else:
            r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach server at {server_url}. Is 3_infererence.py running?")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Server error {r.status_code}: {r.text}") from e


def _complete(server_url: str, messages: List[Dict], greedy: bool = True) -> Dict[str, Any]:
    return _http(server_url, "/v1/chat/completions", {
        "messages": messages,
        "tool_profile": "all_tools",
        "max_new_tokens": 1024,
        "temperature": 0.7,          # ignored when greedy=True
        "greedy": greedy,
        "max_tool_iterations": 3,    # tight limit — degradation turns need single tool calls
    })

# ---------------------------------------------------------------------------
# Answer extraction and scoring
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> Optional[str]:
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _is_correct(answer: Optional[str], expected: Optional[str]) -> Optional[bool]:
    if expected is None or answer is None:
        return None
    return expected.lower() in answer.lower()


def _tool_names(text: str) -> List[str]:
    import re
    return re.findall(r"<tool>\s*(\w+)\s*\(", text)

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_degradation_eval(server_url: str, model_label: str,
                          greedy: bool = True) -> List[Dict[str, Any]]:
    """Run all 12 turns sequentially, accumulating conversation context via the server.
    Uses greedy decoding for reproducibility."""

    print(f"\n{'='*75}")
    print(f"  {model_label}  ({'greedy' if greedy else 'sampled'})")
    print(f"{'='*75}")
    print(f"{'T':<4} {'Category':<26} {'Ctx Toks':<10} {'Tool':<6} {'Expected':<10} {'Correct'}")
    print(f"{'-'*4} {'-'*26} {'-'*10} {'-'*6} {'-'*10} {'-'*8}")

    history: List[Dict] = []
    results: List[Dict[str, Any]] = []

    for turn_def in TURNS:
        q = turn_def["question"]
        history.append({"role": "user", "content": q})

        result = _complete(server_url, history, greedy=greedy)
        response = result["response"]
        metrics = result.get("metrics", {})

        history.append({"role": "assistant", "content": response})

        tools_called = _tool_names(response)
        answer = _extract_answer(response)
        correct = _is_correct(answer, turn_def["expected"])

        # tool_used_correctly: required → must call, not-required → must not call
        tool_used_correctly = (
            bool(tools_called) if turn_def["requires_tool"] else not bool(tools_called)
        )

        row = {
            "turn": turn_def["turn"],
            "question": q[:80],
            "category": turn_def["category"],
            "note": turn_def["note"],
            "input_tokens": metrics.get("input_tokens", 0),      # context size before this response
            "tokens_generated": metrics.get("tokens_generated", 0),
            "latency_s": metrics.get("latency_s", 0),
            "tools_called": tools_called,
            "requires_tool": turn_def["requires_tool"],
            "tool_used_correctly": tool_used_correctly,
            "response_preview": response[:300],
            "extracted_answer": answer,
            "expected": turn_def["expected"],
            "correct": correct,
        }
        results.append(row)

        correct_str = "✓" if correct is True else ("✗" if correct is False else "manual")
        tool_str = ",".join(tools_called)[:5] if tools_called else "none"
        expected_str = (turn_def["expected"] or "–")[:9]
        print(
            f"{turn_def['turn']:<4} {turn_def['category']:<26} "
            f"{metrics.get('input_tokens', 0):<10} {tool_str:<6} {expected_str:<10} {correct_str}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    scored = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in scored if r["correct"])
    print(f"\n  Score: {n_correct}/{len(scored)} auto-scored turns correct  "
          f"({n_correct/len(scored):.0%})")

    first_fail = next((r for r in results if r["correct"] is False), None)
    if first_fail:
        print(f"  First failure: Turn {first_fail['turn']} ({first_fail['category']}) "
              f"at {first_fail['input_tokens']} input tokens")
    else:
        print("  No auto-scored failures detected.")

    tool_mania = [r for r in results if r["tools_called"] and not r["requires_tool"]]
    if tool_mania:
        print(f"  Tool-mania turns: {[r['turn'] for r in tool_mania]} "
              f"(tool called on knowledge questions)")

    return results

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _print_comparison(runs: Dict[str, List[Dict[str, Any]]]) -> None:
    if len(runs) < 2:
        return
    labels = list(runs.keys())
    print(f"\n{'='*75}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*75}")
    print(f"{'T':<4} {'Category':<26} {labels[0][:14]:<16} {labels[1][:14]:<16} {'Ctx Toks'}")
    print(f"{'-'*4} {'-'*26} {'-'*16} {'-'*16} {'-'*10}")
    for r0, r1 in zip(runs[labels[0]], runs[labels[1]]):
        def sym(r):
            return "✓" if r["correct"] is True else ("✗" if r["correct"] is False else "?")
        ctx = r1["input_tokens"]
        print(f"{r0['turn']:<4} {r0['category']:<26} {sym(r0):<16} {sym(r1):<16} {ctx}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context degradation benchmark — accuracy vs context length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tuned model only
  python pipeline/5_context_degradation.py --server_url http://localhost:8000

  # Base vs fine-tuned comparison (start two servers)
  python pipeline/3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001
  python pipeline/5_context_degradation.py \\
      --server_url http://localhost:8000 \\
      --compare_url http://localhost:8001

  # Sampled (non-greedy) run for variance analysis
  python pipeline/5_context_degradation.py --no_greedy
""",
    )
    parser.add_argument("--server_url", default="http://localhost:8000",
                        help="Primary server URL (fine-tuned model)")
    parser.add_argument("--compare_url", default=None,
                        help="Second server URL (e.g. base model on port 8001)")
    parser.add_argument("--no_greedy", action="store_true",
                        help="Use sampled decoding instead of greedy (less reproducible but more realistic)")
    parser.add_argument("--output_dir", default="./pipeline/reports")
    args = parser.parse_args()

    greedy = not args.no_greedy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify primary server
    try:
        h = _http(args.server_url, "/health")
        print(f"Primary server : {args.server_url}  model={h.get('model', '?')}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    compare_label = None
    if args.compare_url:
        try:
            h2 = _http(args.compare_url, "/health")
            compare_label = h2.get("model", args.compare_url)
            print(f"Compare server : {args.compare_url}  model={compare_label}")
        except RuntimeError as e:
            print(f"WARNING: compare server unreachable — {e}. Running single-server mode.")
            args.compare_url = None

    report: Dict[str, Any] = {
        "timestamp": timestamp,
        "description": "Context degradation benchmark — accuracy vs context length (greedy decoding)",
        "greedy": greedy,
        "turns_definition": [{k: v for k, v in t.items() if k != "question"} for t in TURNS],
        "models": {},
    }

    primary_label = h.get("model", args.server_url)
    primary_results = run_degradation_eval(args.server_url, primary_label, greedy=greedy)
    report["models"][primary_label] = primary_results

    if args.compare_url and compare_label:
        compare_results = run_degradation_eval(args.compare_url, compare_label, greedy=greedy)
        report["models"][compare_label] = compare_results
        _print_comparison(report["models"])

    output_path = output_dir / f"degradation_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved → {output_path}")
    print("\nThesis: plot 'input_tokens' vs 'correct' per turn for the degradation curve.")
    print("Ablation: run this script against checkpoints A, B, C, D for the four conditions.")


if __name__ == "__main__":
    main()
