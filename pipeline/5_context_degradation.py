"""
5_context_degradation.py
========================
Measures how model accuracy degrades as conversation context grows.

The professor's question: "Does SFT maintain results under context bloat?"

This script runs a fixed sequence of turns with KNOWN correct answers,
records correctness at each context length, and identifies the exact token
count where each model starts failing.

Usage:
    # Compare base model vs fine-tuned:
    python pipeline/5_context_degradation.py --compare --model_dir ./models/checkpoint_sft

    # Fine-tuned only:
    python pipeline/5_context_degradation.py --model_dir ./models/checkpoint_sft

Output:
    - pipeline/reports/degradation_<timestamp>.json  (full data)
    - Printed table with per-turn correctness + context token counts
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── Benchmark turns ────────────────────────────────────────────────────────
# Each entry has:
#   question     : what the user asks
#   expected     : the correct value the answer must contain (case-insensitive substring match)
#   requires_tool: whether the model MUST use a tool (to detect tool-mania / tool-avoidance)
#   category     : what failure mode this tests

TURNS = [
    {
        "turn": 1,
        "question": "What is 15 + 27?",
        "expected": "42",
        "requires_tool": False,
        "category": "simple_math",
        "note": "Baseline: trivial arithmetic, short context"
    },
    {
        "turn": 2,
        "question": "Now multiply that result by 3 and subtract 10.",
        "expected": "116",
        "requires_tool": True,
        "category": "cross_turn_reference",
        "note": "Must recall Turn 1 result (42), compute 42×3−10=116"
    },
    {
        "turn": 3,
        "question": "Convert 500 USD to EUR.",
        "expected": "425",
        "requires_tool": True,
        "category": "live_data_tool",
        "note": "Must call get_exchange_rate, not guess; rate=0.85 → 425 EUR"
    },
    {
        "turn": 4,
        "question": "What was the very first number I asked you to compute?",
        "expected": "42",
        "requires_tool": False,
        "category": "long_range_recall",
        "note": "Must recall Turn 1 from growing context — tests attention over distance"
    },
    {
        "turn": 5,
        "question": "Calculate (144 / 12) + (25 * 4) - 37.",
        "expected": "75",
        "requires_tool": True,
        "category": "fresh_computation",
        "note": "Fresh maths, no prior context needed; 12 + 100 - 37 = 75"
    },
    {
        "turn": 6,
        "question": "Add the result of that last calculation to the EUR amount from the currency conversion.",
        "expected": "500",
        "requires_tool": True,
        "category": "multi_reference",
        "note": "Must find Turn 5 (75) AND Turn 3 (425) in context: 75+425=500. Common failure point."
    },
    {
        "turn": 7,
        "question": "If I invest that combined amount at 5% annual interest for 3 years (compound), what is the final value?",
        "expected": "578.81",
        "requires_tool": True,
        "category": "compound_on_prior",
        "note": "Depends on correct Turn 6 (500): 500*(1.05)^3 = 578.81"
    },
    {
        "turn": 8,
        "question": "What is the capital city of Ireland?",
        "expected": "dublin",
        "requires_tool": False,
        "category": "knowledge_no_tool",
        "note": "Pure knowledge — no tool needed. Tests tool-mania: does SFT call get_exchange_rate here?"
    },
    {
        "turn": 9,
        "question": "What is the approximate population of that city?",
        "expected": "1.2",   # 1.2 million or similar
        "requires_tool": False,
        "category": "coreference_knowledge",
        "note": "Must resolve 'that city' = Dublin from Turn 8"
    },
    {
        "turn": 10,
        "question": "What is 999 * 999?",
        "expected": "998001",
        "requires_tool": True,
        "category": "fresh_large_math",
        "note": "Fresh computation at high context length — tests if tool-use still works deep in context"
    },
    {
        "turn": 11,
        "question": "List every numerical result we have computed so far in this conversation.",
        "expected": None,   # No single expected value — check manually
        "requires_tool": False,
        "category": "full_recall",
        "note": "Full context recall: 42, 116, 425, 42, 75, 500, 578.81, 998001. Scored manually."
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
        "note": "280-word distractor + hidden number. Tests attention at peak context load."
    },
]

# ─── System prompts (must match training) ───────────────────────────────────
SYSTEM_WITH_TOOLS = (
    "You are a trustworthy assistant that prioritizes accuracy, transparency, and accountability. "
    "Think step by step using <think>, call tools using <tool>, and give your final answer in <answer>. "
    "You must never guess when tools are needed, always acknowledge uncertainty, ask for missing information "
    "rather than assuming, deny impossible tasks honestly, and present tradeoffs for subjective questions.\n\n"
    "Available tools:\n"
    "1. python_execute(code='...') - Execute Python code and return the output.\n"
    "   Example: <tool>python_execute(code='print(15 + 27)')</tool>\n\n"
    "2. get_exchange_rate(from='USD', to='EUR') - Get currency exchange rates.\n"
    "   Example: <tool>get_exchange_rate(from='USD', to='EUR')</tool>\n\n"
    "To use a tool, wrap your call in <tool></tool> tags."
)


# ─── Tool implementations ────────────────────────────────────────────────────
import subprocess

def python_execute(code: str) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=10
        )
        return (result.stdout.strip() or result.stderr.strip()) or "No output"
    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except Exception as e:
        return f"Error: {e}"


def get_exchange_rate(**kwargs) -> str:
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 155.58}
    f = kwargs.get("from", "").upper()
    t = kwargs.get("to", "").upper()
    if f in rates and t in rates:
        return json.dumps({"rate": round(rates[t] / rates[f], 4)})
    return json.dumps({"error": "Currency not supported"})


def execute_tool(text: str) -> Optional[str]:
    """Extract and execute a tool call from model output. Returns result or None."""
    import re
    match = re.search(r'<tool>(.*?)</tool>', text, re.DOTALL)
    if not match:
        return None
    tool_text = match.group(1).strip()
    func_match = re.match(r'(\w+)\((.*)\)', tool_text, re.DOTALL)
    if not func_match:
        return None
    func_name, args_str = func_match.group(1), func_match.group(2)
    kwargs = {}
    if args_str:
        for m in re.finditer(r"(\w+)=(?:(['\"])((?:\\.|(?!\2).)*?)\2|([^,\)]+))", args_str):
            key = m.group(1)
            val = m.group(3) if m.group(2) else (m.group(4) or "").strip()
            kwargs[key] = val
    if func_name == "python_execute":
        return python_execute(**kwargs)
    if func_name == "get_exchange_rate":
        return get_exchange_rate(**kwargs)
    return f"Unknown tool: {func_name}"


def extract_answer(text: str) -> Optional[str]:
    """Pull content from <answer>...</answer> tags."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def is_correct(answer: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Check if model answer contains the expected value (None = manual scoring needed)."""
    if expected is None or answer is None:
        return None
    return expected.lower() in answer.lower()


def count_tokens(tokenizer, messages: list) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(text))


# ─── Core evaluation loop ────────────────────────────────────────────────────
def run_degradation_eval(model, tokenizer, model_label: str, max_new_tokens: int = 1024) -> list:
    """Run all turns, accumulate context, record correctness per turn."""
    conversation = [{"role": "system", "content": SYSTEM_WITH_TOOLS}]
    results = []

    print(f"\n{'='*70}")
    print(f"  Model: {model_label}")
    print(f"{'='*70}")
    print(f"{'Turn':<6} {'Category':<25} {'Ctx Tokens':<12} {'Tool Called':<13} {'Correct'}")
    print(f"{'-'*6} {'-'*25} {'-'*12} {'-'*13} {'-'*8}")

    for turn_def in TURNS:
        turn_num = turn_def["turn"]
        question  = turn_def["question"]
        expected  = turn_def["expected"]

        conversation.append({"role": "user", "content": question})
        context_tokens_before = count_tokens(tokenizer, conversation)

        # Generate
        prompt_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # Greedy — deterministic, reproducible
            temperature=1.0,          # Ignored when do_sample=False
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})

        tool_called = bool(re.search(r'<tool>', response, re.IGNORECASE))
        tool_result_text = None

        # Execute tool if present
        if tool_called:
            tool_result_text = execute_tool(response)
            if tool_result_text:
                conversation.append({
                    "role": "tool",
                    "content": f"Tool result: {tool_result_text}"
                })
                # One follow-up generation after tool result
                prompt_text2 = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                inputs2 = tokenizer(prompt_text2, return_tensors="pt").to("cuda")
                outputs2 = model.generate(
                    **inputs2,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
                response2 = tokenizer.decode(
                    outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
                )
                conversation.append({"role": "assistant", "content": response2})
                response = response2  # Use follow-up for answer extraction

        answer = extract_answer(response)
        correct = is_correct(answer, expected)
        context_tokens_after = count_tokens(tokenizer, conversation)

        result_row = {
            "turn": turn_num,
            "question": question[:80],
            "category": turn_def["category"],
            "note": turn_def["note"],
            "context_tokens_before": context_tokens_before,
            "context_tokens_after": context_tokens_after,
            "tool_called": tool_called,
            "tool_result": tool_result_text,
            "requires_tool": turn_def["requires_tool"],
            "tool_used_correctly": tool_called if turn_def["requires_tool"] else (not tool_called),
            "raw_response": response,
            "extracted_answer": answer,
            "expected": expected,
            "correct": correct,
        }
        results.append(result_row)

        correct_str = "✓" if correct is True else ("✗" if correct is False else "manual")
        print(
            f"{turn_num:<6} {turn_def['category']:<25} {context_tokens_after:<12} "
            f"{'yes' if tool_called else 'no':<13} {correct_str}"
        )

    # Summary
    scored = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in scored if r["correct"])
    print(f"\n  Score: {n_correct}/{len(scored)} auto-scored turns correct")

    # Find first failure
    for r in results:
        if r["correct"] is False:
            print(f"  First failure: Turn {r['turn']} ({r['category']}) at {r['context_tokens_after']} tokens")
            break

    # Find tool-mania instances (tool called when not needed)
    mania = [r for r in results if r["tool_called"] and not r["requires_tool"]]
    if mania:
        print(f"  Tool-mania detected at turns: {[r['turn'] for r in mania]} (tool called on knowledge questions)")

    return results


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Context degradation benchmark")
    parser.add_argument("--model_dir", default="./models/checkpoint_sft",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                        help="Base model identifier")
    parser.add_argument("--compare", action="store_true",
                        help="Also run base model for comparison")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", default="./pipeline/reports")
    args = parser.parse_args()

    from unsloth import FastModel

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"degradation_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "description": "Context degradation benchmark — measures accuracy vs context length",
        "config": vars(args),
        "turns_definition": TURNS,
        "models": {}
    }

    # ── Fine-tuned model ──────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    if model_dir.exists():
        print(f"\nLoading fine-tuned model: {model_dir}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        FastModel.for_inference(model)
        results = run_degradation_eval(model, tokenizer, "SFT Fine-tuned", args.max_new_tokens)
        report["models"]["sft_finetuned"] = results
        del model, tokenizer

    # ── Base model ────────────────────────────────────────────────────────
    if args.compare:
        print(f"\nLoading base model: {args.base_model}")
        base_model, base_tokenizer = FastModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        FastModel.for_inference(base_model)
        base_results = run_degradation_eval(base_model, base_tokenizer, "Base (No SFT)", args.max_new_tokens)
        report["models"]["base"] = base_results
        del base_model, base_tokenizer

    # ── Comparison summary ────────────────────────────────────────────────
    if "sft_finetuned" in report["models"] and "base" in report["models"]:
        print(f"\n{'='*70}")
        print("  HEAD-TO-HEAD COMPARISON")
        print(f"{'='*70}")
        print(f"{'Turn':<6} {'Category':<25} {'Base':<10} {'SFT':<10} {'Ctx Tokens'}")
        print(f"{'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
        for base_r, sft_r in zip(report["models"]["base"], report["models"]["sft_finetuned"]):
            b = "✓" if base_r["correct"] is True else ("✗" if base_r["correct"] is False else "?")
            s = "✓" if sft_r["correct"]  is True else ("✗" if sft_r["correct"]  is False else "?")
            ctx = sft_r["context_tokens_after"]
            print(f"{base_r['turn']:<6} {base_r['category']:<25} {b:<10} {s:<10} {ctx}")

    # ── Save report ────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {output_path}")
    print("\nThesis note: Use 'context_tokens_after' vs 'correct' columns to plot degradation curve.")


if __name__ == "__main__":
    main()
