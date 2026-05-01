"""
SFT Math Question Generator (Part B)
======================================
Generates verifiable math/code questions with known correct answers.
These are used by sft_rejection_sampler.py to filter model responses by correctness.

Question types:
  - Arithmetic (multi-step, requires code)
  - Algebra (solve for x)
  - Geometry (area, volume, distance)
  - Statistics (mean, variance, probability)
  - Unit conversions combined with math
  - Word problems with embedded calculations
  - "No-tool" control cases (question asked in session without python_execute)

Usage:
    python sft_math_question_generator.py --count 200 --output data/questions_partB.jsonl
    python sft_math_question_generator.py --count 10 --type arithmetic --output data/sample_math.jsonl
"""

import argparse
import ast
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# Retry config — overridden by CLI args in main()
_MAX_RETRIES: int = 5
_BASE_DELAY: float = 5.0

# ---------------------------------------------------------------------------
# Question type specifications
# ---------------------------------------------------------------------------

MATH_QUESTION_TYPES = {
    "arithmetic": {
        "count": 250,
        "description": "Multi-step arithmetic that requires code execution for precision",
        "examples": [
            {"question": "What is 15% of 847.50, rounded to 2 decimal places?", "answer": "127.13"},
            {"question": "If I invest €5000 at 4.5% annual compound interest for 7 years, what is the total amount?", "answer": "6832.35"},
            {"question": "A train travels 240km at 80km/h, then 180km at 60km/h. What is the average speed for the whole journey?", "answer": "70.59"},
        ],
    },
    "algebra": {
        "count": 150,
        "description": "Equation solving and algebraic manipulation",
        "examples": [
            {"question": "Solve for x: 3x + 15 = 48", "answer": "11"},
            {"question": "If 2x - 7 = x + 12, what is x?", "answer": "19"},
            {"question": "A rectangle has perimeter 56cm and length 18cm. What is its width?", "answer": "10"},
        ],
    },
    "geometry": {
        "count": 150,
        "description": "Area, volume, distance, and geometric calculations",
        "examples": [
            {"question": "What is the area of a circle with radius 7.3cm? Give answer to 2 decimal places.", "answer": "167.42"},
            {"question": "A cylinder has radius 5cm and height 12cm. What is its volume in cm³ to 2 decimal places?", "answer": "942.48"},
            {"question": "What is the distance between points (3, 4) and (9, 12)?", "answer": "10.0"},
        ],
    },
    "statistics": {
        "count": 150,
        "description": "Statistical calculations: mean, variance, probability",
        "examples": [
            {"question": "Find the standard deviation of: 4, 8, 15, 16, 23, 42", "answer": "13.24"},
            {"question": "What is the probability of rolling a sum of 7 with two fair dice?", "answer": "0.1667"},
            {"question": "Calculate the median of: 17, 3, 24, 8, 12, 36, 5", "answer": "12"},
        ],
    },
    "unit_conversion": {
        "count": 100,
        "description": "Unit conversions combined with arithmetic",
        "examples": [
            {"question": "Convert 75 miles per hour to metres per second. Round to 2 decimal places.", "answer": "33.53"},
            {"question": "A room is 15 feet by 12 feet. What is its area in square metres? (1 foot = 0.3048m)", "answer": "16.72"},
        ],
    },
    "word_problems": {
        "count": 150,
        "description": "Real-world word problems requiring multi-step calculation",
        "examples": [
            {"question": "A shop sells apples for €0.45 each and oranges for €0.65 each. If someone buys 7 apples and 5 oranges and pays with a €10 note, how much change do they get?", "answer": "3.40"},
            {"question": "A car uses 8.5 litres of fuel per 100km. Fuel costs €1.75 per litre. How much does it cost to drive 350km?", "answer": "51.63"},
        ],
    },
    "no_tool_control": {
        "count": 100,
        "description": "Math questions asked in a context where the model explicitly has NO python_execute tool. Correct response = acknowledge inability to compute precisely.",
        "examples": [
            {"question": "[NO_TOOLS_SESSION] What is 847.50 multiplied by 15.7 divided by 3.14?", "answer": "NO_TOOL_RESPONSE"},
            {"question": "[NO_TOOLS_SESSION] Calculate the exact compound interest on €10,000 at 3.75% for 8 years.", "answer": "NO_TOOL_RESPONSE"},
        ],
        "special": "no_tool",
    },
}

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

MATH_GENERATION_PROMPT = """Generate {count} math/calculation questions of type: {question_type}

Description: {description}

Example questions and answers:
{examples}

Requirements:
1. Each question must have an exact, verifiable numerical answer
2. Questions should require multi-step calculation (not trivial single operations)
3. Vary the numbers, contexts, and difficulty
4. Use realistic real-world contexts (shopping, finance, science, engineering)
5. Answers should be computable by executing Python code
6. Do NOT include questions about current data (stock prices, exchange rates) — only
   questions with deterministic mathematical answers

{special_instructions}

Return ONLY a JSON array of objects with this exact format:
[
  {{"question": "question text here", "answer": "exact answer here", "type": "{question_type}"}},
  ...
]

For numeric answers: give the number only (e.g., "127.13" not "€127.13 euros").
For probability answers: give to 4 decimal places.
For geometry: give to 2 decimal places unless exact."""

NO_TOOL_INSTRUCTIONS = """SPECIAL: These questions are for training the model to recognise it cannot
compute precisely without a code execution tool. Prefix each question with [NO_TOOLS_SESSION].
Set answer to "NO_TOOL_RESPONSE" for all questions.
The training goal is: model should say "I don't have code execution available, so I can't compute
this precisely. Here is what I can do: [explain manual approach or ask user to run code]"."""

# ---------------------------------------------------------------------------
# Code safety validation (Security Blocker 1 — OWASP LLM01)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})

_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


def _validate_code(code: str) -> tuple[bool, str]:
    """AST-based validator: only math/statistics stdlib imports allowed.
    Blocks os, sys, subprocess, socket, requests and dangerous builtins.
    Returns (is_safe, reason)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORTS:
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
# Answer verification via Python execution
# ---------------------------------------------------------------------------

VERIFICATION_CODE_PROMPT = """Write Python code that computes the answer to this question:
{question}

Expected answer: {answer}

Requirements:
- Code must print ONLY the final numerical answer on the last line
- Use appropriate precision (2 decimal places for money/geometry, 4 for probability)
- No imports needed except math and statistics from stdlib

Return ONLY the Python code, no explanation."""


def verify_answer_with_execution(question: str, answer: str, model: str,
                                  api_base: str | None = None) -> tuple[bool, str]:
    """Ask the LLM to write verification code, execute it, check if output matches answer."""
    if answer == "NO_TOOL_RESPONSE":
        return True, answer  # No verification needed for no-tool examples

    kwargs = dict(
        model=model, max_tokens=512,
        messages=[{"role": "user", "content": VERIFICATION_CODE_PROMPT.format(
            question=question, answer=answer
        )}],
    )
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
    code = code.strip()

    # Validate then execute the LLM-generated verification code
    safe, reason = _validate_code(code)
    if not safe:
        return False, f"unsafe_code: {reason}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False, f"execution_error: {result.stderr[:200]}"

        computed = result.stdout.strip().split("\n")[-1].strip()

        # Numeric comparison with tolerance
        try:
            computed_f = float(computed)
            expected_f = float(answer)
            close_enough = abs(computed_f - expected_f) / max(abs(expected_f), 1e-9) < 0.01
            return close_enough, computed
        except ValueError:
            return computed == answer, computed

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_math_questions(
    question_type: str,
    count: int,
    model: str,
    verify: bool,
    api_base: str | None = None,
) -> list[dict]:
    spec = MATH_QUESTION_TYPES[question_type]
    is_no_tool = spec.get("special") == "no_tool"

    examples_str = "\n".join(
        f"  Q: {e['question']}\n  A: {e['answer']}"
        for e in spec["examples"]
    )

    prompt = MATH_GENERATION_PROMPT.format(
        count=count,
        question_type=question_type,
        description=spec["description"],
        examples=examples_str,
        special_instructions=NO_TOOL_INSTRUCTIONS if is_no_tool else "",
    )

    kwargs = dict(model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}])
    if api_base:
        kwargs["api_base"] = api_base

    for attempt in range(_MAX_RETRIES):
        try:
            response = litellm.completion(**kwargs)
            break
        except litellm.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 2)
            print(f"  Rate limit. Retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s...")
            time.sleep(wait)
        except (litellm.APIConnectionError, litellm.Timeout):
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BASE_DELAY * (2 ** attempt)
            print(f"  Connection error. Retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s...")
            time.sleep(wait)

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions = json.loads(raw)

    if not verify or is_no_tool:
        return questions

    # Verify a random 20% sample for quality control
    verified = []
    sample_indices = random.sample(range(len(questions)), max(1, len(questions) // 5))

    for i, q in enumerate(questions):
        q["verified"] = False
        q["computed_answer"] = None

        if i in sample_indices:
            ok, computed = verify_answer_with_execution(
                q["question"], q["answer"], model, api_base
            )
            q["verified"] = ok
            q["computed_answer"] = computed
            if not ok:
                print(f"  Warning: answer mismatch for '{q['question'][:60]}...' "
                      f"expected={q['answer']} computed={computed}")

        verified.append(q)

    return verified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part B verifiable math questions")
    parser.add_argument("--count", type=int, default=None,
                        help="Questions per type (overrides per-type defaults)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: generate 1 question per type → data/smoke_questions_partB.jsonl")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all"] + list(MATH_QUESTION_TYPES.keys()),
                        help="Question type (default: all)")
    parser.add_argument("--output", type=str, default="data/questions_partB.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                        help="litellm model string (any small/fast model works for math gen)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify a sample of answers by executing Python (default: True)")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip verification (faster but less quality assurance)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Questions per API call")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on rate limit / connection errors (default: 5)")
    parser.add_argument("--base_delay", type=float, default=5.0,
                        help="Base delay in seconds for exponential backoff (default: 5.0)")
    args = parser.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

    if args.smoke:
        args.count = 1
        args.output = "data/smoke_questions_partB.jsonl"
        args.no_verify = True
        print("Smoke test mode: 1 question per type → data/smoke_questions_partB.jsonl (verification skipped)")

    should_verify = args.verify and not args.no_verify
    types_to_run = list(MATH_QUESTION_TYPES.keys()) if args.type == "all" else [args.type]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for qtype in types_to_run:
            target = args.count or MATH_QUESTION_TYPES[qtype]["count"]
            print(f"\n[{qtype}] Generating {target} questions...")

            generated = []
            remaining = target

            while remaining > 0:
                batch = min(args.batch_size, remaining)
                print(f"  Requesting batch of {batch}...")

                try:
                    batch_qs = generate_math_questions(
                        question_type=qtype,
                        count=batch,
                        model=args.model,
                        verify=should_verify,
                        api_base=args.api_base,
                    )
                    generated.extend(batch_qs)
                    remaining -= len(batch_qs)
                    print(f"  Got {len(batch_qs)}. Total: {len(generated)}")
                    time.sleep(0.5)

                except json.JSONDecodeError as e:
                    print(f"  JSON parse error: {e}. Retrying...")
                    time.sleep(2)
                    continue
                except Exception as e:
                    print(f"  Error: {e}. Skipping type.")
                    break

            for item in generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1

            print(f"  [DONE] {len(generated)} questions for {qtype}")

    print(f"\nTotal written: {total_written} → {args.output}")


if __name__ == "__main__":
    main()
