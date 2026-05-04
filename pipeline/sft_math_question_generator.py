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
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

import litellm
from dotenv import load_dotenv

load_dotenv()

# Retry config — overridden by CLI args in main()
_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0

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
    # "no_tool_control": {
    #     "count": 100,
    #     "description": "Math questions asked in a context where the model explicitly has NO python_execute tool. Correct response = acknowledge inability to compute precisely.",
    #     "examples": [
    #         {"question": "[NO_TOOLS_SESSION] What is 847.50 multiplied by 15.7 divided by 3.14?", "answer": "NO_TOOL_RESPONSE"},
    #         {"question": "[NO_TOOLS_SESSION] Calculate the exact compound interest on €10,000 at 3.75% for 8 years.", "answer": "NO_TOOL_RESPONSE"},
    #     ],
    #     "special": "no_tool",
    # },
}

# ---------------------------------------------------------------------------
# Context hints for word problems — cycled per batch to vary cultural settings
# ---------------------------------------------------------------------------

WORD_PROBLEM_CONTEXTS = [
    {"setting": "South Asian household (India/Bangladesh)", "currency": "₹ rupees", "contexts": "groceries, school fees, auto-rickshaw fares, cooking gas, mobile recharge"},
    {"setting": "West African market (Nigeria/Ghana)", "currency": "₦ naira or GH₵ cedis", "contexts": "produce, mobile data bundles, kerosene, transport, building materials"},
    {"setting": "Southeast Asian setting (Philippines/Vietnam)", "currency": "₱ pesos or ₫ dong", "contexts": "jeepney fares, rice sacks, remittances, sari-sari store goods, internet load"},
    {"setting": "Latin American context (Mexico/Brazil)", "currency": "$ MXN pesos or R$ reais", "contexts": "tortillas, petrol, rent, bus fares, mobile plans, market produce"},
    {"setting": "Middle Eastern context (Egypt/Jordan)", "currency": "£E Egyptian pounds or JD dinars", "contexts": "fuel, utilities, school fees, food staples, water tanks"},
    {"setting": "East Asian context (China/South Korea)", "currency": "¥ yuan or ₩ won", "contexts": "transit cards, delivery food, tutoring, online shopping, utility bills"},
    {"setting": "Eastern European setting (Poland/Ukraine)", "currency": "zł złoty or ₴ hryvnia", "contexts": "groceries, utilities, transport, healthcare copay, school supplies"},
    {"setting": "Scandinavian context (Norway/Sweden)", "currency": "kr NOK or SEK", "contexts": "electricity bills, public transit, childcare, fishing equipment, groceries"},
    {"setting": "UK household context", "currency": "£ GBP", "contexts": "council tax, petrol, broadband, supermarket shop, energy bills"},
    {"setting": "East African context (Kenya/Tanzania)", "currency": "KSh shillings", "contexts": "M-Pesa transactions, matatu fares, maize flour, school uniform, mobile data"},
]

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

MATH_DEDUP_TEMPLATE = """DEDUPLICATION — these questions were already generated. Do NOT repeat or closely paraphrase any of them:
{sample}
Generate questions with entirely different scenarios, numbers, and contexts."""

MATH_GENERATION_PROMPT = """Generate {count} math/calculation questions of type: {question_type}

Description: {description}

Example questions and answers:
{examples}

{context_hint}

Requirements:
1. Each question must have an exact, verifiable numerical answer
2. Questions should require multi-step calculation (not trivial single operations)
3. Vary the numbers, contexts, and difficulty
4. Use realistic real-world contexts (shopping, finance, science, engineering)
5. Answers should be computable by executing Python code
6. Do NOT include questions about current data (stock prices, exchange rates) — only
   questions with deterministic mathematical answers

{special_instructions}

{already_generated}

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
                                  api_base: str | None = None) -> tuple[bool, str, str]:
    """Ask the LLM to write verification code, execute it, check if output matches answer.

    Returns (is_correct, computed_answer, verification_code).
    """
    if answer == "NO_TOOL_RESPONSE":
        return True, answer, ""  # No verification needed for no-tool examples

    kwargs = dict(
        model=model, max_tokens=1024,
        timeout=400,
        messages=[{"role": "user", "content": VERIFICATION_CODE_PROMPT.format(
            question=question, answer=answer
        )}],
    )
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    raw_content = response.choices[0].message.content
    if raw_content is None:
        return False, "null_content", ""
    code = raw_content.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
        if code.startswith("python"):
            code = code[6:]
    code = code.strip()

    print(f"    [verify] generated code:\n      {code.replace(chr(10), chr(10) + '      ')}")

    # Validate then execute the LLM-generated verification code
    safe, reason = _validate_code(code)
    if not safe:
        print(f"    [verify] BLOCKED — {reason}")
        return False, f"unsafe_code: {reason}", code

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:200]
            print(f"    [verify] execution error: {err}")
            return False, f"execution_error: {err}", code

        computed = result.stdout.strip().split("\n")[-1].strip()
        print(f"    [verify] executed → computed={computed!r}  expected={answer!r}")

        # Numeric comparison with tolerance
        try:
            computed_f = float(computed)
            expected_f = float(answer)
            close_enough = abs(computed_f - expected_f) / max(abs(expected_f), 1e-9) < 0.01
            status = "✓ MATCH" if close_enough else "✗ MISMATCH"
            print(f"    [verify] {status}")
            return close_enough, computed, code
        except ValueError:
            match = computed == answer
            print(f"    [verify] {'✓ MATCH' if match else '✗ MISMATCH'} (string compare)")
            return match, computed, code

    except subprocess.TimeoutExpired:
        print("    [verify] timeout")
        return False, "timeout", code
    except Exception as e:
        print(f"    [verify] exception: {e}")
        return False, str(e), code


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_math_questions(
    question_type: str,
    count: int,
    model: str,
    verify: bool,
    api_base: str | None = None,
    existing_questions: list[str] | None = None,
    context_hint: dict | None = None,
) -> list[dict]:
    """Generate math questions for a given type.

    existing_questions: question strings already generated in prior batches for this type,
    injected as dedup context so the model avoids repeating problem structures.

    context_hint: for word_problems/no_tool types, a dict from WORD_PROBLEM_CONTEXTS that
    sets the cultural/geographic setting of the word problem scenario.
    """
    spec = MATH_QUESTION_TYPES[question_type]
    is_no_tool = spec.get("special") == "no_tool"

    examples_str = "\n".join(
        f"  Q: {e['question']}\n  A: {e['answer']}"
        for e in spec["examples"]
    )

    # Build dedup block
    if existing_questions:
        sample_qs = existing_questions[-20:]
        sample_text = "\n".join(f"  - {q}" for q in sample_qs)
        already_generated = MATH_DEDUP_TEMPLATE.format(sample=sample_text)
    else:
        already_generated = ""

    # Build context hint — only meaningful for word problems
    if context_hint and question_type in ("word_problems", "no_tool_control"):
        hint_text = (
            f"WORD PROBLEM CONTEXT FOR THIS BATCH — set at least 60% of questions in this context:\n"
            f"  Setting: {context_hint['setting']}\n"
            f"  Currency: {context_hint['currency']}\n"
            f"  Typical items/scenarios: {context_hint['contexts']}\n"
            f"Use locally realistic numbers (e.g. ₹45 for a vegetable, not €45). "
            f"Names in questions should reflect the setting."
        )
    else:
        hint_text = ""

    prompt = MATH_GENERATION_PROMPT.format(
        count=count,
        question_type=question_type,
        description=spec["description"],
        examples=examples_str,
        context_hint=hint_text,
        special_instructions=NO_TOOL_INSTRUCTIONS if is_no_tool else "",
        already_generated=already_generated,
    )

    kwargs = dict(
        model=model,
        max_tokens=4096 * 2,  # high token limit to avoid truncation of verbose/multi-turn responses; retries with smaller batch if it does truncate
        temperature=0.9,
        timeout=400,
        messages=[{"role": "user", "content": prompt}],
    )
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

    content = response.choices[0].message.content
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if content is None:
        raise ValueError(
            f"Model returned null content (finish_reason={finish_reason!r}). "
            "Response was likely truncated — reduce --batch_size."
        )

    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    if finish_reason == "length":
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(
                "Response truncated at token limit (finish_reason='length'). "
                "Reduce --batch_size and retry."
            )
    else:
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
            print(f"  Verifying: '{q['question'][:70]}...'")
            ok, computed, vcode = verify_answer_with_execution(
                q["question"], q["answer"], model, api_base
            )
            q["verified"] = ok
            q["computed_answer"] = computed
            q["verification_code"] = vcode
            if not ok:
                print(f"  Warning: answer mismatch for '{q['question'][:60]}...' "
                      f"expected={q['answer']} computed={computed}")

        verified.append(q)

    return verified


# ---------------------------------------------------------------------------
# Per-type worker (used by both sequential and parallel modes)
# ---------------------------------------------------------------------------


def _run_type(
    qtype: str,
    target: int,
    batch_size: int,
    model: str,
    api_base: str | None,
    should_verify: bool,
    out_file,
    file_lock: threading.Lock,
    type_idx: int,
    n_types: int,
) -> int:
    """Run the full batch loop for one question type. Thread-safe via file_lock."""
    type_written = 0
    remaining = target
    current_batch_size = batch_size
    batch_num = 0
    type_start = time.monotonic()
    type_questions_seen: list[str] = []
    context_idx = 0

    n_batches = -(-target // batch_size)
    print(f"\n[{type_idx}/{n_types}] {qtype} — target {target} questions "
          f"(~{n_batches} batch{'es' if n_batches != 1 else ''} of {batch_size})")

    while remaining > 0:
        batch = min(current_batch_size, remaining)
        batch_num += 1

        if qtype in ("word_problems", "no_tool_control"):
            ctx = WORD_PROBLEM_CONTEXTS[context_idx % len(WORD_PROBLEM_CONTEXTS)]
            context_idx += 1
            ctx_label = ctx["setting"].split("(")[0].strip()
        else:
            ctx = None
            ctx_label = ""

        label = f", context={ctx_label}" if ctx_label else ""
        print(f"  [{qtype}][batch {batch_num}] requesting {batch} questions "
              f"({remaining} remaining, batch_size={current_batch_size}{label})...")
        t0 = time.monotonic()

        try:
            batch_qs = generate_math_questions(
                question_type=qtype,
                count=batch,
                model=model,
                verify=should_verify,
                api_base=api_base,
                existing_questions=type_questions_seen if type_questions_seen else None,
                context_hint=ctx,
            )
            elapsed = time.monotonic() - t0

            with file_lock:
                for item in batch_qs:
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if isinstance(item.get("question"), str):
                        type_questions_seen.append(item["question"])
                out_file.flush()

            type_written += len(batch_qs)
            remaining -= len(batch_qs)
            print(f"  [{qtype}][batch {batch_num}] ✓ {len(batch_qs)} written in {elapsed:.1f}s "
                  f"(total: {type_written}/{target})")

            if remaining > 0:
                time.sleep(0.5)

        except json.JSONDecodeError as e:
            print(f"  [{qtype}][batch {batch_num}] JSON parse error: {e}. Retrying...")
            time.sleep(2)
            batch_num -= 1
            if qtype in ("word_problems", "no_tool_control"):
                context_idx -= 1
            continue
        except ValueError as e:
            if "truncated" in str(e).lower() or "null content" in str(e).lower():
                new_batch = max(10, current_batch_size // 2)
                print(f"  [{qtype}][batch {batch_num}] Truncation/null — halving: "
                      f"{current_batch_size} → {new_batch}. Retrying...")
                current_batch_size = new_batch
                batch_num -= 1
                if qtype in ("word_problems", "no_tool_control"):
                    context_idx -= 1
                continue
            print(f"  [{qtype}][batch {batch_num}] Error: {e}. Skipping type.")
            break
        except Exception as e:
            print(f"  [{qtype}][batch {batch_num}] Unexpected error: {e}. Skipping type.")
            break

    type_elapsed = time.monotonic() - type_start
    print(f"  [{qtype}] DONE: {type_written}/{target} in {type_elapsed:.1f}s")
    return type_written


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
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Questions per API call (15=safe default; up to 30 for simple math types)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output file instead of appending to it")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on rate limit / connection errors (default: 5)")
    parser.add_argument("--base_delay", type=float, default=3.0,
                        help="Base delay in seconds for exponential backoff (default: 3.0)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel type workers when --type all (default: 5; use 1 to disable)")
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

    file_mode = "w" if args.overwrite else "a"
    if file_mode == "a" and Path(args.output).exists():
        existing = sum(1 for _ in open(args.output, encoding="utf-8"))
        print(f"Appending to existing file: {args.output} ({existing} rows already present; use --overwrite to start fresh)")

    total_written = 0
    n_types = len(types_to_run)
    run_start = time.monotonic()
    file_lock = threading.Lock()

    with open(args.output, file_mode, encoding="utf-8") as f:
        use_parallel = n_types > 1 and not args.smoke and args.workers > 1
        if use_parallel:
            max_workers = min(args.workers, n_types)
            print(f"Running {n_types} types in parallel ({max_workers} workers)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_type,
                        qtype=qtype,
                        target=args.count or MATH_QUESTION_TYPES[qtype]["count"],
                        batch_size=args.batch_size,
                        model=args.model,
                        api_base=args.api_base,
                        should_verify=should_verify,
                        out_file=f,
                        file_lock=file_lock,
                        type_idx=type_idx,
                        n_types=n_types,
                    ): qtype
                    for type_idx, qtype in enumerate(types_to_run, 1)
                }
                for future in concurrent.futures.as_completed(futures):
                    qtype = futures[future]
                    try:
                        written = future.result()
                        total_written += written
                    except Exception as e:
                        print(f"[{qtype}] type thread failed: {e}")
        else:
            for type_idx, qtype in enumerate(types_to_run, 1):
                written = _run_type(
                    qtype=qtype,
                    target=args.count or MATH_QUESTION_TYPES[qtype]["count"],
                    batch_size=args.batch_size,
                    model=args.model,
                    api_base=args.api_base,
                    should_verify=should_verify,
                    out_file=f,
                    file_lock=file_lock,
                    type_idx=type_idx,
                    n_types=n_types,
                )
                total_written += written

    total_elapsed = time.monotonic() - run_start
    print(f"\n{'='*55}")
    print(f"Run complete in {total_elapsed:.1f}s")
    print(f"Questions written this run : {total_written}")
    print(f"Output                     : {args.output}")


if __name__ == "__main__":
    main()
