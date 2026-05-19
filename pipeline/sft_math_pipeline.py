"""
SFT Math Pipeline — Single-Process Verified Training Data Generator
====================================================================
Replaces sft_math_question_generator.py + sft_rejection_sampler.py.

Old approach (fragile):
  LLM generates question + answer  →  answer can be hallucinated
  Another LLM verifies with code   →  two sources of error

New approach (this script):
  Load verified Q+A from trusted datasets (GSM8K, MATH)
  LLM writes step-by-step Python code to solve each question
  Execute the code  →  if output matches trusted answer → save as training example
  No hallucinated answers. One LLM call per question.

Datasets:
  GSM8K  (openai/gsm8k)      — 8,500 grade-school word problems, always numeric
  MATH   (hendrycks/math)    — 12,500 competition problems across 7 subjects
                               algebra, geometry, precalculus (trig), probability, etc.

Output: data/train_partB.jsonl  (consumed by sft_dataset_assembler.py, same format as before)

Usage:
    python sft_math_pipeline.py                              # 500 GSM8K + 500 MATH
    python sft_math_pipeline.py --gsm8k_count 300 --math_count 700
    python sft_math_pipeline.py --math_max_level 2          # easier questions only
    python sft_math_pipeline.py --smoke                     # 5 questions, quick test
    python sft_math_pipeline.py --resume                    # skip already-completed questions
"""

import argparse
import ast
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output on Windows (cp1252 console can't render arrows, ticks, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MAX_RETRIES: int = 5
_BASE_DELAY: float = 15.0  # NIM rate limits are per-minute; 3s was too short

# Global rate-limit coordinator — prevents thundering herd across workers
_rate_limit_lock  = threading.Lock()
_rate_limit_until = 0.0          # monotonic seconds; all threads wait past this


def _wait_for_rate_limit() -> None:
    while True:
        with _rate_limit_lock:
            remaining = _rate_limit_until - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 1.0))


def _set_global_backoff(seconds: float) -> None:
    global _rate_limit_until
    with _rate_limit_lock:
        candidate = time.monotonic() + seconds
        if candidate > _rate_limit_until:
            _rate_limit_until = candidate

# ---------------------------------------------------------------------------
# System prompt — appears in every training example (must match 3_inference.py)
# ---------------------------------------------------------------------------

TRAINING_SYSTEM = (
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

# ---------------------------------------------------------------------------
# Generation prompt — sent to the LLM to produce each training response
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """Solve this math problem as a teaching example. Show your complete reasoning.

Question: {question}

Write your response in EXACTLY this format:

<think>
CAPABILITY_CHECK: python_execute is available. This is a {question_type} problem.
1. Understanding: [one sentence — what is the question asking for]
2. Approach: [which formula or method applies, and why]
3. Steps:
   - [step 1 — what needs to be computed first]
   - [step 2 — next computation]
   - [continue as needed — be specific, not vague]
</think>
<tool>python_execute(code=\"\"\"
# [One-line description of what this block computes]
import math  # only if needed

# Step 1: [describe what this line computes]
variable_name = ...  # use descriptive variable names throughout

# Step 2:
...

# Final result — must be the only print statement
print(round(result, 2))  # use 4dp for probability/fractions, no rounding for integers
\"\"\")</tool>
<answer>[number only — must exactly match what the code prints, nothing else]</answer>

Rules (strictly enforced — violating any will cause this example to be discarded):
- Only allowed imports: math, statistics, decimal, fractions, cmath — nothing else
- Code must print exactly ONE number on its last line
- <answer> must contain only that number — no units, no text, no explanation
- <think> must start with CAPABILITY_CHECK and show genuine step-by-step reasoning
- Variable names must be descriptive (not x, y, a, b)"""

# ---------------------------------------------------------------------------
# MATH dataset subject → question_type mapping
# ---------------------------------------------------------------------------

MATH_SUBJECT_TO_TYPE = {
    "algebra":                  "algebra",
    "intermediate_algebra":     "algebra",
    "prealgebra":               "arithmetic",
    "geometry":                 "geometry",
    "precalculus":              "trigonometry",
    "counting_and_probability": "statistics",
    "number_theory":            "arithmetic",
}

# ---------------------------------------------------------------------------
# Code safety validator (AST-based — same as rest of pipeline)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})
_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


def _validate_code(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
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
# Code extraction and execution
# ---------------------------------------------------------------------------

def extract_code_blocks(response: str) -> list[str]:
    """Extract Python code from <tool>python_execute(code=\"\"\"...\"\"\")"""
    blocks = []
    # Triple-quote variant (preferred — handles multi-line cleanly)
    for m in re.finditer(
        r'<tool>\s*python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)\s*</tool>',
        response, re.DOTALL
    ):
        blocks.append(m.group(1).strip())
    # Single/double-quote variant (fallback)
    if not blocks:
        for m in re.finditer(
            r'<tool>\s*python_execute\s*\(\s*code\s*=\s*["\']+(.*?)["\']+\s*\)\s*</tool>',
            response, re.DOTALL
        ):
            code = m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
            blocks.append(code)
    # Markdown fallback (if LLM ignores the tool format)
    if not blocks:
        for m in re.finditer(r'```python\n(.*?)```', response, re.DOTALL):
            blocks.append(m.group(1).strip())
    return blocks


def execute_code(code_blocks: list[str]) -> tuple[bool, str]:
    """Execute code blocks sequentially. Returns (success, last_stdout)."""
    if not code_blocks:
        return False, "no_code"
    last_output = ""
    for code in code_blocks:
        safe, reason = _validate_code(code)
        if not safe:
            return False, f"unsafe: {reason}"
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return False, f"runtime_error: {result.stderr[:300]}"
            last_output = result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)
    return True, last_output


def extract_number(output: str) -> str | None:
    """Pull the last numeric value from execution output."""
    for line in reversed(output.strip().split("\n")):
        m = re.search(r'[-+]?\d*\.?\d+', line.strip())
        if m:
            return m.group()
    return None


def answers_match(computed: str, expected: str, tolerance: float = 0.01) -> bool:
    try:
        c, e = float(computed), float(expected)
        if abs(e) < 1e-9:
            return abs(c) < 1e-6
        return abs(c - e) / abs(e) < tolerance
    except (ValueError, TypeError):
        return str(computed).strip() == str(expected).strip()


# ---------------------------------------------------------------------------
# Answer extraction from dataset formats
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(answer_text: str) -> str | None:
    """GSM8K answers end with '#### NUMBER'."""
    m = re.search(r'####\s*([\d,\.\-]+)', answer_text)
    if m:
        return m.group(1).replace(',', '').strip()
    return None


def extract_math_answer(solution: str) -> str | None:
    """MATH dataset answers are inside \\boxed{} in the solution LaTeX."""
    m = re.search(r'\\boxed\{([^}]+)\}', solution)
    if not m:
        return None
    raw = m.group(1).strip()

    # Simple fraction: \frac{a}{b}
    frac_m = re.match(r'\\frac\{(\-?\d+)\}\{(\-?\d+)\}', raw)
    if frac_m:
        num, den = int(frac_m.group(1)), int(frac_m.group(2))
        return str(round(num / den, 6)) if den != 0 else None

    # Strip commas and spaces, try float conversion
    try:
        return str(float(raw.replace(',', '').replace(' ', '')))
    except ValueError:
        return None  # Skip non-numeric answers (expressions, variables, etc.)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(count: int, seed: int = 42) -> list[dict]:
    """Sample `count` questions from GSM8K training split."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Loading GSM8K (target: {count} questions)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)

    items = []
    for i in indices:
        if len(items) >= count:
            break
        row = ds[i]
        answer = extract_gsm8k_answer(row["answer"])
        if answer is None:
            continue
        items.append({
            "question":        row["question"].strip(),
            "expected_answer": answer,
            "source":          "gsm8k",
            "question_type":   "word_problems",
            "level":           1,
        })

    print(f"  → {len(items)} questions loaded from GSM8K")
    return items


def load_math_dataset(count: int, seed: int = 42, max_level: int = 3) -> list[dict]:
    """
    Sample `count` questions from the MATH dataset across all 7 subjects,
    filtered to difficulty levels 1–max_level (scale 1–5).
    Skips questions with non-numeric answers (e.g. expressions, variables).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    subjects = list(MATH_SUBJECT_TO_TYPE.keys())
    per_subject = max(1, count // len(subjects))
    items = []
    random.seed(seed)

    for subject in subjects:
        if len(items) >= count:
            break
        print(f"  Loading MATH/{subject} (level ≤ {max_level})...")
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        except Exception as e:
            print(f"    Warning: could not load {subject}: {e}")
            continue

        def level_num(row):
            try:
                return int(row.get("level", "Level 5").split()[-1])
            except (ValueError, IndexError):
                return 5

        rows = [row for row in ds if level_num(row) <= max_level]
        random.shuffle(rows)

        added = 0
        for row in rows:
            if added >= per_subject:
                break
            answer = extract_math_answer(row["solution"])
            if answer is None:
                continue
            items.append({
                "question":        row["problem"].strip(),
                "expected_answer": answer,
                "source":          "math",
                "subject":         subject,
                "question_type":   MATH_SUBJECT_TO_TYPE.get(subject, "algebra"),
                "level":           level_num(row),
            })
            added += 1

        print(f"    → {added} questions from {subject}")

    print(f"  → {len(items)} questions loaded from MATH total")
    return items


# ---------------------------------------------------------------------------
# Core: generate → execute → verify → return training example
# ---------------------------------------------------------------------------

def generate_training_example(
    item: dict,
    model: str,
    api_base: str | None,
    max_retries: int,
) -> dict | None:
    """
    For a single question + trusted answer:
      1. Ask LLM to write a step-by-step Python solution
      2. Execute the code
      3. If output matches trusted answer → return the training example dict
      4. If not → retry with higher temperature → return None after max_retries
    """
    question = item["question"]
    expected = item["expected_answer"]
    q_type   = item["question_type"]

    q_short = question[:120].replace("\n", " ")
    print(f"\n  >> [{item['source']}·{q_type}] {q_short}", flush=True)

    prompt = GENERATION_PROMPT.format(question=question, question_type=q_type)
    kwargs = dict(
        model=model,
        max_tokens=2048 * 4,
        temperature=0.3,
        timeout=600,
        messages=[{"role": "user", "content": prompt}],
    )
    if api_base:
        kwargs["api_base"] = api_base

    for attempt in range(max_retries):
        _wait_for_rate_limit()
        print(f"    attempt {attempt + 1}: calling API [{item['source']}]...", flush=True)
        try:
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            if not content:
                print(f"    attempt {attempt + 1}: null response", flush=True)
                continue

            code_blocks = extract_code_blocks(content)
            if not code_blocks:
                print(f"    attempt {attempt + 1}: no code block found in response", flush=True)
                kwargs["temperature"] = min(kwargs["temperature"] + 0.2, 1.0)
                continue

            success, output = execute_code(code_blocks)
            if not success:
                print(f"    attempt {attempt + 1}: execution failed — {output[:100]}")
                kwargs["temperature"] = min(kwargs["temperature"] + 0.2, 1.0)
                continue

            computed = extract_number(output)
            if computed is None:
                print(f"    attempt {attempt + 1}: no number in output — {repr(output[:80])}")
                kwargs["temperature"] = min(kwargs["temperature"] + 0.2, 1.0)
                continue

            if answers_match(computed, expected):
                print(f"    attempt {attempt + 1}: ✓  computed={computed}  expected={expected}")
                # Build multi-turn v3 format: think → tool call → tool result → answer
                think_m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_m:
                    think_text = think_m.group(1).strip()
                else:
                    # LLM skipped <think> tags — use text before <tool> tag, or reconstruct from comments
                    tool_pos = content.find('<tool>')
                    think_text = content[:tool_pos].strip() if tool_pos != -1 else ""
                    if not think_text:
                        comment_lines = [
                            l.lstrip("# ").strip()
                            for l in code_blocks[0].splitlines()
                            if l.strip().startswith("#") and l.strip() != "#"
                        ]
                        narration = "\n".join(f"- {c}" for c in comment_lines[:10])
                        think_text = f"Let me solve this step-by-step.\n\n{narration}" if narration else "Let me solve this step-by-step."

                code = code_blocks[0]
                return {
                    "messages": [
                        {"role": "system", "content": TRAINING_SYSTEM},
                        {"role": "user",   "content": question},
                        {
                            "role": "assistant",
                            "content": f'<think>\n{think_text}\n</think>\n<tool>python_execute(code="""{code}""")</tool>',
                        },
                        {
                            "role": "tool",
                            "name": "python_execute",
                            "content": f"[TOOL_RESULT: python_execute]\n{output}\n[/TOOL_RESULT]",
                        },
                        {
                            "role": "assistant",
                            "content": f"<answer>{computed}</answer>",
                        },
                    ],
                    "metadata": {
                        "source":          item["source"],
                        "question_id":     uuid.uuid4().hex[:12],
                        "category":        CATEGORY_MAP.get(q_type, q_type),
                        "tool_profile":    "compute_only",
                        "constitution_score": 1.0,
                        "revised":         False,
                        "pipeline":        "part_b_datasets",
                        "original_level":  item.get("level", 1),
                        "subject":         item.get("subject", ""),
                        "expected_answer": expected,
                    },
                }
            else:
                print(f"    attempt {attempt + 1}: ✗  computed={computed}  expected={expected}")
                kwargs["temperature"] = min(kwargs["temperature"] + 0.2, 1.0)

        except litellm.RateLimitError:
            wait = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 5)
            print(f"    attempt {attempt + 1}: rate limit — global backoff {wait:.0f}s (all workers paused)")
            _set_global_backoff(wait)
            _wait_for_rate_limit()
        except (litellm.APIConnectionError, litellm.Timeout):
            wait = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 3)
            print(f"    attempt {attempt + 1}: connection error — waiting {wait:.0f}s")
            _set_global_backoff(wait)
            _wait_for_rate_limit()
        except Exception as e:
            print(f"    attempt {attempt + 1}: unexpected error — {e}")
            time.sleep(_BASE_DELAY)

    return None


# ---------------------------------------------------------------------------
# Processing loop (parallel workers, thread-safe file writes)
# ---------------------------------------------------------------------------

def process_items(
    items: list[dict],
    output_path: str,
    model: str,
    api_base: str | None,
    max_retries: int,
    workers: int,
    resume: bool,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-completed questions so we skip them
    done_questions: set[str] = set()
    if resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    done_questions.add(ex["messages"][1]["content"])
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        print(f"Resume: {len(done_questions)} already completed — skipping.")

    to_run = [it for it in items if it["question"] not in done_questions]
    total  = len(to_run)
    print(f"Processing {total} questions with {workers} workers...\n")

    file_lock = threading.Lock()
    accepted  = 0
    rejected  = 0
    done_count = 0
    start_time = time.monotonic()

    write_mode = "a" if resume else "w"

    with open(output_path, write_mode, encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(generate_training_example, item, model, api_base, max_retries): item
                for item in to_run
            }
            for future in concurrent.futures.as_completed(future_to_item):
                item       = future_to_item[future]
                done_count += 1
                q_short    = item["question"][:70].replace("\n", " ")
                elapsed    = time.monotonic() - start_time

                print(f"\n[{done_count}/{total}] [{item['source']}·{item['question_type']}] {q_short}...")

                try:
                    result = future.result()
                except Exception as e:
                    print(f"  Thread error: {e}")
                    result = None

                if result:
                    with file_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                    accepted += 1
                    rate = accepted / max(elapsed, 1) * 60
                    print(f"  ✓ ACCEPTED  (total: {accepted}  rate: {rate:.1f}/min)")
                else:
                    rejected += 1
                    print(f"  ✗ REJECTED  (exhausted {max_retries} retries)")

    total_time = time.monotonic() - start_time
    print(f"\n{'='*58}")
    print(f"Completed in  : {total_time/60:.1f} min")
    print(f"Processed     : {done_count}")
    print(f"Accepted      : {accepted}  ({100*accepted/max(done_count,1):.1f}%)")
    print(f"Rejected      : {rejected}")
    print(f"Output        : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-process math training data generator from GSM8K + MATH datasets"
    )
    parser.add_argument("--gsm8k_count", type=int, default=500,
                        help="Questions to sample from GSM8K (default: 500)")
    parser.add_argument("--math_count", type=int, default=500,
                        help="Questions to sample from MATH dataset (default: 500)")
    parser.add_argument("--math_max_level", type=int, default=3,
                        help="Max difficulty level for MATH dataset 1–5 (default: 3 = medium)")
    parser.add_argument("--output", type=str, default="data/train_partB_v3.jsonl",
                        help="Output JSONL (default: data/train_partB_v3.jsonl)")
    parser.add_argument("--model", type=str, default="nvidia_nim/minimaxai/minimax-m2.7",
                        help="litellm model string for generating training responses")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. Ollama or local vLLM endpoint)")
    parser.add_argument("--retries", type=int, default=5,
                        help="Retries per question when code fails to verify (default: 5)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers (default: 3; NIM has tight per-minute limits — increase only if you have a paid tier)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible dataset sampling (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing output file, skipping already-completed questions")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test: 3 GSM8K + 2 MATH questions, 1 worker")
    args = parser.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.retries

    if args.smoke:
        args.gsm8k_count   = 3
        args.math_count    = 2
        args.math_max_level = 1
        args.workers       = 1
        args.output        = "data/smoke_train_partB.jsonl"
        print("Smoke test: 3 GSM8K + 2 MATH questions → data/smoke_train_partB.jsonl\n")

    # Load from both datasets
    items  = load_gsm8k(args.gsm8k_count, seed=args.seed)
    items += load_math_dataset(args.math_count, seed=args.seed, max_level=args.math_max_level)

    random.seed(args.seed)
    random.shuffle(items)
    print(f"\nTotal: {len(items)} questions  "
          f"(GSM8K: {sum(1 for i in items if i['source']=='gsm8k')}  "
          f"MATH: {sum(1 for i in items if i['source']=='math')})\n")

    process_items(
        items      = items,
        output_path= args.output,
        model      = args.model,
        api_base   = args.api_base,
        max_retries= args.retries,
        workers    = args.workers,
        resume     = args.resume,
    )

    print(f"\nNext step → assemble final SFT dataset:")
    print(f"  python sft_dataset_assembler.py --part_b {args.output}")


if __name__ == "__main__":
    main()
