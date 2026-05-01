"""
SFT Rejection Sampler (Part B)
================================
For each math/code question, generates N candidate responses from the base model,
executes code blocks in each candidate, and keeps only those with verified-correct answers.

Scoring:
  +1  Code runs AND answer matches expected
   0  No code used (approximated mentally)
  -1  Code runs but wrong answer, OR code crashes

"No-tool" control cases: correct answer is to say "I can't compute precisely without code execution".
These are scored separately: +1 if response acknowledges lack of tool, -1 if it guesses.

Usage:
    python sft_rejection_sampler.py --questions data/questions_partB.jsonl \\
                                     --output data/train_partB.jsonl \\
                                     --model_path outputs/checkpoint_sft \\
                                     --candidates 8

    # Without a local model (uses Claude as stand-in for base model):
    python sft_rejection_sampler.py --questions data/questions_partB.jsonl \\
                                     --output data/train_partB.jsonl \\
                                     --use_api_model \\
                                     --candidates 8
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt for generation (same as final training, but minimal tools)
# ---------------------------------------------------------------------------

TOOL_SESSION_SYSTEM = """You are a trustworthy AI assistant.
Available tools in this session: python_execute

Before answering, complete a CAPABILITY_CHECK in your <think> block.
Use <tool>python_execute(code="...")</tool> for any calculation requiring precision.
Give your final answer in <answer>...</answer>.
Format numbers to 2 decimal places for money/geometry, 4 for probability."""

NO_TOOL_SESSION_SYSTEM = """You are a trustworthy AI assistant.
Available tools in this session: NONE

Before answering, complete a CAPABILITY_CHECK in your <think> block.
You do NOT have code execution available in this session.
You must be honest about this limitation.
Give your final answer in <answer>...</answer>."""

# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------


def extract_code_blocks(response: str) -> list[str]:
    """Extract all Python code from <tool>python_execute(code="...")</tool> tags."""
    blocks = []

    # Pattern 1: <tool>python_execute(code="...")</tool>
    pattern1 = r'<tool>\s*python_execute\s*\(\s*code\s*=\s*["\']+(.*?)["\']+\s*\)\s*</tool>'
    for match in re.finditer(pattern1, response, re.DOTALL):
        code = match.group(1)
        # Unescape common escape sequences
        code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        blocks.append(code)

    # Pattern 2: triple-quoted code in tool tag
    pattern2 = r'<tool>\s*python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)\s*</tool>'
    for match in re.finditer(pattern2, response, re.DOTALL):
        blocks.append(match.group(1).strip())

    return blocks


def execute_code_blocks(code_blocks: list[str]) -> tuple[bool, str]:
    """Execute code blocks sequentially. Return (success, last_output)."""
    if not code_blocks:
        return False, "no_code"

    last_output = ""
    for code in code_blocks:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return False, f"runtime_error: {result.stderr[:300]}"
            last_output = result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    return True, last_output


def extract_final_number(output: str) -> str | None:
    """Extract the last numeric value from execution output."""
    lines = output.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        # Try to extract a number
        match = re.search(r'[-+]?\d*\.?\d+', line)
        if match:
            return match.group()
    return None


def answers_match(computed: str, expected: str, tolerance: float = 0.01) -> bool:
    """Compare computed and expected answers with numeric tolerance."""
    try:
        c = float(computed)
        e = float(expected)
        if abs(e) < 1e-9:
            return abs(c) < 1e-6
        return abs(c - e) / abs(e) < tolerance
    except (ValueError, TypeError):
        return str(computed).strip() == str(expected).strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidate(response: str, expected_answer: str, is_no_tool: bool) -> tuple[int, dict]:
    """Score a candidate response. Returns (score, details)."""

    if is_no_tool:
        # For no-tool questions: model should acknowledge it can't compute precisely
        response_lower = response.lower()
        honest_signals = [
            "don't have", "don't have access", "cannot compute", "can't compute",
            "no code execution", "without a calculator", "i cannot",
            "no tools", "not available", "can't calculate precisely",
            "would need", "cannot perform", "unable to"
        ]
        is_honest = any(signal in response_lower for signal in honest_signals)
        # Also check it doesn't just give a number confidently
        gave_number = bool(re.search(r'\d+\.\d{2,}', response))

        if is_honest and not gave_number:
            return 1, {"reason": "correctly_refused_no_tool", "honest": True}
        elif gave_number and not is_honest:
            return -1, {"reason": "hallucinated_answer_without_tool", "honest": False}
        else:
            return 0, {"reason": "partial_acknowledgment", "honest": True}

    # For normal questions: check code execution
    code_blocks = extract_code_blocks(response)

    if not code_blocks:
        # No code used — penalise mental approximation
        # Check if the answer tag has a number that matches
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1)
            number_in_answer = re.search(r'[-+]?\d*\.?\d+', answer_text)
            if number_in_answer:
                if answers_match(number_in_answer.group(), expected_answer):
                    # Got right answer mentally — score 0 (technically correct but wrong method)
                    return 0, {"reason": "correct_but_no_code", "computed": number_in_answer.group()}
        return 0, {"reason": "no_code_used", "computed": None}

    # Execute the code
    success, output = execute_code_blocks(code_blocks)
    if not success:
        return -1, {"reason": "code_execution_failed", "error": output}

    # Extract numeric answer from output
    computed = extract_final_number(output)
    if computed is None:
        return -1, {"reason": "no_numeric_output", "raw_output": output[:200]}

    if answers_match(computed, expected_answer):
        return 1, {"reason": "correct", "computed": computed, "expected": expected_answer}
    else:
        return -1, {
            "reason": "wrong_answer",
            "computed": computed,
            "expected": expected_answer,
            "output": output[:200],
        }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates_via_api(
    question: str,
    is_no_tool: bool,
    n_candidates: int,
    model: str,
    api_base: str | None = None,
) -> list[str]:
    """Generate N diverse candidates via litellm (any provider)."""
    system = NO_TOOL_SESSION_SYSTEM if is_no_tool else TOOL_SESSION_SYSTEM
    user_question = question.replace("[NO_TOOLS_SESSION] ", "")

    candidates = []
    for _ in range(n_candidates):
        try:
            kwargs = dict(
                model=model, max_tokens=1024,
                temperature=0.8,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_question},
                ],
            )
            if api_base:
                kwargs["api_base"] = api_base
            response = litellm.completion(**kwargs)
            candidates.append(response.choices[0].message.content.strip())
            time.sleep(0.2)
        except Exception as e:
            if "rate" in str(e).lower():
                time.sleep(30)
            else:
                print(f"  Generation error: {e}")

    return candidates


def generate_candidates_via_local_model(
    model_path: str,
    question: str,
    is_no_tool: bool,
    n_candidates: int,
) -> list[str]:
    """Generate candidates using a local HuggingFace model checkpoint."""
    # This requires transformers + unsloth to be installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("transformers and torch required for local model inference. "
                          "Use --use_api_model to use Claude API instead.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    system = NO_TOOL_SESSION_SYSTEM if is_no_tool else TOOL_SESSION_SYSTEM
    user_question = question.replace("[NO_TOOLS_SESSION] ", "")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    candidates = []
    for _ in range(n_candidates):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(decoded.strip())

    return candidates


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_questions(
    questions_path: str,
    output_path: str,
    model_path: str | None,
    use_api_model: bool,
    api_model: str,
    n_candidates: int,
    min_score: int,
    max_examples: int | None,
    resume: bool,
    api_base: str | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    done_questions: set[str] = set()
    if resume and Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    q = ex["messages"][1]["content"]
                    done_questions.add(q)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        print(f"Resuming: {len(done_questions)} already done.")

    processed = 0
    accepted = 0
    rejected_all = 0

    write_mode = "a" if resume else "w"
    with open(questions_path, encoding="utf-8") as qf, \
         open(output_path, write_mode, encoding="utf-8") as out:

        for line in qf:
            if max_examples and processed >= max_examples:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = item.get("question", "").strip()
            expected_answer = item.get("answer", "").strip()
            qtype = item.get("type", "unknown")
            is_no_tool = "[NO_TOOLS_SESSION]" in question or qtype == "no_tool_control"

            if not question or question in done_questions:
                continue

            display_q = question.replace("[NO_TOOLS_SESSION] ", "")
            print(f"\n[{processed + 1}] [{qtype}] {display_q[:80]}...")
            print(f"  Expected: {expected_answer}")

            # Generate candidates
            if use_api_model or model_path is None:
                candidates = generate_candidates_via_api(
                    question, is_no_tool, n_candidates, api_model, api_base
                )
            else:
                candidates = generate_candidates_via_local_model(
                    model_path, question, is_no_tool, n_candidates
                )

            print(f"  Generated {len(candidates)} candidates")

            # Score candidates
            best_score = -2
            best_candidate = None
            best_details = None

            for i, candidate in enumerate(candidates):
                score, details = score_candidate(candidate, expected_answer, is_no_tool)
                print(f"  Candidate {i+1}: score={score} ({details.get('reason', '?')})")
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_details = details

            processed += 1

            if best_score >= min_score and best_candidate:
                # Build training example
                user_question = question.replace("[NO_TOOLS_SESSION] ", "")
                system = NO_TOOL_SESSION_SYSTEM if is_no_tool else TOOL_SESSION_SYSTEM

                example = {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": best_candidate},
                    ],
                    "metadata": {
                        "source": "rejection_sampling",
                        "question_type": qtype,
                        "expected_answer": expected_answer,
                        "best_score": best_score,
                        "score_details": best_details,
                        "candidates_generated": len(candidates),
                        "pipeline": "part_b",
                    },
                }
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                out.flush()
                accepted += 1
                done_questions.add(question)
                print(f"  ACCEPTED (score={best_score})")
            else:
                rejected_all += 1
                print(f"  REJECTED (best score={best_score} < threshold={min_score})")

    print(f"\n{'='*50}")
    print(f"Processed:    {processed}")
    print(f"Accepted:     {accepted} ({100*accepted/max(processed,1):.1f}%)")
    print(f"Rejected all: {rejected_all}")
    print(f"Output:       {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Part B rejection sampler for math/code questions")
    parser.add_argument("--questions", type=str, required=True,
                        help="Input JSONL from sft_math_question_generator.py")
    parser.add_argument("--output", type=str, default="pipeline/data/train_partB.jsonl")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local HuggingFace model checkpoint")
    parser.add_argument("--use_api_model", action="store_true",
                        help="Use litellm model as candidate generator (when no local model)")
    parser.add_argument("--api_model", type=str, default="claude-haiku-4-5-20251001",
                        help="litellm model string for candidate generation (any provider)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--candidates", type=int, default=8,
                        help="Candidate responses to generate per question")
    parser.add_argument("--min_score", type=int, default=1,
                        help="Minimum score to accept a candidate (1=correct, 0=no-code)")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # litellm reads API keys from env vars automatically
    if args.model_path is None:
        args.use_api_model = True

    process_questions(
        questions_path=args.questions,
        output_path=args.output,
        model_path=args.model_path,
        use_api_model=args.use_api_model,
        api_model=args.api_model,
        n_candidates=args.candidates,
        min_score=args.min_score,
        max_examples=args.max,
        resume=args.resume,
        api_base=args.api_base,
    )


if __name__ == "__main__":
    main()
