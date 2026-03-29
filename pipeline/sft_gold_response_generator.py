"""
SFT Gold Response Generator (Part A)
======================================
Takes questions from sft_question_generator.py and produces gold training examples
using a teacher model via litellm (vendor-agnostic: Anthropic, OpenAI, Ollama, Groq…).

Each example goes through:
  1. Draft generation  — teacher generates an initial response
  2. Self-critique     — teacher checks the draft against constitution principles
  3. Revision          — if violations found, teacher rewrites the response
  4. Final formatting  — wraps into the training JSONL format with CAPABILITY_CHECK

Model string examples:
    Anthropic : claude-sonnet-4-5
    OpenAI    : gpt-4o-mini
    Ollama    : ollama/llama3.2  (set OLLAMA_API_BASE or pass --api_base)
    Groq      : groq/llama-3.1-70b-versatile

Usage:
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --output data/train_partA.jsonl
    python sft_gold_response_generator.py --questions data/sample.jsonl \\
                                           --output data/sample_gold.jsonl --max 5
    python sft_gold_response_generator.py --questions data/sample.jsonl \\
                                           --model ollama/llama3.2 --api_base http://localhost:11434 \\
                                           --output data/sample_gold.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# System prompt used during training (what the final model sees)
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT = """You are a trustworthy AI assistant. Before answering any question,
you must complete a CAPABILITY_CHECK inside your <think> block that explicitly identifies:
1. What this question requires to answer correctly
2. What tools/access you currently have in this session
3. Whether there is a gap between (1) and (2)
4. If a gap exists, what the honest response to that gap is

Available tools (check which are present in your session):
- python_execute(code: str)  — run Python code, returns stdout
- web_search(query: str)     — search the web, returns result snippets
- read_url(url: str)         — fetch content from a specific URL
- get_datetime()             — returns current date and time

Tool selection guide:
- Arithmetic / calculation   → python_execute
- Facts about the world that may have changed, proper nouns, entity details → web_search
- Follow up on a search result URL → read_url
- Time-aware responses → get_datetime
- Stable definitions, logic, concepts → no tool needed

You follow these principles in every response:
- Decompose questions into their requirements before answering
- Only call tools actually available in this session — never invent tools
- Use web_search for real-time data AND for proper nouns / entity facts that may be stale
- Use python_execute for any precision arithmetic — never approximate mentally
- If you need user-specific context you don't have, ask one question at a time
- Quantify uncertainty honestly — only hedge what you're genuinely uncertain about
- Acknowledge when tasks are impossible and explain why
- Present tradeoffs for subjective questions — never declare a universal winner
- Hold your position when users pressure you to guess
- Correct your own errors explicitly when you catch them
- Say "I don't know" clearly when you have no basis for an answer

Format:
- Use <think>...</think> for your reasoning (always start with CAPABILITY_CHECK)
- Use <answer>...</answer> for your final response to the user
- For tool calls: <tool>tool_name(param=value)</tool>"""

# ---------------------------------------------------------------------------
# Prompts for draft → critique → revision pipeline
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """You are demonstrating ideal AI assistant behaviour for training purposes.

The user asks: {question}

{follow_up_context}

Write a response that exemplifies trustworthy AI behaviour. Your response MUST:
1. Start your <think> block with a CAPABILITY_CHECK section
2. In CAPABILITY_CHECK: list what the question requires, what session tools are available
   (assume: python_execute is available, no internet/real-time tools unless stated),
   identify any gaps, and state your conclusion
3. Follow the CAPABILITY_CHECK with any needed reasoning (PLAN, EXECUTE, VERIFY)
4. Give the final answer in <answer>...</answer>

For this question type ({category}), the ideal behaviour is:
{ideal_behavior}

Write the complete response now:"""

CRITIQUE_PROMPT = """You are reviewing an AI response against a trustworthiness constitution.

QUESTION: {question}
CATEGORY: {category}

RESPONSE TO REVIEW:
{response}

Check this response against each relevant constitution principle:
1. DECOMPOSE FIRST — Did it identify what the question requires?
2. TOOL INVENTORY — Did it state what session tools are available?
3. TOOL DISCIPLINE — Did it only use real tools?
4. MATH = CODE — If math was needed, was code used?
5. REAL-TIME HONESTY — If live data was needed, did it say so?
6. USER CONTEXT GATE — If user context was needed, did it ask?
7. UNCERTAINTY QUANTIFICATION — Did it hedge appropriately (not too much, not too little)?
8. IMPOSSIBILITY ACKNOWLEDGMENT — If impossible, did it say why + redirect?
9. TRADEOFF PRESENTATION — If subjective, did it enumerate dimensions?
10. HOLD UNDER PRESSURE — (if applicable) Did it maintain position?
11. EXPLICIT I DON'T KNOW — If no basis for answer, did it say so?

List ONLY the violations found. If none, respond with: NO_VIOLATIONS
Format violations as: PRINCIPLE_N: [brief description of what's wrong]

Do not rewrite the response yet. Only list violations."""

REVISION_PROMPT = """You are revising an AI response to fix constitution violations.

QUESTION: {question}
CATEGORY: {category}

ORIGINAL RESPONSE:
{response}

VIOLATIONS TO FIX:
{violations}

Rewrite the response to fix all violations while keeping what was already correct.
The revised response must:
- Start <think> with CAPABILITY_CHECK
- Fix each listed violation
- Keep the same format: <think>...</think><answer>...</answer>

Write the complete revised response:"""

# ---------------------------------------------------------------------------
# Per-category ideal behaviour descriptions
# ---------------------------------------------------------------------------

IDEAL_BEHAVIORS = {
    "user_context_behavioral": (
        "Ask for the single most critical missing context before answering. "
        "CAPABILITY_CHECK should show user-specific info as a gap. "
        "Answer should ask ONE clarifying question, not several."
    ),
    "real_time_dependent": (
        "Clearly state that you cannot access live/current data. "
        "CAPABILITY_CHECK should show real-time tool as unavailable. "
        "Offer alternative help: where to find the data, or what you CAN calculate once they have it."
    ),
    "impossible_tasks": (
        "Explain specifically WHY this is impossible (not just 'I can't do that'). "
        "Redirect to something useful you CAN do. "
        "CAPABILITY_CHECK should identify the fundamental impossibility."
    ),
    "subjective_tradeoffs": (
        "Enumerate tradeoffs across 3-5 relevant dimensions. "
        "End with a question about their specific constraints/priorities. "
        "Never declare a universal winner."
    ),
    "adversarial_pressure": (
        "Acknowledge the user's request, then explain specifically why guessing would be harmful. "
        "Maintain refusal but offer a concrete alternative action. "
        "CAPABILITY_CHECK for turn 2 should show the same gap still exists."
    ),
    "knowledge_boundary": (
        "State your training cutoff when relevant. "
        "Distinguish what you know confidently from what may have changed. "
        "If genuinely uncertain, say so and suggest how to verify."
    ),
    "multi_step_clarification": (
        "Identify the multiple unknowns in CAPABILITY_CHECK. "
        "Ask ONE question — the most critical one. "
        "Explain briefly why that's the most important thing to know first."
    ),
    "ambiguous_underspecified": (
        "Identify the ambiguity explicitly. "
        "Ask the single most important clarifying question. "
        "Give a brief indication of the range of ways you could help, so user knows what to expect."
    ),
}

# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def _call(messages: list, model: str, max_tokens: int, api_base: str | None = None) -> str:
    """Thin litellm wrapper — vendor-agnostic completion call."""
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()


def generate_draft(question: str, category: str, follow_up: str | None,
                   model: str, api_base: str | None = None) -> str:
    follow_up_context = ""
    if follow_up:
        follow_up_context = (
            f"\nNOTE: This is a two-turn scenario. After your initial response, "
            f"the user will push back with: '{follow_up}'\n"
            f"Your response should handle BOTH turns — the initial refusal AND maintaining "
            f"it when challenged. Format as:\n"
            f"<turn_1>[initial response]</turn_1>\n"
            f"<turn_2>[response to follow-up pressure]</turn_2>"
        )

    prompt = DRAFT_PROMPT.format(
        question=question,
        follow_up_context=follow_up_context,
        category=category,
        ideal_behavior=IDEAL_BEHAVIORS.get(category, "Follow constitution principles."),
    )

    return _call(
        messages=[
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model, max_tokens=2048, api_base=api_base,
    )


def critique_draft(question: str, category: str, draft: str,
                   model: str, api_base: str | None = None) -> str:
    prompt = CRITIQUE_PROMPT.format(question=question, category=category, response=draft)
    return _call(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=512, api_base=api_base,
    )


def revise_draft(question: str, category: str, draft: str, violations: str,
                 model: str, api_base: str | None = None) -> str:
    prompt = REVISION_PROMPT.format(
        question=question, category=category, response=draft, violations=violations,
    )
    return _call(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=2048, api_base=api_base,
    )


def build_training_example(question: str, category: str, final_response: str,
                            follow_up: str | None, draft: str,
                            violations: str) -> dict:
    """Convert a question + response into training JSONL format."""
    messages = [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_response},
    ]

    # Two-turn adversarial examples need special handling
    if follow_up and "<turn_1>" in final_response:
        turn_1 = _extract_tag(final_response, "turn_1")
        turn_2 = _extract_tag(final_response, "turn_2")
        messages = [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": turn_1},
            {"role": "user", "content": follow_up},
            {"role": "assistant", "content": turn_2},
        ]

    return {
        "messages": messages,
        "metadata": {
            "source": "constitution_teacher",
            "category": category,
            "constitution_violations_in_draft": 0 if violations == "NO_VIOLATIONS" else len(violations.split("\n")),
            "revised": violations != "NO_VIOLATIONS",
            "pipeline": "part_a",
        },
    }


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>."""
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return text
    return text[start + len(tag) + 2:end].strip()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_questions(
    questions_path: str,
    output_path: str,
    model: str,
    max_examples: int | None,
    resume: bool,
    api_base: str | None = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load already-processed questions if resuming
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
        print(f"Resuming: {len(done_questions)} already processed.")

    processed = 0
    skipped = 0
    errors = 0

    write_mode = "a" if resume else "w"
    with open(questions_path, encoding="utf-8") as qf, \
         open(output_path, write_mode, encoding="utf-8") as out:

        for line in qf:
            if max_examples and processed >= max_examples:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            question = item.get("question", "").strip()
            category = item.get("category", "unknown")
            follow_up = item.get("follow_up")

            if not question:
                errors += 1
                continue

            if question in done_questions:
                skipped += 1
                continue

            print(f"\n[{processed + 1}] [{category}] {question[:80]}...")

            try:
                # Step 1: Generate draft
                draft = generate_draft(question, category, follow_up, model, api_base)
                print(f"  Draft: {len(draft)} chars")

                # Step 2: Critique
                violations = critique_draft(question, category, draft, model, api_base)
                has_violations = violations != "NO_VIOLATIONS"
                print(f"  Critique: {'VIOLATIONS FOUND' if has_violations else 'clean'}")

                # Step 3: Revise if needed
                final = draft
                if has_violations:
                    final = revise_draft(question, category, draft, violations, model, api_base)
                    print(f"  Revised: {len(final)} chars")

                # Step 4: Build training example
                example = build_training_example(question, category, final, follow_up, draft, violations)
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                out.flush()

                processed += 1
                done_questions.add(question)

                # Courtesy pause to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                if "rate" in str(e).lower():
                    print("  Rate limit hit. Waiting 60s...")
                    time.sleep(60)
                    continue  # Retry without incrementing processed
            except Exception as e:
                print(f"  Error: {e}")
                errors += 1
                continue

    print(f"\n{'='*50}")
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part A gold responses via teacher model + self-critique")
    parser.add_argument("--questions", type=str, required=True,
                        help="Input JSONL file from sft_question_generator.py")
    parser.add_argument("--output", type=str, default="pipeline/data/train_partA.jsonl",
                        help="Output training JSONL file")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5",
                        help="litellm model string (e.g. claude-sonnet-4-5, gpt-4o-mini, ollama/llama3.2)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-processed questions)")
    args = parser.parse_args()

    # litellm reads API keys from env vars automatically:
    # ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, etc.

    process_questions(
        questions_path=args.questions,
        output_path=args.output,
        model=args.model,
        max_examples=args.max,
        resume=args.resume,
        api_base=args.api_base,
    )


if __name__ == "__main__":
    main()
