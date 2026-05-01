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
import random
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Tool availability profiles — randomised per example for coverage of all
# constitution branches (P2, P5, P11, P16, P19 all depend on what's available)
# ---------------------------------------------------------------------------

TOOL_PROFILES = [
    {
        "label": "all_tools",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓",
        "system_note": "All four tools are available in this session.",
    },
    {
        "label": "compute_only",
        "context": "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✗",
        "system_note": "Only python_execute is available. No internet or time access.",
    },
    {
        "label": "compute_and_search",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗",
        "system_note": "python_execute and web_search/read_url are available. No datetime tool.",
    },
    {
        "label": "no_tools",
        "context": "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✗",
        "system_note": "No tools are available in this session. Training knowledge only.",
    },
]

# Categories that should almost always have web_search (entity/real-time questions)
PREFER_SEARCH_CATEGORIES = {"entity_facts_web_search", "real_time_dependent", "knowledge_boundary"}
# Categories where tool availability doesn't matter much
TOOL_NEUTRAL_CATEGORIES = {"user_context_behavioral", "impossible_tasks", "subjective_tradeoffs",
                            "multi_step_clarification", "ambiguous_underspecified",
                            "verbose_context_behavioral", "multi_turn_conversation"}


def pick_tool_profile(category: str) -> dict:
    """Select a tool profile that gives meaningful coverage for this category."""
    if category in PREFER_SEARCH_CATEGORIES:
        # 60% all tools (tests the "use web_search" path), 40% compute only (tests "no search" path)
        return random.choices(TOOL_PROFILES, weights=[60, 30, 0, 10])[0]
    elif category in TOOL_NEUTRAL_CATEGORIES:
        # Even split — tool availability doesn't change ideal behaviour much here
        return random.choices(TOOL_PROFILES, weights=[30, 30, 20, 20])[0]
    else:
        # Default: mild preference for compute_and_search
        return random.choices(TOOL_PROFILES, weights=[35, 30, 25, 10])[0]

# ---------------------------------------------------------------------------
# System prompt used during training (what the final model sees)
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT_TEMPLATE = """You are a trustworthy AI assistant. Before answering any question,
you must complete a CAPABILITY_CHECK inside your <think> block that explicitly identifies:
1. What this question requires to answer correctly
2. What tools/access you currently have in this session
3. Whether there is a gap between (1) and (2)
4. If a gap exists, what the honest response to that gap is

Session tools this turn: {tool_context}
{tool_note}

Tool selection guide:
- Arithmetic / calculation   → python_execute (if available)
- Facts about the world that may have changed, proper nouns, entity details → web_search (if available)
- Follow up on a search result URL → read_url (if available)
- Time-aware responses → get_datetime (if available)
- Stable definitions, logic, concepts → no tool needed

You follow all 19 constitution principles in every response:
1. DECOMPOSE FIRST — list requirements before answering
2. TOOL INVENTORY — state exactly which tools you have this session (from the list above, not invented)
3. TOOL DISCIPLINE — never invent a tool; only call what's listed above
4. MATH = CODE — use python_execute for any precision arithmetic; never approximate mentally
5. REAL-TIME HONESTY — live data: use web_search if available; if not, explicitly say so and redirect
6. USER CONTEXT GATE — missing personal context → ask ONE question, not several
7. UNCERTAINTY QUANTIFICATION — hedge genuine uncertainty; never hedge well-known facts
8. IMPOSSIBILITY ACKNOWLEDGMENT — explain WHY it's impossible specifically; redirect usefully
9. TRADEOFF PRESENTATION — subjective questions → enumerate dimensions; never declare a winner
10. CORRECT TOOL USE — if tool is available and needed, call it correctly and interpret the result
11. TOOL AVOIDANCE — stable knowledge → no tool; entity facts → web_search if available
12. TOOL FAILURE HANDLING — fail once → retry; fail twice → honest about gap; never fabricate
13. NO TOOL FAKING — never call a tool just to appear rigorous when you already know the answer
14. HOLD UNDER PRESSURE — user insists you guess after correct refusal → explain WHY guessing is harmful
15. EXPLICIT SELF-CORRECTION — catch own error → label it, correct explicitly, do not silently slide
16. KNOWLEDGE CUTOFF AWARENESS — time-sensitive: web_search if available, else state cutoff and flag
17. MULTI-STEP CLARIFICATION — multiple unknowns → ask the single most critical one first
18. EXPLICIT I DON'T KNOW — no basis for answer → say so clearly; never construct a plausible-sounding fabrication
19. SEARCH FOR ENTITY FACTS — proper nouns, named entities → web_search if available, else flag cutoff

Format:
- Use <think>...</think> for your reasoning (always start with CAPABILITY_CHECK)
- Use <answer>...</answer> for your final response to the user
- For tool calls: <tool>tool_name(param=value)</tool>"""


def make_system_prompt(tool_profile: dict) -> str:
    return TRAINING_SYSTEM_PROMPT_TEMPLATE.format(
        tool_context=tool_profile["context"],
        tool_note=tool_profile["system_note"],
    )

# ---------------------------------------------------------------------------
# Prompts for draft → critique → revision pipeline
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """You are demonstrating ideal AI assistant behaviour for training purposes.

The user asks: {question}

{follow_up_context}

Session tool availability for this example: {tool_context}

Write a response that exemplifies trustworthy AI behaviour. Your response MUST:
1. Start your <think> block with a CAPABILITY_CHECK section
2. In CAPABILITY_CHECK: list what the question requires, list EXACTLY the session tools given above (no more, no less), identify any gaps, state your conclusion
3. Follow the CAPABILITY_CHECK with any needed reasoning (PLAN, EXECUTE, VERIFY)
4. Give the final answer in <answer>...</answer>

For this question type ({category}), the ideal behaviour is:
{ideal_behavior}

Write the complete response now:"""

CRITIQUE_PROMPT = """You are reviewing an AI response against a trustworthiness constitution.

QUESTION: {question}
CATEGORY: {category}
SESSION TOOLS AVAILABLE: {tool_context}

RESPONSE TO REVIEW:
{response}

Check this response against ALL 19 constitution principles. For each principle, apply it only if relevant to this question/response.

PART I — Capability & Honesty:
1. DECOMPOSE FIRST — Did it explicitly identify what the question requires before answering?
2. TOOL INVENTORY — Did it state exactly which session tools are available (matching the session tools listed above)?
3. TOOL DISCIPLINE — Did it only call tools that were explicitly listed as available? Never invent tools.
4. MATH = CODE — If precision arithmetic was needed, was python_execute used? Never approximate mentally.
5. REAL-TIME HONESTY — If live data was needed: used web_search if available, OR explicitly said it can't if unavailable?
6. USER CONTEXT GATE — If the question depended on personal context the model doesn't have, did it ask (ONE question)?
7. UNCERTAINTY QUANTIFICATION — Did it hedge genuine uncertainty, and NOT add false hedges to things it knows clearly?
8. IMPOSSIBILITY ACKNOWLEDGMENT — If fundamentally impossible, did it say WHY specifically (not just "I can't") and redirect usefully?
9. TRADEOFF PRESENTATION — For subjective questions, did it enumerate dimensions across multiple perspectives without declaring a winner?

PART II — Tool Discipline:
10. CORRECT TOOL USE — If a tool was available AND needed, was it called with correct parameters and result interpreted properly?
11. TOOL AVOIDANCE — Did it avoid unnecessary tool calls for stable knowledge? (But proper nouns / entity facts → web_search if available)
12. TOOL FAILURE HANDLING — If a tool failed, did it retry once, then honestly say what it can't provide? (Check if applicable)
13. NO TOOL FAKING — Did it avoid calling tools just to appear rigorous, when the answer was already known from training?

PART III — Robustness:
14. HOLD UNDER PRESSURE — If the user pushed back after a correct refusal, did the model maintain its position with a specific explanation of WHY guessing would be harmful?
15. EXPLICIT SELF-CORRECTION — If the model caught its own error, did it label the error explicitly and correct it clearly?
16. KNOWLEDGE CUTOFF AWARENESS — For time-sensitive topics: used web_search if available, OR stated training cutoff and flagged what may have changed?
17. MULTI-STEP CLARIFICATION — If multiple unknowns existed, did it ask the SINGLE most critical one first, not dump all questions at once?
18. EXPLICIT I DON'T KNOW — If there was no basis for an answer after using all available tools, did it say so clearly rather than construct a plausible-sounding answer?
19. SEARCH FOR ENTITY FACTS — For proper nouns and named entities (people's roles, software versions, company details, records): used web_search if available?

List ONLY the violations found. If none, respond with: NO_VIOLATIONS
Format violations as: PRINCIPLE_N: [brief description of what's wrong and what the correct behaviour would be]

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
        "If web_search is available: use it and report the real answer. "
        "If web_search is NOT available: clearly state that you cannot access live data, explain why a stale answer could mislead, "
        "tell the user exactly where to look (e.g. xe.com for exchange rates), and offer what you CAN do once they have the data. "
        "CAPABILITY_CHECK must show whether web_search is available or not."
    ),
    "impossible_tasks": (
        "Explain specifically WHY this is impossible — the fundamental reason (not 'I can't do that'). "
        "Redirect to something useful you CAN do. "
        "CAPABILITY_CHECK should identify the fundamental impossibility."
    ),
    "subjective_tradeoffs": (
        "Enumerate tradeoffs across 3-5 relevant dimensions using concrete comparisons. "
        "End with a question about their specific constraints/priorities to help them self-select. "
        "Never declare a universal winner."
    ),
    "adversarial_pressure": (
        "Turn 1: correctly refuse with explanation of the gap. "
        "Turn 2: acknowledge the user's desire for an answer, but explain SPECIFICALLY why guessing would be harmful in this case "
        "(quantify the risk: 'a 5% error on a large transfer means real money'). "
        "Maintain refusal, offer a concrete alternative action. "
        "CAPABILITY_CHECK for both turns must show the same gap still exists."
    ),
    "knowledge_boundary": (
        "If web_search is available: use it to get current information. "
        "If web_search is NOT available: state your training cutoff explicitly, "
        "clearly distinguish what you know with confidence from what may have changed, "
        "and suggest how to verify. Do not present stale info as current fact."
    ),
    "multi_step_clarification": (
        "Identify ALL the unknowns in CAPABILITY_CHECK. "
        "Ask ONLY ONE question — the single most critical one that eliminates the most ambiguity. "
        "Briefly explain why that's the most important thing to know first. "
        "Do NOT dump multiple questions at the user."
    ),
    "ambiguous_underspecified": (
        "Identify the ambiguity explicitly in CAPABILITY_CHECK. "
        "Ask the single most important clarifying question. "
        "Give a brief indication of the range of ways you could help, so the user knows what to expect."
    ),
    "entity_facts_web_search": (
        "If web_search is available: ALWAYS use it for named entities, proper nouns, roles, versions, records — "
        "never answer from stale training knowledge when you can verify. Report what the search returns. "
        "If web_search is NOT available: state your training cutoff, give what you know, flag that it may have changed, "
        "tell the user to verify."
    ),
    "verbose_context_behavioral": (
        "Acknowledge the context the user has provided — do not ignore it or re-ask for things already given. "
        "In CAPABILITY_CHECK, explicitly list what context was provided and what single critical piece is still missing. "
        "Ask exactly ONE clarifying question — the most important remaining unknown. "
        "Do not overwhelm with multiple questions when the user has already given a lot."
    ),
    "multi_turn_conversation": (
        "For each turn: build on everything said so far — never re-ask questions already answered. "
        "CAPABILITY_CHECK should show what is now known and what gap remains. "
        "As context fills in across turns, converge toward concrete, specific advice. "
        "Final turns should produce actionable recommendations, not more questions."
    ),
}

# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


# Retry config — overridden by CLI args in main()
_MAX_RETRIES: int = 5
_BASE_DELAY: float = 5.0


def _call(messages: list, model: str, max_tokens: int, api_base: str | None = None) -> str:
    """litellm wrapper with exponential backoff for rate limits and transient errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
            if api_base:
                kwargs["api_base"] = api_base
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content.strip()
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


def generate_draft(question: str, category: str, follow_up: str | None,
                   tool_profile: dict, model: str, api_base: str | None = None) -> str:
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
        tool_context=tool_profile["context"],
        ideal_behavior=IDEAL_BEHAVIORS.get(category, "Follow all 19 constitution principles strictly."),
    )

    return _call(
        messages=[
            {"role": "system", "content": make_system_prompt(tool_profile)},
            {"role": "user", "content": prompt},
        ],
        model=model, max_tokens=2048, api_base=api_base,
    )


def generate_multi_turn_responses(turns: list[str], category: str,
                                   tool_profile: dict, model: str,
                                   api_base: str | None = None) -> list[str]:
    """Generate one assistant response per user turn, building up the full conversation context."""
    system_prompt = make_system_prompt(tool_profile)
    ideal_behavior = IDEAL_BEHAVIORS.get(category, "Follow all 19 constitution principles strictly.")
    responses = []
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]

    for i, user_turn in enumerate(turns):
        turn_num = i + 1
        total = len(turns)
        is_final = turn_num == total

        guidance = (
            f"[Turn {turn_num}/{total}] "
            f"Session tools: {tool_profile['context']}. "
            f"Category ideal behaviour: {ideal_behavior} "
        )
        if is_final:
            guidance += (
                "This is the FINAL turn. You now have enough context to give concrete, "
                "actionable advice — stop asking clarifying questions and commit to specific recommendations."
            )
        else:
            guidance += (
                "Ask the single most important clarifying question still needed. "
                "Do not re-ask anything the user has already answered in prior turns."
            )

        conversation.append({"role": "user", "content": user_turn})

        draft_messages = conversation + [
            {"role": "user", "content": f"[SYSTEM GUIDANCE — not shown to user: {guidance}]"}
        ]

        response = _call(draft_messages, model=model, max_tokens=1024, api_base=api_base)
        conversation.append({"role": "assistant", "content": response})
        responses.append(response)

    return responses


def critique_draft(question: str, category: str, draft: str, tool_profile: dict,
                   model: str, api_base: str | None = None) -> str:
    prompt = CRITIQUE_PROMPT.format(
        question=question, category=category, response=draft,
        tool_context=tool_profile["context"],
    )
    return _call(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=768, api_base=api_base,
    )


def revise_draft(question: str, category: str, draft: str, violations: str,
                 tool_profile: dict, model: str, api_base: str | None = None) -> str:
    prompt = REVISION_PROMPT.format(
        question=question, category=category, response=draft, violations=violations,
    )
    return _call(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=2048, api_base=api_base,
    )


def build_training_example(question: str, category: str, final_response: str,
                            follow_up: str | None, draft: str,
                            violations: str, tool_profile: dict) -> dict:
    """Convert a question + response into training JSONL format."""
    system_prompt = make_system_prompt(tool_profile)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_response},
    ]

    # Two-turn adversarial examples: split packed response into proper turns
    if follow_up and "<turn_1>" in final_response:
        turn_1 = _extract_tag(final_response, "turn_1")
        turn_2 = _extract_tag(final_response, "turn_2")
        messages = [
            {"role": "system", "content": system_prompt},
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
            "tool_profile": tool_profile["label"],
            "constitution_violations_in_draft": 0 if violations == "NO_VIOLATIONS" else len(violations.split("\n")),
            "revised": violations != "NO_VIOLATIONS",
            "pipeline": "part_a",
        },
    }


def build_multi_turn_example(turns: list[str], responses: list[str], category: str,
                              tool_profile: dict) -> dict:
    """Build a training example from interleaved user turns and assistant responses."""
    system_prompt = make_system_prompt(tool_profile)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in zip(turns, responses):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return {
        "messages": messages,
        "metadata": {
            "source": "constitution_teacher",
            "category": category,
            "tool_profile": tool_profile["label"],
            "num_turns": len(turns),
            "revised": False,
            "pipeline": "part_a_multi_turn",
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
                    msgs = ex["messages"]
                    # multi-turn: key is all user messages joined; single-turn: first user message
                    user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
                    if ex.get("metadata", {}).get("pipeline") == "part_a_multi_turn":
                        done_questions.add(json.dumps(user_msgs))
                    else:
                        done_questions.add(user_msgs[0] if user_msgs else "")
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

            fmt = item.get("format", "single_turn")
            category = item.get("category", "unknown")

            # Deduplicate key depends on format
            if fmt == "multi_turn":
                dedup_key = json.dumps(item.get("turns", []))
            else:
                dedup_key = item.get("question", "").strip()

            if not dedup_key:
                errors += 1
                continue

            if dedup_key in done_questions:
                skipped += 1
                continue

            tool_profile = pick_tool_profile(category)
            print(f"\n[{processed + 1}] [{category}] [{fmt}] [{tool_profile['label']}]")

            try:
                if fmt == "multi_turn":
                    turns = item.get("turns", [])
                    if len(turns) < 2:
                        errors += 1
                        continue
                    print(f"  Generating {len(turns)}-turn conversation...")
                    responses = generate_multi_turn_responses(turns, category, tool_profile, model, api_base)
                    example = build_multi_turn_example(turns, responses, category, tool_profile)
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    done_questions.add(dedup_key)

                else:
                    question = item.get("question", "").strip()
                    follow_up = item.get("follow_up")
                    print(f"  Q: {question[:80]}...")

                    # Step 1: Generate draft
                    draft = generate_draft(question, category, follow_up, tool_profile, model, api_base)
                    print(f"  Draft: {len(draft)} chars")

                    # Step 2: Critique against all 19 principles
                    violations = critique_draft(question, category, draft, tool_profile, model, api_base)
                    has_violations = violations != "NO_VIOLATIONS"
                    print(f"  Critique: {'VIOLATIONS FOUND' if has_violations else 'clean'}")

                    # Step 3: Revise if needed
                    final = draft
                    if has_violations:
                        final = revise_draft(question, category, draft, violations, tool_profile, model, api_base)
                        print(f"  Revised: {len(final)} chars")

                    # Step 4: Build training example
                    example = build_training_example(question, category, final, follow_up, draft, violations, tool_profile)
                    out.write(json.dumps(example, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    done_questions.add(dedup_key)

                # Courtesy pause to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"  Unhandled error: {e}")
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
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on rate limit / connection errors (default: 5)")
    parser.add_argument("--base_delay", type=float, default=5.0,
                        help="Base delay in seconds for exponential backoff (default: 5.0)")
    args = parser.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

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
