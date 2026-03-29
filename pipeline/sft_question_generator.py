"""
SFT Question Generator (Part A)
================================
Generates diverse training questions for the constitution-based SFT pipeline.
Uses litellm for vendor-agnostic LLM access (Anthropic, OpenAI, Ollama, Groq, etc.)

Model string examples:
    Anthropic : claude-sonnet-4-5
    OpenAI    : gpt-4o-mini
    Ollama    : ollama/llama3.2  (set OLLAMA_API_BASE=http://localhost:11434)
    Groq      : groq/llama-3.1-70b-versatile

Usage:
    python sft_question_generator.py --count 200 --category all --output data/questions_partA.jsonl
    python sft_question_generator.py --count 10 --category real_time_dependent --output data/sample.jsonl
    python sft_question_generator.py --count 10 --model ollama/llama3.2 --output data/sample.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "user_context_behavioral": {
        "count": 300,
        "description": "Questions requiring user-specific context before answering. The correct response is to ask for that context, not guess.",
        "examples": [
            "Should I invest in index funds or individual stocks?",
            "What programming language should I learn first?",
            "Is it worth getting a master's degree?",
            "Should I take this job offer?",
            "What diet should I follow to lose weight?",
        ],
        "domains": ["personal finance", "career advice", "health", "education", "relationships", "technology choices"],
    },
    "real_time_dependent": {
        "count": 200,
        "description": "Questions that require live/current data the model cannot have. Correct response acknowledges inability to get current data.",
        "examples": [
            "What's the current price of Bitcoin?",
            "What's the weather like in Dublin today?",
            "What did the Fed announce at yesterday's meeting?",
            "Who won last night's Champions League match?",
            "What's the EUR to USD exchange rate right now?",
        ],
        "domains": ["financial markets", "weather", "sports scores", "breaking news", "live events", "current prices"],
    },
    "impossible_tasks": {
        "count": 150,
        "description": "Tasks that are genuinely impossible for fundamental reasons. Correct response explains why it's impossible and redirects.",
        "examples": [
            "Predict the exact Dow Jones closing price next Tuesday.",
            "Tell me what my friend is thinking right now.",
            "Access my Gmail account and read my emails.",
            "Show me an image of the Eiffel Tower.",
            "Tell me the lottery numbers for this Saturday.",
        ],
        "domains": ["prediction", "private data access", "physical impossibility", "mind reading", "future knowledge"],
    },
    "subjective_tradeoffs": {
        "count": 200,
        "description": "Questions with no single right answer — correct response enumerates tradeoffs across relevant dimensions.",
        "examples": [
            "Is Python or Java better for backend development?",
            "Should I rent or buy a home?",
            "Is a PhD worth doing?",
            "Which is better: remote work or office work?",
            "Should I use React or Vue for my frontend?",
        ],
        "domains": ["technology choices", "career decisions", "lifestyle choices", "financial decisions", "tooling comparisons"],
    },
    "adversarial_pressure": {
        "count": 100,
        "description": "Multi-turn scenarios where the user pushes back after a correct refusal. Correct response maintains position with explanation.",
        "examples": [
            "User first asks for current stock price, model correctly refuses. User then says: 'Just give me a rough estimate, I won't hold you to it.'",
            "User asks model to predict election result, model refuses. User says: 'Come on, just guess, everyone's doing it.'",
            "User asks for exchange rate, model refuses. User says: 'Even just an approximate range would help.'",
        ],
        "domains": ["financial pressure", "prediction pressure", "false permission", "social engineering"],
        "format": "two_turn",
    },
    "knowledge_boundary": {
        "count": 200,
        "description": "Questions near or beyond training cutoff, or about obscure/niche topics. Correct response quantifies uncertainty or says I don't know.",
        "examples": [
            "What happened at the recent UN climate summit?",
            "Who is the current Taoiseach of Ireland?",
            "What is the latest version of PyTorch?",
            "What did the latest IPCC report say about 2°C targets?",
            "Who won the 2024 Irish general election?",
        ],
        "domains": ["recent politics", "current technology versions", "recent scientific findings", "current office holders", "recent legislation"],
    },
    "multi_step_clarification": {
        "count": 150,
        "description": "Ambiguous questions with multiple unknowns. Correct response asks the single most critical clarifying question first.",
        "examples": [
            "Help me plan my workout routine.",
            "I want to start investing.",
            "Help me learn to code.",
            "I need help with my diet.",
            "I want to change careers.",
        ],
        "domains": ["fitness planning", "financial planning", "learning paths", "nutrition", "career transition"],
    },
    "ambiguous_underspecified": {
        "count": 200,
        "description": "Requests that are too vague to answer without clarification. Correct response identifies the ambiguity and asks for the most critical specification.",
        "examples": [
            "Help me with Python.",
            "Can you fix my code?",
            "Tell me about machine learning.",
            "Write me a letter.",
            "Help me prepare for my interview.",
        ],
        "domains": ["programming help", "writing assistance", "learning", "interview prep", "general requests"],
    },
    "entity_facts_web_search": {
        "count": 200,
        "description": "Questions about proper nouns and named entities where training data may be stale. Correct response uses web_search if available, or flags the knowledge cutoff if not.",
        "examples": [
            "Who is the current Prime Minister of the UK?",
            "What is the latest version of PyTorch?",
            "What are the current visa requirements to visit Japan from Ireland?",
            "Who won the most recent FIFA World Cup?",
            "What is the current population of Dublin?",
            "Is Python still the most popular programming language?",
        ],
        "domains": ["current office holders", "software versions", "sports records", "population statistics", "legal/regulatory info", "company leadership", "product releases"],
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """You are generating diverse training questions for an AI assistant.

Category: {category_name}
Description: {description}
Target domains: {domains}

Example questions from this category:
{examples}

Generate {count} diverse questions that:
1. Fit this category clearly
2. Come from varied domains (don't repeat the same domain more than twice)
3. Are realistic — the kind of thing real users ask
4. Range from simple to complex
5. Are specific enough to have a clear "correct behavior" (the constitution principle it tests)

{format_instruction}

Return ONLY a JSON array of question strings. No explanation, no numbering outside JSON.
Example format: ["question 1", "question 2", "question 3"]
"""

TWO_TURN_FORMAT = """For the adversarial_pressure category, each item should be a JSON object:
{"turn_1": "the initial question", "turn_2": "the follow-up pressure after model correctly refuses"}
Return an array of these objects."""

SINGLE_TURN_FORMAT = "Return a plain JSON array of question strings."

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_questions_for_category(
    category_name: str,
    count: int,
    model: str = "claude-sonnet-4-5",
    api_base: str | None = None,
) -> list:
    """Generate `count` questions for a single category via litellm (any provider)."""
    spec = CATEGORIES[category_name]
    is_two_turn = spec.get("format") == "two_turn"

    prompt = QUESTION_GENERATION_PROMPT.format(
        category_name=category_name,
        description=spec["description"],
        domains=", ".join(spec["domains"]),
        examples="\n".join(f"- {e}" for e in spec["examples"]),
        count=count,
        format_instruction=TWO_TURN_FORMAT if is_two_turn else SINGLE_TURN_FORMAT,
    )

    kwargs = dict(model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}])
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    questions = json.loads(raw)

    # Normalise to list of dicts with consistent schema
    result = []
    for q in questions:
        if isinstance(q, str):
            result.append({
                "question": q,
                "category": category_name,
                "format": "single_turn",
            })
        elif isinstance(q, dict):
            result.append({
                "question": q.get("turn_1", ""),
                "follow_up": q.get("turn_2", ""),
                "category": category_name,
                "format": "two_turn",
            })

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part A questions for constitution-based SFT")
    parser.add_argument("--count", type=int, default=None,
                        help="Questions per category (overrides per-category defaults)")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all"] + list(CATEGORIES.keys()),
                        help="Which category to generate (default: all)")
    parser.add_argument("--output", type=str, default="pipeline/data/questions_partA.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5",
                        help="Model string for litellm (e.g. claude-sonnet-4-5, gpt-4o-mini, ollama/llama3.2)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Questions to request per API call (reduce if hitting token limits)")
    args = parser.parse_args()

    # litellm reads API keys from env vars automatically:
    # ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, etc.
    # For Ollama: set OLLAMA_API_BASE or pass --api_base

    categories_to_run = list(CATEGORIES.keys()) if args.category == "all" else [args.category]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for cat_name in categories_to_run:
            target_count = args.count or CATEGORIES[cat_name]["count"]
            print(f"\n[{cat_name}] Generating {target_count} questions...")

            generated = []
            remaining = target_count

            while remaining > 0:
                batch = min(args.batch_size, remaining)
                print(f"  Requesting batch of {batch}...")

                try:
                    batch_questions = generate_questions_for_category(
                        category_name=cat_name,
                        count=batch,
                        model=args.model,
                        api_base=args.api_base,
                    )
                    generated.extend(batch_questions)
                    remaining -= len(batch_questions)
                    print(f"  Got {len(batch_questions)} questions. Total so far: {len(generated)}")

                    # Courtesy pause
                    if remaining > 0:
                        time.sleep(1)

                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON parse error in batch: {e}. Retrying...")
                    time.sleep(2)
                    continue
                except Exception as e:
                    if "rate" in str(e).lower():
                        print("  Rate limit hit. Waiting 30s...")
                        time.sleep(30)
                    else:
                        print(f"  Error: {e}. Retrying...")
                        time.sleep(3)
                    continue

            # Write to file
            for item in generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1

            print(f"  [DONE] {len(generated)} questions written for {cat_name}")

    print(f"\nTotal questions written: {total_written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
