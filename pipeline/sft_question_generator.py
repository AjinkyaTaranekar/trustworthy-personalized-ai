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
import random
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# Retry config — overridden by CLI args in main()
_MAX_RETRIES: int = 5
_BASE_DELAY: float = 5.0

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
    "verbose_context_behavioral": {
        "count": 200,
        "description": "User narrates a paragraph (or several) of personal context before asking a question. Tests whether the model identifies the actual question, picks out the relevant context, and asks for the single remaining critical unknown rather than overwhelming the user with follow-ups or ignoring the context they already gave.",
        "examples": [
            "I'm a 34-year-old software engineer, been at the same company 6 years, earning €85k. I have two kids aged 5 and 8, a mortgage with 15 years left, and my company is about to restructure. I've been offered a position at a startup that pays €110k but they only have 6 months runway. My wife is nervous but supportive. Should I take the job?",
            "So I've been trying to lose weight for 3 months. I've cut down on sugar, I'm walking 30 minutes a day, and I've reduced my portions. I've only lost 2kg. I'm 47, 5'9\", about 95kg. Desk job, sleep 5-6 hours because of deadlines. Tried low-carb twice before with mixed results. What am I doing wrong?",
            "Hi, bit of a long one sorry. I'm finishing my undergrad in computer science, first in my family to go to college. I've got a graduate job offer for €42k from a large bank doing internal tooling. I also got accepted to an MSc in ML at a decent university — full-time, €12k fees, no scholarship. I have €8k saved and some student debt. I'm 22. My parents think I should take the job. What would you do?",
        ],
        "domains": ["career decisions with rich context", "health and fitness after failed attempts", "financial decisions under constraints", "education vs employment dilemmas", "housing decisions with family context", "relationship advice with backstory"],
        "format": "verbose_single_turn",
    },
    "multi_turn_conversation": {
        "count": 150,
        "description": "A 3-5 turn user-side scaffold where the user progressively reveals context across turns. The model must ask good clarifying questions, track what it already knows, and converge on useful advice as context fills in — not repeat questions it already asked or ignore information already given.",
        "examples": [
            '{"turns": ["I want to start investing.", "I have about €500 a month I can put away.", "I\'m 29, no dependents, emergency fund is sorted.", "I\'m comfortable with medium risk, I\'d be upset but not panicked by a 20% dip."]}',
            '{"turns": ["Help me plan a birthday dinner for my friend.", "She\'s vegetarian and has a nut allergy.", "We\'re in Dublin, budget is around €40 per person.", "About 8 people — mix of her close friends and some people she doesn\'t know that well."]}',
            '{"turns": ["I\'m thinking about doing a PhD.", "In computer science, probably focusing on NLP or ML.", "I have a first-class undergrad and a distinction in my MSc.", "I\'m 26, no partner, no mortgage. The academic job market worries me though."]}',
        ],
        "domains": ["financial planning", "event planning", "career advice", "learning paths", "health consultation", "travel planning", "technology choices"],
        "format": "multi_turn",
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

MULTI_TURN_FORMAT = """For the multi_turn_conversation category, each item is a JSON object with a "turns" array containing 3-5 user messages that progressively reveal context. Only user messages — do NOT write assistant responses.
Example: {"turns": ["opening message", "adds more context", "answers the model's implied question", "final constraint"]}
Return an array of these objects. Make turns feel natural — the user is typing messages in a real chat, not filling out a form."""

VERBOSE_SINGLE_TURN_FORMAT = """For the verbose_context_behavioral category, each question is a long paragraph (3-8 sentences) where the user provides rich personal context before asking. Return a plain JSON array of these paragraph-style question strings."""

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

    fmt = spec.get("format", "single_turn")
    if fmt == "two_turn":
        format_instruction = TWO_TURN_FORMAT
    elif fmt == "multi_turn":
        format_instruction = MULTI_TURN_FORMAT
    elif fmt == "verbose_single_turn":
        format_instruction = VERBOSE_SINGLE_TURN_FORMAT
    else:
        format_instruction = SINGLE_TURN_FORMAT

    prompt = QUESTION_GENERATION_PROMPT.format(
        category_name=category_name,
        description=spec["description"],
        domains=", ".join(spec["domains"]),
        examples="\n".join(f"- {e}" for e in spec["examples"]),
        count=count,
        format_instruction=format_instruction,
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

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    questions = json.loads(raw)

    # Normalise to list of dicts with consistent schema
    fmt = spec.get("format", "single_turn")
    result = []
    for q in questions:
        if isinstance(q, str):
            result.append({
                "question": q,
                "category": category_name,
                "format": fmt if fmt == "verbose_single_turn" else "single_turn",
            })
        elif isinstance(q, dict):
            if "turns" in q:
                result.append({
                    "turns": q["turns"],
                    "category": category_name,
                    "format": "multi_turn",
                })
            elif "turn_1" in q:
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
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: generate 1 question per category and write to data/smoke_questions.jsonl")
    parser.add_argument("--category", type=str, default="all",
                        choices=["all"] + list(CATEGORIES.keys()),
                        help="Which category to generate (default: all)")
    parser.add_argument("--output", type=str, default="data/questions_partA.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-5",
                        help="Model string for litellm (e.g. claude-sonnet-4-5, gpt-4o-mini, ollama/llama3.2)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Questions to request per API call (reduce if hitting token limits)")
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
        args.output = "data/smoke_questions.jsonl"
        print("Smoke test mode: 1 question per category → data/smoke_questions.jsonl")

    categories_to_run = list(CATEGORIES.keys()) if args.category == "all" else [args.category]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for cat_name in categories_to_run:
            target_count = args.count or CATEGORIES[cat_name]["count"]
            print(f"\n[{cat_name}] Generating {target_count} question(s)...")

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

                    if remaining > 0:
                        time.sleep(1)

                except json.JSONDecodeError as e:
                    print(f"  JSON parse error: {e}. Retrying...")
                    time.sleep(2)
                    continue
                except Exception as e:
                    print(f"  Error: {e}. Skipping category.")
                    break

            for item in generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1

            print(f"  [DONE] {len(generated)} question(s) written for {cat_name}")

    print(f"\nTotal questions written: {total_written}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
