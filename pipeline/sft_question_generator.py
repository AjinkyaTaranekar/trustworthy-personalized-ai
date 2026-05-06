"""
SFT Question Generator (Part A)
================================
Generates diverse training questions for the constitution-based SFT pipeline.
Uses litellm for vendor-agnostic LLM access (Anthropic, OpenAI, Ollama, Groq, NVIDIA NIM, etc.)

Model string examples:
    NVIDIA NIM : nvidia_nim/minimaxai/minimax-m2.7   (recommended — free, frontier quality)
    NVIDIA NIM : nvidia_nim/moonshotai/kimi-k2.6
    Anthropic  : claude-sonnet-4-6
    Groq       : groq/llama-3.3-70b-versatile
    Ollama     : ollama/llama3.2  (set OLLAMA_API_BASE=http://localhost:11434)

Batch size guide (--batch_size):
    15  — safe default; works for all categories including verbose ones (~2,250 output tokens)
    30  — fine for simple categories (single-turn, real-time, etc.)
    40  — max recommended for simple categories; avoid for verbose/multi-turn

Usage:
    python sft_question_generator.py --count 200 --type all --output data/questions_partA.jsonl
    python sft_question_generator.py --count 30 --type real_time_dependent --output data/sample.jsonl
    python sft_question_generator.py --count 15 --model nvidia_nim/minimaxai/minimax-m2.7 --output data/questions_partA.jsonl
    python sft_question_generator.py --count 15 --model nvidia_nim/minimaxai/minimax-m2.7 --type adversarial_pressure --output data/adv.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import random
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
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    "user_context_behavioral": {
        "count": 150,
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
        "count": 100,
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
        "count": 75,
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
        "count": 100,
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
        "count": 50,
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
        "count": 100,
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
        "count": 75,
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
        "count": 100,
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
        "count": 100,
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
        "count": 100,
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
        "count": 75,
        "description": "A 3-5 turn user-side scaffold where the user progressively reveals context across turns. The model must ask good clarifying questions, track what it already knows, and converge on useful advice as context fills in — not repeat questions it already asked or ignore information already given.",
        "examples": [
            '{"turns": ["I want to start investing.", "I have about €500 a month I can put away.", "I\'m 29, no dependents, emergency fund is sorted.", "I\'m comfortable with medium risk, I\'d be upset but not panicked by a 20% dip."]}',
            '{"turns": ["Help me plan a birthday dinner for my friend.", "She\'s vegetarian and has a nut allergy.", "We\'re in Dublin, budget is around €40 per person.", "About 8 people — mix of her close friends and some people she doesn\'t know that well."]}',
            '{"turns": ["I\'m thinking about doing a PhD.", "In computer science, probably focusing on NLP or ML.", "I have a first-class undergrad and a distinction in my MSc.", "I\'m 26, no partner, no mortgage. The academic job market worries me though."]}',
        ],
        "domains": ["financial planning", "event planning", "career advice", "learning paths", "health consultation", "travel planning", "technology choices"],
        "format": "multi_turn",
    },
    # ── Empathy category ─────────────────────────────────────────────────────
    # This category does NOT call the LLM to generate questions.
    # Instead it loads from data/appraisal_labels.jsonl (produced by
    # appraisal_labeller.py).  The special "loader" key signals that
    # generate_questions_for_category() should read from file rather than prompt.
    "appraisal_empathy": {
        "count": 150,
        "description": (
            "User utterances from EmpatheticDialogues labelled with 21-dim OCC "
            "appraisal vectors by AppraisePLM. The model must produce an <appraisal> "
            "block inside <think> that identifies the dominant dimensions, then give "
            "an empathetically conditioned <answer>."
        ),
        "examples": [
            "I finally got the promotion I've been working towards for three years!",
            "My dog passed away last night. I can't stop crying.",
            "I got rejected from every grad school I applied to.",
            "My partner and I had a huge fight and I don't know what to do.",
            "I just found out I'm pregnant — it wasn't planned.",
        ],
        "domains": ["personal achievement", "loss and grief", "rejection", "relationship conflict",
                    "unexpected life events", "anxiety", "excitement", "frustration"],
        "format": "appraisal_empathy",
        "loader": "appraisal_labels",           # load from file, not LLM generation
        "labels_path": "data/appraisal_labels.jsonl",
    },
}

# ---------------------------------------------------------------------------
# Diversity axes — one is assigned per batch to anchor geographic/cultural context.
# The list is cycled sequentially so across a full category run every axis appears.
# ---------------------------------------------------------------------------

DIVERSITY_AXES = [
    {"region": "South Asia (India, Bangladesh, Pakistan, Sri Lanka)", "culture": "Hindu or Muslim cultural context", "demographic": "lower-middle-income urban professional or rural family"},
    {"region": "East Africa (Kenya, Tanzania, Ethiopia, Uganda)", "culture": "Christian or Muslim African cultural context", "demographic": "young entrepreneur or informal-sector worker"},
    {"region": "Southeast Asia (Philippines, Indonesia, Vietnam, Thailand, Malaysia)", "culture": "Catholic, Muslim, or Buddhist cultural context", "demographic": "OFW/migrant worker, gig-economy worker, or urban student"},
    {"region": "Latin America (Brazil, Mexico, Colombia, Argentina, Peru)", "culture": "Catholic or secular Latin American context", "demographic": "middle-class family, recent graduate, or informal worker"},
    {"region": "Middle East (Egypt, Saudi Arabia, UAE, Turkey, Jordan)", "culture": "Sunni or Shia Muslim cultural context", "demographic": "urban professional, university student, or expatriate worker"},
    {"region": "East Asia (China, Japan, South Korea, Taiwan)", "culture": "Confucian, Buddhist, or secular East Asian context", "demographic": "technology worker, university student, or factory worker"},
    {"region": "West Africa (Nigeria, Ghana, Senegal, Côte d'Ivoire)", "culture": "Christian or Muslim West African context", "demographic": "young professional, small-business owner, or student"},
    {"region": "Eastern Europe (Poland, Ukraine, Romania, Czechia, Hungary)", "culture": "Catholic or Orthodox Christian cultural context", "demographic": "skilled tradesperson, academic, or migrant worker"},
    {"region": "North Africa (Morocco, Algeria, Tunisia)", "culture": "Muslim cultural context with French colonial heritage", "demographic": "bilingual urban professional or rural youth"},
    {"region": "South America (Chile, Venezuela, Bolivia, Ecuador, Paraguay)", "culture": "Indigenous or mestizo Latin American context", "demographic": "lower-income family, small farmer, or indigenous community member"},
    {"region": "Central Asia (Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan)", "culture": "Muslim context with Soviet cultural legacy", "demographic": "migrant worker, government employee, or semi-nomadic background"},
    {"region": "South Asian diaspora (UK, Canada, UAE, Australia)", "culture": "Hindu, Sikh, or Muslim diaspora context", "demographic": "second-generation immigrant professional or student"},
    {"region": "African diaspora (France, UK, US, Belgium)", "culture": "African diaspora cultural context", "demographic": "first-generation immigrant student or essential worker"},
    {"region": "Scandinavia (Sweden, Norway, Denmark, Finland)", "culture": "Lutheran or secular Nordic welfare-state context", "demographic": "public-sector worker, student, or new immigrant"},
    {"region": "Southern Europe (Italy, Spain, Greece, Portugal)", "culture": "Catholic Mediterranean cultural context", "demographic": "young adult facing economic uncertainty or retiree"},
    {"region": "Caribbean (Jamaica, Trinidad, Haiti, Cuba, Dominican Republic)", "culture": "Afro-Caribbean Christian or Vodou cultural context", "demographic": "lower-income worker, small entrepreneur, or diaspora returnee"},
    {"region": "Jewish communities (Israel, US diaspora, France, Argentina)", "culture": "Ashkenazi or Sephardi Jewish cultural/religious context", "demographic": "urban professional or observant family with mixed secular ties"},
    {"region": "Buddhist communities (Myanmar, Thailand, Sri Lanka, Tibet, Japan)", "culture": "Theravada, Mahayana, or Vajrayana Buddhist context", "demographic": "monastic, lay practitioner, or secular Buddhist professional"},
    {"region": "Pacific Islands (Papua New Guinea, Fiji, Samoa, Tonga)", "culture": "Christian or indigenous Melanesian/Polynesian context", "demographic": "rural community member, subsistence farmer, or remittance-receiving family"},
    {"region": "Horn of Africa (Somalia, Eritrea, Djibouti)", "culture": "Muslim cultural context with pastoralist or urban refugee experience", "demographic": "displaced person, aid-sector worker, or diaspora member"},
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DEDUP_INSTRUCTION_TEMPLATE = """DEDUPLICATION — the questions below were already generated in earlier batches. Do NOT repeat or closely paraphrase any of them. Generate questions on entirely different topics, scenarios, and situations.
{sample}"""

QUESTION_GENERATION_PROMPT = """You are generating diverse training questions for an AI assistant.

Category: {category_name}
Description: {description}
Target domains: {domains}

Example questions from this category:
{examples}

BATCH GEOGRAPHIC/CULTURAL FOCUS — at least 60% of questions in this batch MUST reflect this specific context:
  Region: {batch_region}
  Cultural/religious background: {batch_culture}
  User demographic: {batch_demographic}

Use country-specific details: local currencies, laws, healthcare systems, financial instruments, naming conventions, social norms, and cultural practices (e.g. halal finance, joint family decisions, bride price, chit funds, stokvel savings, M-Pesa, UPI payments, NHS vs private care, mandatory military service). Do NOT default to US/UK context unless the batch region specifies it.

Generate {count} diverse questions that:
1. Fit this category clearly
2. Come from varied domains within the batch region (don't repeat the same domain more than twice)
3. Are realistic — the kind of thing a real user in that region and culture would ask
4. Range from simple to complex
5. Are specific enough to have a clear "correct behavior" (the constitution principle it tests)

{already_generated}

{format_instruction}

Return ONLY a JSON array. No explanation, no numbering outside JSON.
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


def load_appraisal_questions(category_name: str, count: int) -> list:
    """
    Load appraisal_empathy questions from data/appraisal_labels.jsonl.
    Each row in the labels file becomes one training question entry with
    appraisal_meta attached so the gold-response generator can use it.
    """
    spec = CATEGORIES[category_name]
    labels_path = Path(spec.get("labels_path", "data/appraisal_labels.jsonl"))

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Appraisal labels file not found: {labels_path}\n"
            "Run:  python pipeline/appraisal_labeller.py\n"
            "Or:   python pipeline/appraisal_labeller.py --mock_model  (for testing)"
        )

    rows = []
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    random.shuffle(rows)
    rows = rows[:count]

    result = []
    for row in rows:
        result.append({
            "question":       row["utterance"],
            "category":       category_name,
            "format":         "appraisal_empathy",
            "appraisal_meta": {
                "emotion":           row.get("emotion", ""),
                "top3":              row.get("top3", []),
                "appraisal_reading": row.get("appraisal_reading", ""),
                "valence":           row.get("valence", 0.5),
                "appraisal_named":   row.get("appraisal_named", {}),
            },
        })

    return result


def generate_questions_for_category(
    category_name: str,
    count: int,
    model: str = "claude-sonnet-4-5",
    api_base: str | None = None,
    diversity_slot: dict | None = None,
    existing_questions: list[str] | None = None,
) -> list:
    """Generate `count` questions for a single category via litellm (any provider).
    For the appraisal_empathy category the questions are loaded from file instead.

    diversity_slot: dict with keys region/culture/demographic — anchors the batch to a specific
    geographic and cultural context. If None, a random axis is chosen.

    existing_questions: questions already generated in earlier batches for this category,
    injected as a dedup list so the model avoids repeating them.
    """
    spec = CATEGORIES[category_name]

    # File-loader path: skip LLM generation entirely
    if spec.get("loader") == "appraisal_labels":
        return load_appraisal_questions(category_name, count)

    fmt = spec.get("format", "single_turn")
    if fmt == "two_turn":
        format_instruction = TWO_TURN_FORMAT
    elif fmt == "multi_turn":
        format_instruction = MULTI_TURN_FORMAT
    elif fmt == "verbose_single_turn":
        format_instruction = VERBOSE_SINGLE_TURN_FORMAT
    else:
        format_instruction = SINGLE_TURN_FORMAT

    # Build dedup block — cap token use for verbose/multi-turn formats
    if existing_questions:
        if fmt in ("verbose_single_turn", "multi_turn"):
            # First sentence only to keep prompt size manageable
            sample_qs = [q[:100] + "…" for q in existing_questions[-10:]]
        else:
            sample_qs = existing_questions[-30:]
        sample_text = "\n".join(f"  - {q}" for q in sample_qs)
        already_generated = DEDUP_INSTRUCTION_TEMPLATE.format(sample=sample_text)
    else:
        already_generated = ""

    slot = diversity_slot or random.choice(DIVERSITY_AXES)

    prompt = QUESTION_GENERATION_PROMPT.format(
        category_name=category_name,
        description=spec["description"],
        domains=", ".join(spec["domains"]),
        examples="\n".join(f"- {e}" for e in spec["examples"]),
        count=count,
        format_instruction=format_instruction,
        batch_region=slot["region"],
        batch_culture=slot["culture"],
        batch_demographic=slot["demographic"],
        already_generated=already_generated,
    )

    kwargs = dict(
        model=model,
        max_tokens=4096*2,  # high token limit to avoid truncation of verbose/multi-turn responses; retries with smaller batch if it does truncate
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

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    if finish_reason == "length":
        # Response was cut off mid-JSON — try to salvage, else raise so caller retries with smaller batch
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(
                f"Response truncated at token limit (finish_reason='length'). "
                "Reduce --batch_size and retry."
            )
    else:
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
# Per-category worker (used by both sequential and parallel modes)
# ---------------------------------------------------------------------------


def _run_category(
    cat_name: str,
    target_count: int,
    batch_size: int,
    model: str,
    api_base: str | None,
    out_file,
    file_lock: threading.Lock,
    cat_idx: int,
    n_cats: int,
) -> int:
    """Run the full batch loop for one category. Thread-safe via file_lock."""
    cat_written = 0
    remaining = target_count
    current_batch_size = batch_size
    batch_num = 0
    cat_start = time.monotonic()
    cat_questions_seen: list[str] = []
    diversity_idx = 0

    n_batches = -(-target_count // batch_size)
    print(f"\n[{cat_idx}/{n_cats}] {cat_name} — target {target_count} questions "
          f"(~{n_batches} batch{'es' if n_batches != 1 else ''} of {batch_size})")

    while remaining > 0:
        batch = min(current_batch_size, remaining)
        batch_num += 1
        diversity_slot = DIVERSITY_AXES[diversity_idx % len(DIVERSITY_AXES)]
        diversity_idx += 1
        print(f"  [{cat_name}][batch {batch_num}] requesting {batch} questions "
              f"({remaining} remaining, batch_size={current_batch_size}, "
              f"region={diversity_slot['region'].split('(')[0].strip()})...")
        t0 = time.monotonic()

        try:
            batch_questions = generate_questions_for_category(
                category_name=cat_name,
                count=batch,
                model=model,
                api_base=api_base,
                diversity_slot=diversity_slot,
                existing_questions=cat_questions_seen if cat_questions_seen else None,
            )
            elapsed = time.monotonic() - t0

            with file_lock:
                for item in batch_questions:
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if isinstance(item.get("question"), str):
                        cat_questions_seen.append(item["question"])
                    elif isinstance(item.get("turns"), list) and item["turns"]:
                        cat_questions_seen.append(item["turns"][0])
                out_file.flush()

            cat_written += len(batch_questions)
            remaining -= len(batch_questions)
            print(f"  [{cat_name}][batch {batch_num}] ✓ {len(batch_questions)} written in {elapsed:.1f}s "
                  f"(total: {cat_written}/{target_count})")

            if remaining > 0:
                time.sleep(1)

        except json.JSONDecodeError as e:
            print(f"  [{cat_name}][batch {batch_num}] JSON parse error: {e}. Retrying same batch...")
            time.sleep(2)
            batch_num -= 1
            diversity_idx -= 1
            continue
        except ValueError as e:
            if "truncated" in str(e).lower() or "null content" in str(e).lower():
                new_batch = max(10, current_batch_size // 2)
                print(f"  [{cat_name}][batch {batch_num}] Truncation/null — halving batch size: "
                      f"{current_batch_size} → {new_batch}. Retrying...")
                current_batch_size = new_batch
                batch_num -= 1
                diversity_idx -= 1
                continue
            print(f"  [{cat_name}][batch {batch_num}] Error: {e}. Skipping category.")
            break
        except Exception as e:
            print(f"  [{cat_name}][batch {batch_num}] Unexpected error: {e}. Skipping category.")
            break

    cat_elapsed = time.monotonic() - cat_start
    print(f"  [{cat_name}] DONE: {cat_written}/{target_count} in {cat_elapsed:.1f}s")
    return cat_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part A questions for constitution-based SFT")
    parser.add_argument("--count", type=int, default=None,
                        help="Questions per category (overrides per-category defaults)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: generate 1 question per category and write to data/smoke_questions.jsonl")
    parser.add_argument("--type", "--category", dest="category", type=str, default="all",
                        choices=["all"] + list(CATEGORIES.keys()),
                        help="Question category to generate (default: all). "
                             "E.g. --type appraisal_empathy or --type adversarial_pressure")
    parser.add_argument("--output", type=str, default="data/questions_partA.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="nvidia_nim/minimaxai/minimax-m2.7",
                        help="litellm model string (e.g. nvidia_nim/minimaxai/minimax-m2.7, claude-sonnet-4-6)")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Questions per API call. 15=safe for all types; 30=fine for simple types; "
                             "avoid >20 for verbose_context_behavioral/multi_turn (hits token limits)")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on rate limit / connection errors (default: 5)")
    parser.add_argument("--base_delay", type=float, default=3.0,
                        help="Base delay in seconds for exponential backoff (default: 3.0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output file instead of appending to it")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel category workers when --type all (default: 5; use 1 to disable)")
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

    file_mode = "w" if args.overwrite else "a"
    if file_mode == "a" and Path(args.output).exists():
        existing = sum(1 for _ in open(args.output, encoding="utf-8"))
        print(f"Appending to existing file: {args.output} ({existing} rows already present; use --overwrite to start fresh)")

    total_written = 0
    n_cats = len(categories_to_run)
    run_start = time.monotonic()
    file_lock = threading.Lock()

    with open(args.output, file_mode, encoding="utf-8") as f:
        use_parallel = n_cats > 1 and not args.smoke and args.workers > 1
        if use_parallel:
            max_workers = min(args.workers, n_cats)
            print(f"Running {n_cats} categories in parallel ({max_workers} workers)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_category,
                        cat_name=cat_name,
                        target_count=args.count or CATEGORIES[cat_name]["count"],
                        batch_size=args.batch_size,
                        model=args.model,
                        api_base=args.api_base,
                        out_file=f,
                        file_lock=file_lock,
                        cat_idx=cat_idx,
                        n_cats=n_cats,
                    ): cat_name
                    for cat_idx, cat_name in enumerate(categories_to_run, 1)
                }
                for future in concurrent.futures.as_completed(futures):
                    cat_name = futures[future]
                    try:
                        written = future.result()
                        total_written += written
                    except Exception as e:
                        print(f"[{cat_name}] category thread failed: {e}")
        else:
            for cat_idx, cat_name in enumerate(categories_to_run, 1):
                written = _run_category(
                    cat_name=cat_name,
                    target_count=args.count or CATEGORIES[cat_name]["count"],
                    batch_size=args.batch_size,
                    model=args.model,
                    api_base=args.api_base,
                    out_file=f,
                    file_lock=file_lock,
                    cat_idx=cat_idx,
                    n_cats=n_cats,
                )
                total_written += written

    total_elapsed = time.monotonic() - run_start
    print(f"\n{'='*55}")
    print(f"Run complete in {total_elapsed:.1f}s")
    print(f"Questions written this run : {total_written}")
    print(f"Output                     : {args.output}")


if __name__ == "__main__":
    main()
