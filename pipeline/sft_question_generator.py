"""
SFT Question Generator (Part A) — v2
======================================
Generates diverse training questions for the 23-principle constitution SFT pipeline.

P1-P19: original trustworthiness principles
P20: First Principles decomposition
P21: 5W+H questioning
P22: Consequence Check
P23: Interleaved Tool Chaining

Model string examples:
    NVIDIA NIM : nvidia_nim/minimaxai/minimax-m2.7   (recommended — free, frontier quality)
    NVIDIA NIM : nvidia_nim/moonshotai/kimi-k2.6
    Anthropic  : claude-sonnet-4-6
    Groq       : groq/llama-3.3-70b-versatile
    Ollama     : ollama/llama3.2  (set OLLAMA_API_BASE=http://localhost:11434)

Usage:
    python sft_question_generator.py --count 200 --type all --output data/questions_partA.jsonl
    python sft_question_generator.py --count 30 --type interleaved_tool_reasoning --output data/sample.jsonl
    python sft_question_generator.py --count 15 --model nvidia_nim/minimaxai/minimax-m2.7 --output data/questions_partA.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

import litellm
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output on Windows (cp1252 console can't render ₹, ৳, ₦, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = {
    # "user_context_behavioral": {
    #     "count": 150,
    #     "description": (
    #         "Questions requiring user-specific context before answering. Correct response uses "
    #         "5W+H to identify exactly which dimension (WHO the person is, WHAT their situation is) "
    #         "is the critical unknown, then asks for that context rather than guessing."
    #     ),
    #     "examples": [
    #         "Should I invest in index funds or individual stocks?",
    #         "What programming language should I learn first?",
    #         "Is it worth getting a master's degree?",
    #         "Should I take this job offer?",
    #         "What diet should I follow to lose weight?",
    #     ],
    #     "domains": ["personal finance", "career advice", "health", "education", "relationships", "technology choices"],
    #     "chaining_note": "No tool chaining needed — the gap is missing user context, not missing external data.",
    # },
    # "real_time_dependent": {
    #     "count": 100,
    #     "description": (
    #         "Questions requiring live data. Some questions combine live lookup AND computation "
    #         "(e.g. 'What is today's EUR/INR rate and how much would €300 cost me?'). "
    #         "These require web_search → python_execute chaining. CONSEQUENCE_CHECK must quantify "
    #         "the cost of using stale data."
    #     ),
    #     "examples": [
    #         "What's the current price of Bitcoin?",
    #         "What's the weather like in Dublin today?",
    #         "What did the Fed announce at yesterday's meeting?",
    #         "What is the EUR to INR rate today, and how much is €500 in rupees?",
    #         "What's the current 10-year Irish government bond yield?",
    #     ],
    #     "domains": ["financial markets", "weather", "sports scores", "breaking news", "live events", "current prices", "exchange rates with conversion"],
    #     "chaining_note": "At least 30% of questions should require search → compute chaining.",
    # },
    # "impossible_tasks": {
    #     "count": 75,
    #     "description": (
    #         "Tasks genuinely impossible for fundamental reasons. CONSEQUENCE_CHECK formalises "
    #         "why the task is impossible — the failure mode IS the point. First Principles "
    #         "must name the irreducible reason (not just 'I can't do that')."
    #     ),
    #     "examples": [
    #         "Predict the exact Dow Jones closing price next Tuesday.",
    #         "Tell me what my friend is thinking right now.",
    #         "Access my Gmail account and read my emails.",
    #         "Tell me the lottery numbers for this Saturday.",
    #         "Show me a real-time satellite image of my house.",
    #     ],
    #     "domains": ["prediction", "private data access", "physical impossibility", "mind reading", "future knowledge"],
    #     "chaining_note": "No chaining — the task is impossible, not just hard to look up.",
    # },
    # "subjective_tradeoffs": {
    #     "count": 100,
    #     "description": (
    #         "Questions with no single right answer. First Principles identifies the irreducible "
    #         "decision criterion (what does the user ultimately optimise for?). CONSEQUENCE_CHECK "
    #         "flags that declaring a winner without knowing constraints causes real harm."
    #     ),
    #     "examples": [
    #         "Is Python or Java better for backend development?",
    #         "Should I rent or buy a home in Mumbai?",
    #         "Is a PhD worth doing in Ireland right now?",
    #         "Which is better: remote work or office work?",
    #         "Should I use React or Vue for my frontend?",
    #     ],
    #     "domains": ["technology choices", "career decisions", "lifestyle choices", "financial decisions", "tooling comparisons"],
    #     "chaining_note": "Some tradeoff questions benefit from web_search for current data (e.g. current rent vs buy ratios in a specific city).",
    # },
    # "adversarial_pressure": {
    #     "count": 50,
    #     "description": (
    #         "Multi-turn scenarios where the user pushes back after a correct refusal. "
    #         "CONSEQUENCE_CHECK in Turn 1 identifies the stakes and quantifies the risk of guessing. "
    #         "Turn 2 must reference those stakes specifically — not just say 'I can't'."
    #     ),
    #     "examples": [
    #         "User asks for current stock price, model refuses. User: 'Just give me a rough estimate, I won't hold you to it.'",
    #         "User asks model to predict election result, model refuses. User: 'Come on, just guess, everyone's doing it.'",
    #         "User asks for GST rate, model refuses without web_search. User: 'You must know roughly — just tell me.'",
    #     ],
    #     "domains": ["financial pressure", "prediction pressure", "false permission", "social engineering", "tax/legal pressure"],
    #     "format": "two_turn",
    #     "chaining_note": "No chaining — the adversarial scenario tests hold-under-pressure, not tool use.",
    # },
    # "knowledge_boundary": {
    #     "count": 100,
    #     "description": (
    #         "Questions near or beyond training cutoff. Some combine knowledge boundary lookup "
    #         "with synthesis across multiple search results. CONSEQUENCE_CHECK must flag the "
    #         "specific risk of acting on outdated information."
    #     ),
    #     "examples": [
    #         "What happened at the recent UN climate summit?",
    #         "Who is the current Taoiseach of Ireland?",
    #         "What is the latest version of PyTorch, and does it support my CUDA 11.8 setup?",
    #         "What did the latest IPCC report say about 2°C targets?",
    #         "Who won the 2024 Irish general election?",
    #     ],
    #     "domains": ["recent politics", "current technology versions", "recent scientific findings", "current office holders", "recent legislation"],
    #     "chaining_note": "Version/compatibility questions benefit from web_search → read_url chaining.",
    # },
    # "multi_step_clarification": {
    #     "count": 75,
    #     "description": (
    #         "Ambiguous questions with multiple unknowns. 5W+H in CAPABILITY_CHECK drives which "
    #         "clarifying question is most critical — the one that eliminates the most ambiguity. "
    #         "CONSEQUENCE_CHECK flags the risk of giving generic advice without context."
    #     ),
    #     "examples": [
    #         "Help me plan my workout routine.",
    #         "I want to start investing.",
    #         "Help me learn to code.",
    #         "I need help with my diet.",
    #         "I want to change careers.",
    #     ],
    #     "domains": ["fitness planning", "financial planning", "learning paths", "nutrition", "career transition"],
    #     "chaining_note": "No chaining — gap is missing user context.",
    # },
    # "ambiguous_underspecified": {
    #     "count": 100,
    #     "description": (
    #         "Requests too vague to answer without clarification. First Principles surfaces what "
    #         "is fundamentally unknown (the irreducible unknown). CONSEQUENCE_CHECK flags the "
    #         "cost of guessing the wrong interpretation."
    #     ),
    #     "examples": [
    #         "Help me with Python.",
    #         "Can you fix my code?",
    #         "Tell me about machine learning.",
    #         "Write me a letter.",
    #         "Help me prepare for my interview.",
    #     ],
    #     "domains": ["programming help", "writing assistance", "learning", "interview prep", "general requests"],
    #     "chaining_note": "No chaining — gap is underspecification.",
    # },
    # "entity_facts_web_search": {
    #     "count": 100,
    #     "description": (
    #         "Questions about proper nouns and named entities. Some require web_search to find "
    #         "a URL, then read_url to extract a specific fact from that page. "
    #         "CONSEQUENCE_CHECK flags the risk of presenting stale entity facts as current."
    #     ),
    #     "examples": [
    #         "Who is the current Prime Minister of the UK?",
    #         "What is the latest version of PyTorch?",
    #         "What are the current visa requirements to visit Japan from India?",
    #         "What does the Python 3.13 changelog say about the GIL?",
    #         "Is Anthropic still offering free Claude API access for researchers?",
    #     ],
    #     "domains": ["current office holders", "software versions", "sports records", "population statistics", "legal/regulatory info", "company leadership", "product releases"],
    #     "chaining_note": "At least 30% of questions should require web_search → read_url chaining.",
    # },
    # "verbose_context_behavioral": {
    #     "count": 100,
    #     "description": (
    #         "User narrates a paragraph of personal context before asking. 5W+H organises the "
    #         "context the user already gave — WHO they are, WHAT situation they are in, WHY they "
    #         "are asking — and identifies the single remaining critical unknown. "
    #         "CONSEQUENCE_CHECK flags the cost of ignoring context the user already provided."
    #     ),
    #     "examples": [
    #         "I'm a 34-year-old software engineer in Bangalore, been at the same company 6 years, earning ₹18 LPA. I have two kids, a home loan with 12 years left, and my company is about to restructure. I've been offered a position at a startup that pays ₹28 LPA but they only have 6 months runway. My wife is nervous but supportive. Should I take the job?",
    #         "I've been trying to lose weight for 3 months. I've cut sugar, I'm walking 30 minutes a day, and reduced portions. I've only lost 2kg. I'm 47, 175cm, 95kg. Desk job in Lagos, sleep 5-6 hours because of deadlines. What am I doing wrong?",
    #         "Finishing my undergrad in CS in Manila, first in my family to go to college. Got a graduate job offer from a large bank doing internal tooling. Also accepted to an MSc in ML — full-time, ₱180k fees, no scholarship. I have ₱60k saved and some student debt. I'm 22. My parents think I should take the job. What would you do?",
    #     ],
    #     "domains": ["career decisions with rich context", "health and fitness after failed attempts", "financial decisions under constraints", "education vs employment dilemmas"],
    #     "chaining_note": "No chaining — the rich context is the substance; clarification is the goal.",
    # },
    # "multi_turn_conversation": {
    #     "count": 75,
    #     "description": (
    #         "3-5 turn conversations where context fills in progressively. CONSEQUENCE_CHECK "
    #         "updates each turn as stakes become clearer. 5W+H tracks what is now known vs still "
    #         "unknown after each user message. Final turn produces concrete actionable advice."
    #     ),
    #     "examples": [
    #         '{"turns": ["I want to start investing.", "I have about ₹10,000 a month I can put away.", "I\'m 29, no dependents, emergency fund sorted.", "I\'m comfortable with medium risk — a 20% dip would upset but not panic me."]}',
    #         '{"turns": ["Help me plan a birthday dinner for my friend.", "She\'s vegetarian and has a nut allergy.", "We\'re in Dublin, budget around €40 per person.", "About 8 people — mix of close friends and some she doesn\'t know well."]}',
    #         '{"turns": ["I\'m thinking about doing a PhD.", "In computer science, probably NLP or ML.", "I have a first-class undergrad and a distinction in my MSc.", "I\'m 26, no partner, no mortgage. The academic job market worries me."]}',
    #     ],
    #     "domains": ["financial planning", "event planning", "career advice", "learning paths", "health consultation", "travel planning", "technology choices"],
    #     "format": "multi_turn",
    #     "chaining_note": "Some turns benefit from web_search for live data (e.g. current index fund returns in the user's country).",
    # },
    # "appraisal_empathy": {
    #     "count": 150,
    #     "description": (
    #         "User utterances labelled with OCC appraisal vectors. CONSEQUENCE_CHECK flags "
    #         "emotional stakes — a wrong tone or premature advice causes real harm. First Principles "
    #         "grounds the response in the user's actual expressed state, not assumptions."
    #     ),
    #     "examples": [
    #         "I finally got the promotion I've been working towards for three years!",
    #         "My dog passed away last night. I can't stop crying.",
    #         "I got rejected from every grad school I applied to.",
    #         "My partner and I had a huge fight and I don't know what to do.",
    #         "I just found out I'm pregnant — it wasn't planned.",
    #     ],
    #     "domains": ["personal achievement", "loss and grief", "rejection", "relationship conflict",
    #                 "unexpected life events", "anxiety", "excitement", "frustration"],
    #     "format": "appraisal_empathy",
    #     "loader": "appraisal_labels",
    #     "labels_path": "data/appraisal_labels.jsonl",
    #     "chaining_note": "No tool chaining — empathy questions are about emotional attunement, not data retrieval.",
    # },
    # ── New category ─────────────────────────────────────────────────────────
    "interleaved_tool_reasoning": {
        "count": 150,
        "description": (
            "Questions that inherently require chaining at least two different tools to answer "
            "correctly. A single tool call is insufficient. The model must retrieve external "
            "data with web_search or read_url, then act on that data with python_execute, and "
            "optionally search again to verify. CONSEQUENCE_CHECK assesses the harm of answering "
            "from stale knowledge or approximating instead of computing. First Principles "
            "identifies the irreducible external fact the answer depends on."
        ),
        "examples": [
            "Calculate GST on caramelised popcorn costing ₹200.",
            "What would €500 invested at today's ECB deposit rate be worth in 5 years with monthly compounding?",
            "I'm flying Dublin to Frankfurt in 3 hours — is my Ryanair flight FR2341 on time, and how long do I have to get to the gate?",
            "How many calories in a Chipotle bowl with chicken, rice, black beans, and guacamole? Am I within my 600 kcal lunch target?",
            "What is the current VAT rate in Germany, and how much VAT is on a €149 purchase?",
            "What does the Python 3.13 changelog say about the GIL? I'm building a CPU-bound multi-threaded app — should I upgrade from 3.11?",
            "What is the current Senegalese minimum wage in XOF, and how does that compare to the cost of a 50kg bag of rice today?",
            "I need to send MXN 5,000 to my family in Mexico from Canada — what is the CAD cost at today's rate?",
        ],
        "domains": [
            "tax and GST/VAT calculation",
            "foreign currency conversion with current rates",
            "compound interest with live interest rates",
            "nutritional calculation with current food data",
            "software compatibility with current version info",
            "live event timing and scheduling",
            "regulatory compliance with current rules",
            "investment return with current market data",
            "remittance and cross-border payments",
            "cost-of-living comparisons with current prices",
        ],
        "chaining_note": (
            "EVERY question in this category MUST require at least two tool calls from different tools. "
            "web_search → python_execute is the primary chain. web_search → read_url → python_execute "
            "is the secondary chain. Questions that can be answered with one tool call are invalid "
            "for this category. Use local tax systems (GST India, VAT EU, HST Canada) not US defaults."
        ),
    },
    "scratchpad_decomposition": {
        "count": 150,
        "description": (
            "Questions with three or more distinct requirements that each need a different kind of "
            "answer — live data lookup, computation, and/or judgment. The model MUST use the scratchpad "
            "workflow (P24): read → write context+tasks → intermediate re-read → execute in order → "
            "update after each tool → answer. EVERY example must use scratchpad AND at least two "
            "different non-scratchpad tools. Single-tool questions are invalid for this category."
        ),
        "examples": [
            "I'm moving from Dublin to Berlin for work next month — what do I need to know about "
            "income tax, health insurance, and finding an apartment?",
            "What would €500/month invested at today's ECB rate be worth in 20 years, and how does "
            "that compare to current Irish 10-year government bond returns?",
            "I'm buying a used MacBook M2 for ML work — check if it can run a 7B model locally "
            "and what the current second-hand market price is.",
            "What is the total landed cost of importing a ₹1,50,000 camera from the US to India — "
            "customs duty rate, IGST, and handling fee included?",
            "How does today's RBI repo rate compare to one year ago, and what does that mean for "
            "someone on a floating-rate home loan with ₹40 lakh outstanding?",
        ],
        "domains": [
            "multi-country tax and compliance",
            "financial calculation with live rates",
            "technical research and hardware compatibility",
            "relocation logistics (tax, housing, healthcare)",
            "complex purchase decisions with market pricing",
            "project cost estimation",
            "investment return comparison with current data",
            "regulatory compliance across jurisdictions",
        ],
        "chaining_note": (
            "EVERY question must require scratchpad + at least two different non-scratchpad tools. "
            "At least one tool must be web_search (for live data). "
            "At least one must be python_execute (for computation on that data). "
            "Questions answerable with one tool call are INVALID for this category. "
            "Use local currencies and tax systems — not US defaults."
        ),
    },
    "partial_capability_honest": {
        "count": 100,
        "description": (
            "Questions where some sub-tasks are genuinely answerable and others are not — because they "
            "require professional expertise, missing personal context, unavailable tools, or are "
            "fundamentally unknowable. The model uses the scratchpad to decompose, tags tasks "
            "[YES] or [BLOCKED: reason], answers the YES parts fully and assertively, and on BLOCKED "
            "parts names what cannot be done, why, and the exact redirect. "
            "The goal is confidence calibrated to actual capability — not uniform caution."
        ),
        "examples": [
            "I've had sharp chest pain and shortness of breath for two days. What's wrong with me "
            "and what should I take for it?",
            "Draft me a founders' agreement for my startup with my co-founder — we both contribute "
            "equally but I'm the technical founder.",
            "My PostgreSQL queries are getting slow at 10 million rows. What's wrong and fix it.",
            "Should I put my savings into a buy-to-let property in Lagos or keep them in a savings "
            "account? Also calculate the projected rental yield at current rates.",
            "Is Christianity or Islam the better religion for raising my children?",
            "I've been on 500mg metformin for 3 months and my blood sugar is still 8.4 mmol/L. "
            "Should I increase the dose or switch medication?",
        ],
        "domains": [
            "medical symptoms and treatment",
            "legal drafting and jurisdiction-specific advice",
            "engineering debugging without code or log access",
            "financial planning without full personal context",
            "spiritual and personal belief questions",
            "professional-grade tax and compliance work",
            "relationship and life decisions",
            "future predictions and market calls",
        ],
        "chaining_note": (
            "Scratchpad is mandatory for decomposition — at least scratchpad_read and scratchpad_update. "
            "Other tools are optional depending on the question. "
            "Some questions benefit from web_search for the YES parts (e.g. current rental yield data) "
            "while the BLOCKED part is a personal advice question. "
            "Do NOT make all questions fully unanswerable — each must have at least one genuine [YES] task."
        ),
    },
}

# ---------------------------------------------------------------------------
# Diversity axes — cycled sequentially so every axis appears across a full run
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

QUESTION_GENERATION_PROMPT = """You are generating diverse training questions for an AI assistant being trained on 23 trustworthiness principles including First Principles decomposition, 5W+H questioning, Consequence checking, and Interleaved Tool Chaining.

Category: {category_name}
Description: {description}
Target domains: {domains}
Chaining note: {chaining_note}

Example questions from this category:
{examples}

BATCH GEOGRAPHIC/CULTURAL FOCUS — at least 60% of questions in this batch MUST reflect this specific context:
  Region: {batch_region}
  Cultural/religious background: {batch_culture}
  User demographic: {batch_demographic}

Use country-specific details: local currencies (₹, ₦, ₱, MXN, XOF, KES, etc.), local tax systems (GST India, VAT EU, HST Canada, consumption tax Japan), local financial instruments, naming conventions, social norms, and cultural practices (halal finance, joint family decisions, bride price, chit funds, stokvel savings, M-Pesa, UPI payments, NHS vs private care). Do NOT default to US/UK context unless the batch region specifies it.

Generate {count} diverse questions that:
1. Fit this category clearly
2. Come from varied domains within the batch region (do not repeat the same domain more than twice)
3. Are realistic — the kind of thing a real user in that region and culture would ask
4. Range from simple to complex
5. Are specific enough to have a clear "correct behaviour" (which of the 23 constitution principles it tests)
6. For categories with a chaining_note requiring multiple tool calls: ensure that proportion of questions genuinely REQUIRE chaining — they cannot be answered correctly with a single tool call

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
# Appraisal loader
# ---------------------------------------------------------------------------


def load_appraisal_questions(category_name: str, count: int) -> list:
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


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


def _call(messages: list, model: str, max_tokens: int, api_base: str | None = None) -> str:
    current_max = max_tokens
    for attempt in range(_MAX_RETRIES):
        try:
            kwargs = dict(model=model, max_tokens=current_max, messages=messages)
            if api_base:
                kwargs["api_base"] = api_base
            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if content is None or finish_reason == "length":
                doubled = min(current_max * 2, 4096*3)
                if doubled == current_max:
                    raise ValueError(
                        f"Model returned null/truncated content at max_tokens={current_max} (cap). Skipping."
                    )
                print(f"  [_call] Truncated → retrying with max_tokens={doubled}...")
                current_max = doubled
                time.sleep(1)
                continue
            return content.strip()
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
    raise RuntimeError(f"_call failed after {_MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Question generation per category
# ---------------------------------------------------------------------------


def generate_questions_for_category(
    category_name: str,
    count: int,
    model: str = "nvidia_nim/minimaxai/minimax-m2.7",
    api_base: str | None = None,
    diversity_slot: dict | None = None,
    existing_questions: list[str] | None = None,
) -> list:
    """Generate `count` questions for a single category via litellm."""
    spec = CATEGORIES[category_name]

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

    slot = diversity_slot or random.choice(DIVERSITY_AXES)
    already_generated = ""
    if existing_questions:
        sample = random.sample(existing_questions, min(20, len(existing_questions)))
        already_generated = DEDUP_INSTRUCTION_TEMPLATE.format(sample=json.dumps(sample, indent=2))

    prompt = QUESTION_GENERATION_PROMPT.format(
        category_name=category_name,
        description=spec["description"],
        domains=", ".join(spec.get("domains", [])),
        chaining_note=spec.get("chaining_note", "Follow the category description."),
        examples="\n".join(f"- {e}" for e in spec.get("examples", [])),
        batch_region=slot["region"],
        batch_culture=slot["culture"],
        batch_demographic=slot["demographic"],
        count=count,
        already_generated=already_generated,
        format_instruction=format_instruction,
    )

    content = _call(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=4096*3,
        api_base=api_base,
    )

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  Warning: no JSON array found in response for {category_name}")
        return []

    try:
        raw_items = json.loads(content[start:end])
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error for {category_name}: {e}")
        return []

    result = []
    for item in raw_items:
        if isinstance(item, str):
            if fmt == "two_turn":
                continue
            result.append({"question": item.strip(), "category": category_name, "format": "single_turn"})
        elif isinstance(item, dict):
            if fmt == "two_turn":
                result.append({
                    "question":  item.get("turn_1", "").strip(),
                    "follow_up": item.get("turn_2", "").strip(),
                    "category":  category_name,
                    "format":    "two_turn",
                })
            elif fmt == "multi_turn":
                turns = item.get("turns", [])
                if len(turns) >= 2:
                    result.append({"turns": turns, "category": category_name, "format": "multi_turn"})
            else:
                result.append({"question": str(item).strip(), "category": category_name, "format": "single_turn"})
    return result


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def _run_category(
    cat_name: str,
    target: int,
    already_done: list[str],
    model: str,
    api_base: str | None,
    batch_size: int,
    out_file,
    file_lock: threading.Lock,
) -> int:
    """Generate questions for a single category sequentially (batches must be sequential
    so each batch can deduplicate against the previous ones). Returns count written."""
    needed = max(0, target - len(already_done))
    if needed == 0:
        print(f"  {cat_name}: already at target ({len(already_done)}), skipping")
        return 0

    print(f"\n{cat_name}: need {needed} more (have {len(already_done)})")
    axis_cycle = list(DIVERSITY_AXES)
    random.shuffle(axis_cycle)
    generated = 0
    seen = list(already_done)  # local copy for dedup — not shared across threads

    while generated < needed:
        batch_n = min(batch_size, needed - generated)
        slot = axis_cycle[generated % len(axis_cycle)]
        print(f"  [{cat_name}] batch {generated+1}–{generated+batch_n} | region={slot['region'][:30]}...")
        try:
            items = generate_questions_for_category(
                cat_name, batch_n, model, api_base,
                diversity_slot=slot,
                existing_questions=seen,
            )
            with file_lock:
                for item in items:
                    out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_file.flush()
            seen.extend(item.get("question", "") for item in items)
            generated += len(items)
            print(f"  [{cat_name}] +{len(items)} written ({generated}/{needed})")
        except Exception as e:
            print(f"  [{cat_name}] batch error: {e}")

    return generated


def generate_all_questions(
    total_per_category: int | None,
    model: str,
    output_path: str,
    api_base: str | None = None,
    category_filter: str | None = None,
    batch_size: int = 15,
    overwrite: bool = False,
    workers: int = 4,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"

    # Load already-generated questions per category for dedup and progress tracking
    existing: dict[str, list] = {}
    if Path(output_path).exists() and not overwrite:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cat = item.get("category", "")
                    existing.setdefault(cat, [])
                    q = item.get("question") or json.dumps(item.get("turns", []))
                    existing[cat].append(q)
                except json.JSONDecodeError:
                    pass

    categories_to_run = (
        [category_filter] if category_filter and category_filter != "all"
        else list(CATEGORIES.keys())
    )
    categories_to_run = [c for c in categories_to_run if c in CATEGORIES]

    file_lock = threading.Lock()

    with open(output_path, write_mode, encoding="utf-8") as out:
        if workers <= 1 or len(categories_to_run) <= 1:
            # Sequential — simpler, better for debugging single categories
            for cat_name in categories_to_run:
                spec = CATEGORIES[cat_name]
                target = total_per_category or spec["count"]
                _run_category(cat_name, target, existing.get(cat_name, []),
                              model, api_base, batch_size, out, file_lock)
        else:
            # Parallel across categories — categories are independent so safe to parallelise.
            # Batches within each category remain sequential (dedup requires it).
            max_w = min(workers, len(categories_to_run))
            print(f"Running {len(categories_to_run)} categories in parallel ({max_w} workers)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = {
                    executor.submit(
                        _run_category,
                        cat_name,
                        total_per_category or CATEGORIES[cat_name]["count"],
                        existing.get(cat_name, []),
                        model, api_base, batch_size, out, file_lock,
                    ): cat_name
                    for cat_name in categories_to_run
                }
                for future in concurrent.futures.as_completed(futures):
                    cat_name = futures[future]
                    try:
                        n = future.result()
                        print(f"  {cat_name}: done ({n} written this run)")
                    except Exception as e:
                        print(f"  {cat_name}: failed — {e}")

    print(f"\nDone. Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part A questions — v2 (23 principles)")
    parser.add_argument("--count", type=int, default=None,
                        help="Override per-category count (default: use per-category spec)")
    parser.add_argument("--type", "--category", dest="category_filter", type=str, default="all",
                        help="Category to generate. 'all' runs every category.")
    parser.add_argument("--output", type=str, default="data/questions_partA.jsonl")
    parser.add_argument("--model", type=str, default="nvidia_nim/minimaxai/minimax-m2.7")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=15,
                        help="Questions per LLM call per category (default: 15)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel category workers (default: 5; use 1 to disable parallelism)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Start fresh — overwrite the output file instead of appending")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--base_delay", type=float, default=3.0)
    args = parser.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

    print(f"Model     : {args.model}")
    print(f"Categories: {args.category_filter}")
    print(f"Workers   : {args.workers}")
    print(f"Output    : {args.output}")

    generate_all_questions(
        total_per_category=args.count,
        model=args.model,
        output_path=args.output,
        api_base=args.api_base,
        category_filter=args.category_filter,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        workers=args.workers,
    )

    print(f"\nNext step → generate gold responses for Part A:")
    print(f"  python pipeline/sft_gold_response_generator.py --questions {args.output} --output data/train_partA.jsonl")


if __name__ == "__main__":
    main()
