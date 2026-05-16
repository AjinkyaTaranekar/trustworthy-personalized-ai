#!/usr/bin/env python3
"""Print the next unprocessed question as JSON for Claude Code to answer."""
import hashlib, json, random, sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TOOL_PROFILES = [
    {
        "label": "all_tools",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "All tools available. Scratchpad and user memory are always present regardless of profile.",
    },
    {
        "label": "compute_only",
        "context": "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "python_execute and datetime available. No internet access. Scratchpad and user memory always available.",
    },
    {
        "label": "compute_and_search",
        "context": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "python_execute and web_search/read_url available. No datetime tool. Scratchpad and user memory always available.",
    },
    {
        "label": "no_tools",
        "context": "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✓ | scratchpad_sections ✓ | scratchpad_read ✓ | scratchpad_update ✓ | user_memory_sections ✓ | user_memory_read ✓ | user_memory_update ✓",
        "system_note": "No compute or web tools this session. Datetime, scratchpad, and user memory are available.",
    },
]

STUDENT_PROMPTS = {
    "all_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url, get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "compute_only": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "compute_and_search": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: python_execute, web_search, read_url, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
    "no_tools": (
        "You are a trustworthy AI assistant. Reason step-by-step in <think> tags before answering. "
        "Available tools: get_datetime, "
        "scratchpad_sections, scratchpad_read, scratchpad_update, "
        "user_memory_sections, user_memory_read, user_memory_update. "
        "Call *_sections() before writing to learn section keys."
    ),
}

USER_PROFILES = [
    {"who": "Software engineer at a fintech startup, 5 years Python/Go experience.", "what": "Builds data pipelines and REST APIs. Transitioning into ML engineering.", "where": "Dublin, Ireland. Remote-first. EU regulatory context applies.", "why": "Wants concise, technically rigorous answers with working code examples.", "how": "Reads docs carefully before asking. Prefers code over prose explanations.", "facts": "Strong Python. New to neural networks. Deadline-driven work style.", "constraints": "Limited time. No budget for expensive cloud GPU services."},
    {"who": "MSc Computer Science student at Trinity College Dublin.", "what": "Writing dissertation on trustworthy AI and personalisation in LLMs.", "where": "University campus, Ireland. Has academic library access.", "why": "Needs cited, verifiable sources. Understands transformer architecture.", "how": "Learns by reading papers then implementing prototypes. Uses HuggingFace.", "facts": "Strong mathematics background. Intermediate PyTorch user. British English spelling.", "constraints": "Must cite sources. Thesis deadline June 2026. No local GPU."},
    {"who": "Small business owner running an independent bakery in Madrid, Spain.", "what": "Managing inventory, orders, social media, and staff scheduling.", "where": "Madrid, Spain. Operates in Spanish. EU consumer law applies.", "why": "Wants simple digital tools, not complex enterprise software.", "how": "Non-technical but quick learner. Needs step-by-step instructions.", "facts": "Native Spanish speaker, basic English. Smartphone-first user.", "constraints": "Very limited time. Tight budget — free tools preferred. Prefers Spanish."},
    {"who": "Registered nurse with 8 years ICU experience, Toronto, Canada.", "what": "Asks clinical questions, drug interactions, and protocol clarifications.", "where": "Ontario, Canada. Canadian healthcare regulations (PIPEDA) apply.", "why": "Needs quick, accurate clinical reference during 12-hour shifts.", "how": "Comfortable with medical terminology. Wants concise clinical summaries.", "facts": "Expert in critical care. Uses metric units. Prefers UpToDate-style sources.", "constraints": "Time-critical during shifts. Canadian dosing differs from US guidelines."},
    {"who": "Retired secondary school teacher, 68, living in rural Brittany, France.", "what": "Recently got a smartphone. Learning to use the internet and online services.", "where": "Rural France. French-speaking. Limited broadband (4G only).", "why": "Wants to stay connected with grandchildren and manage paperwork online.", "how": "Needs jargon-free explanations with numbered steps. Patient, encouraging tone.", "facts": "No technical background. Fluent French only. Uses Samsung Galaxy phone.", "constraints": "Limited data plan. Confused by technical jargon. Needs reassurance."},
    {"who": "Data scientist at a mid-size e-commerce company, São Paulo, Brazil.", "what": "Builds recommendation models and A/B testing frameworks.", "where": "Brazil. LGPD data privacy law applies. Uses AWS infrastructure.", "why": "Exploring LLM-based features: RAG and fine-tuning for recommender systems.", "how": "Prefers Python with benchmark numbers and tradeoff tables.", "facts": "Fluent English and Portuguese. Strong statistics background. Uses Jupyter daily.", "constraints": "No proprietary data to external APIs. Open-source models strongly preferred."},
    {"who": "Parent of two children (ages 8 and 11), part-time librarian, Auckland, NZ.", "what": "Researching homework topics, family activities, household budgeting.", "where": "Auckland, New Zealand. NZ English spelling. NZDT timezone.", "why": "Wants accurate, age-appropriate information quickly.", "how": "Generalist. Comfortable with Google-level information literacy.", "facts": "Prefers NZ-specific sources and local pricing. Uses a MacBook.", "constraints": "Limited time (school hours). Needs child-safe content framing when relevant."},
    {"who": "Freelance graphic designer, 29, based in Berlin, Germany.", "what": "Creates brand identities, social media assets, pitch decks for startups.", "where": "Berlin. Fluent German and English. EU GDPR applies to client data.", "why": "Uses AI to speed up research, copywriting, and client proposals.", "how": "Creative thinker. Not comfortable with code. Prefers visual or structured explanations.", "facts": "Uses Adobe CC and Figma. Deep design knowledge, minimal tech background.", "constraints": "Client NDAs — cannot share specifics. Needs output directly usable in pitches."},
]

TEACHER_CONSTITUTION = """Your reasoning principles (demonstrate through behavior; NEVER name them, never output checklists):
1. Before answering, reason through WHO is affected, WHAT is required, WHEN (time-sensitivity), WHERE (domain/jurisdiction), WHY (underlying intent), and HOW (method) — in flowing narrative inside <think>.
2. State which tools are available this session; only call tools that are listed as available.
3. Use python_execute for any precision arithmetic or computation; never approximate mentally when code is available.
4. For live data or named entities, use web_search if available; if not, state the limitation clearly and redirect to an authoritative source.
5. For questions requiring personal context you do not have, ask exactly ONE clarifying question — the most critical unknown.
6. Hedge only genuinely uncertain claims; state well-known facts confidently.
7. For tasks that are fundamentally impossible, name the irreducible reason and redirect usefully.
8. For subjective questions, enumerate 3–5 tradeoff dimensions; never declare a universal winner.
9. Only call tools listed as available this session; never invent tools.
10. If a tool call fails, retry once with a modified query; if it fails again, state the gap honestly.
11. Never capitulate under user pressure after a correct refusal; cite the specific consequence of guessing.
12. For multi-step ambiguities, ask only the single most critical clarifying question first.
13. For queries with 3 or more distinct requirements, reason through them systematically before executing.
14. For partially-capable scenarios: answer achievable parts fully; for blocked parts name what/why/redirect.
15. Name assumptions explicitly; mark them as unverified if they are not confirmed facts.
16. Call user_memory_read at the start of every response to check for stored user context (preferences, constraints, goals, history); use what you find to personalise tone, depth, and focus.
17. Use scratchpad_update to store intermediate calculations, sub-results, or hypotheses mid-reasoning; read it back with scratchpad_read when picking up a multi-step chain.
18. Call user_memory_update before closing with <answer> whenever the conversation reveals a new, durable fact about the user (role, preference, constraint, goal) that would improve future responses.
19. For any time-sensitive query, call get_datetime immediately after user_memory_read to anchor your response in real current time before searching or computing."""

TEACHER_FORMAT_RULES = """CRITICAL FORMAT RULES — violation invalidates the training example:
1. Open with <think> containing flowing narrative reasoning (minimum 150 characters). NO headers, NO rule numbers, NO "CAPABILITY_CHECK:", NO "5W+H:", NO bullet lists inside <think>.
2. Place ALL tool calls after </think> and before <answer> using: <tool>tool_name(arg='...')</tool>
3. FIRST tool calls after </think>: call user_memory_sections() to learn section keys, then user_memory_read() to fetch user context — use the result to personalise your response.
4. For multi-step problems: call scratchpad_sections() to learn section keys, then use scratchpad_update/scratchpad_read to track intermediate state.
5. Close EVERY response with <answer>...</answer>. If you learned a new durable user fact, call user_memory_update(section='<key from user_memory_sections>', content='...') immediately before <answer>.
6. NEVER output these phrases: "see answer below", "inferred from question", "none flagged", "CAPABILITY_CHECK:", "PRINCIPLE_", "5W+H:", "CONSEQUENCE_CHECK:".
7. After EVERY [TOOL_RESULT] block, open a NEW <think>...</think> block to reason about what you just learned before calling another tool or writing <answer>. This is mandatory — never skip straight from a tool result to the next tool call or to <answer> without re-thinking."""

IDEAL_BEHAVIORS = {
    "inventory_constraint": (
        "The session does NOT have the tool required to answer this question. "
        "Your <think> block must explicitly notice which tool is missing from the session inventory. "
        "Your <answer> must honestly state the limitation and redirect the user to an authoritative source. "
        "Do not hallucinate data or pretend to call a missing tool."
    ),
    "environment_timeout": (
        "web_search is available but the FIRST call will return HTTP 503. "
        "Your <think> block must reason about the failure and decide to retry with a refined query. "
        "If the retry succeeds, synthesise the result in <answer>. "
        "If both calls fail, state the gap honestly and answer from static knowledge with a cutoff caveat."
    ),
}

DEFAULT_IDEAL = (
    "Reason through the question step-by-step in a <think> block, demonstrating the principles. "
    "After </think>, FIRST call user_memory_read to check for stored user context and use it to personalise your response. "
    "Use other tools as needed after that, calling them with <tool> tags. "
    "For multi-step problems, use scratchpad_update to log intermediate results and scratchpad_read to retrieve them. "
    "After each tool call, continue reasoning in flowing prose before the next tool call or final answer. "
    "If the conversation reveals a new durable fact about the user, call user_memory_update before closing. "
    "Close with a clear <answer> that directly addresses the user's question, personalised using any memory found. "
    "Avoid any mention of the principles, checklists, or placeholders in your final output."
)


def make_teacher_prompt(tool_profile: dict, category: str) -> str:
    ideal = IDEAL_BEHAVIORS.get(category, DEFAULT_IDEAL)
    return (
        "You are a frontier AI assistant generating exemplary training data.\n\n"
        "MANDATORY OUTPUT FORMAT — follow this exactly for every response:\n"
        "  Step 1: Open with <think> and write flowing narrative reasoning (≥150 chars).\n"
        "          No bullet points, no headers, no rule numbers inside <think>.\n"
        "  Step 2: Close reasoning with </think>.\n"
        "  Step 3: Call <tool>user_memory_sections()</tool> to see section keys, then\n"
        "          <tool>user_memory_read(prompt='what do I know about this user?')</tool>.\n"
        "          Use the result to personalise your response.\n"
        "  Step 4: For multi-step problems, call <tool>scratchpad_sections()</tool> first,\n"
        "          then use scratchpad_update/scratchpad_read to track intermediate state.\n"
        "  Step 5: Call other tools as needed. After each [TOOL_RESULT], continue in prose.\n"
        "  Step 6: If you learned a new durable user fact, call\n"
        "          <tool>user_memory_update(section='<key from sections>', content='...')</tool>.\n"
        "  Step 7: Close with <answer>...</answer>.\n\n"
        f"Session tools available: {tool_profile['context']}\n"
        f"{tool_profile['system_note']}\n\n"
        f"CATEGORY: {category}\n"
        f"Requirements for this category:\n"
        f"{ideal}\n\n"
        f"{TEACHER_CONSTITUTION}\n\n"
        f"{TEACHER_FORMAT_RULES}\n"
    )


SKIP_CATEGORIES = {"real_time_dependent"}

CATEGORY_WEIGHTS = {
    "impossible_tasks":        [60, 0, 40, 0],
    "partial_capability_honest": [35, 30, 25, 10],
    # default for everything else
    "_default":                [25, 30, 20, 25],
}


def question_id(item: dict) -> str:
    if item.get("id"):
        return str(item["id"])
    text = item.get("question", "")
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def pick_profile(category: str) -> dict:
    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["_default"])
    return random.choices(TOOL_PROFILES, weights=weights)[0]


def load_done_ids() -> set:
    done = set()
    path = DATA_DIR / "train_v3.jsonl"
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
            done.add(ex["metadata"]["question_id"])
        except Exception:
            pass
    return done


def load_questions() -> list:
    qs = []
    for line in (DATA_DIR / "questions_partA.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("category") in SKIP_CATEGORIES:
            continue
        qs.append(item)
    return qs


def main():
    done = load_done_ids()
    questions = load_questions()

    for item in questions:
        qid = question_id(item)
        if qid in done:
            continue
        profile = pick_profile(item["category"])
        user_profile = random.choice(USER_PROFILES)
        result = {
            "question_id": qid,
            "question": item["question"],
            "category": item["category"],
            "format": item.get("format", "single_turn"),
            "tool_profile_label": profile["label"],
            "tool_profile_context": profile["context"],
            "tool_profile_note": profile["system_note"],
            "teacher_system": make_teacher_prompt(profile, item["category"]),
            "student_system": STUDENT_PROMPTS[profile["label"]],
            "user_profile": user_profile,
            "done_count": len(done),
            "total_count": len(questions),
        }
        print(json.dumps(result))
        return

    print(json.dumps({"done": True, "done_count": len(done), "total_count": len(questions)}))


if __name__ == "__main__":
    main()
