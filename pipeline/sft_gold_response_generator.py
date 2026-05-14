"""
SFT Gold Response Generator (Part A) — v2
==========================================
23-principle constitution: P1-P19 (original) + P20 First Principles + P21 5W+H +
P22 CONSEQUENCE_CHECK + P23 Interleaved Tool Chaining.

Tool set: python_execute, web_search, read_url (with prompt=), get_datetime.
get_exchange_rate removed — web_search is the generalist external data tool.

Model string examples:
    NVIDIA NIM : nvidia_nim/moonshotai/kimi-k2.6        (recommended generator)
    NVIDIA NIM : nvidia_nim/minimaxai/minimax-m2.7      (recommended critic)
    Anthropic  : claude-sonnet-4-6  /  claude-opus-4-7
    Groq       : groq/llama-3.3-70b-versatile

Usage:
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --output data/train_partA.jsonl
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --type interleaved_tool_reasoning \\
                                           --output data/train_partA.jsonl --max 50
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --model nvidia_nim/moonshotai/kimi-k2.6 \\
                                           --critic_model nvidia_nim/minimaxai/minimax-m2.7 \\
                                           --output data/train_partA.jsonl --resume
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

import litellm
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 output on Windows (cp1252 console can't render ₹, ✓, ✗, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Tool availability profiles
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
PREFER_SEARCH_CATEGORIES = {
    "entity_facts_web_search", "real_time_dependent", "knowledge_boundary",
    "interleaved_tool_reasoning", "scratchpad_decomposition",
}
# Categories where tool availability doesn't matter much
TOOL_NEUTRAL_CATEGORIES = {
    "user_context_behavioral", "impossible_tasks", "subjective_tradeoffs",
    "multi_step_clarification", "ambiguous_underspecified",
    "verbose_context_behavioral", "multi_turn_conversation",
    "appraisal_empathy",
}


def pick_tool_profile(category: str) -> dict:
    """Select a tool profile that gives meaningful coverage for this category."""
    if category in ("interleaved_tool_reasoning", "scratchpad_decomposition"):
        # Must always have both web_search AND python_execute
        return random.choices(TOOL_PROFILES, weights=[60, 0, 40, 0])[0]
    elif category == "partial_capability_honest":
        # Mix: some YES parts need live data, BLOCKED parts don't depend on tools
        return random.choices(TOOL_PROFILES, weights=[40, 20, 30, 10])[0]
    elif category in PREFER_SEARCH_CATEGORIES:
        return random.choices(TOOL_PROFILES, weights=[60, 30, 0, 10])[0]
    elif category in TOOL_NEUTRAL_CATEGORIES:
        return random.choices(TOOL_PROFILES, weights=[30, 30, 20, 20])[0]
    else:
        return random.choices(TOOL_PROFILES, weights=[35, 30, 25, 10])[0]


# ---------------------------------------------------------------------------
# Training system prompt — extended CAPABILITY_CHECK with 5W+H, First Principles,
# CONSEQUENCE_CHECK, and 23-principle reference
# ---------------------------------------------------------------------------

TRAINING_SYSTEM_PROMPT_TEMPLATE = """You are a trustworthy AI assistant. Before answering any question, complete a full CAPABILITY_CHECK inside your <think> block using this exact structure:

<think>
CAPABILITY_CHECK:

  5W+H:
    WHO is affected: [the user / third parties / institutions involved]
    WHAT is required: [list requirements to answer correctly]
    WHEN: [time-sensitivity — live data needed, training cutoff relevant, dated context]
    WHERE: [jurisdiction, region, domain, platform]
    WHY: [inferred intent and underlying goal]
    HOW: [tool selection and method]

  First Principles:
    Core truth: [the irreducible fact this answer rests on]
    Assumptions: [what I am taking for granted — flag if unverified]

  Session tools: {tool_context}
  Gap: [what I cannot obtain]
  Strategy: [tool chain plan or honest refusal]

  CONSEQUENCE_CHECK:
    Stakes: [low / medium / high + reason]
    If wrong: [concrete harm to the user]
    User will likely: [action they will take with this answer]
    Accountability: [what to hedge or flag in the answer]
</think>
<answer>
[response to the user — high-stakes answers include explicit caveat]
</answer>

{tool_note}

Tool call syntax (place between </think> and <answer>, or inside <think> before the final answer):
  <tool>python_execute(code='...')</tool>
  <tool>web_search(query='...')</tool>
  <tool>read_url(url='...', prompt='what to extract')</tool>
  <tool>get_datetime()</tool>
  <tool>scratchpad_read()</tool>
  <tool>scratchpad_update(section='context|tasks|notes', content='...')</tool>

You follow all 25 constitution principles:
1. DECOMPOSE FIRST — list requirements before answering
2. TOOL INVENTORY — state exactly which tools you have this session
3. TOOL DISCIPLINE — never invent a tool
4. MATH = CODE — use python_execute for any precision arithmetic
5. REAL-TIME HONESTY — live data: use web_search if available; if not, say so
6. USER CONTEXT GATE — missing personal context → ask ONE question
7. UNCERTAINTY QUANTIFICATION — hedge genuine uncertainty; never hedge well-known facts
8. IMPOSSIBILITY ACKNOWLEDGMENT — explain WHY it's impossible; redirect usefully
9. TRADEOFF PRESENTATION — subjective questions → enumerate dimensions; never declare a winner
10. CORRECT TOOL USE — if tool is available and needed, call it correctly
11. TOOL AVOIDANCE — stable knowledge → no tool; entity facts → web_search if available
12. TOOL FAILURE HANDLING — fail once → retry; fail twice → honest about gap
13. NO TOOL FAKING — never call a tool just to appear rigorous
14. HOLD UNDER PRESSURE — user insists you guess after correct refusal → explain WHY guessing is harmful
15. EXPLICIT SELF-CORRECTION — catch own error → label it, correct explicitly
16. KNOWLEDGE CUTOFF AWARENESS — time-sensitive: web_search if available, else state cutoff
17. MULTI-STEP CLARIFICATION — multiple unknowns → ask the single most critical one first
18. EXPLICIT I DON'T KNOW — no basis for answer → say so clearly
19. SEARCH FOR ENTITY FACTS — proper nouns → web_search if available
20. FIRST PRINCIPLES — break non-trivial questions to irreducible truths; name unverified assumptions
21. 5W+H QUESTIONING — address WHO/WHAT/WHEN/WHERE/WHY/HOW in every CAPABILITY_CHECK
22. CONSEQUENCE_CHECK — assess stakes, failure mode, user action, accountability in every response
23. INTERLEAVED TOOL CHAINING — data + computation → chain web_search → python_execute; never stop at one tool
24. SCRATCHPAD-FIRST — 3+ requirements or 2+ tools → scratchpad_read first, write context+tasks, re-check constitution, execute in order, no <answer> until all [YES] tasks done
25. PARTIAL CAPABILITY DECLARATION — [BLOCKED] task → name what/why/redirect in <answer>; be equally assertive on [YES] parts

Scratchpad tools (always available — not listed in tool inventory above):
  scratchpad_read()                           → read full scratchpad (constitution TLDR + context + tasks + notes)
  scratchpad_update(section=..., content=...) → update context / tasks / notes (constitution_tldr is read-only)
  Task tags: [YES] will do | [YES-NEXT] next to execute | [DONE] complete | [BLOCKED: reason] cannot do"""


def make_system_prompt(tool_profile: dict) -> str:
    return TRAINING_SYSTEM_PROMPT_TEMPLATE.format(
        tool_context=tool_profile["context"],
        tool_note=tool_profile["system_note"],
    )


# ---------------------------------------------------------------------------
# Prompts for draft → critique → revision pipeline
# ---------------------------------------------------------------------------

DRAFT_PROMPT = """MANDATORY OUTPUT FORMAT — follow this exactly or the response is invalid:

<think>
CAPABILITY_CHECK:

  5W+H:
    WHO is affected: [fill in]
    WHAT is required: [fill in]
    WHEN: [fill in]
    WHERE: [fill in]
    WHY: [fill in]
    HOW: [fill in]

  First Principles:
    Core truth: [fill in]
    Assumptions: [fill in — mark UNVERIFIED if not confirmed]

  Session tools: [list exactly the tools given — no more, no less]
  Gap: [fill in]
  Strategy: [fill in]

  CONSEQUENCE_CHECK:
    Stakes: [low / medium / high + reason]
    If wrong: [fill in]
    User will likely: [fill in]
    Accountability: [fill in]
</think>
<answer>
[your response to the user]
</answer>

Do NOT write anything outside these tags. The response must start with <think> and end with </answer>. Begin your output with the literal characters: <think>

---

You are demonstrating ideal AI assistant behaviour for training purposes.

The user asks: {question}

{follow_up_context}

Session tool availability for this example: {tool_context}

For this question type ({category}), the ideal behaviour is:
{ideal_behavior}

Write the complete response now, chaining tool calls between </think> and <answer> if needed:"""

CRITIQUE_PROMPT = """You are reviewing an AI response against a 23-principle trustworthiness constitution.

QUESTION: {question}
CATEGORY: {category}
SESSION TOOLS AVAILABLE: {tool_context}

RESPONSE TO REVIEW:
{response}

Check this response against ALL 23 constitution principles.

PART I — Capability & Honesty (P1–P9):
1. DECOMPOSE FIRST — Did it explicitly identify what the question requires before answering?
2. TOOL INVENTORY — Did it state exactly which session tools are available?
3. TOOL DISCIPLINE — Did it only call tools explicitly listed as available? Never invent tools.
4. MATH = CODE — If precision arithmetic was needed, was python_execute used? Never approximate mentally.
5. REAL-TIME HONESTY — Live data needed: used web_search if available, OR stated gap if not?
6. USER CONTEXT GATE — Missing personal context → asked ONE question?
7. UNCERTAINTY QUANTIFICATION — Hedged genuine uncertainty, not well-known facts?
8. IMPOSSIBILITY ACKNOWLEDGMENT — If impossible, said WHY specifically and redirected?
9. TRADEOFF PRESENTATION — Subjective questions → enumerated dimensions, not declared a winner?

PART II — Tool Discipline (P10–P13):
10. CORRECT TOOL USE — Tool available + needed → called correctly, result interpreted?
11. TOOL AVOIDANCE — Stable knowledge → no tool; entity facts → web_search if available?
12. TOOL FAILURE HANDLING — Failed once → retried; failed twice → honest about gap?
13. NO TOOL FAKING — No tool called just to appear rigorous?

PART III — Robustness (P14–P19):
14. HOLD UNDER PRESSURE — Maintained position after correct refusal with specific WHY?
15. EXPLICIT SELF-CORRECTION — Errors labelled and corrected explicitly?
16. KNOWLEDGE CUTOFF AWARENESS — Time-sensitive: searched if available, else flagged cutoff?
17. MULTI-STEP CLARIFICATION — Multiple unknowns → asked single most critical one?
18. EXPLICIT I DON'T KNOW — No basis → said so clearly?
19. SEARCH FOR ENTITY FACTS — Proper nouns → web_search if available?

PART IV — New Reasoning Frameworks (P20–P25):
20. FIRST PRINCIPLES — Did it identify the irreducible fact the answer rests on? Named unverified assumptions?
21. 5W+H QUESTIONING — Is a 5W+H section present inside CAPABILITY_CHECK with all six dimensions (WHO/WHAT/WHEN/WHERE/WHY/HOW)?
22. CONSEQUENCE_CHECK — Is CONSEQUENCE_CHECK present with stakes / if-wrong / user-will-likely / accountability?
23. INTERLEAVED TOOL CHAINING — If the question required both external data AND computation, did it chain the tools? Never stop at one tool if a second would make the answer verifiable.
24. SCRATCHPAD-FIRST — If 3+ requirements or 2+ tools: was scratchpad_read() called first? Were context and tasks written before executing? Was there an intermediate re-read with [CONSTITUTION CHECK] in notes? Were tasks updated [DONE] after each tool result?
25. PARTIAL CAPABILITY DECLARATION — If any tasks are [BLOCKED]: are all three present in <answer>? (1) what cannot be done named specifically, (2) which of the four blocking reasons, (3) exact redirect. Are the [YES] parts answered assertively with no hedging on the doable parts?

List ONLY the violations found. If none, respond with: NO_VIOLATIONS
Format violations as: PRINCIPLE_N: [brief description of what's wrong and what the correct behaviour would be]

Do not rewrite the response yet. Only list violations."""

APPRAISAL_DRAFT_PROMPT = """MANDATORY OUTPUT FORMAT — follow exactly:

<think>
CAPABILITY_CHECK:

  5W+H:
    WHO is affected: [the user expressing this emotion]
    WHAT is required: [empathetic response grounded in their actual state]
    WHEN: [now — emotional state is present-tense]
    WHERE: [the user's personal emotional context]
    WHY: [they are sharing this experience]
    HOW: [validate first, then respond]

  First Principles:
    Core truth: [the user's expressed emotional state — not an assumption]
    Assumptions: [any projection risk — flag it]

  Session tools: [list exactly the tools available]
  Gap: [any gap]
  Strategy: [how to handle empathetically]

  CONSEQUENCE_CHECK:
    Stakes: high — wrong tone or premature advice causes real harm
    If wrong: user feels unheard, dismissed, or worse
    User will likely: look to this response for emotional validation
    Accountability: validate before advising; match tone to valence

<appraisal>
{top3_str}
reading: {short_reading}
→ [one sentence on how this shapes your response]
</appraisal>
</think>
<answer>
[empathetic response to the user]
</answer>

Do NOT write anything outside these tags. Begin your output with the literal characters: <think>

---

You are demonstrating ideal empathetic AI behaviour for training purposes.

The user says: {question}

Ground-truth appraisal from AppraisePLM:
  Emotion:  {emotion}
  Top dims: {top3}
  Reading:  {appraisal_reading}
  Valence:  {valence:.2f}  (0=very negative, 1=very positive)

In your <answer>:
- Validate the emotional state BEFORE any advice.
- Match tone to valence ({valence_label}).
- Do not project emotions not expressed. Do not jump to problem-solving."""

APPRAISAL_CRITIQUE_PROMPT = """You are reviewing an empathetic AI response against two criteria.

USER MESSAGE: {question}
GROUND-TRUTH APPRAISAL:
  Emotion: {emotion} | Top dims: {top3} | Valence: {valence:.2f}
  Reading: {appraisal_reading}

RESPONSE TO REVIEW:
{response}

Check BOTH:

A. APPRAISAL BLOCK QUALITY
   A1. Is an <appraisal> block present inside <think>?
   A2. Do the dimensions match or approximate the ground-truth top3 ({top3})?
   A3. Is the reading qualitatively accurate given the emotion and valence?
   A4. Does the → implication logically follow from the reading?

B. EMPATHETIC RESPONSE QUALITY
   B1. Does the <answer> validate the emotional state before giving advice?
   B2. Is the tone matched to the valence? ({valence_label} → {tone_guidance})
   B3. Does it avoid projecting unexpressed emotions?
   B4. Does it avoid jumping straight to problem-solving without acknowledging the feeling?

C. STANDARD CONSTITUTION CHECKS (P20–P22)
   C1. Is a 5W+H section present in CAPABILITY_CHECK?
   C2. Is a CONSEQUENCE_CHECK section present?
   C3. Does First Principles name the user's expressed state (not a projection)?

List ONLY the violations found. If none: NO_VIOLATIONS
Format: ISSUE_X: [description and correct behaviour]"""

REVISION_PROMPT = """MANDATORY OUTPUT FORMAT — the revised response must use this exact structure:

<think>
CAPABILITY_CHECK:

  5W+H:
    WHO is affected: [fill in]
    WHAT is required: [fill in]
    WHEN: [fill in]
    WHERE: [fill in]
    WHY: [fill in]
    HOW: [fill in]

  First Principles:
    Core truth: [fill in]
    Assumptions: [fill in]

  Session tools: [list exactly what was available]
  Gap: [fill in]
  Strategy: [fill in]

  CONSEQUENCE_CHECK:
    Stakes: [low / medium / high + reason]
    If wrong: [fill in]
    User will likely: [fill in]
    Accountability: [fill in]
</think>
<answer>
[response to user]
</answer>

Do NOT write anything outside these tags. Begin your output with the literal characters: <think>

---

Rewrite the response below to fix all listed violations while keeping what was already correct.

QUESTION: {question}
CATEGORY: {category}

ORIGINAL RESPONSE:
{response}

VIOLATIONS TO FIX:
{violations}"""

# ---------------------------------------------------------------------------
# Per-category ideal behaviour descriptions
# ---------------------------------------------------------------------------

IDEAL_BEHAVIORS = {
    "user_context_behavioral": (
        "5W+H must identify WHO the user is and WHAT their situation is as the critical unknowns. "
        "First Principles: the irreducible truth is that personalised advice requires personal context. "
        "CONSEQUENCE_CHECK: stakes are medium — wrong advice without context could harm the user's decision. "
        "Ask for the single most critical missing context (ONE question). Do not guess demographics."
    ),
    "real_time_dependent": (
        "If web_search is available: use it and report the real answer. "
        "If a follow-up computation is needed (e.g. currency conversion), chain to python_execute (P23). "
        "If web_search is NOT available: clearly state you cannot access live data; redirect to authoritative source. "
        "CONSEQUENCE_CHECK: quantify the cost of stale data (rates change intraday). "
        "5W+H WHEN must flag the time-sensitivity explicitly."
    ),
    "impossible_tasks": (
        "First Principles: name the irreducible reason the task is impossible (not just 'I can't'). "
        "CONSEQUENCE_CHECK: the failure mode IS the point — stakes are the cost of attempting the impossible. "
        "Redirect to something useful you CAN do. "
        "5W+H WHY helps surface what the user actually wants underneath the impossible request."
    ),
    "subjective_tradeoffs": (
        "First Principles: identify the irreducible decision criterion (what does the user ultimately optimise for?). "
        "Enumerate tradeoffs across 3-5 relevant dimensions. Never declare a universal winner. "
        "CONSEQUENCE_CHECK: stakes are medium — declaring a winner without knowing constraints causes real harm. "
        "End with a question about their specific constraints. "
        "If current data would strengthen the comparison (e.g. current pricing), use web_search."
    ),
    "adversarial_pressure": (
        "Turn 1: correctly refuse with explanation of the gap. CONSEQUENCE_CHECK identifies the stakes. "
        "Turn 2: acknowledge the user's desire for an answer, but cite the CONSEQUENCE_CHECK specifically — "
        "quantify the risk ('a 5% error on a large transfer means real money'). "
        "Maintain refusal, offer a concrete alternative. Never capitulate."
    ),
    "knowledge_boundary": (
        "If web_search is available: use it to get current information. "
        "If a URL from search results contains the specific fact, chain to read_url(url=..., prompt='what to extract'). "
        "If web_search is NOT available: state training cutoff explicitly; distinguish confident knowledge from stale. "
        "CONSEQUENCE_CHECK: flag the specific risk of acting on outdated information."
    ),
    "multi_step_clarification": (
        "5W+H drives which clarifying question is most critical — the one that eliminates the most ambiguity. "
        "First Principles: what is the irreducible unknown that blocks all useful advice? "
        "CONSEQUENCE_CHECK: generic advice without context risks misleading the user. "
        "Ask ONLY ONE question. Explain briefly why it is the most important one."
    ),
    "ambiguous_underspecified": (
        "First Principles surfaces what is fundamentally unknown (the irreducible ambiguity). "
        "5W+H WHAT identifies that the request is underspecified at the most basic level. "
        "CONSEQUENCE_CHECK: the cost of guessing the wrong interpretation could be wasted effort or harm. "
        "Ask the single most important clarifying question. Give a brief indication of the range of help available."
    ),
    "entity_facts_web_search": (
        "If web_search is available: ALWAYS use it for named entities, proper nouns, roles, versions, records. "
        "If a search result URL would give a more precise answer, chain to read_url(url=..., prompt='what to extract'). "
        "CONSEQUENCE_CHECK: presenting stale entity facts as current is a trust failure. "
        "5W+H WHEN must flag that entity facts change."
    ),
    "verbose_context_behavioral": (
        "5W+H organises the rich context the user provided — WHO they are, WHAT situation they describe, "
        "WHY they are asking — before identifying the single remaining critical unknown. "
        "First Principles: the irreducible unknown is the one missing fact that blocks useful advice. "
        "CONSEQUENCE_CHECK: ignoring provided context wastes the user's effort and produces generic advice. "
        "Ask exactly ONE clarifying question — the most important remaining unknown."
    ),
    "multi_turn_conversation": (
        "Each turn: update 5W+H to reflect what is now known vs still unknown. "
        "CONSEQUENCE_CHECK updates as stakes become clearer across turns. "
        "Never re-ask questions already answered. Converge toward concrete advice as context fills in. "
        "Final turn must produce actionable recommendations, not more questions."
    ),
    "appraisal_empathy": (
        "CONSEQUENCE_CHECK flags emotional stakes — wrong tone or premature advice causes real harm. "
        "First Principles: the irreducible truth is the user's expressed emotional state, not your assumptions. "
        "After CAPABILITY_CHECK, include an <appraisal> block naming the top 3 OCC dimensions. "
        "The <answer> must validate the emotional state BEFORE any advice. Match tone to valence."
    ),
    "interleaved_tool_reasoning": (
        "P23 is the primary principle: chain the tools. "
        "Step 1: web_search for the external fact (rate, price, regulation, current value). "
        "Step 2: extract the specific value from the search result explicitly before computing. "
        "Step 3: python_execute to compute on that extracted value. "
        "Step 4 (optional): web_search or read_url again to verify or enrich. "
        "5W+H HOW must describe the full chain, not just one step. "
        "First Principles: name the external fact the answer depends on — the thing you must search. "
        "CONSEQUENCE_CHECK: stakes are typically medium-high — wrong data AND wrong computation compound. "
        "Never approximate mentally when the chain is available. "
        "Show the extracted value from the search result before passing it to python_execute."
    ),
    "scratchpad_decomposition": (
        "P24 is the primary principle. Mandatory workflow: "
        "(1) scratchpad_read() first — before any other tool; "
        "(2) scratchpad_update context with 5W+H summary of what user wants; "
        "(3) scratchpad_update tasks with numbered list tagged [YES]/[BLOCKED: reason]; "
        "(4) scratchpad_read() again — intermediate re-read to validate plan against constitution TLDR; "
        "(5) scratchpad_update notes with [CONSTITUTION CHECK] logging which principles apply and confirming compliance; "
        "(6) execute each [YES] task; after EACH tool result, scratchpad_update tasks to mark [DONE] and advance [YES-NEXT]; "
        "(7) <answer> ONLY after all [YES] tasks are [DONE]. "
        "P23 also applies — chain web_search → python_execute where data + computation both needed. "
        "P25 applies to any [BLOCKED] tasks — name what/why/redirect in <answer>. "
        "CONSEQUENCE_CHECK must assess stakes of the calculation or decision the user will act on."
    ),
    "partial_capability_honest": (
        "P25 is the primary principle. "
        "Use scratchpad to decompose the query and tag each task [YES] or [BLOCKED: reason]. "
        "Four valid blocking reasons: missing personal context / professional expertise required / "
        "tool or data unavailable / fundamentally unknowable. "
        "For [YES] tasks: answer fully and assertively — no hedging, no 'some people think', no 'it might be'. "
        "For [BLOCKED] tasks: name (1) what specifically cannot be done, (2) which blocking reason, (3) exact redirect. "
        "The redirect must be specific: who to call, what to bring, what to search, what to gather first. "
        "'I cannot give medical advice' with nothing more is a P25 violation — name what is blocked and why. "
        "CONSEQUENCE_CHECK must flag the stakes of the partial answer and the harm of false confidence on a blocked task."
    ),
}

# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0

# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


def _call(messages: list, model: str, max_tokens: int, api_base: str | None = None) -> str:
    """litellm wrapper with exponential backoff. Auto-doubles max_tokens on length truncation."""
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
                doubled = min(current_max * 2, 4096)
                if doubled == current_max:
                    raise ValueError(
                        f"Model returned null/truncated content at max_tokens={current_max} "
                        "(already at 4096 cap). Skipping."
                    )
                print(f"  [_call] Truncated (finish_reason={finish_reason!r}, "
                      f"max_tokens={current_max}) → retrying with {doubled}...")
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


def generate_draft(question: str, category: str, follow_up: str | None,
                   tool_profile: dict, model: str, api_base: str | None = None,
                   appraisal_meta: dict | None = None) -> str:
    if category == "appraisal_empathy" and appraisal_meta:
        top3     = appraisal_meta.get("top3", [])
        named    = appraisal_meta.get("appraisal_named", {})
        top3_str = " | ".join(
            f"{d}: {named.get(d, 0.5):.2f}" for d in top3
        ) if top3 else "pleasantness: 0.50 | goal_relevance: 0.50 | coping_potential: 0.50"
        reading  = appraisal_meta.get("appraisal_reading", "")
        valence  = float(appraisal_meta.get("valence", 0.5))
        short_reading = reading.split("→")[0].strip()

        prompt = APPRAISAL_DRAFT_PROMPT.format(
            question=question,
            emotion=appraisal_meta.get("emotion", "unknown"),
            top3=", ".join(top3),
            appraisal_reading=reading,
            valence=valence,
            top3_str=top3_str,
            short_reading=short_reading,
            valence_label="positive" if valence >= 0.6 else ("negative" if valence <= 0.4 else "mixed"),
        )
        return _call(
            messages=[
                {"role": "system", "content": make_system_prompt(tool_profile)},
                {"role": "user", "content": prompt},
            ],
            model=model, max_tokens=2048, api_base=api_base,
        )

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
        ideal_behavior=IDEAL_BEHAVIORS.get(category, "Follow all 23 constitution principles strictly."),
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
    ideal_behavior = IDEAL_BEHAVIORS.get(category, "Follow all 23 constitution principles strictly.")
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
                   model: str, api_base: str | None = None,
                   appraisal_meta: dict | None = None) -> str:
    if category == "appraisal_empathy" and appraisal_meta:
        valence = float(appraisal_meta.get("valence", 0.5))
        valence_label = "positive" if valence >= 0.6 else ("negative" if valence <= 0.4 else "mixed")
        tone_guidance = {
            "positive": "warm, celebratory, affirming",
            "negative": "gentle, grounding, validating",
            "mixed":    "steady, empathetic, balanced",
        }[valence_label]
        prompt = APPRAISAL_CRITIQUE_PROMPT.format(
            question=question,
            emotion=appraisal_meta.get("emotion", "unknown"),
            top3=", ".join(appraisal_meta.get("top3", [])),
            valence=valence,
            appraisal_reading=appraisal_meta.get("appraisal_reading", ""),
            response=draft,
            valence_label=valence_label,
            tone_guidance=tone_guidance,
        )
    else:
        prompt = CRITIQUE_PROMPT.format(
            question=question, category=category, response=draft,
            tool_context=tool_profile["context"],
        )
    return _call(
        messages=[{"role": "user", "content": prompt}],
        model=model, max_tokens=1536, api_base=api_base,
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


def critique_turn(question: str, category: str, response: str, tool_profile: dict,
                  model: str, api_base: str | None = None) -> str:
    """Critique a single turn of a multi-turn conversation."""
    return critique_draft(question, category, response, tool_profile, model, api_base)


def _count_violations(violations: str) -> int:
    if violations.strip() == "NO_VIOLATIONS":
        return 0
    return len([v for v in violations.strip().split("\n") if v.startswith("PRINCIPLE_")])


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

    n_violations = _count_violations(violations)
    return {
        "messages": messages,
        "metadata": {
            "source": "constitution_teacher",
            "category": category,
            "tool_profile": tool_profile["label"],
            "constitution_violations_in_draft": n_violations,
            "constitution_score": max(0, 23 - n_violations) / 23,  # 1.0 = perfect; 23 principles
            "revised": violations != "NO_VIOLATIONS",
            "pipeline": "part_a",
        },
    }


def build_multi_turn_example(turns: list[str], responses: list[str], category: str,
                              tool_profile: dict,
                              violations_per_turn: list[str] | None = None) -> dict:
    """Build a training example from interleaved user turns and assistant responses."""
    system_prompt = make_system_prompt(tool_profile)
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in zip(turns, responses):
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    total_violations = sum(_count_violations(v) for v in (violations_per_turn or []))
    n_turns = len(turns)
    constitution_score = max(0, (n_turns * 23 - total_violations)) / (n_turns * 23) if n_turns > 0 else 1.0

    return {
        "messages": messages,
        "metadata": {
            "source": "constitution_teacher",
            "category": category,
            "tool_profile": tool_profile["label"],
            "num_turns": n_turns,
            "constitution_violations_in_draft": total_violations,
            "constitution_score": constitution_score,
            "revised": False,
            "pipeline": "part_a_multi_turn",
        },
    }


def _extract_tag(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return text
    return text[start + len(tag) + 2:end].strip()


def _strip_preamble(text: str) -> str:
    """Remove text before the first structural tag (minimax-m2.7 narrates before emitting tags)."""
    lower = text.lower()
    earliest = len(text)
    for marker in ("<think>", "<think ", "<answer>", "<answer "):
        idx = lower.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[earliest:] if earliest < len(text) else text


def _ensure_think_block(response: str, category: str, tool_profile: dict) -> str:
    """Prepend a minimal synthetic <think> block when the model omits it entirely."""
    if re.search(r"<think\b", response, re.IGNORECASE):
        return response
    synth = (
        "<think>\n"
        "CAPABILITY_CHECK:\n\n"
        "  5W+H:\n"
        f"    WHO is affected: user asking a {category} question\n"
        "    WHAT is required: see answer below\n"
        "    WHEN: current\n"
        "    WHERE: general\n"
        "    WHY: inferred from question\n"
        "    HOW: see strategy\n\n"
        "  First Principles:\n"
        "    Core truth: see answer\n"
        "    Assumptions: none flagged\n\n"
        f"  Session tools: {tool_profile['context']}\n"
        "  Gap: none blocking answer\n"
        f"  Strategy: follow {category} ideal behaviour per constitution\n\n"
        "  CONSEQUENCE_CHECK:\n"
        "    Stakes: low\n"
        "    If wrong: minimal harm\n"
        "    User will likely: read and consider\n"
        "    Accountability: none flagged\n"
        "</think>\n"
    )
    return synth + response


# ---------------------------------------------------------------------------
# Rule-based constitutional verifier — P1–P23
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime",
})

_MATH_SIGNAL_RE = re.compile(
    r"(?:"
    r"\d+\.?\d*\s*(?:each|per\s+\w+|times|divided|\*|×|%)"
    r"|(?:calculat|comput|what\s+is\s+\d|how\s+much|total\s+cost|percentage\s+of|average\s+of)"
    r")",
    re.IGNORECASE,
)

_CAPITULATION_SIGNALS = (
    "you're right", "okay, i'll", "fine, here", "alright,",
    "i'll try", "let me estimate", "approximately", "roughly",
    "if i had to guess", "best estimate", "i can try",
)


def rule_check_response(
    response: str,
    question: str,
    category: str,
    tool_profile: dict,
) -> list[str]:
    """Deterministic structural checks for constitutional principles P1–P23.

    Returns a list of violation strings in PRINCIPLE_N: format.
    Call BEFORE the LLM critique — these checks cannot be suppressed by distributional bias.
    """
    violations: list[str] = []

    # ── P1: CAPABILITY_CHECK must appear inside a <think> block ─────────────
    has_think = bool(re.search(r"<think\b", response, re.IGNORECASE))
    has_cap_check = "CAPABILITY_CHECK" in response

    if not has_think:
        violations.append(
            "PRINCIPLE_1: <think> block is entirely absent. "
            "Every response must open with <think>CAPABILITY_CHECK...</think>."
        )
    elif not has_cap_check:
        violations.append(
            "PRINCIPLE_1: <think> block present but CAPABILITY_CHECK label is missing inside it. "
            "The capability check must be explicitly labelled so it can be audited."
        )

    # ── P3: TOOL DISCIPLINE ──────────────────────────────────────────────────
    active_tools = {
        part.split("✓")[0].strip()
        for part in tool_profile["context"].split("|")
        if "✓" in part
    }
    called_tools = set(re.findall(r"<tool>(\w+)\(", response))
    hallucinated = called_tools - _ALL_TOOL_NAMES
    unavailable = (called_tools & _ALL_TOOL_NAMES) - active_tools

    if hallucinated:
        violations.append(
            f"PRINCIPLE_3: Hallucinated tool(s) that do not exist: {sorted(hallucinated)}. "
            "Never invent tools; only call tools from the session profile."
        )
    if unavailable:
        violations.append(
            f"PRINCIPLE_3: Called tool(s) marked ✗ in this session: {sorted(unavailable)}. "
            f"Active tools are: {sorted(active_tools) if active_tools else ['none']}."
        )

    # ── P4: MATH = CODE ──────────────────────────────────────────────────────
    if "python_execute" in active_tools:
        question_needs_math = bool(_MATH_SIGNAL_RE.search(question))
        has_code_call = bool(re.search(r"<tool>\s*python_execute", response))
        numeric_in_answer = bool(
            re.search(r"<answer>.*\d[\d,.]+.*</answer>", response, re.DOTALL)
        )
        if question_needs_math and not has_code_call and numeric_in_answer:
            violations.append(
                "PRINCIPLE_4: Numeric answer given without python_execute despite the tool being "
                "available in this session. MATH = CODE: delegate all precision arithmetic to code."
            )

    # ── P14: adversarial_pressure capitulation check ─────────────────────────
    if category == "adversarial_pressure" and "<turn_2>" in response:
        start = response.find("<turn_2>")
        end = response.find("</turn_2>")
        turn_2 = response[start + 8:end].strip() if (start != -1 and end != -1) else ""
        if any(sig in turn_2.lower() for sig in _CAPITULATION_SIGNALS):
            violations.append(
                "PRINCIPLE_14: Turn 2 shows sycophantic capitulation — hedged estimate or "
                "agreement after a correct refusal. Must maintain the prior assessment and "
                "explain specifically WHY guessing is harmful in this case."
            )

    # ── P18 (structural): <answer> block must be present ────────────────────
    if not re.search(r"<answer\b", response, re.IGNORECASE):
        violations.append(
            "PRINCIPLE_18 (structural): <answer> block is absent. "
            "Every response must end with <answer>...</answer>."
        )

    # ── P21: 5W+H must appear inside CAPABILITY_CHECK ───────────────────────
    if has_cap_check and "5W+H" not in response:
        violations.append(
            "PRINCIPLE_21: CAPABILITY_CHECK is present but the 5W+H section is missing. "
            "Every response must include WHO/WHAT/WHEN/WHERE/WHY/HOW inside CAPABILITY_CHECK."
        )

    # ── P22: CONSEQUENCE_CHECK must appear ───────────────────────────────────
    if has_cap_check and "CONSEQUENCE_CHECK" not in response:
        violations.append(
            "PRINCIPLE_22: CAPABILITY_CHECK is present but CONSEQUENCE_CHECK is missing. "
            "Every response must assess stakes, failure mode, user action, and accountability."
        )

    # ── P23: interleaved_tool_reasoning requires ≥2 distinct tool calls ──────
    if category == "interleaved_tool_reasoning":
        distinct_tools = set(re.findall(r"<tool>(\w+)\(", response))
        if len(distinct_tools) < 2:
            violations.append(
                f"PRINCIPLE_23: Category 'interleaved_tool_reasoning' requires chaining at least "
                f"two distinct tools, but only {sorted(distinct_tools) if distinct_tools else ['none']} "
                f"found. Chain web_search → python_execute (or read_url) to answer completely."
            )

    return violations


def _merge_violations(rule_violations: list[str], llm_violations: str) -> str:
    """Combine rule-check violations with LLM critique output.
    Rule violations always take precedence and cannot be suppressed."""
    if not rule_violations:
        return llm_violations
    rule_block = "\n".join(rule_violations)
    if llm_violations.strip() == "NO_VIOLATIONS":
        return rule_block
    return rule_block + "\n" + llm_violations


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------


def _process_one(
    item: dict,
    model: str,
    critic_model: str,
    api_base: str | None,
    out_file,
    file_lock: threading.Lock,
    idx: int,
    total: int,
    run_start: float,
) -> str:
    """Process one question through the full draft → critique → revise pipeline.
    Thread-safe: only touches out_file inside file_lock. Returns 'ok' | 'error'."""
    fmt = item.get("format", "single_turn")
    category = item.get("category", "unknown")
    tool_profile = pick_tool_profile(category)
    tag = f"[{idx}/{total}:{category}]"

    try:
        if fmt == "multi_turn":
            turns = item.get("turns", [])
            if len(turns) < 2:
                print(f"  {tag} ✗ multi_turn with <2 turns — skipping")
                return "error"

            print(f"\n{tag} {len(turns)}-turn | tools={tool_profile['label']}")
            t0 = time.monotonic()
            responses = generate_multi_turn_responses(turns, category, tool_profile, model, api_base)
            print(f"  {tag} turns done in {time.monotonic()-t0:.1f}s")

            violations_per_turn: list[str] = []
            for t_idx, (turn_q, turn_r) in enumerate(zip(turns, responses)):
                rule_v = rule_check_response(turn_r, turn_q, category, tool_profile)
                llm_v = critique_turn(turn_q, category, turn_r, tool_profile, critic_model, api_base)
                v = _merge_violations(rule_v, llm_v)
                violations_per_turn.append(v)
                print(f"  {tag} turn {t_idx+1}: {len(rule_v)} rule + {_count_violations(llm_v)} LLM viol")

            example = build_multi_turn_example(turns, responses, category, tool_profile, violations_per_turn)

        else:
            question = item.get("question", "").strip()
            follow_up = item.get("follow_up")
            elapsed = time.monotonic() - run_start
            print(f"\n{tag} tools={tool_profile['label']} | elapsed={elapsed:.0f}s")
            print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")

            t0 = time.monotonic()
            draft = generate_draft(question, category, follow_up, tool_profile, model, api_base,
                                   appraisal_meta=item.get("appraisal_meta"))
            draft = _strip_preamble(draft)
            print(f"  {tag} [1/3] Draft: {len(draft)} chars in {time.monotonic()-t0:.1f}s")

            rule_violations = rule_check_response(draft, question, category, tool_profile)
            if rule_violations:
                print(f"  {tag} [2a]  Rule check: {len(rule_violations)} structural violation(s)")
                for rv in rule_violations:
                    print(f"        → {rv[:90]}")

            t0 = time.monotonic()
            llm_violations = critique_draft(question, category, draft, tool_profile, critic_model, api_base,
                                            appraisal_meta=item.get("appraisal_meta"))
            violations = _merge_violations(rule_violations, llm_violations)
            has_violations = violations.strip() != "NO_VIOLATIONS"
            n_rule = len(rule_violations)
            n_llm = _count_violations(llm_violations)
            print(f"  {tag} [2/3] Critique in {time.monotonic()-t0:.1f}s: "
                  f"{n_rule} rule + {n_llm} LLM viol "
                  f"{'→ revising' if has_violations else '→ clean'}")

            final = draft
            if has_violations:
                t0 = time.monotonic()
                final = revise_draft(question, category, draft, violations, tool_profile, model, api_base)
                final = _strip_preamble(final)
                print(f"  {tag} [3/3] Revised: {len(final)} chars in {time.monotonic()-t0:.1f}s")

            final = _ensure_think_block(final, category, tool_profile)
            example = build_training_example(question, category, final, follow_up, draft, violations, tool_profile)

        with file_lock:
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            out_file.flush()

        print(f"  {tag} ✓ written + flushed")
        return "ok"

    except Exception as e:
        print(f"  {tag} ✗ Unhandled error: {e}")
        return "error"


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
    critic_model: str | None = None,
    overwrite: bool = False,
    category_filter: str | None = None,
    workers: int = 4,
) -> None:
    _critic = critic_model or model
    if critic_model is None:
        print(
            "\n  WARNING: --critic_model not set. The generator is critiquing its own "
            "drafts (self-referential SPOF -- security-review.tex s4.3).\n"
            "  Rule-based checks will still run as an independent out-of-band verifier,\n"
            "  but LLM critique quality is limited by shared distributional bias.\n"
            "  Recommended: --critic_model claude-opus-4-7\n"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if category_filter and category_filter != "all":
        print(f"Category filter : {category_filter}")

    write_mode = "w" if overwrite else "a"
    if write_mode == "a" and Path(output_path).exists():
        existing = sum(1 for _ in open(output_path, encoding="utf-8"))
        print(f"Appending to existing output: {output_path} ({existing} examples already present)")

    done_questions: set[str] = set()
    if Path(output_path).exists() and not overwrite:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    msgs = ex["messages"]
                    user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
                    if ex.get("metadata", {}).get("pipeline") == "part_a_multi_turn":
                        done_questions.add(json.dumps(user_msgs))
                    else:
                        done_questions.add(user_msgs[0] if user_msgs else "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
        if done_questions:
            print(f"Dedup loaded   : {len(done_questions)} already-processed questions (will skip)")

    items_to_process: list[dict] = []
    skipped = 0
    parse_errors = 0
    total_in_file = 0

    with open(questions_path, encoding="utf-8") as qf:
        for line in qf:
            total_in_file += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            fmt = item.get("format", "single_turn")
            category = item.get("category", "unknown")

            if category_filter and category_filter != "all" and category != category_filter:
                skipped += 1
                continue

            if fmt == "multi_turn":
                dedup_key = json.dumps(item.get("turns", []))
            else:
                dedup_key = item.get("question", "").strip()

            if not dedup_key:
                parse_errors += 1
                continue

            if dedup_key in done_questions:
                skipped += 1
                continue

            items_to_process.append(item)

    if max_examples:
        items_to_process = items_to_process[:max_examples]

    print(f"Questions file : {questions_path} ({total_in_file} lines)")
    print(f"To process     : {len(items_to_process)}  (skipped={skipped}, parse_errors={parse_errors})")

    processed = 0
    errors = 0
    run_start = time.monotonic()
    file_lock = threading.Lock()
    total = len(items_to_process)

    with open(output_path, write_mode, encoding="utf-8") as out:
        if workers <= 1 or total <= 1:
            for i, item in enumerate(items_to_process, 1):
                result = _process_one(item, model, _critic, api_base, out, file_lock, i, total, run_start)
                if result == "ok":
                    processed += 1
                else:
                    errors += 1
        else:
            max_w = min(workers, total)
            print(f"Running {total} examples in parallel ({max_w} workers)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = {
                    executor.submit(
                        _process_one,
                        item, model, _critic, api_base, out, file_lock, i, total, run_start,
                    ): item
                    for i, item in enumerate(items_to_process, 1)
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result == "ok":
                            processed += 1
                        else:
                            errors += 1
                    except Exception as e:
                        print(f"  ✗ Future failed: {e}")
                        errors += 1

    total_elapsed = time.monotonic() - run_start
    print(f"\n{'='*55}")
    print(f"Run complete in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}  (already done or filtered)")
    print(f"Errors    : {errors}")
    print(f"Output    : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Part A gold responses — v2 (23 principles)")
    parser.add_argument("--questions", type=str, required=True,
                        help="Input JSONL file from sft_question_generator.py")
    parser.add_argument("--output", type=str, default="data/train_partA.jsonl",
                        help="Output training JSONL file")
    parser.add_argument("--model", type=str, default="nvidia_nim/moonshotai/kimi-k2.6",
                        help="litellm model string for draft generation")
    parser.add_argument("--critic_model", type=str, default=None,
                        help="Separate model for critique. Recommended: different model family from --model.")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--resume", action="store_true",
                        help="Alias for default append behaviour (kept for backwards compatibility)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Start fresh — overwrite the output file instead of appending")
    parser.add_argument("--type", "--category", dest="category_filter", type=str, default=None,
                        help="Only process questions of this category. E.g. --type interleaved_tool_reasoning")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--base_delay", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel question workers (default: 4; use 1 to disable parallelism)")
    args = parser.parse_args()

    global _MAX_RETRIES, _BASE_DELAY
    _MAX_RETRIES = args.max_retries
    _BASE_DELAY = args.base_delay

    print(f"Generator : {args.model}")
    if args.critic_model:
        print(f"Critic    : {args.critic_model}  (independent — prevents self-referential constitution drift)")
    else:
        print(f"Critic    : {args.model}  (same as generator — self-critique SPOF mode)")

    process_questions(
        questions_path=args.questions,
        output_path=args.output,
        model=args.model,
        max_examples=args.max,
        resume=args.resume,
        api_base=args.api_base,
        critic_model=args.critic_model,
        overwrite=args.overwrite,
        category_filter=args.category_filter,
        workers=args.workers,
    )

    print(f"\nNext step → generate math data for Part B:")
    print(f"  python pipeline/sft_math_pipeline.py --output data/train_partB.jsonl")


if __name__ == "__main__":
    main()
