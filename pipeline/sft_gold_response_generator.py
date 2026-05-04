"""
SFT Gold Response Generator (Part A)
======================================
Takes questions from sft_question_generator.py and produces gold training examples
using a teacher model via litellm (vendor-agnostic: Anthropic, OpenAI, Ollama, Groq, NVIDIA NIM…).

Each example goes through:
  1. Draft generation  — teacher generates an initial response
  2. Self-critique     — teacher checks the draft against constitution principles
  3. Revision          — if violations found, teacher rewrites the response
  4. Final formatting  — wraps into the training JSONL format with CAPABILITY_CHECK

Model string examples:
    NVIDIA NIM : nvidia_nim/moonshotai/kimi-k2.6        (recommended generator — free, frontier)
    NVIDIA NIM : nvidia_nim/minimaxai/minimax-m2.7      (recommended critic   — free, different family)
    Anthropic  : claude-sonnet-4-6  /  claude-opus-4-7
    Groq       : groq/llama-3.3-70b-versatile

Usage:
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --output data/train_partA.jsonl
    python sft_gold_response_generator.py --questions data/questions_partA.jsonl \\
                                           --type adversarial_pressure \\
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
import threading
import time
from pathlib import Path
from datetime import datetime

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
                            "verbose_context_behavioral", "multi_turn_conversation",
                            "appraisal_empathy"}


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

DRAFT_PROMPT = """MANDATORY OUTPUT FORMAT — you must follow this exactly or the response is invalid:

<think>
CAPABILITY_CHECK:
  This question requires: [list what's needed]
  Session tools: [list exactly the tools given below — no more, no less]
  Gap: [what's missing, if anything]
  Strategy: [how to handle the gap]
[any further reasoning here]
</think>
<answer>
[your response to the user]
</answer>

Do NOT write anything outside these tags. Do NOT use markdown headers or bullet points outside the tags. The response must start with <think> and end with </answer>.

---

You are demonstrating ideal AI assistant behaviour for training purposes.

The user asks: {question}

{follow_up_context}

Session tool availability for this example: {tool_context}

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

APPRAISAL_DRAFT_PROMPT = """MANDATORY OUTPUT FORMAT — follow exactly:

<think>
CAPABILITY_CHECK:
  This question requires: [list what's needed]
  Session tools: [list exactly the tools available]
  Gap: [any gap]
  Strategy: [how to handle]
<appraisal>
{top3_str}
reading: {short_reading}
→ [one sentence on how this shapes your response]
</appraisal>
</think>
<answer>
[empathetic response to the user]
</answer>

Do NOT write anything outside these tags. Start with <think>, end with </answer>.

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

Also check the standard constitution principles (CAPABILITY_CHECK present? format correct?).

List ONLY the violations found. If none: NO_VIOLATIONS
Format: ISSUE_X: [description and correct behaviour]"""

REVISION_PROMPT = """MANDATORY OUTPUT FORMAT — the revised response must use this exact structure:

<think>
CAPABILITY_CHECK:
  This question requires: [list]
  Session tools: [list exactly what was available]
  Gap: [any gap]
  Strategy: [conclusion]
[further reasoning]
</think>
<answer>
[response to user]
</answer>

Do NOT write anything outside these tags. Start with <think>, end with </answer>.

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
    "appraisal_empathy": (
        "The user has shared an emotionally significant message. "
        "After CAPABILITY_CHECK, include an <appraisal> block that names the top 3 OCC appraisal "
        "dimensions most relevant to this message (e.g. pleasantness, goal_relevance, coping_potential), "
        "gives each a 0–1 value, provides a one-line qualitative reading, and concludes with a "
        "'→ implication' sentence about how this should shape the response. "
        "The <answer> must be empathetically conditioned on that reading — validate the emotional state "
        "BEFORE giving any advice or information. Match the tone to the valence: warm and celebratory "
        "for positive events, gentle and grounding for distress, steady and practical for mixed signals. "
        "Do not project emotions the user hasn't expressed. Never skip straight to problem-solving."
    ),
}

# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


# Retry config — overridden by CLI args in main()
_MAX_RETRIES: int = 5
_BASE_DELAY: float = 3.0


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
                # Response truncated — double max_tokens and retry (cap at 4096)
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
    # Appraisal empathy category uses a specialised prompt
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

    n_violations = _count_violations(violations)
    return {
        "messages": messages,
        "metadata": {
            "source": "constitution_teacher",
            "category": category,
            "tool_profile": tool_profile["label"],
            "constitution_violations_in_draft": n_violations,
            "constitution_score": max(0, 19 - n_violations) / 19,  # 1.0 = perfect; used as GRPO reward signal
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
    constitution_score = max(0, (n_turns * 19 - total_violations)) / (n_turns * 19) if n_turns > 0 else 1.0

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
    """Extract content between <tag>...</tag>."""
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return text
    return text[start + len(tag) + 2:end].strip()


# ---------------------------------------------------------------------------
# Rule-based constitutional verifier (Security Blocker 2 — OWASP LLM04)
#
# Checks 5 principles with unambiguous structural signals that the LLM critic
# cannot miss due to shared distributional bias with the generator.
# Violations use the same PRINCIPLE_N: format so _count_violations picks them
# up correctly and they flow into the revision step automatically.
# ---------------------------------------------------------------------------

_ALL_TOOL_NAMES = frozenset({
    "python_execute", "web_search", "read_url", "get_datetime", "get_exchange_rate",
})

_MATH_SIGNAL_RE = re.compile(
    r"(?:"
    r"\d+\.?\d*\s*(?:each|per\s+\w+|times|divided|\*|×|%)"  # "10.50 each", "3 times", etc.
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
    """Deterministic structural checks for the 5 highest-signal constitutional principles.

    Returns a list of violation strings in PRINCIPLE_N: format.
    Call BEFORE the LLM critique and merge results so the revision step
    always receives these violations even if the LLM critic missed them.
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

    # ── P3: TOOL DISCIPLINE — only call tools that are ✓ in active profile ──
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

    # ── P4: MATH = CODE — numeric answer without code when code is available ─
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

    # ── P14: adversarial_pressure — turn_2 must not capitulate ──────────────
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
                print(f"  {tag} [3/3] Revised: {len(final)} chars in {time.monotonic()-t0:.1f}s")

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
    # ── Critic selection ─────────────────────────────────────────────────────
    # A frozen larger model (e.g. claude-opus-4-7) prevents the self-referential
    # critique SPOF where the model being trained also sets the grading standard.
    # Rule-based checks (rule_check_response) run regardless and cannot be
    # suppressed by the critic — they are the Blocker 2 independent verifier.
    _critic = critic_model or model
    if critic_model is None:
        print(
            "\n  ⚠  WARNING: --critic_model not set. The generator is critiquing its own "
            "drafts (self-referential SPOF — security-review.tex §4.3).\n"
            "  Rule-based checks will still run as an independent out-of-band verifier,\n"
            "  but LLM critique quality is limited by shared distributional bias.\n"
            "  Recommended: --critic_model claude-opus-4-7\n"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if category_filter and category_filter != "all":
        print(f"Category filter : {category_filter}")

    # Append by default; overwrite only if explicitly requested
    write_mode = "w" if overwrite else "a"
    if write_mode == "a" and Path(output_path).exists():
        existing = sum(1 for _ in open(output_path, encoding="utf-8"))
        print(f"Appending to existing output: {output_path} ({existing} examples already present)")

    # Load already-processed questions for deduplication (always, not just on --resume)
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

    # Load and filter all items upfront so parallel workers start immediately
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
    parser = argparse.ArgumentParser(description="Generate Part A gold responses via teacher model + self-critique")
    parser.add_argument("--questions", type=str, required=True,
                        help="Input JSONL file from sft_question_generator.py")
    parser.add_argument("--output", type=str, default="pipeline/data/train_partA.jsonl",
                        help="Output training JSONL file")
    parser.add_argument("--model", type=str, default="nvidia_nim/moonshotai/kimi-k2.6",
                        help="litellm model string for draft generation")
    parser.add_argument("--critic_model", type=str, default=None,
                        help="Separate model for critique. Recommended: different model family from --model. "
                             "E.g. --model nvidia_nim/moonshotai/kimi-k2.6 --critic_model nvidia_nim/minimaxai/minimax-m2.7")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--resume", action="store_true",
                        help="Alias for default append behaviour (kept for backwards compatibility)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Start fresh — overwrite the output file instead of appending")
    parser.add_argument("--type", "--category", dest="category_filter", type=str, default=None,
                        help="Only process questions of this category. E.g. --type adversarial_pressure")
    parser.add_argument("--max_retries", type=int, default=5,
                        help="Max retry attempts on rate limit / connection errors (default: 5)")
    parser.add_argument("--base_delay", type=float, default=3.0,
                        help="Base delay in seconds for exponential backoff (default: 3.0)")
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


if __name__ == "__main__":
    main()
