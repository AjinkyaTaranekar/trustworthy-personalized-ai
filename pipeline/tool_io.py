"""tool_io.py — single source of truth for how tool results are presented to the model.

This is the canonical sanitiser used in BOTH training-data generation and inference, so the
representation a model sees is byte-identical across the two (the train/serve contract). It was
previously inlined in 3_infererence.py as `_sanitise_tool_output`; it now lives here and is
imported by:

  * 3_infererence.py            — single-model tool loop + the dual Thinker–Executor loop
  * thinker_executor_orchestrator.py — standalone --serve/--chat/--question default sanitiser
  * sft_trajectory_splitter.py  — stamps Thinker tool turns with THIS exact representation

Two responsibilities, both safety- and parity-critical:
  1. Injection stripping — neutralise XML control tags / "ignore previous instructions" style
     content that arrives inside a tool result (which the model reads as a user turn).
  2. Per-tool output budgets — keep each result readable and the context affordable on a 0.6B.

The Qwen3 chat template already delimits a tool turn natively as
`<|im_start|>user\n<tool_response>…</tool_response>`, so the `[TOOL_RESULT: name]` text wrapper is
a SECOND, redundant delimiter. It is retained because the single-model SFT data was built with it
(removing it would put the single model off-distribution); the fix for the Thinker is to stamp the
SAME wrapper into its training data rather than to drop the wrapper at serve time.
"""
from __future__ import annotations

import re
from typing import Dict, Set


# ---------------------------------------------------------------------------
# Tool profiles — the single source of truth for which tools are active per session.
# Imported by 3_infererence.py (serving) and restamp_native_tools.py (training-data stamping)
# so the tool SET a model is trained on equals the set it is served (train/serve parity). Memory
# and scratchpad tools are ALWAYS on (the thesis's personalisation/state mechanism); the profile
# label only gates the external-world tools (python_execute / web_search / read_url).
# ---------------------------------------------------------------------------
ALWAYS_ON_TOOLS: Set[str] = {
    "get_datetime",
    "scratchpad_sections", "scratchpad_read", "scratchpad_update",
    "user_memory_sections", "user_memory_read", "user_memory_update",
}

TOOL_PROFILES: Dict[str, set] = {
    "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime"} | ALWAYS_ON_TOOLS,
    "compute_only":       {"python_execute"} | ALWAYS_ON_TOOLS,
    "compute_and_search": {"python_execute", "web_search", "read_url"} | ALWAYS_ON_TOOLS,
    "no_tools":           set(ALWAYS_ON_TOOLS),
}

# Patterns that could hijack the model's instruction context if returned by a tool.
# Strips XML control tags the model reads, and common injection phrases from web content.
INJECTION_RE = re.compile(
    r"</?tool>|</?think>|</?answer>|CAPABILITY_CHECK|\[/?TOOL_RESULT[^\]]*\]"
    r"|ignore\s+(all\s+)?previous\s+(instructions?|prompts?|context)"
    r"|disregard\s+previous|you\s+are\s+now\s+|new\s+instructions?\s*:",
    re.IGNORECASE,
)

# 1500 chars ≈ 375 tokens per tool result. With 4096 token context and ~400 token system prompt,
# 4 tool calls at 375 tokens each = 1500 tokens, leaving ~2200 for reasoning and answer.
MAX_TOOL_OUTPUT = 1500

# Per-tool output budgets — tools that return structured multi-result data get tighter limits so
# each result stays readable rather than one result dominating.
TOOL_OUTPUT_BUDGETS: Dict[str, int] = {
    "web_search":     1200,   # multi-result; truncate aggressively
    "read_url":       1000,   # raw HTML → text can be very long
    "python_execute":  800,   # stdout only; long output usually means debug noise
}


def sanitise_tool_result(tool_name: str, raw: str) -> str:
    """Strip injection patterns and apply per-tool token budgets before context injection.

    For web_search: keeps the first N chars of the *combined* result, preserving result
    boundaries so the model sees [title] + snippet for each hit rather than one result in full
    and nothing else. Returns the `[TOOL_RESULT: name] … [/TOOL_RESULT]` wrapped form.
    """
    raw = raw or ""
    cleaned = INJECTION_RE.sub("[FILTERED]", raw)
    budget = TOOL_OUTPUT_BUDGETS.get(tool_name, MAX_TOOL_OUTPUT)

    if tool_name == "web_search" and len(cleaned) > budget:
        # Split on blank lines (result boundaries) and keep as many full results as fit
        results = [r.strip() for r in cleaned.split("\n\n") if r.strip()]
        kept, total = [], 0
        for r in results:
            if total + len(r) + 2 > budget:
                break
            kept.append(r)
            total += len(r) + 2
        cleaned = "\n\n".join(kept) if kept else cleaned[:budget]
        if len(cleaned) < len(raw):
            cleaned += " … [truncated]"
    elif len(cleaned) > budget:
        cleaned = cleaned[:budget] + " … [truncated]"

    return f"[TOOL_RESULT: {tool_name}]\n{cleaned}\n[/TOOL_RESULT]"
