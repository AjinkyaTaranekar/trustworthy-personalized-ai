"""
Empathy Module (Runtime)
========================
Appraisal-conditioned generation support for the inference server.

At inference time Qwen produces its own <appraisal> block from its fine-tuning
on AppraisePLM-labelled data — no external model is called at runtime.
This module provides:

  analyse_appraisal()  — optional system-prompt injection for the <appraisal> nudge
  APPRAISAL_SYSTEM_PREFIX — the prefix injected when ENABLE_EMPATHY=True
  format_appraisal_block() — format a dict into the canonical <appraisal> XML element

The <appraisal> block lives inside <think>...</think>, before the main reasoning.
When ENABLE_EMPATHY=False, no injection happens and the model generates normally.

Wiring (Phase 4 — 3_infererence.py):
    from empathy import analyse_appraisal, APPRAISAL_SYSTEM_PREFIX

    if cfg.ENABLE_EMPATHY:
        appraisal_ctx = analyse_appraisal(user_turn, llm_fn)
        system_prompt = APPRAISAL_SYSTEM_PREFIX + base_system_prompt
        # appraisal_ctx is also returned in response metadata
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Appraisal format spec
# ---------------------------------------------------------------------------
# The canonical <appraisal> block that Qwen is trained to produce.
# It lives at the start of <think>, before the First Principles reasoning.
#
# Example:
#   <appraisal>
#     pleasantness: 0.95 | goal_relevance: 0.85 | coping_potential: 0.23
#     reading: high pleasantness, high goal relevance, low coping potential
#     → user achieved something important but feels they barely held it together
#   </appraisal>
#
# When the user's turn has no emotional signal the block is omitted entirely.

APPRAISAL_OPEN  = "<appraisal>"
APPRAISAL_CLOSE = "</appraisal>"

# System-prompt prefix injected when ENABLE_EMPATHY=True.
# Nudges the model to include an appraisal block in its think chain.
APPRAISAL_SYSTEM_PREFIX = (
    "For every user turn, at the start of your <think> block (before First Principles "
    "reasoning), include an <appraisal> block if the user's message carries emotional content. "
    "The block format is:\n"
    "<appraisal>\n"
    "  <dim1>: <val> | <dim2>: <val> | <dim3>: <val>\n"
    "  reading: <one-line qualitative summary>\n"
    "  → <one sentence: what this means for how you should respond>\n"
    "</appraisal>\n"
    "If the turn is purely factual or task-based with no emotional signal, omit the block.\n\n"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AppraisalContext:
    """Appraisal analysis for one user turn."""
    top3:    List[str]           = field(default_factory=list)
    values:  Dict[str, float]    = field(default_factory=dict)
    reading: str                 = ""
    valence: float               = 0.5    # 0 = very negative, 1 = very positive
    present: bool                = False  # False when no emotional signal detected

    def to_prompt_block(self) -> str:
        """Format for optional system-prompt injection."""
        if not self.present or not self.top3:
            return ""
        dim_str = " | ".join(
            f"{dim}: {self.values.get(dim, 0.5):.2f}" for dim in self.top3
        )
        return (
            f"{APPRAISAL_OPEN}\n"
            f"  {dim_str}\n"
            f"  reading: {self.reading}\n"
            f"{APPRAISAL_CLOSE}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Appraisal format helpers (used by SFT data generation)
# ---------------------------------------------------------------------------

def format_appraisal_block(
    top3:    List[str],
    values:  Dict[str, float],
    reading: str,
) -> str:
    """
    Build the canonical <appraisal> XML element for insertion into a <think> block.
    Used by sft_gold_response_generator to construct training examples.
    """
    if not top3:
        return ""
    dim_str = " | ".join(f"{dim}: {values.get(dim, 0.5):.2f}" for dim in top3)
    narrative = reading.split("→", 1)[-1].strip() if "→" in reading else reading
    return (
        f"{APPRAISAL_OPEN}\n"
        f"  {dim_str}\n"
        f"  reading: {reading.split('→')[0].strip()}\n"
        f"  → {narrative}\n"
        f"{APPRAISAL_CLOSE}"
    )


def parse_appraisal_block(text: str) -> Optional[AppraisalContext]:
    """
    Extract and parse an <appraisal> block from a model response.
    Returns None if no block is found.
    Used by the critic in sft_gold_response_generator to validate training examples.
    """
    match = re.search(
        r"<appraisal>(.*?)</appraisal>", text, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return None

    content = match.group(1).strip()
    values: Dict[str, float] = {}
    top3: List[str] = []
    reading = ""

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("→") or line.startswith("reading:"):
            if line.startswith("reading:"):
                reading = line[len("reading:"):].strip()
            continue
        # Parse "dim1: val | dim2: val | dim3: val"
        for part in line.split("|"):
            part = part.strip()
            if ":" in part:
                key, _, val_str = part.partition(":")
                key = key.strip().replace(" ", "_")
                try:
                    val = float(val_str.strip())
                    values[key] = val
                    top3.append(key)
                except ValueError:
                    pass

    if not values:
        return None

    valence = (values.get("pleasantness", 0.5) + (1.0 - values.get("unpleasantness", 0.5))) / 2

    return AppraisalContext(
        top3=top3[:3],
        values=values,
        reading=reading,
        valence=round(valence, 4),
        present=True,
    )


# ---------------------------------------------------------------------------
# Runtime analysis (called from 3_infererence.py)
# ---------------------------------------------------------------------------

_APPRAISAL_ANALYSE_PROMPT = """\
Analyse the following user message for emotional appraisal dimensions.
Identify up to three dimensions from this list that are most salient:
pleasantness, unpleasantness, goal_relevance, coping_potential,
meaning_significance, self_control, fairness, suddenness, anticipated_effort

Return ONLY a JSON object (no prose, no markdown):
{{
  "present": true,
  "top3": ["pleasantness", "goal_relevance", "coping_potential"],
  "values": {{"pleasantness": 0.9, "goal_relevance": 0.85, "coping_potential": 0.3}},
  "reading": "high pleasantness, high goal relevance, low coping potential",
  "valence": 0.85
}}

If the message has no emotional content, return:
{{
  "present": false,
  "top3": [], "values": {{}}, "reading": "", "valence": 0.5
}}

User message: {turn_text}"""


def analyse_appraisal(
    turn_text: str,
    llm_fn: Callable[[str], str],
) -> AppraisalContext:
    """
    Analyse a user turn for appraisal dimensions.
    Returns an empty AppraisalContext when the message has no emotional signal.

    llm_fn is the server's own _raw_generate() — no HTTP round-trip.
    """
    empty = AppraisalContext()

    if not turn_text or not turn_text.strip():
        return empty

    prompt = _APPRAISAL_ANALYSE_PROMPT.format(turn_text=turn_text.strip())

    try:
        raw = llm_fn(prompt)
    except Exception as exc:
        logger.debug("Appraisal analysis failed: %s", exc)
        return empty

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```", "", raw).strip()

    # Extract first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return empty

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return empty

    if not data.get("present", False):
        return empty

    return AppraisalContext(
        top3    = [str(d) for d in data.get("top3", [])[:3]],
        values  = {str(k): float(v) for k, v in data.get("values", {}).items()},
        reading = str(data.get("reading", "")),
        valence = float(data.get("valence", 0.5)),
        present = True,
    )
