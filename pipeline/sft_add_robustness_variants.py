"""
sft_add_robustness_variants.py
================================
Adds "prompt robustness" variants to an SFT JSONL dataset.

Goal
----
After standard SFT the model follows the constitution only when it sees the
full 23-principle system prompt it was trained on.  Adding variants with
shorter prompts — where the response is identical — trains the model to treat
constitutional behaviour as intrinsic rather than prompt-dependent.

Three variant levels are used, applied randomly at the configured ratios:

  MINIMAL  (default 15 %)
    "You are a trustworthy AI assistant."
    Teaches the model to produce CAPABILITY_CHECK + constitution behaviour
    even with a bare role description.

  BRIEF    (default 10 %)
    Role + one-line tool mention + short instruction.
    Closer to what a real deployment might set, without spelling out principles.

  NO_PRINCIPLES (default 5 %)
    Full CAPABILITY_CHECK template structure without the 23-principle list.
    Tests whether the template alone (without the enumerated list) is enough.

Only the system message is changed.  The conversation, tool turns, and answer
are identical to the original — this is the key training signal: "regardless
of how much the prompt tells you, produce this quality of output."

Usage
-----
    python pipeline/sft_add_robustness_variants.py \\
        --input  pipeline/data/train_sft_v3.jsonl \\
        --output pipeline/data/train_sft_v3_robust.jsonl

    python pipeline/sft_add_robustness_variants.py \\
        --input   pipeline/data/train_sft_v3.jsonl \\
        --output  pipeline/data/train_sft_v3_robust.jsonl \\
        --minimal 0.20 --brief 0.10 --no_principles 0.05 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Variant system prompts
# ---------------------------------------------------------------------------

# Tool context strings — mirrors sft_gold_response_generator.py TOOL_PROFILES[*]["context"]
_TOOL_CONTEXTS = {
    "all_tools":          "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✓",
    "compute_only":       "python_execute ✓ | web_search ✗ | read_url ✗ | get_datetime ✗",
    "compute_and_search": "python_execute ✓ | web_search ✓ | read_url ✓ | get_datetime ✗",
    "no_tools":           "python_execute ✗ | web_search ✗ | read_url ✗ | get_datetime ✗",
}

_TOOL_NOTES = {
    "all_tools":          "All four tools are available in this session.",
    "compute_only":       "Only python_execute is available. No internet or time access.",
    "compute_and_search": "python_execute and web_search/read_url are available. No datetime tool.",
    "no_tools":           "No tools are available in this session. Training knowledge only.",
}

# ── Level 1: MINIMAL — bare role description ─────────────────────────────────
# For tool-using examples we keep a one-liner about available tools so the
# model is not left completely in the dark; the key drop is the 23 principles
# and the CAPABILITY_CHECK template.

def _minimal_prompt(tool_profile: str) -> str:
    has_tools = tool_profile not in ("no_tools", "unknown", "")
    if has_tools:
        ctx = _TOOL_CONTEXTS.get(tool_profile, "python_execute ✓")
        return (
            f"You are a trustworthy AI assistant.\n"
            f"Available tools in this session: {ctx}"
        )
    return "You are a trustworthy AI assistant."


# ── Level 2: BRIEF — role + short behavioural instruction ────────────────────
# The key behavioural expectations are stated in one or two sentences but the
# CAPABILITY_CHECK template and the 23 principles are absent.

def _brief_prompt(tool_profile: str) -> str:
    ctx = _TOOL_CONTEXTS.get(tool_profile, "")
    tool_line = f"\nSession tools: {ctx}" if ctx else ""
    return (
        "You are a trustworthy AI assistant. Think carefully before every "
        "response: identify what the question requires, which tools (if any) "
        "are needed, what the stakes are if you are wrong, and what you cannot "
        f"know.{tool_line}"
    )


# ── Level 3: NO_PRINCIPLES — full template without the 23-principle list ─────
# Keeps the CAPABILITY_CHECK scaffold so the model still sees the right
# structure, but removes the enumerated list of principles.  Tests whether
# the model internalised the principles or merely pattern-matched the headers.

def _no_principles_prompt(tool_profile: str) -> str:
    ctx = _TOOL_CONTEXTS.get(tool_profile, "python_execute ✓")
    note = _TOOL_NOTES.get(tool_profile, "")

    has_python  = "python_execute ✓" in ctx
    has_search  = "web_search ✓" in ctx
    has_url     = "read_url ✓" in ctx
    has_dt      = "get_datetime ✓" in ctx

    call_lines = []
    if has_python:  call_lines.append("  <tool>python_execute(code='...')</tool>")
    if has_search:  call_lines.append("  <tool>web_search(query='...')</tool>")
    if has_url:     call_lines.append("  <tool>read_url(url='...', prompt='what to extract')</tool>")
    if has_dt:      call_lines.append("  <tool>get_datetime()</tool>")
    call_examples = "\n".join(call_lines) if call_lines else "  (no tools available this session)"

    return (
        "You are a trustworthy AI assistant. Before answering any question, complete a full "
        "CAPABILITY_CHECK inside your <think> block using this exact structure:\n\n"
        "<think>\n"
        "CAPABILITY_CHECK:\n\n"
        "  5W+H:\n"
        "    WHO is affected: [the user / third parties / institutions involved]\n"
        "    WHAT is required: [list requirements to answer correctly]\n"
        "    WHEN: [time-sensitivity — live data needed, training cutoff relevant, dated context]\n"
        "    WHERE: [jurisdiction, region, domain, platform]\n"
        "    WHY: [inferred intent and underlying goal]\n"
        "    HOW: [tool selection and method]\n\n"
        "  First Principles:\n"
        "    Core truth: [the irreducible fact this answer rests on]\n"
        "    Assumptions: [what I am taking for granted — flag if unverified]\n\n"
        f"  Session tools: {ctx}\n"
        "  Gap: [what I cannot obtain]\n"
        "  Strategy: [tool chain plan or honest refusal]\n\n"
        "  CONSEQUENCE_CHECK:\n"
        "    Stakes: [low / medium / high + reason]\n"
        "    If wrong: [concrete harm to the user]\n"
        "    User will likely: [action they will take with this answer]\n"
        "    Accountability: [what to hedge or flag in the answer]\n"
        "</think>\n"
        "<answer>\n"
        "[response to the user — high-stakes answers include explicit caveat]\n"
        "</answer>\n\n"
        f"{note}\n\n"
        "Tool call syntax (place between </think> and <answer>, or inside <think> before the final answer):\n"
        f"{call_examples}"
    )


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

_VARIANT_BUILDERS = {
    "minimal":       _minimal_prompt,
    "brief":         _brief_prompt,
    "no_principles": _no_principles_prompt,
}


def make_variant(example: dict, level: str) -> dict:
    """Return a copy of example with the system message replaced by a minimal prompt."""
    messages = example.get("messages", [])
    tool_profile = example.get("metadata", {}).get("tool_profile", "all_tools")

    builder = _VARIANT_BUILDERS[level]
    new_system = builder(tool_profile)

    new_messages = []
    for m in messages:
        if m["role"] == "system":
            new_messages.append({"role": "system", "content": new_system})
        else:
            new_messages.append(m)

    # If no system message existed, prepend one
    if not any(m["role"] == "system" for m in messages):
        new_messages = [{"role": "system", "content": new_system}] + new_messages

    out = dict(example)
    out["messages"] = new_messages
    meta = dict(out.get("metadata", {}))
    meta["robustness_variant"] = level
    out["metadata"] = meta
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add prompt-robustness variants to an SFT JSONL dataset"
    )
    parser.add_argument("--input",         default="pipeline/data/train_sft_v3.jsonl")
    parser.add_argument("--output",        default="pipeline/data/train_sft_v3_robust.jsonl")
    parser.add_argument("--minimal",       type=float, default=0.15,
                        help="Fraction of examples to get a MINIMAL variant (default: 0.15)")
    parser.add_argument("--brief",         type=float, default=0.10,
                        help="Fraction of examples to get a BRIEF variant (default: 0.10)")
    parser.add_argument("--no_principles", type=float, default=0.05,
                        help="Fraction of examples to get a NO_PRINCIPLES variant (default: 0.05)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    in_path  = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    total_orig = len(examples)

    # Assign each example independently to zero, one, or more variant levels.
    # Using independent Bernoulli draws so the fractions don't compete.
    variants: list[dict] = []
    counts = {level: 0 for level in _VARIANT_BUILDERS}

    for ex in examples:
        for level, frac in [
            ("minimal",       args.minimal),
            ("brief",         args.brief),
            ("no_principles", args.no_principles),
        ]:
            if rng.random() < frac:
                variants.append(make_variant(ex, level))
                counts[level] += 1

    all_examples = examples + variants
    rng.shuffle(all_examples)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Input examples   : {total_orig}")
    print(f"Variants added   :")
    for level, n in counts.items():
        print(f"  {level:<15} {n:>5}  ({100*n/total_orig:.1f}%)")
    print(f"Total output     : {len(all_examples)}")
    print(f"Written to       : {out_path}")
    print()
    print("Next step → train on the robust dataset:")
    print(f"  python pipeline/2_model_trainer.py --mode sft --data_dir pipeline/data")
    print(f"  # (update trainer to use train_sft_v3_robust.jsonl, or symlink it to train_sft_v3.jsonl)")


if __name__ == "__main__":
    main()
