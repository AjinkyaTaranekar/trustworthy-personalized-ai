"""
Pre-Flight SFT Dataset Validation
==================================
Enforces 5 quality invariants on every row of a training JSONL before
the Unsloth training loop starts. If >5% of rows fail, the pipeline is
fundamentally broken — fix the generator, not the validator.

Invariants:
  1. System prompt contains no leaked teacher constitution
     (the verbose student prompt is intentional; the check is for teacher-only
     phrases that should never reach the student JSONL)
  2. <think> block ≥ 150 chars  (substantive reasoning, not synthetic laziness)
  3. No banned placeholders in <think>  (no v2-style shortcuts)
  4. Tool call immediately followed by tool role  (sequence integrity)
  5. Last message is assistant with <answer>  (end-to-end resolution)

Usage:
    python validate_sft_data.py --input data/train_v3.jsonl
    python validate_sft_data.py --input data/train_v3.jsonl --fix --output data/train_v3_clean.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

_BANNED_PHRASES = frozenset({
    "see answer below", "inferred from question", "none flagged",
    "capability_check:", "principle_", "5w+h:", "consequence_check:",
})

_MIN_THINK_CHARS = 150

# Phrases that appear ONLY in the teacher system prompt and must never leak into
# the student JSONL. The student prompt is now intentionally verbose (400+ words
# with First Principles + 5W+H instructions), so a word-count cap is wrong;
# instead we detect specific teacher-only phrases.
_TEACHER_LEAK_PHRASES = frozenset({
    "demonstrate through behavior; never name them",   # teacher constitution header
    "mandatory output format — follow this exactly",   # teacher format rules header
    "you are a frontier ai assistant generating",      # teacher identity line
    "violation invalidates the training example",      # teacher format rule label
})


def validate_row(row: dict) -> tuple[bool, str]:
    """Return (is_valid, failure_reason). Returns (True, 'ok') on success."""
    messages = row.get("messages", [])
    if not messages:
        return False, "empty_messages"

    # ── 1. No leaked teacher constitution ────────────────────────────────────
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    if system_msg is None:
        return False, "missing_system_message"
    sys_lower = system_msg.get("content", "").lower()
    for phrase in _TEACHER_LEAK_PHRASES:
        if phrase in sys_lower:
            return False, f"teacher_constitution_leaked: '{phrase[:60]}'"

    # ── 2 & 3: <think> block length + banned placeholders ───────────────────
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not asst_msgs:
        return False, "no_assistant_message"

    first_asst = asst_msgs[0].get("content", "")
    think_m = re.search(r"<think>(.*?)</think>", first_asst, re.DOTALL | re.IGNORECASE)
    if not think_m:
        return False, "missing_think_block"
    think_text = think_m.group(1).strip()
    if len(think_text) < _MIN_THINK_CHARS:
        return False, f"think_block_too_short: {len(think_text)} chars (min {_MIN_THINK_CHARS})"
    lower_think = think_text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower_think:
            return False, f"banned_placeholder_in_think: '{phrase}'"

    # ── 4. Tool sequence integrity ───────────────────────────────────────────
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "<tool>" not in content:
            continue
        if i + 1 >= len(messages) or messages[i + 1].get("role") != "tool":
            next_role = messages[i + 1].get("role", "missing") if i + 1 < len(messages) else "missing"
            return False, (
                f"tool_sequence_violation: <tool> in assistant[{i}] "
                f"not followed by tool role (got '{next_role}')"
            )

    # ── 5. End-to-end resolution ─────────────────────────────────────────────
    last = messages[-1]
    if last.get("role") != "assistant":
        return False, f"last_message_not_assistant: role='{last.get('role')}'"
    if "<answer>" not in last.get("content", ""):
        return False, "last_assistant_missing_answer_tag"

    return True, "ok"


def validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Partition rows into (valid, invalid) and tally failure reasons."""
    from collections import Counter
    valid, invalid, reasons = [], [], Counter()
    for row in rows:
        ok, reason = validate_row(row)
        if ok:
            valid.append(row)
        else:
            invalid.append(row)
            reasons[reason] += 1
    return valid, invalid, dict(reasons)


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-flight SFT dataset validation")
    p.add_argument("--input", required=True, help="JSONL file to validate")
    p.add_argument("--fix", action="store_true", help="Write valid rows to --output")
    p.add_argument("--output", default=None,
                   help="Output path for valid rows (default: <input>_clean.jsonl)")
    p.add_argument("--max_drop_pct", type=float, default=5.0,
                   help="Exit with error if more than this %% of rows fail (default: 5.0)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error — {e}")

    print(f"Loaded {len(rows)} rows from {input_path}")
    valid, invalid, reasons = validate_rows(rows)

    drop_pct = 100 * len(invalid) / max(len(rows), 1)
    print(f"\nValid   : {len(valid)}")
    print(f"Invalid : {len(invalid)}  ({drop_pct:.1f}%)")
    if reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<55} {count}")

    if drop_pct > args.max_drop_pct:
        print(f"\nFAIL: drop rate {drop_pct:.1f}% exceeds threshold {args.max_drop_pct:.1f}%.")
        print("The generation pipeline is fundamentally broken — fix the generator.")
        sys.exit(1)

    if args.fix:
        out_path = (Path(args.output) if args.output
                    else input_path.with_stem(input_path.stem + "_clean"))
        with open(out_path, "w", encoding="utf-8") as f:
            for row in valid:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(valid)} valid rows → {out_path}")

    print("\nPASS: dataset is within quality threshold.")


if __name__ == "__main__":
    main()
