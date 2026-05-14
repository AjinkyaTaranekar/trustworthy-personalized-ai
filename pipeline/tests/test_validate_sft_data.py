"""Tests for validate_sft_data.py — all 5 quality gate assertions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def _make_row(
    system="You are a trustworthy AI assistant. Reason in think tags.",
    think_content="x" * 200,
    tool_call=None,
    tool_result=None,
    final_answer="<answer>42</answer>",
):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What is 6 * 7?"},
    ]
    asst_content = f"<think>{think_content}</think>"
    if tool_call:
        asst_content += f"\n<tool>{tool_call}</tool>"
    msgs.append({"role": "assistant", "content": asst_content})
    if tool_result is not None:
        msgs.append({"role": "tool", "content": tool_result})
    msgs.append({"role": "assistant", "content": final_answer})
    return {"messages": msgs, "metadata": {"category": "arithmetic", "pipeline": "sft_v3"}}


# ── Assertion 1: System prompt length ────────────────────────────────────────

def test_valid_row_passes():
    import validate_sft_data as v
    row = _make_row()
    ok, reason = v.validate_row(row)
    assert ok, f"Expected valid row to pass, got: {reason}"


def test_long_system_prompt_fails():
    import validate_sft_data as v
    leaked = " ".join(["word"] * 60)
    row = _make_row(system=leaked)
    ok, reason = v.validate_row(row)
    assert not ok
    assert "system_prompt" in reason or "constitution" in reason.lower()


def test_exactly_50_word_system_passes():
    import validate_sft_data as v
    system = " ".join(["word"] * 50)
    row = _make_row(system=system)
    ok, _ = v.validate_row(row)
    assert ok


# ── Assertion 2: Think block length ──────────────────────────────────────────

def test_short_think_block_fails():
    import validate_sft_data as v
    row = _make_row(think_content="ok")
    ok, reason = v.validate_row(row)
    assert not ok
    assert "think" in reason.lower()


def test_missing_think_fails():
    import validate_sft_data as v
    row = _make_row()
    row["messages"][2]["content"] = "<answer>42</answer>"
    ok, reason = v.validate_row(row)
    assert not ok


# ── Assertion 3: Banned placeholders ─────────────────────────────────────────

def test_banned_placeholder_see_answer_below_fails():
    import validate_sft_data as v
    row = _make_row(think_content="Core truth: see answer below. " + "x" * 180)
    ok, reason = v.validate_row(row)
    assert not ok
    assert "placeholder" in reason.lower() or "banned" in reason.lower()


def test_banned_placeholder_none_flagged_fails():
    import validate_sft_data as v
    row = _make_row(think_content="Assumptions: none flagged. " + "x" * 180)
    ok, reason = v.validate_row(row)
    assert not ok


def test_clean_think_passes():
    import validate_sft_data as v
    row = _make_row(think_content="I need to compute 6 times 7. I have python_execute. " + "x" * 150)
    ok, _ = v.validate_row(row)
    assert ok


# ── Assertion 4: Tool sequence integrity ─────────────────────────────────────

def test_tool_call_followed_by_tool_role_passes():
    import validate_sft_data as v
    row = _make_row(
        tool_call="python_execute(code='print(42)')",
        tool_result="42",
    )
    ok, _ = v.validate_row(row)
    assert ok


def test_tool_call_not_followed_by_tool_role_fails():
    import validate_sft_data as v
    row = _make_row(
        tool_call="python_execute(code='print(42)')",
        tool_result=None,
    )
    ok, reason = v.validate_row(row)
    assert not ok
    assert "sequence" in reason.lower() or "tool" in reason.lower()


# ── Assertion 5: End-to-end resolution ───────────────────────────────────────

def test_missing_answer_tag_fails():
    import validate_sft_data as v
    row = _make_row(final_answer="The answer is 42.")
    ok, reason = v.validate_row(row)
    assert not ok
    assert "answer" in reason.lower()


def test_last_message_not_assistant_fails():
    import validate_sft_data as v
    row = _make_row()
    row["messages"].append({"role": "user", "content": "Thanks!"})
    ok, reason = v.validate_row(row)
    assert not ok


# ── Drop rate calculation ─────────────────────────────────────────────────────

def test_validate_file_drop_rate():
    import validate_sft_data as v
    rows = [_make_row() for _ in range(95)] + [
        _make_row(system=" ".join(["word"] * 60)) for _ in range(5)
    ]
    valid, invalid, _ = v.validate_rows(rows)
    assert len(valid) == 95
    assert len(invalid) == 5
    drop_rate = len(invalid) / len(rows)
    assert drop_rate < 0.06
