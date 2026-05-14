"""Unit tests for sft_v3_generator.py helpers — no LLM or network calls."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ── Student prompt length ────────────────────────────────────────────────────

def test_all_student_prompts_under_50_words():
    import sft_v3_generator as gen
    for profile_label, prompt in gen.STUDENT_PROMPTS.items():
        word_count = len(prompt.split())
        assert word_count <= 50, (
            f"Student prompt for '{profile_label}' is {word_count} words (max 50): {prompt!r}"
        )


def test_student_prompts_cover_all_profiles():
    import sft_v3_generator as gen
    for profile in gen.TOOL_PROFILES:
        assert profile["label"] in gen.STUDENT_PROMPTS, (
            f"No student prompt for profile '{profile['label']}'"
        )


# ── Tool executor ─────────────────────────────────────────────────────────────

def test_execute_tool_v3_python_execute():
    import sft_v3_generator as gen
    result = gen._execute_tool_v3(
        tool_inner="python_execute(code='print(2 + 2)')",
        active_tools={"python_execute"},
        failure_config=None,
    )
    assert "4" in result


def test_execute_tool_v3_python_unavailable():
    import sft_v3_generator as gen
    result = gen._execute_tool_v3(
        tool_inner="python_execute(code='print(1)')",
        active_tools=set(),
        failure_config=None,
    )
    assert "not available" in result.lower()


def test_execute_tool_v3_web_search_503_first_call():
    import sft_v3_generator as gen
    fc = {"inject_503": True}
    result = gen._execute_tool_v3(
        tool_inner="web_search(query='current gold price')",
        active_tools={"web_search"},
        failure_config=fc,
    )
    assert "503" in result
    assert fc["web_search_count"] == 1


def test_execute_tool_v3_web_search_503_only_first_call():
    """Second call should NOT return 503."""
    import sft_v3_generator as gen
    fc = {"inject_503": True}
    with patch("sft_v3_generator._exa_search", return_value="Gold: $2300/oz"):
        gen._execute_tool_v3(
            tool_inner="web_search(query='gold price')",
            active_tools={"web_search"},
            failure_config=fc,
        )
        result = gen._execute_tool_v3(
            tool_inner="web_search(query='current gold price site:ft.com')",
            active_tools={"web_search"},
            failure_config=fc,
        )
    assert "Gold" in result or "2300" in result


def test_execute_tool_v3_get_datetime():
    import sft_v3_generator as gen
    result = gen._execute_tool_v3(
        tool_inner="get_datetime()",
        active_tools={"get_datetime"},
        failure_config=None,
    )
    assert "UTC" in result or "202" in result


def test_execute_tool_v3_unknown_tool():
    import sft_v3_generator as gen
    result = gen._execute_tool_v3(
        tool_inner="fly_to_moon(destination='Mars')",
        active_tools={"python_execute"},
        failure_config=None,
    )
    assert "unknown" in result.lower() or "error" in result.lower()


# ── Context swap ─────────────────────────────────────────────────────────────

def _make_conversation(profile_label: str = "compute_only"):
    import sft_v3_generator as gen
    teacher_system = gen._make_teacher_prompt(
        next(p for p in gen.TOOL_PROFILES if p["label"] == profile_label)
    )
    return [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": "What is 7 * 8?"},
        {"role": "assistant", "content": "<think>I need to compute 7 * 8.</think><answer>56</answer>"},
    ]


def test_context_swap_replaces_system_prompt():
    import sft_v3_generator as gen
    conversation = _make_conversation("compute_only")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "compute_only")
    example = gen._build_v3_example(conversation, "What is 7 * 8?", "arithmetic", profile)
    system_msgs = [m for m in example["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == gen.STUDENT_PROMPTS["compute_only"]


def test_context_swap_student_prompt_is_under_50_words():
    import sft_v3_generator as gen
    conversation = _make_conversation("all_tools")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "all_tools")
    example = gen._build_v3_example(conversation, "What is 2+2?", "arithmetic", profile)
    system_msgs = [m for m in example["messages"] if m["role"] == "system"]
    assert len(system_msgs[0]["content"].split()) <= 50


def test_context_swap_teacher_constitution_not_in_output():
    import sft_v3_generator as gen
    conversation = _make_conversation("all_tools")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "all_tools")
    example = gen._build_v3_example(conversation, "q", "arithmetic", profile)
    full_text = json.dumps(example)
    assert "DECOMPOSE FIRST" not in full_text
    assert "TOOL INVENTORY" not in full_text
    assert "25 constitution principles" not in full_text


def test_context_swap_metadata_has_pipeline_v3():
    import sft_v3_generator as gen
    conversation = _make_conversation("compute_only")
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == "compute_only")
    example = gen._build_v3_example(conversation, "q", "arithmetic", profile)
    assert example["metadata"]["pipeline"] == "sft_v3"


# ── Banned placeholder detection ────────────────────────────────────────────

def test_has_banned_placeholder_detects_shortcuts():
    import sft_v3_generator as gen
    assert gen._has_banned_placeholder("see answer below") is True
    assert gen._has_banned_placeholder("inferred from question") is True
    assert gen._has_banned_placeholder("none flagged") is True
    assert gen._has_banned_placeholder("The radius is 4.5 cm.") is False


# ── Think block length ───────────────────────────────────────────────────────

def test_think_block_length_short():
    import sft_v3_generator as gen
    assert gen._think_block_length("<think>ok</think>") < 50


def test_think_block_length_long():
    import sft_v3_generator as gen
    content = "<think>" + "x" * 200 + "</think>"
    assert gen._think_block_length(content) >= 200


def test_think_block_length_absent():
    import sft_v3_generator as gen
    assert gen._think_block_length("<answer>hello</answer>") == 0
