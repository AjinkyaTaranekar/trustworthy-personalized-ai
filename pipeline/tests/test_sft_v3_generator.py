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


# ── Tool registry (via _TOOL_REGISTRY) ────────────────────────────────────────

def test_tool_registry_python_execute():
    import sft_v3_generator as gen
    result = gen._TOOL_REGISTRY.execute(
        tool_inner="python_execute(code='print(2 + 2)')",
        active_tools={"python_execute"},
    )
    assert "4" in result


def test_tool_registry_python_unavailable():
    import sft_v3_generator as gen
    result = gen._TOOL_REGISTRY.execute(
        tool_inner="python_execute(code='print(1)')",
        active_tools=set(),
    )
    assert "not available" in result.lower()


def test_tool_registry_web_search_503_first_call():
    import sft_v3_generator as gen
    fc = {"inject_503": True}
    result = gen._TOOL_REGISTRY.execute(
        tool_inner="web_search(query='current gold price')",
        active_tools={"web_search"},
        failure_config=fc,
    )
    assert "503" in result
    assert fc["web_search_count"] == 1


def test_tool_registry_web_search_503_only_first_call():
    """Second call should NOT return 503."""
    import sft_v3_generator as gen
    from pipeline_tools import web_search as _ws
    fc = {"inject_503": True}
    with patch("pipeline_tools.web_search", return_value="Gold: $2300/oz"):
        gen._TOOL_REGISTRY.execute(
            tool_inner="web_search(query='gold price')",
            active_tools={"web_search"},
            failure_config=fc,
        )
        result = gen._TOOL_REGISTRY.execute(
            tool_inner="web_search(query='current gold price site:ft.com')",
            active_tools={"web_search"},
            failure_config=fc,
        )
    assert "Gold" in result or "2300" in result


def test_tool_registry_get_datetime():
    import sft_v3_generator as gen
    result = gen._TOOL_REGISTRY.execute(
        tool_inner="get_datetime()",
        active_tools={"get_datetime"},
    )
    assert "UTC" in result or "202" in result


def test_tool_registry_unknown_tool():
    import sft_v3_generator as gen
    result = gen._TOOL_REGISTRY.execute(
        tool_inner="fly_to_moon(destination='Mars')",
        active_tools={"python_execute"},
    )
    assert "unknown" in result.lower() or "error" in result.lower()


# ── Context swap ─────────────────────────────────────────────────────────────

def _make_conversation(profile_label: str = "compute_only"):
    import sft_v3_generator as gen
    profile = next(p for p in gen.TOOL_PROFILES if p["label"] == profile_label)
    teacher_system = gen._make_teacher_prompt(
        profile,
        category="arithmetic",
        ideal_behavior=gen._DEFAULT_IDEAL_V3,
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


def test_question_id_uses_existing_id():
    from sft_v3_generator import _question_id
    item = {"id": "q-001", "question": "What is 1+1?", "category": "arithmetic"}
    assert _question_id(item) == "q-001"


def test_question_id_hashes_text_when_no_id():
    from sft_v3_generator import _question_id
    item = {"question": "What is 1+1?", "category": "arithmetic"}
    result = _question_id(item)
    assert len(result) == 12
    assert result == _question_id(item)  # deterministic


def test_question_id_different_for_different_questions():
    from sft_v3_generator import _question_id
    a = _question_id({"question": "Question A"})
    b = _question_id({"question": "Question B"})
    assert a != b


def test_build_v3_example_includes_question_id():
    from sft_v3_generator import _build_v3_example, TOOL_PROFILES
    profile = next(p for p in TOOL_PROFILES if p["label"] == "compute_only")
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>reasoning here and more text to reach length</think><answer>ans</answer>"},
    ]
    example = _build_v3_example(conv, "q", "arithmetic", profile, question_id="abc123def456")
    assert example["metadata"]["question_id"] == "abc123def456"


def test_wrap_missing_think_no_op_when_present():
    from sft_v3_generator import _wrap_missing_think
    content = "<think>Some reasoning here that is long enough.</think><answer>Done.</answer>"
    assert _wrap_missing_think(content) == content


def test_wrap_missing_think_wraps_before_answer():
    from sft_v3_generator import _wrap_missing_think
    reasoning = "x" * 100
    content = f"{reasoning}<answer>Done.</answer>"
    result = _wrap_missing_think(content)
    assert result.startswith("<think>")
    assert reasoning in result
    assert "<answer>Done.</answer>" in result


def test_wrap_missing_think_wraps_before_tool():
    from sft_v3_generator import _wrap_missing_think
    reasoning = "I need to look this up. " * 5
    content = f"{reasoning}<tool>web_search(query='test')</tool>"
    result = _wrap_missing_think(content)
    assert result.startswith("<think>")
    assert "<tool>web_search" in result


def test_wrap_missing_think_skips_short_reasoning():
    from sft_v3_generator import _wrap_missing_think
    content = "Short.<answer>Answer.</answer>"
    assert _wrap_missing_think(content) == content


def test_think_block_length_min_threshold():
    import sft_v3_generator as gen
    short = "<think>Too short.</think><answer>Answer.</answer>"
    long_think = "<think>" + "x" * 200 + "</think><answer>Answer.</answer>"
    assert gen._think_block_length(short) < 150
    assert gen._think_block_length(long_think) >= 150


def test_process_one_v3_rejects_missing_think(monkeypatch):
    """_process_one_v3 must return 'error' when all generation attempts lack <think>."""
    import io
    import threading
    import sft_v3_generator as gen

    no_think_conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Direct answer without thinking tags.<answer>Answer.</answer>"},
    ]
    monkeypatch.setattr(
        "sft_v3_generator._generate_with_intercept",
        lambda **kwargs: no_think_conv,
    )

    buf = io.StringIO()
    lock = threading.Lock()
    item = {"category": "impossible_tasks", "question": "What is 1 + 1?"}
    result = gen._process_one_v3(item, "test_model", None, buf, lock, 1, 1, 0.0)
    assert result == "error"
