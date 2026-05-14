"""Unit tests for scratchpad.py — run with: pytest pipeline/tests/test_scratchpad.py -v"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scratchpad import ScratchpadStore


def test_read_initialises_pad_on_first_call():
    store = ScratchpadStore()
    result = store.read("sess1")
    assert "SCRATCHPAD" in result
    assert "sess1" in result
    assert "CONSTITUTION TLDR" in result
    assert "[CONTEXT]" in result
    assert "[TASKS]" in result
    assert "[NOTES]" in result


def test_constitution_tldr_contains_key_principles():
    store = ScratchpadStore()
    result = store.read("sess1")
    for p in ("P1", "P3", "P5", "P8", "P14", "P18", "P24", "P25"):
        assert p in result, f"Missing {p} from constitution TLDR"


def test_update_context_section():
    store = ScratchpadStore()
    store.read("sess1")
    result = store.update("sess1", "context", "user wants X")
    assert result == "✓ context updated"
    assert "user wants X" in store.read("sess1")


def test_update_tasks_section():
    store = ScratchpadStore()
    store.read("sess1")
    store.update("sess1", "tasks", "1. [YES] do thing")
    assert "1. [YES] do thing" in store.read("sess1")


def test_update_notes_section():
    store = ScratchpadStore()
    store.read("sess1")
    store.update("sess1", "notes", "[CONSTITUTION CHECK] P3 ✓")
    assert "[CONSTITUTION CHECK]" in store.read("sess1")


def test_update_constitution_tldr_is_rejected():
    store = ScratchpadStore()
    store.read("sess1")
    result = store.update("sess1", "constitution_tldr", "overwrite attempt")
    assert "Error" in result
    pad = store.read("sess1")
    assert "overwrite attempt" not in pad
    assert "P1" in pad


def test_update_unknown_section_is_rejected():
    store = ScratchpadStore()
    store.read("sess1")
    result = store.update("sess1", "invalid_section", "content")
    assert "Error" in result


def test_update_initialises_pad_if_not_yet_read():
    store = ScratchpadStore()
    result = store.update("sess2", "context", "hello")
    assert "✓" in result
    assert "hello" in store.read("sess2")


def test_get_section_returns_empty_string_for_unknown_session():
    store = ScratchpadStore()
    assert store.get_section("nonexistent", "tasks") == ""


def test_get_task_status_empty_when_pad_not_initialised():
    store = ScratchpadStore()
    assert store.get_task_status("sess1") == ""


def test_get_task_status_empty_when_tasks_section_is_default():
    store = ScratchpadStore()
    store.read("sess1")
    assert store.get_task_status("sess1") == ""


def test_get_task_status_returns_compact_summary():
    store = ScratchpadStore()
    store.read("sess1")
    store.update("sess1", "tasks",
        "1. [YES] get rate\n2. [YES-NEXT] calculate\n3. [BLOCKED: needs context] advise")
    status = store.get_task_status("sess1")
    assert "TASK STATUS" in status
    assert "1." in status


def test_destroy_resets_session():
    store = ScratchpadStore()
    store.read("sess1")
    store.update("sess1", "context", "some context")
    store.destroy("sess1")
    pad = store.read("sess1")
    assert "(empty)" in pad
    assert "some context" not in pad


def test_multiple_sessions_are_independent():
    store = ScratchpadStore()
    store.update("sessA", "context", "user A content")
    store.update("sessB", "context", "user B content")
    assert "user A content" in store.read("sessA")
    assert "user B content" in store.read("sessB")
    assert "user B content" not in store.read("sessA")
