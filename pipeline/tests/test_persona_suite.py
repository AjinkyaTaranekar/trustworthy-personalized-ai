"""Tests for Suite E persona generation in 4_benchmark.py (judging lives in 5_judgement_day.py)."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub optional deps so the benchmark module loads cleanly (mirrors test_multi_benchmark.py)
_PROMPTS_STUB = {"all_tools": "", "compute_only": "", "compute_and_search": "", "no_tools": ""}
sys.modules.setdefault("sft_v3_generator", MagicMock(STUDENT_PROMPTS=_PROMPTS_STUB))
sys.modules.setdefault("litellm", MagicMock())

_spec = importlib.util.spec_from_file_location(
    "benchmark_module",
    Path(__file__).parent.parent / "4_benchmark.py",
)
bm = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_module"] = bm
_spec.loader.exec_module(bm)


# ---------------------------------------------------------------------------
# PERSONAS data integrity
# ---------------------------------------------------------------------------

class TestPersonasData:
    _VALID_PROFILES = {"all_tools", "compute_only", "compute_and_search", "no_tools"}

    def test_personas_nonempty(self):
        assert len(bm.PERSONAS) >= 5

    def test_persona_ids_unique(self):
        ids = [p["persona_id"] for p in bm.PERSONAS]
        assert len(ids) == len(set(ids))

    def test_each_persona_has_required_keys(self):
        for p in bm.PERSONAS:
            assert {"persona_id", "profile", "goal", "tool_profile", "script", "expectations"} <= set(p)

    def test_script_is_nonempty_list_of_strings(self):
        for p in bm.PERSONAS:
            assert isinstance(p["script"], list) and len(p["script"]) >= 2
            assert all(isinstance(t, str) and t.strip() for t in p["script"])

    def test_tool_profile_is_known(self):
        for p in bm.PERSONAS:
            assert p["tool_profile"] in self._VALID_PROFILES


# ---------------------------------------------------------------------------
# _format_persona_transcript (generation-side; the judge reads this text)
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def _records(self):
        return [
            {"turn": 1, "user": "hello there", "think": "reasoning here",
             "tools_called": ["python_execute"], "answer": "hi back"},
            {"turn": 2, "user": "second question", "think": "",
             "tools_called": [], "answer": "second answer"},
        ]

    def test_includes_turn_numbers_users_tools_answers(self):
        out = bm._format_persona_transcript(self._records())
        assert "[Turn 1]" in out and "[Turn 2]" in out
        assert "hello there" in out and "second question" in out
        assert "python_execute" in out and "hi back" in out

    def test_omits_empty_think_and_tools(self):
        out = bm._format_persona_transcript(self._records())
        turn2 = out.split("[Turn 2]")[1]
        assert "<think>" not in turn2 and "tools called" not in turn2


# ---------------------------------------------------------------------------
# run_persona_suite — generation only (no judge)
# ---------------------------------------------------------------------------

class TestRunPersonaSuiteGenerationOnly:
    def _fake_complete(self, *args, **kwargs):
        return {
            "response": "<think>t</think><answer>a</answer>",
            "think_content": "t",
            "answer_content": "a",
            "tool_trace": [{"tool": "python_execute"}],
            "conversation": [],
            "metrics": {},
        }

    def _run(self, **kw):
        with patch.object(bm, "_complete", side_effect=self._fake_complete), \
             patch.object(bm, "_live_init", lambda *a, **k: None), \
             patch.object(bm, "_live", lambda *a, **k: None), \
             patch.object(bm, "_build_run_metadata", lambda *a, **k: {}):
            return bm.run_persona_suite("http://x", **kw)

    def test_generates_all_personas_without_judging(self):
        result = self._run()
        assert result["personas_total"] == len(bm.PERSONAS)
        assert result["persona_score"] is None          # filled later by 5_judgement_day.py
        assert result["personas_judged"] == 0
        assert set(result["dimension_means"]) == set(bm._PERSONA_DIMENSIONS)
        assert all(v is None for v in result["dimension_means"].values())

    def test_records_are_self_contained_for_the_judge(self):
        result = self._run(quick=True)
        rec = result["persona_results"][0]
        # everything 5_judgement_day.py needs is present
        assert rec["transcript"] and rec["profile"] and rec["expectations"]
        assert rec["judge"] is None
        assert len(rec["turns"]) >= 2

    def test_accepts_no_judge_model_kwarg(self):
        # run_persona_suite must no longer accept judge_model (judging moved out)
        import inspect
        assert "judge_model" not in inspect.signature(bm.run_persona_suite).parameters

    def test_judge_symbols_removed_from_benchmark(self):
        for gone in ("_llm_judge", "_batch_judge", "_conversation_judge", "generate_llm_report"):
            assert not hasattr(bm, gone), f"{gone} should have moved to 5_judgement_day.py"
