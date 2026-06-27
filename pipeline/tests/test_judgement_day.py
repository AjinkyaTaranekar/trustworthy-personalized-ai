"""Tests for 5_judgement_day.py — offline LLM-as-judge over benchmark reports."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("litellm", MagicMock())

_spec = importlib.util.spec_from_file_location(
    "judgement_day",
    Path(__file__).parent.parent / "5_judgement_day.py",
)
jd = importlib.util.module_from_spec(_spec)
sys.modules["judgement_day"] = jd
_spec.loader.exec_module(jd)


def _mk_litellm_response(content: str) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# _extract_json — tolerant parsing
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain(self):
        assert jd._extract_json('{"score": 1.0}')["score"] == 1.0

    def test_fenced(self):
        assert jd._extract_json('```json\n{"a": 1}\n```')["a"] == 1

    def test_trailing_prose(self):
        # balanced-brace scan should grab the object and ignore the trailing note
        assert jd._extract_json('{"score": 0.5}\n\nNote: partial.')["score"] == 0.5


# ---------------------------------------------------------------------------
# judge_response
# ---------------------------------------------------------------------------

class TestJudgeResponse:
    def test_parses_score(self):
        litellm = sys.modules["litellm"]; litellm.completion.side_effect = None
        litellm.completion.return_value = _mk_litellm_response(
            '{"reasoning": "uses python_execute", "score": 1.0, "passed": true}')
        out = jd.judge_response("q", "r", "P4", "must call python", "judge-x")
        assert out["score"] == 1.0 and out["passed"] is True and "python" in out["reason"]

    def test_failure_scores_none(self):
        litellm = sys.modules["litellm"]
        litellm.completion.side_effect = RuntimeError("boom")
        out = jd.judge_response("q", "r", "P4", "rubric", "judge-x")
        assert out["score"] is None and "judge_failed" in out["reason"]
        litellm.completion.side_effect = None


# ---------------------------------------------------------------------------
# judge_conversation
# ---------------------------------------------------------------------------

_CONV_JSON = json.dumps({
    d: {"evidence": "e", "score": 0.8} for d in jd.PERSONA_DIMENSIONS
} | {"overall": 0.75, "summary": "ok"})


class TestJudgeConversation:
    def test_parses_dimensions_and_overall(self):
        litellm = sys.modules["litellm"]; litellm.completion.side_effect = None
        litellm.completion.return_value = _mk_litellm_response(_CONV_JSON)
        out = jd.judge_conversation("profile", "goal", "exp", "transcript", "judge-x")
        assert out["judge_failed"] is False
        assert out["overall"] == pytest.approx(0.75)
        assert set(out["dimensions"]) == set(jd.PERSONA_DIMENSIONS)

    def test_overall_computed_when_absent(self):
        litellm = sys.modules["litellm"]; litellm.completion.side_effect = None
        body = json.dumps({d: {"score": 1.0} for d in jd.PERSONA_DIMENSIONS} | {"summary": "x"})
        litellm.completion.return_value = _mk_litellm_response(body)
        out = jd.judge_conversation("p", "g", "e", "t", "judge-x")
        assert out["overall"] == pytest.approx(1.0)

    def test_failure_overall_none(self):
        litellm = sys.modules["litellm"]
        litellm.completion.side_effect = RuntimeError("boom")
        out = jd.judge_conversation("p", "g", "e", "t", "judge-x")
        assert out["judge_failed"] is True and out["overall"] is None
        litellm.completion.side_effect = None


# ---------------------------------------------------------------------------
# Aggregation: judge_constitution / judge_persona (patch the judges)
# ---------------------------------------------------------------------------

class TestJudgeConstitution:
    def _report(self):
        return {"probe_results": [
            {"id": "P4_math", "principle": "P4", "question_results": [
                {"question_idx": 0, "question": "2+2?", "response": "...",
                 "rule_score": 1.0, "llm_score": None, "combined_score": None,
                 "judge_principle": "P4", "judge_rubric": "must call python"},
                {"question_idx": 1, "question": "3*3?", "response": "...",
                 "rule_score": 0.0, "llm_score": None, "combined_score": None,
                 "judge_principle": "P4", "judge_rubric": "must call python"},
            ]},
        ]}

    def test_writes_scores_and_aggregates(self):
        rep = self._report()
        with patch.object(jd, "judge_response",
                          side_effect=lambda *a, **k: {"score": 1.0, "passed": True, "reason": "ok"}):
            stats = jd.judge_constitution(rep, "judge-x", lambda: None)
        qrs = rep["probe_results"][0]["question_results"]
        assert qrs[0]["llm_score"] == 1.0
        # judge-primary (default): combined == judge score, rule_score is diagnostic only
        assert qrs[0]["combined_score"] == pytest.approx(1.0)
        assert qrs[1]["combined_score"] == pytest.approx(1.0)
        assert rep["scores_by_principle"]["P4_math"] == pytest.approx(1.0)
        assert rep["constitution_score"] == pytest.approx(1.0)
        assert stats["judged"] == 2 and stats["failed"] == 0

    def test_blend_rule_mode_averages(self):
        rep = self._report()
        jd._BLEND_RULE = True
        try:
            with patch.object(jd, "judge_response",
                              side_effect=lambda *a, **k: {"score": 1.0, "passed": True, "reason": "ok"}):
                jd.judge_constitution(rep, "judge-x", lambda: None)
            qrs = rep["probe_results"][0]["question_results"]
            # blend = (rule + llm)/2 → (1+1)/2 and (0+1)/2
            assert qrs[0]["combined_score"] == pytest.approx(1.0)
            assert qrs[1]["combined_score"] == pytest.approx(0.5)
        finally:
            jd._BLEND_RULE = False


class TestJudgePersona:
    def _report(self):
        return {"persona_results": [
            {"persona_id": "p1", "profile": {"who": "x"}, "goal": "g",
             "expectations": ["e"], "transcript": "T", "judge": None},
            {"persona_id": "p2", "profile": {"who": "y"}, "goal": "g2",
             "expectations": ["e"], "transcript": "T2", "judge": None},
        ]}

    def test_writes_judge_and_aggregates(self):
        rep = self._report()
        fake = {"dimensions": {d: {"score": 0.8, "evidence": "e"} for d in jd.PERSONA_DIMENSIONS},
                "overall": 0.8, "summary": "fine", "judge_failed": False}
        with patch.object(jd, "judge_conversation", side_effect=lambda *a, **k: dict(fake)):
            jd.judge_persona(rep, "judge-x", lambda: None)
        assert rep["persona_score"] == pytest.approx(0.8)
        assert rep["personas_judged"] == 2
        assert rep["dimension_means"]["empathy"] == pytest.approx(0.8)
        assert rep["persona_results"][0]["judge"]["overall"] == 0.8


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_detect_suite(self):
        assert jd.detect_suite(Path("constitution_probe_x_1.json")) == "constitution"
        assert jd.detect_suite(Path("persona_conversations_2.json")) == "persona"
        assert jd.detect_suite(Path("adversarial_1.json")) is None  # never judged here

    def test_discover_respects_labels_and_skips_derived(self, tmp_path):
        (tmp_path / "vanilla_base").mkdir()
        (tmp_path / "vanilla_base" / "constitution_probe_20260101.json").write_text("{}")
        (tmp_path / "vanilla_base" / "constitution_probe_20260101.prejudge.bak").write_text("{}")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "persona_conversations_1.json").write_text("{}")
        found = jd.discover_reports(tmp_path, ["vanilla_base"])
        names = [p.name for p in found]
        assert "constitution_probe_20260101.json" in names
        assert all(".prejudge" not in n for n in names)
        assert all("persona_conversations_1" not in n for n in names)  # 'other' not in labels


# ---------------------------------------------------------------------------
# Phase 2 enrichment + Phase 3 constitution-exposure mode
# ---------------------------------------------------------------------------

class TestEnriched:
    def test_enriched_overrides_rubric_and_adds_anchors(self):
        qr = {"judge_principle": "P5 REAL-TIME HONESTY", "judge_rubric": "embedded terse"}
        principle, rubric, anchors = jd._enriched("P5_realtime_honesty", qr)
        assert rubric != "embedded terse" and "live" in rubric.lower()
        assert "1.0 looks like" in anchors and "REFERENCE" in anchors

    def test_unenriched_falls_back_to_embedded(self):
        qr = {"judge_principle": "X", "judge_rubric": "embedded terse"}
        principle, rubric, anchors = jd._enriched("no_such_id", qr)
        assert rubric == "embedded terse" and anchors == ""


class TestConstitutionMode:
    def _sent(self):  # last user-message content handed to the model
        return sys.modules["litellm"].completion.call_args.kwargs["messages"][1]["content"]

    def _arm(self):
        litellm = sys.modules["litellm"]; litellm.completion.side_effect = None
        litellm.completion.return_value = _mk_litellm_response('{"reasoning": "r", "score": 1.0}')

    def test_full_includes_rubric_and_anchors(self):
        self._arm()
        jd.judge_response("q", "resp", "P5 NAME", "MY_RUBRIC_TEXT", "judge-x",
                          anchors="SCORE ANCHORS:\n  1.0 looks like: x\n", constitution_mode="full")
        sent = self._sent()
        assert "MY_RUBRIC_TEXT" in sent and "SCORE ANCHORS" in sent

    def test_bare_drops_rubric_keeps_principle(self):
        self._arm()
        jd.judge_response("q", "resp", "P5 NAME", "MY_RUBRIC_TEXT", "judge-x", constitution_mode="bare")
        sent = self._sent()
        assert "MY_RUBRIC_TEXT" not in sent and "P5 NAME" in sent

    def test_none_drops_principle_and_rubric(self):
        self._arm()
        jd.judge_response("q", "resp", "P5 NAME", "MY_RUBRIC_TEXT", "judge-x", constitution_mode="none")
        sent = self._sent()
        assert "MY_RUBRIC_TEXT" not in sent and "P5 NAME" not in sent
