"""Tests for analyze_experiments.py — cross-condition benchmark consolidator."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_experiments",
    Path(__file__).parent.parent / "analyze_experiments.py",
)
ae = importlib.util.module_from_spec(_spec)
sys.modules["analyze_experiments"] = ae
_spec.loader.exec_module(ae)


# ---------------------------------------------------------------------------
# Synthetic report fixtures
# ---------------------------------------------------------------------------

def _constitution_report(rule_pass: list, combined: float) -> dict:
    """rule_pass: list of bools across two principle groups."""
    return {
        "constitution_score": combined,
        "scores_by_principle": {"P1_x": combined, "P4_y": combined},
        "probe_results": [
            {"id": "P1_x", "question_results": [
                {"question_idx": i, "rule_passed": rp, "combined_score": 1.0 if rp else 0.0}
                for i, rp in enumerate(rule_pass)]},
        ],
    }


def _write(reports_dir: Path, prefix: str, label: str, ts: str, payload: dict) -> None:
    (reports_dir / f"{prefix}_{label}_{ts}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def reports_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    # base: 1/3 rule pass; sft: 3/3 rule pass
    _write(d, "constitution_probe", "base", "20260101_000000",
           _constitution_report([True, False, False], 0.40))
    _write(d, "constitution_probe", "sft", "20260101_000000",
           _constitution_report([True, True, True], 0.90))
    _write(d, "persona_conversations", "base", "20260101_000000",
           {"persona_score": 0.30, "dimension_means": {d_: 0.3 for d_ in ae.PERSONA_DIMENSIONS},
            "persona_results": [{"persona_id": "p1", "judge": {"overall": 0.3}}]})
    _write(d, "persona_conversations", "sft", "20260101_000000",
           {"persona_score": 0.70, "dimension_means": {d_: 0.7 for d_ in ae.PERSONA_DIMENSIONS},
            "persona_results": [{"persona_id": "p1", "judge": {"overall": 0.7}}]})
    return d


# ---------------------------------------------------------------------------
# find_report
# ---------------------------------------------------------------------------

class TestFindReport:
    def test_finds_latest(self, reports_dir):
        _write(reports_dir, "constitution_probe", "base", "20260202_000000",
               _constitution_report([True, True, True], 0.99))
        p = ae.find_report(reports_dir, "constitution", "base")
        assert "20260202" in p.name  # latest by sorted timestamp

    def test_excludes_derived_artifacts(self, reports_dir):
        (reports_dir / "constitution_probe_base_20260303_000000_rescored.json").write_text("{}")
        p = ae.find_report(reports_dir, "constitution", "base")
        assert not p.stem.endswith("_rescored")

    def test_returns_none_when_missing(self, reports_dir):
        assert ae.find_report(reports_dir, "drift", "base") is None

    def test_finds_per_condition_subdir_layout(self, reports_dir):
        # Runbook layout: reports/<label>/<prefix>_<ts>.json (no label in filename)
        sub = reports_dir / "thinker_executor"
        sub.mkdir()
        _write(sub, "constitution_probe", "anything", "20260601_000000",
               _constitution_report([True, True, True], 0.88))
        p = ae.find_report(reports_dir, "constitution", "thinker_executor")
        assert p is not None and p.parent.name == "thinker_executor"


# ---------------------------------------------------------------------------
# bootstrap_delta
# ---------------------------------------------------------------------------

class TestBootstrapDelta:
    def test_point_delta_matches_means(self):
        a = {"k0": 0.0, "k1": 0.0, "k2": 0.0}
        b = {"k0": 1.0, "k1": 1.0, "k2": 1.0}
        delta, lo, hi = ae.bootstrap_delta(a, b, n=500)
        assert delta == pytest.approx(1.0)
        assert lo <= delta <= hi

    def test_none_without_shared_keys(self):
        assert ae.bootstrap_delta({"a": 1.0}, {"b": 1.0}) is None

    def test_is_deterministic_given_seed(self):
        a = {f"k{i}": float(i % 2) for i in range(10)}
        b = {f"k{i}": float((i + 1) % 2) for i in range(10)}
        r1 = ae.bootstrap_delta(a, b, n=300, seed=7)
        r2 = ae.bootstrap_delta(a, b, n=300, seed=7)
        assert r1 == r2


# ---------------------------------------------------------------------------
# build_ladder
# ---------------------------------------------------------------------------

class TestBuildLadder:
    def test_rule_and_combined_rows(self, reports_dir):
        rows, _ = ae.build_ladder(["base", "sft"], ["constitution"], reports_dir, bootstrap_n=200)
        metrics = [r["metric"] for r in rows]
        assert "constitution (rule)" in metrics
        assert "constitution (combined)" in metrics

    def test_rule_score_from_items(self, reports_dir):
        rows, _ = ae.build_ladder(["base", "sft"], ["constitution"], reports_dir, bootstrap_n=200)
        rule_row = next(r for r in rows if r["metric"] == "constitution (rule)")
        assert rule_row["scores"]["base"] == pytest.approx(1 / 3)
        assert rule_row["scores"]["sft"] == pytest.approx(1.0)

    def test_delta_has_ci_for_paired_items(self, reports_dir):
        rows, _ = ae.build_ladder(["base", "sft"], ["constitution"], reports_dir, bootstrap_n=200)
        rule_row = next(r for r in rows if r["metric"] == "constitution (rule)")
        d = rule_row["deltas"][0]
        assert d["from"] == "base" and d["to"] == "sft"
        assert d["delta"] == pytest.approx(1.0 - 1 / 3)
        assert d["ci"] is not None

    def test_missing_report_yields_none_score(self, reports_dir):
        rows, found = ae.build_ladder(["base", "sft", "ghost"], ["constitution"], reports_dir, 200)
        rule_row = next(r for r in rows if r["metric"] == "constitution (rule)")
        assert rule_row["scores"]["ghost"] is None
        assert "constitution" not in found["ghost"]


# ---------------------------------------------------------------------------
# persona matrix + H3 inventory
# ---------------------------------------------------------------------------

class TestPersonaMatrix:
    def test_dimension_means_per_condition(self, reports_dir):
        m = ae.persona_dimension_matrix(["base", "sft"], reports_dir)
        assert set(m) == set(ae.PERSONA_DIMENSIONS)
        assert m["empathy"]["base"] == pytest.approx(0.3)
        assert m["empathy"]["sft"] == pytest.approx(0.7)


class TestH3Inventory:
    def test_flags_failing_top_principle(self, reports_dir):
        # top rung = "base" here (0.40 < 0.6 on both principles) → both flagged FAIL
        inv = ae.h3_failure_inventory(["sft", "base"], reports_dir)
        flagged = {row["principle"] for row in inv}
        assert "P1_x" in flagged and "P4_y" in flagged
        assert all(row["flag"] in ("FAIL", "REGRESS", "FAIL+REGRESS") for row in inv)

    def test_clean_top_rung_yields_empty(self, reports_dir):
        # top rung = "sft" (0.90 ≥ 0.6, and improves vs base) → nothing flagged
        inv = ae.h3_failure_inventory(["base", "sft"], reports_dir)
        assert inv == []


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

class TestWriters:
    def test_csv_written_with_headers(self, reports_dir, tmp_path):
        rows, _ = ae.build_ladder(["base", "sft"], ["constitution"], reports_dir, 200)
        out = tmp_path / "ladder.csv"
        ae.write_csv(rows, ["base", "sft"], out)
        assert out.exists()
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert "metric" in header and "base" in header and "delta_base__sft" in header

    def test_latex_written(self, reports_dir, tmp_path):
        rows, _ = ae.build_ladder(["base", "sft"], ["constitution"], reports_dir, 200)
        out = tmp_path / "ladder.tex"
        ae.write_latex(rows, ["base", "sft"], out)
        body = out.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in body and r"\toprule" in body

    def test_h3_csv_written(self, reports_dir, tmp_path):
        inv = ae.h3_failure_inventory(["sft", "base"], reports_dir)
        out = tmp_path / "h3.csv"
        ae.write_h3_csv(inv, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Persona dimension correlation (Owen's "overlapping metrics" check)
# ---------------------------------------------------------------------------

class TestPearson:
    def test_perfect_positive(self):
        assert ae._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert ae._pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_too_few_points(self):
        assert ae._pearson([1, 2], [1, 2]) is None

    def test_zero_variance(self):
        assert ae._pearson([5, 5, 5], [1, 2, 3]) is None


class TestDimensionCorrelation:
    def _persona_report(self, scores_per_persona):
        # scores_per_persona: list of {dim: score} dicts
        return {"persona_results": [
            {"persona_id": f"p{i}", "judge": {"overall": 0.5,
             "dimensions": {d: {"score": s.get(d, 0.5), "evidence": ""} for d in ae.PERSONA_DIMENSIONS}}}
            for i, s in enumerate(scores_per_persona)]}

    def test_correlation_pools_and_flags_redundant(self, tmp_path):
        d = tmp_path / "reports"; d.mkdir()
        # personalisation and memory_consistency move together (redundant);
        # empathy moves opposite.
        rows = [{"personalisation": v, "memory_consistency": v, "empathy": 1 - v}
                for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
        _write(d, "persona_conversations", "x", "20260101_000000", self._persona_report(rows))
        matrix, n = ae.persona_dimension_correlation(["x"], d)
        assert n == 5
        assert matrix["personalisation"]["memory_consistency"] == pytest.approx(1.0)
        assert matrix["personalisation"]["empathy"] == pytest.approx(-1.0)
        assert matrix["empathy"]["empathy"] == 1.0

    def test_writer_runs_and_emits_csv(self, tmp_path):
        d = tmp_path / "reports"; d.mkdir()
        rows = [{dim: v for dim in ae.PERSONA_DIMENSIONS} for v in (0.2, 0.4, 0.6, 0.8)]
        _write(d, "persona_conversations", "x", "20260101_000000", self._persona_report(rows))
        matrix, n = ae.persona_dimension_correlation(["x"], d)
        out = tmp_path / "corr.csv"
        ae.write_dimension_correlation(matrix, n, out)
        assert out.exists()
        assert out.read_text(encoding="utf-8").splitlines()[0].startswith("dimension,")

    def test_skips_with_too_few_personas(self, tmp_path, capsys):
        d = tmp_path / "reports"; d.mkdir()
        _write(d, "persona_conversations", "x", "20260101_000000",
               self._persona_report([{dim: 0.5 for dim in ae.PERSONA_DIMENSIONS}]))  # n=1
        matrix, n = ae.persona_dimension_correlation(["x"], d)
        ae.write_dimension_correlation(matrix, n, tmp_path / "corr.csv")
        assert n == 1
        assert "skipped" in capsys.readouterr().out
