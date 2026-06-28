"""Tests for judge_reliability.py — agreement / self-consistency statistics."""
import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "judge_reliability", Path(__file__).parent.parent / "judge_reliability.py")
jr = importlib.util.module_from_spec(_spec)
sys.modules["judge_reliability"] = jr
_spec.loader.exec_module(jr)


class TestContinuous:
    def test_pearson_perfect(self):
        assert jr.pearson([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(1.0)

    def test_pearson_constant_side_is_none(self):
        assert jr.pearson([1, 1, 1], [0, 1, 2]) is None

    def test_spearman_monotonic_not_linear(self):
        # monotonic but non-linear → spearman 1.0, pearson < 1.0
        x = [1, 2, 3, 4]; y = [1, 4, 9, 16]
        assert jr.spearman(x, y) == pytest.approx(1.0)
        assert jr.pearson(x, y) < 1.0

    def test_mae_and_exact(self):
        assert jr.mae([1.0, 0.0], [0.5, 0.0]) == pytest.approx(0.25)
        assert jr.exact_agreement([1.0, 0.5, 0.0], [1.0, 0.5, 1.0]) == pytest.approx(2 / 3)


class TestChanceCorrected:
    def test_kappa_and_ac1_perfect(self):
        a = [1, 1, 0, 0, 1]
        assert jr.cohens_kappa(a, a) == 1.0
        assert jr.gwet_ac1(a, a) == 1.0

    def test_ac1_beats_kappa_under_skew(self):
        # 9/10 agree, both raters almost always "pass" → kappa paradox: low kappa,
        # but high observed agreement → Gwet's AC1 stays high.
        a = [1] * 9 + [0]
        b = [1] * 8 + [0, 1]
        k = jr.cohens_kappa(a, b)
        ac1 = jr.gwet_ac1(a, b)
        assert ac1 > k  # AC1 is the more honest statistic here

    def test_kappa_zero_when_chance(self):
        a = [1, 0, 1, 0]; b = [1, 1, 0, 0]  # independent → ~0
        assert abs(jr.cohens_kappa(a, b)) < 1e-9


class TestKrippendorff:
    def test_perfect_interval(self):
        data = [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]
        assert jr.krippendorff_alpha(data, "interval") == pytest.approx(1.0)

    def test_nominal_perfect(self):
        assert jr.krippendorff_alpha([[1, 1], [0, 0]], "nominal") == pytest.approx(1.0)

    def test_disagreement_lowers_alpha(self):
        agree = jr.krippendorff_alpha([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], "interval")
        disagree = jr.krippendorff_alpha([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], "interval")
        assert disagree < agree

    def test_drops_singleton_units(self):
        # a unit with one rating carries no pairing info and must be ignored, not crash
        assert jr.krippendorff_alpha([[1.0, 1.0], [0.5]], "interval") == pytest.approx(1.0)


class TestReports:
    def test_agreement_report_keys_and_bias(self):
        rep = jr.agreement_report([1.0, 0.0, 0.5], [1.0, 0.5, 0.5])
        assert rep["n"] == 3
        assert rep["judge_bias"] == pytest.approx((2.0 / 3) - 0.5)  # judge a touch lenient
        for key in ("pearson", "spearman", "krippendorff_alpha", "gwet_ac1_pass", "mae"):
            assert key in rep

    def test_agreement_report_drops_missing(self):
        rep = jr.agreement_report([1.0, None, 0.0], [1.0, 0.5, None])
        assert rep["n"] == 1  # only the first pair is fully present

    def test_self_consistency_flip_and_spread(self):
        sc = jr.self_consistency([[1.0, 1.0, 1.0], [0.4, 0.7, 0.5]])
        assert sc["items"] == 2
        assert sc["verdict_flip_rate"] == pytest.approx(0.5)  # second item straddles 0.6
        assert sc["mean_within_item_std"] > 0
