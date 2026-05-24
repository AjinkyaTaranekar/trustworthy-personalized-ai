"""Tests for multi-model benchmark helpers in 4_benchmark.py."""
import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Stub optional deps so the benchmark module loads cleanly
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
# _derive_label
# ---------------------------------------------------------------------------

class TestDeriveLabel:
    def test_uses_explicit_label_when_provided(self):
        assert bm._derive_label("./models/sft", ["vanilla", "sft"], 1) == "sft"

    def test_uses_path_stem_when_no_labels(self):
        assert bm._derive_label("./models/checkpoint_sft", None, 0) == "checkpoint_sft"

    def test_uses_path_stem_when_labels_list_too_short(self):
        assert bm._derive_label("./models/checkpoint_sft_v2", ["vanilla"], 1) == "checkpoint_sft_v2"

    def test_handles_huggingface_id(self):
        label = bm._derive_label("unsloth/Qwen3-0.6B", None, 0)
        assert label == "Qwen3-0.6B"

    def test_handles_trailing_slash(self):
        label = bm._derive_label("./models/sft/", None, 0)
        assert label == "sft"


# ---------------------------------------------------------------------------
# _swap_model
# ---------------------------------------------------------------------------

class TestSwapModel:
    def test_posts_to_swap_endpoint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "model": "checkpoint_sft"}
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = bm._swap_model(
                "http://localhost:8000", "./models/checkpoint_sft", "unsloth/Qwen3-0.6B"
            )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert body["model_dir"] == "./models/checkpoint_sft"
        assert body["base_model"] == "unsloth/Qwen3-0.6B"
        assert body["reset_metrics"] is True
        assert result == "checkpoint_sft"

    def test_uses_300s_timeout(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "model": "m"}
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response) as mock_post:
            bm._swap_model("http://localhost:8000", "./models/m", "base")

        call_kwargs = mock_post.call_args
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs.args[-1]
        assert timeout >= 300

    def test_returns_model_label_from_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "model": "my-label"}
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            label = bm._swap_model("http://localhost:8000", "./models/x", "base")

        assert label == "my-label"


# ---------------------------------------------------------------------------
# Sample results fixture
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = {
    "vanilla": {
        "constitution": {
            "constitution_score": 0.600,
            "scores_by_principle": {
                "P1_decompose_first": 0.500,
                "P4_math_code":       0.667,
                "P7_uncertainty":     0.833,
            },
        }
    },
    "sft": {
        "constitution": {
            "constitution_score": 0.750,
            "scores_by_principle": {
                "P1_decompose_first": 0.667,
                "P4_math_code":       0.833,
                "P7_uncertainty":     1.000,
            },
        }
    },
}


# ---------------------------------------------------------------------------
# print_multi_comparison
# ---------------------------------------------------------------------------

class TestPrintMultiComparison:
    def test_prints_without_error(self, capsys):
        bm.print_multi_comparison(_SAMPLE_RESULTS)
        out = capsys.readouterr().out
        assert "vanilla" in out
        assert "sft" in out

    def test_output_contains_all_principle_ids(self, capsys):
        bm.print_multi_comparison(_SAMPLE_RESULTS)
        out = capsys.readouterr().out
        assert "P1_decompose_first" in out
        assert "P4_math_code" in out
        assert "P7_uncertainty" in out

    def test_shows_delta_for_two_models(self, capsys):
        bm.print_multi_comparison(_SAMPLE_RESULTS)
        out = capsys.readouterr().out
        # Delta for P1: 0.667 - 0.500 = +0.167
        assert "+0.167" in out

    def test_shows_overall_row(self, capsys):
        bm.print_multi_comparison(_SAMPLE_RESULTS)
        out = capsys.readouterr().out
        assert "OVERALL" in out

    def test_handles_empty_results_gracefully(self, capsys):
        bm.print_multi_comparison({"a": {}, "b": {}})
        out = capsys.readouterr().out
        assert "No constitutional scores" in out


# ---------------------------------------------------------------------------
# _save_comparison_csv
# ---------------------------------------------------------------------------

class TestSaveComparisonCsv:
    def test_creates_csv_file(self, tmp_path):
        out = tmp_path / "comparison.csv"
        bm._save_comparison_csv(_SAMPLE_RESULTS, out)
        assert out.exists()

    def test_csv_has_correct_headers(self, tmp_path):
        out = tmp_path / "comparison.csv"
        bm._save_comparison_csv(_SAMPLE_RESULTS, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "principle_id" in headers
        assert "vanilla" in headers
        assert "sft" in headers
        assert "delta_sft" in headers

    def test_csv_contains_principle_rows(self, tmp_path):
        out = tmp_path / "comparison.csv"
        bm._save_comparison_csv(_SAMPLE_RESULTS, out)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        pids = [r["principle_id"] for r in rows]
        assert "P1_decompose_first" in pids
        assert "OVERALL" in pids

    def test_csv_delta_is_b_minus_a(self, tmp_path):
        out = tmp_path / "comparison.csv"
        bm._save_comparison_csv(_SAMPLE_RESULTS, out)
        with open(out, newline="") as f:
            rows = {r["principle_id"]: r for r in csv.DictReader(f)}
        p1 = rows["P1_decompose_first"]
        expected_delta = round(float(p1["sft"]) - float(p1["vanilla"]), 4)
        assert float(p1["delta_sft"]) == pytest.approx(expected_delta, abs=1e-4)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.csv"
        bm._save_comparison_csv(_SAMPLE_RESULTS, out)
        assert out.exists()
