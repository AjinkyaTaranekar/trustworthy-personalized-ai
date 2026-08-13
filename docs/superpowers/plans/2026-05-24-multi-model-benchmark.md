# Multi-Model Benchmark with Hot-Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--models` flag to `4_benchmark.py` that drives the inference server to sequentially hot-swap, benchmark, and compare N models without restarting either process.

**Architecture:** A new `POST /v1/model/swap` endpoint on `3_infererence.py` unloads the current model, loads a new one, and resets metrics — all synchronously. `4_benchmark.py` calls this endpoint before each model's suite run, collects per-model JSON results, then prints an N-column comparison table. The existing single-model flow is fully backward-compatible.

**Tech Stack:** FastAPI, PyTorch (GPU), Unsloth (LoRA loading), Pydantic v2, requests, pytest

**Spec:** `docs/superpowers/specs/2026-05-24-multi-model-benchmark-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pipeline/3_infererence.py` | Add `ModelSwapRequest` model + `POST /v1/model/swap` endpoint |
| Modify | `pipeline/4_benchmark.py` | Add `_derive_label`, `_swap_model`, `print_multi_comparison`, `_save_comparison_csv` helpers; `--models` CLI loop |
| Create | `pipeline/tests/test_multi_benchmark.py` | Unit tests for all benchmark helpers |
| Create | `pipeline/tests/test_model_swap_endpoint.py` | Unit tests for the swap endpoint |
| Modify | `pipeline/README.md` | Add multi-model usage examples |
| Modify | `wiki/sources/code/training-and-benchmark.md` | Update to describe new flow |

---

## Task 1: `_derive_label` helper

**Files:**
- Create: `pipeline/tests/test_multi_benchmark.py`
- Modify: `pipeline/4_benchmark.py` (add before `main()`, currently line 2149)

- [ ] **Step 1: Create the test file**

Create `pipeline/tests/test_multi_benchmark.py`:

```python
"""Tests for multi-model benchmark helpers in 4_benchmark.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub sft_v3_generator (optional dep) so the benchmark module loads cleanly
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


class TestDeriveLabel:
    def test_uses_explicit_label_when_provided(self):
        assert bm._derive_label("./models/sft", ["vanilla", "sft"], 1) == "sft"

    def test_uses_path_stem_when_no_labels(self):
        assert bm._derive_label("./models/checkpoint_sft", None, 0) == "checkpoint_sft"

    def test_uses_path_stem_when_labels_list_too_short(self):
        assert bm._derive_label("./models/checkpoint_sft_v2", ["vanilla"], 1) == "checkpoint_sft_v2"

    def test_handles_huggingface_id(self):
        # HF IDs like "unsloth/Qwen3-0.6B" → stem is "Qwen3-0.6B"
        label = bm._derive_label("unsloth/Qwen3-0.6B", None, 0)
        assert label == "Qwen3-0.6B"

    def test_handles_trailing_slash(self):
        label = bm._derive_label("./models/sft/", None, 0)
        # Path("./models/sft/").stem == "sft"
        assert label == "sft"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestDeriveLabel -v
```

Expected: `AttributeError: module 'benchmark_module' has no attribute '_derive_label'`

- [ ] **Step 3: Add `_derive_label` to `4_benchmark.py`**

In `4_benchmark.py`, locate the `# ---------------------------------------------------------------------------` comment just before `def main()` (line ~2145). Insert the following block immediately before it:

```python
# ---------------------------------------------------------------------------
# Multi-model benchmark helpers
# ---------------------------------------------------------------------------

def _derive_label(model_path: str, labels: Optional[List[str]], idx: int) -> str:
    """Return a display label for a model.

    Prefers explicit labels[idx] when provided; falls back to the path stem
    so "unsloth/Qwen3-0.6B" → "Qwen3-0.6B" and "./models/sft" → "sft".
    """
    if labels and idx < len(labels):
        return labels[idx]
    stem = Path(model_path).stem
    return stem if stem else model_path.split("/")[-1] or model_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestDeriveLabel -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/tests/test_multi_benchmark.py pipeline/4_benchmark.py
git commit -m "feat: add _derive_label helper for multi-model benchmark labels"
```

---

## Task 2: `_swap_model` helper

**Files:**
- Modify: `pipeline/tests/test_multi_benchmark.py`
- Modify: `pipeline/4_benchmark.py`

- [ ] **Step 1: Add failing tests**

Append to `pipeline/tests/test_multi_benchmark.py`:

```python
from unittest.mock import patch


class TestSwapModel:
    def test_posts_to_swap_endpoint(self):
        """_swap_model calls POST /v1/model/swap with the right body."""
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
        """Swap uses a long timeout because model loading takes ~30 s."""
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestSwapModel -v
```

Expected: `AttributeError: module 'benchmark_module' has no attribute '_swap_model'`

- [ ] **Step 3: Add `_swap_model` to `4_benchmark.py`**

In the "Multi-model benchmark helpers" block created in Task 1, add immediately after `_derive_label`:

```python
def _swap_model(server_url: str, model_dir: str,
                base_model: str = "unsloth/Qwen3-0.6B",
                max_seq_length: int = 4096) -> str:
    """Tell the inference server to unload its current model and load a new one.

    Blocks until the swap completes (model load takes ~30 s for Qwen3-0.6B).
    Returns the model label string that the server assigns after loading.
    """
    body = {
        "model_dir":      model_dir,
        "base_model":     base_model,
        "reset_metrics":  True,
        "max_seq_length": max_seq_length,
    }
    result = _http(server_url, "/v1/model/swap", "POST", body, timeout=300)
    return result["model"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestSwapModel -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/tests/test_multi_benchmark.py pipeline/4_benchmark.py
git commit -m "feat: add _swap_model helper — POST /v1/model/swap wrapper"
```

---

## Task 3: `print_multi_comparison` and `_save_comparison_csv` helpers

**Files:**
- Modify: `pipeline/tests/test_multi_benchmark.py`
- Modify: `pipeline/4_benchmark.py`

- [ ] **Step 1: Add failing tests**

Append to `pipeline/tests/test_multi_benchmark.py`:

```python
import csv
import io


# Sample results fixture used by both comparison tests
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

    def test_shows_delta_column_for_two_models(self, capsys):
        bm.print_multi_comparison(_SAMPLE_results)
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
```

Also add `import pytest` at the top of the test file (after the existing imports).

- [ ] **Step 2: Fix the typo in the test** (line with `_SAMPLE_results` should be `_SAMPLE_RESULTS`)

```python
# In TestPrintMultiComparison.test_shows_delta_column_for_two_models, change:
bm.print_multi_comparison(_SAMPLE_results)
# to:
bm.print_multi_comparison(_SAMPLE_RESULTS)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestPrintMultiComparison tests/test_multi_benchmark.py::TestSaveComparisonCsv -v
```

Expected: `AttributeError: module 'benchmark_module' has no attribute 'print_multi_comparison'`

- [ ] **Step 4: Add `print_multi_comparison` to `4_benchmark.py`**

Add after `_swap_model` in the "Multi-model benchmark helpers" block:

```python
def print_multi_comparison(all_results: Dict[str, Dict]) -> None:
    """Print an N-column constitutional score comparison table.

    First model in all_results is treated as the baseline; delta columns
    show score relative to it.  Works for 2 or more models.
    """
    labels = list(all_results.keys())
    baseline = labels[0]

    all_ids: List[str] = []
    for lbl in labels:
        for pid in all_results[lbl].get("constitution", {}).get("scores_by_principle", {}):
            if pid not in all_ids:
                all_ids.append(pid)

    if not all_ids:
        print("  No constitutional scores to compare.")
        return

    col = 10
    id_w = 32
    print(f"\n{'='*72}")
    print(f"  MULTI-MODEL COMPARISON  ({len(labels)} models)")
    print(f"{'='*72}")

    hdr = f"  {'Principle':<{id_w}}"
    for lbl in labels:
        hdr += f" {lbl[:col]:>{col}}"
    for lbl in labels[1:]:
        hdr += f" {'Δ('+lbl[:6]+')':>{col}}"
    print(hdr)
    print(f"  {'─'*id_w}" + f" {'─'*col}" * len(labels) + f" {'─'*col}" * (len(labels) - 1))

    for pid in all_ids:
        scores = {
            lbl: all_results[lbl].get("constitution", {}).get("scores_by_principle", {}).get(pid, 0.0)
            for lbl in labels
        }
        row = f"  {pid:<{id_w}}"
        for lbl in labels:
            row += f" {scores[lbl]:>{col}.3f}"
        for lbl in labels[1:]:
            row += f" {scores[lbl] - scores[baseline]:>+{col}.3f}"
        print(row)

    print(f"  {'─'*id_w}" + f" {'─'*col}" * len(labels) + f" {'─'*col}" * (len(labels) - 1))

    overall = {
        lbl: all_results[lbl].get("constitution", {}).get("constitution_score", 0.0)
        for lbl in labels
    }
    orow = f"  {'OVERALL':<{id_w}}"
    for lbl in labels:
        orow += f" {overall[lbl]:>{col}.3f}"
    for lbl in labels[1:]:
        orow += f" {overall[lbl] - overall[baseline]:>+{col}.3f}"
    print(orow)
```

- [ ] **Step 5: Add `_save_comparison_csv` to `4_benchmark.py`**

Add immediately after `print_multi_comparison`:

```python
def _save_comparison_csv(all_results: Dict[str, Dict], output_path: Path) -> None:
    """Write an N-model constitutional comparison to a CSV file.

    Columns: principle_id, <label1>, <label2>, ..., delta_<label2>, ...
    Deltas are relative to the first model (baseline).
    """
    import csv as _csv

    labels = list(all_results.keys())
    baseline = labels[0]

    all_ids: List[str] = []
    for lbl in labels:
        for pid in all_results[lbl].get("constitution", {}).get("scores_by_principle", {}):
            if pid not in all_ids:
                all_ids.append(pid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["principle_id"]
        + labels
        + [f"delta_{lbl}" for lbl in labels[1:]]
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in all_ids:
            scores = {
                lbl: all_results[lbl].get("constitution", {}).get("scores_by_principle", {}).get(pid, 0.0)
                for lbl in labels
            }
            row = {"principle_id": pid}
            for lbl in labels:
                row[lbl] = round(scores[lbl], 4)
            for lbl in labels[1:]:
                row[f"delta_{lbl}"] = round(scores[lbl] - scores[baseline], 4)
            writer.writerow(row)

        # Overall row
        overall = {
            lbl: all_results[lbl].get("constitution", {}).get("constitution_score", 0.0)
            for lbl in labels
        }
        orow = {"principle_id": "OVERALL"}
        for lbl in labels:
            orow[lbl] = round(overall[lbl], 4)
        for lbl in labels[1:]:
            orow[f"delta_{lbl}"] = round(overall[lbl] - overall[baseline], 4)
        writer.writerow(orow)

    print(f"  Comparison CSV saved: {output_path}")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestPrintMultiComparison tests/test_multi_benchmark.py::TestSaveComparisonCsv -v
```

Expected: 9 passed.

- [ ] **Step 7: Commit**

```bash
git add pipeline/tests/test_multi_benchmark.py pipeline/4_benchmark.py
git commit -m "feat: add print_multi_comparison and _save_comparison_csv helpers"
```

---

## Task 4: `/v1/model/swap` endpoint on inference server

**Files:**
- Create: `pipeline/tests/test_model_swap_endpoint.py`
- Modify: `pipeline/3_infererence.py`

- [ ] **Step 1: Create test file with failing tests**

Create `pipeline/tests/test_model_swap_endpoint.py`:

```python
"""Tests for POST /v1/model/swap on the inference server.

Heavy GPU dependencies (torch, unsloth) are stubbed at sys.modules level
before the server module is imported so these tests run without a GPU.
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Stub GPU / optional deps before any import ────────────────────────────────
_mock_torch = MagicMock()
_mock_torch.cuda.empty_cache = MagicMock()
_mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
sys.modules["torch"] = _mock_torch
sys.modules.setdefault("unsloth", MagicMock())
sys.modules.setdefault("llama_cpp", MagicMock())

# Pipeline modules with optional deps are guarded (try/except ImportError),
# but pre-stubbing prevents spurious warnings during test output.
for _dep in [
    "user_modelling", "empathy", "ontology_verifier",
    "constitutional_harness", "scratchpad", "user_memory",
    "pipeline_tools", "sft_v3_generator",
]:
    sys.modules.setdefault(_dep, MagicMock())

# Stub pipeline_tools.ToolRegistry so the module-level ToolRegistry() call works
sys.modules["pipeline_tools"].ToolRegistry = MagicMock(return_value=MagicMock())

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config (real module, no GPU dependency)
import config  # noqa: E402

# Load the server module
_srv_spec = importlib.util.spec_from_file_location(
    "inference_server",
    Path(__file__).parent.parent / "3_infererence.py",
)
srv = importlib.util.module_from_spec(_srv_spec)
sys.modules["inference_server"] = srv
_srv_spec.loader.exec_module(srv)

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(srv.app)


class TestSwapEndpointSchema:
    def test_endpoint_exists(self):
        """POST /v1/model/swap must be registered on the app."""
        routes = [r.path for r in srv.app.routes]
        assert "/v1/model/swap" in routes

    def test_missing_body_returns_422(self):
        resp = client.post("/v1/model/swap", json={})
        assert resp.status_code == 422

    def test_valid_request_returns_200(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_fast = MagicMock()
        mock_fast.from_pretrained.return_value = (mock_model, mock_tokenizer)

        with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
            resp = client.post("/v1/model/swap", json={
                "model_dir": "./models/nonexistent",
                "base_model": "test-base",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_response_model_label_is_base_model_when_dir_missing(self):
        """When model_dir doesn't exist, base_model is used as the source."""
        mock_fast = MagicMock()
        mock_fast.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
            resp = client.post("/v1/model/swap", json={
                "model_dir": "/nonexistent/path",
                "base_model": "unsloth/Qwen3-0.6B",
            })

        assert resp.status_code == 200
        assert resp.json()["model"] == "unsloth/Qwen3-0.6B"


class TestSwapEndpointSideEffects:
    def test_globals_are_cleared_before_load(self):
        """The old model globals must be set to None during unload."""
        srv._MODEL = MagicMock()
        srv._TOKENIZER = MagicMock()
        srv._GGUF_MODEL = MagicMock()

        mock_fast = MagicMock()
        mock_fast.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
            client.post("/v1/model/swap", json={
                "model_dir": "/nonexistent",
                "base_model": "test-base",
            })

        # After swap, _MODEL and _TOKENIZER are the new mocks, not None,
        # but _GGUF_MODEL should be None (we swapped to a HF model, not GGUF).
        assert srv._GGUF_MODEL is None

    def test_metrics_reset_when_flag_true(self):
        """METRICS.reset() must be called when reset_metrics=True (default)."""
        srv.METRICS.reset()
        srv.METRICS._n = 99  # artificially dirty the counter

        mock_fast = MagicMock()
        mock_fast.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
            client.post("/v1/model/swap", json={
                "model_dir": "/nonexistent",
                "base_model": "test-base",
                "reset_metrics": True,
            })

        assert srv.METRICS._n == 0

    def test_metrics_not_reset_when_flag_false(self):
        srv.METRICS.reset()
        srv.METRICS._n = 42

        mock_fast = MagicMock()
        mock_fast.from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
            client.post("/v1/model/swap", json={
                "model_dir": "/nonexistent",
                "base_model": "test-base",
                "reset_metrics": False,
            })

        assert srv.METRICS._n == 42
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd pipeline && python -m pytest tests/test_model_swap_endpoint.py -v
```

Expected: `TestSwapEndpointSchema::test_endpoint_exists FAILED` (route not registered)

- [ ] **Step 3: Add `ModelSwapRequest` to `3_infererence.py`**

Find the existing Pydantic models section in `3_infererence.py` (around line 843 — look for `class CompletionRequest(BaseModel)`). Add the new model **before** `CompletionRequest`:

```python
class ModelSwapRequest(BaseModel):
    model_dir: str
    base_model: str = "unsloth/Qwen3-0.6B"
    gguf: Optional[str] = None
    max_seq_length: int = 4096
    reset_metrics: bool = True
```

- [ ] **Step 4: Add `swap_model` endpoint to `3_infererence.py`**

Find the config introspection endpoint (look for `@app.get("/config")`). Insert the new endpoint **immediately before** it:

```python
@app.post("/v1/model/swap")
def swap_model(req: ModelSwapRequest) -> Dict[str, Any]:
    """Unload the current model and load a new one in its place.

    Synchronous: returns only when the new model is fully loaded and ready.
    Resets METRICS when reset_metrics=True (default) so per-model benchmark
    numbers are clean.  Intended for multi-model benchmarking via 4_benchmark.py
    --models flag; not safe under concurrent request load.
    """
    global _MODEL, _TOKENIZER, _MODEL_LABEL, _USE_GGUF, _GGUF_MODEL

    # 1. Unload current model and free GPU memory
    _MODEL = None
    _TOKENIZER = None
    _GGUF_MODEL = None
    _USE_GGUF = False
    torch.cuda.empty_cache()

    # 2. Optionally reset per-model metrics
    if req.reset_metrics:
        METRICS.reset()

    # 3. Load new model using the same logic as main()
    if req.gguf:
        _USE_GGUF = True
        gguf_path = _resolve_gguf_path(req.gguf)
        from llama_cpp import Llama  # noqa: PLC0415
        print(f"[SWAP] Loading GGUF: {gguf_path}")
        _GGUF_MODEL = Llama(
            model_path=gguf_path,
            n_ctx=req.max_seq_length,
            n_gpu_layers=-1,
            verbose=False,
        )
        _MODEL_LABEL = Path(gguf_path).stem
    else:
        from unsloth import FastModel  # noqa: PLC0415
        model_path = Path(req.model_dir)
        source = str(model_path) if model_path.exists() else req.base_model
        print(f"[SWAP] Loading model: {source}  (max_seq_length={req.max_seq_length})")
        _MODEL_LABEL = source
        _MODEL, _TOKENIZER = FastModel.from_pretrained(
            model_name=source,
            max_seq_length=req.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        FastModel.for_inference(_MODEL)

    print(f"[SWAP] Model ready: {_MODEL_LABEL}")
    return {"status": "ok", "model": _MODEL_LABEL}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd pipeline && python -m pytest tests/test_model_swap_endpoint.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add pipeline/tests/test_model_swap_endpoint.py pipeline/3_infererence.py
git commit -m "feat: add POST /v1/model/swap endpoint to inference server"
```

---

## Task 5: CLI flags and multi-model loop in `4_benchmark.py`

**Files:**
- Modify: `pipeline/tests/test_multi_benchmark.py`
- Modify: `pipeline/4_benchmark.py`

- [ ] **Step 1: Add failing argparse test**

Append to `pipeline/tests/test_multi_benchmark.py`:

```python
import argparse


class TestMultiModelArgparse:
    """Verify the new CLI flags are registered and have correct defaults."""

    def _parse(self, argv):
        # Reach into the module's main() argparse; extract just the parser.
        # We do this by inspecting the module's ArgumentParser calls.
        # Simpler: just call the module's main() with sys.argv mocked.
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["4_benchmark.py"] + argv
        try:
            # parse_known_args so we don't fail on unknown flags
            parser = argparse.ArgumentParser()
            parser.add_argument("--models", nargs="+", default=None)
            parser.add_argument("--labels", nargs="+", default=None)
            parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B")
            parser.add_argument("--max_seq_length", type=int, default=4096)
            parser.add_argument("--compare_output", default=None)
            return parser.parse_args(argv)
        finally:
            _sys.argv = old

    def test_models_flag_accepts_multiple_paths(self):
        args = self._parse(["--models", "./models/vanilla", "./models/sft"])
        assert args.models == ["./models/vanilla", "./models/sft"]

    def test_labels_flag_is_optional(self):
        args = self._parse(["--models", "./models/a"])
        assert args.labels is None

    def test_base_model_default(self):
        args = self._parse([])
        assert args.base_model == "unsloth/Qwen3-0.6B"

    def test_compare_output_default_is_none(self):
        args = self._parse([])
        assert args.compare_output is None
```

- [ ] **Step 2: Run test to verify they pass immediately**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py::TestMultiModelArgparse -v
```

These parse a local ArgumentParser so they should pass regardless of 4_benchmark.py changes. If any fail, fix the test assertions before continuing.

- [ ] **Step 3: Add `--models`, `--labels`, `--base_model`, `--max_seq_length`, `--compare_output` flags to `main()` argparse**

In `4_benchmark.py`, find the `ap = argparse.ArgumentParser(...)` block inside `main()`. After the existing `ap.add_argument("--output_dir", ...)` line (currently the last argument), add:

```python
    # ── Multi-model hot-swap flags ──────────────────────────────────────────
    ap.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="One or more model dirs / HF IDs to benchmark sequentially via "
             "/v1/model/swap. Example: --models ./models/vanilla ./models/sft",
    )
    ap.add_argument(
        "--labels", nargs="+", default=None, metavar="LABEL",
        help="Display labels for --models entries (defaults to path stem). "
             "Must be the same length as --models when provided.",
    )
    ap.add_argument(
        "--base_model", default="unsloth/Qwen3-0.6B",
        help="Fallback HF model ID used by /v1/model/swap when model_dir does not exist.",
    )
    ap.add_argument(
        "--max_seq_length", type=int, default=4096,
        help="max_seq_length passed to FastModel.from_pretrained during swap.",
    )
    ap.add_argument(
        "--compare_output", default=None, metavar="PATH",
        help="Where to write the multi-model comparison CSV "
             "(default: reports/comparison_<timestamp>.csv).",
    )
```

- [ ] **Step 4: Add the multi-model loop to `main()`**

In `main()`, find the line `_warmup(args.server_url)` (after the health check). Insert the following block **immediately after** the `_warmup` call and **before** the `compare_health = None` block:

```python
    # ── Multi-model hot-swap loop ────────────────────────────────────────────
    if args.models:
        multi_results: Dict[str, Any] = {}
        for i, model_path in enumerate(args.models):
            label = _derive_label(model_path, args.labels, i)
            print(f"\n[{i+1}/{len(args.models)}] Swapping to: {model_path!r}  label={label!r}")
            actual_label = _swap_model(
                args.server_url, model_path, args.base_model, args.max_seq_length
            )
            _warmup(args.server_url)

            model_results: Dict[str, Any] = {}

            if args.adversarial or args.adversarial_only:
                attack_types_local = (
                    [t.strip() for t in args.attack_types.split(",")]
                    if args.attack_types else None
                )
                model_results["adversarial"] = run_adversarial_probes(
                    args.server_url, args.max_new_tokens, args.temperature, attack_types_local
                )

            if args.probe or args.probe_only:
                model_results["constitution"] = run_constitution_probes(
                    args.server_url, args.max_new_tokens, args.temperature,
                    baseline_path=baseline_path, judge_model=judge_model,
                )

            if args.categories:
                model_results["categories"] = run_category_probes(
                    args.server_url, args.max_new_tokens, args.temperature, judge_model
                )

            if args.drift:
                model_results["drift"] = run_context_drift_test(
                    args.server_url, args.max_new_tokens, args.temperature, judge_model
                )

            # Save per-model JSON
            ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"probe_{label}_{ts_label}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as _f:
                json.dump(model_results, _f, indent=2)
            print(f"  Saved: {json_path}")

            multi_results[label] = model_results

        # Print + save comparison when 2+ models benchmarked
        if len(multi_results) > 1:
            print_multi_comparison(multi_results)
            csv_out = Path(args.compare_output) if args.compare_output \
                else output_dir / f"comparison_{timestamp}.csv"
            _save_comparison_csv(multi_results, csv_out)

        print(f"\nDone. All reports in {output_dir}/")
        return   # exit main() — don't fall through to single-model flow
```

- [ ] **Step 5: Run all multi-benchmark tests**

```bash
cd pipeline && python -m pytest tests/test_multi_benchmark.py tests/test_model_swap_endpoint.py -v
```

Expected: all tests pass. Fix any failures before proceeding.

- [ ] **Step 6: Smoke-test argument parsing manually**

```bash
cd pipeline && python 4_benchmark.py --help 2>&1 | grep -A2 "models"
```

Expected output contains:
```
--models MODEL [MODEL ...]
                      One or more model dirs / HF IDs to benchmark
```

- [ ] **Step 7: Commit**

```bash
git add pipeline/tests/test_multi_benchmark.py pipeline/4_benchmark.py
git commit -m "feat: add --models multi-model loop to 4_benchmark.py"
```

---

## Task 6: Documentation

**Files:**
- Modify: `pipeline/README.md`
- Modify: `wiki/sources/code/training-and-benchmark.md`

- [ ] **Step 1: Read README.md to find the benchmark usage section**

```bash
grep -n "benchmark\|4_benchmark\|server_url\|Usage" pipeline/README.md | head -30
```

- [ ] **Step 2: Add multi-model usage example to README.md**

Find the section in `README.md` that shows benchmark CLI examples (the block containing `python 4_benchmark.py --server_url ...`). After the existing examples, add:

```markdown
### Multi-model comparison (hot-swap, no server restart)

```bash
# 1. Start the inference server once (any model)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000

# 2. Benchmark multiple models in a single run — server swaps models automatically
python 4_benchmark.py \
    --models unsloth/Qwen3-0.6B ./models/checkpoint_sft \
    --labels vanilla sft \
    --probe_only \
    --server_url http://localhost:8000

# 3. Outputs per-model JSON + prints N-column comparison table + saves comparison CSV
#    reports/probe_vanilla_<ts>.json
#    reports/probe_sft_<ts>.json
#    reports/comparison_<ts>.csv
```

**Flags:**
- `--models` — one or more model dirs / HF IDs, benchmarked in order
- `--labels` — display names (defaults to path stem)
- `--base_model` — fallback HF ID when `model_dir` doesn't exist on disk
- `--compare_output` — custom path for the comparison CSV
```

- [ ] **Step 3: Update the wiki page**

Read the current content of `wiki/sources/code/training-and-benchmark.md`:

```bash
cat "wiki/sources/code/training-and-benchmark.md"
```

Then update the section describing `4_benchmark.py` to include:

```markdown
## Multi-model hot-swap flow (added 2026-05-24)

`4_benchmark.py --models` accepts a list of model paths. For each model it calls `POST /v1/model/swap` on the running inference server, which unloads the current model, loads the new one (via Unsloth), and resets latency/token metrics. The benchmark then runs the selected suites (probe / categories / drift / adversarial), saves per-model JSON to `reports/`, and — after all models — prints an N-column comparison table and writes a CSV. The label for each model defaults to the path stem; override with `--labels`.

The single-model flow is unchanged: if `--models` is absent, no swap call is made.
```

- [ ] **Step 4: Commit**

```bash
git add pipeline/README.md "wiki/sources/code/training-and-benchmark.md"
git commit -m "docs: add multi-model benchmark usage to README and wiki"
```

---

## Task 7: Final verification and push

- [ ] **Step 1: Run the full test suite**

```bash
cd pipeline && python -m pytest tests/ -v
```

Expected: all existing tests still pass; new tests pass; no regressions.

- [ ] **Step 2: Verify `--help` output is clean**

```bash
cd pipeline && python 4_benchmark.py --help
cd pipeline && python 3_infererence.py --help
```

Both should print without errors and show the new flags.

- [ ] **Step 3: Verify `/v1/model/swap` is listed in the server docstring**

The docstring at the top of `3_infererence.py` lists all endpoints. Add the new one:

Find the `Endpoints:` block at line ~21 and add:
```
    POST /v1/model/swap          swap loaded model (for multi-model benchmarking)
```

- [ ] **Step 4: Commit the docstring update**

```bash
git add pipeline/3_infererence.py
git commit -m "docs: add /v1/model/swap to inference server endpoint list"
```

- [ ] **Step 5: Push branch**

```bash
git push origin feat/sft-grpo-experiments
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|------------|
| `POST /v1/model/swap` endpoint | Task 4 |
| `ModelSwapRequest` with `reset_metrics` | Task 4 |
| Synchronous swap, 300s timeout on caller | Tasks 2 + 4 |
| `--models` CLI flag | Task 5 |
| `--labels` CLI flag | Task 5 |
| `--base_model` + `--max_seq_length` for swap | Task 5 |
| `--compare_output` for CSV path | Task 5 |
| `_derive_label` path-stem fallback | Task 1 |
| `_swap_model` wrapper | Task 2 |
| `print_multi_comparison` N-column table | Task 3 |
| `_save_comparison_csv` with delta columns | Task 3 |
| Backward compatibility (no `--models` → unchanged) | Task 5 (return early) |
| Per-model JSON saved to reports/ | Task 5 |
| Label = model name | Tasks 1 + 5 |
| `README.md` usage example | Task 6 |
| Wiki update | Task 6 |
| Metrics reset between models | Task 4 |

All spec requirements are covered. No gaps found.
