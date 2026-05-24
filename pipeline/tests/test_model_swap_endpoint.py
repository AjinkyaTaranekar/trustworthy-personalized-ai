"""Tests for the swap_model function in 3_infererence.py.

FastAPI and GPU deps are fully stubbed so these tests run without hardware
or the installed FastAPI version. We call swap_model() as a plain Python
function rather than through HTTP since the logic under test is the Python
side (globals management, metrics reset, model loading calls).
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ── Passthrough FastAPI mock ──────────────────────────────────────────────────
# Real FastAPI's route decorators return the original function unchanged.
# We replicate that here so srv.swap_model stays the real callable after
# @app.post("/v1/model/swap") is applied.
class _PassthroughDec:
    def __call__(self, fn):
        return fn


_mock_app = MagicMock()
for _http_method in ("get", "post", "delete", "put"):
    setattr(_mock_app, _http_method, lambda *a, **kw: _PassthroughDec())
_mock_app.add_middleware = MagicMock()
_mock_app.routes = []


class _FakeHTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        self.status_code = status_code
        self.detail = detail


_mock_fastapi_mod = MagicMock()
_mock_fastapi_mod.FastAPI = lambda *a, **kw: _mock_app
_mock_fastapi_mod.HTTPException = _FakeHTTPException

sys.modules["fastapi"] = _mock_fastapi_mod
sys.modules["fastapi.middleware"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()

# ── GPU / optional deps ───────────────────────────────────────────────────────
_mock_torch = MagicMock()
_mock_torch.cuda.empty_cache = MagicMock()
_mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
_mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
sys.modules["torch"] = _mock_torch
sys.modules.setdefault("unsloth", MagicMock())
sys.modules.setdefault("llama_cpp", MagicMock())
sys.modules.setdefault("uvicorn", MagicMock())

# ── Pipeline optional deps (guarded by try/except in the server, but
#    pre-stubbing avoids warning noise in test output) ────────────────────────
for _dep in [
    "user_modelling", "empathy", "ontology_verifier",
    "constitutional_harness", "scratchpad", "user_memory",
    "sft_v3_generator",
]:
    sys.modules.setdefault(_dep, MagicMock())

_mock_pt = MagicMock()
_mock_pt.ToolRegistry = MagicMock(return_value=MagicMock())
sys.modules["pipeline_tools"] = _mock_pt

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load server module ────────────────────────────────────────────────────────
_srv_spec = importlib.util.spec_from_file_location(
    "inference_server",
    Path(__file__).parent.parent / "3_infererence.py",
)
srv = importlib.util.module_from_spec(_srv_spec)
sys.modules["inference_server"] = srv
_srv_spec.loader.exec_module(srv)


# ── Request factory ───────────────────────────────────────────────────────────

def _req(**overrides):
    """Build a fake ModelSwapRequest-like namespace for direct function calls."""
    defaults = dict(
        model_dir="./models/nonexistent",
        base_model="test-base",
        gguf=None,
        max_seq_length=4096,
        reset_metrics=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _call_swap(**overrides):
    mock_fast = MagicMock()
    mock_fast.from_pretrained.return_value = (MagicMock(), MagicMock())
    with patch.dict(sys.modules, {"unsloth": MagicMock(FastModel=mock_fast)}):
        return srv.swap_model(_req(**overrides))


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------

class TestSwapEndpointRegistered:
    def test_swap_model_is_callable(self):
        assert callable(srv.swap_model)

    def test_model_swap_request_class_exists(self):
        assert hasattr(srv, "ModelSwapRequest")


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestSwapReturnValue:
    def test_returns_ok_status(self):
        result = _call_swap()
        assert result["status"] == "ok"

    def test_returns_model_key(self):
        result = _call_swap()
        assert "model" in result

    def test_uses_base_model_when_dir_missing(self):
        result = _call_swap(
            model_dir="/nonexistent/path",
            base_model="unsloth/Qwen3-0.6B",
        )
        assert result["model"] == "unsloth/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Side effects
# ---------------------------------------------------------------------------

class TestSwapSideEffects:
    def test_gguf_global_cleared_on_hf_swap(self):
        srv._GGUF_MODEL = MagicMock()
        _call_swap()
        assert srv._GGUF_MODEL is None

    def test_metrics_reset_when_flag_true(self):
        srv.METRICS.reset()
        srv.METRICS._n = 99
        _call_swap(reset_metrics=True)
        assert srv.METRICS._n == 0

    def test_metrics_preserved_when_flag_false(self):
        srv.METRICS.reset()
        srv.METRICS._n = 42
        _call_swap(reset_metrics=False)
        assert srv.METRICS._n == 42

    def test_cuda_empty_cache_called(self):
        _mock_torch.cuda.empty_cache.reset_mock()
        _call_swap()
        _mock_torch.cuda.empty_cache.assert_called()
