# Multi-Model Benchmark with Hot-Swap

**Date:** 2026-05-24
**Status:** Approved

## Summary

Add a multi-model benchmarking flow where `4_benchmark.py` drives the inference server to sequentially load, benchmark, and compare N models — all without stopping either process. The inference server gains a `/v1/model/swap` endpoint; the benchmark gains a `--models` flag. The existing single-model flow is unchanged.

## Problem

Running vanilla vs SFT comparisons currently requires: stop inference server, restart with new model, rerun benchmark, manually compare JSONs. This is error-prone and slow for multi-model experiments.

## Proposed Flow

```
python 4_benchmark.py \
    --models ./models/vanilla ./models/checkpoint_sft \
    --probe \
    --server_url http://localhost:8000

For each model in --models:
  1. POST /v1/model/swap     → inference server unloads current model, loads new one (~30s)
  2. POST /metrics/reset     → triggered automatically by swap (reset_metrics=True)
  3. Run selected suites     → probe / categories / drift / adversarial (same as today)
  4. Save per-model JSON     → reports/probe_<label>_<ts>.json

After all models:
  Print N-column comparison table (generalised compare_runs.py logic)
  Save comparison CSV → reports/comparison_<ts>.csv
```

Model label defaults to the path stem (e.g. `./models/checkpoint_sft` → `checkpoint_sft`). Overridable via `--labels vanilla sft`.

## Inference Server Changes — `3_infererence.py`

### New Pydantic model

```python
class ModelSwapRequest(BaseModel):
    model_dir: str
    base_model: str = "unsloth/Qwen3-0.6B"
    gguf: Optional[str] = None
    max_seq_length: int = 4096
    reset_metrics: bool = True
```

### New endpoint

```python
@app.post("/v1/model/swap")
def swap_model(req: ModelSwapRequest) -> Dict[str, Any]:
    global _MODEL, _TOKENIZER, _MODEL_LABEL, _USE_GGUF, _GGUF_MODEL
    # 1. Unload
    _MODEL = None
    _TOKENIZER = None
    _GGUF_MODEL = None
    _USE_GGUF = False
    torch.cuda.empty_cache()
    # 2. Optionally reset metrics
    if req.reset_metrics:
        METRICS.reset()
    # 3. Load new model (same logic as main())
    if req.gguf:
        _USE_GGUF = True
        gguf_path = _resolve_gguf_path(req.gguf)
        from llama_cpp import Llama
        _GGUF_MODEL = Llama(model_path=gguf_path, n_ctx=req.max_seq_length, n_gpu_layers=-1, verbose=False)
        _MODEL_LABEL = Path(gguf_path).stem
    else:
        from unsloth import FastModel
        model_path = Path(req.model_dir)
        source = str(model_path) if model_path.exists() else req.base_model
        _MODEL_LABEL = source
        _MODEL, _TOKENIZER = FastModel.from_pretrained(
            model_name=source, max_seq_length=req.max_seq_length, load_in_4bit=True, dtype=None,
        )
        FastModel.for_inference(_MODEL)
    return {"status": "ok", "model": _MODEL_LABEL}
```

**Concurrency:** No lock needed. This is a single-user research tool; the benchmark serialises swap → probe → swap → probe, so no requests arrive during a swap.

**HTTP timeout on caller side:** 300s (model load is ~30s; 300s is a safe ceiling).

## Benchmark Changes — `4_benchmark.py`

### New CLI flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--models` | `str+` | `None` | One or more model dirs/IDs to benchmark in sequence |
| `--labels` | `str+` | `None` | Display labels (defaults to path stem of each model) |
| `--compare_output` | `str` | `reports/comparison_<ts>.csv` | Where to save the comparison CSV |

### Backward compatibility

- `--models` absent → existing behaviour, no swap call made at all.
- `--models` with one entry → swap to that model, run, no comparison table.
- `--models` with 2+ entries → full multi-model loop + comparison.

### New helpers in `4_benchmark.py`

```python
def _swap_model(server_url: str, model_dir: str, base_model: str, max_seq_length: int = 4096) -> str:
    """POST /v1/model/swap and return the new model label."""
    body = {"model_dir": model_dir, "base_model": base_model, "reset_metrics": True,
            "max_seq_length": max_seq_length}
    result = _http(server_url, "/v1/model/swap", "POST", body, timeout=300)
    return result["model"]

def _derive_label(model_path: str, labels: Optional[List[str]], idx: int) -> str:
    if labels and idx < len(labels):
        return labels[idx]
    return Path(model_path).stem or model_path.split("/")[-1] or model_path

def print_multi_comparison(all_results: Dict[str, Dict]) -> None:
    """N-column generalisation of compare_runs.py comparison table."""
    ...
```

### Multi-model loop (inside `main()`)

```python
if args.models:
    all_results = {}
    for i, model_path in enumerate(args.models):
        label = _derive_label(model_path, args.labels, i)
        print(f"\n[{i+1}/{len(args.models)}] Swapping to model: {model_path} (label={label})")
        actual_label = _swap_model(args.server_url, model_path, args.base_model)
        _warmup(args.server_url)

        run_results = {}
        if args.probe or args.probe_only:
            run_results["constitution"] = run_constitution_probes(...)
        if args.categories:
            run_results["categories"] = run_category_probes(...)
        if args.drift:
            run_results["drift"] = run_context_drift_test(...)
        if args.adversarial or args.adversarial_only:
            run_results["adversarial"] = run_adversarial_probes(...)

        _save_results(run_results, label)
        all_results[label] = run_results

    if len(all_results) > 1:
        print_multi_comparison(all_results)
        _save_comparison_csv(all_results, args.compare_output)
```

## Comparison Table (N-model generalisation)

For 2 models: identical to `compare_runs.py` output (Δ column).
For N > 2 models: one score column per model, sorted by principle ID. No delta column; deltas shown relative to first model (baseline).

```
Principle                   vanilla   sft_v1    sft_v2    Δ(v1)   Δ(v2)
────────────────────────   ──────── ──────── ──────── ──────── ────────
P1_decompose_first           0.667    0.833    0.900   +0.167   +0.233
...
OVERALL                      0.712    0.798    0.841   +0.086   +0.129
```

## Files Changed

| File | Change |
|------|--------|
| `pipeline/3_infererence.py` | Add `ModelSwapRequest` + `POST /v1/model/swap` endpoint (~50 lines) |
| `pipeline/4_benchmark.py` | Add `--models`, `--labels`, `--compare_output` flags; multi-model loop; `_swap_model()`, `_derive_label()`, `print_multi_comparison()`, `_save_comparison_csv()` helpers (~120 lines) |
| `pipeline/README.md` | Add multi-model usage example |
| `wiki/sources/code/training-and-benchmark.md` | Update to describe new multi-model flow |

## Out of Scope

- Concurrent multi-model serving (different models on different ports) — not needed for dissertation timeline.
- GRPO model support — current scope is SFT only per research pivot.
- Automatic model discovery (scanning a directory) — explicit paths are clearer for academic reproducibility.

## Usage Examples

```bash
# Compare vanilla base model vs two SFT checkpoints
python 4_benchmark.py \
    --models unsloth/Qwen3-0.6B ./models/checkpoint_sft_v1 ./models/checkpoint_sft_v2 \
    --labels vanilla sft_v1 sft_v2 \
    --probe --categories \
    --server_url http://localhost:8000

# Minimal: just constitutional probes, two models
python 4_benchmark.py \
    --models ./models/vanilla ./models/sft \
    --probe_only \
    --server_url http://localhost:8000
```
