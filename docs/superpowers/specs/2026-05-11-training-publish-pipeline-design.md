# Training & Publish Pipeline Design

**Date:** 2026-05-11
**Status:** Approved

## Overview

Extend `2_model_trainer.py` so that after each training phase (SFT or GRPO) it automatically merges LoRA adapters, pushes the full merged model and a GGUF-quantised version to HuggingFace, and computes ROUGE scores against two references. Update `3_infererence.py` to support loading GGUF models via `llama-cpp-python`. Add two new sections to `analysis.ipynb` for ROUGE visualisation and training loss curves.

---

## Section 1 — 90/10 Train/Eval Split

### SFT
In `ModelTrainer.train_sft()`, change `test_size=0.05` → `test_size=0.10`. The eval split is already used for `eval_loss` during training; no other changes needed.

### GRPO
In `ModelTrainer.train_grpo()`, after `build_grpo_dataset()` returns the full dataset, apply `.train_test_split(test_size=0.10, seed=42)`. The `train` subset feeds the `GRPOTrainer`. The `test` subset is held out; after training completes the reward function runs over it to produce a held-out reward score that is written to `reports/rouge_{output_name}.json` alongside the ROUGE scores.

---

## Section 2 — Post-Training Publish Flow

### New `ModelTrainer.publish()` method
Called automatically at the end of `train_sft()` and `train_grpo()`. Controlled by a `--no_publish` CLI flag that skips the whole block (for fast local iteration).

Steps executed in order:

1. **Merge LoRA into base weights** — `model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")` produces a standard safetensors checkpoint with no LoRA dependency.
2. **Push merged model to HuggingFace** — `model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit", token=HF_TOKEN)`. Repo IDs:
   - SFT → `{hf_username}/trustworthy-ai-sft`
   - GRPO-C → `{hf_username}/trustworthy-ai-grpo-c`
   - GRPO-D → `{hf_username}/trustworthy-ai-grpo-d`
   Each run creates a new commit; HuggingFace commit history serves as version control.
3. **Export GGUF** — `model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")`. Saves locally to `models/{output_name}_gguf/`.
4. **Push GGUF to HuggingFace** — `model.push_to_hub_gguf(repo_id, tokenizer, quantization_method="q4_k_m", token=HF_TOKEN)`. Pushed to the same repo as step 2; GGUF file appears alongside safetensors.
5. **Compute and save ROUGE** — see Section 3.

### Configuration
- `HF_TOKEN` read from environment variable (standard HuggingFace convention).
- `--hf_username` CLI arg, default `AjinkyaTaranekar`.
- `--no_publish` CLI flag skips the entire publish block.

---

## Section 3 — ROUGE Evaluation

### References
Two reference sources, both evaluated in `publish()`:

1. **Eval split gold responses** — the 10% held-out SFT JSONL rows. Assistant content extracted from each `messages` list and compared to the model's greedy-decoded output on the same prompt.
2. **Constitution probe baseline** — `reports/constitution_baseline.json` produced by `4_benchmark.py --probe_only --save_as_baseline`. Each probe question's saved baseline response is the reference; the current checkpoint's response is the hypothesis. If this file does not exist at publish time, probe ROUGE is skipped and a warning is printed; the field is set to `null` in the output JSON.

### Metrics
ROUGE-1, ROUGE-2, ROUGE-L using the `rouge-score` package (`rouge_scorer.RougeScorer`). Scores averaged across all pairs; both precision/recall/F1 stored.

### Output
`reports/rouge_{output_name}.json`:
```json
{
  "checkpoint": "checkpoint_sft",
  "eval_split_rouge": {
    "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
  },
  "probe_baseline_rouge": {
    "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
  },
  "grpo_held_out_reward": null
}
```
`grpo_held_out_reward` is populated only after GRPO runs; `null` for SFT.

---

## Section 4 — GGUF Inference in `3_infererence.py`

### New CLI argument
`--gguf <path_or_hf_repo>` — accepts either a local `.gguf` file path or a HuggingFace repo ID. When present, the server uses `llama-cpp-python` for generation; the existing Unsloth/LoRA path is used otherwise.

### Loading
```python
from llama_cpp import Llama
llm = Llama(model_path=gguf_path, n_ctx=4096, n_gpu_layers=-1)
```
`n_gpu_layers=-1` offloads all layers to GPU if available; falls back to CPU automatically.

### Generation
The existing tool loop and all endpoints (`/v1/chat/completions`, `/health`, `/metrics`, tool registry) remain unchanged. The only difference is the generation call: instead of `FastModel.generate()`, the server calls `llm.create_chat_completion(messages, ...)` and maps the response to the same internal format.

### HuggingFace GGUF download
If `--gguf` receives a HuggingFace repo ID (contains `/` but not `.gguf`), the server uses `huggingface_hub.hf_hub_download()` to fetch the GGUF file into the local cache before loading.

---

## Section 5 — Analysis Notebook Additions

Two new sections appended to `pipeline/analysis.ipynb`. Both follow the existing `save_fig()` pattern for SVG + PNG export.

### Section 8 — ROUGE Scores
- Loads all `reports/rouge_*.json`.
- Grouped bar chart: x-axis = checkpoint name, y-axis = F1 score, grouped by ROUGE-1 / ROUGE-2 / ROUGE-L, two sub-charts side by side (eval split vs probe baseline).
- Exports to `exports/17_rouge_scores.svg/.png`.

### Section 9 — Training Loss Curves
- Loads `models/checkpoint_sft/loss_history.json` (SFT) and `models/checkpoint_grpo_*/grpo_loss_history.json` (GRPO, all checkpoints found).
- Line chart per checkpoint: train loss and eval loss over steps (SFT); reward score over steps (GRPO).
- Exports to `exports/18_sft_loss_curves.svg/.png` and `exports/19_grpo_reward_curves.svg/.png`.

---

## Files Changed

| File | Change |
|------|--------|
| `pipeline/2_model_trainer.py` | 90/10 split; `publish()` method; `--hf_username`, `--no_publish` CLI args; ROUGE computation |
| `pipeline/3_infererence.py` | `--gguf` CLI arg; GGUF loading via `llama-cpp-python`; HF download helper |
| `pipeline/analysis.ipynb` | Section 8 (ROUGE) and Section 9 (loss curves) appended |

## Dependencies Added

```
rouge-score
llama-cpp-python
huggingface_hub   # already a transitive dep of transformers; explicit pin for hf_hub_download
```
