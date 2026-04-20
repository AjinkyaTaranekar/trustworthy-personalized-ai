---
title: Training and Benchmark Scripts
type: source
kind: code
tags: [code, training, lora, benchmark, context-degradation]
sources:
  - pipeline/1_dataset_generator.py
  - pipeline/2_model_trainer.py
  - pipeline/3_infererence.py
  - pipeline/4_benchmark.py
  - pipeline/5_context_degradation.py
  - README.md
updated: 2026-04-19
status: current
---

# Training & Benchmark Scripts

**The numbered v1 entry points plus the context-degradation evaluator. Together they cover data generation → training → inference → benchmark → multi-turn degradation testing.**

## Scripts at a glance

| Script | Role |
| ------ | ---- |
| `1_dataset_generator.py` | Generates the v1 interleaved training set (`data/train_interleaved.jsonl`). Superseded by the SFT v2 flow for most purposes. |
| `2_model_trainer.py` | LoRA fine-tune of [[entities/qwen3-0.6b\|Qwen3-0.6B]]. Output: `./models/<output_name>/`. |
| `3_infererence.py` | Single-prompt inference; optional `--compare` runs base model alongside. Saves timestamped JSON reports. |
| `4_benchmark.py` | Multi-question benchmark across a model (or `--compare` base vs custom). Per-turn metrics: tokens, generation time, tool calls. Output: `./reports/benchmark_*.json`. |
| `5_context_degradation.py` | Measures correctness decay as context length grows. Shares TURNS with the dataset builder. |

## Base model & training config

- Base: `unsloth/Qwen3-0.6B`.
- Fine-tuning: LoRA (low-rank adapters) — small enough to run on a single GPU, cheap enough to iterate on.
- Inference defaults: `max_new_tokens=2048`, `max_iterations=10` (tool loop), `temperature=0.7`.

## Reports surface

- Inference reports: `./reports/inference_*.json` or `./reports/comparison_*.json`.
- Benchmark reports: `./reports/benchmark_*.json`.
- Local viewer: `python3 server.py` in `reports/`, open `view_benchmark.html`.

## Ablation conditions

| Condition | Base/trained | Notes |
| --------- | ------------ | ----- |
| A | Base Qwen3-0.6B, no training | Baseline |
| B | SFT only (`checkpoint_sft`) | Proves SFT format value |
| C | SFT → GRPO, format + accuracy only | Proves RL correctness signal |
| D | SFT → GRPO, all rewards incl. tool_integrity + behavioural | Full thesis contribution |

**Branch note.** The GRPO RL trainer (`2c_rl_trainer.py` in memory records) lives on a separate branch, not on `main`. The `main` branch in this repo contains the SFT v2 data pipeline + benchmark infrastructure only. Expect to switch branches (or merge) before running Conditions C and D.

## Related

- [[sources/code/sft-v2-pipeline]] — the data source for training
- [[entities/grpo]] · [[entities/qwen3-0.6b]] · [[entities/constitution]]
- [[experiments/experiment-catalog]] — where ablation A/B/C/D is referenced

## Raw

- `pipeline/1_dataset_generator.py`, `2_model_trainer.py`, `3_infererence.py`, `4_benchmark.py`, `5_context_degradation.py`
- `README.md`
