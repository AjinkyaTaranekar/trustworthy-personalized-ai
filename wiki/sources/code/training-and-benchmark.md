---
title: Training and Benchmark Scripts
type: source
kind: code
tags: [code, training, lora, benchmark, context-degradation, constitution, drift-detection]
sources:
  - pipeline/1_dataset_generator.py
  - pipeline/2_model_trainer.py
  - pipeline/3_infererence.py
  - pipeline/4_benchmark.py
  - pipeline/5_context_degradation.py
  - README.md
updated: 2026-05-01
status: current
---

# Training & Benchmark Scripts

**The numbered entry points covering data generation → SFT training → inference server → benchmark client → constitutional drift detection → context degradation testing.**

## Scripts at a glance

| Script | Role |
| ------ | ---- |
| `1_dataset_generator.py` | V1 interleaved training set. Superseded by SFT v2 for dissertation purposes. |
| `2_model_trainer.py` | LoRA SFT of [[entities/qwen3-0.6b\|Qwen3-0.6B]]. Outputs `./models/<name>/`. |
| `3_infererence.py` | **FastAPI inference server.** Loads model once; serves via HTTP. 5 built-in tools: `python_execute`, `web_search`, `read_url`, `get_datetime`, `get_exchange_rate`. Dynamic tool registration via `POST /v1/tools/register`. Metrics at `GET /metrics`. |
| `4_benchmark.py` | **HTTP benchmark client.** Zero GPU dependency — calls `3_infererence.py` via HTTP. Two suites: (1) constitutional drift probes and (2) multi-turn conversation benchmark + 6 edge cases. Supports `--compare_url` for base-vs-fine-tuned comparison. |
| `5_context_degradation.py` | **HTTP client** (upgraded). Measures correctness decay as context grows using greedy decoding. 12 turns with known correct answers. Detects tool-mania, coreference failure, needle-in-haystack. Produces degradation curve data (input_tokens vs correct). Use `--compare_url` for base vs fine-tuned. |

## Architecture: server + client

`3_infererence.py` and `4_benchmark.py` form a server/client pair — mirroring how Anthropic, OpenAI, and DeepSeek separate model serving from evaluation. The server loads the model once; the benchmark client calls it over HTTP with no GPU requirement. This enables: running multiple benchmark suites without reloading the model, comparing two models by pointing two server instances at different ports, and adding tools at runtime without restart.

```bash
# Start server (terminal 1)
python 3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Run benchmark (terminal 2 — no GPU needed)
python 4_benchmark.py --server_url http://localhost:8000

# Compare base vs fine-tuned (start a second server on 8001)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001
python 4_benchmark.py --server_url http://localhost:8000 --compare_url http://localhost:8001
```

## Constitutional drift detection

The probe suite runs 12 fixed questions (one per testable constitution principle) against the model via the server. All checks are regex / rule-based — no model judge, which would itself drift. The overall `constitution_score` is compared against a baseline saved right after SFT.

```bash
# Save baseline immediately after SFT (before any GRPO)
python 4_benchmark.py --probe_only --save_as_baseline

# Check drift after each GRPO checkpoint
python 4_benchmark.py --probe_only --baseline reports/constitution_baseline.json
# → drift_warning: true if score drops ≥ 5pp from baseline
```

Drift mitigation responses (escalating): (1) rollback to last good checkpoint, (2) increase KL coefficient β in GRPO config, (3) add SFT replay buffer (20%) to GRPO batches.

## Base model & training config

- Base: `unsloth/Qwen3-0.6B`.
- LoRA: r=16, α=32, 3 epochs, lr=2e-4.
- Inference defaults: `max_new_tokens=1024`, `max_tool_iterations=8`, `temperature=0.7`.

## Reports surface

- Constitutional probe reports: `./reports/constitution_probe_*.json`.
- Benchmark reports: `./reports/benchmark_*.json`.
- Comparison reports: `./reports/benchmark_compare_*.json`.
- Local viewer: `python3 server.py` in `reports/`, open `view_benchmark.html`.

## Ablation conditions

| Condition | Training | Notes |
| --------- | -------- | ----- |
| A | Base Qwen3-0.6B, no training | Baseline — run against port 8001 |
| B | SFT only (`checkpoint_sft`) | Proves SFT format value |
| C | SFT → GRPO, format + accuracy rewards only | Proves RL correctness signal |
| D | SFT → GRPO, all rewards incl. `constitution_score` | Full thesis contribution |

Run probe suite on A, B, C, D to produce the constitutional drift trajectory across training phases — this is a primary thesis experiment.

## Context degradation study (`5_context_degradation.py`)

Separate from the general benchmark by design: uses greedy (deterministic) decoding, has known correct answers per turn, and measures a specific quantity — context token count at first failure. The 12 TURNS overlap ~10 questions with the benchmark but serve a different purpose: plotting accuracy-vs-context-length for the thesis degradation curve.

Key failure modes tested: cross-turn reference, long-range recall, multi-reference (Turn 6 combining values from turns 3 and 5 — historically the most common failure point), coreference resolution, tool-mania detection, and needle-in-haystack at peak context load.

Run against all four ablation conditions (A/B/C/D) to show whether SFT and GRPO training improve degradation tolerance relative to the base model.

## Related

- [[sources/code/sft-v2-pipeline]] — data source for training
- [[entities/grpo]] · [[entities/qwen3-0.6b]] · [[entities/constitution]]
- [[experiments/experiment-catalog]] — where ablation A/B/C/D is referenced
- [[decisions/2026-05-01-constitutional-drift-mitigation]] — design decisions for drift prevention

## Raw

- `pipeline/1_dataset_generator.py`, `2_model_trainer.py`, `3_infererence.py`, `4_benchmark.py`, `5_context_degradation.py`, `README.md`
