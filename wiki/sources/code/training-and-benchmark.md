---
title: Training and Benchmark Scripts
type: source
kind: code
tags: [code, training, lora, grpo, dapo, benchmark, context-degradation, constitution, drift-detection, small-model]
sources:
  - pipeline/1_dataset_generator.py
  - pipeline/2_model_trainer.py
  - pipeline/3_infererence.py
  - pipeline/4_benchmark.py
  - pipeline/5_context_degradation.py
  - pipeline/experiment0_reasoning_comparison.py
  - pipeline/run_all.sh
  - pipeline/preflight_check.sh
  - README.md
updated: 2026-05-02
status: current
---

# Training & Benchmark Scripts

**The entry-point scripts covering SFT data generation → SFT training → GRPO RL training → inference server → Experiment 0 reasoning comparison → constitutional probe suite → adversarial probe suite → context degradation study. All orchestrated by `run_all.sh`.**

## Scripts at a glance

| Script | Role |
| ------ | ---- |
| `preflight_check.sh` | Pre-flight validation: Python version, packages, API key, file integrity, all 10 security blocker symbols, 7-stage training status. Run first on any new machine. |
| `run_all.sh` | Master orchestration: runs all 8 pipeline stages in order, resumable via `--from N`. |
| `1_dataset_generator.py` | V1 template-based interleaved data. Legacy prototype; superseded by `sft_*.py` for dissertation. |
| `2_model_trainer.py` | **Phase 1: SFT** — LoRA fine-tuning of [[entities/qwen3-0.6b\|Qwen3-0.6B]]. **Phase 2: GRPO** — DAPO-improved RL training with composite constitutional reward. CLI: `--mode {sft,grpo}`, `--reward_type {c,d}`. |
| `3_infererence.py` | **FastAPI inference server.** Loads model once on startup; all evaluation scripts call it over HTTP. 5 built-in tools. Security: AST code sandbox + tool-output injection sanitiser (Blocker 1). Dependency monitor with wellbeing disclosure (Blocker 4). Endpoints: `/health`, `/v1/chat/completions`, `/metrics`, `/dependency/status/{id}`. |
| `4_benchmark.py` | **Benchmark client.** Zero GPU dependency. Three suites: (1) 12-probe constitutional drift suite, (2) 14-turn multi-turn conversation + 6 edge cases, (3) 14-probe adversarial suite (Blocker 3). CLI: `--probe_only`, `--adversarial_only`, `--compare_url`. |
| `experiment0_reasoning_comparison.py` | **Experiment 0** (researchplan.tex Phase 3): compares baseline / CoT / interleaved / ToT on GSM8K + 10 logic puzzles. Must run before GRPO to determine which reasoning format to use. |
| `5_context_degradation.py` | Context-length degradation study. Greedy decoding, 12 turns with known correct answers, plots accuracy-vs-context-token-count. |

## Architecture: server + client

`3_infererence.py` is the hub. Everything else calls it over HTTP, mirroring how Anthropic/OpenAI/DeepSeek separate model serving from evaluation. No model reloading between evaluations.

```bash
# Start server (terminal 1)
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Any evaluation (terminal 2 — no GPU needed)
python pipeline/4_benchmark.py --server_url http://localhost:8000
python pipeline/experiment0_reasoning_comparison.py --server_url http://localhost:8000
python pipeline/5_context_degradation.py --server_url http://localhost:8000
```

## Phase 1: SFT

`2_model_trainer.py --mode sft` trains on `data/train_interleaved.jsonl` (output of `sft_dataset_assembler.py`). LoRA r=16, α=32, 3 epochs, lr=2e-4. Outputs `models/checkpoint_sft/adapter_config.json`.

## Phase 2: GRPO (DAPO improvements)

`2_model_trainer.py --mode grpo` starts from the SFT checkpoint (reference policy anchor) and applies GRPO with DAPO improvements:

- **Token-level loss normalisation (Dr.GRPO)**: divide policy gradient loss by completion length, not sequence count. Eliminates the verbosity bias where longer wrong answers receive smaller penalties.
- **Clip-Higher**: asymmetric ε clipping (ε_low=0.2, ε_high=0.28). Lets high-reward completions update more freely, preventing entropy collapse where all G completions become identical.
- **Dynamic sampling**: skip prompts where all G completions score identically (zero gradient, wasted compute).

**Composite reward (all verifiable — no judge model):**

| Component | Weight | How measured |
|---|---|---|
| Format score | 0.30 | `<think>` + `CAPABILITY_CHECK` + `<answer>` present (regex) |
| Accuracy score | 0.40 | Code executes + answer matches expected (tool execution) |
| Tool integrity | 0.15 | No hallucinated or unavailable tool calls (set diff against profile) |
| Constitution score | 0.15 | P1+P3+P4+P14+P18 rule check (Blocker 2 `rule_check_response`) |

Two ablation conditions:
- **Condition C** (`--reward_type c`): format + accuracy only. Proves whether RL correctness signal matters.
- **Condition D** (`--reward_type d`): full composite. Full thesis contribution.

Reference policy = `checkpoint_sft` (not base model) — this anchors the constitution throughout RL.

## Experiment 0: Reasoning Paradigm Comparison

Required before GRPO by researchplan.tex Phase 3. Compares four strategies on GSM8K (up to 100 questions) + 10 built-in logic puzzles. Result determines which reasoning format becomes the target for GRPO training.

| Strategy | System prompt | Tool profile |
|---|---|---|
| baseline | Direct answer, no think tags | no_tools |
| cot | Step-by-step chain of thought | no_tools |
| interleaved | CAPABILITY_CHECK native format | compute_only |
| tot | Generate 3 candidates, self-rank | no_tools |

Metrics: accuracy, CAPABILITY_CHECK rate, tool use rate, answer-tag rate, latency, tokens generated.

## Constitutional drift detection

The 12 constitutional probes are the monitoring mechanism for GRPO training. All checks are regex/rule-based — a judge model that drifts with the trained model would undermine the measurement.

```bash
# Step 1: after SFT, before any GRPO
python 4_benchmark.py --probe_only --save_as_baseline

# Step 2: after each GRPO checkpoint
python 4_benchmark.py --probe_only --baseline reports/constitution_baseline.json
# → drift_warning: true if constitution_score drops ≥5pp
```

Drift mitigation: (1) rollback → (2) increase `kl_coef` → (3) SFT replay buffer.

## Adversarial probe suite (Blocker 3)

14 probes in three categories. Run on SFT baseline before GRPO; re-run on GRPO checkpoints to detect reward-hacking regression.

| Category | Probes | Tests |
|---|---|---|
| jailbreak | JB1–JB5 | Direct override, DAN persona, system-prompt fishing, harmful roleplay, false authority |
| injection | INJ1–INJ3 | Embedded instruction in structured content, false constitution update, numeric payload |
| regression | REG1–REG6 | Sycophancy under expert pressure, format under pressure, tool hallucination, calibrated confidence ×2, social-pressure honesty |

Score = fraction of attacks resisted. `adversarial_score < 0.8` before GRPO = do not proceed.

## Security blockers summary

All four blockers are verified by `preflight_check.sh` section 9. Each is a structural code check, not a runtime policy:

- **Blocker 1** (`3_infererence.py`, `sft_rejection_sampler.py`, `sft_math_question_generator.py`): AST-based import whitelist + dangerous-builtin block before any `subprocess.run`; `_sanitise_tool_output` strips injection patterns from web content.
- **Blocker 2** (`sft_gold_response_generator.py`): `rule_check_response()` provides deterministic out-of-band checks for P1/P3/P4/P14/P18 before the LLM critique; `_merge_violations()` ensures rule violations survive a `NO_VIOLATIONS` LLM response.
- **Blocker 3** (`4_benchmark.py`): 14-probe adversarial suite; run before GRPO.
- **Blocker 4** (`3_infererence.py`): `DependencyMonitor` tracks interaction frequency + burst patterns, appends wellbeing disclosure when dependency signals detected.

## Ablation A/B/C/D

| Condition | Training | Measures |
|---|---|---|
| A | Base Qwen3-0.6B, no training | Zero-shot floor |
| B | SFT only (`checkpoint_sft`) | Value of constitutional formatting |
| C | SFT → GRPO, format+accuracy only | Value of RL correctness signal |
| D | SFT → GRPO, full composite reward | Full thesis contribution |

The primary thesis argument: D outperforms A on constitutional adherence, accuracy, and sycophancy resistance — at a fraction of the compute of frontier models.

## Hyperparameters

**SFT**: LoRA r=16 α=32, 3 epochs, lr=2e-4, batch=2, grad_accum=4, bf16, adamw_8bit.
**GRPO**: G=8, β=0.001, lr=1e-6, ε_low=0.2, ε_high=0.28, rollout temp=1.0, max_new_tokens=512, 1 epoch.

## Related

- [[sources/code/sft-v2-pipeline]] — data source for SFT training
- [[entities/grpo]] — GRPO + DAPO algorithm notes and hyperparameters
- [[entities/qwen3-0.6b]] — base model
- [[entities/constitution]] — 19 constitutional principles
- [[decisions/2025-10-01-four-module-architecture]] — why SFT+GRPO trains only the Reasoning Module
- [[queries/grpo-and-personalisation-master-plan]] — full build plan including User Modelling stack
- [[experiments/experiment-catalog]] — ablation A/B/C/D + Experiment 0 context

## Raw

- `pipeline/2_model_trainer.py`, `3_infererence.py`, `4_benchmark.py`, `5_context_degradation.py`
- `pipeline/experiment0_reasoning_comparison.py`, `run_all.sh`, `preflight_check.sh`
- `README.md`
