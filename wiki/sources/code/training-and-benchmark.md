---
title: Training and Benchmark Scripts
type: source
kind: code
tags: [code, training, lora, grpo, dapo, benchmark, context-degradation, constitution, drift-detection, small-model, personalisation, empathy, ontology, tool-use]
sources:
  - pipeline/config.py
  - pipeline/1_dataset_generator.py
  - pipeline/2_model_trainer.py
  - pipeline/3_infererence.py
  - pipeline/4_benchmark.py
  - pipeline/5_context_degradation.py
  - pipeline/experiment0_reasoning_comparison.py
  - pipeline/user_modelling.py
  - pipeline/empathy.py
  - pipeline/appraisal_labeller.py
  - pipeline/ontology_verifier.py
  - pipeline/run_all.sh
  - pipeline/preflight_check.sh
  - docker-compose.yml
  - README.md
updated: 2026-05-02
status: current
---

# Training & Benchmark Scripts

**The full four-module pipeline: SFT data generation → SFT training → GRPO RL training → User Modelling (FalkorDB + 5W+H + scrutability) → Appraisal-conditioned empathy → Ontology post-hoc verifier → Experiment 0 reasoning comparison → constitutional and adversarial probe suites → context degradation study. All feature-flag gated; all orchestrated by `run_all.sh`.**

## Scripts at a glance

| Script | Role |
| ------ | ---- |
| `config.py` | Feature flags singleton — six `ENABLE_*` flags read from `PIPELINE_*` env vars; `validate()` enforces dependency rules; `summary()` for startup logs. |
| `preflight_check.sh` | Pre-flight: Python version, packages, API key, file integrity, security blockers, feature-flag state, 7-stage training status. Run first on any new machine. |
| `run_all.sh` | Master orchestration: Stage 0 (FalkorDB), Stage 0.5 (appraisal labelling), Stages 1–8 (SFT + GRPO + evaluation). Fully resumable. Forwards `PIPELINE_*` env vars to every subprocess. |
| `1_dataset_generator.py` | V1 template-based interleaved data. Legacy prototype. |
| `2_model_trainer.py` | **Phase 1: SFT** — LoRA fine-tuning of [[entities/qwen3-0.6b\|Qwen3-0.6B]]. **Phase 2: GRPO** — DAPO RL training. CLI: `--mode {sft,grpo}`, `--reward_type {c,d}`. |
| `3_infererence.py` | **FastAPI inference server.** Model loaded once. Five built-in tools. Full security hardening (Blockers 1–4). Module lifecycle hooks: write_pipeline → retrieve → analyse_appraisal → generate → onto_score. New endpoints: `/memory/inspect\|contest\|correct`, `/config`. |
| `4_benchmark.py` | **Benchmark client** (zero GPU). Three suites: 12-probe constitutional drift, 14-turn conversation + 6 edge cases, 14-probe adversarial suite. |
| `experiment0_reasoning_comparison.py` | **Experiment 0**: baseline / CoT / interleaved / ToT on GSM8K + logic puzzles. Must run before GRPO. |
| `5_context_degradation.py` | Context-length degradation study. Greedy decoding, 12 turns with known correct answers. |
| `user_modelling.py` | 5W+H FalkorDB graph client; Mem0g 4-stage write pipeline; retrieval gating; scrutability handlers. Loaded by `3_infererence.py` when `ENABLE_USER_MODELLING=true`. |
| `empathy.py` | Runtime appraisal helpers: `analyse_appraisal()`, `format_appraisal_block()`, `parse_appraisal_block()`, `APPRAISAL_SYSTEM_PREFIX`. |
| `appraisal_labeller.py` | **Offline one-time script.** Runs AppraisePLM over EmpatheticDialogues → `data/appraisal_labels.jsonl`. AppraisePLM used as labeller only; not a runtime dependency. |
| `ontology_verifier.py` | Post-hoc claim scorer (Experiment 6 Approach B). Dual backend: local OWL via rdflib or remote SPARQL endpoint. Loaded by `3_infererence.py` when `ENABLE_ONTOLOGY_VERIF=true`. |

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

## Feature flags and module lifecycle

All optional modules are gated by `config.py`. The inference server reads flags at startup via `PIPELINE_*` env vars or a `--config yaml` file. Each module degrades to a neutral stub when its flag is off — the server never crashes on a missing dependency.

Dependency rules enforced by `cfg.validate()`:
- `ENABLE_GRPO` → `checkpoint_sft` must exist
- `ENABLE_PERSONALISATION` → `ENABLE_USER_MODELLING` must be true
- `ENABLE_EMPATHY` → `data/appraisal_labels.jsonl` must exist
- `ENABLE_ONTOLOGY_VERIF` → `ONTOLOGY_PATH` file or `ONTOLOGY_SPARQL_ENDPOINT` must be set

`run_all.sh` forwards all flags to every subprocess so the server, trainer, and benchmark client always run with a consistent configuration.

## `run_all.sh` stage map (updated)

| Stage | Condition | Output |
|---|---|---|
| 0 | `ENABLE_USER_MODELLING=true` | FalkorDB on port 6379 |
| 0.5 | `ENABLE_EMPATHY=true` | `data/appraisal_labels.jsonl` |
| 1–8 | always | SFT data, SFT checkpoint, baselines, GRPO C/D, ablation |

## Related

- [[sources/code/sft-v2-pipeline]] — data source for SFT training
- [[entities/grpo]] — GRPO + DAPO algorithm notes
- [[entities/qwen3-0.6b]] — base model
- [[entities/constitution]] — 19 constitutional principles
- [[topics/personalisation]] — 5W+H schema + Mem0g design + scrutability gap
- [[topics/empathy]] — appraisal-conditioned generation design
- [[topics/ontology-integration]] — Experiment 6 design; Approach A vs B
- [[decisions/2025-10-01-four-module-architecture]] — why the pipeline is split this way
- [[queries/grpo-and-personalisation-master-plan]] — detailed module design rationale
- [[queries/full-pipeline-implementation-plan]] — phase-by-phase build plan for all modules

## Raw

- `pipeline/config.py`, `pipeline/user_modelling.py`, `pipeline/empathy.py`
- `pipeline/appraisal_labeller.py`, `pipeline/ontology_verifier.py`
- `pipeline/2_model_trainer.py`, `3_infererence.py`, `4_benchmark.py`, `5_context_degradation.py`
- `pipeline/experiment0_reasoning_comparison.py`, `run_all.sh`, `preflight_check.sh`
- `docker-compose.yml`, `README.md`
