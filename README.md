# Trustworthy Personalised AI

MSc dissertation pipeline — *Architecting Trust and Empathy in Conversational AI* (Trinity College Dublin, CS7CS6).

Trains Qwen3-0.6B (a 0.6-billion-parameter model) to be honest about its own capabilities, use tools correctly, delegate maths to code, and refuse to hallucinate — then evaluates it with a full adversarial test suite.

The thesis claim: a small model with the right **modular architecture** (constitutional SFT → GRPO reinforcement learning → graph-based user modelling → appraisal-conditioned empathy) is more trustworthy and scrutable than a larger monolithic model with no structure.

---

## Before you start — preflight check

Always run this first. It checks Python version, packages, API keys, file integrity, security blockers, and training status in one pass:

```bash
bash pipeline/preflight_check.sh
```

If any `[FAIL]` lines appear, fix them before continuing. `[WARN]` lines are safe to proceed on a development machine; fix them before running on a GPU cluster.

**Two hard dependencies you will need to install:**

```bash
pip install datasets trl
pip install unsloth accelerate bitsandbytes  # GPU training only
```

**API key** (needed for SFT data generation with a critic model):

```bash
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FOUR-MODULE ARCHITECTURE                      │
│  (Pivot 1, October 2025 — Professor Conlan feedback)            │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────────────────────┐   │
│  │  Reasoning       │   │  User Modelling Module           │   │
│  │  Module          │   │  5W+H graph (FalkorDB+Cognee)    │   │
│  │  Qwen3-0.6B      │   │  local MCP server                │   │
│  │  SFT + GRPO      │   │  NOT a neural network            │   │
│  └──────────────────┘   └──────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────────────────────┐   │
│  │  Tool Integration│   │  Generator Module                │   │
│  │  Layer           │   │  Base LLM + prompting + RAG      │   │
│  │  MCP + full logs │   │  No fine-tuning on user data     │   │
│  └──────────────────┘   └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

The **Reasoning Module** is what this pipeline trains. The other three modules are the next build phase (User Modelling stack — see `wiki/queries/grpo-and-personalisation-master-plan.md`).

---

## Repository layout

```
pipeline/
├── preflight_check.sh              Pre-flight validation (run this first)
├── run_all.sh                      Master orchestration — runs every stage in order
│
│   ─── Data generation ───
├── sft_question_generator.py       SFT step 1a — behavioural questions (11 categories)
├── sft_gold_response_generator.py  SFT step 1b — teacher generates + critiques (19 principles)
├── sft_math_question_generator.py  SFT step 2a — math/code questions (7 types)
├── sft_rejection_sampler.py        SFT step 2b — keep only correct code executions
├── sft_dataset_assembler.py        SFT step 3  — merge, filter, train/eval split
├── 1_dataset_generator.py          V1 template-based generator (legacy prototype)
│
│   ─── Training ───
├── 2_model_trainer.py              Phase 1: SFT  |  Phase 2: GRPO (DAPO improvements)
│
│   ─── Inference + evaluation ───
├── 3_infererence.py                FastAPI inference server — loads model once, serves HTTP
├── 4_benchmark.py                  Benchmark client — constitutional probes + adversarial suite
├── 5_context_degradation.py        Context-length degradation study (greedy decoding)
├── experiment0_reasoning_comparison.py  Experiment 0 — CoT/ToT/interleaved/baseline comparison
│
│   ─── Reference ───
├── constitution.md                 The 19 constitutional principles the model is trained on
├── .env.example                    Copy to .env and add your API keys
├── data/                           Generated JSONL datasets (git-ignored)
├── models/                         Saved checkpoints (git-ignored)
└── reports/                        Benchmark JSON + HTML viewer

wiki/                               Living research wiki (Obsidian vault)
docs/                               PDFs, dissertation drafts, literature notes
```

> **3 and 4 are server/client.** Start `3_infererence.py` first (it loads the GPU model), then every other script calls it over HTTP. You never reload the model between runs.

---

## Training pipeline: two phases

### Phase 1 — Supervised Fine-Tuning (SFT)

Teaches the model the constitutional format: `<think>CAPABILITY_CHECK...</think><answer>...</answer>`.

```bash
cd pipeline

# Step 1: Generate SFT data (~$10–15 with Claude Sonnet as generator + critic)
python sft_question_generator.py --output data/questions_partA.jsonl

python sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output    data/train_partA.jsonl \
  --critic_model claude-opus-4-7   # frozen critic prevents self-referential bias

python sft_math_question_generator.py --output data/questions_partB.jsonl

python sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output    data/train_partB.jsonl \
  --use_api_model

python sft_dataset_assembler.py \
  --input_a data/train_partA.jsonl \
  --input_b data/train_partB.jsonl \
  --output  data/train_interleaved.jsonl

# Step 2: Train (requires GPU — ~2–4h on A100)
python 2_model_trainer.py \
  --mode sft \
  --data_dir data \
  --output_dir models \
  --output_name checkpoint_sft

# Step 3: Save SFT constitutional baseline (run BEFORE any GRPO)
python 3_infererence.py --model_dir models/checkpoint_sft --port 8000 &
python 4_benchmark.py --probe_only --save_as_baseline
```

### Phase 2 — GRPO Reinforcement Learning

Improves constitutional adherence using verifiable rewards. Implements [DAPO](https://arxiv.org/abs/2503.14476) improvements over vanilla GRPO (entropy collapse fix, length-bias fix, dynamic sampling).

**Ablation conditions:**

| Condition | Command | What it proves |
|---|---|---|
| A | Base model, no training | Zero-shot baseline |
| B | SFT only | Does constitutional formatting transfer? |
| C | `--reward_type c` | Does RL improve correctness? |
| D | `--reward_type d` | Full thesis contribution |

```bash
# Condition C — format + accuracy rewards only
python 2_model_trainer.py \
  --mode grpo \
  --sft_checkpoint models/checkpoint_sft \
  --reward_type c \
  --output_name checkpoint_grpo_c

# Condition D — full composite reward (format + accuracy + tool integrity + constitution)
python 2_model_trainer.py \
  --mode grpo \
  --sft_checkpoint models/checkpoint_sft \
  --reward_type d \
  --output_name checkpoint_grpo_d
```

**Composite reward (Condition D):**
```
reward = 0.30 × format_score       (CAPABILITY_CHECK + think + answer tags present)
       + 0.40 × accuracy_score     (code execution gives correct answer)
       + 0.15 × tool_integrity     (no hallucinated or unavailable tools)
       + 0.15 × constitution_score (broader rule check: P1 + P3 + P4 + P14 + P18)
```

---

## Run everything at once

On a GPU machine, use the master orchestration script. It checks each stage's output before running — fully resumable if interrupted:

```bash
# Run all stages from scratch
bash pipeline/run_all.sh

# Dry run — see what would happen without executing
bash pipeline/run_all.sh --dry_run

# Resume from a specific stage
bash pipeline/run_all.sh --from 4

# Run only specific stages
bash pipeline/run_all.sh --stages 3,4,5
```

**Stages:**

| # | Stage | Output |
|---|---|---|
| 1 | SFT data check | `data/train_interleaved.jsonl` |
| 2 | SFT training | `models/checkpoint_sft/` |
| 3 | SFT constitutional baseline | `reports/constitution_baseline.json` |
| 4 | Experiment 0 (reasoning comparison) | `reports/experiment0_*.json` |
| 5 | Adversarial baseline | `reports/adversarial_baseline.json` |
| 6 | GRPO Condition C + drift check | `models/checkpoint_grpo_c/` |
| 7 | GRPO Condition D + drift check | `models/checkpoint_grpo_d/` |
| 8 | Final ablation A/B/C/D | `reports/ablation_*_*.json` |

---

## Experiment 0 — Reasoning Paradigm Comparison

Required before GRPO (researchplan.tex Phase 3). Compares four reasoning approaches on GSM8K maths and logic puzzles to determine which format the Reasoning Module should use.

```bash
# Start inference server (any checkpoint or base model)
python pipeline/3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000

# Run all four strategies (100 GSM8K questions)
python pipeline/experiment0_reasoning_comparison.py --n 100

# Quick smoke test
python pipeline/experiment0_reasoning_comparison.py --smoke

# Single strategy
python pipeline/experiment0_reasoning_comparison.py --strategy interleaved --n 50
```

Four strategies: `baseline` (direct answer), `cot` (chain-of-thought), `interleaved` (native CAPABILITY_CHECK format), `tot` (Tree-of-Thoughts: generate 3 candidates, self-rank, pick best).

---

## Evaluation: constitutional probes

Run the 12-question constitutional probe suite after every training phase to catch drift before it compounds.

```bash
# After SFT — save the baseline
python 4_benchmark.py --probe_only --save_as_baseline

# After each GRPO checkpoint — compare against baseline
python 4_benchmark.py --probe_only --baseline reports/constitution_baseline.json
# Prints "DRIFT WARNING" if constitution_score drops ≥5 percentage points
```

Drift mitigation (in order of escalation):
1. Roll back to last good checkpoint
2. Increase the KL coefficient `kl_coef` in `GRPO_CONFIG` in `2_model_trainer.py`
3. Add SFT replay buffer — mix 20% SFT examples into each GRPO batch

---

## Evaluation: adversarial probes

New in this branch. Tests attack resistance across three categories:

```bash
# Run all adversarial probes
python 4_benchmark.py --adversarial_only

# Specific attack type
python 4_benchmark.py --adversarial_only --attack_types jailbreak
python 4_benchmark.py --adversarial_only --attack_types injection
python 4_benchmark.py --adversarial_only --attack_types regression
```

- **Jailbreak (5 probes)**: direct instruction override, DAN persona injection, system-prompt fishing, harmful roleplay, false authority claim
- **Injection (3 probes)**: adversarial instructions embedded in structured content, false constitution update, numeric expression payload
- **Regression (6 probes)**: sycophancy under expert pressure, format preservation under pressure, tool hallucination, confidence calibration (known vs unknown facts), honesty under social pressure

Score = fraction of attacks successfully resisted. Run on SFT checkpoint before GRPO; re-run on GRPO checkpoints to catch reward-hacking regression.

---

## Evaluation: multi-turn benchmark + comparison

```bash
# Standard 14-turn benchmark
python 4_benchmark.py --server_url http://localhost:8000

# Compare base model vs fine-tuned
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001  # second terminal
python 4_benchmark.py --server_url http://localhost:8000 --compare_url http://localhost:8001
```

---

## Context degradation study

Tests whether accuracy holds as conversation history grows. Uses greedy decoding for reproducibility.

```bash
python pipeline/5_context_degradation.py --server_url http://localhost:8000

# Compare base vs fine-tuned
python pipeline/5_context_degradation.py \
    --server_url http://localhost:8000 \
    --compare_url http://localhost:8001
```

---

## Inference server reference

`3_infererence.py` is the central serving component. All other scripts call it over HTTP.

```bash
# Start server
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Key endpoints
GET  /health                       → liveness + model name
GET  /v1/tools                     → list registered tools
POST /v1/chat/completions          → generate (tool loop handled server-side)
GET  /metrics                      → latency p50/p95/p99, throughput, tool call counts
POST /metrics/reset                → reset counters

# Dependency monitoring (Blocker 4 — OWASP LLM09)
GET  /dependency/status/{session_id}    → interaction frequency + disclosure state
POST /dependency/reset/{session_id}     → reset session monitor
```

**Add a tool at runtime (no restart required):**

```bash
curl -X POST http://localhost:8000/v1/tools/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
    "python_code": "def tool_fn(city=\"Dublin\", **_): return f\"Weather for {city}: 12°C, cloudy (mock)\""
  }'
```

---

## Security hardening (this branch)

Four pre-GRPO security blockers are implemented and verified by `preflight_check.sh`:

| Blocker | File | What it does |
|---|---|---|
| 1a — code sandbox | `3_infererence.py` | AST-validates LLM-generated code before `subprocess.run`; blocks `os`, `sys`, `socket`, dangerous builtins |
| 1b — injection sanitiser | `3_infererence.py` | Strips prompt-injection patterns from web/URL tool outputs before injecting into model context |
| 1c/1d — sampler sandbox | `sft_rejection_sampler.py`, `sft_math_question_generator.py` | Same AST validation for verification code |
| 2a — rule verifier | `sft_gold_response_generator.py` | `rule_check_response()`: deterministic P1/P3/P4/P14/P18 checks before LLM critique |
| 2b — violation merge | `sft_gold_response_generator.py` | `_merge_violations()`: rule violations survive even if LLM critic says NO_VIOLATIONS |
| 3a/3b — adversarial suite | `4_benchmark.py` | 14 probes across jailbreak/injection/regression; run before GRPO |
| 4a/4b — dependency monitor | `3_infererence.py` | `DependencyMonitor`: tracks per-session frequency + burst patterns; appends wellbeing disclosure |

---

## Viewing benchmark results

```bash
cd pipeline/reports
python server.py
# Open http://localhost:8000/view_benchmark.html
```

---

## Constitution

The 19 principles in `pipeline/constitution.md` define the model's target behaviour: capability honesty, correct tool use, honest refusal, uncertainty quantification, sycophancy resistance. Read it first to understand what the model is being trained to do and why.

---

## Research wiki

The `wiki/` directory is an Obsidian vault containing the full thesis knowledge graph. Start at `wiki/index.md`. Key entry points:

- `wiki/overview.md` — thesis synthesis and official research question
- `wiki/queries/grpo-and-personalisation-master-plan.md` — full implementation plan for this branch
- `wiki/experiments/experiment-catalog.md` — all six experiments + ablation conditions
- `wiki/decisions/2025-10-01-four-module-architecture.md` — the binding architectural decision
