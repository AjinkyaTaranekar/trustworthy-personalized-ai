# Trustworthy Personalised AI

MSc dissertation pipeline — *Architecting Trust and Empathy in Conversational AI* (Trinity College Dublin, CS7CS6).

Trains Qwen3-0.6B (a 0.6-billion-parameter model) to be honest about its own capabilities, use tools correctly, delegate maths to code, and refuse to hallucinate — then evaluates it with a full adversarial test suite.

The thesis claim: a small model with the right **modular architecture** (constitutional SFT → GRPO reinforcement learning → graph-based user modelling → appraisal-conditioned empathy) is more trustworthy and scrutable than a larger monolithic model with no structure.

---

## Before you start — preflight check

Always run this first. It checks Python version, packages, API keys, file integrity, security blockers, feature-flag state, and training status in one pass:

```bash
bash pipeline/preflight_check.sh
```

If any `[FAIL]` lines appear, fix them before continuing. `[WARN]` lines are safe on a development machine; fix before GPU cluster runs.

**Core Python dependencies:**

```bash
pip install datasets trl fastapi uvicorn pydantic requests litellm python-dotenv exa-py
pip install unsloth accelerate bitsandbytes  # GPU training only
pip install rouge-score huggingface_hub      # ROUGE evaluation + HuggingFace publish
pip install rdflib                           # Ontology Verifier (local OWL)
pip install SPARQLWrapper                    # Ontology Verifier (remote endpoint)
pip install falkordb                         # User Modelling graph backend
```

**API key** (needed for SFT data generation with a critic model):

```bash
cp pipeline/.env.example pipeline/.env
# Edit pipeline/.env and set one of the providers below
```

Supported providers via [litellm](https://github.com/BerriAI/litellm) — swap with `--model`:

| Provider | Env var | Model string | Cost |
|---|---|---|---|
| **NVIDIA NIM** ✅ confirmed | `NVIDIA_NIM_API_KEY=nvapi-...` | `nvidia_nim/moonshotai/kimi-k2.6` | Free tier |
| **NVIDIA NIM** ✅ confirmed | `NVIDIA_NIM_API_KEY=nvapi-...` | `nvidia_nim/minimaxai/minimax-m2.7` | Free tier |
| **exa.ai** ✅ web search | `EXA_API_KEY=...` | Used by `sft_v3_generator.py` for live semantic web search | $10 free credits |
| Groq | `GROQ_API_KEY=gsk_...` | `groq/llama-3.3-70b-versatile` | Free tier |
| Anthropic | `ANTHROPIC_API_KEY=sk-ant-...` | `claude-sonnet-4-6` | ~$10–15 for full run |
| OpenAI | `OPENAI_API_KEY=sk-...` | `gpt-4o-mini` | Paid |
| Ollama (local) | `OLLAMA_API_BASE=http://localhost:11434` | `ollama/llama3.2` | Free |

---

## Feature flags

All optional modules are off by default. Enable them via environment variables (prefix `PIPELINE_`) or a YAML config file before running the server or `run_all.sh`.

| Flag | Default | Controls |
|---|---|---|
| `PIPELINE_ENABLE_SFT` | `true` | Phase 1 SFT training — always on, the constitutional baseline |
| `PIPELINE_ENABLE_GRPO` | `false` | Phase 2 GRPO/DAPO RL training; requires `checkpoint_sft` first |
| `PIPELINE_ENABLE_USER_MODELLING` | `false` | FalkorDB 5W+H graph + Mem0g write pipeline + scrutability endpoints |
| `PIPELINE_ENABLE_EMPATHY` | `false` | Appraisal-conditioned generation; requires `data/appraisal_labels.jsonl` |
| `PIPELINE_ENABLE_PERSONALISATION` | `false` | Per-query retrieval gating; requires `ENABLE_USER_MODELLING` |
| `PIPELINE_ENABLE_ONTOLOGY_VERIF` | `false` | Post-hoc SPARQL claim scoring against a loaded OWL ontology |
| `PIPELINE_ENABLE_HARNESS` | `false` | Inference-time constitutional validation loop — checks P1/P3/P4/P18 on every response, retries with corrective prompt on failure, adapts system prompt to reinforce weak principles |

```bash
# Enable flags inline for a single run
PIPELINE_ENABLE_USER_MODELLING=true \
PIPELINE_ENABLE_EMPATHY=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft

# Or via a YAML config file
python pipeline/3_infererence.py --config my_config.yaml
```

**Dependency rules (enforced at startup):**
- `ENABLE_GRPO` requires `ENABLE_SFT` to have produced `models/checkpoint_sft/`
- `ENABLE_PERSONALISATION` requires `ENABLE_USER_MODELLING`
- `ENABLE_EMPATHY` requires `data/appraisal_labels.jsonl` (run `appraisal_labeller.py` first)
- `ENABLE_ONTOLOGY_VERIF` requires an OWL file at `PIPELINE_ONTOLOGY_PATH` or a remote SPARQL endpoint at `PIPELINE_ONTOLOGY_SPARQL_ENDPOINT`

---

## Module prerequisites

### User Modelling (FalkorDB)

Requires Docker. Start before launching the inference server:

```bash
docker compose up -d          # starts FalkorDB on port 6379
docker compose down           # stop (graph data persists)
docker compose down -v        # stop + wipe all graph data
```

### Empathy (AppraisePLM labels)

One-time offline step on CPU. Run before SFT data generation:

```bash
# Clone AppraisePLM (supervisor co-authored paper — Debnath, Graham, Conlan, CoNLL 2025)
git clone https://github.com/alokdebnath/appraise-PLM

# Label EmpatheticDialogues (generates data/appraisal_labels.jsonl)
python pipeline/appraisal_labeller.py --appraise_plm_path appraise-PLM

# Smoke test without the model (pipeline testing only — random vectors)
python pipeline/appraisal_labeller.py --mock_model --smoke
```

### Ontology Verifier (Experiment 6 Approach B)

Either place a valid OWL/RDF file at `data/ontology.owl`, or point to a remote endpoint:

```bash
# Local OWL file (pip install rdflib)
# Place file at: data/ontology.owl

# Remote SPARQL endpoint (pip install SPARQLWrapper)
export PIPELINE_ONTOLOGY_SPARQL_ENDPOINT=https://dbpedia.org/sparql
```

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FOUR-MODULE ARCHITECTURE                         │
│   (Pivot 1, October 2025 — Professor Conlan feedback)               │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────┐    │
│  │  Reasoning Module  │    │  User Modelling Module           │    │
│  │  Qwen3-0.6B        │    │  5W+H graph — FalkorDB           │    │
│  │  SFT + GRPO/DAPO   │    │  Mem0g write pipeline            │    │
│  │  Constitution: 23P │    │  Scrutability layer              │    │
│  │  ENABLE_GRPO       │    │  ENABLE_USER_MODELLING           │    │
│  └────────────────────┘    └──────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────┐    │
│  │  Tool Integration  │    │  Generator Module                │    │
│  │  Layer             │    │  Appraisal-conditioned empathy   │    │
│  │  MCP + full logs   │    │  Ontology post-hoc verifier      │    │
│  │  Dep. monitor      │    │  ENABLE_EMPATHY                  │    │
│  │  Adversarial probes│    │  ENABLE_ONTOLOGY_VERIF           │    │
│  └────────────────────┘    └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

All four modules are now implemented. Feature flags control which are active at runtime — see the **Feature flags** section above.

---

## Repository layout

```
pipeline/
├── preflight_check.sh              Pre-flight validation (run this first)
├── run_all.sh                      Master orchestration — runs every stage in order
├── config.py                       Feature flags singleton — single source of truth
│
│   ─── Data generation ───
├── sft_question_generator.py       SFT step 1a — 13 categories incl. appraisal_empathy + interleaved_tool_reasoning
├── sft_gold_response_generator.py  SFT step 1b — teacher generates + critiques (23 principles)
├── sft_math_pipeline.py            SFT step 2  — math/code questions (7 types) + rejection sampling
├── sft_dataset_assembler.py        SFT step 3  — merge, filter, v2→v3 multi-turn transform, native JSON tools, robustness variants, train/eval split
├── appraisal_labeller.py           Offline: AppraisePLM → EmpatheticDialogues labels
│
│   ─── Training ───
├── 2_model_trainer.py              Phase 1: SFT  |  Phase 2: GRPO (DAPO improvements)  |  Publish: upload checkpoint to HuggingFace
│
│   ─── Inference + evaluation ───
├── 3_infererence.py                FastAPI server — model + all four module hooks
├── 4_benchmark.py                  Constitutional probes + adversarial suite
├── 5_context_degradation.py        Context-length degradation study (greedy decoding)
├── experiment0_reasoning_comparison.py  Experiment 0 — CoT/ToT/interleaved/baseline
│
│   ─── Runtime modules (loaded by 3_infererence.py via feature flags) ───
├── user_modelling.py               FalkorDB 5W+H graph + Mem0g write + scrutability
├── empathy.py                      Appraisal-conditioned generation helpers
├── ontology_verifier.py            Post-hoc SPARQL claim scorer (Experiment 6 Approach B)
│
│   ─── Reference ───
├── constitution.md                 19 constitutional principles
├── .env.example                    Copy to .env and add API keys
├── data/                           Datasets + appraisal_labels.jsonl (git-ignored)
├── models/                         Checkpoints (git-ignored)
└── reports/                        Benchmark JSON reports

docker-compose.yml                  FalkorDB service (required for ENABLE_USER_MODELLING)
wiki/                               Living research wiki (Obsidian vault)
docs/                               PDFs, dissertation drafts, literature notes
```

> **3 and 4 are server/client.** Start `3_infererence.py` first, then every other script calls it over HTTP. You never reload the model between runs.

---

## GPU setup (vast.ai)

Training requires a GPU. The pipeline is tested on RTX 4090 (24 GB VRAM) — ~3 hours per SFT stage. An A100 40 GB is ~2×faster but costs more.

### 1. Rent an instance

- Go to **vast.ai → Search**
- Filter: GPU = `RTX 4090`, Disk ≥ `50 GB`, Reliability ≥ `0.98`
- Sort by **DLPerf** descending — higher score = healthier GPU
- Pick **On-demand** (not spot/interruptible) — spot instances can be preempted mid-run
- Use the **vastai/pytorch cuda-12.1.1** template image

> If you see `Error: GPU error, unable to start instance` or repeated `retries exceeded` in the instance log, the host machine's GPU is faulty. Destroy the instance and pick a different one — this is common on consumer hardware.

### 2. Verify the GPU on first login

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Both must succeed before proceeding. If `nvidia-smi` hangs or errors, destroy and repick.

### 3. Set up the environment

vast.ai auto-starts a `tmux` session on SSH. Your terminal is already inside tmux — if you disconnect and reconnect, run `tmux attach -t ssh_tmux` to resume without losing your process.

```bash
# Clone the repo
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai

# Install dependencies
pip install -r pipeline/requirements.txt

# Install unsloth (needs a CUDA-specific build — do this separately)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Add your API keys
cp pipeline/.env.example pipeline/.env
nano pipeline/.env   # set NVIDIA_NIM_API_KEY, EXA_API_KEY, HF_TOKEN
```

### 4. Smoke test before the full run

Confirms VRAM is sufficient and the training loop starts without errors:

```bash
python pipeline/2_model_trainer.py --mode sft --curriculum_stage 1 \
    --output_name checkpoint_test --max_steps 20
```

Watch for tokens/sec in the output and no OOM errors. If OOM, lower `max_seq_length` from 2048 → 1024 in `2_model_trainer.py`.

### 5. Save checkpoints before destroying

The instance disk is wiped on Destroy. Upload to HuggingFace after each stage:

```bash
huggingface-cli login   # paste your HF token (same as HF_TOKEN in .env)
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_sft \
    --hf_username AjinkyaTaranekar
```

Or copy back locally (run from your local machine, not the instance):

```bash
scp -P <port> -r root@<host>:~/trustworthy-personalized-ai/models ./
```

### Data persistence

| Action | Data |
|--------|------|
| SSH disconnect | Survives (tmux keeps processes running) |
| **Stop** instance | Survives — can restart and continue (small storage fee) |
| **Destroy** instance | Wiped — upload checkpoints first |

> No volume setup needed. The disk you selected at rental time is automatically available at `/root/`.

---

## Training pipeline: two phases

### Phase 1 — Supervised Fine-Tuning (SFT)

Teaches the model the constitutional format: `<think>CAPABILITY_CHECK...</think><answer>...</answer>`.

```bash
cd pipeline

# Step 1: Generate SFT data
# Recommended (free): NVIDIA NIM — Kimi K2.6 as generator, Minimax M2.7 as independent critic
python sft_question_generator.py \
  --model  nvidia_nim/moonshotai/kimi-k2.6 \
  --output data/questions_partA.jsonl

python sft_gold_response_generator.py \
  --questions   data/questions_partA.jsonl \
  --output      data/train_partA.jsonl \
  --model       nvidia_nim/moonshotai/kimi-k2.6 \
  --critic_model nvidia_nim/minimaxai/minimax-m2.7   # different model family = genuine independence

# Alternative (also free): Groq
# python sft_question_generator.py --model groq/llama-3.3-70b-versatile --output data/questions_partA.jsonl
# python sft_gold_response_generator.py --questions data/questions_partA.jsonl \
#   --model groq/llama-3.3-70b-versatile --critic_model groq/gemma2-9b-it --output data/train_partA.jsonl

# Alternative (paid ~$10–15): Anthropic
# python sft_gold_response_generator.py --questions data/questions_partA.jsonl \
#   --model claude-sonnet-4-6 --critic_model claude-opus-4-7 --output data/train_partA.jsonl

python sft_math_pipeline.py --output data/train_partB.jsonl

python sft_dataset_assembler.py \
  --part_a    data/train_partA.jsonl \
  --part_b    data/train_partB.jsonl \
  --output_dir data
# Produces data/train_sft_v3_robust.jsonl (1,983 examples — multi-turn, native JSON tools, robustness variants)

# Step 2: Train (requires GPU — ~50 min on A4000 / 2–4h on A100)
python 2_model_trainer.py \
  --mode sft \
  --data_dir data \
  --output_dir models \
  --output_name checkpoint_sft \
  --resume

# Step 3: Save SFT constitutional baseline (run BEFORE any GRPO)
python 3_infererence.py --model_dir models/checkpoint_sft --port 8000 &
python 4_benchmark.py --probe_only --save_as_baseline
```

### SFT v3 — Asymmetric Distillation (recommended for sub-1B models)

The v3 pipeline eliminates context-window starvation in 0.6B models by keeping the 25-principle constitution teacher-side only. The student model only sees a ≤50-word system prompt.

    # 1. Generate questions (same as v2)
    python pipeline/sft_question_generator.py --count 200 --output pipeline/data/questions_v3.jsonl

    # 2. Generate gold responses with live tool execution and exa.ai web search
    # (nohup form for long runs — auto-commits every 50 lines via --watch_commit)
    nohup python -u pipeline/sft_v3_generator.py \
        --questions pipeline/data/questions_partA.jsonl \
        --output pipeline/data/train_v3.jsonl \
        --model nvidia_nim/minimaxai/minimax-m2.7 \
        --workers 5 \
        --watch_commit \
        > pipeline/nohup_generator.out 2>&1 &

    # 2b. Negative trajectories (inventory constraints + environment timeouts)
    python pipeline/sft_v3_generator.py \
        --questions pipeline/data/questions_v3.jsonl \
        --type inventory_constraint \
        --output pipeline/data/train_v3_negative.jsonl

    # 3. Pre-flight validation (fails if >5% of rows are malformed)
    python pipeline/validate_sft_data.py --input pipeline/data/train_v3.jsonl

    # 4. Assemble dataset (same assembler as v2)
    python pipeline/sft_dataset_assembler.py \
        --part_a pipeline/data/train_v3.jsonl \
        --output_dir pipeline/data/

    # 5. Curriculum training — 3 stages
    python pipeline/2_model_trainer.py --mode sft --curriculum_stage 1 --output_name checkpoint_sft_s1
    python pipeline/2_model_trainer.py --mode sft --curriculum_stage 2 --from_checkpoint models/checkpoint_sft_s1 --output_name checkpoint_sft_s2
    python pipeline/2_model_trainer.py --mode sft --curriculum_stage 3 --from_checkpoint models/checkpoint_sft_s2 --output_name checkpoint_sft

    # 6. GRPO (add --v3_format flag when base is a v3-trained model)
    python pipeline/2_model_trainer.py --mode grpo --sft_checkpoint models/checkpoint_sft --v3_format

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
  --output_name checkpoint_grpo_c \
  --resume

# Condition D — full composite reward (format + accuracy + tool integrity + constitution)
python 2_model_trainer.py \
  --mode grpo \
  --sft_checkpoint models/checkpoint_sft \
  --reward_type d \
  --output_name checkpoint_grpo_d \
  --resume
```

**Composite reward (Condition D):**
```
reward = 0.30 × format_score       (CAPABILITY_CHECK + think + answer tags present)
       + 0.40 × accuracy_score     (code execution gives correct answer)
       + 0.15 × tool_integrity     (no hallucinated or unavailable tools)
       + 0.15 × constitution_score (broader rule check: P1 + P3 + P4 + P14 + P18)
```

Each component is logged separately by TRL's multi-function reward system — visible as `rewards/format_reward_mean`, `rewards/accuracy_reward_mean`, etc. in `grpo_loss_history.json`.

### Publishing a checkpoint to HuggingFace

After training, the model is automatically pushed to HuggingFace unless `--no_publish` is set. If the push failed (network error, missing token), re-upload without retraining:

```bash
# Set HF_TOKEN in pipeline/.env first: HF_TOKEN=hf_...

# Re-upload SFT checkpoint
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_sft --hf_username AjinkyaTaranekar

# Re-upload a GRPO checkpoint
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_grpo_d --hf_username AjinkyaTaranekar

# Skip GGUF export (use on machines without llama.cpp or sudo access)
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_sft --skip_gguf
```

Publish merges LoRA adapters → 16-bit safetensors, exports GGUF (Q4_K_M), computes ROUGE vs baseline, and pushes both formats with retry (3 attempts, exponential backoff). If `HF_TOKEN` is unset, it saves locally and skips the upload. Pass `--skip_gguf` to bypass GGUF export and push entirely (required on machines without `llama.cpp` or `sudo` access).

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

| # | Stage | Flag condition | Output |
|---|---|---|---|
| 0 | Infrastructure setup (FalkorDB) | `ENABLE_USER_MODELLING=true` | FalkorDB running on port 6379 |
| 0.5 | Appraisal labelling | `ENABLE_EMPATHY=true` | `data/appraisal_labels.jsonl` |
| 1 | SFT data check | always | `data/train_sft_v2.jsonl` (existence check; `train_sft_v3_robust.jsonl` consumed at training time) |
| 2 | SFT training | always | `models/checkpoint_sft/` |
| 3 | SFT constitutional baseline | always | `reports/constitution_baseline.json` |
| 4 | Experiment 0 (reasoning comparison) | always | `reports/experiment0_*.json` |
| 5 | Adversarial baseline | always | `reports/adversarial_baseline.json` |
| 6 | GRPO Condition C + drift check | `ENABLE_GRPO=true` | `models/checkpoint_grpo_c/` |
| 7 | GRPO Condition D + drift check | `ENABLE_GRPO=true` | `models/checkpoint_grpo_d/` |
| 8 | Final ablation A/B/C/D | always | `reports/ablation_*_*.json` |

Enable optional modules for a full run:

```bash
PIPELINE_ENABLE_USER_MODELLING=true \
PIPELINE_ENABLE_EMPATHY=true \
PIPELINE_ENABLE_GRPO=true \
bash pipeline/run_all.sh
```

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

### Harness comparison benchmark

Run the constitutional probe suite with and without the inference-time harness to quantify its contribution:

```bash
# Start server with harness enabled
PIPELINE_ENABLE_HARNESS=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft

# In a second terminal — runs probes twice and prints per-principle delta
python pipeline/4_benchmark.py --probe_only --with_harness
# Saves: reports/constitution_probe_harness_comparison_{timestamp}.json
```

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
# Start server (base flags only)
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Start with optional modules enabled
PIPELINE_ENABLE_USER_MODELLING=true \
PIPELINE_ENABLE_EMPATHY=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Or via YAML config
python pipeline/3_infererence.py --config pipeline_config.yaml --port 8000
```

**Endpoints:**

```
GET  /health                            liveness + model name
GET  /config                            active feature-flag state
GET  /v1/models                         list loaded model
GET  /v1/tools                          list registered tools
POST /v1/tools/register                 add a tool at runtime
DELETE /v1/tools/{name}                 remove a tool
POST /v1/chat/completions               generate (tool loop server-side)
GET  /metrics                           latency p50/p95/p99, throughput, tool counts
POST /metrics/reset                     reset counters
GET  /harness/metrics                   per-principle failure rates, retry stats, adaptation state
POST /harness/reset                     reset rolling harness counters

# Dependency monitoring (Blocker 4 — OWASP LLM09)
GET  /dependency/status/{session_id}    interaction frequency + disclosure state
POST /dependency/reset/{session_id}     reset session monitor

# Scrutability (ENABLE_USER_MODELLING only)
GET  /memory/inspect/{session_id}       NL summary of user's 5W+H belief graph
POST /memory/contest                    flag a belief node as wrong
POST /memory/correct                    apply user-supplied correction (audit trail preserved)
```

**`/v1/chat/completions` response envelope** (new fields when modules are active):

```json
{
  "response":              "...",
  "dependency_disclosure": false,
  "metrics":               { "latency_s": 1.2, "tokens_generated": 180, ... },
  "user_modelling":        { "nodes_written": 2, "conflicts": [], "conflict_count": 0 },
  "appraisal":             { "top3": ["pleasantness", "goal_relevance", "coping_potential"],
                             "valence": 0.91, "reading": "...", "present": true },
  "ontology_score":        { "ontology_score": 0.85, "total_claims": 4, "verified_count": 3,
                             "unverified_claims": ["..."] }
}
```

All three extra fields are `null` when their flag is off — existing clients are unaffected.

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

## User Modelling (ENABLE_USER_MODELLING)

Implements the thesis's scrutable 5W+H user graph. After every user turn the write pipeline extracts WHO/WHAT/WHERE/WHY/HOW entities, detects contradictions with existing beliefs, and writes with `:DEPRECATED_BY` edges (never deletes — the full audit trail is always preserved).

```bash
# Prerequisites
docker compose up -d   # starts FalkorDB on port 6379

# Start server with user modelling
PIPELINE_ENABLE_USER_MODELLING=true \
PIPELINE_ENABLE_PERSONALISATION=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft

# Scrutability API
curl http://localhost:8000/memory/inspect/my_session_id
curl -X POST http://localhost:8000/memory/contest \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my_session_id", "node_id": "abc123"}'
curl -X POST http://localhost:8000/memory/correct \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my_session_id", "old_node_id": "abc123",
          "correction": "I am an intermediate Python developer, not a beginner",
          "label": "Skill"}'
```

The scrutability layer is the thesis's named contribution: **no current production system** (Mem0, ChatGPT memory, Gemini) exposes inspect/contest/correct/audit to end users. See `wiki/topics/personalisation.md` for the design rationale.

---

## Empathy (ENABLE_EMPATHY)

Qwen is fine-tuned on AppraisePLM-labelled EmpatheticDialogues data to produce an `<appraisal>` block inside its `<think>` chain — no external model at inference time. AppraisePLM (Debnath, Graham, Conlan — CoNLL 2025) is used **offline as a labeller only**.

```bash
# Step 1 (one-time, CPU, ~30 min for 5000 examples)
git clone https://github.com/alokdebnath/appraise-PLM
python pipeline/appraisal_labeller.py --appraise_plm_path appraise-PLM
# Output: data/appraisal_labels.jsonl

# Step 2: include appraisal_empathy category in SFT data generation
# (sft_question_generator.py reads from data/appraisal_labels.jsonl automatically)
python sft_question_generator.py --category appraisal_empathy \
       --output data/questions_empathy.jsonl

# Step 3: run server with empathy enabled
PIPELINE_ENABLE_EMPATHY=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft
```

The model's `<think>` block will include:
```xml
<appraisal>
  pleasantness: 0.91 | goal_relevance: 0.85 | coping_potential: 0.23
  reading: high pleasantness, high goal relevance, low coping potential
  → user achieved something important but feels they barely held it together
</appraisal>
```

---

## Ontology Verifier (ENABLE_ONTOLOGY_VERIF)

Implements Experiment 6 Approach B. After each assistant response, the verifier extracts atomic factual claims and checks them against a SPARQL-queryable knowledge base. Returns an `ontology_score` (mean confidence) per response.

```bash
# Option A — local OWL file (pip install rdflib)
# Place an OWL/RDF file at: data/ontology.owl
PIPELINE_ENABLE_ONTOLOGY_VERIF=true \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft

# Option B — remote SPARQL endpoint (pip install SPARQLWrapper)
PIPELINE_ENABLE_ONTOLOGY_VERIF=true \
PIPELINE_ONTOLOGY_SPARQL_ENDPOINT=https://dbpedia.org/sparql \
python pipeline/3_infererence.py --model_dir models/checkpoint_sft
```

---

## Security hardening (this branch)

Four pre-GRPO security blockers are implemented and verified by `preflight_check.sh`:

| Blocker | File | What it does |
|---|---|---|
| 1a — code sandbox | `3_infererence.py` | AST-validates LLM-generated code before `subprocess.run`; blocks `os`, `sys`, `socket`, dangerous builtins |
| 1b — injection sanitiser | `3_infererence.py` | Strips prompt-injection patterns from web/URL tool outputs before injecting into model context |
| 1c/1d — sampler sandbox | `sft_math_pipeline.py` | Same AST validation for verification code (math question generation + rejection sampling merged into one script) |
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

The 23 principles in `pipeline/constitution.md` define the model's target behaviour: capability honesty, correct tool use, honest refusal, uncertainty quantification, sycophancy resistance, first-principles reasoning, 5W+H framing, consequence checking, and interleaved tool chaining. Read it first to understand what the model is being trained to do and why.

---

## Research wiki

The `wiki/` directory is an Obsidian vault containing the full thesis knowledge graph. Start at `wiki/index.md`. Key entry points:

- `wiki/overview.md` — thesis synthesis and official research question
- `wiki/queries/grpo-and-personalisation-master-plan.md` — full implementation plan for this branch
- `wiki/experiments/experiment-catalog.md` — all six experiments + ablation conditions
- `wiki/decisions/2025-10-01-four-module-architecture.md` — the binding architectural decision
