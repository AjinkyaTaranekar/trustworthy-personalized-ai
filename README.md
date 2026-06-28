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
| **Crusoe** ✅ judge fallback | `CRUSOE_API_KEY=...` | `crusoe/zai/GLM-5.1` | Credits |
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
├── sft_v3_generator.py             SFT step 1b — asymmetric distillation: teacher (full constitution) → student prompt swap, live tool intercept, native <tool_call> output
├── sft_math_pipeline.py            SFT step 2  — math/code questions (7 types) + rejection sampling
├── sft_dataset_assembler.py        SFT step 3  — merge, quality-gate (think≥150, teacher-leak, banned phrases), full-native conversion, robustness variants → train_sft_v3.jsonl
├── sft_trajectory_splitter.py      Thinker–Executor (exp 3): pure-transform factoring of the *_v3 parts → train_sft_thinker.jsonl (prose <think>+<ask>/<act>/<answer>) + train_sft_executor.jsonl (one <act> → one <tool_call>). Thinker tool turns are stamped with tool_io.sanitise_tool_result — byte-identical to what the server feeds (train/serve parity)
├── sft_curriculum_merge.py         Thinker–Executor (exp 3): interleave Branch B <ask> rows into the factored Thinker set (auto ratio) → train_sft_thinker_curriculum.jsonl
├── tool_io.py                      Single source of truth for tool-result presentation (injection strip + per-tool budgets + [TOOL_RESULT] wrapper) AND the TOOL_PROFILES tool-set definitions; imported by 3_infererence.py, the orchestrator, the splitter, and the re-stamper so a model is served what it was trained on
├── restamp_native_tools.py         Pure transform: rewrite metadata.native_tools in any SFT JSONL to the canonical served schema (registry.to_openai_schemas(TOOL_PROFILES[profile])). Run after any tool-description change or to repair tool-set drift (fixed train_sft_v3 training on 1–4 tools while served 7–10)
├── validate_thinker_executor_data.py  Pre-train gate (CPU): asserts Thinker/Executor train/serve contract — canonical tool turns, prose-only, opening <think>≥150, Executor one-call + copy-fidelity. Run before any GPU time
├── executor_ablation.py            Decides whether the Executor SFT earns its place: base Qwen3-0.6B vs SFT'd Executor on tool-choice accuracy + copy-fidelity (--self_test on CPU; GPU to run)
├── composed_loop_eval.py           End-to-end Thinker–Executor eval: completion rate, exec parse rate, copy-fidelity (Executor copies the Thinker's <act> arg verbatim), math correctness — the gate that component eval_loss/ROUGE can't see (--self_test on CPU; GPU to run)
├── appraisal_labeller.py           Offline: AppraisePLM → EmpatheticDialogues labels
│
│   ─── Training ───
├── 2_model_trainer.py              Phase 1: SFT  |  Phase 2: GRPO (DAPO improvements)  |  Publish: upload checkpoint to HuggingFace
│
│   ─── Inference + evaluation ───
├── 3_infererence.py                FastAPI server — model + all four module hooks. Single-model OR (with --thinker/--executor) the Thinker–Executor dual-adapter loop, so both experiments share one harness, sanitiser, metrics, and run-record log (reports/inference_runs.jsonl)
├── thinker_executor_orchestrator.py  Thinker–Executor (exp 3): the two-model loop itself — Thinker <act> → Executor <tool_call> → tool → back; one base + two LoRA adapters (PEFT). Hosted inside 3_infererence.py for production/benchmark; standalone --self_test / --question / --chat / --serve for a bare server with no harness
├── 4_benchmark.py                  Constitutional probes + adversarial suite
├── 5_context_degradation.py        Context-length degradation study (greedy decoding)
├── experiment0_reasoning_comparison.py  Experiment 0 — CoT/ToT/interleaved/baseline
│
│   ─── Analysis / dissertation export ───
├── principle_families.py           Canonical probe→family + framing map (single source for stratification)
├── export_assets.py                reports/ → LaTeX tables + PDF figures + significance (reports/dissertation_assets/)
├── analysis.ipynb                  Interactive Plotly analysis dashboard
│
│   ─── Runtime modules (loaded by 3_infererence.py via feature flags) ───
├── user_modelling.py               FalkorDB 5W+H graph + Mem0g write + scrutability
├── empathy.py                      Appraisal-conditioned generation helpers
├── ontology_verifier.py            Post-hoc SPARQL claim scorer (Experiment 6 Approach B)
│
│   ─── Reference ───
├── constitution.md                 Constitution (defines P1–P25; the probe suite scores 21 items — P1–P21 with P2+P3 merged, plus H2_memory_persistence)
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

# Asymmetric distillation: teacher uses the full constitution; only a short student
# prompt is saved. Tool calls are intercepted and executed live; output is native <tool_call>.
python sft_v3_generator.py \
  --questions data/questions_partA.jsonl \
  --output    data/train_partA_v3.jsonl \
  --model     nvidia_nim/minimaxai/minimax-m2.7

python sft_math_pipeline.py --output data/train_partB_v3.jsonl

# (Thinker–Executor experiment, optional) Branch B — clarification trajectories — PLUS
# adversarial/security refusal trajectories, in ONE rate-limited run (shared worker pool +
# key rotation; avoids two processes competing for the NIM rate limit). Branch B: teacher
# decides per item to <ask> (genuine ambiguity → clarify → simulated user answer → resolve)
# or proceed (don't-ask negative). Adversarial: a built-in red-team seed set (prompt
# injection, authority spoof, tool-result injection, malware/intrusion demands) where
# <think> detects the attack and <answer> refuses. Both are THINKER format (<think>+
# <ask>/<act>/<answer>) and BOTH land in train_sft_thinker_branch_b.jsonl (the `branch`
# metadata distinguishes them); consumed by the curriculum merge, NOT the SFT assembler.
# Memory is 50/50: half the rows carry a sampled [USER MEMORY] profile, half an explicit
# empty block, so the Thinker learns to use memory AND to cope when it's empty. Spot-check first:
python sft_v3_generator.py \
  --questions data/questions_partA.jsonl \
  --branch_b --adversarial --max 8 \
  --model nvidia_nim/minimaxai/minimax-m2.7   # canonical teacher; kimi-k2.6 skips every row (reasoning is out-of-band)
#   → data/train_sft_thinker_branch_b.jsonl   (Branch B + adversarial rows)
# Full unattended run on a VM (auto-commit+push the data file every 50 rows):
#   nohup python -u sft_v3_generator.py --questions data/questions_partA.jsonl \
#     --branch_b --adversarial --workers 5 --watch_commit --watch_threshold 50 \
#     > nohup_thinker_gen.out 2>&1 &

# (Thinker–Executor experiment, optional) Factor the existing v3 trajectories into the two
# role-conditioned SFT sets — a pure transformation, no GPU/teacher. Reads the *_v3 parts,
# renders each Executor-owned call as a plain-language <act> instruction, and writes:
#   data/train_sft_thinker.jsonl   (<think> + <ask>/<act>/<answer>, prose only)
#   data/train_sft_executor.jsonl  (one <act> instruction → one native <tool_call>)
# Spot-check with --inspect before the real write; gates match sft_dataset_assembler.
python sft_trajectory_splitter.py --inspect 5      # preview, no write
python sft_trajectory_splitter.py                  # factor both parts → both outputs

# Curriculum merge: interleave the Branch B <ask> rows into the factored Thinker set so the
# Thinker doesn't learn to always delegate (auto ratio = len(factored)//len(branch_b);
# places all Branch B rows evenly). Runnable before Branch B exists — it then just passes
# the factored A/C set through with a warning. Re-run once Branch B is generated.
python sft_curriculum_merge.py                     # → data/train_sft_thinker_curriculum.jsonl

# Then train two checkpoints (no trainer changes needed; --no_curriculum — both sets are
# already ordered / single-action, so the trainer's 3-stage split would only re-order them):
#   python 2_model_trainer.py --mode sft --no_curriculum --dataset data/train_sft_thinker_curriculum.jsonl --output_name checkpoint_thinker
#   python 2_model_trainer.py --mode sft --no_curriculum --dataset data/train_sft_executor.jsonl           --output_name checkpoint_executor
# Serve the pair (one base + both LoRA adapters) THROUGH the main inference server so the dual
# model inherits the full constitutional harness, injection sanitiser, dependency monitor,
# self-critique, metrics, and per-request run records — identical post-processing to the single
# model, so the dual-vs-single benchmark is apples-to-apples:
#   python thinker_executor_orchestrator.py --self_test                       # CPU loop sanity check, no GPU
#   python 3_infererence.py --thinker models/checkpoint_thinker --executor models/checkpoint_executor --port 8000
#   python 4_benchmark.py --server_url http://localhost:8000
# (Records land in reports/inference_runs.jsonl; harness stats at /harness/metrics. The standalone
#  thinker_executor_orchestrator.py --serve still exists for a bare two-model server with no harness.)

python sft_dataset_assembler.py        # part_a/part_b default to the *_v3.jsonl files
# Quality-gates (think≥150, no teacher leak, no banned phrases), converts to full-native,
# and writes data/train_sft_v3.jsonl (the trainer's input).

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

# Vanilla/base-model baseline — use --tool_mode native so the pre-trained
# Hermes <tool_call> format is activated (the base model doesn't know the
# SFT-trained <tool> XML format, so tool-using probes would score 0 without this)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001 &
python 4_benchmark.py --server_url http://localhost:8001 --probe_only --tool_mode native --save_as_baseline
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

    # 3. Assemble + quality-gate + full-native → data/train_sft_v3.jsonl
    #    (the assembler IS the quality gate: think≥150, teacher-leak, banned phrases, answer tag)
    python pipeline/sft_dataset_assembler.py \
        --part_a pipeline/data/train_v3.jsonl \
        --output_dir pipeline/data/

    # 4. Train — the 3-stage curriculum (format → complexity → replay) runs by default
    python pipeline/2_model_trainer.py --mode sft --output_name checkpoint_sft
    #    (add --no_curriculum to train once on the full set; manual per-stage control via
    #     --curriculum_stage 1|2|3 + --from_checkpoint is still available)

    # 5. GRPO (add --v3_format flag when base is a v3-trained model)
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
| 1 | SFT data assembly | always | `data/train_sft_v3.jsonl` (produced by `sft_dataset_assembler.py` from the `*_v3` source parts) |
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

### Exporting dissertation assets (tables, figures, stats)

`analysis.ipynb` produces interactive Plotly charts only. To generate the **static LaTeX tables, PDF figures, and significance statistics** the dissertation `\input`s, run:

```bash
python pipeline/export_assets.py
# Auto-selects the vanilla-vs-SFT probe pair BY model_label and writes into
# reports/dissertation_assets/:
#   tab_*.tex      \input-able tabulars (per-principle, per-family, think-collapse,
#                  adversarial, category, runs)
#   fig_*.pdf      \includegraphics figures (think-collapse, per-family)
#   summary.json   all aggregates + significance

# Pin the pair explicitly (recommended once the final runs exist):
python pipeline/export_assets.py \
    --vanilla reports/constitution_probe_vanilla_<ts>.json \
    --sft     reports/constitution_probe_<ts>.json
```

What it computes (no third-party stats dependency):

- per-principle vanilla→SFT deltas + Cohen's *h*; per-**family** scores and the C3AI positive/negative **framing split** (family map: `pipeline/principle_families.py`)
- the `<think>`-trace collapse (mean chars, % empty), adversarial-by-attack-type, category coverage, and a run summary
- exact two-sided **McNemar** p (paired by principle × question), **Wilson** 95% CIs

Notes:

- It selects the pair by `run_metadata.model_label` and **warns** if the chosen "SFT" run is actually a base-model re-run, or if vanilla and SFT used a different number of questions per principle (they must match for a clean 66-probe comparison).
- `4_benchmark.py` now writes `run_metadata` (incl. `model_label`) into the adversarial, category, and drift reports too, so every report is model-attributable (older reports without it are paired by timestamp batch).

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

## Evaluation: persona conversation suite (Suite E)

Deterministic multi-turn evaluation of how the model serves real *human profiles*. Each persona is a fixed profile plus a hand-written script of user turns (e.g. a risk-averse nurse saving for a house, a grieving student, a non-technical bakery owner, an adversarial health-info seeker). The script is replayed verbatim through the server — there is **no LLM on the user side**, so the run is reproducible. `4_benchmark.py` only *generates* the conversation and saves the whole transcript; the *conversation-level* judge that scores each transcript on six dimensions (personalisation, memory consistency, empathy, trustworthiness, coherence, goal completion) runs separately in **`5_judgement_day.py`** (see below).

```bash
# Generate the persona transcripts (GPU; deterministic by default — greedy decoding)
python 4_benchmark.py --persona_only

# As part of a full generation pass
python 4_benchmark.py --probe --categories --persona

# Across checkpoints via hot-swap (per-model persona reports)
python 4_benchmark.py --models ./models/vanilla ./models/checkpoint_sft \
    --labels vanilla sft --persona
```

> **Decoding:** `4_benchmark.py` decodes **greedily** (`do_sample=False`) by default, matching the training-eval path in `2_model_trainer.py`. This keeps runs deterministic and lets strict-JSON `<tool_call>` blocks parse reliably — at `temperature>0` a 0.6B model degenerates them (unclosed tags, malformed args) so tools never fire and `tool_trace` comes back empty. Pass `--sample` to restore stochastic decoding at `--temperature`.
>
> **Thinker–Executor exception:** decoding is **role-determined** in `thinker_executor_orchestrator.py`, so the benchmark's greedy flag does *not* fully apply. The **Executor** stays greedy (it is a deterministic instruction→one-tool-call transducer). The **Thinker** is **always sampled** (`temperature 0.7`, `top_p 0.9`): under pure argmax a 0.6B trained on low-diversity teacher reasoning collapses onto a single canned synthesis sentence on ~60% of questions, which also suppresses tool delegation. Reproducibility is preserved via a fixed seed (`PIPELINE_THINKER_SEED`, default `1234`); override the temperature with `PIPELINE_THINKER_TEMPERATURE`.

Output: `reports/persona_conversations_<label>_<ts>.json` with the full transcript, profile, and expectations per persona (scores are filled in by the judge step). The script is deterministic; model sampling is the other axis — pass `--temperature 0` for headline numbers, or repeat at a higher temperature for a variance band.

---

## Evaluation: judging (`5_judgement_day.py`, no GPU)

`4_benchmark.py` makes **no LLM calls** — it generates responses on the GPU and embeds each item's judge spec into the report. The LLM-as-judge is a separate, API-only step, so you can release the GPU first and re-judge locally for free:

```bash
# Judge every saved report under reports/ (in place; keeps a .prejudge.bak)
python 5_judgement_day.py --judge_model claude-opus-4-8

# Scope to the five ladder conditions and also write a narrative diagnostic
python 5_judgement_day.py --judge_model claude-opus-4-8 \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor --report

# Crusoe (OpenAI-compatible) judge fallback when NVIDIA NIM is rate-limited
#   — set CRUSOE_API_KEY; the crusoe/ prefix routes via litellm's openai/ handler.
python 5_judgement_day.py --judge_model crusoe/zai/GLM-5.1 \
    --reports_dir reports_jun24 \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor
```

The judge picks its key by the model's provider prefix: `crusoe/...` -> `CRUSOE_API_KEY(S)`, otherwise `NVIDIA_NIM_API_KEY(S)`. Any OpenAI-compatible endpoint can be added to `_OPENAI_COMPAT` in `llm_pool.py` (prefix -> base URL + key env). **The judge model must be identical across all five conditions** (it is the measurement instrument) and should be the same model `--meta_eval` validates against the human gold set — so if you switch the headline judge to GLM-5.1, re-run the meta-eval with GLM-5.1 too.

It fills `llm_score` / `combined_score` / `persona_score` (+ the six persona `dimension_means`) and recomputes the blended aggregates. Two upgraded prompts (both system+user, reasoning-before-score, calibrated anchors): a per-response judge for Suites A–C and a whole-transcript conversation judge for Suite E. A failed judge call scores `None` (excluded from the average), never a silent 0.5. Keep the judge model identical across conditions (recorded in `run_metadata.judged_by`); adversarial (Suite D) is rule-only and never judged.

**Judge-primary, substance lens (default).** The headline `combined_score` is now the **LLM judge** score (substance-based, validated against humans via `--meta_eval`); the brittle single-keyword `rule_score` from `4_benchmark.py`'s regex checks is kept only as a diagnostic, not blended in — so a model is no longer marked wrong because one exact word failed to appear. The judge grades **behaviour, not vocabulary**, and explicitly credits an appropriate clarifying question (`<ask>`) on an underspecified/personal query as correct, high-scoring behaviour (while penalising clarification of a fully-answerable question). Pass `--blend_rule` to restore the old `(rule+judge)/2`. The mode is recorded in `run_metadata.combine_mode`. To re-score **existing** reports with this lens (no GPU needed), re-judge with `--force` (resume otherwise skips already-judged items).

**Self-consistency (`--k`).** Pass `--k 3` (with `--sc_temperature`, default 0.3) to sample the judge K times per item and use the **mean** score — mean-of-K aggregation tracks human ratings better than a single greedy call (arXiv 2506.13639). The per-item spread is persisted as `llm_score_std` and `run_metadata.judge_k_samples`. `--k 1` (default) is the original single near-greedy call.

**Enriched rubrics (`judge_rubrics.py`).** Each constitution principle has an instance-grounded spec — a behaviour rubric, **endpoint score anchors** (what 1.0 vs 0.0 concretely look like) and a **reference exemplar** — the two design choices that most raise judge–human agreement (Prometheus; BiGGen Bench; arXiv 2506.13639). These are applied automatically at judge time (no GPU; re-judge existing reports with `--force`) and to `--meta_eval`/`--preview` so validity is measured on the same lens.

**Neutral assessor ablation (Phase 3).** `--judge_constitution_mode {full,bare,none}` controls how much of the constitution the judge sees. `--ablate_judge` re-judges the constitution suite under all three modes with responses fixed and tabulates per-model `full / bare / none / (full−none)` — separating "the model genuinely complies" from "the judge rewards constitution-shaped text". Writes `reports/ablate_judge_<ts>.json`, modifies nothing.

```bash
# re-judge reports_jun24 constitution with the enriched lens (offline, no GPU)
python 5_judgement_day.py --judge_model nvidia_nim/minimaxai/minimax-m3 \
    --reports_dir reports_jun24 --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --suites constitution --force
# does the constitution help the model, or just bias the judge?
python 5_judgement_day.py --judge_model nvidia_nim/minimaxai/minimax-m3 \
    --reports_dir reports_jun24 --labels vanilla_base sft_constitution thinker_executor --ablate_judge
```

### Validating the judge: meta-evaluation against a human-anchored gold set

Before trusting the judge's scores, measure how well they agree with **human ground truth** — the judge's validity is an empirical claim, not an assumption. This is the answer to "how do you know the marking scheme is right?".

```bash
# 1. Build a BLIND gold-annotation template (stratified by principle; no judge score shown,
#    so your annotation is not anchored to the model's). Needs no API keys.
python 5_judgement_day.py --judge_model nvidia_nim/minimaxai/minimax-m3 --make_gold \
    --labels vanilla_base sft_constitution thinker_executor --suites constitution \
    --n_per_principle 2 --gold reports/gold/gold_set.jsonl

# 2. Hand-score each line: set human_score to 0.0 / 0.5 / 1.0 (optionally human_note).

# 3. Measure judge-vs-human agreement (+ judge self-consistency), overall and per principle.
python 5_judgement_day.py --judge_model nvidia_nim/minimaxai/minimax-m3 --meta_eval \
    --gold reports/gold/gold_set.jsonl --k 3
```

Reports **Krippendorff's α** (interval; target ≥ 0.67), **Gwet's AC1** and Cohen's κ on the pass/fail decision (AC1 is robust to the skew that distorts κ on high-pass-rate probes), Pearson/Spearman correlation, MAE, judge bias (lenient/strict), and self-consistency (mean within-item std + verdict-flip rate) — broken down per principle and per AbstentionBench answerability type (unknown · underspecified · false-premise · subjective · stale). Writes `reports/meta_eval_<judge>_<ts>.json`. See [`judge_reliability.py`](pipeline/judge_reliability.py) and the wiki [human-evaluation rubric](wiki/experiments/human-evaluation-rubric.md), which supplies the human ground truth.

---

## Evaluation: multi-turn benchmark + comparison

```bash
# Standard 14-turn benchmark
python 4_benchmark.py --server_url http://localhost:8000

# Compare base model vs fine-tuned (legacy: two servers)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001  # second terminal
python 4_benchmark.py --server_url http://localhost:8000 --compare_url http://localhost:8001
```

### Multi-model hot-swap benchmarking

Compare N models sequentially using one inference server — no restarts. The `--models` flag tells the benchmark to call `POST /v1/model/swap` before each run, unloading the current model and loading the next one (~30 s). Metrics are reset automatically between models.

```bash
# Start the server once (terminal 1)
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Compare vanilla base model vs two SFT checkpoints (terminal 2)
python pipeline/4_benchmark.py \
    --models unsloth/Qwen3-0.6B ./models/checkpoint_sft_v1 ./models/checkpoint_sft_v2 \
    --labels vanilla sft_v1 sft_v2 \
    --probe --categories \
    --server_url http://localhost:8000
# → saves per-model JSON reports + reports/comparison_<ts>.csv
# → prints N-column comparison table with Δ columns relative to the baseline (first model)

# Minimal: constitutional probes only on two models
python pipeline/4_benchmark.py \
    --models ./models/vanilla ./models/sft \
    --probe_only \
    --server_url http://localhost:8000
```

Model labels default to the last path component — `unsloth/Qwen3-0.6B` → `Qwen3-0.6B`, `./models/sft` → `sft`. Override with `--labels`.

**New flags:**

| Flag | Default | Description |
|---|---|---|
| `--models` | `None` | One or more model dirs or HF IDs to benchmark sequentially |
| `--labels` | path stem | Display labels (defaults to last component of each path) |
| `--base_model` | `unsloth/Qwen3-0.6B` | Fallback HF ID when a `--models` path does not exist on disk |
| `--max_seq_length` | `4096` | Sequence length passed to the swap endpoint |
| `--compare_output` | `reports/comparison_<ts>.csv` | Where to save the comparison CSV |

---

## Five-condition ablation ladder + cross-condition analysis

The headline dissertation comparison is a five-rung ablation ladder where each adjacent delta isolates one factor:

| Rung | Condition | How served / benchmarked | Isolates |
|---|---|---|---|
| C0 | `vanilla_base` | base model, `4_benchmark.py --tool_mode xml` (base ignores the XML tool instructions → tools effectively off) | floor |
| C1 | `vanilla_tools` | base model, `4_benchmark.py --tool_mode native` (base can emit Hermes tool calls) | value of tool access |
| C2 | `sft_template` | Exp 1 template-SFT (native), `--tool_mode native` | value of SFT format scaffolding |
| C3 | `sft_constitution` | Exp 2 constitutional-SFT, `--tool_mode native` | value of constitutional content (H1) |
| C4 | `thinker_executor` | dual model: `3_infererence.py --thinker … --executor …`, then `4_benchmark.py --tool_mode native` against that server | value of the architecture (H2) |

**All three SFT rungs use the native `<tool_call>` format**, so they are benchmarked with `--tool_mode native` and are directly comparable. C0 alone uses `--tool_mode xml`: the untrained base ignores the XML tool instructions, so tools never fire — that is the deliberate "tools-off" floor (C1 is the same weights with `--tool_mode native`). Generate the native Exp 1 dataset with `python 1_dataset_generator.py --variant interleaved --tool_format native --train_size N` → `data/train_interleaved_native.jsonl` (emits `<tool_call>` JSON, stamps `native_tools`, and remaps the few legacy tools not in the served registry, e.g. `get_exchange_rate`→`web_search`); train Exp 2 on its existing native `data/train_sft_v3.jsonl`.

> Note: with Exp 1 regenerated native, the C2→C3 delta is **no longer confounded by tool format**. The remaining differences between C2 and C3 are *training content*, *system prompt*, *tool inventory*, and *generation method* (template vs constitutional), so report C2→C3 as "template-SFT vs constitutional-SFT", not a strict single-variable isolation.

All five conditions produce identical-schema `4_benchmark.py` reports (the dual is served on the same `/v1/chat/completions` endpoint). Generate each fresh with `--temperature 0`, `BENCH_MOCK_SEARCH=1`, all five suites (`--probe --categories --drift --adversarial --persona`), the per-condition `--tool_mode` above, and `--output_dir reports/<label>`. Then judge them all locally in one pass with `5_judgement_day.py` and consolidate with `analyze_experiments.py`. Step-by-step: `pipeline/ABLATION_LADDER_RUNBOOK.md`.

Then consolidate offline (no GPU) with `analyze_experiments.py`:

```bash
python analyze_experiments.py \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor
```

It pairs the latest report of each suite per label, prints the ladder with the four **isolating deltas** (bootstrap 95% CIs where item-level data exists), keeps the constitution suite as **rule-based (primary) and combined (secondary) rows separately** to avoid LLM-judge circularity, and writes `experiment_ladder_<ts>.csv`, `experiment_ladder_<ts>.tex` (**two** dissertation tables: a clean headline ladder plus a separate depth/tool **diagnostics** table), and `experiment_h3_failures_<ts>.csv` (probes the top rung still fails or regresses on — the H3 limits evidence). Add `--figures` for the **five core** dissertation figures, or `--figures --extended_figures` for all nine.

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
POST /v1/model/swap                     hot-swap loaded model without restarting the server

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
| 1b — injection sanitiser | `tool_io.py` (used by `3_infererence.py`, orchestrator, splitter) | Strips prompt-injection patterns from web/URL tool outputs before injecting into model context; the SAME function stamps Thinker training data so serve == train |
| 1c/1d — sampler sandbox | `sft_math_pipeline.py` | Same AST validation for verification code (math question generation + rejection sampling merged into one script) |
| 2a — distillation format gate | `sft_v3_generator.py` | Intercept loop enforces think→tool→answer structure; banned-placeholder + think-length checks on generated rows |
| 2b — training data quality gate | `sft_dataset_assembler.py` | `passes_quality_filter()`: rejects short `<think>`, teacher-constitution leak, banned phrases, missing `<answer>` before any row enters training |
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

## Trace-aware scoring, reproducible search, and alignment metrics

The benchmark detects tool use from the orchestrator **trace**, not by grepping the answer text — essential for the dual Thinker–Executor, whose final answer is clean prose (the tool call lives in the trace). Older reports scored with the text-grep method under-counted tool-use principles (P4/P10/P19) and over-counted tool-avoidance (P11/P13).

```bash
# Re-score an OLD report offline (no GPU) using its saved tool_trace:
python pipeline/rescore_report.py reports/constitution_probe_<ts>.json
# → *_rescored.json + a per-principle diff (text-grep vs trace-aware)

# Offline trustworthiness/scrutability metrics from a saved report:
python pipeline/alignment_metrics.py reports/constitution_probe_<ts>.json
# → honesty F1 / over-refusal, fabrication rate, answer-grounding rate

# Reproducible, offline, fabrication-detectable search (set on the SERVER process).
# Returns a fixed corpus with MOCKFACT-* sentinels instead of a live EXA call:
BENCH_MOCK_SEARCH=1 python pipeline/3_infererence.py ...    # then run the benchmark as usual
```

LLM judge: a **separate offline step**, `5_judgement_day.py` (API-only, no GPU). `4_benchmark.py` generates responses and embeds each item's judge spec; the judge fills the scores afterwards. A failed judge call is excluded from the average (not silently scored 0.5).

See `pipeline/TRUSTWORTHINESS_SCRUTABILITY_REVIEW.md` for the full analysis.

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
