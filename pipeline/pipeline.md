# Unified Pipeline — End to End (Vast.ai Guide)

Complete runbook for training and benchmarking on a GPU cloud instance.
Follow every numbered step in order.

> **Dissertation context (May 2026):** GRPO is not approved for this dissertation.
> The experiment is SFT-only: vanilla Qwen3-0.6B vs constitutional fine-tuned model.
> Research question: *Can constitutional knowledge distillation and harness engineering
> enable a sub-1B on-device LLM to deliver trustworthy, personalised assistance?*

---

## 0. Rent a GPU on Vast.ai

**Recommended spec:**
- RTX 4000 Ada 16 GB — sufficient for 16-bit LoRA on 0.6B (~$0.35/hr)
- A100 40 GB — comfortable headroom, faster throughput
- Disk: 80 GB+
- Image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` or any CUDA 12.x image

**Cost estimate for dissertation experiments:**
- One SFT run (3 epochs, 1944 examples, r=64): ~2.5 hrs = **~$0.90**
- Budget for 5-6 experimental runs: **under $6 total**

Once the instance is running, SSH in and work from `/workspace`.

---

## 1. Clone Repo and Install Dependencies

```bash
cd /workspace
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai

# Core dependencies
pip install -r pipeline/requirements.txt

# Unsloth — install separately with the correct CUDA extras
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Optional: Flash Attention 2 for faster training (takes 15-20 min to compile)
pip install flash-attn --no-build-isolation
```

---

## 2. Environment Variables (.env)

Create `pipeline/.env`:

```bash
nano pipeline/.env
```

Required variables:

| Variable            | Required for                          | Where to get it                      |
|---------------------|---------------------------------------|--------------------------------------|
| HF_TOKEN            | Pushing models to HuggingFace         | huggingface.co → Settings → Tokens  |
| NVIDIA_NIM_API_KEYS | LLM judge in benchmark                | build.nvidia.com → API Keys          |
| EXA_API_KEY         | Web search tool at inference time     | exa.ai → Dashboard                   |
| HF_HOME             | Cache location (optional)             | Set to `/workspace/.hf_home`         |

---

## 3. Pull Latest Code

```bash
cd /workspace/trustworthy-personalized-ai
git pull
git checkout feat/sft-grpo-experiments
cd pipeline
```

---

## 4. Data (already assembled — skip unless regenerating)

The training data is committed at `data/train_partA_v3.jsonl` (1944 examples).
**No regeneration needed for the dissertation experiments.**

If you ever need to re-validate the data:

```bash
python validate_sft_data.py --input data/train_partA_v3.jsonl
# Checks: no leaked teacher constitution, think block ≥150 chars,
#         no banned placeholders, tool sequence integrity, answer tag present
# If >5% fail:
python validate_sft_data.py --input data/train_partA_v3.jsonl --fix \
    --output data/train_partA_v3_clean.jsonl
```

If you update student prompts in `sft_v3_generator.py`, patch existing JSONL:

```bash
python patch_student_prompts.py               # dry-run: shows counts
python patch_student_prompts.py --apply       # writes changes
```

---

## 5. SFT Training

All commands from `pipeline/`. Current config (set in `2_model_trainer.py`):

| Parameter           | Value                    | Why                                               |
|---------------------|--------------------------|---------------------------------------------------|
| `base_model`        | `unsloth/Qwen3-0.6B`     | IS the instruct model (no separate -Instruct variant exists) |
| `load_in_4bit`      | `False`                  | 16-bit LoRA — eliminates QLoRA quantisation noise, 0.6B fits in 16 GB |
| `lora_r`            | `64`                     | 4× capacity vs previous r=16; matches benchmark that beat 120B teacher |
| `lora_alpha`        | `16`                     | Standard α=r/4 ratio for complex multi-behaviour tasks |
| `num_train_epochs`  | `3`                      | Eval loss plateaued at epoch 2.5-3.0 in previous run |
| `learning_rate`     | `2e-4`                   | Standard for SFT                                  |
| `lr_scheduler_type` | `cosine`                 | Better convergence than linear for behavioural SFT |
| `load_best_model_at_end` | `True`              | Saves best eval-loss checkpoint, not the overfitted final one |
| `packing`           | `False`                  | Disabled — packing splits multi-turn tool sequences |

### Run SFT

```bash
python 2_model_trainer.py \
    --mode sft \
    --data_dir data \
    --output_dir models \
    --no_publish
```

Saves to `models/checkpoint_sft/`. The trainer automatically picks the best
checkpoint (lowest eval loss) at the end.

### Monitor training

```bash
# Tail loss in another terminal
watch -n 10 'python3 -c "
import json; s=json.load(open(\"models/checkpoint_sft/trainer_state.json\"))
for e in s[\"log_history\"][-5:]: print(e)
"'
```

Loss report auto-saved to `reports/training/checkpoint_sft/loss_history_<ts>.json`.

### After SFT — commit results

```bash
git add pipeline/reports/training/
git commit -m "results: SFT loss history r64 16bit cosine"
git push
```

---

## 6. Dissertation Experiment — Vanilla vs Fine-Tuned Benchmark

This is the core experimental workflow. Requires running two server sessions.
Use `tmux` to manage terminals:

```bash
tmux new-session -d -s server
tmux new-session -d -s bench
```

---

### Step A — Vanilla baseline (run BEFORE or independently of SFT)

**Terminal 1 — start vanilla server:**
```bash
tmux attach -t server
cd /workspace/trustworthy-personalized-ai/pipeline

python 3_infererence.py \
    --base_model unsloth/Qwen3-0.6B \
    --port 8000
# Wait for: "Model ready"
```

**Terminal 2 — run and save vanilla benchmark:**
```bash
tmux attach -t bench
cd /workspace/trustworthy-personalized-ai/pipeline

python 4_benchmark.py \
    --probe_only \
    --save_as_baseline \
    --model_label vanilla \
    --no_judge \
    --output_dir reports
```

Saves:
- `reports/constitution_probe_<ts>.json` — full per-question data
- `reports/constitution_probe_<ts>.csv` — per-principle scores table
- `reports/constitution_baseline.json` — drift reference for later runs

Stop vanilla server before loading SFT:
```bash
tmux send-keys -t server C-c
```

---

### Step B — Fine-tuned model benchmark

**Terminal 1 — start SFT server:**
```bash
tmux attach -t server

python 3_infererence.py \
    --model_dir models/checkpoint_sft \
    --port 8000
```

**Terminal 2 — run and save SFT benchmark:**
```bash
tmux attach -t bench

python 4_benchmark.py \
    --probe_only \
    --baseline reports/constitution_baseline.json \
    --model_label sft \
    --no_judge \
    --output_dir reports
```

Saves:
- `reports/constitution_probe_<ts2>.json`
- `reports/constitution_probe_<ts2>.csv`
- Prints drift vs baseline automatically

---

### Step C — Comparison table (offline, no server needed)

```bash
python compare_runs.py \
    reports/constitution_probe_<vanilla_ts>.json \
    reports/constitution_probe_<sft_ts>.json \
    --label_a vanilla \
    --label_b sft \
    --output_dir reports
```

Saves:
- `reports/comparison_vanilla_vs_sft.csv` — **paste directly into dissertation**
- `reports/comparison_vanilla_vs_sft.json` — full data with per-principle deltas

The CSV shows: `principle_id | vanilla | sft | delta | direction | hypothesis`
where hypothesis maps each probe to H1/H2/H3.

---

### Optional — Harness contribution (H2 experiment)

Measures how much the constitutional harness adds on top of model weights alone:

```bash
python 4_benchmark.py \
    --probe_only \
    --with_harness \
    --model_label sft_with_harness \
    --no_judge \
    --output_dir reports
```

This runs probes twice (with and without harness) and records the per-principle delta.
The delta isolates the harness contribution from the SFT contribution — evidence for H2.

---

### Commit all results

```bash
git add pipeline/reports/
git commit -m "results: vanilla vs sft constitutional probes"
git push
```

---

## 7. Benchmark Probe Coverage

Suite A (`--probe_only`) runs **22 probe groups × 3 questions = 66 probes**:

| Probe group              | What it tests                                  | Hypothesis |
|--------------------------|------------------------------------------------|------------|
| P1 Decompose First       | Substantive `<think>` block (>80 chars)        | H1         |
| P2+P3 Tool Discipline    | No hallucinated/unavailable tool calls         | H1         |
| P4 Math = Code           | `python_execute` for arithmetic                | H1         |
| P5 Real-Time Honesty     | Acknowledges missing live data                 | H1         |
| P6 Context Gate          | Asks clarifying question when context missing  | H1         |
| P7 Uncertainty           | Hedges uncertain, confident on known facts     | H1         |
| P8 Impossibility         | Names irreducible reason + redirects           | H1         |
| P9 Tradeoff              | Enumerates tradeoffs, no universal winner      | H3         |
| P10 Correct Tool Use     | Right tool for the question type               | H1         |
| P11 Tool Avoidance       | No tool for stable factual knowledge           | H1         |
| P12–P19 (remaining)      | Multi-step, assumptions, partial capability…   | H1/H3      |
| **P20 First Principles** | `<think>` contains decomposition + 5W+H scan  | H1 (new)   |
| **P21 Greedy Follow-up** | `<answer>` ends with targeted 5W+H question    | H1+H2 (new)|
| **H2 Memory Persistence**| Multi-turn: user fact in T1 applied in T2      | H2 (new)   |

Each probe records: `rule_score` (deterministic), `llm_score` (judge, optional),
`combined_score`, full `response`, `tool_trace`, `think_content`, `answer_content`,
`harness_violations`, `harness_retries`.

---

## 8. Running with LLM Judge (optional, costs API credits)

Add `--judge_model` to any probe run for semantic scoring on top of rule checks:

```bash
python 4_benchmark.py \
    --probe_only \
    --model_label sft \
    --judge_model nvidia_nim/moonshotai/kimi-k2.6 \
    --output_dir reports
```

Each probe question gets an LLM score (0–1) in addition to the regex rule check.
`combined_score = (rule_score + llm_score) / 2`.
Use `--no_judge` for fast rule-only runs during iteration.

---

## 8a. Auto-push results to GitHub (`--push`)

Add `--push` to any benchmark run to git-commit and push the saved report files to origin after each suite completes. Useful for long multi-suite runs on a remote GPU where you want results available as they arrive:

```bash
python 4_benchmark.py --probe --categories --adversarial --push
# Each suite (A, B, D) commits and pushes its JSON/CSV to origin immediately after saving.

python 4_benchmark.py --models unsloth/Qwen3-0.6B ./models/checkpoint_sft \
    --labels vanilla sft --probe --categories --push
# Hot-swap: each model × suite combo is pushed as it completes.
```

The commit message format is: `benchmark: <suite_name> results YYYY-MM-DD HH:MM [label]`. Failures (no remote, nothing-to-commit) are printed as warnings and do not abort the run.

---

## 9. Inference Server Reference

```bash
# Load base model (vanilla)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000

# Load SFT checkpoint
python 3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Health check
curl http://localhost:8000/health

# View server metrics (latency, tool call counts)
curl http://localhost:8000/metrics
```

The inference server:
- Imports student prompts from `sft_v3_generator.py` (canonical source — always in sync)
- Runs the constitutional harness (P1–P21 checks + steering on violation)
- Executes tools server-side: `python_execute`, `web_search`, `scratchpad`, `user_memory`
- Auto-injects 5W+H state and task status after every tool result

---

## 10. GRPO (Not applicable for this dissertation)

GRPO training is implemented in `2_model_trainer.py --mode grpo` but is not
part of the approved dissertation experiments. SFT-only is the research scope.

The GRPO reward weights (if ever needed) are:

| Component        | Weight | What it rewards                         |
|------------------|--------|-----------------------------------------|
| format           | 0.20   | Substantive `<think>` + `<answer>` tag  |
| accuracy         | 0.35   | Math code executes and matches answer   |
| tool_integrity   | 0.10   | No hallucinated tools                   |
| tool_quality     | 0.15   | Correct tool for question type          |
| constitution     | 0.10   | Constitutional harness rule pass rate   |
| greedy_followup  | 0.10   | `<answer>` ends with 5W+H question      |

---

## 11. Push Models to HuggingFace (optional)

```bash
# Upload SFT checkpoint after training
python 2_model_trainer.py \
    --mode publish \
    --output_name checkpoint_sft \
    --hf_username AjinkyaTaranekar
```

Requires `HF_TOKEN` with write access to `AjinkyaTaranekar/trustworthy-ai-*`.

---

## 12. Troubleshooting

### CUDA out of memory during SFT

The model loads in bf16 (`load_in_4bit=False`). On 16 GB this is fine for 0.6B.
If OOM occurs (e.g. on a smaller GPU):

```bash
# Edit MODEL_CONFIG in 2_model_trainer.py — add load_in_4bit back temporarily:
# "load_in_4bit": True,   ← QLoRA fallback
# "max_seq_length": 3072  ← or reduce to 2048
```

### Stale file handle on /workspace

```bash
python3 -c "import shutil; shutil.rmtree('/workspace/.hf_home/hub/', ignore_errors=True)"
export $(grep -v '^#' pipeline/.env | xargs)
```

### Benchmark returns SERVER ERROR

```bash
curl http://localhost:8000/health   # check server is up
# Server logs show [REQ] and [RESP] lines when healthy
```

### Instance ran out of disk

```bash
rm -rf models/checkpoint_sft_merged models/checkpoint_sft_gguf
```

### Validation rejects all rows after student prompt update

The validator now checks for leaked teacher constitution phrases, not word count.
If rows fail with `teacher_constitution_leaked`, re-patch:
```bash
python patch_student_prompts.py --apply
```

---

## 13. Script Reference

| Script                     | Role                                           | Step       |
|----------------------------|------------------------------------------------|------------|
| `sft_v3_generator.py`      | Teacher→student distillation + canonical prompts | data gen |
| `validate_sft_data.py`     | Quality gate (5 invariants, checks teacher leak) | validation |
| `patch_student_prompts.py` | Sync student prompts across all JSONL entries  | maintenance|
| `2_model_trainer.py`       | SFT training (+ GRPO reward fns, unused)       | step 5     |
| `3_infererence.py`         | FastAPI inference server (loads prompts from sft_v3_generator) | step 9 |
| `4_benchmark.py`           | Constitutional benchmark — 22 probe groups     | step 6     |
| `compare_runs.py`          | Offline comparison: two JSON files → CSV table | step 6C    |
| `scratchpad.py`            | Session working memory (5wh_state, tasks, notes)| all       |
| `user_memory.py`           | Persistent 5W+H user ontology (JSON on disk)   | all        |
| `constitutional_harness.py`| Real-time P1–P21 checks + steering at inference| all        |
| `pipeline_tools.py`        | Tool implementations (python, web, memory)     | all        |
| `config.py`                | Shared pipeline configuration                  | all        |
| `empathy.py`               | Appraisal analysis module                      | all        |
| `ontology_verifier.py`     | Ontology-grounded response check               | all        |
| `user_modelling.py`        | GraphRAG user model                            | all        |

---

## 14. Data Files

| File                                        | Producer              | Consumer                   |
|---------------------------------------------|-----------------------|----------------------------|
| `data/train_partA_v3.jsonl`                 | committed (1944 rows) | `2_model_trainer.py`       |
| `reports/constitution_baseline.json`        | `4_benchmark.py`      | `4_benchmark.py` (drift)   |
| `reports/constitution_probe_<ts>.json`      | `4_benchmark.py`      | `compare_runs.py`          |
| `reports/constitution_probe_<ts>.csv`       | `4_benchmark.py`      | dissertation tables        |
| `reports/comparison_<a>_vs_<b>.csv`         | `compare_runs.py`     | dissertation results table |
| `reports/training/*/loss_history_<ts>.json` | `2_model_trainer.py`  | git / loss analysis        |
