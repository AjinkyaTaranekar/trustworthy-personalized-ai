# Unified Pipeline — End to End (Vast.ai Guide)

Complete runbook for training on a GPU cloud instance from scratch.
Follow every numbered step in order. SFT and GRPO are separate sections.

---

## 0. Rent a GPU on Vast.ai

**Minimum spec:**
- A100 40 GB (recommended — fits max_seq_length=3072 comfortably)
- A4000 16 GB (marginal — drop max_seq_length to 2048 in MODEL_CONFIG if OOM)
- Disk: 80 GB+
- Image: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` or any CUDA 12.x image

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
# Skip if you're in a hurry — Xformers fallback is fine
pip install flash-attn --no-build-isolation
```

---

## 2. Environment Variables (.env)

Create `pipeline/.env` with all required keys:

```bash
nano pipeline/.env
```

## 3. Pull Latest Code

If the repo was already cloned and you're continuing a session:

```bash
cd /workspace/trustworthy-personalized-ai
git pull
```

Switch to the `experiments` branch if not already on it:

```bash
git checkout feat/sft-grpo-experiments
```
---

## 4. Data Pipeline (skip if data/train_sft_v3.jsonl already exists)

All commands run from `/workspace/trustworthy-personalized-ai/pipeline`.

```bash
cd /workspace/trustworthy-personalized-ai/pipeline
```

### Step 4a — Generate Behavioural Questions

```bash
python sft_question_generator.py \
    --count 200 --type all \
    --output data/questions_v3.jsonl
```

Produces a JSONL of diverse questions across all 19 categories.

### Step 4b — Generate Verified Math Data (Part B)

```bash
python sft_math_pipeline.py --resume
```

Loads GSM8K + MATH datasets, uses the teacher LLM to write and execute Python
for each question, verifies against the trusted answer.
Outputs `data/train_partB.jsonl` (~794 examples).

Optional flags:
```bash
--gsm8k_count 300 --math_count 700   # custom split
--math_max_level 2                   # easier questions only
--smoke                              # 5-question quick test
```

### Step 4c — Generate Behavioural Gold Responses (Teacher Distillation)

```bash
python sft_v3_generator.py \
    --questions data/questions_v3.jsonl \
    --output data/train_v3.jsonl \
    --model nvidia_nim/moonshotai/kimi-k2.6
```

For each question: teacher generates with 25-principle constitution, tool calls
execute live (web search via Exa, Python via subprocess), teacher system prompt
is swapped to the short student prompt before saving.

### Step 4d — Validate

```bash
python validate_sft_data.py --input data/train_v3.jsonl
# If >5% fail:
python validate_sft_data.py --input data/train_v3.jsonl --fix
```

### Step 4e — Assemble Final Dataset

```bash
python sft_dataset_assembler.py \
    --part_a data/train_v3.jsonl \
    --part_b data/train_partB.jsonl \
    --output_dir data/
```

Outputs `data/train_sft_v3.jsonl` (train) + `data/eval_sft_v3.jsonl` (eval).
Check the summary printed at the end — should show 1800+ train examples.

---

## 5. SFT Training

All commands from `pipeline/`. Uses max_seq_length=3072 and 4-bit LoRA.

### Option A — Single-run SFT (faster, less optimal)

```bash
python 2_model_trainer.py \
    --mode sft \
    --data_dir data \
    --output_dir models \
    --no_publish
```

Saves to `models/checkpoint_sft/`. Skip `--no_publish` if you want auto-upload
to HuggingFace (requires HF_TOKEN with write permissions).

### Option B — Curriculum SFT (recommended for better format grounding)

Three staged runs — each builds on the previous checkpoint:

```bash
# Stage 1: short no-tool examples → teach <think>...<answer> syntax
python 2_model_trainer.py --mode sft \
    --curriculum_stage 1 \
    --output_name checkpoint_sft_s1 \
    --data_dir data --output_dir models --no_publish

# Stage 2: all examples → complex multi-tool reasoning
python 2_model_trainer.py --mode sft \
    --curriculum_stage 2 \
    --from_checkpoint models/checkpoint_sft_s1 \
    --output_name checkpoint_sft_s2 \
    --data_dir data --output_dir models --no_publish

# Stage 3: all + 20% stage-1 replay → anti-drift
python 2_model_trainer.py --mode sft \
    --curriculum_stage 3 \
    --from_checkpoint models/checkpoint_sft_s2 \
    --output_name checkpoint_sft \
    --data_dir data --output_dir models --no_publish
```

### Monitor training loss

```bash
# While training runs, tail the loss in another terminal
tail -f models/checkpoint_sft/loss_history.json 2>/dev/null || \
    watch -n 10 'python3 -c "
import json; s=json.load(open(\"models/checkpoint_sft/trainer_state.json\"))
for e in s[\"log_history\"][-5:]: print(e)
"'
```

Loss report is also saved to `reports/training/checkpoint_sft/loss_history_<timestamp>.json`
which is git-tracked — commit and push after training.

### After SFT completes — copy loss report to git

```bash
# Already saved automatically to reports/training/ — just commit
git add pipeline/reports/training/
git commit -m "chore: add SFT loss history"
git push
```

---

## 6. SFT Inference and Constitution Baseline

Use tmux to run inference server and benchmark in parallel.

```bash
tmux new-session -d -s inference
tmux new-session -d -s benchmark
```

### Terminal 1 — Start SFT inference server

```bash
tmux attach -t inference
cd /workspace/trustworthy-personalized-ai/pipeline

python 3_infererence.py \
    --model_dir models/checkpoint_sft \
    --port 8000

# Wait for: "Model ready: models/checkpoint_sft"
```

### Terminal 2 — Save constitution baseline (run once before GRPO)

```bash
tmux attach -t benchmark
cd /workspace/trustworthy-personalized-ai/pipeline

# Save baseline — this is the SFT reference point for drift detection
python 4_benchmark.py \
    --probe_only \
    --save_as_baseline \
    --no_judge \
    --output_dir reports

# Full benchmark suite (rule-based only, fast)
python 4_benchmark.py \
    --probe --categories --adversarial \
    --no_judge \
    --output_dir reports

# Full benchmark with LLM judge (costs API credits, more accurate)
python 4_benchmark.py \
    --probe --categories \
    --judge_model nvidia_nim/moonshotai/kimi-k2.6 \
    --output_dir reports
```

Results saved to `reports/constitution_probe_<timestamp>.json`.

### Commit SFT benchmark results

```bash
git add pipeline/reports/
git commit -m "results: SFT constitution probe baseline"
git push
```

Stop the SFT inference server before starting GRPO training (releases VRAM):
```bash
tmux send-keys -t inference C-c
```

---

## 7. GRPO Training

Starts from the SFT checkpoint. Runs reward_type=d (full 5-component reward).

```bash
cd /workspace/trustworthy-personalized-ai/pipeline

python 2_model_trainer.py \
    --mode grpo \
    --sft_checkpoint models/checkpoint_sft \
    --data_dir data \
    --output_dir models \
    --reward_type d \
    --no_publish
```

**What the reward signal trains toward:**
- Format (0.25): non-empty `<think>` block (>100 chars = full score) + `<answer>`
- Accuracy (0.35): math code executes and matches expected answer
- Tool integrity (0.10): no hallucinated or profile-unavailable tools
- Tool quality (0.20): correct tool for question type (get_datetime for time questions,
  user_memory_read for user context, web_search for entity facts, python_execute for math)
- Constitution (0.10): rule_check_response pass rate

Saves to `models/checkpoint_grpo_d/`.

### Monitor GRPO rewards

```bash
# In a separate terminal — watch reward components per step
tail -f models/checkpoint_grpo_d/grpo_loss_history.json 2>/dev/null || \
    watch -n 15 'python3 -c "
import json; s=json.load(open(\"models/checkpoint_grpo_d/trainer_state.json\"))
for e in s[\"log_history\"][-3:]: print(e)
"'
```

Healthy GRPO: reward should increase from ~0.3 toward ~0.6-0.7 over training.
If reward stays flat or drops, check that tool_quality and format rewards are firing.

### After GRPO completes — copy loss report

```bash
git add pipeline/reports/training/
git commit -m "chore: add GRPO loss history"
git push
```

---

## 8. GRPO Inference and Final Benchmark

### Terminal 1 — Start GRPO inference server

```bash
tmux attach -t inference
cd /workspace/trustworthy-personalized-ai/pipeline

python 3_infererence.py \
    --model_dir models/checkpoint_grpo_d \
    --port 8000
```

### Terminal 2 — Run full benchmark and compare against SFT baseline

```bash
tmux attach -t benchmark
cd /workspace/trustworthy-personalized-ai/pipeline

# Rule-based only (fast, ~10 min)
python 4_benchmark.py \
    --probe --categories --adversarial \
    --baseline reports/constitution_baseline.json \
    --no_judge \
    --output_dir reports

# With LLM judge (full accuracy, ~30-40 min, costs API credits)
python 4_benchmark.py \
    --probe --categories \
    --baseline reports/constitution_baseline.json \
    --judge_model nvidia_nim/moonshotai/kimi-k2.6 \
    --output_dir reports
```

The drift report shows GRPO score vs SFT baseline — this delta is your core result.

### Run Ablation C (format+accuracy only) for dissertation comparison

```bash
# Train ablation variant
python 2_model_trainer.py \
    --mode grpo \
    --sft_checkpoint models/checkpoint_sft \
    --data_dir data \
    --output_dir models \
    --reward_type c \
    --output_name checkpoint_grpo_c \
    --no_publish

# Serve and benchmark ablation
python 3_infererence.py --model_dir models/checkpoint_grpo_c --port 8001 &
python 4_benchmark.py \
    --server_url http://localhost:8001 \
    --probe --no_judge \
    --baseline reports/constitution_baseline.json \
    --output_dir reports
```

### Commit all results

```bash
git add pipeline/reports/
git commit -m "results: GRPO_d and GRPO_c constitution probes"
git push
```

---

## 9. Push Models to HuggingFace (optional)

If you ran with `--no_publish` and want to upload separately:

```bash
cd /workspace/trustworthy-personalized-ai/pipeline

# Upload SFT checkpoint
python 2_model_trainer.py \
    --mode publish \
    --output_name checkpoint_sft \
    --hf_username AjinkyaTaranekar

# Upload GRPO checkpoint
python 2_model_trainer.py \
    --mode publish \
    --output_name checkpoint_grpo_d \
    --hf_username AjinkyaTaranekar
```

Requires HF_TOKEN with write permissions to `AjinkyaTaranekar/trustworthy-ai-*`.

---

## 10. Troubleshooting

### Stale file handle on /workspace

```bash
python3 -c "import shutil; shutil.rmtree('/workspace/.hf_home/hub/', ignore_errors=True)"
# Re-export env and retry
export $(grep -v '^#' pipeline/.env | xargs)
```

### CUDA out of memory during SFT

```bash
# Edit MODEL_CONFIG in 2_model_trainer.py:
# "max_seq_length": 2048   ← drop from 3072
# Then reduce batch if still OOM:
# "per_device_train_batch_size": 1
# "gradient_accumulation_steps": 16
```

### GRPO TypeError: unexpected keyword argument

```bash
# All TRL param names are auto-detected at runtime via inspect.
# If a new name mismatch appears, check:
python3 -c "from trl import GRPOConfig; help(GRPOConfig.__init__)" | grep -A1 "epsilon\|beta\|completion"
```

### Model loaded but benchmark returns SERVER ERROR

```bash
# Check server health
curl http://localhost:8000/health

# Check server logs — inference terminal should show [REQ] and [RESP] lines
# If tool calls return empty code, confirm you have the parser fix:
git log --oneline pipeline/3_infererence.py | head -3
```

### Instance ran out of disk

```bash
# Merged model dirs are large (~3GB each) — delete if not needed
rm -rf models/checkpoint_sft_merged models/checkpoint_grpo_d_merged
# GGUF dirs too
rm -rf models/checkpoint_sft_gguf models/checkpoint_grpo_d_gguf
```

---

## 11. Script Reference

| Script                    | Role                              | Step |
|---------------------------|-----------------------------------|------|
| sft_question_generator.py | Generate behavioural questions    | 4a   |
| sft_math_pipeline.py      | Generate verified math data       | 4b   |
| sft_v3_generator.py       | Teacher→student distillation      | 4c   |
| validate_sft_data.py      | Quality gate (5 invariants)       | 4d   |
| sft_dataset_assembler.py  | Filter, dedupe, augment, split    | 4e   |
| 2_model_trainer.py        | SFT curriculum + GRPO             | 5, 7 |
| 3_infererence.py          | FastAPI inference server          | 6, 8 |
| 4_benchmark.py            | Constitutional benchmark client   | 6, 8 |
| 5_context_degradation.py  | Ablation: context window stress   | opt  |
| config.py                 | Shared pipeline configuration     | all  |
| pipeline_tools.py         | Tool registry (python/web/etc.)   | all  |
| constitutional_harness.py | Real-time constitution scoring    | all  |
| scratchpad.py             | Session scratchpad store          | all  |
| user_memory.py            | Persistent user memory store      | all  |
| empathy.py                | Appraisal analysis module         | all  |
| ontology_verifier.py      | Ontology-grounded response check  | all  |
| user_modelling.py         | GraphRAG user model               | all  |

---

## 12. Data Files

| File                      | Producer              | Consumer              |
|---------------------------|-----------------------|-----------------------|
| data/questions_v3.jsonl   | sft_question_generator| sft_v3_generator      |
| data/train_v3.jsonl       | sft_v3_generator      | sft_dataset_assembler |
| data/train_partB.jsonl    | sft_math_pipeline     | sft_dataset_assembler |
| data/train_sft_v3.jsonl   | sft_dataset_assembler | 2_model_trainer       |
| data/eval_sft_v3.jsonl    | sft_dataset_assembler | 2_model_trainer       |
| reports/constitution_baseline.json | 4_benchmark  | 4_benchmark (drift)   |
| reports/training/*/loss_history_*.json | 2_model_trainer | git / analysis |

---

## 13. Environment Variable Reference

| Variable              | Required for               | Where to get it                        |
|-----------------------|----------------------------|----------------------------------------|
| HF_TOKEN              | Pushing models to HuggingFace | huggingface.co → Settings → Tokens |
| NVIDIA_NIM_API_KEYS   | Teacher distillation + LLM judge | build.nvidia.com → API Keys     |
| EXA_API_KEY           | Web search tool (data gen + inference) | exa.ai → Dashboard          |
| HF_HOME               | Cache location (optional)  | Set to /workspace/.hf_home             |
