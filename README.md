# Trustworthy Personalised AI

MSc dissertation pipeline — *Architecting Trust and Empathy in Conversational AI* (Trinity College Dublin, CS7CS6).

Trains a small language model (Qwen3-0.6B) to be honest about its own capabilities, use tools correctly, and refuse to hallucinate. Two pipelines are included: a quick V1 for prototyping and a constitution-driven V2 for the full dissertation dataset.

---

## Quick start

```bash
cd pipeline
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**API keys (V2 pipeline only):** copy `.env.example` to `.env` and fill in your key — no `export` needed.

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

**See [PIPELINE.md](PIPELINE.md) for the full step-by-step guide.**

---

## Two pipelines at a glance

| | V1 — Interleaved | V2 — Constitution |
|---|---|---|
| Purpose | Rapid prototype | Dissertation-quality dataset |
| Dataset size | ~5–50 examples | ~2,700 examples |
| Data source | Fixed templates | LLM-generated + critique loop |
| API needed | No | Yes (Anthropic / OpenAI / Ollama) |
| Scripts | `1_dataset_generator.py` → `2_model_trainer.py` | `sft_*.py` × 5 → `2_model_trainer.py` |
| Cost | Free | ~$10–15 (Claude Sonnet) |

---

## Repository layout

```
pipeline/
├── 1_dataset_generator.py      V1: generate training data from templates
├── 2_model_trainer.py          SFT training with LoRA (used by both V1 and V2)
├── 3_infererence.py            Inference server — loads model once, serves via HTTP API
├── 4_benchmark.py              Benchmark client — calls 3_infererence.py via HTTP
├── 5_context_degradation.py    Context length degradation study
├── sft_question_generator.py   V2 step 1a — behavioural questions (11 categories)
├── sft_gold_response_generator.py  V2 step 1b — teacher/critic loop (--critic_model supported)
├── sft_math_question_generator.py  V2 step 2a — math/code questions (7 types)
├── sft_rejection_sampler.py    V2 step 2b — keep only correct executions
├── sft_dataset_assembler.py    V2 step 3 — merge, filter, train/eval split
├── constitution.md             The 19 principles the model is trained against
├── .env.example                Copy to .env and add your API keys
├── data/                       Generated JSONL datasets (git-ignored)
├── models/                     Saved checkpoints (git-ignored)
└── reports/                    Benchmark JSON + HTML viewer
wiki/                           Living research wiki (Obsidian)
docs/                           PDFs, dissertation drafts, literature notes
```

> **3 + 4 are server/client.** Start `3_infererence.py` first (it loads the GPU model), then `4_benchmark.py` calls it over HTTP. This means evals require no model reloading between runs.

---

## V1 — Interleaved (quick prototype)

```bash
# 1. Generate dataset
python 1_dataset_generator.py --variant interleaved --train_size 5

# 2. Train
python 2_model_trainer.py --data_dir data --output_dir models --output_name qwen3-0.6b

# 3. Start inference server (keep this running)
python 3_infererence.py --model_dir ./models/qwen3-0.6b --port 8000

# 4. Benchmark (in a second terminal)
python 4_benchmark.py --server_url http://localhost:8000

# 4b. Base vs fine-tuned comparison (start a second server on port 8001)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001
python 4_benchmark.py --server_url http://localhost:8000 --compare_url http://localhost:8001
```

---

## Context degradation study

Tests whether SFT accuracy holds as conversation context grows. Uses greedy decoding for reproducibility. Maps the "first failure token count" for each ablation condition.

```bash
# Fine-tuned model only
python pipeline/5_context_degradation.py --server_url http://localhost:8000

# Base vs fine-tuned (start base model on port 8001)
python pipeline/3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8001
python pipeline/5_context_degradation.py \
    --server_url http://localhost:8000 \
    --compare_url http://localhost:8001
```

---

## V2 — Constitution-based (full pipeline)

```bash
# 1a. Behavioural questions (~1,700)
python sft_question_generator.py --output data/questions_partA.jsonl

# 1b. Teacher generates + critiques gold responses (~$10–15)
python sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output data/train_partA.jsonl

# 2a. Math/code questions (~1,050)
python sft_math_question_generator.py --output data/questions_partB.jsonl

# 2b. Rejection sampling — keep only correct executions
python sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output data/train_partB.jsonl \
  --use_api_model

# 3. Assemble final dataset (~2,700 examples)
python sft_dataset_assembler.py

# 4. Train on V2 data
python 2_model_trainer.py --data_path data/train_sft_v2.jsonl
```

Keys are read from `pipeline/.env` automatically — no shell exports needed. All V2 scripts accept `--model` and `--api_base` to swap providers.

---

## Viewing benchmark results

```bash
cd reports
python server.py
# open http://localhost:8000/view_benchmark.html
```

---

## Constitutional drift detection

Run the constitutional probe suite after every training phase to catch drift before it compounds.

```bash
# Step 1: right after SFT — save the baseline score
python 4_benchmark.py --server_url http://localhost:8000 --probe_only --save_as_baseline

# Step 2: after each GRPO checkpoint — compare against baseline
python 4_benchmark.py --server_url http://localhost:8000 \
    --probe_only --baseline reports/constitution_baseline.json
# Prints "DRIFT WARNING" and sets drift_warning: true in JSON if score drops ≥ 5pp

# Mitigation if drift detected (in order of escalation):
# 1. Roll back to last good checkpoint
# 2. Increase β (KL coefficient) in GRPO config
# 3. Add SFT replay buffer (20% of each batch) to GRPO training
```

12 probes cover: CAPABILITY_CHECK presence, tool inventory/discipline, math=code, real-time honesty, context gate, impossibility acknowledgment, tradeoff presentation, tool avoidance, hold under pressure, knowledge cutoff, single clarification, and explicit I-don't-know. All checks are regex/rule-based — no model judge.

---

## Adding new tools to the inference server

Tools can be registered at runtime without restarting:

```bash
curl -X POST http://localhost:8000/v1/tools/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},
    "python_code": "import urllib.request, json\ndef tool_fn(city=\"Dublin\", **_):\n    return f\"Weather for {city}: 12°C, cloudy (mock)\""
  }'

# List all registered tools
curl http://localhost:8000/v1/tools

# View live metrics
curl http://localhost:8000/metrics
```

---

## Constitution

The 19 principles the model is trained to follow — capability honesty, correct tool use, honest refusal — live in `pipeline/constitution.md`. Read it first to understand the target behaviour.
