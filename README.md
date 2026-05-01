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
├── 3_infererence.py            Single-prompt inference + comparison mode
├── 4_benchmark.py              Multi-question benchmark with JSON reports
├── 5_context_degradation.py    Context length degradation study
├── sft_question_generator.py   V2 step 1a — behavioural questions (9 categories)
├── sft_gold_response_generator.py  V2 step 1b — teacher critique + revise loop
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

---

## V1 — Interleaved (quick prototype)

```bash
# 1. Generate dataset
python 1_dataset_generator.py --variant interleaved --train_size 5

# 2. Train
python 2_model_trainer.py --data_dir data --output_dir models --output_name qwen3-0.6b

# 3. Inference
python 3_infererence.py --prompt "calculate 100*20-10+(50/12.5)"

# 4. Benchmark
python 4_benchmark.py --model_dir ./models/qwen3-0.6b --compare
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

## Constitution

The 19 principles the model is trained to follow — capability honesty, correct tool use, honest refusal — live in `pipeline/constitution.md`. Read it first to understand the target behaviour.
