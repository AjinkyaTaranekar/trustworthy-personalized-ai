# Trustworthy Personalized AI

Building a Trustworthy Personalized LLM with Interleaved Thinking and First Principles

## Setup

### Create and Activate Virtual Environment

Navigate to the `pipeline` directory and create a virtual environment:

```bash
cd pipeline
python3 -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Install Requirements

Install dependencies in the virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

Generate synthetic training data:

```bash
python3 1_dataset_generator.py --variant interleaved --train_size 5
```

**Options:**
- `--variant`: Dataset variant (e.g., `interleaved`)
- `--train_size`: Number of training examples to generate

This will create a JSONL dataset in `data/train_interleaved.jsonl`

### 2. Train Model

Train the model on generated data:

```bash
python3 2_model_trainer.py --data_dir data --output_dir models --output_name qwen3-0.6b
```

**Options:**
- `--data_dir`: Directory containing training data
- `--output_dir`: Directory to save trained models
- `--output_name`: Name of the output checkpoint directory (default: `checkpoint_sft`)
- `--skip_if_exists`: Skip training if model already exists

The trainer uses **Qwen3-0.6B** as the base model with LoRA fine-tuning.

### 3. Run Inference on the Trained Model

To run inference using the trained model, execute the following command:

```bash
python3 3_infererence.py --prompt "calculate 100*20-10+(50/12.5)"
```

Compare the model with default base model:
```bash
python3 3_infererence.py --compare --prompt "calculate 159*234/24-5"
```

Compare the model with a different base model:
```bash
python3 3_infererence.py --base_model "unsloth/Qwen3-4B" --compare --prompt "calculate 159*234/24-5"
```

**Options:**
- `--prompt`: User prompt for the model
- `--model_dir`: Path to the fine-tuned model (default: `./models/checkpoint_sft`)
- `--base_model`: Base model for comparison (default: `unsloth/Qwen3-0.6B`)
- `--compare`: Compare custom model with base model
- `--max_new_tokens`: Max tokens to generate (default: 2048)
- `--max_iterations`: Max tool call iterations (default: 10)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_dir`: Directory to save reports (default: `./reports`)

**Report Saving:**

Inference results are automatically saved as JSON reports in the `reports/` directory:
- Single model: `inference_YYYYMMDD_HHMMSS.json`
- Comparison: `comparison_YYYYMMDD_HHMMSS.json`

Each report includes:
- Timestamp and prompt
- Model configuration
- Complete conversation history
- Number of turns for each model (in comparison mode)

### 4. Run Benchmark

Run a comprehensive benchmark with multiple questions to evaluate model performance:

```bash
python3 4_benchmark.py --compare --model_dir ./models/trustworthy-qwen3-0.6b/
```

Run benchmark on a single model:
```bash
python3 4_benchmark.py --model_dir ./models/trustworthy-qwen3-0.6b/
```

**Options:**
- `--model_dir`: Path to the fine-tuned model checkpoint (default: `./models/checkpoint_sft`)
- `--base_model`: Base model name on HF Hub (default: `unsloth/Qwen3-0.6B`)
- `--compare`: Run both base and custom models for comparison
- `--questions`: Comma-separated list of custom questions (overrides default benchmark questions)
- `--max_new_tokens`: Max tokens to generate per turn (default: 2048)
- `--max_tool_iters`: Max tool-call iterations per question (default: 10)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_dir`: Directory to save benchmark reports (default: `./reports`)

**Benchmark Reports:**

Results are saved as JSON in the `reports/` directory with filename `benchmark_YYYYMMDD_HHMMSS.json`. Each report includes:
- Configuration details
- Per-turn metrics (tokens, generation time, tool calls)
- Context growth analysis
- Summary statistics across all turns
- Full conversation history

**Viewing Results:**

To view benchmark results in a web interface:

```bash
cd reports
python3 server.py
```

Then open `view_benchmark.html` in your browser at `http://localhost:8000/view_benchmark.html`

---

## SFT v2 Pipeline (Constitution-Based, Domain-Unbounded)

The v2 pipeline replaces the 42-template approach with a **constitution-driven** data generation
system. Instead of scripting specific scenarios, the model learns 19 principles that apply to
any question — covering capability honesty, tool discipline, and honest refusal.

**Tools the model learns to use:**
| Tool | Purpose |
|------|---------|
| `python_execute(code)` | Precision arithmetic and computation |
| `web_search(query)` | Real-time data, current events, entity facts, proper nouns |
| `read_url(url)` | Follow up on a specific search result |
| `get_datetime()` | Current date/time for time-aware responses |

**All v2 scripts use [litellm](https://github.com/BerriAI/litellm) — swap the `--model` string
to use any provider:**

```
Anthropic : claude-sonnet-4-5            (set ANTHROPIC_API_KEY)
OpenAI    : gpt-4o-mini                  (set OPENAI_API_KEY)
Ollama    : ollama/llama3.2              (set --api_base http://localhost:11434)
Groq      : groq/llama-3.1-70b-versatile (set GROQ_API_KEY)
```

### V2 Step 1a — Generate Behavioral Questions (Part A)

Generates diverse questions across 9 behavioral categories using an LLM.
Categories: user-context, real-time, impossible tasks, subjective tradeoffs,
adversarial pressure, knowledge boundary, multi-step clarification, ambiguous requests,
entity facts requiring web search.

```bash
# Generate all categories (default counts, ~1,700 total)
python3 sft_question_generator.py --output data/questions_partA.jsonl

# Single category, 10 questions, using Ollama
python3 sft_question_generator.py \
  --category real_time_dependent \
  --count 10 \
  --model ollama/llama3.2 \
  --api_base http://localhost:11434 \
  --output data/sample_questions.jsonl
```

**Options:**
- `--category`: Category to generate (`all` or a specific one, default: `all`)
- `--count`: Questions per category (overrides per-category defaults)
- `--model`: litellm model string (default: `claude-sonnet-4-5`)
- `--api_base`: Custom API base URL (for Ollama or other local servers)
- `--batch_size`: Questions per API call (default: 50, reduce if hitting limits)
- `--output`: Output JSONL path

### V2 Step 1b — Generate Gold Responses (Part A)

For each question: teacher model generates a draft → critiques it against the constitution
→ revises if violations found. Training examples use only the revised response.

```bash
# Process all questions
python3 sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output data/train_partA.jsonl

# Quick smoke test (5 examples)
python3 sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output data/sample_gold.jsonl \
  --max 5

# Resume interrupted run
python3 sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output data/train_partA.jsonl \
  --resume

# Use Ollama locally
python3 sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --model ollama/llama3.2 \
  --api_base http://localhost:11434 \
  --output data/train_partA.jsonl
```

**Options:**
- `--questions`: Input JSONL from Step 1a
- `--output`: Output training JSONL
- `--model`: litellm model string (default: `claude-sonnet-4-5`)
- `--api_base`: Custom API base URL
- `--max`: Max examples to process
- `--resume`: Skip already-processed questions (safe to re-run)

**Estimated API cost:** ~$10–15 for 1,500 examples with Claude Sonnet.

### V2 Step 2a — Generate Math Questions (Part B)

Generates verifiable math/code questions with known correct answers across 7 types:
arithmetic, algebra, geometry, statistics, unit conversions, word problems, and
no-tool control cases (where the correct answer is to refuse to compute without a tool).

```bash
# Generate all types (~1,050 questions)
python3 sft_math_question_generator.py --output data/questions_partB.jsonl

# Single type, 10 questions
python3 sft_math_question_generator.py \
  --type arithmetic \
  --count 10 \
  --output data/sample_math.jsonl

# Skip answer verification (faster, less quality assurance)
python3 sft_math_question_generator.py \
  --output data/questions_partB.jsonl \
  --no_verify
```

**Options:**
- `--type`: Question type (`all` or specific: `arithmetic`, `algebra`, `geometry`, `statistics`, `unit_conversion`, `word_problems`, `no_tool_control`)
- `--count`: Questions per type (overrides defaults)
- `--model`: litellm model string (default: `claude-haiku-4-5-20251001`)
- `--api_base`: Custom API base URL
- `--verify` / `--no_verify`: Whether to execute a sample of answers for QA (default: verify)
- `--output`: Output JSONL path

### V2 Step 2b — Rejection Sampling (Part B)

For each math question, generates N candidate responses and keeps only those where
the code executes correctly and produces the right answer.

```bash
# Using litellm model as candidate generator
python3 sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output data/train_partB.jsonl \
  --use_api_model \
  --candidates 8

# Using a local model checkpoint
python3 sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output data/train_partB.jsonl \
  --model_path ./models/checkpoint_sft \
  --candidates 8

# Use Ollama
python3 sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output data/train_partB.jsonl \
  --use_api_model \
  --api_model ollama/llama3.2 \
  --api_base http://localhost:11434
```

**Scoring:**
- `+1` Code executes AND answer matches expected
- ` 0` No code used (mental approximation)
- `-1` Code fails or wrong answer

For no-tool questions: `+1` if model honestly refuses to compute, `-1` if it guesses.

**Options:**
- `--questions`: Input JSONL from Step 2a
- `--output`: Output training JSONL
- `--model_path`: Local HuggingFace checkpoint (uses local model for generation)
- `--use_api_model`: Use litellm instead of local model
- `--api_model`: litellm model string (default: `claude-haiku-4-5-20251001`)
- `--api_base`: Custom API base URL
- `--candidates`: Candidates per question (default: 8)
- `--min_score`: Minimum score to accept (default: 1)
- `--resume`: Safe to re-run — skips already-accepted questions

### V2 Step 3 — Assemble Final Dataset

Merges Part A + Part B, applies quality filters (requires `CAPABILITY_CHECK` in every
`<think>` block), deduplicates, balances categories, and produces the train/eval split.

```bash
python3 sft_dataset_assembler.py

# Custom paths
python3 sft_dataset_assembler.py \
  --part_a data/train_partA.jsonl \
  --part_b data/train_partB.jsonl \
  --output_dir data/
```

**Output files:**
- `data/train_sft_v2.jsonl` — training set (~2,700 examples)
- `data/eval_sft_v2.jsonl` — eval set (10% held out)
- `data/sft_v2_stats.json` — category/pipeline breakdown

**Options:**
- `--part_a` / `--part_b`: Input JSONL paths
- `--output_dir`: Where to write output files
- `--eval_frac`: Eval fraction (default: 0.10)
- `--max_per_category`: Cap per category to prevent imbalance (default: 400)

### V2 Full Pipeline (Quick Reference)

```bash
# 1. Generate questions
python3 sft_question_generator.py --output data/questions_partA.jsonl
python3 sft_math_question_generator.py --output data/questions_partB.jsonl

# 2. Generate gold responses
python3 sft_gold_response_generator.py \
  --questions data/questions_partA.jsonl \
  --output data/train_partA.jsonl
python3 sft_rejection_sampler.py \
  --questions data/questions_partB.jsonl \
  --output data/train_partB.jsonl \
  --use_api_model

# 3. Assemble final dataset
python3 sft_dataset_assembler.py

# 4. Train (same script as v1, update --data_path)
python3 2_model_trainer.py --data_path data/train_sft_v2.jsonl
```

### Constitution

The 19 principles governing all training data live in `pipeline/constitution.md`.
Read it to understand what behaviors the model is trained to exhibit and why.

