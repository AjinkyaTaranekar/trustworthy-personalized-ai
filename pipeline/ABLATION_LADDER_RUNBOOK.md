# Ablation Ladder — Simplified Runbook (5 conditions)

Command-by-command guide to train, benchmark, and compare the five-rung ablation ladder.
Run **every command from `pipeline/`** on the GPU box, on branch `feat/thinker-executor-sft`
(the benchmark `--tool_mode`, Suite E, and `analyze_experiments.py` do **not** exist on `main`).
For deeper detail see `pipeline.md` (single-model) and `THINKER_EXECUTOR_RUNBOOK.md` (dual).

## The five conditions

| # | Label | Model | Trained on | Benchmark `--tool_mode` |
|---|---|---|---|---|
| C0 | `vanilla_base` | base Qwen3-0.6B | — | `xml` (base ignores XML tools → tools off) |
| C1 | `vanilla_tools` | base Qwen3-0.6B | — | `native` |
| C2 | `sft_template` | Exp 1 | `train_interleaved_native.jsonl` | `native` |
| C3 | `sft_constitution` | Exp 2 | `train_sft_v3.jsonl` | `native` |
| C4 | `thinker_executor` | Exp 3 | `train_sft_thinker_curriculum.jsonl` + `train_sft_executor.jsonl` | `native` |

C0 and C1 are the **same base weights** — only `--tool_mode` differs. C2/C3/C4 are all native, so the rungs are directly comparable.

---

## 0. Setup (once)

```bash
cd /workspace
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai
git checkout feat/thinker-executor-sft
pip install -r pipeline/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" peft
pip install -U "fastapi>=0.110" "starlette>=0.37" "pydantic>=2"
cd pipeline
# pipeline/.env needs: HF_TOKEN (publish), EXA_API_KEY (live web_search), and a judge key
#   (NVIDIA_NIM_API_KEYS, or ANTHROPIC_API_KEY for a frontier judge) used later in §4
```

Use two terminals (server + benchmark):
```bash
tmux new-session -d -s server
tmux new-session -d -s bench
```

---

## 1. Build the three SFT datasets (vanilla needs none)

```bash
# Exp 1 — template, native format (2895 to match Exp 2's size for a fair delta)
python 1_dataset_generator.py --variant interleaved --tool_format native \
    --train_size 2895 --output_dir data
#   → data/train_interleaved_native.jsonl

# Exp 2 — constitutional (assemble from committed parts; quality-gated + full native)
python sft_dataset_assembler.py
#   → data/train_sft_v3.jsonl

# Exp 3 — Thinker + Executor (pure transforms, no GPU)
python sft_trajectory_splitter.py     # → train_sft_thinker.jsonl + train_sft_executor.jsonl
python sft_curriculum_merge.py        # → train_sft_thinker_curriculum.jsonl

# MANDATORY pre-train gate for Exp 3 (CPU) — must be green
python validate_thinker_executor_data.py
```

---

## 2. Train (GPU) — 4 SFT runs

`--no_curriculum` on both single-model runs so C2 and C3 share an identical training regime (only the data differs).

```bash
# C2 — Exp 1 template
python 2_model_trainer.py --mode sft \
    --dataset data/train_interleaved_native.jsonl \
    --output_name checkpoint_sft_template --no_curriculum --no_publish

# C3 — Exp 2 constitutional
python 2_model_trainer.py --mode sft \
    --dataset data/train_sft_v3.jsonl \
    --output_name checkpoint_sft_constitution --no_curriculum --no_publish

# C4 — Exp 3 Thinker, then Executor
python 2_model_trainer.py --mode sft \
    --dataset data/train_sft_thinker_curriculum.jsonl \
    --output_name checkpoint_thinker --no_curriculum --no_publish

python 2_model_trainer.py --mode sft \
    --dataset data/train_sft_executor.jsonl \
    --output_name checkpoint_executor --no_curriculum --no_publish
```

Outputs: `models/checkpoint_sft_template`, `models/checkpoint_sft_constitution`, `models/checkpoint_thinker`, `models/checkpoint_executor`. Watch the `[collapse-monitor]` line — if `think_empty%` climbs toward 100, reasoning is collapsing.

Exp 3 readiness gate before benchmarking:
```bash
python composed_loop_eval.py --thinker models/checkpoint_thinker --executor models/checkpoint_executor --n 60
#   → want high completion_rate + copy_fidelity
```

---

## 2b. Publish the checkpoints to HuggingFace (save the models)

Training above used `--no_publish`. Publish once afterwards (needs `HF_TOKEN` with write access).
Each command merges the LoRA into 16-bit safetensors and pushes to a repo
`{hf_username}/trustworthy-ai-{suffix}` (suffix = output name minus `checkpoint_`, underscores → hyphens):

```bash
python 2_model_trainer.py --mode publish --output_name checkpoint_sft_template     --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_sft_constitution --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_thinker          --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_executor         --hf_username AjinkyaTaranekar
```

**Four repos are created** (the vanilla base needs none — it is `unsloth/Qwen3-0.6B`):

| Checkpoint | HuggingFace repo |
|---|---|
| `checkpoint_sft_template`     | `AjinkyaTaranekar/trustworthy-ai-sft-template` |
| `checkpoint_sft_constitution` | `AjinkyaTaranekar/trustworthy-ai-sft-constitution` |
| `checkpoint_thinker`          | `AjinkyaTaranekar/trustworthy-ai-thinker` |
| `checkpoint_executor`         | `AjinkyaTaranekar/trustworthy-ai-executor` |

Add `--no_skip_gguf` to also export a GGUF (needs llama.cpp). Publishing is decoupled from training and from serving — benchmarking works fine from the local `models/checkpoint_*` dirs, so this step is for saving/sharing the weights. Do it on the GPU box before teardown.

---

## 2c. Sanity-check serving BEFORE benchmarking (5 min — saves hours)

Three serving bugs were fixed (2026-06-21) — make sure the box has **all three** (`git pull`) before any long run, or the tool dimension comes back empty:
- `3_infererence.py` — single-model anti-repetition no longer hard-bans 3-grams (the ban mangled tool-call JSON so it never parsed).
- `sft_v3_generator.py` — `litellm` is now a guarded import, so the server loads the **canonical** student prompt instead of a drifted fallback.
- `4_benchmark.py` — decodes **greedy** by default (was temp 0.7, which degraded the JSON).

**(a) Do the weights call tools at all?** Loads each published model, replays its own training rows greedy, checks for a parseable `<tool_call>` (or `<act>` for the thinker):
```bash
python test_tool_calling.py                                         # all published models
python test_tool_calling.py --only constitution --n 8 --max_new_tokens 2048
```
PASS across the fine-tuned models = weights are good; any empty `tool_trace` later is serving plumbing.

**(b) Per-condition server check.** For each condition's server, confirm the canonical-prompt log line, then one curl proves tools fire end-to-end:
```bash
# server startup MUST log: [INFO] Student prompts loaded from sft_v3_generator.py (canonical source)
#   (if it logs "[WARN] ... using built-in fallback prompts", the litellm guard didn't reach the box — stop)
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 9847 times 23.5?"}],"tool_profile":"compute_only","greedy":true}'
```
Look for `"tool_trace"` containing `"tool":"python_execute"`. Empty `tool_trace` with a `<tool_call>` still in the response → the box is missing a fix; re-pull before benchmarking.

---

## 3. Benchmark each condition (GPU — generation only, no LLM judge)

`4_benchmark.py` only GENERATES responses and saves self-contained reports; **LLM judging is a
separate local step (§5)** so you can release the GPU first. Every run uses the same flags: all
five suites, greedy decoding (the benchmark default — pass `--sample` only to opt out), the condition's `--tool_mode`, and a per-condition
`--output_dir reports/<label>`. `web_search`/`read_url` hit **live Exa** (set `EXA_API_KEY`) — see
the reproducibility caveat in Notes.

### C0 + C1 — vanilla base (one server, two benchmark runs)
```bash
# server (terminal 1)
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000
```
```bash
# bench (terminal 2) — C0: tools effectively off (base ignores XML tool instructions)
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
     --tool_mode xml --model_label vanilla_base --output_dir reports/vanilla_base

# C1: tools on
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
     --tool_mode native --model_label vanilla_tools --output_dir reports/vanilla_tools
```
Stop the server (`tmux send-keys -t server C-c`) before loading the next model.

### C2 — Exp 1 template
```bash
# server
python 3_infererence.py --model_dir models/checkpoint_sft_template --port 8000
```
```bash
# bench
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
     --tool_mode native --model_label sft_template --output_dir reports/sft_template
```

### C3 — Exp 2 constitutional
```bash
# server
python 3_infererence.py --model_dir models/checkpoint_sft_constitution --port 8000
```
```bash
# bench
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
     --tool_mode native --model_label sft_constitution --output_dir reports/sft_constitution
```

### C4 — Exp 3 Thinker–Executor
```bash
# server  (/health must report "mode":"dual")
python 3_infererence.py \
    --thinker models/checkpoint_thinker --executor models/checkpoint_executor \
    --base_model unsloth/Qwen3-0.6B --port 8000
```
```bash
# bench
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
     --tool_mode native --model_label thinker_executor --output_dir reports/thinker_executor
```

Each run writes `reports/<label>/{constitution_probe,category_probes,context_drift,adversarial,persona_conversations}_<ts>.json` (rule scores only; `llm_score` is null until §5).

**You can release the GPU now** — everything below is API-only.

---

## 4. Judge (local machine — LLM API only, no GPU)

Copy `reports/` to your machine and run the judge once over all conditions. It fills in
`llm_score` / `combined_score` / `persona_score` and recomputes the blended aggregates, editing
each report in place (keeps a `.prejudge.bak`).

```bash
python 5_judgement_day.py \
    --judge_model claude-opus-4-8 \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --report
```

The judge model must be **identical across all conditions** (recorded in each report's
`run_metadata.judged_by`). Use a strong frontier judge — it tracks facts across the persona
transcripts far better than a small one. Re-judging is free (no GPU); to compare judges, re-run
with `--out_suffix .kimi` etc. instead of editing in place.

---

## 5. Consolidate — the ladder table (offline, no GPU)

```bash
python analyze_experiments.py \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --reports_dir reports --output_dir reports
```

Outputs:
- `reports/experiment_ladder_<ts>.csv` + `.tex` — scores per condition with the four **isolating deltas** (C1−C0 tools, C2−C1 SFT scaffolding, C3−C2 constitutional content, C4−C3 architecture) and bootstrap 95% CIs.
- `reports/experiment_h3_failures_<ts>.csv` — probes the top rung still fails or regresses on (H3 limits).
- `reports/persona_dimension_correlation_<ts>.csv` — 6×6 Pearson matrix over the judged personas; flags any distinct dimension pair with |r| ≥ 0.9 as near-redundant (the "are the trust/empathy metrics overlapping?" check). Needs ≥3 judged personas.
- Console also prints the persona dimension means per condition.

---

## 5b. Side-by-side comparison + comparative judge (offline, no GPU)

`analyze_experiments.py` gives the **numbers**; `compare_report.py` gives the **answers** — every condition's response to the same question, aligned side by side in one HTML page (think / answer / tools / score per column, across all five suites). Because the questions are shared, you read each row across conditions and see exactly where they diverge.

```bash
# side-by-side HTML only (no API key needed)
python compare_report.py \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor

# + comparative LLM judge: ranks the answers head-to-head per question, builds a win leaderboard
python compare_report.py \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --judge --judge_model claude-opus-4-8
```

Output: `reports/comparison_<ts>.html` (open in a browser). The comparative judge is **relative** (which answer is best for this question) and complements `5_judgement_day.py`, which scores each answer in **isolation** against a fixed rubric — use both: the absolute scores for the ladder table, the head-to-head wins + leaderboard for "which condition actually answers better".

---

## 6. Commit results

```bash
git add pipeline/reports/ && git commit -m "results: five-condition ablation ladder" && git push
```

---

## Notes

- **Generation (GPU) and judging (API) are separate steps on purpose** — `4_benchmark.py` makes no LLM calls, so you pay GPU only for model inference and judge locally afterwards (and re-judge for free).
- **Judge must be identical across all 5 conditions** (recorded in `run_metadata.judged_by`). A strong frontier judge (e.g. `claude-opus-4-8`) gives more reliable Suite E conversation scores — if you switch, switch for every condition and re-run §4.
- **Scoring discipline:** `analyze_experiments.py` reports constitution as rule-based (primary) and combined (secondary) separately. For judge-independent anchors run `alignment_metrics.py` and `rescore_report.py` on the saved reports.
- **Determinism:** scripts + greedy decoding (the benchmark default; the server applies `do_sample=False` on the `greedy` request flag) make decoding deterministic; the one remaining nondeterministic input is live web search (next bullet). Re-running a step overwrites nothing destructive (timestamped files; the judge keeps a `.prejudge.bak`; `analyze_experiments.py` picks the latest per `reports/<label>/`).
- **Web search is live (Exa), not mocked.** `web_search`/`read_url` use your `EXA_API_KEY`, so the few web-grounded items (P5/P19, `real_time_data`) reflect current results and are **not byte-reproducible** across runs — everything else is deterministic. If you ever need exact web reproduction (e.g. to re-run a result months later), restart that server with `BENCH_MOCK_SEARCH=1` to swap in the fixed offline corpus.
- **Persona suite (Suite E)** runs 8 scripted personas including an error-prone "incompetent" user (states wrong facts + self-contradicts) — the conversation judge scores six dimensions; the correlation CSV above is how you defend those dimensions as non-redundant.
- **Write-up:** the methodology section (justifying scripted users + LLM assessor with literature, plus the reproducibility protocol and threats to validity) is drafted as `%`-commented LaTeX in `methodology-draft.tex` at the repo root — uncomment what you accept.
