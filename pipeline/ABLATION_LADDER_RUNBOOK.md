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
git checkout feat/thinker-executor-sft && git pull
pip install -r pipeline/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" peft
pip install -U "fastapi>=0.110" "starlette>=0.37" "pydantic>=2"
cd pipeline
# pipeline/.env needs: HF_TOKEN, NVIDIA_NIM_API_KEYS (judge), EXA_API_KEY (web_search)
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

## 3. Benchmark each condition (GPU — generation only, no LLM judge)

`4_benchmark.py` only GENERATES responses and saves self-contained reports; **LLM judging is a
separate local step (§5)** so you can release the GPU first. Same recipe for all conditions:
`--temperature 0`, all five suites, the condition's `--tool_mode`, and a per-condition
`--output_dir reports/<label>`. Start every server with `BENCH_MOCK_SEARCH=1` so web grounding is
reproducible.

Reusable benchmark command:
```bash
run_bench() {   # usage: run_bench <tool_mode> <label>   (generation only)
  python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --temperature 0 --tool_mode "$1" --model_label "$2" --output_dir "reports/$2"
}
```

### C0 + C1 — vanilla base (one server, two runs)
```bash
# server
BENCH_MOCK_SEARCH=1 python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000
```
```bash
# bench
run_bench xml    vanilla_base      # C0 — tools off
run_bench native vanilla_tools     # C1 — tools on
```
Stop the server (`tmux send-keys -t server C-c`) before the next model.

### C2 — Exp 1 template
```bash
# server
BENCH_MOCK_SEARCH=1 python 3_infererence.py --model_dir models/checkpoint_sft_template --port 8000
```
```bash
# bench
run_bench native sft_template
```

### C3 — Exp 2 constitutional
```bash
# server
BENCH_MOCK_SEARCH=1 python 3_infererence.py --model_dir models/checkpoint_sft_constitution --port 8000
```
```bash
# bench
run_bench native sft_constitution
```

### C4 — Exp 3 Thinker–Executor
```bash
# server  (/health must report "mode":"dual")
BENCH_MOCK_SEARCH=1 python 3_infererence.py \
    --thinker models/checkpoint_thinker --executor models/checkpoint_executor \
    --base_model unsloth/Qwen3-0.6B --port 8000
```
```bash
# bench
run_bench native thinker_executor
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

## 6. Commit results

```bash
git add pipeline/reports/ && git commit -m "results: five-condition ablation ladder" && git push
```

---

## Notes

- **Generation (GPU) and judging (API) are separate steps on purpose** — `4_benchmark.py` makes no LLM calls, so you pay GPU only for model inference and judge locally afterwards (and re-judge for free).
- **Judge must be identical across all 5 conditions** (recorded in `run_metadata.judged_by`). A strong frontier judge (e.g. `claude-opus-4-8`) gives more reliable Suite E conversation scores — if you switch, switch for every condition and re-run §4.
- **Scoring discipline:** `analyze_experiments.py` reports constitution as rule-based (primary) and combined (secondary) separately. For judge-independent anchors run `alignment_metrics.py` and `rescore_report.py` on the saved reports.
- **Determinism:** the scripts are deterministic; `--temperature 0` removes the only other source of variance. Re-running a step overwrites nothing destructive (timestamped files; the judge keeps a `.prejudge.bak`; `analyze_experiments.py` picks the latest per `reports/<label>/`).
- **Persona suite (Suite E)** runs 8 scripted personas including an error-prone "incompetent" user (states wrong facts + self-contradicts) — the conversation judge scores six dimensions; the correlation CSV above is how you defend those dimensions as non-redundant.
- **Write-up:** the methodology section (justifying scripted users + LLM assessor with literature, plus the reproducibility protocol and threats to validity) is drafted as `%`-commented LaTeX in `methodology-draft.tex` at the repo root — uncomment what you accept.
