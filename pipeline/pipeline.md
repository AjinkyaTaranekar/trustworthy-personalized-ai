# Pipeline: End to End (all 3 experiments)

Command-by-command runbook to reproduce the five-condition ablation ladder on one GPU.
Run every command from `pipeline/`. Generation needs a GPU; judging and analysis are API/CPU only.

For the research question, hypotheses and what each condition is testing, see the [project README](../README.md).

```
Experiment 1  ->  sft_template        (template SFT, format only)
Experiment 2  ->  sft_constitution    (constitutional distillation from a frontier teacher)
Experiment 3  ->  thinker_executor    (two 0.6B models: Thinker + Executor)
Baselines     ->  vanilla_base (tools off), vanilla_tools (tools on): same base weights
```

## Code and models

- Code: https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai
- Base model: `unsloth/Qwen3-0.6B` (the vanilla conditions need no publish)
- Published checkpoints (created in step 3):
  - Exp 1: https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-sft-template
  - Exp 2: https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-sft-constitution
  - Exp 3 Thinker: https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-thinker
  - Exp 3 Executor: https://huggingface.co/AjinkyaTaranekar/trustworthy-ai-executor

---

## 0. GPU requirement (Vast.ai)

Rent one GPU instance; the whole study fits a single consumer-class device.

Minimum / recommended configuration:

| Item    | Minimum                         | Comfortable                     |
|---------|---------------------------------|---------------------------------|
| GPU     | RTX 4000 Ada 16 GB (~$0.35/hr)  | A100 40 GB (faster throughput)  |
| VRAM    | 16 GB (16-bit LoRA on 0.6B)     | 40 GB                           |
| Disk    | 80 GB+                          | 100 GB+                         |
| Image   | CUDA 12.x, e.g. `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` | same |

Cost: one SFT run (3 epochs, r=64) is ~2.5 hr, roughly $0.90; all four training runs plus benchmarking stay well under $10. Judging and analysis run off the GPU, so release the instance before scoring.

---

## 1. Setup (once)

```bash
cd /workspace
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai
git checkout main
pip install -r pipeline/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" peft
cd pipeline

# pipeline/.env needs:
#   HF_TOKEN             : publish checkpoints
#   EXA_API_KEY          : live web_search / read_url at inference
#   NVIDIA_NIM_API_KEYS  : LLM judge (comma-separated keys; rotated automatically)

tmux new-session -d -s server        # two terminals: one serves, one benchmarks
tmux new-session -d -s bench
```

---

## 2. Build the training data (Exp 1, 2, 3)

```bash
# Exp 1: template corpus (native tools; size-matched to Exp 2 for a fair delta)
python 1_dataset_generator.py --variant interleaved --tool_format native \
    --train_size 2895 --output_dir data          # -> data/train_interleaved_native.jsonl

# Exp 2: constitutional corpus (assemble + quality-gate the committed teacher parts)
python sft_dataset_assembler.py                  # -> data/train_sft_v3.jsonl

# Exp 3: Thinker + Executor views (CPU-only transforms of the Exp 2 trajectories)
python sft_trajectory_splitter.py                # -> train_sft_thinker.jsonl + train_sft_executor.jsonl
python sft_curriculum_merge.py                   # -> train_sft_thinker_curriculum.jsonl
python validate_thinker_executor_data.py         # gate: must pass before training Exp 3
```

---

## 3. Train: 4 SFT runs (GPU)

`--no_curriculum` on the single-model runs so Exp 1 and Exp 2 share an identical regime (only the data differs).

```bash
# Exp 1
python 2_model_trainer.py --mode sft --dataset data/train_interleaved_native.jsonl \
    --output_name checkpoint_sft_template --no_curriculum --no_publish

# Exp 2
python 2_model_trainer.py --mode sft --dataset data/train_sft_v3.jsonl \
    --output_name checkpoint_sft_constitution --no_curriculum --no_publish

# Exp 3: Thinker, then Executor
python 2_model_trainer.py --mode sft --dataset data/train_sft_thinker_curriculum.jsonl \
    --output_name checkpoint_thinker --no_curriculum --no_publish
python 2_model_trainer.py --mode sft --dataset data/train_sft_executor.jsonl \
    --output_name checkpoint_executor --no_curriculum --no_publish

# Watch the [collapse-monitor] line: if think_empty% climbs toward 100, reasoning is collapsing.
```

Exp 3 readiness check before benchmarking:

```bash
python composed_loop_eval.py --thinker models/checkpoint_thinker \
    --executor models/checkpoint_executor --n 60      # want high completion_rate + copy_fidelity
```

Publish the checkpoints (optional; needs `HF_TOKEN` write access). Benchmarking works from the local `models/checkpoint_*` dirs, so this is only for saving/sharing:

```bash
python 2_model_trainer.py --mode publish --output_name checkpoint_sft_template     --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_sft_constitution --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_thinker          --hf_username AjinkyaTaranekar
python 2_model_trainer.py --mode publish --output_name checkpoint_executor         --hf_username AjinkyaTaranekar
```

---

## 4. Benchmark each condition (GPU: generation only, greedy by default)

All five suites, same flags per run; only the model and `--tool_mode` change. `4_benchmark.py` makes no LLM calls, so the GPU can be released after this.

```bash
# --- Baselines: one base server, two benchmark runs ---
python 3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000        # terminal: server

# vanilla_base: tools off (base ignores XML tool instructions)
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --tool_mode xml    --model_label vanilla_base  --output_dir reports/vanilla_base
# vanilla_tools: tools on
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --tool_mode native --model_label vanilla_tools --output_dir reports/vanilla_tools
# stop server before loading the next model:  tmux send-keys -t server C-c

# --- Exp 1: sft_template ---
python 3_infererence.py --model_dir models/checkpoint_sft_template --port 8000
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --tool_mode native --model_label sft_template --output_dir reports/sft_template

# --- Exp 2: sft_constitution ---
python 3_infererence.py --model_dir models/checkpoint_sft_constitution --port 8000
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --tool_mode native --model_label sft_constitution --output_dir reports/sft_constitution

# --- Exp 3: thinker_executor (/health must report "mode":"dual") ---
python 3_infererence.py --thinker models/checkpoint_thinker \
    --executor models/checkpoint_executor --base_model unsloth/Qwen3-0.6B --port 8000
python 4_benchmark.py --probe --categories --drift --adversarial --persona \
    --tool_mode native --model_label thinker_executor --output_dir reports/thinker_executor
```

Decode every condition identically (greedy is the default; keep `--max_new_tokens` the same across all five). Release the GPU after this step; everything below is API/CPU only.

---

## 5. Judge: absolute lens (LLM API, no GPU)

Fills `llm_score` / `combined_score` / `persona_score` into each report in place (keeps a `.prejudge.bak`). Resumable and rate-limit tolerant. The judge model must be identical across all five conditions.

```bash
python 5_judgement_day.py --judge_model nvidia_nim/minimaxai/minimax-m3 \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --workers 4 --rpm 36 --checkpoint_every 10 --report

# unattended: loops until everything is judged, pushing checkpoints after each pass
nohup python run_judge_loop.py --push > judge_loop.log 2>&1 &
```

---

## 6. Consolidate: ladder table + figures (no GPU)

```bash
python analyze_experiments.py \
    --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --reports_dir reports --output_dir reports --figures
```

Outputs: `reports/experiment_ladder_<ts>.{csv,tex}` (headline + diagnostics tables with isolating deltas and bootstrap CIs) and `reports/dissertation_assets/*.pdf` (drop `--figures` for the table only; add `--extended_figures` for the secondary plots).

---

## 7. Compare: comparative lens (no GPU)

Side-by-side HTML of every condition's answer to the same question, plus an optional head-to-head judge that ranks the answers and builds the win leaderboard used in the dissertation.

```bash
# side-by-side HTML only (no API key needed)
python compare_report.py --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor

# + comparative judge (ranks answers per question, builds leaderboard) and rank figures
python compare_report.py --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor \
    --judge --judge_model claude-opus-4-8
python rank_figures.py --reports_dir reports        # -> reports/dissertation_assets/fig_rank_*.pdf
```

Output: `reports/comparison_<ts>.html` and the `fig_rank_*` / `tab_rank_*` assets.

---

## 8. Commit results

```bash
git add pipeline/reports/ && git commit -m "results: five-condition ablation ladder" && git push
```

---

## Notes

- Generation (GPU) and judging (API) are deliberately separate: pay GPU only for inference, then judge/re-judge locally for free (`--force` re-judges).
- Web search is live (Exa), so the few web-grounded probes (P5/P19) are not byte-reproducible; set `BENCH_MOCK_SEARCH=1` on the server for a fixed offline corpus if exact reproduction is needed. Everything else is deterministic under greedy decoding.
- Judge-free anchors: `experiment_metrics.py` (structural depth metrics) and `alignment_metrics.py` / `rescore_report.py` (rule-only rescoring) run on the saved reports.
