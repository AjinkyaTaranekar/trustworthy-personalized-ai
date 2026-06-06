# Thinker–Executor — Train & Run Runbook (Vast.ai)

End-to-end runbook for the **Thinker–Executor dual-SFT** experiment (Experiment 3).
Follow the numbered steps in order. Companion to `pipeline.md` (single-model constitution
experiment); this file is the two-model variant.

> **Context (June 2026):** SFT-only, sub-1B. Two LoRA adapters on ONE shared `unsloth/Qwen3-0.6B`
> base: a **Thinker** (prose `<think>` + exactly one of `<ask>/<act>/<answer>`; never calls tools)
> and an **Executor** (one plain-language instruction → exactly one native `<tool_call>`).
> The contract is byte-identical between training and serving — that is the whole point, and the
> June-2026 cleanup (`THINKER_EXECUTOR_FIXES.md`) repaired every place it had drifted.

> **What is verified vs not.** All data/contract/decoding fixes are verified on CPU (tokenizer
> render parity, schema parity, validator, self-tests). The GPU path (two-adapter PEFT load,
> generation quality) can only be confirmed on the box — §6 and §9 include smoke tests to catch
> issues before a full run. Do not skip the pre-train gate (§5) or the smoke tests.

---

## 0. Rent a GPU

- RTX 4000 Ada 16 GB (~$0.35/hr) is enough for 16-bit LoRA on 0.6B; A100 40 GB for headroom.
- Disk 80 GB+. Image: any CUDA 12.x (e.g. `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`).
- Two adapters train independently — budget ~2× a single SFT run (~5 hrs total ≈ $2).

---

## 1. Clone + install

```bash
cd /workspace
git clone https://github.com/AjinkyaTaranekar/trustworthy-personalized-ai.git
cd trustworthy-personalized-ai
git checkout feat/thinker-executor-sft && git pull

pip install -r pipeline/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install peft                       # two-adapter (PeftModel) serving
# Pin FastAPI/Starlette together — a mismatch throws `Router.__init__() got 'on_startup'`
# at server startup. If you only train, you can skip this; you need it for §9.
pip install -U "fastapi>=0.110" "starlette>=0.37" "pydantic>=2"
```

---

## 2. Environment (`pipeline/.env`)

| Variable      | Required for                         |
|---------------|--------------------------------------|
| `HF_TOKEN`    | Publishing checkpoints to HuggingFace |
| `EXA_API_KEY` | `web_search` at serve/benchmark time  |
| `HF_HOME`     | Cache location (optional; `/workspace/.hf_home`) |

Decoding overrides (optional, **Thinker only** — the Executor always decodes clean):
`PIPELINE_REPETITION_PENALTY` (default 1.1), `PIPELINE_NO_REPEAT_NGRAM` (default 0).
Eval-sample printing: `PIPELINE_EVAL_SHOW_SAMPLES` (default 2), `PIPELINE_EVAL_SAMPLE_CHARS` (default 700).

```bash
cd pipeline
```

---

## 3. Data — already committed; regenerate only if you change prompts/schemas/sanitiser

The datasets are committed and consistent. Regenerate **in this order** only after editing a
prompt (`sft_v3_generator.py`), a tool schema/description (`pipeline_tools.py`), or the sanitiser
(`tool_io.py`). All three steps are pure transforms (no GPU, no teacher):

```bash
python sft_trajectory_splitter.py        # partA/partB → train_sft_thinker.jsonl + train_sft_executor.jsonl
python sft_curriculum_merge.py           # → train_sft_thinker_curriculum.jsonl  (the Thinker training file)
python restamp_native_tools.py --data data/train_sft_v3.jsonl   # single-model schema repair (only if training it)
```

Why order matters: the splitter stamps the canonical tool-result wrapper + Executor schema; the
merge interleaves the Branch-B `<ask>` rows + adversarial/negative rows into the Thinker set; the
re-stamp aligns the single-model tool schema to what the server serves.

---

## 4. (Skip unless changing the question set) regenerate source parts

`train_partA_v3.jsonl` / `train_partB_v3.jsonl` are committed teacher-distilled sources. Only
regenerate them (needs GPU/teacher) when changing categories or distillation prompts —
see `pipeline.md` §13. Normal runs start from §5.

---

## 5. Pre-train gate — CPU, MANDATORY before any GPU time

```bash
python validate_thinker_executor_data.py
```
Asserts the full train/serve contract: system prompt parity, prose-only Thinker, opening
`<think>` ≥150, **no empty `<think>` anywhere** (T6), canonical `[TOOL_RESULT]` tool turns,
Executor one-call + copy-fidelity, and Executor schema **byte-identical** to what the server
serves (T1–T6 + E1–E4). **If this is not green, do not train — fix the data first.**

Optional extra confidence (CPU, no model):
```bash
python thinker_executor_orchestrator.py --self_test     # control flow + parsing + sanitiser
python composed_loop_eval.py --self_test                # end-to-end scoring logic
python executor_ablation.py   --self_test               # ablation scoring logic
```

---

## 6. Train the two adapters (GPU)

Config lives in `2_model_trainer.py` (`MODEL_CONFIG`/`SFT_CONFIG`): `unsloth/Qwen3-0.6B`, 16-bit
LoRA r=64 α=16, lr 1e-4, cosine, `load_best_model_at_end`. Publishing is **decoupled** — pass
`--no_publish` and publish once later (§11). Use `--no_curriculum`: the merged Thinker file already
encodes its ordering, and the Executor is a flat transducer set.

```bash
# Thinker (reasoning; prose only)
python 2_model_trainer.py --mode sft \
    --dataset data/train_sft_thinker_curriculum.jsonl \
    --output_name checkpoint_thinker --no_curriculum --no_publish

# Executor (one instruction → one tool call)
python 2_model_trainer.py --mode sft \
    --dataset data/train_sft_executor.jsonl \
    --output_name checkpoint_executor --no_curriculum --no_publish
```
Outputs: `models/checkpoint_thinker/`, `models/checkpoint_executor/` (LoRA adapters + tokenizer).

### Watch the model during eval (this answers "can I see outputs?")
At every eval step the trainer now **prints sample generations** plus the collapse monitor:
```
  ┌─ [eval-sample 1] step=50
  │ Q: What is 17 times 23?
  │ A: <think> … </think> <act>Use python_execute to run this code: …</act>
  └─
  [collapse-monitor] step=50 think_empty=0/5 (0%) mean_tool_calls=0.80
```
- **Live persistence (survives a disconnect/crash):**
  - `reports/training/<name>/loss_live.jsonl` — every loss/lr/eval_loss record appended as it happens.
  - `reports/training/<name>/eval_samples.jsonl` — every sampled generation per eval.
  - (HF also checkpoints + writes `trainer_state.json` every `save_steps`.)
- **Watch `think_empty%`** — if it climbs toward 100%, reasoning is collapsing (lower LR / fewer epochs).
- Show more/fewer inline: `PIPELINE_EVAL_SHOW_SAMPLES=4 python 2_model_trainer.py …`.
- Tail live from another shell:
  ```bash
  tail -f reports/training/checkpoint_thinker/loss_live.jsonl       # metrics as they stream
  tail -f reports/training/checkpoint_thinker/eval_samples.jsonl    # generations as they stream
  ```

---

## 7. (Optional) single-model constitution checkpoint

If you also want the single-model baseline (its schema drift was repaired in §3):
```bash
python 2_model_trainer.py --mode sft --output_name checkpoint_sft --no_publish
```
Benchmark it per `pipeline.md` §6.

---

## 8. Decision gates (GPU) — run BEFORE benchmarking

```bash
# (a) Does the Executor SFT beat the base model at its narrow job? If not, drop it
#     and serve base Qwen3-0.6B + EXECUTOR_STUDENT_PROMPT + schema instead.
python executor_ablation.py \
    --base unsloth/Qwen3-0.6B --sft models/checkpoint_executor \
    --data data/train_sft_executor.jsonl --n 150
#   → reports/executor_ablation.json + KEEP/DROP verdict (tool-choice acc + copy-fidelity)

# (b) Does the composed loop work end to end? (eval_loss per component can't see this)
python composed_loop_eval.py \
    --thinker models/checkpoint_thinker --executor models/checkpoint_executor --n 60
#   → completion_rate, exec_parse_rate, copy_fidelity, math_correct
```
Gate "ready to benchmark" on (b): a healthy run has high `completion_rate` and `copy_fidelity`.

---

## 9. Serve (GPU) — full harness, metrics, run-records

```bash
python 3_infererence.py \
    --thinker models/checkpoint_thinker \
    --executor models/checkpoint_executor \
    --base_model unsloth/Qwen3-0.6B --port 8000
# /health → {"mode":"dual","architecture":"thinker_executor"}.  --max_steps N caps cycles (default 6).
```

Smoke-test BEFORE the full benchmark (the two-adapter load only runs on a real GPU):
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17 times 23?"}]}' | python -m json.tool
# Healthy: "type":"answer", a non-empty tool_trace with tool":"python_execute", answer contains 391.
tail -n 1 reports/inference_runs.jsonl | python -m json.tool   # durable audit record
```
Bare CPU sanity (no harness, local checkpoints): `thinker_executor_orchestrator.py --question "…" --verbose`.

---

## 10. Benchmark

```bash
python 4_benchmark.py --probe_only --model_label thinker_executor --no_judge --output_dir reports
# then: compare_runs.py vanilla.json thinker_executor.json → CSV   (see pipeline.md §6C)
```
The dual model is benchmarked exactly like a single model (apples-to-apples, harness on both arms).

**Live persistence:** each suite streams per-item results to
`reports/live/<suite>_<model>_<ts>.jsonl` as each probe/category/adversarial item completes — so a
disconnect or crash mid-run loses nothing (the aggregated JSON/CSV is still written at the end).
Watch it with `tail -f reports/live/probe_*.jsonl`.

---

## 11. Publish (once, optional)

Serving from local checkpoint dirs needs no publish. To upload:
```bash
python 2_model_trainer.py --mode publish --output_name checkpoint_thinker
python 2_model_trainer.py --mode publish --output_name checkpoint_executor
```
Requires `HF_TOKEN`. Publishes merged 16-bit + (optionally) GGUF and computes ROUGE.

---

## 12. On tool definitions in the dataset (production-grade notes)

**Yes — tool definitions belong in the dataset, and they already are.** Each Executor (and
single-model) row carries `metadata.native_tools`; the trainer renders them into the prompt via
`apply_chat_template(tools=native_tools)`, exactly as the server does. That is how the model learns
to read a schema and emit a conformant call, and it is standard practice (Qwen/Hermes/ToolLLM).
The thing that matters is that the schema is rendered **identically** at train and serve — names,
descriptions, parameters, **and order** — which the cleanup fixed and the validator now enforces.
The Thinker correctly gets **no** tools (it is prose-only; it delegates via `<act>`).

What a production-grade tool-use dataset adds beyond what we have (prioritised):

| Property | Status here | Worth adding? |
|---|---|---|
| Schema in-context, byte-identical to serving | ✅ done (validator E3) | — |
| Gold target is always a valid, parseable call | ✅ (validator E2) | — |
| Verbatim argument copy fidelity | ✅ 980 python + 1263 search/url, asserted (E4) | — |
| **Distractor tools** (more advertised than used) | Executor sees only its 3 owned tools | Optional — improves tool *selection*; the Thinker already does selection via `<act>`, so low value for a pure transducer |
| **Negatives / robustness** (injection inside the instruction, no-op instruction) | None on the Executor (security clause unreinforced) | **Recommended** — a small adversarial slice makes the security line real |
| **Tool-error recovery** (tool returns an error → cope) | Thinker has some from source trajectories | Nice-to-have; ensure a few error results survive factoring |
| Schema-phrasing variation (same tool, reworded) | Single canonical schema | Skip for a dissertation — overf-robustness, low ROI |
| Balanced / deduped / decontaminated vs eval | ✅ dedup + seed-42 held-out split | — |

Bottom line: the contract fixes (not more data) are what move results. The one genuinely
worthwhile addition for "one-go" quality is a **small Executor adversarial/negative slice**
(~50–100 rows) so the security instruction is trained, not decorative. Everything else is polish.

---

## 13. Troubleshooting

- **`Router.__init__() got 'on_startup'`** at server start → FastAPI/Starlette mismatch. Fix:
  `pip install -U "fastapi>=0.110" "starlette>=0.37"`.
- **CUDA OOM** → set `load_in_4bit=True` (or `max_seq_length=3072`) in `MODEL_CONFIG`, or serve with `--load_in_4bit`.
- **Executor emits prose instead of a call** → confirm you passed both `--thinker` and `--executor`; check the smoke test; verify decoding is clean (Executor must have no repetition penalty — it does by default).
- **Thinker loops / never closes `</think>`** → raise `PIPELINE_REPETITION_PENALTY` (e.g. 1.2) for the Thinker only.
- **`web_search unavailable`** in answers → set `EXA_API_KEY`.
- **Validator fails after editing a prompt/schema** → re-run §3 in order, then §5; never hand-edit the assembled JSONL.

---

## 14. Script & data reference (Thinker–Executor)

| Script | Role |
|---|---|
| `sft_trajectory_splitter.py` | partA/partB → Thinker + Executor SFT sets (canonical tool turns) |
| `sft_curriculum_merge.py` | interleave Branch-B `<ask>` + adversarial rows → Thinker curriculum |
| `restamp_native_tools.py` | re-stamp `metadata.native_tools` to the canonical served schema |
| `validate_thinker_executor_data.py` | pre-train gate (T1–T6, E1–E4) |
| `tool_io.py` | canonical tool-result sanitiser + `TOOL_PROFILES` (single source) |
| `2_model_trainer.py` | SFT trainer (per-role decode untouched; eval prints samples) |
| `executor_ablation.py` | base vs SFT Executor (tool-choice + copy-fidelity) |
| `composed_loop_eval.py` | end-to-end loop (completion / parse / copy-fidelity / math) |
| `thinker_executor_orchestrator.py` | the two-model loop (hosted in `3_infererence.py`) |
| `3_infererence.py` | server; `--thinker/--executor` for dual mode + full harness |
| `4_benchmark.py` | constitutional probes |

| Data file | Producer | Consumer |
|---|---|---|
| `data/train_sft_thinker_curriculum.jsonl` | splitter + merge | Thinker SFT |
| `data/train_sft_executor.jsonl` | splitter | Executor SFT |
| `data/train_sft_v3.jsonl` | assembler + restamp | single-model SFT |
| `reports/training/<name>/eval_samples.jsonl` | trainer (each eval) | inspect generations |
| `reports/executor_ablation.json` | `executor_ablation.py` | KEEP/DROP decision |
| `reports/composed_loop_eval.json` | `composed_loop_eval.py` | readiness gate |
| `reports/inference_runs.jsonl` | `3_infererence.py` | audit trail |
