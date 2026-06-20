---
title: Training and Benchmark Scripts
type: source
kind: code
tags: [code, training, lora, grpo, dapo, benchmark, context-degradation, constitution, drift-detection, small-model, personalisation, empathy, ontology, tool-use]
sources:
  - pipeline/config.py
  - pipeline/2_model_trainer.py
  - pipeline/3_infererence.py
  - pipeline/4_benchmark.py
  - pipeline/5_judgement_day.py
  - pipeline/analyze_experiments.py
  - pipeline/rescore_report.py
  - pipeline/alignment_metrics.py
  - pipeline/pipeline_tools.py
  - pipeline/5_context_degradation.py
  - pipeline/experiment0_reasoning_comparison.py
  - pipeline/sft_math_pipeline.py
  - pipeline/sft_dataset_assembler.py
  - pipeline/user_modelling.py
  - pipeline/empathy.py
  - pipeline/appraisal_labeller.py
  - pipeline/ontology_verifier.py
  - pipeline/run_all.sh
  - pipeline/preflight_check.sh
  - docker-compose.yml
  - README.md
updated: 2026-06-17
status: current
---

# Training & Benchmark Scripts

**The full four-module pipeline: SFT data generation → SFT training → GRPO RL training → User Modelling (FalkorDB + 5W+H + scrutability) → Appraisal-conditioned empathy → Ontology post-hoc verifier → Experiment 0 reasoning comparison → constitutional and adversarial probe suites → context degradation study. All feature-flag gated; all orchestrated by `run_all.sh`.**

## Scripts at a glance

| Script | Role |
| ------ | ---- |
| `config.py` | Feature flags singleton — six `ENABLE_*` flags read from `PIPELINE_*` env vars; `validate()` enforces dependency rules; `summary()` for startup logs. |
| `preflight_check.sh` | Pre-flight: Python version, packages, API key, file integrity, security blockers, feature-flag state, 7-stage training status. Run first on any new machine. |
| `run_all.sh` | Master orchestration: Stage 0 (FalkorDB), Stage 0.5 (appraisal labelling), Stages 1–8 (SFT + GRPO + evaluation). Fully resumable. Forwards `PIPELINE_*` env vars to every subprocess. |
| `1_dataset_generator.py` | V1 template-based interleaved data (Exp 1 of the ablation ladder). Restored 2026-06-19; gained `--tool_format {xml,native}`. Native mode converts `<tool>`→`<tool_call>`, remaps legacy off-registry tools to served ones, wraps tool results via `tool_io`, and stamps `native_tools` — output matches the Exp 2 served contract. |
| `2_model_trainer.py` | **Phase 1: SFT** — LoRA fine-tuning of [[entities/qwen3-0.6b\|Qwen3-0.6B]] on `train_sft_v3_robust.jsonl`. Loss masked on system+user tokens via `train_on_responses_only` (fixes large train/eval gap from ~400-token system prompt). **Phase 2: GRPO** — DAPO RL training with per-component reward functions (`make_reward_fns`) so TRL logs each component separately. **Publish** — merges LoRA → safetensors, exports GGUF, pushes to HuggingFace with retry. CLI: `--mode {sft,grpo,publish}`, `--reward_type {c,d}`, `--resume`. |
| `3_infererence.py` | **FastAPI inference server.** Model loaded once. Four built-in tools. Dual tool-call mode: `tool_mode="xml"` (custom `<tool>` tags, default) or `tool_mode="native"` (Qwen3 JSON `<tool_call>` via `apply_chat_template(tools=…)` — zero-shot new tools without retraining). Tool loop treats safety-validator failures as non-retryable and falls back to a no-tool answer after repeated tool errors. Full security hardening (Blockers 1–4). Module lifecycle hooks: write_pipeline → retrieve → analyse_appraisal → generate → onto_score. Endpoints: `/memory/inspect\|contest\|correct`, `/config`, `/dependency/status\|reset`, `POST /v1/model/swap` (hot-swap loaded model for multi-model benchmarking). |
| `4_benchmark.py` | **Benchmark client — generation only** (zero GPU, makes NO LLM calls as of 2026-06-19). Five suites: constitutional probes, category coverage, context drift, adversarial, persona conversations (Suite E, `--persona`). Embeds each item's `judge_principle`/`judge_rubric` into the saved report so judging is a separable step. **Trace-aware tool detection** (2026-06-07): tool-use checks read the orchestrator `tool_trace` via a centralised `_TRACE_TOOLS_OVERRIDE` + `_trace_tools()` context manager, not a grep of the prose answer — essential for the dual Thinker–Executor (whose answer carries no `<tool>` syntax). Multi-model hot-swap mode: `--models path1 path2 ...`. |
| `5_judgement_day.py` | **Offline LLM-as-judge** (zero GPU, 2026-06-19). Reads the saved `4_benchmark.py` reports and scores them via LLM API — so the GPU can be released before any judging, and re-judging is free. Two upgraded prompts (system+user, reasoning-before-score, calibrated anchors): a per-response judge (Suites A–C) and a whole-transcript conversation judge (Suite E, six dimensions). Writes `llm_score`/`combined_score`/`persona_score` + `dimension_means` back, recomputes blended aggregates, keeps a `.prejudge.bak`. Failed judge call → `score=None` (excluded). `--report` emits a diagnostic narrative. Adversarial (Suite D) is rule-only, never judged. |
| `analyze_experiments.py` | **Cross-condition consolidator** (zero GPU, 2026-06-17). Ingests the per-condition `4_benchmark.py` report JSONs for the five-rung ablation ladder (`vanilla_base` → `vanilla_tools` → `sft_template` → `sft_constitution` → `thinker_executor`), pairs the latest report of each suite per `--labels`, and emits the ladder table with the four **isolating deltas** + bootstrap 95% CIs. Keeps constitution **rule-based (primary) vs combined (secondary)** as separate rows (circularity discipline). Outputs `experiment_ladder_<ts>.{csv,tex}` + `experiment_h3_failures_<ts>.csv` (H3 limits inventory) + `persona_dimension_correlation_<ts>.csv` (6×6 Pearson over judged personas; flags |r|≥0.9 redundant pairs — the metric-overlap check). Console output is ASCII-safe (runs on Windows). |
| `rescore_report.py` | **Offline re-scorer** (zero GPU). Re-runs each probe's exact check against a saved report's `tool_trace`, swapping only the tool-detection source. Emits `saved` vs `replicated_textgrep` vs `rescored_trace` columns (isolates the tool-detection fix from check-version drift). On `a8d0c2a`: text-grep 0.4762 → trace-aware 0.5238 (P4/P10 corrected to PASS; P11 corrected to FAIL — a real failure the grep had masked). |
| `alignment_metrics.py` | **Offline trustworthiness/scrutability metrics** (zero GPU). Honesty F1 / over-refusal (positive class = abstain), fabrication rate (specific factual tokens absent from tool outputs), answer-grounding rate. Pairs with the deterministic mock search for exact, reproducible grounding. |
| `experiment0_reasoning_comparison.py` | **Experiment 0**: baseline / CoT / interleaved / ToT on GSM8K + logic puzzles. Must run before GRPO. |
| `5_context_degradation.py` | Context-length degradation study. Greedy decoding, 12 turns with known correct answers. |
| `user_modelling.py` | 5W+H FalkorDB graph client; Mem0g 4-stage write pipeline; retrieval gating; scrutability handlers. Loaded by `3_infererence.py` when `ENABLE_USER_MODELLING=true`. |
| `empathy.py` | Runtime appraisal helpers: `analyse_appraisal()`, `format_appraisal_block()`, `parse_appraisal_block()`, `APPRAISAL_SYSTEM_PREFIX`. |
| `appraisal_labeller.py` | **Offline one-time script.** Runs AppraisePLM over EmpatheticDialogues → `data/appraisal_labels.jsonl`. AppraisePLM used as labeller only; not a runtime dependency. |
| `ontology_verifier.py` | Post-hoc claim scorer (Experiment 6 Approach B). Dual backend: local OWL via rdflib or remote SPARQL endpoint. Loaded by `3_infererence.py` when `ENABLE_ONTOLOGY_VERIF=true`. |
| `sft_math_pipeline.py` | **Math question generation + rejection sampling** (merged from `sft_math_question_generator.py` + `sft_rejection_sampler.py`). 7 question types, EleutherAI/hendrycks_math dataset, Kimi K2.6 default model. Same AST sandbox as inference server (Blocker 1c/1d). |

## Architecture: server + client

`3_infererence.py` is the hub. Everything else calls it over HTTP, mirroring how Anthropic/OpenAI/DeepSeek separate model serving from evaluation. No model reloading between evaluations.

```bash
# Start server (terminal 1)
python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000

# Any evaluation (terminal 2 — no GPU needed)
python pipeline/4_benchmark.py --server_url http://localhost:8000
python pipeline/experiment0_reasoning_comparison.py --server_url http://localhost:8000
python pipeline/5_context_degradation.py --server_url http://localhost:8000
```

## Phase 1: SFT

`2_model_trainer.py --mode sft` trains on `data/train_sft_v3_robust.jsonl` (1,983 examples — see [[sources/code/sft-v2-pipeline]] for dataset composition). LoRA r=16, α=32, 3 epochs, lr=2e-4, effective batch=8 (batch=1, grad_accum=8), bf16, adamw_8bit. Outputs `models/checkpoint_sft/adapter_config.json`. The trainer does its own 90/10 split from the JSONL at runtime — `eval_sft_v2.jsonl` is not consumed by the trainer.

**Loss masking:** `train_on_responses_only` is applied after SFTTrainer construction so gradients flow only from assistant turns. This was critical: computing loss over the ~400-token system prompt (23 principles + CAPABILITY_CHECK template) on every example caused a large train/eval gap. Hardware config tuned for NVIDIA RTX A4000 (16 GB VRAM): `max_seq_length=2048`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, `save_steps=25`, `save_total_limit=4`.

The system prompt used at training time (`_system_prompt_for_profile()`) is now byte-identical to the inference server system prompt. The `_CONSTITUTION` constant in `3_infererence.py` is a verbatim copy of the 23-principle block in `sft_gold_response_generator.py:TRAINING_SYSTEM_PROMPT_TEMPLATE` — both must be updated together when principles change.

## Phase 2: GRPO (DAPO improvements)

`2_model_trainer.py --mode grpo` starts from the SFT checkpoint (reference policy anchor) and applies GRPO with DAPO improvements:

- **Token-level loss normalisation (Dr.GRPO)**: divide policy gradient loss by completion length, not sequence count. Eliminates the verbosity bias where longer wrong answers receive smaller penalties.
- **Clip-Higher**: asymmetric ε clipping (ε_low=0.2, ε_high=0.28). Lets high-reward completions update more freely, preventing entropy collapse where all G completions become identical.
- **Dynamic sampling**: skip prompts where all G completions score identically (zero gradient, wasted compute).

**Composite reward (all verifiable — no judge model):**

| Component | Weight | How measured |
|---|---|---|
| Format score | 0.30 | `<think>` + `CAPABILITY_CHECK` + `<answer>` present (regex) |
| Accuracy score | 0.40 | Code executes + answer matches expected (tool execution) |
| Tool integrity | 0.15 | No hallucinated or unavailable tool calls (set diff against profile) |
| Constitution score | 0.15 | P1+P3+P4+P14+P18 rule check (Blocker 2 `rule_check_response`) |

Two ablation conditions:
- **Condition C** (`--reward_type c`): format + accuracy only. Proves whether RL correctness signal matters.
- **Condition D** (`--reward_type d`): full composite. Full thesis contribution.

Reference policy = `checkpoint_sft` (not base model) — this anchors the constitution throughout RL. GRPO `num_generations=4` (reduced from 8 to fit A4000 16 GB VRAM; 8 requires 24 GB+).

**Per-component reward logging:** `make_reward_fns()` returns a list of functions (one per component). TRL sums them and automatically logs each under `rewards/{fn_name}_mean` in `grpo_loss_history.json` — enables per-component breakdown in the analysis notebook without extra instrumentation. The old `make_reward_fn()` (singular) is retained only for ROUGE evaluation during publish.

## Publish mode (`--mode publish`)

Re-uploads an already-trained checkpoint without retraining. Use when the automatic post-training push failed (network error, expired token). Loads the LoRA adapter via `load_checkpoint()`, merges → 16-bit safetensors, exports GGUF (Q4_K_M), computes ROUGE vs baseline, pushes both formats with `_retry_hf_push()` (3 attempts, exponential backoff). Falls back to local save if `HF_TOKEN` is unset.

```bash
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_sft --hf_username AjinkyaTaranekar
python pipeline/2_model_trainer.py --mode publish --output_name checkpoint_grpo_d --hf_username AjinkyaTaranekar
```

## Experiment 0: Reasoning Paradigm Comparison

Required before GRPO by researchplan.tex Phase 3. Compares four strategies on GSM8K (up to 100 questions) + 10 built-in logic puzzles. Result determines which reasoning format becomes the target for GRPO training.

| Strategy | System prompt | Tool profile |
|---|---|---|
| baseline | Direct answer, no think tags | no_tools |
| cot | Step-by-step chain of thought | no_tools |
| interleaved | CAPABILITY_CHECK native format | compute_only |
| tot | Generate 3 candidates, self-rank | no_tools |

Metrics: accuracy, CAPABILITY_CHECK rate, tool use rate, answer-tag rate, latency, tokens generated.

## Constitutional drift detection

The 12 constitutional probes are the monitoring mechanism for GRPO training. All checks are regex/rule-based — a judge model that drifts with the trained model would undermine the measurement.

```bash
# Step 1: after SFT, before any GRPO
python 4_benchmark.py --probe_only --save_as_baseline

# Step 2: after each GRPO checkpoint
python 4_benchmark.py --probe_only --baseline reports/constitution_baseline.json
# → drift_warning: true if constitution_score drops ≥5pp
```

Drift mitigation: (1) rollback → (2) increase `kl_coef` → (3) SFT replay buffer.

## Adversarial probe suite (Blocker 3)

14 probes in three categories. Run on SFT baseline before GRPO; re-run on GRPO checkpoints to detect reward-hacking regression.

| Category | Probes | Tests |
|---|---|---|
| jailbreak | JB1–JB5 | Direct override, DAN persona, system-prompt fishing, harmful roleplay, false authority |
| injection | INJ1–INJ3 | Embedded instruction in structured content, false constitution update, numeric payload |
| regression | REG1–REG6 | Sycophancy under expert pressure, format under pressure, tool hallucination, calibrated confidence ×2, social-pressure honesty |

Score = fraction of attacks resisted. `adversarial_score < 0.8` before GRPO = do not proceed.

## Persona conversation suite (Suite E, 2026-06-17)

`--persona` / `--persona_only` adds a fifth suite that evaluates how the model serves *human profiles* across a multi-turn conversation — the trust/empathy/personalisation lens the per-turn principle probes (Suites A–D) cannot capture. Eight scripted personas (`PERSONAS` in `4_benchmark.py`): risk-averse nurse, grieving student, non-technical bakery owner, ML-PhD depth-seeker, adversarial health-info seeker, budget-constrained parent, vague career-changer, and an error-prone "incompetent" user (2026-06-20, per Owen — states wrong facts + self-contradicts to test self-correction/non-sycophancy/memory). Each persona is a fixed `profile` + `goal` + ordered `script` of user turns + `expectations`. The six conversation dimensions are checked for redundancy by `analyze_experiments.py` (pooled Pearson matrix; flags |r|≥0.9) — Owen's "overlapping metrics compound" concern.

**Deterministic by construction:** the script is replayed verbatim through `_complete()` (no LLM on the user side); a fresh `session_id` per persona stops server-side memory/scratchpad state leaking between runs. The *script* is deterministic; model sampling is a separate axis — run `--temperature 0` for headline numbers.

**Conversation-level judge (`_conversation_judge`):** unlike the per-turn `_llm_judge` (which scores one `(question, response, principle, rubric)` tuple), this reads the WHOLE transcript — every user turn, the model's `<think>`/`<answer>`, and the tools it called (`_format_persona_transcript`) — against the persona's profile + goal + expectations, and returns structured JSON scoring six dimensions: `personalisation`, `memory_consistency`, `empathy`, `trustworthiness`, `coherence`, `goal_completion`, plus an `overall` and a one-paragraph `summary`. Judging runs concurrently across personas (`ThreadPoolExecutor`, 5 workers). A failed judge call returns `overall=None` and is excluded from the average — never a silent 0.5, matching the Suite A judge discipline.

**Output:** `reports/persona_conversations_<label>_<ts>.json` with `persona_score` (mean of per-persona `overall`), `dimension_means` across personas, and full per-persona transcripts + evidence. Live-streamed to `reports/live/persona_*.jsonl`. Threaded into the multi-model hot-swap loop and into `generate_llm_report` (section 5, "Persona conversation analysis"). The judge model is the shared `--judge_model`; a strong frontier judge (e.g. `claude-opus-4-8`) is recommended for conversation-level scoring even when the per-turn suites use a cheaper judge.

No constitution change: Suite E is an evaluation lens over existing principles (personalisation P6/P20/P21, trustworthiness P5/P7/P14/P16/P18/P25, memory persistence) plus the empathy module — it adds no new trained behaviour, tool, or category.

## Five-condition ablation ladder (2026-06-17)

The headline dissertation comparison is a five-rung ablation ladder; each adjacent delta isolates exactly one factor, which *is* the argument:

| Rung | Label | Serving | Isolating delta | Hypothesis |
|---|---|---|---|---|
| C0 | `vanilla_base` | base, `4_benchmark.py --tool_mode xml` (tools effectively off) | floor | — |
| C1 | `vanilla_tools` | base, `--tool_mode native` | C1−C0 = tool access | — |
| C2 | `sft_template` | Exp 1 template SFT (native), `--tool_mode native` | C2−C1 = SFT format scaffolding | H1 |
| C3 | `sft_constitution` | Exp 2 constitutional SFT (`train_sft_v3`), `--tool_mode native` | C3−C2 = constitutional content | **H1 core** |
| C4 | `thinker_executor` | dual: `3_infererence.py --thinker --executor`, benchmarked via `4_benchmark.py --tool_mode native` | C4−C3 = architecture | **H2 / capacity-displacement** |

**Tool-format per condition (verified 2026-06-19):** each model is benchmarked in the format it was trained on, else the score measures format mismatch not capability. All three SFT rungs are native `<tool_call>`: Exp 1 is generated native via `1_dataset_generator.py --tool_format native` (see below); Exp 2 (`train_sft_v3`, `tool_format: native`) and Exp 3's executor (`train_sft_executor`) are native (the thinker emits `<act>`). So C2/C3/C4 = native; C0 = xml is the deliberate "tools-off" rung (the untrained base ignores XML tool instructions), C1 = the same base with native.

> With Exp 1 regenerated native, the **C2→C3 delta is no longer confounded by tool format**. Remaining differences between C2 and C3 are training content, system prompt, tool inventory, and generation method (template vs constitutional) — report C2→C3 as "template-SFT vs constitutional-SFT", not a strict single-variable isolation. (Note: `restamp_native_tools.py` does NOT convert `<tool>`→`<tool_call>` — it only rewrites `metadata.native_tools`; the message-level conversion lives in the generator's `--tool_format native`.)

All five conditions go through the SAME `4_benchmark.py` suites — the Thinker–Executor dual is served on the same `/v1/chat/completions` endpoint (`_DUAL_MODE` in `3_infererence.py`), and trace-aware scoring reads its tools from the orchestrator trace — so every report shares one schema and pairs cleanly in `analyze_experiments.py`. `composed_loop_eval.py` is a separate narrower loop-eval artifact, not used for this comparison.

**Exp 1 generator restored + native support (2026-06-19):** `1_dataset_generator.py` was deleted in PR #1 (`e20c190`); restored from `e373779` (the exact pre-PR blob), then extended with `--tool_format {xml,native}`. `templates.json` (42 `interleaved` templates) was never deleted. Native mode (`_to_native`) converts assistant `<tool>name(args)</tool>` → `<tool_call>{json}</tool_call>` (parsing via `_parse_xml_tool`), remaps legacy tools not in the served registry to real ones (`get_exchange_rate`→`web_search`, `get_user_profile`→`user_memory_read`, `update_profile`→`user_memory_update`, `analyze_sentiment`→`python_execute`), wraps `role:tool` content via `tool_io.sanitise_tool_result`, and stamps `metadata.tool_profile`/`native_tools`/`tool_format` — matching the served contract and Exp 2. Output: `data/train_interleaved_native.jsonl` (generated 2000 examples, seed 42). Command: `python 1_dataset_generator.py --variant interleaved --tool_format native --train_size N`. Plan: retrain all conditions fresh, re-run the full benchmark on each (per-condition `--tool_mode`), then consolidate with `analyze_experiments.py`. **Step-by-step commands: `pipeline/ABLATION_LADDER_RUNBOOK.md`.**

**Run hygiene per condition:** `--temperature 0` (scripts are deterministic; sampling is the only other axis), one fixed `--judge_model` recorded in `run_metadata`, `BENCH_MOCK_SEARCH=1` for reproducible grounding, all five suites. Then `python analyze_experiments.py --labels vanilla_base vanilla_tools sft_template sft_constitution thinker_executor`.

## Security blockers summary

All four blockers are verified by `preflight_check.sh` section 9. Each is a structural code check, not a runtime policy:

- **Blocker 1** (`3_infererence.py`, `sft_math_pipeline.py`): AST-based import whitelist + dangerous-builtin block before any `subprocess.run`; `_sanitise_tool_output` strips injection patterns from web content. (`sft_math_question_generator.py` + `sft_rejection_sampler.py` merged into `sft_math_pipeline.py`.)
- **Blocker 2** (`sft_gold_response_generator.py`): `rule_check_response()` provides deterministic out-of-band checks for P1/P3/P4/P14/P18 before the LLM critique; `_merge_violations()` ensures rule violations survive a `NO_VIOLATIONS` LLM response.
- **Blocker 3** (`4_benchmark.py`): 14-probe adversarial suite; run before GRPO.
- **Blocker 4** (`3_infererence.py`): `DependencyMonitor` tracks interaction frequency + burst patterns, appends wellbeing disclosure when dependency signals detected.

## Ablation A/B/C/D

| Condition | Training | Measures |
|---|---|---|
| A | Base Qwen3-0.6B, no training | Zero-shot floor |
| B | SFT only (`checkpoint_sft`) | Value of constitutional formatting |
| C | SFT → GRPO, format+accuracy only | Value of RL correctness signal |
| D | SFT → GRPO, full composite reward | Full thesis contribution |

The primary thesis argument: D outperforms A on constitutional adherence, accuracy, and sycophancy resistance — at a fraction of the compute of frontier models.

## Inference server: dual tool-call modes (2026-05-12)

`3_infererence.py` supports two tool-call modes selectable per request via `tool_mode`:

| Mode | Format | Parser | Training | New tools |
|---|---|---|---|---|
| `xml` (default) | `<tool>name(arg='val')</tool>` | `_parse_tool_call()` regex | ✓ SFT examples | Requires retraining |
| `native` | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` | `_parse_native_tool_call()` JSON | ✓ 133 native SFT examples | Zero-shot via pre-training |

In native mode `_generate()` passes `tools=[…]` (OpenAI schemas from `_to_openai_schemas()`) to `apply_chat_template` — identical to what the training examples were rendered with via `messages_to_text()`. GGUF path normalises structured `tool_calls` output to inline `<tool_call>` text so both backends share one parser.

## Benchmark tool mode (2026-05-28)

`4_benchmark.py` now exposes `--tool_mode {xml,native}` (default `xml`) which is threaded through every suite runner (`run_constitution_probes`, `run_category_probes`, `run_context_drift_test`, `run_adversarial_probes`, `run_benchmark`) and ultimately into every `_complete()` call as `tool_mode` in the POST body. For vanilla/base-model benchmarking use `--tool_mode native`: the base Qwen3-0.6B only knows the Hermes `<tool_call>` format (pre-training) and ignores the `<tool>` XML instructions in the system prompt, causing all tool-using probes (P4, P10, P19, …) to score 0.0 without this flag. For SFT model evaluation keep the default `xml`. For `--probe_compare` (vanilla vs fine-tuned on two servers) use `--compare_tool_mode native` so server A uses `xml` and server B uses `native` (or vice versa).

## Multi-model benchmarking (hot-swap, 2026-05-24)

`POST /v1/model/swap` on `3_infererence.py` + `--models` flag on `4_benchmark.py` together enable sequential multi-model benchmarking without restarting either process. The benchmark calls the swap endpoint, which unloads the current model, calls `torch.cuda.empty_cache()`, optionally resets the inference metrics (`reset_metrics=True` by default), then loads the new model via `FastModel.from_pretrained` (HF checkpoint) or `llama_cpp.Llama` (GGUF). The server blocks during load (~30 s); the benchmark uses a 300 s timeout.

**Swap endpoint request body (`ModelSwapRequest`):**

| Field | Default | Description |
|---|---|---|
| `model_dir` | required | Path to LoRA checkpoint dir on disk (or any string — used as fallback key) |
| `base_model` | `unsloth/Qwen3-0.6B` | HF model ID used when `model_dir` does not exist on disk |
| `gguf` | `None` | GGUF file path; if set, loads via llama_cpp instead of unsloth |
| `max_seq_length` | `4096` | Passed to `FastModel.from_pretrained` or `Llama(n_ctx=…)` |
| `reset_metrics` | `True` | Clear latency/throughput counters before loading |

**Benchmark `--models` flags:**

| Flag | Default | Description |
|---|---|---|
| `--models` | `None` | One or more model dirs/HF IDs; triggers hot-swap loop |
| `--labels` | path stem | Display labels; defaults to last path component (so `unsloth/Qwen3-0.6B` → `Qwen3-0.6B`) |
| `--base_model` | `unsloth/Qwen3-0.6B` | Fallback HF ID when a `--models` path is not on disk |
| `--max_seq_length` | `4096` | Forwarded to the swap endpoint |
| `--compare_output` | `reports/comparison_<ts>.csv` | Output path for the N-model comparison CSV |

**Output artefacts per run:** `constitution_probe_<label>_<ts>.json`, `category_probes_<label>_<ts>.json`, `adversarial_<label>_<ts>.json`, one CSV per invocation with principle scores and Δ columns relative to the first (baseline) model.

**Backward compatibility:** `--models` absent → existing single-server flow, no swap calls made. `--models` with one entry → swap + run, no comparison table. `--models` with 2+ → full loop + comparison.

## Hyperparameters

**SFT**: LoRA r=16 α=32, 3 epochs, lr=2e-4, batch=1, grad_accum=8 (effective batch=8), bf16, adamw_8bit, packing=True, max_seq_length=2048, save_steps=25, eval_steps=25, save_total_limit=4. Hardware: NVIDIA RTX A4000 16.8 GB. Expected ~50 min.
**GRPO**: G=4 (reduced from 8 for 16 GB VRAM), β=0.001, lr=1e-6, ε_low=0.2, ε_high=0.28, rollout temp=1.0, max_new_tokens=512, 1 epoch.

## Feature flags and module lifecycle

All optional modules are gated by `config.py`. The inference server reads flags at startup via `PIPELINE_*` env vars or a `--config yaml` file. Each module degrades to a neutral stub when its flag is off — the server never crashes on a missing dependency.

Dependency rules enforced by `cfg.validate()`:
- `ENABLE_GRPO` → `checkpoint_sft` must exist
- `ENABLE_PERSONALISATION` → `ENABLE_USER_MODELLING` must be true
- `ENABLE_EMPATHY` → `data/appraisal_labels.jsonl` must exist
- `ENABLE_ONTOLOGY_VERIF` → `ONTOLOGY_PATH` file or `ONTOLOGY_SPARQL_ENDPOINT` must be set

`run_all.sh` forwards all flags to every subprocess so the server, trainer, and benchmark client always run with a consistent configuration.

## `run_all.sh` stage map (updated)

| Stage | Condition | Output |
|---|---|---|
| 0 | `ENABLE_USER_MODELLING=true` | FalkorDB on port 6379 |
| 0.5 | `ENABLE_EMPATHY=true` | `data/appraisal_labels.jsonl` |
| 1–8 | always | SFT data, SFT checkpoint, baselines, GRPO C/D, ablation |

## Related

- [[sources/code/sft-v2-pipeline]] — data source for SFT training
- [[entities/grpo]] — GRPO + DAPO algorithm notes
- [[entities/qwen3-0.6b]] — base model
- [[entities/constitution]] — 19 constitutional principles
- [[topics/personalisation]] — 5W+H schema + Mem0g design + scrutability gap
- [[topics/empathy]] — appraisal-conditioned generation design
- [[topics/ontology-integration]] — Experiment 6 design; Approach A vs B
- [[decisions/2025-10-01-four-module-architecture]] — why the pipeline is split this way
- [[queries/grpo-and-personalisation-master-plan]] — detailed module design rationale
- [[queries/full-pipeline-implementation-plan]] — phase-by-phase build plan for all modules

## Raw

- `pipeline/config.py`, `pipeline/user_modelling.py`, `pipeline/empathy.py`
- `pipeline/appraisal_labeller.py`, `pipeline/ontology_verifier.py`
- `pipeline/2_model_trainer.py`, `3_infererence.py`, `4_benchmark.py`, `5_context_degradation.py`
- `pipeline/experiment0_reasoning_comparison.py`, `run_all.sh`, `preflight_check.sh`
- `docker-compose.yml`, `README.md`
