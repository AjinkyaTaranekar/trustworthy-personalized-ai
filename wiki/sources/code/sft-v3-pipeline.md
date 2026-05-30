---
title: SFT v3 Asymmetric Distillation Pipeline
type: source
tags: [sft, distillation, training, tool-use, constitutional-ai, curriculum-learning]
sources:
  - pipeline/sft_v3_generator.py
  - pipeline/sft_dataset_assembler.py
  - pipeline/sft_trajectory_splitter.py
  - pipeline/2_model_trainer.py
updated: 2026-05-30
status: current
---

**The v3 pipeline replaces constitution-in-student-prompt with asymmetric context distillation, live tool execution via an intercept loop, and failure injection for negative trajectories.**

> **Update (2026-05-29) — pipeline coherence + anti-collapse fixes.** Diagnosing the 2026-05-25 benchmark (SFT 0.4286 < vanilla 0.4603; `think_empty` 0%→95%; tool-calls 1→56) traced the collapse to dirty training data, not the architecture. Changes: (1) the standalone `validate_sft_data.py` was removed and its quality gates folded into `sft_dataset_assembler.py` `passes_quality_filter()` — now requires first-assistant `<think>` ≥150 chars, rejects teacher-constitution leak + banned placeholders, requires `<answer>`; (2) the assembler is now **full-native** by default (all tool examples in `<tool_call>` JSON; legacy `<tool>` XML dropped) — the data was already ~99% native, the old 4-tool XML→native converter was silently dropping 98% of examples and is now registry-sourced for all 10 tools; (3) the trainer's empty default `train_sft_v3.jsonl` landmine was fixed by regenerating from the clean `*_v3` source parts; (4) trainer SFT LR 2e-4→1e-4, 3-stage curriculum on by default, and a `[collapse-monitor]` callback reports `think_empty%`/tool-calls each eval. The `enable_thinking` train/inference settings were verified correct (no change). (5) The student prompt (`_make_student_prompt`/`STUDENT_PROMPTS`, the single source `3_infererence.py` imports) was rewritten to **native function-calling** — no XML `<tool>` call-syntax, ~230 words, behaviours preserved — and the assembler now **re-stamps it onto every training example** (`restamp_student_prompt`) so training data matches the served prompt byte-for-byte. See [[experiments/sft-benchmark-analysis-20260525]].

The core diagnosis from the v2 post-mortem: a 0.6B parameter model cannot simultaneously track a 24-point checklist and reason about the user's problem. The v2 system prompt consumed roughly 200 words of the student's attention budget for rule-recitation instead of reasoning. V3 eliminates this by keeping the constitution entirely on the teacher side and distilling only the behaviours into the student data.

## Architecture: Three Phases

**Phase A — Teacher generation.** `sft_v3_generator.py` calls the teacher model (Kimi K2.6 or Minimax M2.7) with a system prompt containing all 25 constitution principles. Crucially, the teacher prompt explicitly forbids outputting rule names, checklist headers (`CAPABILITY_CHECK:`, `5W+H:`), or placeholder phrases. The result is a flowing narrative `<think>` block that implicitly demonstrates the principles rather than naming them. This is the behaviour the student model learns to imitate.

**Phase B — Reality-anchored execution.** Generation is not a single API call but a state machine. The Python script passes `stop=["</tool>"]` to litellm so generation halts immediately when a tool call body is emitted. The script then executes the tool (exa.ai for web search, subprocess for python_execute), appends the real `[TOOL_RESULT]` to the conversation, and resumes generation. The teacher's synthesis is therefore grounded in actual tool outputs, not imagined ones. This eliminates the "hallucinated execution" failure mode where the model writes "I ran a script and got X" without ever emitting a parseable `<tool>` tag.

**Phase C — Context swap.** Before writing the completed conversation to JSONL, the teacher's system prompt is replaced with the ≤50-word student prompt (`STUDENT_PROMPTS` dict keyed by tool profile). The full constitution never appears in the saved file. The 0.6B student model is trained on: [short system prompt] + [user question] + [narrative `<think>` with embedded tool calls] + [real tool results] + [`<answer>`].

## Negative Trajectories

Two new question categories train the model on failure recovery:

**`inventory_constraint`** — the question requires web_search but the session profile is `compute_only`. Correct behaviour: check the tool inventory inside `<think>`, recognise the gap, refuse honestly in `<answer>`, and redirect to an authoritative source. Trains constitution principles P3 (tool discipline) and P18 (explicit I don't know).

**`environment_timeout`** — web_search is available but the first call returns HTTP 503. Correct behaviour: retry once with a refined query; if the second attempt also fails, state the gap and answer from static knowledge with a knowledge-cutoff caveat. Trains P12 (tool failure handling).

## Quality Gate

`validate_sft_data.py` enforces five invariants before training: (1) system prompt ≤ 50 words — proves the constitution was not leaked to the student; (2) `<think>` block ≥ 50 characters — prevents synthetic laziness (empty think blocks); (3) no banned placeholders in `<think>` — blocks v2-era shortcuts like "see answer below"; (4) tool call immediately followed by tool role — sequence integrity, no hallucinated execution; (5) last message is assistant with `<answer>` — end-to-end resolution guaranteed. If >5% of rows fail, the generation pipeline is broken; fix the generator, not the validator.

## Branch B — Thinker Clarification Trajectories (`--branch_b`)

`sft_v3_generator.py --branch_b` generates training data for the Thinker model of the [[experiments/thinker-executor-experiment|Thinker–Executor experiment]] — specifically the behaviour no public dataset and no existing trajectory contains: deciding to **stop and ask** one targeted clarifying question (grounded in first-principles + 5W+H decomposition) rather than proceeding on a silent assumption. It re-uses the ambiguous seed questions already in `data/questions_partA.jsonl` (categories `multi_step_clarification`, `ambiguous_underspecified`, `user_context_behavioral`, `verbose_context_behavioral`) — no new question generation.

The Thinker vocabulary is **prose only**: a `<think>` block followed by exactly one of `<ask>` / `<act>` / `<answer>` (step-by-step, ReAct-like — confirmed 2026-05-29). This is the structural fix for the capacity-displacement collapse: the Thinker has one output modality, so structured tool-calling (which displaced reasoning in the single model) cannot compete. All tool-call syntax lives in the separate Executor.

Per seed, the teacher (cold-start, no user memory injected, so genuine ambiguity about the user cannot be silently resolved) decides:
- **Genuinely ambiguous → `<ask>`** (positive, `branch: "B"`): validated by `_validate_ask` (exactly one question; `<think>` names a 5W+H dimension). A second teacher call role-plays the user answering; the teacher then resolves with `<act>` or `<answer>`. Row shape: `system → user → <think>+<ask> → human → <think>+<act|answer>`.
- **Specifiable → `<act>`/`<answer>`** (don't-ask negative, `branch: "B_negative"`): the teacher proceeds without asking — the negative example that prevents over-clarification.

> **Priming + teacher model (validated 2026-05-30 spot-check).** Both teacher calls are primed with `_BRANCH_B_FEWSHOT` — a one-shot prose demonstration (one ambiguous→`<ask>`, one specifiable→`<answer>`). Without it the teacher emits a bare tag with no `<think>` and every row is rejected by the ≥150-char think gate (`processed=0`). Use `--model nvidia_nim/minimaxai/minimax-m2.7` (now the default): the first-assistant `<think>` auto-wrap is built around minimax's flowing-prose style, whereas kimi-k2.6 returns reasoning out-of-band (empty-content `<think>`) and skips every row.

Output: `data/train_sft_thinker_branch_b.jsonl` in **Thinker format**, consumed by `sft_trajectory_splitter.py` at the Thinker curriculum-merge step — **not** by `sft_dataset_assembler.py` (whose quality gate requires `<answer>`). `THINKER_STUDENT_PROMPT` and the `<ask>`/`<act>`/`<answer>` vocabulary in `sft_v3_generator.py` are the single source of truth shared with the splitter. Spot-check before a full run:

```bash
python sft_v3_generator.py --questions data/questions_partA.jsonl --branch_b --max 5
```

## Trajectory Splitter — Thinker · Executor Factoring (`sft_trajectory_splitter.py`)

**Implemented 2026-05-30.** A pure transformation (no GPU, no teacher) that projects the already-generated v3 trajectories (`data/train_partA_v3.jsonl` + `data/train_partB_v3.jsonl`) onto the two role-conditioned SFT sets for the [[experiments/thinker-executor-experiment|Thinker–Executor experiment]] (§7.2–7.3, §7.9–7.10). Because both views are read off the *same* trajectory, the `<act>` the Thinker learns to emit and the `<tool_call>` the Executor learns to produce are the same action — alignment is structural, not stitched. It reuses the assembler's parsing helpers (`_THINK_BLOCK_RE`, `_TOOL_RE`, `_xml_tool_to_native`, `_unwrap_tool_result`, `MIN_THINK_CHARS`, `MAX_RESULT_CHARS`) and adds a Hermes `<tool_call>{json}</tool_call>` text parser (the format the v3 generator actually emits — not legacy `<tool>` XML, not a structured `tool_calls` field). Role prompts `THINKER_STUDENT_PROMPT` and the new `EXECUTOR_STUDENT_PROMPT` are imported from `sft_v3_generator.py` (single source of truth shared with the served models).

> **Parser gotcha (load-bearing).** The Hermes body must be captured up to the `</tool_call>` **tag**, not via `\{.*?\}`, and parsed with `json.loads(..., strict=False)`. `python_execute` `code` arguments contain `{ }` (dicts, f-strings, sets) **and raw newlines**; a brace-delimited capture truncates at the first inner `}`, and strict JSON rejects the raw newlines — together these silently dropped ~780 of the ~800 Part B maths `python_execute` calls in the first cut (executor `python_execute` pairs were 177 instead of 980). The same brace-truncating regex + strict parse also lived in `3_infererence.py` `_parse_native_tool_call` (~line 510) — **fixed 2026-05-30** with the identical `</tool_call>`-delimited capture + `strict=False`. Before the fix, the model's `python_execute` calls failed to parse at inference (JSONDecodeError on the first inner brace / raw newline), so maths tool-calls were silently dropped and never executed — a likely contributor to weak P4/P10 maths-probe scores. The return shape (`{"function", "kwargs"}`) is unchanged, so the execution loop needed no other change.

**Tool ownership at factor time:** Executor-owned (delegated as `<act>`) = `python_execute`, `web_search`, `read_url`. The Thinker's own state tools (`user_memory_*`, `scratchpad_*`, `get_datetime`) are dropped from the factored Thinker stream — their call turns are handled in-context by the orchestrator at inference, so the Thinker never learns to emit tool-call syntax, while their preceding `<think>` reasoning is carried forward to the next emitted Thinker turn. `get_datetime` is still advertised in the Executor's schema list but never delegated.

**Outputs (validated 2026-05-30):** from 2,274 source trajectories → `train_sft_thinker.jsonl` (2,220 rows, prose-only `<think>` + `<ask>`/`<act>`/`<answer>`; 1,366 with ≥1 `<act>`, 854 reason→answer; 54 dropped for missing `<answer>`, opening `<think>` <150 chars, or tool-syntax leak) and `train_sft_executor.jsonl` (2,243 deduped one-`<act>`→one-`<tool_call>` pairs: 1,055 web_search, 980 python_execute, 208 read_url). The Executor target preserves the source Hermes call **verbatim** (raw newlines and all) so it matches what the model emits and what the inference parser consumes. All 2,243 Executor rows and the Thinker rows render cleanly through the Qwen3 chat template in `2_model_trainer.messages_to_text` (Thinker with no tools block; Executor with the 4-tool schema + an empty `<think></think>` before the call), so training needs **no** trainer changes — just two `--dataset … --output_name …` runs.

```bash
python sft_trajectory_splitter.py --inspect 5   # preview factored rows, no write
python sft_trajectory_splitter.py               # → train_sft_thinker.jsonl + train_sft_executor.jsonl
```

### Curriculum merge (`sft_curriculum_merge.py`)

**Implemented 2026-05-30.** The factored Thinker set is entirely Branch A/C — every row eventually acts — so a Thinker trained on it alone learns to *always delegate* and never `<ask>`. This step interleaves the synthesised Branch B rows (`train_sft_thinker_branch_b.jsonl`, from `sft_v3_generator.py --branch_b`) into the factored set for curriculum ordering (§7.5, §7.9 #5), writing the Thinker trainer input `data/train_sft_thinker_curriculum.jsonl`. Default ratio is **auto** = `len(factored)//len(branch_b)` so all Branch B rows are placed evenly (with ~500 Branch B + 2,220 factored that is ~1 per 4; `--ratio 11` forces the sparser 1-per-10–12 the experiment text assumed for the larger external-topped mix). Every row is re-stamped with `THINKER_STUDENT_PROMPT` so both sources carry a byte-identical system prompt. It is runnable **before** Branch B exists: if the Branch B file is missing/empty it warns loudly and passes the factored A/C set through unchanged, so Thinker training can still proceed on A/C alone and the merge is simply re-run once Branch B is generated (pending supervisor sign-off).

```bash
python sft_curriculum_merge.py   # → data/train_sft_thinker_curriculum.jsonl
```

## Curriculum Learning

`2_model_trainer.py` gains a `--curriculum_stage {1,2,3}` flag: Stage 1 uses short, no-tool examples to establish `<think>...</think><answer>...</answer>` syntax; Stage 2 uses all examples to introduce multi-tool reasoning trajectories; Stage 3 uses all examples plus 20% Stage-1 replay to prevent anti-drift loss of basic instruction-following. Pass each stage's output checkpoint as `--from_checkpoint` for the next stage.

## V3 Format Compatibility

GRPO `_format_reward` previously required `CAPABILITY_CHECK` in every response. V3-trained models use narrative think blocks without this header. Pass `--v3_format` to the GRPO trainer to switch to the v3 reward, which checks `<think>` + `<answer>` only.

## Background Watch-Commit

Passing `--watch_commit` to `sft_v3_generator.py` starts a daemon thread that runs alongside the main worker pool. Every `--watch_threshold` new non-blank lines written to the output file (default 50), the thread stages the file, commits, and pushes — identical behaviour to the standalone `watch_and_commit.py`. This removes the need to run two separate nohup processes. Typical invocation for long unattended runs:

```bash
nohup python -u pipeline/sft_v3_generator.py \
    --questions pipeline/data/questions_partA.jsonl \
    --model nvidia_nim/minimaxai/minimax-m2.7 \
    --workers 5 --watch_commit \
    > pipeline/nohup_generator.out 2>&1 &
```

## Related

- [[sources/code/sft-v2-pipeline]] — prior pipeline (preserved, v3 is additive)
- [[entities/grpo]] — GRPO/DAPO training on top of v3 SFT base
- [[sources/code/training-and-benchmark]] — benchmark integration

## Sources

- `pipeline/sft_v3_generator.py`
- `pipeline/validate_sft_data.py`
- `pipeline/2_model_trainer.py`
