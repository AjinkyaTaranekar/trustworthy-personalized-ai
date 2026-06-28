---
title: SFT v3 Asymmetric Distillation Pipeline
type: source
tags: [sft, distillation, training, tool-use, constitutional-ai, curriculum-learning]
sources:
  - pipeline/sft_v3_generator.py
  - pipeline/sft_dataset_assembler.py
  - pipeline/2_model_trainer.py
updated: 2026-05-29
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
