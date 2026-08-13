---
title: "Replacing Thinking with Tool Usage Enables Reasoning in Small Language Models"
type: source
tags: [reasoning, tool-use, small-model, rl, sft, lora, trade-off]
sources:
  - https://arxiv.org/abs/2507.05065
updated: 2026-07-18
status: current
---

# Replacing Thinking with Tool Usage Enables Reasoning in Small Language Models

**Formatting inference-time "thinking" as multi-turn edits to a stateful code editor (a small DSL over a tool), rather than free-form language chain-of-thought, shrinks the RL action space and densifies reward — letting SFT+RLVR teach genuine reasoning to 1B–3B models, though the advantage over plain CoT reverses at 8B.**

## Summary

Rainone, Bakker and Memisevic (Qualcomm AI Research, 2025) argue that long chain-of-thought and RLVR recipes that lift large models fail on small ones for two structural reasons: free-form CoT gives a combinatorially huge action space (the whole vocabulary at every step) and only a single sparse end-of-trajectory reward, so a low-capacity model cannot explore usefully and collapses into repeating common patterns. Their fix — "Chain-of-Edits" (CoE) — recasts reasoning as a sequence of structured edits to a Python scratchpad through a five-command DSL (`ADDL`, `REPL`, `REPW`, `DELL`, `EXIT`), interleaved with execution feedback, trained by SFT on synthetic repair traces then a GRPO-style RL with a per-turn verifiable reward. The result is a clean small-model win that vanishes with scale, which makes it one of the closest external analogues to this dissertation's tool-use-versus-thinking axis.

## Why it matters here

This is direct evidence for the core on-device bet: a sub-3B model can be given real, verifiable reasoning through **structured tool use** rather than language deliberation. It maps onto the project's student conditions (no-tools / compute-only / all-tools) and onto [[experiments/thinker-executor-experiment]], where a thin Executor issues tool calls. The per-turn "solved now and not before" reward is a transferable RLVR trick for constitution-guided training at small scale. The load-bearing caution is the crossover.

## Method

- **Scratchpad environment.** Line-numbered Python plus an execution-feedback region separated by a `***` delimiter. Each turn the model emits one DSL edit, the code is re-run against unit tests, and feedback is appended for the next turn. Trajectories capped at 10 turns.
- **Stage 1 — SFT.** Demonstrations built by corrupting correct code with 1–5 random edits then *reversing* the corruption to get a ground-truth repair trace. From MBPP: 35,223 training demos (~19.7M tokens). LoRA rank 16 on all linear layers, context 2048, on Llama-3.2-1B/3B and Llama-3.1-8B-Instruct.
- **Stage 2 — RLVR.** Modified GRPO with per-turn reward normalisation: reward 1.0 exactly at the turn that fixes the code (dense signal), −0.5 format penalties for malformed DSL or missing `EXIT`. Group size 4, KL β=0.01, clip ε=0.2, LoRA-only updates.
- **Harder eval set.** 3-shot prompting Llama-3.1-8B to produce faulty solutions (mean edit distance 135.6 from ground truth, vs 39.93 for the SFT demos) — so the RL/eval task is genuinely harder than the imitation data. Failure modes: 81.2% unit-test failures, the rest syntax/name/argument errors.

## Key results (pass@1 / pass@4)

| Model | CoE | s1K (CoT) | Direct |
|---|---|---|---|
| Llama-3.2-1B | **7.82** / 11.0 | 0.15 / 0.53 | 1.3 / 3.1 |
| Llama-3.2-3B | **13.8** / 19.0 | 1.44 / 5.24 | 6.9 / 12.0 |
| Llama-3.1-8B | 21.7 / 32.7 | 23.3 / 46.2 | **33.4** / 42.9 |

At 1B, CoE beats the natural-language s1K baseline by ~50× and Direct by ~6×. **At 8B the ordering flips** — Direct > s1K > CoE — so the tool-over-thinking advantage is a small-model phenomenon, not a universal law. Small models trained on language CoT "regularly get stuck repeating common patterns"; the DSL sidesteps that failure mode.

## Critical appraisal

Trust the qualitative claim (constrained, stateful tool interaction plus dense per-turn reward is what unlocks RL on small models) more than the magnitudes, which sit on a bespoke code-repair benchmark. Caveats: single task family and single model family (Llama), LoRA-only, low absolute accuracies (7.82% at 1B), and a title that generalises past the evidence. The s1K comparison is arguably unfavourable to the baseline (math-reasoning data applied to code repair), so part of the gap may be task-format mismatch rather than a pure tool-vs-thinking effect.

> ⚠ Conflict / caution: the 8B crossover means results at 0.6B must **not** be extrapolated upward, and larger-model behaviour must not be read down onto 0.6B. This tempers any claim that a small-model harness result generalises.

## Related

- [[experiments/thinker-executor-experiment]] — thin tool-calling Executor; this paper is the strongest external support for tool-use substituting for thinking at small scale
- [[sources/papers/beyond-react]] — adjacent small-model tool-use / planner finding; also confirms 0.6B RL instability
- [[sources/papers/reason-plan-react]] — planner supervising a ReAct executor; complementary structural argument
- [[sources/papers/self-enhanced-reasoning]] — small-model reasoning activation via self-training
- [[sources/papers/dual-head-reasoning-distillation]] — train-time-only reasoning at small scale
- [[entities/grpo]] — the RL algorithm modified here with a per-turn reward
- [[topics/tool-use-and-verification]] — tool delegation theory
- [[topics/reasoning]] — trustworthy reasoning across SFT + RL

## Sources

- Rainone, Bakker, Memisevic (2025) — arXiv:2507.05065 — [arxiv.org/abs/2507.05065](https://arxiv.org/abs/2507.05065)
