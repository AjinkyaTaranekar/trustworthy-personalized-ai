---
title: "ToolMind Technical Report: A Large-Scale, Reasoning-Enhanced Tool-Use Dataset"
type: source
tags: [tool-use, sft, agents, small-model]
sources:
  - https://arxiv.org/abs/2511.15718
updated: 2026-07-19
status: current
---

# ToolMind: A Large-Scale, Reasoning-Enhanced Tool-Use Dataset

**A 360k-sample tool-use dataset (160k synthetic via 20k+ tools + 200k augmented) whose distinguishing move is fine-grained turn-level filtering — beyond the usual trajectory-level correctness check — removing turn-level errors before training while preserving self-corrective reasoning signals; without it, the entire τ-bench SFT gain evaporates.**

## Summary

Yang et al. (Nanbeige Lab / RUC, 2025) argue that most tool-use data is validated only at the *trajectory* level (does the whole dialogue reach the goal?), which overlooks *turn-level* errors that propagate during training. A trajectory can reach the right answer through locally-wrong steps; training on those teaches bad habits. ToolMind builds a function graph from parameter correlations, samples tool chains by random walk, simulates user/assistant/tool dialogues, then applies two-stage filtering (trajectory then turn level). SFT on it lifts Qwen3-14B on τ-bench 38.78 → 53.00 and reaches GPT-4o parity on BFCL-v4 (50.54 vs 50.27). The decisive ablation: removing turn-level filtering drops Qwen3-8B on τ-bench from 46.70 back to 35.31 (≈ baseline). The durable lesson is conceptual — per-turn verification of training data is what makes small-model tool use work.

## Why it matters here

A direct blueprint for building the Executor's SFT corpus (function graph → chains → multi-agent simulation, no real tool infrastructure needed), and the trajectory-vs-turn-level distinction *is* the verification question for a thinker/executor split: the +11 τ-bench gain vanishes without per-turn filtering, arguing for turn-level verification of the executor's SFT data rather than only end-to-end trajectory success. The deliberate retention of self-correction traces is a nuance a small model could exploit (learn to recover from a bad call, not just emit perfect calls).

## Method

- **Function graph:** parameters embedded (`DESC ∥ desc ∥ TYPE ∥ type`); edges where cosine similarity exceeds a threshold, LLM-validated (one tool's output plausibly feeds another's input).
- **Chain sampling:** random walks, chain length `L ∼ Uniform{5,…,20}`.
- **Multi-agent synthesis:** User / Assistant / Tool agents simulate realistic multi-turn dialogue.
- **Two-stage filtering:** trajectory-level (goal coverage/coherence) then **turn-level** (remove erroneous/suboptimal turns, keep self-correction). 160k synthetic + 200k augmented = 360k. SFT of Qwen3-8B/14B.

## Key results

- **τ-bench:** 8B 35.83 → 46.70 (+10.87); 14B 38.78 → 53.00 (+14.22).
- **τ²-bench:** 8B +11.73, 14B +8.44. **BFCL-v4:** 8B +4.71, 14B +5.40 (tuned 14B 50.54 ≈ GPT-4o 50.27, > DeepSeek-R1 48.97).
- **Ablation:** without turn-level filtering, 8B τ-bench 46.70 → 35.31 (gain gone).

## Critical appraisal

The durable contribution is the trajectory-vs-turn-level distinction and the finding that turn-level filtering is the difference between +11 points and *no gain*. Weakness: all quality control is **LLM-judge-based over *simulated* tool outputs**, so "correctness" is softer than execution-verified alternatives (T1's sandbox, CoVe's rule verifier), and simulated outputs may embed hallucinations a judge can't catch.

> ⚠ 0.6B caution: floor is 8B; the +11 τ-bench gain is unproven at 0.6B and may not survive the capacity drop, since turn-level self-correction is itself a capability-hungry behaviour. For a *trustworthy* thesis, prefer execution-grounded verification over ToolMind's LLM-judge-over-simulation.

## Related

- [[sources/papers/t1]] — execution-grounded tool-use dataset/harness
- [[sources/papers/cove]] — deterministic constraint-based verification (stronger than LLM-judge)
- [[experiments/thinker-executor-experiment]] — the Executor whose SFT data this informs
- [[sources/code/sft-v3-pipeline]] — the project's own trajectory data generation
- [[sources/papers/qwen3-tr]] — the Qwen3 base tuned here
- [[topics/tool-use-and-verification]] — turn-level verification of tool calls

## Sources

- Yang, Le, Xing, An, Chen, Zhao, Song, Zhang (2025) — arXiv:2511.15718 — [arxiv.org/abs/2511.15718](https://arxiv.org/abs/2511.15718)
