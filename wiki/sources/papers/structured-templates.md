---
title: "Can Structured Templates Facilitate LLMs in Tackling Harder Tasks? An Exploration of Scaling Laws by Difficulty"
type: source
tags: [reasoning, small-model, sft, grpo, distillation]
sources:
  - https://arxiv.org/abs/2508.19069
updated: 2026-07-22
status: current
---

# Can Structured Templates Facilitate LLMs in Tackling Harder Tasks? (SST)

**More synthetic training data can hurt hard-task reasoning because most generated problems are too easy; instead, extract abstract solution templates ("chains") from expert solutions and use them as cognitive scaffolds — the Structured Solution Template (SST) framework lifts a 1.5B model by +6.2 points on GSM8K and +2.2 on AIME24.**

## Summary

Yang et al. (2025) document a "scaling law by difficulty": as a synthetic dataset grows 0k→100k, DeepSeek-R1-Distill-Qwen accuracy on AIME24 consistently *drops*, because LLM-generated problems are mostly easier than the originals and induce shortcut heuristics — a U-shaped curve where excess low-difficulty data erodes abstraction. Their fix, SST, has three stages: weighted `<chain>` supervision (chain tokens up-weighted early), a lightweight LoRA "chain generator" that injects a problem-specific template at prompt time, and curriculum SFT+GRPO on the hardest problems in a "Plan-then-Think" format. On DeepSeek-R1-Distill-Qwen-1.5B it reaches GSM8K 84.80 (+6.18) and AIME24 30.48 (+2.18). This is the most directly on-thesis of its cluster — concrete evidence that template scaffolding raises a *small* model's reasoning without more data.

## Why it matters here

Directly supports the template-scaffolding-for-weaker-models line: an explicit structural template forces a small model to represent the *procedure* rather than memorise surface forms. The "scaling law by difficulty" is a useful caution for the project's own data-generation — more synthetic data is not automatically better; favour hard, well-structured data. Reported on GSM8K with GRPO, aligning with the pipeline's benchmark and RL choices. Potentially load-bearing if the pipeline adopts template scaffolding.

## Method

- **Stage 1 — Weighted chain supervision:** templates extracted by Qwen2.5-14B-Instruct, `<chain>`-tagged, trained with a dynamically weighted cross-entropy (chain tokens weighted high early, decaying to 1) to prevent template overfitting.
- **Stage 2 — Prompt-time chain injection:** a Qwen2.5-Math-1.5B-Instruct LoRA generator emits a problem-specific template prepended at inference (main-model weights untouched).
- **Stage 3 — Curriculum SFT + GRPO:** rejection-sample ~20K hardest Open-R1 problems, DeepSeek-R1 API generates `<think><chain>…</chain>…</think>` solutions, distil to 3,843 examples, SFT cold-start then GRPO.

## Key results

- **Headline (1.5B):** GSM8K 84.80 (+6.18 over 78.62), AIME24 30.48 (+2.18), MATH500 84.15 (+1.05).
- **Token efficiency:** on easy problems chains *compress* reasoning (GSM8K −62.4% tokens); on hard problems they scaffold with little token change.
- **GRPO amplifies:** up to +6.72 on the new "Dynamic Math" benchmark.
- **Stage-specialised:** Stage 1 → MATH-style, Stage 2 → competition/AIME, Stage 3 → balanced.

## Critical appraisal

The strongest small-model-scaffolding evidence in its cluster, with a genuinely useful data-curation lesson (the U-shaped difficulty law). Cautions: **Stage 2 alone drops GSM8K to 73.79 — below baseline** — so "templates help" is conditional and only the full three-stage pipeline recovers (a heavy pipeline: separate LoRA generator + R1-API distillation); large error bars on small competition sets (AMC23 ±8.64) make several gains statistically soft; evaluation is maths-only.

> ⚠ 0.6B caution: the student is 1.5B and the scaffolds come from a **14B extractor and R1-scale teacher**, so gains may not transfer cleanly to a 0.6B student, and the teacher-dependence is a real caveat for a fully on-device setting.

## Related

- [[sources/papers/gsm8k]] — the benchmark these results are on
- [[sources/papers/deepseekmath]] / [[sources/papers/dapo]] — the GRPO stage
- [[sources/papers/thinker]] — SFT-taught reasoning structure at small scale; convergent
- [[sources/papers/qwen25-math]] — data-curation-over-scale for small math models
- [[sources/papers/lima]] — data quality > quantity (a related curation argument)
- [[topics/reasoning]] — procedural scaffolding vs surface pattern-matching
- [[experiments/thinker-executor-experiment]] — structured reasoning for a small model

## Sources

- Yang, Fan, Li, Hu, Wang, Qiu, Wang, Sun, Wu (2025) — arXiv:2508.19069 — [arxiv.org/abs/2508.19069](https://arxiv.org/abs/2508.19069)
