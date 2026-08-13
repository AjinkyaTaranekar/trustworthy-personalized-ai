---
title: "Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition"
type: source
tags: [reasoning, metacognition, distillation, grpo]
sources:
  - https://arxiv.org/abs/2505.22375
updated: 2026-07-19
status: current
---

# Pangu Embedded: A Dual-system LLM Reasoner with Metacognition

**A single 7B model can hold both a "fast" System-1 (direct answer) and a "slow" System-2 (explicit chain-of-thought) mode, switching between them either by an explicit user meta-prompt or adaptively by self-assessing query complexity ("metacognition"), so it spends reasoning tokens only when needed — a dual-system-in-one-model design for latency-constrained deployment.**

## Summary

Huawei's Pangu Team (2025) borrow Kahneman's System-1/System-2 to avoid always-on long CoT, which is wasteful on easy queries but necessary on hard ones. Their 7B model fuses both modes via a two-stage recipe — build a strong reasoner (iterative distillation with inter-iteration model merging + GRPO with the MARS multi-source reward), then fuse fast+slow behaviour and teach mode-switching, both manual (`META_PROMPT: system 1/2`) and adaptive. System-2 reaches AIME'24 81.9 / GPQA 68.0; System-1 is much cheaper (AIME 35.8 / GPQA 58.0). The large mode gap quantifies exactly what System-2 buys — and what adaptive switching risks losing if it mis-routes. Conceptually it supports a thinker that decides *how hard to think*, but "embedded" here means a 7B model on datacentre NPUs, not phone-scale.

## Why it matters here

Strong conceptual support for a **dual-system fast/slow thinker** and *metacognitive mode-switching* as a first-class design element — architecturally cheaper than running two models, and a natural fit for a Thinker that decides *how hard to think* before dispatching to a thin Executor. It aligns with Qwen3's thinking/non-thinking modes ([[sources/papers/qwen3-tr]]) the project's base already offers, and the complexity-conditioned data-selection recipe is borrowable for teaching a small model when to escalate.

## Method

- **Stage 1 — reasoner:** SFT via model-aware iterative distillation with inter-iteration model merging (harder-for-current-model examples prioritised by a complexity score), then GRPO guided by MARS (rule + lightweight-LLM verifiers, ~95% precision), on 1,024 Ascend NPUs with a stale-synchronous-parallel scheduler (~30% idle-time cut).
- **Stage 2 — dual-system:** manual switching via meta-prompts implemented by "fusion training" (replay mastered slow-thinking data + introduce fast-thinking examples); adaptive switching learned from curated difficulty distributions.

## Key results

- **System-2:** AIME'24 81.9, GPQA 68.0, LiveCodeBench 67.1, MMLU-Pro 79.0 (beats Qwen3-8B / GLM-4-9B).
- **System-1:** AIME'24 35.8, GPQA 58.0 — "highly competitive" while much cheaper.
- Complexity-based data selection lifts AIME'24 50.42 vs 43.33.

> Note: the exact adaptive-switching efficiency ablation and full baseline rows were truncated in the fetched HTML; the qualitative "dynamically allocates compute" claim is confirmed but the precise reduction figure is unverified.

## Critical appraisal

The dual-system-in-one-model idea is clean and the mode-gap numbers usefully quantify the fast/slow trade-off. But this is a capabilities/infrastructure report, not a study of the *controller*: the adaptive switch — the most trust-relevant part — is asserted and lightly ablated (no router accuracy, calibration, or cost-of-error analysis), and "metacognition" oversells a complexity classifier. Efficiency claims are real but Ascend-specific.

> ⚠ 0.6B cautions (significant): "embedded" is a misnomer for phone-scale — this is a **7B** model on datacentre NPUs, so no number evidences 0.6B feasibility; the AIME 81.9-vs-35.8 mode gap warns that at small scale the fast mode may be too weak for adaptive routing to help (the slow mode carries the model), and a 0.6B model may lack headroom for a useful System-2 at all; the router-calibration risk (silently mis-routing a hard query to fast mode, losing ~46 points) is a concrete trust failure the project must evaluate explicitly, since Pangu does not.

## Related

- [[sources/papers/qwen3-tr]] — the project base's built-in thinking/non-thinking modes + budget
- [[sources/papers/hierarchical-reasoning-model]] — another slow+fast dual-module design
- [[experiments/thinker-executor-experiment]] — a Thinker deciding how hard to think
- [[sources/papers/deepseekmath]] — the GRPO stage Pangu's RL uses
- [[entities/qwen3-0.6b]] — the sub-1B target below Pangu's 7B
- [[topics/reasoning]] — adaptive reasoning-depth allocation

## Sources

- Pangu Team, Huawei (2025) — arXiv:2505.22375 — [arxiv.org/abs/2505.22375](https://arxiv.org/abs/2505.22375)
