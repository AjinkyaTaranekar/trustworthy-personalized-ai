---
title: "Phi-4 Technical Report"
type: source
tags: [small-model, distillation, foundations]
sources:
  - https://arxiv.org/abs/2412.08905
updated: 2026-07-20
status: current
---

# Phi-4 Technical Report

**A 14B open-weight model matches or beats far larger frontier models on STEM reasoning — even surpassing its own teacher GPT-4o on GPQA (56.1 vs 50.6) and MATH (80.4 vs 74.6) — when the training corpus is dominated by carefully engineered synthetic data (40% of a 10T-token mixture) rather than raw web scrape: data quality, not parameter count, is the lever.**

## Summary

Abdin et al. (Microsoft Research, 2024) build Phi-4 around a synthetic-data-centric mixture (~400B synthetic tokens across ~50 dataset types, re-epoched ~12–14× without overfitting), engineered on four principles — diversity, complexity, accuracy, chain-of-thought — with execution/test validation, instruction reversal, and majority-voting difficulty balancing. Post-training adds Pivotal Token Search DPO (isolating the single tokens that flip a solution's correctness) and judge-guided DPO. Validated on the fresh Nov-2024 AMC-10/12 (post-dating all training data) to rebut contamination. The defensible claim isn't "14B beats GPT-4o" (it doesn't overall — GPT-4o still wins MMLU, HumanEval, IFEval, SimpleQA) but the sharper one: on well-specified reasoning tasks, a curated synthetic corpus lets a small model exceed the very model that generated its data. This is the flagship "data-quality small models" evidence.

## Why it matters here

Central to the data-quality-small-models pillar. The size-agnostic recipe — four data principles, execution/test validation, instruction reversal, majority-voting difficulty balancing — is directly portable to seeding a constitution-guided data pipeline for a 0.6B student, and Pivotal Token Search DPO is a candidate to sharpen reasoning trust where every preference token counts. The teacher-dependence caveat mirrors the project's teacher/executor distillation framing.

## Key results (Phi-4 14B)

- **Beats GPT-4o on STEM:** GPQA 56.1 (vs 50.6), MATH 80.4 (vs 74.6); MMLU 84.8 (vs 88.1 — GPT-4o still leads broadly).
- **Data mixture:** 40% synthetic, 15% web-rewrites, 15% filtered web, 20% code, 10% academic; many synthetic epochs *outperform* seeing more unique web tokens.
- **Fresh AMC Nov-2024:** scores above its weight class — clean anti-contamination evidence.
- **Post-training ablation:** Pivotal-token DPO helps reasoning most; judge DPO helps style/ArenaHard — complementary.
- **Safety:** GCG adversarial suffixes "did not transfer to Phi-4"; jailbreak DR 0.073.

## Critical appraisal

The enduring contribution is the data-generation stack, well-evidenced by the fresh AMC test. Honest caveats: "synthetic data" is largely generated *by GPT-4o*, so the recipe presupposes a strong frontier teacher (efficient distillation-plus, not bootstrapping); it's a non-peer-reviewed report with the pipeline described qualitatively (exact prompts/thresholds withheld); benchmark selection favours reasoning where synthetic shines.

> ⚠ For a 0.6B thesis: pure synthetic erodes factual recall (TriviaQA −14.8 in ablation; Phi-4 hallucinates biographies, SimpleQA 3.0) — reinforcing that trust/refusal behaviour and retrieval must compensate for a small model's thin world-knowledge. Weak instruction-following (IFEval 63.0 vs GPT-4o 84.8) and long-context lag (HELMET, below) also carry down-scale.

## Related

- [[sources/papers/phi3-tr]] — the on-phone predecessor; same data-quality thesis
- [[sources/papers/lima]] / [[sources/papers/qlora]] — data quality > quantity, independently
- [[sources/papers/context-length-hurts]] — the HELMET long-context lag, convergent evidence
- [[sources/papers/hallucination-survey]] — the factual-recall weakness this trades for reasoning
- [[entities/qwen3-0.6b]] — the sub-1B student the recipe would seed
- [[topics/reasoning]] — synthetic reasoning data for small models

## Sources

- Abdin et al. (Microsoft Research, 2024) — arXiv:2412.08905 — [arxiv.org/abs/2412.08905](https://arxiv.org/abs/2412.08905)
