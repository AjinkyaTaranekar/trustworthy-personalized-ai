---
title: "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions"
type: source
tags: [evaluation, abstention, alignment]
sources:
  - https://arxiv.org/abs/2506.09038
  - https://github.com/facebookresearch/AbstentionBench
updated: 2026-07-20
status: current
---

# AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions

**Appropriate abstention on unanswerable or underspecified questions is an unsolved capability that does not improve with scale and is actively degraded by reasoning fine-tuning (−24% on average) — so a model can be more "accurate" yet less honest about what it cannot answer.**

## Summary

Kirichenko et al. (FAIR / Meta, 2025) build a ~35,000-question benchmark across 20 datasets and six scenario categories (answer-unknown, false-premise, stale, subjective, underspecified-context, underspecified-intent) and evaluate ~20 models. Two findings matter most: **scale is nearly inert** for abstention (Llama 8B/70B/405B show almost no change; frontier models sit only marginally above small ones), and **reasoning fine-tuning *degrades* abstention by ~24%** even where it raises accuracy — with staged Tülu 3 checkpoints pinning the RLVR (verifiable-reward RL) stage as a key degrader. A carefully crafted system prompt helps but is a band-aid. This is a cornerstone for the honesty/abstention trust axis and unusually encouraging for small models.

## Why it matters here

It operationalises "knowing and saying when you don't know" as a measurable behaviour *orthogonal to accuracy* — the argument for treating abstention as its own [[entities/constitution|constitution]] principle rather than assuming a smarter model abstains better. The six scenario categories are a ready taxonomy for constitution clauses and an abstention eval set. Crucially for sub-1B: because scale barely moves abstention, **a 0.6B model is not inherently doomed on this axis relative to 405B** — a place where a well-constituted small model can be competitive, with SFT a promising lever (their SFT stage *improved* abstention) and RLVR a caution.

## Key results

- **Reasoning fine-tuning:** −24% abstention on average (DeepSeek-R1, s1 families), even on math/science where accuracy rises.
- **Scale inert:** near-flat mean abstention across Llama 8B→405B; best average performers GPT-4o and Qwen2.5-32B.
- **Staged attribution:** SFT and DPO improve abstention; **RLVR degrades it** (sharp SFT→RLVR drop on underspecified context).
- **Test-time compute:** 512→4,096 reasoning tokens improves accuracy but *hurts* abstention.
- **Reasoning paradox:** models voice uncertainty inside the chain-of-thought yet still emit a definitive final answer.

## Critical appraisal

A strong, well-scoped negative result that isolates a capability scaling and reasoning-tuning fail to fix (and can worsen), with a genuinely useful causal-ish handle via staged checkpoints. The −24% is a clean, quotable headline. Weaknesses: the single 8B LLM-judge is itself a non-abstaining reasoning-style model (judge error may correlate with the failure being measured), and "abstention recall" flattens six heterogeneous failure causes into one axis. English-only; possible CoCoNot/Tülu leakage.

> ⚠ Design warning for a think-then-answer harness on a small model: the reasoning-paradox (doubt in CoT that doesn't propagate to the answer) is a concrete failure mode; and any RLVR-style stage risks trading honesty for answer-completion.

## Related

- [[sources/papers/general-language-assistant]] — HHH "honest" = calibrated uncertainty; abstention operationalises it
- [[sources/papers/hallucination-survey]] — the faithfulness/factuality failure abstention guards against
- [[sources/papers/reducing-safety-tax]] — RLVR/RL alignment interactions with capability
- [[sources/papers/reducing-safety-tax]] — on-policy alignment as an alternative to RLVR degradation
- [[entities/constitution]] — abstention as an explicit principle
- [[experiments/human-evaluation-rubric]] — honesty/uncertainty as a scored trust axis
- [[topics/explainability]] — honest "I don't know" as a trust behaviour

## Sources

- Kirichenko, Ibrahim, Chaudhuri, Bell (2025) — arXiv:2506.09038 — [arxiv.org/abs/2506.09038](https://arxiv.org/abs/2506.09038)
