---
title: "Length-Unbiased Sequence Policy Optimization (LUSPO): Revealing and Controlling Response Length Variation in RLVR"
type: source
tags: [rl, grpo, caveat]
sources:
  - https://arxiv.org/abs/2602.05261
updated: 2026-07-21
status: current
---

# Length-Unbiased Sequence Policy Optimization (LUSPO)

**Mainstream RLVR objectives (GRPO, GSPO) carry a hidden response-length bias baked into their per-token/per-sequence gradient normalisation; removing GSPO's `1/|y|` length normalisation makes the loss length-unbiased, cures GSPO's response-length collapse, and yields higher reasoning accuracy across dense, MoE, and vision-language models.**

## Summary

Liu et al. (Meituan, 2026) diagnose a concrete artefact in the dominant RLVR objectives: because [[entities/grpo|GRPO]] averages each token's contribution within a trajectory (divides by response length), a fixed sequence-level advantage is spread thinner over long responses and concentrated over short ones — a length bias favouring short-correct and long-incorrect answers, which GSPO's sequence-level clipping amplifies into *response-length collapse*. The fix (LUSPO) scales each sequence's loss by its length `|y_i|` (drops the `1/|y_i|` divisor), making gradient magnitude length-independent. It cures the collapse and lifts accuracy by +4.0 avg over GSPO on Qwen2.5-7B (MATH500 +7.4; Qwen3-30B AIME25 +17.1). This is the RL-family length/entropy-stability contribution, and its documented RLVR fragility is indirect support for the SFT-only pivot.

> Naming: `ref.bib` and the paper title use "Length-Unbiased Sequence Policy Optimization"; the method is abbreviated **LUSPO** in the paper (the earlier working label "LSPO" refers to the same method).

## Why it matters here

Maps to the project's RL axis (RLVR length/entropy stability). If the project ever reports GRPO length or entropy instability, LUSPO *names the cause* (length-dependent gradient normalisation) and gives the correction. More usefully for the current direction: it documents that RLVR objectives are *fragile* — GSPO can collapse response length outright and needs a bespoke correction to behave — supporting evidence that RLVR adds optimisation-stability risk a small-model constitution project may reasonably avoid, favouring SFT. Contrast partner for the forgetting papers: they attack SFT-side forgetting, LUSPO attacks RL-side instability — together the project can argue *both* fine-tuning families have known failure modes and its SFT-only path is lower-risk.

## Method

- **Diagnosis:** decompose GRPO/GSPO — both carry a `1/|y_i|` factor making per-token gradient length-dependent; GSPO's sequence ratio + clip-higher amplifies it into collapse.
- **LUSPO:** scale the GSPO per-sequence loss by `|y_i|` (remove the `1/|y_i|` divisor) → length-unbiased gradient, no new reward or sampling scheme.

## Key results

- **Qwen2.5-7B (vs GSPO):** AIME24 +2.9, AIME25 +2.7, MATH500 +7.4 (+4.0 avg).
- **Qwen3-30B-A3B:** AIME25 76.3% vs 59.2% (+17.1 — largest single gain).
- **Multimodal (Qwen2.5-VL-7B):** LogicVista +6.0 over GSPO.
- **Length:** LUSPO sustains ~1.5× longer, stable responses (3940 vs 2611 tokens) where GSPO collapses.

## Critical appraisal

A focused, mechanistic optimiser analysis: it isolates a real artefact (length-dependent gradient normalisation) in the dominant RLVR objectives and offers a one-line correction with consistent, sometimes large gains. The diagnosis is the real contribution. Cautions: all tasks are verifiable-reward STEM/math/multimodal reasoning (no open-ended/dialogue), so external validity beyond RLVR is untested; removing `1/|y|` mechanically *rewards* longer sequences, so the "unbiased" label may in practice be "biased in the pro-length direction" — verify the neutrality claim rather than accept the branding; AIME sets are tiny (30 questions); no inference-cost accounting for the ~1.5× longer outputs.

> ⚠ 0.6B relevance is weak/indirect: all models are 7B–30B; RLVR on a 0.6B constitution model is exactly the regime the project is de-scoping. Treat LUSPO as *background/related-work* framing for the RL family, not a method to run.

## Related

- [[entities/grpo]] — the objective whose length bias this corrects
- [[sources/papers/dapo]] — a different GRPO length/entropy-stability fix (clip-higher, token-level loss)
- [[sources/papers/understanding-r1-zero]] — Dr. GRPO, the length-bias critique lineage
- [[sources/papers/ragen]] — multi-turn RL instability (Echo Trap) at small scale
- [[sources/papers/entropy-adaptive-ft]] — the SFT-side failure-mode partner
- [[decisions/2026-05-03-research-question-reframe]] — RLVR fragility supports the SFT-only pivot

## Sources

- Liu, Yin, Shi, Yang, Zeng, Qiu (Meituan, 2026) — arXiv:2602.05261 — [arxiv.org/abs/2602.05261](https://arxiv.org/abs/2602.05261)
