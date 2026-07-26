---
title: "An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability"
type: source
tags: [evaluation, llm-as-judge]
sources:
  - https://arxiv.org/abs/2506.13639
updated: 2026-07-22
status: current
---

# An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Reliability

**For open-ended LLM-as-a-Judge evaluation, explicit evaluation criteria (rubrics) are the dominant lever on human-agreement reliability; non-deterministic sampling with mean aggregation beats greedy decoding; and Chain-of-Thought adds essentially nothing once clear criteria are present — so spend the design budget on the rubric, not on CoT.**

## Summary

Yamauchi, Yano and Oyamada (U. Tokyo / NEC, 2025) run a controlled ablation of judge-design choices — reference answer, score-description rubric, rubric granularity, decoding (greedy vs sampled+aggregated), CoT — measuring both *alignment* (Pearson vs humans) and *consistency* (Krippendorff's α) on GPT-4o and LLaMA-3.1-70B judges over BiGGenBench and EvalBiasBench. The ordered recipe: **criteria > reference > decoding-aggregation > CoT**. Removing criteria drops Pearson 0.666→0.591 (GPT-4o) and 0.641→0.555 (LLaMA), and removing both collapses the weaker judge to 0.346; mean-of-5 sampling beats greedy by ~2–3 points; CoT adds ~0.002 with criteria present (it only helps as a *substitute* for a missing rubric); anchoring only score levels 1 and 5 preserves alignment cheaply. A tightly scoped, directly actionable companion to [[sources/papers/biggen-bench]].

## Why it matters here

The empirically-grounded recipe for tuning the project's LLM-judge, mapping almost one-to-one: (1) always give instance criteria (biggest lever, *most* important for weaker judges); (2) include a reference answer (secondary); (3) anchor only the extreme score levels to save rubric-writing cost; (4) prefer sampled decoding with **mean** aggregation over greedy; (5) drop CoT when criteria are strong to save tokens. Criteria help *more* for the weaker judge and CoT can partly *substitute* for a missing rubric — both suggest a small judge can be materially rescued by good scaffolding.

## Method

- **Factors:** reference present/absent; rubric present/absent; granularity (all five levels vs only 1 & 5); decoding (greedy vs sampling + majority/median/mean); CoT vs direct.
- **Judges:** GPT-4o-2024-08-06 (strong closed) + LLaMA-3.1-70B (weaker open). Metrics: Pearson (alignment), Krippendorff α (consistency), 5 seeds at temp 1.0.

## Key results

- **Criteria dominate:** removing them costs ~0.07–0.10 Pearson; removing criteria + reference collapses GPT-4o to 0.487, LLaMA to 0.346.
- **Mean > median > majority > greedy:** mean-of-5 beats greedy by ~+0.02–0.03 everywhere (averaging preserves fractional scores).
- **CoT ≈ free lunch only without criteria:** with good criteria it adds ~0.002.
- **Endpoint-only rubrics (levels 1 & 5)** give the best alignment cheaply.

## Critical appraisal

Turns folklore ("add CoT", "use greedy for determinism") into measured effects; the cleanest defensible claim is that a good rubric dominates every other knob and CoT is redundant given one. Cautions: only two *large* judges (≥70B), English-only, modest effect sizes without confidence intervals (median-vs-majority may be noise), and — crucially — mixed criteria provenance (EvalBiasBench criteria are GPT-4o-generated while GPT-4o judges, a mild self-preference confound).

> ⚠ 0.6B: every judge here is ≥70B — the sub-1B regime, its most valuable question, is untested; mean-aggregation needs 5× inference (costly on-device); and **consistency (α) stays high even when alignment collapses**, so a small judge can be *reliably wrong* — consistency metrics alone can't validate an on-device judge; human-alignment spot-checks stay mandatory.

## Related

- [[sources/papers/biggen-bench]] — the rubric half; both use BiGGenBench, they stack
- [[sources/papers/prometheus]] — rubric + reference + feedback-before-score judge design
- [[sources/papers/mt-bench]] — the LLM-as-judge bias/reliability tradition
- [[experiments/human-evaluation-rubric]] — the project's judge to tune with this recipe
- [[topics/explainability]] — alignment vs consistency as distinct judge properties
- [[sources/code/training-and-benchmark]] — where the judge runs

## Sources

- Yamauchi, Yano, Oyamada (2025) — arXiv:2506.13639 — [arxiv.org/abs/2506.13639](https://arxiv.org/abs/2506.13639)
