---
title: "Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation"
type: source
tags: [constitutional-ai, constitution, evaluation, principles]
sources:
  - https://arxiv.org/abs/2605.24247
updated: 2026-07-20
status: current
---

# Improving Labeling Consistency with Detailed Constitutional Definitions

**Replacing short one-or-two-sentence category definitions with detailed per-category "constitutions" — long structured specifications that frontier LLMs interpret literally — produces golden labels that are dramatically more consistent across models (up to 57× less cross-model disagreement) and more unanimous than a panel of human annotators.**

## Summary

Berlin and Swanda (Cisco AI Defense, 2026) extend Constitutional AI from model *training* to the *labelling* context. Each moderation category (harassment, hate speech, non-violent crime) is specified by a ~10-component constitution — definitions, key terms, decision logic, conservatism level, an intent axis and a content axis, boundaries, worked examples, edge cases — and multiple frontier LLMs read it to produce labels; cross-model disagreement both signals quality and drives iterative refinement. Short definitions force annotators (human or model) to fill boundary cases from private priors; a detailed spec "fixes a shared prior". Result: up to 57× less cross-model disagreement, three LLMs more unanimous than three human annotators on all three categories, and — the key caution — **nano-class models an order of magnitude worse**, because reading and applying a long spec is capability-gated.

## Why it matters here

The primary hook is *how to write* the project's constitution: this is a concrete template for turning each of the 19 principles into an operational spec (definitions, decision logic, conservatism, worked examples, edge cases) rather than a one-line rule, with the predicted payoff (much higher inter-model/inter-rater consistency). The "use cross-model disagreement to find spec gaps, then refine" loop is a ready methodology for validating the constitution and the [[sources/papers/mt-bench|LLM-judge]] pipeline — disagreement localises which principle wording is ambiguous, directly supporting the "validated judge over regex" stance.

## Method

- **Constitutional structure:** ~10 components per category; **dual-axis** intent + content as independent binary judgements over a full conversation.
- **Refinement loop:** humans surface problems → AI revises the relevant constitutional section; 6 LLMs from 3 vendors surface disagreements that pinpoint spec gaps; implicit rulings iteratively become explicit rules.
- **Data:** HarmBench (392, 4 human annotators), WildChat (~1M organic; ~200 suspected-positive + ~1,000 conservative-negative per category). Models: GPT-5.4/Mini/Nano, Opus 4.6, Gemini 3.1 Pro, Safeguard 20B.

## Key results

- **Cross-model disagreement:** up to **57×** lower than paragraph definitions; frontier models disagree <3 per 1,000 conversations on most category/axis pairs.
- **LLMs > humans on unanimity** (per 1,000, HarmBench): Non-Violent Crime human 301.0 vs LLM 84.2; Harassment 43.4 vs 37.9; Hate Speech 23.0 vs 13.5.
- **Accuracy:** Harassment F1 0.47 → 0.65 via iterative refinement (the only hard accuracy anchor).
- **Model class:** frontier ≈ mini within range; **nano-class an order of magnitude worse**.

## Critical appraisal

The most directly useful paper here for operationalising a written constitution, and the cross-model-disagreement-as-refinement-signal loop is elegant and reusable. The weakness is the **consistency-vs-correctness gap**: it proves reproducibility far more than correctness (only one modest F1 anchor, and six models can consistently follow a *flawed* rule). No component ablation, so the recipe's active ingredient (the 10 components vs just the worked examples) is unidentified; tiny positive counts (as few as 16) make some numbers fragile; proprietary-frontier, English moderation only.

> ⚠ Small-model caution (important): the nano-class result is the key warning — faithfully applying a long constitution is capability-gated and the smallest models degrade sharply, so on a sub-1B student inference-time constitutional prompting alone may fail. This strengthens the case for baking the constitution in via SFT, and for a *distilled/compressed* constitution rather than handing the model the full document.

## Related

- [[sources/papers/c3ai]] — crafting/pruning constitutions; compact vs full
- [[sources/papers/effective-cai-small-llms]] — small-model constitutional capability floor
- [[sources/papers/mt-bench]] — judge disagreement as a calibration signal
- [[entities/constitution]] — the principles to expand into operational specs
- [[sources/code/constitution-document]] — the project's full constitution source
- [[topics/explainability]] — explicit decision logic and auditability

## Sources

- Berlin, Swanda (Cisco AI Defense, 2026) — arXiv:2605.24247 — [arxiv.org/abs/2605.24247](https://arxiv.org/abs/2605.24247)
