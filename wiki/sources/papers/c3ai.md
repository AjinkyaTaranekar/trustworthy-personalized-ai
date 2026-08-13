---
title: "C3AI: Crafting and Evaluating Constitutions for Constitutional AI"
type: source
tags: [constitutional-ai, constitution, principles, evaluation]
sources:
  - https://arxiv.org/abs/2502.15861
updated: 2026-07-18
status: current
---

# C3AI: Crafting and Evaluating Constitutions for Constitutional AI

**An end-to-end framework that crafts a CAI constitution by collecting, standardising and selecting principles from psychology and AI-ethics sources, then evaluates both which principles match human preferences and whether a fine-tuned model actually adheres to them — surfacing a design-adherence gap: the positively-framed principles humans prefer are exactly the ones fine-tuned models follow worst.**

## Summary

Kyrychenko et al. (2025) treat constitution design as an empirical discipline rather than an art. They compile 495 candidate items (Anthropic's constitution, Collective CAI, Schwartz values, Jigsaw bridging attributes) into 185 machine-readable principles, measure principle-objective alignment over 333,000 measurements, and distil to ~15 principles via psychometric graph analysis (EGA/UVA). Two robust findings: **positive framing aligns +27%** with human preference and behaviour-framing beats trait-framing (+5%); yet after ORPO fine-tuning, models adhere best to *negatively-framed prohibitions* (F5 win-rate >0.55) and worst to the *positive abstract values humans prefer* (F6 <0.35). Crucially, the compact 15-principle constitution matched the full 58 on TrustLLM safety **without a reasoning tax** (MMLU/GSM8K held). This is the project's closest methodological neighbour.

## Why it matters here

It gives the dissertation a citable *methodology for crafting and evaluating* its own [[entities/constitution|written constitution]] rather than hand-waving it, plus concrete framing rules (prefer positive, behaviour-oriented principles for human acceptability) and evidence that a compact constitution preserves capability. It also supplies an adherence-evaluation template (an independent LLM judge scoring responses against each principle) consistent with the project's substance-based, no-hardcoded-eval stance. And it explicitly calls for *personalised* constitutions — directly resonant with the personalisation angle.

## Method

- **Item pipeline:** 495 items → 185 standardised principles (88.5% needed no modification).
- **Principle-objective alignment:** a Llama-3-8B evaluator (3-shot) chooses between two responses per principle over 1,800 conversations (five human-preference datasets) → 333,000 measurements; avg alignment 57.8%.
- **Framing regression (mixed-effects):** positive vs negative OR=1.27; trait vs behaviour OR=0.95.
- **Psychometric distillation (EGA + UVA):** 185 → 14–15 non-redundant, stable principles; six latent factors (F6 ethics/rights strongest human alignment, OR 1.86).
- **Fine-tuning:** Orpo-Llama-3-8B + ORPO on 11,230 HH-RLHF conversations; Anthropic (58) vs Anthropic-EGA (15); adherence scored by an independent Llama-3-8B judge (90% selection accuracy).

## Key results

- **Design-adherence gap:** models score >0.55 win-rate on negatively-framed F5 (Non-Aggression 0.627) but <0.35 on positively-framed F6 (Friendly Response 0.257) — the inverse of human preference.
- **Compact ≈ full on safety:** Anthropic-EGA (15) vs Anthropic (58) vs baseline — Jailbreak 0.679 / 0.580 / 0.447; Exaggerated Safety 0.390 / 0.420 / 0.560; MMLU 0.663 / 0.660 / 0.658; GSM8K 0.484 / 0.492 / 0.460. Safety up, reasoning preserved.

## Critical appraisal

The first systematic, quantitative treatment of how to write and prune a constitution, well-instrumented (regressions, psychometrics, safety benchmarks). Cautions: everything is 8B, single-turn, LLM-judged (a Llama-3-8B judge scoring a Llama-3-8B-derived model risks circularity); "human preference" is dataset labels, not fresh raters; per-principle alignment hovers near 58% (barely above chance) and only becomes decisive in aggregate.

> ⚠ Design hook + caution: prefer positively-framed, behaviour-oriented principles for acceptability, but the design-adherence gap warns a small SFT'd model follows prohibitions more reliably. For a 0.6B model (likely *more* rule-rigid than 8B) expect the gap to widen — pair aspirational positive principles with concrete behavioural operationalisations, and report *per-principle* adherence. The traits this thesis wants (warmth, empathy, honesty) are exactly the abstract positive kind models adhere to worst.

## Related

- [[sources/papers/constitutional-ai-bai]] — the original CAI this operationalises how to design for
- [[sources/papers/constitution-or-collapse]] — CAI at 8B; helpfulness cost
- [[sources/papers/effective-cai-small-llms]] — small-model CAI architecture dependence
- [[entities/constitution]] — the project's ~23-principle constitution to audit for redundancy/framing
- [[topics/constitution-psychological-grounding]] — principles mapped to psychology theory
- [[topics/personalisation]] — the personalised-constitution direction C3AI calls for

## Sources

- Kyrychenko et al. (2025) — arXiv:2502.15861 — [arxiv.org/abs/2502.15861](https://arxiv.org/abs/2502.15861)
