---
title: "Generative Value Conflicts Reveal LLM Priorities"
type: source
tags: [alignment, constitution, evaluation, principles]
sources:
  - https://arxiv.org/abs/2509.25369
updated: 2026-07-20
status: current
---

# Generative Value Conflicts Reveal LLM Priorities

**Automatically generated value-conflict scenarios (ConflictScope) show an LLM's revealed value priorities depend heavily on evaluation format — models look protective under multiple-choice but shift toward personal/autonomy values in open-ended interaction — and detailed value orderings in the system prompt only partly (~14%) re-steer them.**

## Summary

Liu et al. (CMU / UW, 2025) build ConflictScope: given a value set, it generates scenarios pitting two sampled values against each other, elicits free-text responses from a simulated user, and fits a Bradley-Terry model to extract a value ranking. Across 14 models and three value sets (HHH, Personal-vs-Protective, ModelSpec), **every model except Claude shifts away from protective values toward personal ones when moving from multiple-choice to open-ended** — a large, consistent format effect. System-prompt value orderings recover only ~14% on average (0.01 for Claude Haiku, 0.27 for OLMo-2-32B). This is the closest paper to the project's problem that its constitution contains principles that can conflict, and a direct warning against validating a constitution via MCQ-style probes.

## Why it matters here

ConflictScope is a ready template for an eval that pits two dissertation principles against each other and measures which the SFT'd small model actually prioritises, with Bradley-Terry giving a principled value ordering. The multiple-choice-vs-open-ended divergence reinforces the project's substance-based (not hardcoded/MCQ) evaluation commitment — MCQ probes overstate protective behaviour. And the finding that a system prompt only recovers ~14% is a quantitative baseline the project's *SFT* approach should beat: if constitutional SFT moves value prioritisation more than a system prompt does, that is a headline result.

## Method

- Sample a value pair → generate a scenario where satisfying one violates the other (two-stage, Claude 3.5 Sonnet) → six-dimensional LLM filtering (realism, specificity, feasibility, impossibility, value-guidedness, genuine dilemma).
- Present in **multiple-choice** and **open-ended** modes with a simulated user; fit **Bradley-Terry** over response comparisons to get a value ranking. Optionally inject a target value ordering into the system prompt to test steerability.
- Generated sets: HHH 1,109 / ModelSpec 602 / Personal-Protective 1,187 scenarios; reported Pareto-optimal vs 7 prior dilemma datasets on agreement/strength.

## Key results

- **Format effect:** all models except Claude shift toward personal values in open-ended mode.
- **Steering ceiling:** detailed value orderings recover ~14% (avg normalised effect 0.145); highly model-dependent (Claude Haiku 0.01, OLMo-2-32B 0.27).
- Generated conflicts are harder/sharper than prior datasets (Pareto-optimal on agreement vs preference strength).

## Critical appraisal

The transferable result — a model that looks safe under multiple-choice can systematically deprioritise protective values when actually responding — is a direct warning to anyone validating a constitution via MCQ checks, and the pipeline is elegant and value-set-agnostic. Caveats: heavy reliance on Claude as generator/judge (Claude's apparent exceptionalism may be an in-group artefact); single-turn scope; Bradley-Terry compresses rich behaviour into a scalar order; "personal vs protective" is a coarse axis.

> ⚠ Small-model caution: steerability varies wildly by model, so a sub-1B student may be much harder to steer under conflict — making the conflict-resolution eval especially informative to run on the project's own model.

## Related

- [[sources/papers/c3ai]] — constitution design; framing effects on adherence
- [[sources/papers/hierarchical-safety-adherence]] — adherence under goal/principle conflict
- [[sources/papers/safety-tax]] — the protective-vs-capability trade-off
- [[entities/constitution]] — the 19 principles that can conflict
- [[decisions/2026-05-03-research-question-reframe]] — SFT over prompt-only steering
- [[experiments/human-evaluation-rubric]] — open-ended, substance-based evaluation

## Sources

- Liu, Ghate, Diab, Fried, Kasirzadeh, Kleiman-Weiner (2025) — arXiv:2509.25369 — [arxiv.org/abs/2509.25369](https://arxiv.org/abs/2509.25369)
