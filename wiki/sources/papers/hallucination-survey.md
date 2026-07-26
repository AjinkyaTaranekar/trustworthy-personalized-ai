---
title: "Survey of Hallucination in Natural Language Generation"
type: source
tags: [evaluation, hallucination]
sources:
  - https://arxiv.org/abs/2202.03629
updated: 2026-07-18
status: current
---

# Survey of Hallucination in Natural Language Generation

**The canonical taxonomy and framework for hallucination in NLG: it defines hallucination as output unfaithful to source, splits it into intrinsic (contradicts source) vs extrinsic (unverifiable against source), catalogues causes, measurement metrics, and mitigations, and insists on the faithfulness-vs-factuality distinction.**

## Summary

Ji et al. (CAiRE / HKUST, ACM Computing Surveys 2022; LLM section added 2024) unify the fragmented hallucination literature under one taxonomy. The load-bearing move is separating **faithfulness** (consistency with the provided source) from **factuality** (consistency with world knowledge), and splitting hallucination into **intrinsic** (contradicts the source — e.g. wrong vaccine-approval year) and **extrinsic** (ungrounded, unverifiable against the source, though possibly world-true). It maps causes (data misalignment, innate divergence, exposure bias, parametric-knowledge bias, decoding), metrics (statistical: PARENT, Knowledge-F1; model-based: NLI/QA/IE, FactCC, FEQA/QAGS), and mitigations (data cleaning/augmentation; architecture, training, generate-then-refine). Cited statistics — 25% of summaries hallucinate, 62% of WIKIBIO first sentences, 74% of QMSum samples inconsistent — show hallucination is pervasive. This is the backbone reference for the thesis's trustworthiness chapter.

## Why it matters here

It gives the dissertation a precise vocabulary for a personalised on-device model's errors: an **intrinsic** hallucination contradicts the user's stored profile/memory; an **extrinsic** one asserts a preference the profile neither supports nor contradicts (inventing a user fact). The faithfulness-vs-factuality split is directly usable — "faithful to the user's memory" and "factual about the world" are separate axes a trustworthy personalised agent must satisfy at once. Its NLI/QA-based faithfulness metrics inform the project's substance-based, judge-driven evaluation.

## Framework

- **Definitions:** hallucination = ungrounded/unfaithful output; faithfulness = source-consistency; factuality = world-truth (distinct).
- **Taxonomy:** intrinsic (contradicts source) vs extrinsic (unverifiable against source; extrinsic ≠ false).
- **Causes:** data-related (heuristic collection → misaligned pairs; innate divergence in open-ended tasks) and training/inference (imperfect encoding; erroneous decoding; exposure bias; parametric-knowledge bias).
- **Metrics:** statistical (PARENT/PARENT-T, Knowledge-F1, BVSS); model-based (IE-, QA- (FEQA/QAGS/QuestEval), NLI-, classifier- (FactCC), LM-based); human eval as gold standard.
- **Mitigations:** data (faithful datasets, cleaning, information augmentation) and modelling/inference (encoder/attention/decoder changes, RL, controllable generation, generate-then-refine).
- **Task-by-task:** summarisation, dialogue, task-oriented dialogue, generative QA, data-to-text, MT, vision-language, and LLMs.

## Key observations

- Faithfulness and factuality must be measured **separately**; source-only metrics can penalise correct extrinsic additions.
- Hallucination is often baked in at the **data** stage, so modelling fixes alone are insufficient.
- Higher decoding diversity/temperature trades off against faithfulness.
- No single automatic metric suffices; metric choice is task-dependent.

## Critical appraisal

The field's canonical reference, with a clean, widely-adopted taxonomy and unusually complete coverage of causes/metrics/mitigations. Cautions: as a 2022 survey (LLM section bolted on in 2024) it predates the retrieval-augmentation, self-consistency and LLM-as-judge era that now dominates hallucination detection/mitigation, so its metric/mitigation catalogue is foundational but not current SOTA; it is descriptive, not prescriptive (maps the space, does not rank methods).

> ⚠ For the on-device 0.6B thesis: hallucination is pervasive and partly data/scale-driven, which raises the bar for a small model — the causes framework flags that a 0.6B model is *more* prone to parametric-knowledge bias and decoding errors, motivating grounding in retrieved memory (ties to [[sources/papers/memmachine|MemMachine's]] ground-truth preservation) and conservative decoding rather than assuming a small personalised model is safe.

## Related

- [[sources/papers/memmachine]] — ground-truth-preserving memory as a hallucination mitigation
- [[sources/papers/transparent-scrutable-recs]] — fabricated profile facts as an extrinsic hallucination
- [[experiments/human-evaluation-rubric]] — human eval as the faithfulness gold standard
- [[topics/explainability]] — honest, grounded, citable outputs
- [[topics/personalisation]] — faithfulness to the user model as a trust axis
- [[entities/graph-rag]] — retrieval grounding as mitigation

## Sources

- Ji, Lee, Frieske, Yu, Su, Xu, Ishii, Bang, Chen, Dai, Chan, Madotto, Fung (2022/2024) — arXiv:2202.03629 (ACM CSUR) — [arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629)
