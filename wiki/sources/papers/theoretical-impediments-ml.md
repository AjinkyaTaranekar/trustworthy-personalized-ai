---
title: "Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution"
type: source
tags: [foundations, psychology]
sources:
  - https://arxiv.org/abs/1801.04016
updated: 2026-07-22
status: current
---

# Theoretical Impediments to Machine Learning (Seven Sparks from the Causal Revolution)

**Current machine learning is stuck at the lowest rung of a three-level "Ladder of Causation" because it is model-blind, purely data-driven curve-fitting; genuine human-level reasoning requires causal models that let a system reason about interventions ("doing") and counterfactuals ("imagining"), not just associations ("seeing").**

## Summary

Pearl (UCLA, 2018) argues that optimising any function of the *observed* data distribution — however expressive — can never, in principle, answer questions about what *would* happen under intervention or *would have* happened under a different action, because those questions require information absent from the observational distribution. He formalises the **Ladder of Causation** — Level 1 Association `P(y|x)` ("what is?"), Level 2 Intervention `P(y|do(x))` ("what if I do X?"), Level 3 Counterfactuals `P(y_x|x',y')` ("why? what if I had acted differently?") — and shows the hierarchy is unidirectional: no amount of Level-1 data answers Level-2/3 questions without a causal model. The seven "sparks" (pillars) are capabilities the Structural Causal Model framework mechanises (confounding control via do-calculus, counterfactual algorithmisation, mediation, transportability, missing-data recovery, causal discovery). BACKGROUND/motivation for the trust-foundations framing — conceptual, not methodological, with no LLM or small-model angle.

## Why it matters here

The correlation-vs-causation distinction is the companion to [[sources/papers/talking-about-llms|Shanahan]] in the background chapter's philosophical scaffolding: Shanahan argues an LLM producing language does not thereby *mean* or *know*; Pearl argues a system fitting `P(y|x)` does not thereby *understand* or reason about interventions. Together they justify why fluent, empathetic-sounding small-model output must not be mistaken for genuine understanding or reliable reasoning — and why trust must be evaluated on substance. The Ladder is also a lens on the limits of LLM "reasoning": chain-of-thought and pattern completion operate at Level 1 (association over token distributions) with no guarantee of the causal/counterfactual reasoning humans use.

## Argument structure

- **Ladder of Causation:** Association (seeing) → Intervention (doing) → Counterfactuals (imagining); questions at level *i* need knowledge at level ≥ *i*.
- **Structural Causal Models:** graphical models (assumptions) + structural equations + counterfactual/interventional logic — an inference engine over (Assumptions, Queries, Data).
- **The seven sparks:** (1) transparent, testable causal assumptions (d-separation); (2) do-calculus + back-door confounding control; (3) algorithmisation of counterfactuals; (4) mediation (direct/indirect effects); (5) external validity / transportability / selection-bias correction; (6) causal missing-data recovery; (7) causal discovery.
- Closing device: Babylonian curve-fitters vs Greek model-builders — creativity comes from models, not fitting.

## Critical appraisal

A foundational framing that gave the field the ladder, `do(x)`, and the three-level vocabulary; its enduring value is conceptual clarity about *what kind* of question a purely associational system can and cannot answer. Its weakness is prescriptive thinness at scale — it diagnoses the limitation more convincingly than it operationalises the cure for high-dimensional perceptual learning, and the framework's power depends on *having the right causal graph* (substantive, possibly-wrong assumptions). Best read as the definitive statement of the *problem*, paired with later work attempting the synthesis.

> Note: no on-device/small-model angle and predates LLMs — background weight for the "seeing is not doing, correlation is not causation" argument, not a technique the pipeline implements.

## Related

- [[sources/papers/talking-about-llms]] — Shanahan; the first philosophical pillar (language ≠ mind)
- [[topics/llm-foundations]] — the limits of association-only learning
- [[topics/reasoning]] — CoT-as-Level-1-association; faithful reasoning as a higher rung
- [[sources/papers/hallucination-survey]] — factuality/faithfulness as a trust concern
- [[topics/ontology-integration]] — structured/causal knowledge as a complement to a statistical model
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — the background argument

## Sources

- Pearl (2018) — arXiv:1801.04016 (WSDM 2018 keynote) — [arxiv.org/abs/1801.04016](https://arxiv.org/abs/1801.04016)
