---
title: "State Stream Transformer (SST)"
type: source
arxiv_id: 2501.18356v1
authors: Aviss
year: 2025
tags: [reasoning, architecture, metacognition, state]
sources:
  - docs/Assets/State Stream Transformer (SST) Emergent Metacognitive Behaviours Through Latent State Persistence (2501.18356v1).pdf
  - docs/Literature Notes/State Stream Transformer (SST) Emergent Metacognitive Behaviours Through Latent State Persistence (2501.18356v1).md
updated: 2026-04-19
status: current
---

# SST — State Stream Transformer

**Adds a sliding-window FFN-cache with weighted decay that persists latent
state across autoregressive generations — using the *same* frozen weights,
elicits emergent metacognitive behaviour (89.01% GSM8K, 91.04% ARC).**

## What it does
Architectural modification only; no retraining. Removes the
across-generation discontinuity that autoregression imposes. Attributes
gains purely to the "state stream" via controlled ablations.

## Why it matters for this thesis
Direct evidence that **metacognition can be architectural**, not behavioural
— counterweight to the "sociopath yapper" framing that treats metacognition
as a training-time problem. Tension with the thesis's external-verification
approach: if metacognition is unlockable from frozen weights, the need for
an external ontology verifier weakens. Useful to cite as an alternative
hypothesis in the literature review section of
[[sources/dissertation/road-towards-trustworthy-empathetic-ai]].

## Related

- [[topics/reasoning]]
- [[sources/papers/hidden-reasoners]]
- [[sources/papers/looped-transformers-reasoning]]
- [[sources/papers/hierarchical-reasoning-model]]

## Sources

- `docs/Assets/State Stream Transformer (SST) Emergent Metacognitive Behaviours Through Latent State Persistence (2501.18356v1).pdf`
- `docs/Literature Notes/State Stream Transformer (SST) Emergent Metacognitive Behaviours Through Latent State Persistence (2501.18356v1).md`
