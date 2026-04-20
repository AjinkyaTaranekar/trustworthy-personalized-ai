---
title: "Coconut: Training LLMs to Reason in a Continuous Latent Space"
type: source
arxiv_id: 2412.06769v3
authors: Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Meta)
year: 2024
venue: COLM 2025
tags: [reasoning, latent, continuous, search]
sources:
  - docs/Assets/Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v3).pdf
  - docs/Literature Notes/Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v3).md
updated: 2026-04-19
status: current
---

# Coconut — Continuous Latent Reasoning

**Feeds the last hidden state back as next input embedding — "continuous thought" — instead of decoding to words. Each latent thought encodes multiple alternative next steps, enabling implicit breadth-first search instead of committing prematurely.**

## What it does
Replaces the CoT text interface with a continuous-vector reasoning loop. Outperforms CoT on logical-reasoning tasks that reward planning search.

## Why it matters for this thesis
A direct rebuttal to one core thesis claim — that **textual reasoning is scrutable** and therefore more trustworthy. Coconut trades scrutability for performance. Worth citing as the "alternative trade-off" the thesis argues against: if we accept latent reasoning, we lose the auditability story in [[topics/reasoning]]. File as an explicit tension in [[decisions/2025-11-10-ontology-focus-shift]] — ontology verification **requires** textual claims to extract, making Coconut-style representations a poor fit for the chosen direction.

## Related

- [[topics/reasoning]]
- [[sources/papers/looped-transformers-reasoning]]
- [[sources/papers/ladir]]
- [[sources/papers/hidden-reasoners]]

## Sources

- `docs/Assets/Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v3).pdf`
- `docs/Literature Notes/Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v3).md`
