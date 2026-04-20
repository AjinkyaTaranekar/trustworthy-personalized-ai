---
title: Attention Is All You Need
type: source
arxiv_id: 1706.03762v7
authors: Vaswani et al.
year: 2017
venue: NeurIPS
tags: [foundations, transformers, attention]
sources:
  - docs/Assets/Attention Is All You Need (1706.03762v7).pdf
  - docs/Literature Notes/Attention Is All You Need (1706.03762v7).md
updated: 2026-04-19
status: current
---

# Attention Is All You Need

**Introduces the Transformer — a sequence model built entirely from self-attention, dispensing with recurrence and convolution.**

## What it does
Proposes a purely attention-based encoder–decoder that parallelises cleanly across a sequence. Establishes state-of-the-art on WMT 2014 translation with a fraction of prior training cost. This architecture is the substrate every modern LLM sits on.

## Why it matters for this thesis
Self-attention **is** the explainability bottleneck this thesis works around. The standard autoregressive decoding process is feed-forward with no stateful backtracking — a wrong early token corrupts the rest. This is the architectural reason post-hoc rationalisation happens: the model cannot revisit intermediate steps, so any "explanation" it emits is newly generated, not retrieved. Every later paper ingested here either (a) accepts the architecture and adds scaffolding on top, or (b) proposes deliberation mechanisms to recover what autoregression lost.

## Related

- [[topics/llm-foundations]]
- [[sources/papers/bert]] — the bidirectional counterpart
- [[sources/papers/tree-of-thoughts]] — recovers backtracking externally

## Sources

- `docs/Assets/Attention Is All You Need (1706.03762v7).pdf`
- `docs/Literature Notes/Attention Is All You Need (1706.03762v7).md`
