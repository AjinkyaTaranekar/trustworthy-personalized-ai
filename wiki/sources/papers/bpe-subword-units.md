---
title: Neural Machine Translation of Rare Words with Subword Units (BPE)
type: source
arxiv_id: 1508.07909v5
authors: Sennrich, Haddow, Birch
year: 2015
venue: ACL 2016
tags: [foundation, tokenization, bpe]
sources:
  - docs/Assets/Neural Machine Translation of Rare Words with Subword Units (1508.07909v5).pdf
  - docs/Literature Notes/Neural Machine Translation of Rare Words with Subword Units (1508.07909v5).md
updated: 2026-04-19
status: current
---

# BPE — Subword Units

**Applies byte-pair encoding to NMT so rare words are represented as
sequences of subword units, eliminating the closed-vocabulary problem.**

## What it does
Treats translation as an open-vocabulary problem by encoding words as
compositions of frequent subword fragments. Improves WMT EN–DE / EN–RU by
~1.1–1.3 BLEU vs a dictionary back-off baseline.

## Why it matters for this thesis
BPE is the **technical root** of LLM arithmetic failure as framed in the
dissertation: numbers like `183491` get split into `["183", "491"]`, stripping
the digit-place structure that makes arithmetic compositional. This is not a
training-data issue — it is baked into the tokeniser. The thesis uses this to
justify treating computation as an external tool call
([[sources/papers/pal]]) rather than expecting the LLM to calculate. A
cornerstone fact for the "capability-honesty" principle in
[[entities/constitution]].

## Related

- [[topics/llm-foundations]]
- [[topics/tool-use-and-verification]]
- [[sources/papers/pal]]
- [[entities/constitution]]

## Sources

- `docs/Assets/Neural Machine Translation of Rare Words with Subword Units (1508.07909v5).pdf`
- `docs/Literature Notes/Neural Machine Translation of Rare Words with Subword Units (1508.07909v5).md`
