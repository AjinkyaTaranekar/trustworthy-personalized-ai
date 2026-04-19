---
title: "BERT: Pre-training of Deep Bidirectional Transformers"
type: source
arxiv_id: 1810.04805v2
authors: Devlin, Chang, Lee, Toutanova
year: 2018
venue: NAACL
tags: [foundation, transformers, bidirectional]
sources:
  - docs/Assets/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding (1810.04805v2).pdf
  - docs/Literature Notes/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding (1810.04805v2).md
updated: 2026-04-19
status: current
---

# BERT

**Bidirectional pre-training of Transformers via masked-language-modelling —
context both before and after each token, unlike GPT's causal mask.**

## What it does
Pre-trains an encoder on unlabelled text with two objectives (masked LM,
next-sentence prediction), producing contextual embeddings that fine-tune to
state-of-the-art on eleven NLP tasks with minimal task-specific architecture.

## Why it matters for this thesis
BERT establishes the critical distinction between **understanding**
(bidirectional) and **generation** (causal). The dissertation draft flags this
as an open design lever: could a hybrid — BERT-like understanding to classify
or verify, GPT-like generation to respond — close the "sociopath yapper" gap
in [[topics/reasoning]]? Approach B of the ontology-verifier direction
([[decisions/2025-11-10-ontology-focus-shift]]) is structurally similar:
generate with a causal LM, verify with a bidirectional check.

## Related

- [[topics/llm-foundations]]
- [[sources/papers/attention-is-all-you-need]]
- [[decisions/2025-11-10-ontology-focus-shift]]

## Sources

- `docs/Assets/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding (1810.04805v2).pdf`
- `docs/Literature Notes/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding (1810.04805v2).md`
