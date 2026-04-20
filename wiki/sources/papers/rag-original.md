---
title: "Retrieval-Augmented Generation (original RAG paper)"
type: source
arxiv_id: 2005.11401v4
authors: Lewis et al. (FAIR)
year: 2020
venue: NeurIPS 2020
tags: [retrieval, rag, knowledge, foundations]
sources:
  - docs/Assets/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).pdf
  - docs/Literature Notes/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).md
updated: 2026-04-19
status: current
---

# RAG — the original

**Combines parametric memory (seq2seq weights) with non-parametric memory
(dense Wikipedia index + neural retriever) end-to-end, giving generations
provenance + updatable knowledge.**

## What it does
Two RAG formulations (same-passages vs per-token-passages). Sets SOTA on
three open-domain QA tasks. Produces "more specific, diverse, and factual"
language than parametric-only baselines.

## Why it matters for this thesis
The foundational paper for every retrieval-based approach in the pipeline,
including the planned [[entities/graph-rag|GraphRAG]] user model. More
importantly, RAG's key insight — **don't bake facts into weights, retrieve
them** — is the core mechanism the thesis uses to dodge catastrophic
forgetting in [[topics/personalisation]]. Provenance is a first-class
design goal here, directly feeding the scrutability story.

## Related

- [[topics/personalisation]] · [[topics/tool-use-and-verification]]
- [[entities/rag]] · [[entities/graph-rag]]
- [[sources/papers/search-r1]]

## Sources

- `docs/Assets/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).pdf`
- `docs/Literature Notes/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).md`
