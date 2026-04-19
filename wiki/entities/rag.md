---
title: Retrieval-Augmented Generation (RAG)
type: entity
tags: [retrieval, rag, memory, knowledge]
sources:
  - docs/Assets/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).pdf
updated: 2026-04-19
status: current
---

# RAG

**Pattern where a model's generation is conditioned on retrieved
non-parametric memory (typically dense-vector chunks) at inference time,
rather than only on weights.**

## Why it matters for this thesis
RAG is the **root mechanism** that lets the architecture separate facts
(retrievable) from style (trained). Three uses in this thesis:
1. **Personalisation** — [[entities/graph-rag|GraphRAG]]-style retrieval
   over a per-user knowledge graph; base model stays frozen, user state
   lives externally → no catastrophic forgetting.
2. **Grounding** — provenance for factual claims (original paper showed
   RAG outputs are "more specific, diverse, and factual").
3. **Ontology verification** — Approach B in
   [[decisions/2025-11-10-ontology-focus-shift]] uses ontology retrieval to
   check LLM claims.

## Related

- [[topics/personalisation]] · [[topics/tool-use-and-verification]]
- [[sources/papers/rag-original]]
- [[sources/papers/search-r1]] — RL for retrieval
- [[entities/graph-rag]] _(to be filed when ingested)_

## Sources

- `docs/Assets/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).pdf`
