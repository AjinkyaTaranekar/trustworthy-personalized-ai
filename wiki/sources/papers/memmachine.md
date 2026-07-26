---
title: "MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents"
type: source
tags: [memory, personalisation, graph-memory, agents]
sources:
  - https://arxiv.org/abs/2604.04853
  - https://github.com/MemMachine/MemMachine
updated: 2026-07-18
status: current
---

# MemMachine: A Ground-Truth-Preserving Memory System

**A memory system for LLM agents that stores whole raw conversational episodes (rather than distilling each message through an LLM extractor) and layers short-term, long-term episodic, and profile memory plus contextualized retrieval — preserving factual "ground truth" while cutting input-token cost ~80% versus Mem0 and beating open memory baselines on LoCoMo.**

## Summary

Wang et al. (MemVerge, 2026) argue that context windows and naive RAG degrade across multi-session interactions, and that extraction-heavy systems (e.g. Mem0) run an LLM over every message — lossy (paraphrase corrupts the original), expensive, and hard to audit. MemMachine instead keeps raw turns verbatim and indexes them at the sentence level, reserving LLMs for summarisation, profile extraction, and agent inference only. Its contextualized retrieval (nucleus episode + neighbour expansion + cross-encoder rerank) reaches LoCoMo 0.9169 (gpt-4.1-mini) and LongMemEvalS 93.0%, at ~78% fewer input tokens than Mem0 (4.20M vs 19.21M). For this thesis it is a leading exemplar of the memory-systems line — and a pointed cloud/big-model contrast to the on-device target.

## Why it matters here

MemMachine's three layers (short-term / episodic / profile) map onto the dissertation's notion of persistent user memory, and its updatable profile memory is conceptually adjacent to a scrutable, editable user model. Its "ground-truth-preserving" design is itself a *faithfulness* argument — retaining the original turn to avoid extraction distortion is hallucination mitigation at the memory layer ([[sources/papers/hallucination-survey]]). But it is the cloud/frontier-model contrast case, not a solved on-device solution.

## Method

- **Short-term memory:** recent-episode window; overflow triggers LLM summaries migrated to LTM.
- **Long-term episodic memory:** four-stage indexing (NLTK sentence extraction → metadata → relational mapping → embeddings) on Postgres/pgvector + Neo4j.
- **Profile/semantic memory:** SQL-backed user preferences/demographics that support *updates* (correct a changed fact).
- **Contextualized retrieval:** find a nucleus episode by embedding similarity, expand to neighbours (1 before, 2 after), cross-encoder rerank, dedupe, chronological sort.
- **Retrieval agent (optional):** ToolSelect router over direct lookup / SplitQuery / ChainOfQuery, bounded to 3 iterations.

## Key results

- **LoCoMo:** 0.8747 (gpt-4o-mini) vs Memobase 0.7578, Mem0 0.6688; Agent mode 0.9169 (gpt-4.1-mini).
- **LongMemEvalS:** 0.930 (best config); knowledge-update 0.949, single-session user facts 1.000, multi-session 0.872 (weakest).
- **Tokens:** 4.20M vs Mem0 19.21M (~78% fewer). Retrieval-stage tuning dominates ingestion-stage; a smaller answer model (gpt-5-mini) beat gpt-5 by +2.6% with good retrieval.

## Critical appraisal

A defensible core idea (don't paraphrase away the ground truth), broad benchmark coverage, an honest cost-accuracy Pareto, and open source. Cautions: it is a vendor publishing its own product with self-reported, model-version-fragile numbers over mixed re-run/published baselines.

> ⚠ Conflict / tension for an on-device 0.6B thesis: every result uses **frontier hosted models** (gpt-4.1-mini/gpt-5-mini, OpenAI embeddings, Cohere reranker) — the "efficiency" is *token cost to a paid API*, not local compute/RAM, and there is no evidence it works under a sub-1B on-device budget. Cite it as the "big-model, cloud" contrast and motivation, not a solved on-device solution. Also: verbatim retention of all episodes is a **privacy liability** the paper frames as a virtue — exactly the retention/redaction tension a privacy-on-mobile framing must critique ([[sources/papers/what-should-llms-forget]]).

## Related

- [[entities/graph-rag]] — the project's KG-backed user memory; MemMachine is a leading comparator
- [[sources/papers/op-bench]] — over-personalisation from memory augmentation; the failure mode
- [[sources/papers/what-should-llms-forget]] — the retention/erasure critique of verbatim storage
- [[sources/papers/hallucination-survey]] — grounding in retrieved memory as hallucination mitigation
- [[topics/personalisation]] — memory-augmented personalisation
- [[topics/security-and-privacy]] — on-device vs server storage of user memory

## Sources

- Wang, Yu, Love, Zhang, Wong, Scargall, Fan (MemVerge, 2026) — arXiv:2604.04853 — [arxiv.org/abs/2604.04853](https://arxiv.org/abs/2604.04853)
- Code — [github.com/MemMachine/MemMachine](https://github.com/MemMachine/MemMachine)
