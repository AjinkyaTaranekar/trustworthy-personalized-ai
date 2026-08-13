---
title: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
type: source
tags: [memory, personalisation, graph-memory, agents]
sources:
  - https://arxiv.org/abs/2504.19413
updated: 2026-07-19
status: current
---

# Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

**A scalable, LLM-driven long-term memory layer that extracts, consolidates (ADD/UPDATE/DELETE/NOOP), and retrieves only salient facts from ongoing conversations — delivering near full-context answer quality at a fraction of the tokens and latency, with an optional graph variant (Mem0g) that adds relational/temporal structure.**

## Summary

Chhikara et al. (Mem0, 2025) address the finite-context problem for long multi-session dialogue: full-context prompting is high-quality but expensive (~26k tokens, ~17s p95), and RAG chunking caps ~61% quality and mishandles multi-hop/temporal questions. Mem0 distils conversations into a compact self-maintaining store via a two-phase pipeline — an LLM extracts candidate facts, then for each picks ADD/UPDATE/DELETE/NOOP against the top-10 similar existing memories. On LOCOMO it reaches 66.9 overall LLM-judge (vs full-context 72.9) at ~1,764 tokens (>90% savings) and ~1.44s p95 (~91% lower); the graph variant Mem0g leads temporal (58.13) and posts the highest overall J (68.44). This is the canonical memory extraction/update *contract* — and a cloud/frontier-model result that is a caution, not proof, for on-device.

## Why it matters here

The ADD/UPDATE/DELETE/NOOP operation set and the extract-then-consolidate loop are exactly the primitives a [[entities/5w-h|5W+H]] KG user-memory layer would implement, and Mem0g's entity+triplet+timestamp graph is a close cousin of a 5W+H personalisation graph ([[entities/graph-rag]]). Its bounded footprint (~1.8–3.6k tokens) and sub-2.6s p95 latency are the kind of budget a mobile deployment needs. It is the strong baseline a small-model memory manager must be compared against — and a likely upper bound.

## Method

- **Extraction:** an LLM reads a running summary + last 10 messages + the current pair, emitting candidate facts.
- **Update (Algorithm 1):** for each candidate, retrieve top-10 similar memories, then the LLM chooses ADD / UPDATE / DELETE / NOOP via function-calling (no separate classifier).
- **Mem0 (base):** dense NL facts in a vector DB (~7k tokens/conversation). **Mem0g:** labelled directed graph (entities + typed relationships + timestamps), Neo4j; conflicts marked *invalid* rather than hard-deleted (preserving history for temporal reasoning); ~14k tokens.
- All operations run on GPT-4o-mini + text-embedding-3-small.

## Key results (LOCOMO, LLM-judge J)

- **Mem0** wins single-hop (67.13) and multi-hop (51.15); **Mem0g** wins temporal (58.13) and highest overall J (68.44).
- **Cost:** Mem0 66.9 J at 1,764 tokens / 1.44s p95 vs full-context 72.9 J at ~26k tokens / 17.1s p95; best RAG ~61 J. Zep balloons to 600k+ tokens.

## Critical appraisal

Clean, reusable architecture and genuinely useful cost/latency numbers. Cautions: it is a **vendor-authored benchmark with LLM-judge as the headline metric** (F1 ~38 vs J ~67 signals a generous judge — treat absolute J as directional); write-time cost (an LLM call per extraction/update) is under-reported; single benchmark (LOCOMO, 10 conversations); baseline configs may not be tuned charitably.

> ⚠ Conflict / caution for a 0.6B thesis: *every* memory operation runs on GPT-4o-mini — there is no evidence a sub-1B model can reliably do the extraction, contradiction detection, and ADD/UPDATE/DELETE reasoning. That is exactly the open question this project's work must answer. DELETE/invalidation semantics do map onto privacy retention/forgetting requirements ([[sources/papers/what-should-llms-forget]]).

## Related

- [[sources/papers/memmachine]] — the ground-truth-preserving contrast (keep raw episodes vs extract facts)
- [[sources/papers/personalai]] — KG storage/retrieval sweep that *does* vary model scale
- [[sources/papers/graph-agent-memory-survey]] — the taxonomy that places Mem0/Mem0g
- [[entities/graph-rag]] — KG-backed user memory; Mem0g is a close cousin
- [[entities/5w-h]] — the who/what/when/where schema the ops would populate
- [[topics/personalisation]] — memory-augmented personalisation
- [[topics/security-and-privacy]] — DELETE/invalidation as retention/forgetting

## Sources

- Chhikara, Khant, Aryan, Singh, Yadav (2025) — arXiv:2504.19413 — [arxiv.org/abs/2504.19413](https://arxiv.org/abs/2504.19413)
