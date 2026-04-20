---
title: Personalisation
type: topic
tags: [personalisation, 5w-h, graph-rag, privacy]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - researchplan.tex
updated: 2026-04-19
status: stub
---

# Personalisation

**How to make responses personal to a user without baking user state into
model weights (which triggers catastrophic forgetting).**

## Summary
The thesis treats personalisation as an **external memory problem**, not a
fine-tuning problem. A **5W+H user model** (who, what, when, where, why, how)
captures structured user context; a [[entities/graph-rag|GraphRAG]] layer
retrieves it at inference time. Base model weights stay frozen → no
catastrophic forgetting, weights can be shared across users, and user data
stays locally controlled (privacy property).

## Key sub-ideas

- **5W+H schema** — structured slots for user context, updated per session.
- **GraphRAG memory** — retrieval over a user-specific graph instead of a
  vector store; keeps entity relations explicit.
- **Catastrophic-forgetting avoidance** — move user state out of weights and
  into retrievable memory; the base model is a frozen reasoner.
- **Privacy by architecture** — user-graph can live locally (MCP-style), never
  sent to cloud LLMs.

## Open questions

- How much personalisation is achievable from in-context retrieval vs.
  per-user LoRA adapters? (Trade-off: inference cost vs. adaptation depth.)
- How to keep the 5W+H graph fresh without the LLM silently editing user
  facts?

## Cold start — the thesis's answer
- Start with [[entities/5w-h|5W+H]] structured inquiry in the first
  conversation.
- Don't ask "what do you want?" — ask "what are you trying to achieve?
  Why?"
- Use few-shot: "Users like you typically prefer X" — with a visible
  source, not a silent demographic proxy.

## Privacy paradox
- Users want personalisation but fear surveillance.
- Three levers the thesis considers: **local-only storage** (user model
  on localhost MCP server), **federated learning** (patterns across
  users without centralising), **differential privacy** (noise on
  aggregate signals). Research question: how much personalisation can
  we achieve with privacy guarantees?

## Open vehicle: graph tools
Candidates for the [[entities/graph-rag|GraphRAG]] implementation:
**Cognee**, **FalkorDB**, **Neo4J**. Decision deferred — see
[[questions/2026-04-19-initial-questions]].

## Related

- [[topics/empathy]] — empathy grounds on user state; depends on this layer
- [[topics/tool-use-and-verification]] — memory lookup as a tool call
- [[topics/explainability]] — user-visible memory is part of scrutability
- [[entities/5w-h]] · [[entities/rag]] · [[entities/mcp]]
- [[entities/graph-rag]]

## Sources (ingested)

- [[sources/papers/rag-original]] — non-parametric memory foundation

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §3.2, §5.2
- `researchplan.tex`
