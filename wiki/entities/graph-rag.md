---
title: GraphRAG
type: entity
tags: [retrieval, graph, rag, personalisation, user-modelling]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - researchplan.tex
  - docs/Assets/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401v4).pdf
updated: 2026-04-19
status: current
---

# GraphRAG

**A variant of [[entities/rag|RAG]] where the non-parametric memory is a
structured knowledge graph (entities + typed relations) rather than a flat
dense-vector index. The retriever walks the graph; generation is
conditioned on the retrieved sub-graph, not on cosine-nearest chunks.**

## Why this thesis cares

In the proposed architecture, GraphRAG is the **User Modelling backend** —
the place where a per-user [[entities/5w-h|5W+H]] profile lives. The
research plan (`researchplan.tex` Phase 4) makes this explicit: "prepare a
Knowledge Graph, and eventually be used as GraphRAG." Three properties
matter:

- **Typed relations beat dense vectors** for reasoning about user context
  — who-did-what-with-whom is a graph query, not a similarity match.
- **Structured slots map to graph nodes** — the 5W+H schema gives clean
  node types (Person, Goal, Location, Time, Motivation, Method).
- **Privacy by locality** — the graph can live on the user's device
  behind a local [[entities/mcp|MCP]] server; nothing leaves.

## Backend candidates (decision pending)

| Candidate | Strengths | Filed under |
| --------- | --------- | ----------- |
| **Cognee** | LLM-native KG construction, Python-first | TODO |
| **FalkorDB** | Redis-backed, low-latency graph queries | TODO |
| **Neo4J** | Mature, Cypher query language, rich tooling | TODO |

See [[questions/2026-04-19-initial-questions]] for the decision item.

## Upstream literature

- [[sources/papers/rag-original]] — the parametric / non-parametric
  memory split that GraphRAG specialises.
- Think-on-Graph (Sun et al. 2024) — LLM reasoning over a knowledge
  graph. Cited in the research plan but **not yet in `docs/Assets/`**;
  listed as a gap in [[questions/2026-04-19-initial-questions]].

## Related

- [[topics/personalisation]] · [[topics/tool-use-and-verification]] · [[topics/ontology-integration]]
- [[entities/rag]] · [[entities/5w-h]] · [[entities/mcp]]
- [[sources/dissertation/research-plan]] — Phase 4 "User Modelling"
- [[sources/papers/rag-original]]

## Sources

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]]
- `researchplan.tex` — Phase 4
