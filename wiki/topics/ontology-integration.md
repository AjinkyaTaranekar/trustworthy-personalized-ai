---
title: Ontology–LLM Integration
type: topic
tags: [ontology, verification, neuro-symbolic]
sources:
  - docs/Dissertation/Experimental Planning Document.md
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - researchplan.tex
updated: 2026-04-19
status: current
---

# Ontology–LLM Integration

**The flagship experimental direction after the 2025-11-10 advisor
meeting. Treat an ontology as a structured, verifiable reasoning partner
to the LLM — either as the authoritative knowledge base (Approach A) or
as a post-hoc verifier of LLM claims (Approach B).**

## Why this became the flagship

The advisor's argument: current industry practice (MCP + tools)
**delegates** computation but does not **verify** reasoning. Ontologies
encode typed entities, relations, and inference rules — exactly the
substrate you need to check a claim like "X causes Y" or "A is a member
of B". This closes the scrutability gap in [[topics/explainability]] and
answers the "sociopath yapper" critique in [[topics/reasoning]] from a
*structural* angle rather than a training-signal angle.

## The two approaches

### Approach A — ontology as core knowledge base
- **LLM role:** NL interface. Translate user query → ontology query
  (SPARQL / Cypher / custom); format ontology answer → natural language.
- **Routing module:** classify incoming questions as factual/logical
  (→ ontology) vs subjective/creative (→ LLM generation) vs hybrid.
- **Explainability:** the answer is traceable to concepts and relations
  inside the ontology — "based on concept X, relation Y…".

### Approach B — ontology as post-hoc verifier
- **LLM generates first;** claim-extraction pipeline pulls factual
  assertions from the response.
- **Ontology checks** each claim for support + logical consistency.
- **Output:** verified, corrected, flagged, or rejected. Rejected
  responses can be regenerated with the contradiction fed back.

## Open design decisions (all in [[questions/2026-04-19-initial-questions]])

- **Ontology choice** — DBpedia, Wikidata, or domain-specific (political,
  medical, legal)?
- **Query language** — SPARQL, Cypher, or a custom NL→query model?
- **Inference engine** — OWL reasoner, rule-based, or neural
  approximation?
- **Claim extraction** (Approach B) — NER + relation extraction? Reuse
  existing fact-check models?
- **Evaluation ground truth** — for political questions: who defines
  "correct"? Test cross-ontology consistency?
- **Latency budget** — ontology inference is expensive; live chat needs
  <1–2 s.

## Design tensions

- **Scrutability vs performance.** Latent-reasoning architectures
  ([[sources/papers/coconut-continuous-latent]],
  [[sources/papers/ladir]], [[sources/papers/hierarchical-reasoning-model]])
  outperform on reasoning benchmarks but kill the ontology-verifier
  story — there's no textual claim to extract.
- **Every ontology encodes a worldview.** Presenting Western vs Eastern
  political ontologies side-by-side may make bias visible rather than
  eliminating it.
- **Scope.** Ontologies are narrow by construction; out-of-ontology
  questions need a graceful fallback.

## Adjacent: Think-on-Graph
The research plan cites **Sun 2024 — Think-on-Graph** (LLM reasoning
over a knowledge graph) as a core precedent. Not yet in `docs/Assets/`;
ingest when obtained. This paper and the upcoming
[[entities/graph-rag|GraphRAG]] work share substrate: both are LLMs
coordinating with typed relations, just at different stages of the
pipeline.

## Related

- [[decisions/2025-11-10-ontology-focus-shift]] — the commit to this direction
- [[experiments/experiment-catalog]] — Experiment 6
- [[topics/tool-use-and-verification]] — ontology is another MCP-exposable tool
- [[topics/explainability]] — ontology trace = scrutability route 4
- [[topics/reasoning]] — sibling: structural answer to the sociopath-yapper problem
- [[entities/rag]] · [[entities/graph-rag]] · [[entities/mcp]]

## Sources

- [[sources/dissertation/experimental-planning-document]]
- [[sources/dissertation/research-plan]]
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]]
- [[sources/papers/rag-original]]
