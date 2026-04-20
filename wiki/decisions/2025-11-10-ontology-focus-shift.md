---
title: Shift primary focus to Ontology-LLM integration
type: decision
tags: [direction, ontology, verification]
sources:
  - docs/Dissertation/Experimental Planning Document.md
updated: 2026-04-19
status: current
---

# Shift primary focus to Ontology-LLM integration

**Date:** 2025-11-10 (advisor meeting)

## Decision
Make **ontology-LLM integration** the primary experimental lens, demoting the earlier focus on pure process-reward RL to a secondary experiment. Pursue two complementary approaches in parallel:

- **Approach A** — ontology as core knowledge base; LLM as NL interface.
- **Approach B** — ontology as post-hoc verifier of LLM claims.

## Rationale
The advisor pressed for a **concrete, measurable** experimental basis over theoretical framing. Two structural problems motivate the shift:

1. Transformers cannot explain token generation from inside (tokens are IDs; output complexity is multi-parameter). Explainability has to come from an external structure.
2. Current industry practice (MCP / tools) **delegates** but does not **verify**. Ontology-based verification fills that gap.

## Consequences

- Reasoning / RL work becomes a supporting component, not the flagship.
- Need to choose an ontology (OWL / RDF) and a test domain. Political or geopolitical reasoning was suggested.
- New evaluation axes: verification accuracy, routing precision, cross-cultural consistency.

## Open items (to resolve before next advisor meeting)

- Which ontology + domain?
- Claim-extraction pipeline for Approach B.
- Baseline: MCP-only vs. MCP + ontology-verify.

## Related

- [[topics/ontology-integration]] — full topic treatment of A + B approaches
- [[topics/tool-use-and-verification]] · [[topics/reasoning]] · [[topics/explainability]]
- [[experiments/experiment-catalog]] — Experiment 6
- [[sources/dissertation/research-plan]] · [[sources/dissertation/experimental-planning-document]]
- [[overview]]

## Sources

- `docs/Dissertation/Experimental Planning Document.md` — §"Meeting Summary (November 10, 2025)" and §"Experiment 6"
