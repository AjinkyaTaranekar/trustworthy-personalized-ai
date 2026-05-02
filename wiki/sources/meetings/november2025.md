---
title: Meeting Notes — November 2025
type: source
tags: [advisor-meeting, ontology, reasoning, scrutability, planning]
sources:
  - docs/meetings-notes/november2025.md
updated: 2026-05-02
status: current
---

# Meeting Notes — November 2025

**Two meetings in November: the first (Nov 11, in-person) pivoted the research toward ontology-LLM integration; the second (late November) introduced interleaved thinking as a scrutability mechanism.**

## Meeting 1 — November 11 (in-person, Owen's office)

### Summary

Ajinkya presented three approaches to LLM reasoning: brain-frequency-mode models (hierarchical), neurosymbolic AI, and the Qwen model. Owen pushed for a concrete, practically grounded research question rather than theoretical exploration. The key insight that emerged: LLMs cannot explain their token generation not just because of opacity but because transformers fundamentally lack understanding of token meaning — they see tokens as IDs. This reframe directed the discussion toward post-hoc verification: rather than making the LLM explain itself, use an external ontology to verify or rate LLM outputs. The meeting ended with an action for Ajinkya to write a brief experiment outline based on using an extant ontology as a knowledge base with the LLM as the interface.

### Key concepts

**Ontology as KB vs post-hoc verifier.** Owen proposed two roles for an ontology: (A) a primary knowledge base that the LLM queries as an intelligent interface, or (B) a post-hoc verifier that assesses LLM claims for accuracy after generation. This A/B distinction became the flagship experimental question, now represented in [[topics/ontology-integration]] and [[decisions/2025-11-10-ontology-focus-shift]].

**Post-hoc processes to verify LLM responses.** The core insight distinguishing this thesis from industry approaches: rather than delegating to external APIs at generation time (MCP server calls), use a verification layer after generation to ratify outputs. This contrasts with the "MCP as tool delegation" model and opens a research question about when each is appropriate.

**Ontologies from multiple global perspectives.** Owen suggested testing the ontology-LLM approach in politically sensitive areas where LLMs may have Western-centric biases — using ontologies built from different geopolitical perspectives as verifiers reveals hidden model biases.

### Action items

- Ajinkya: write a brief bullet-point experiment outline.
- Ajinkya: briefly document related research around the proposed experiment ideas.
- Next meeting: in person, November 11th at 10 AM (same day — this was the scheduling discussion).

---

## Meeting 2 — Late November (online)

### Summary

Ajinkya shared experience from a ClickHouse hiring assessment (building a Postgres AI extension) and introduced *interleaved thinking* — a mechanism from Apple (conceptualised May 2023) where thinking, questions, and answers are repeatedly fed back into the model in a circular loop. Owen responded by asking whether interleaved thinking could be implemented or simulated in a less performant base model (Llama, Minimax M2) to demonstrate scrutability improvements. The key reframe: rather than chasing the best LLM, show that a mundane LLM *with* interleaved thinking has better perceived scrutability than one without. They agreed on December 11 for an implementation plan meeting.

### Key concepts

**Interleaved thinking as scrutability mechanism.** The circular feedback loop (think → question → answer → repeat) produces a visible reasoning chain that users can inspect and contest. This is distinct from CoT prompting (which is one-pass) and closer to the iterative deliberation in [[sources/papers/interleaved-reasoning]].

**Pseudo-scrutability.** Owen raised the concept: a system that *appears* scrutable to users even if the internal process is not fully transparent. This is a pragmatic concession — full transparency may be architecturally impossible, so the research question is whether useful partial transparency is achievable and whether it changes user trust.

**Minimax M2 and Kimi K2.** Owen and Ajinkya identified these as potential open-source models for experimentation, noting their research repos as starting points. Neither was ultimately chosen — Qwen3-0.6B was selected later as the base model for small-model constraints.

### Action items

- Ajinkya: share interleaved thinking papers with Owen via chat.
- Ajinkya: research Minimax M2 and Kimi K2 repos.
- Ajinkya: create a plan for getting interleaved thinking operational.
- Ajinkya: investigate open-source graph representation tools for reasoning visualisation.
- Owen: send meeting transcript.
- Next meeting: December 11, noon, online.

## Related

- [[decisions/2025-11-10-ontology-focus-shift]] — this meeting produced the decision
- [[topics/ontology-integration]] — the flagship experimental lens seeded here
- [[topics/explainability]] — scrutability and pseudo-scrutability concepts raised
- [[sources/papers/interleaved-reasoning]] — the interleaved thinking paper discussed in meeting 2
- [[entities/qwen3-0.6b]] — the model that replaced Minimax M2 / Llama

## Sources

- `docs/meetings-notes/november2025.md`
