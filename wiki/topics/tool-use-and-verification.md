---
title: Tool Use and Verification
type: topic
tags: [tool-use, mcp, ontology, verification]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
  - docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol (2504.21030v1).pdf
  - docs/Assets/PAL Program-aided Language Models (2211.10435v2).pdf
  - docs/Assets/ReAct Synergizing Reasoning and Acting (2210.03629v3).pdf
  - pipeline/constitution.md
updated: 2026-04-19
status: stub
---

# Tool Use and Verification

**Delegate what the LLM cannot do reliably (computation, current facts, logical verification). Verify what it can do, before presenting the answer.**

## Summary
The thesis treats the LLM as a **reasoner and interface**, not a solver. Arithmetic, currency, time, entity facts → [[entities/mcp|MCP]] tools (`python_execute`, `web_search`, `read_url`, `get_datetime`). The output of those tools — and the LLM's own claims — are then passed to a **verification layer**. After the 2025-11-10 direction shift, that verification layer is the centrepiece: an ontology-based check for logical and factual consistency (see [[decisions/2025-11-10-ontology-focus-shift]]).

## Two verification modes

- **Approach A — ontology as knowledge base.** LLM acts as NL interface; ontology is authoritative for factual/logical queries.
- **Approach B — ontology as post-hoc verifier.** LLM generates first; claims extracted and checked against ontology; flagged / corrected / regenerated.

## Key sub-ideas

- **MCP as "USB for AI"** — one protocol for tools, any compatible model.
- **Tool discipline reward** — behavioural reward signal in RL that checks whether the model delegated when appropriate (per the constitution).
- **Scrutability** — every tool call logged and human-inspectable.
- **Privacy** — sensitive tools can stay local.

## Open questions

- Which ontology? OWL/RDF of what domain? (Political/geopolitical was proposed but scope is huge.)
- Claim-extraction accuracy from free-form LLM responses determines the verifier's ceiling.
- Latency budget for post-hoc verification in interactive use.

## Related

- [[topics/ontology-integration]] — the ontology-verifier direction is a specialisation of this topic
- [[topics/reasoning]] — delegation is a trust property of reasoning
- [[topics/personalisation]] — local MCP servers for private user memory
- [[entities/constitution]] — codifies tool-use rules
- [[entities/mcp]]
- [[decisions/2025-11-10-ontology-focus-shift]]

## Sources (ingested)

- [[sources/papers/pal]] — code-as-reasoning delegation
- [[sources/papers/react]] — interleaved reason-act loop
- [[sources/papers/mcp-multi-agent]] — MCP as the coordination protocol
- [[sources/papers/search-r1]] — RL-trained tool-use for search
- [[sources/papers/ui-r1]] — rule-based RL for GUI-agent action choice
- [[sources/papers/vlm-r1]] — rule-based RL for vision-language tasks (reward-hacking cautions)
- [[sources/papers/rag-original]] — non-parametric retrieval as a "memory tool"

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §2 "MCP Servers"
- [[sources/dissertation/experimental-planning-document]] — Experiment 6
- [[sources/code/constitution-document]] · [[sources/code/sft-v2-pipeline]]
