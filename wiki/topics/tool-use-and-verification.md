---
title: Tool Use and Verification
type: topic
tags: [tool-use, mcp, ontology, verification, security]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
  - docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1).pdf
  - docs/Assets/PAL Program-aided Language Models (2211.10435v2).pdf
  - docs/Assets/ReAct Synergizing Reasoning and Acting in Language Models (2210.03629v3).pdf
  - pipeline/constitution.md
  - docs/security-analysis/security-review.tex
updated: 2026-04-30
status: current
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

## Security Risk: Prompt Injection via Tool Outputs

Live web content retrieved by `read_url` and `web_search` is the primary prompt-injection surface (OWASP LLM01). **Principle 10** of the [[entities/constitution|constitution]] makes the model structurally disposed to follow tool-returned content — correct in normal operation, but an amplified attack surface when content is adversarially crafted. The **Log-To-Leak** attack (Hu et al. 2026, unacquired) shows a malicious MCP server can exfiltrate user queries through a logging tool with no task-performance degradation. No runtime defence currently exists; a separate extraction layer converting raw tool output to structured data is the required fix before any public deployment. See [[entities/mcp]] and [[topics/security-and-privacy]].

## Related

- [[topics/ontology-integration]] — the ontology-verifier direction is a specialisation of this topic
- [[topics/reasoning]] — delegation is a trust property of reasoning
- [[topics/personalisation]] — local MCP servers for private user memory
- [[topics/security-and-privacy]] — prompt injection is the primary security risk in this topic
- [[entities/constitution]] — codifies tool-use rules; Principle 10 interacts with injection risk
- [[entities/mcp]] — the injection vector + the local privacy architecture
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
- [[sources/dissertation/security-privacy-social-ethics]] — §4.1 prompt injection analysis
- [[sources/code/constitution-document]] · [[sources/code/sft-v2-pipeline]]
