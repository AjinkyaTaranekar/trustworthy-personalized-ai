---
title: Model Context Protocol (MCP)
type: entity
tags: [tool-use, mcp, protocol, anthropic, security]
sources:
  - docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1).pdf
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/security-analysis/security-review.tex
updated: 2026-04-30
status: current
---

# Model Context Protocol (MCP)

**An open standard from Anthropic for connecting LLMs to external tools, resources, and data sources. "USB for AI" — write a server once, use with any MCP-compatible model.**

## Architecture
- **MCP Client** — the LLM application (Claude, custom chatbot, etc.).
- **MCP Server** — exposes tools / resources (file system, database, Python sandbox, calculator, ontology).
- **Transport** — stdio, HTTP, WebSocket.

## Why it matters for this thesis
MCP is the transport substrate for three pillars of the architecture:
- Tool use (`python_execute`, `web_search`, `read_url`, `get_datetime` in [[entities/constitution]]).
- [[topics/personalisation|Personalisation]] — local MCP servers keep user memory off cloud LLMs.
- Ontology verification ([[decisions/2025-11-10-ontology-focus-shift]]) — the ontology becomes another MCP server in Approach B.

Advantage over native function calling: **model-agnostic**. The same reasoner can swap between Python sandbox, GraphRAG memory, and ontology verifier without per-vendor glue.

## Security Risk: Prompt Injection via MCP Tool Outputs

MCP is both the thesis's privacy architecture (keeping user data local) and its primary prompt-injection surface (live web content enters the context via MCP tool calls). The **Log-To-Leak** attack (Hu et al. 2026, unacquired — see [[questions/2026-04-30-asset-acquisition-todo]]) demonstrates the specific risk: a malicious MCP server embeds instructions in its tool response, the model follows those instructions per Principle 10, and the attacker silently receives exfiltrated user queries through the same logging tool — with no degradation in task performance that would alert the user. This attack has no current runtime defence in the architecture. The required mitigation is a separate extraction layer that converts raw MCP tool output to structured data before the main model sees it.

## Related

- [[topics/tool-use-and-verification]] · [[topics/personalisation]]
- [[topics/security-and-privacy]] — MCP as both the privacy architecture and the injection vector
- [[sources/papers/mcp-multi-agent]]
- [[sources/papers/react]] · [[sources/papers/search-r1]]
- [[entities/constitution]] — Principle 10 interacts with MCP tool execution
- [[sources/dissertation/security-privacy-social-ethics]] — Log-To-Leak analysis

## Sources

- `docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1).pdf`
- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md` — §2 "MCP Servers"
- `docs/security-analysis/security-review.tex` — §4.1 Prompt Injection via Tool Outputs
