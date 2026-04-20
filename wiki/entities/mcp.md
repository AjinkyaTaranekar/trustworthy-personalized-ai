---
title: Model Context Protocol (MCP)
type: entity
tags: [tool-use, mcp, protocol, anthropic]
sources:
  - docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1).pdf
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
updated: 2026-04-19
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

## Related

- [[topics/tool-use-and-verification]] · [[topics/personalisation]]
- [[sources/papers/mcp-multi-agent]]
- [[sources/papers/react]] · [[sources/papers/search-r1]]
- [[entities/constitution]]

## Sources

- `docs/Assets/Advancing Multi-Agent Systems Through Model Context Protocol Architecture, Implementation, and Applications (2504.21030v1).pdf`
- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md` — §2 "MCP Servers"
