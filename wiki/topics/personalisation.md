---
title: Personalisation
type: topic
tags: [personalisation, 5w-h, graph-rag, privacy, over-personalisation, sycophancy, scrutability]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - researchplan.tex
  - docs/overpersonalisation/paper.tex
updated: 2026-04-30
status: current
---

# Personalisation

**How to make responses personal to a user without baking user state into model weights (which triggers catastrophic forgetting).**

## Summary
The thesis treats personalisation as an **external memory problem**, not a fine-tuning problem. A **5W+H user model** (who, what, when, where, why, how) captures structured user context; a [[entities/graph-rag|GraphRAG]] layer retrieves it at inference time. Base model weights stay frozen → no catastrophic forgetting, weights can be shared across users, and user data stays locally controlled (privacy property).

## Key sub-ideas

- **5W+H schema** — structured slots for user context, updated per session.
- **GraphRAG memory** — retrieval over a user-specific graph instead of a vector store; keeps entity relations explicit.
- **Catastrophic-forgetting avoidance** — move user state out of weights and into retrievable memory; the base model is a frozen reasoner.
- **Privacy by architecture** — user-graph can live locally (MCP-style), never sent to cloud LLMs.

## Open questions

- How much personalisation is achievable from in-context retrieval vs. per-user LoRA adapters? (Trade-off: inference cost vs. adaptation depth.)
- How to keep the 5W+H graph fresh without the LLM silently editing user facts?

## Cold start — the thesis's answer
- Start with [[entities/5w-h|5W+H]] structured inquiry in the first conversation.
- Don't ask "what do you want?" — ask "what are you trying to achieve? Why?"
- Use few-shot: "Users like you typically prefer X" — with a visible source, not a silent demographic proxy.

## Privacy paradox
- Users want personalisation but fear surveillance.
- Three levers the thesis considers: **local-only storage** (user model on localhost MCP server), **federated learning** (patterns across users without centralising), **differential privacy** (noise on aggregate signals). Research question: how much personalisation can we achieve with privacy guarantees?

## Over-Personalisation: When It Goes Wrong

The thesis's own CS7IS5 paper (see [[sources/dissertation/overpersonalisation-paper]]) documents three failure modes that emerge when personalisation fires unconditionally on every query: stored preferences override explicit task intent, context-window inflation degrades reasoning (13.9–85% performance loss from input length alone, Du et al. 2025), and opacity prevents users from correcting the model's beliefs about them. These are not hypothetical — OP-Bench (Hu et al. 2026) provides 1,700 verified instances; SycEval finds 58.19% sycophancy rates with 14.66% incorrect-answer agreement rates. The 5W+H + GraphRAG design is the architectural response: retrieval fires only when a schema slot is relevant to the query, implementing the per-query relevance gate that commercial systems currently lack.

## Sycophancy as the Over-Personalisation Mechanism

Sycophancy — the model agreeing rather than performing — is the proximate failure. RLHF structurally trains it in because human raters prefer validating responses, so the training and over-personalisation objectives align for the wrong reasons. Memory injection worsens this: Jain et al. CHI 2026 shows persistent user-profile injection produces the largest sycophancy increases across 4 of 5 LLMs tested. Persona injection (Zheng et al. EMNLP 2024) is useless or actively harmful for objective tasks — the persona interferes rather than sitting quietly. All these papers are unacquired; see [[questions/2026-04-30-asset-acquisition-todo]].

## Scrutability as the Design Response

The UMAP community's 20-year tradition of scrutable user models provides the design response: the user must be able to inspect, contest, and correct the model's beliefs about them before the system acts on those beliefs. Kay and Kummerfeld (2013) catalogued the same five problems now measurable in LLM agents. Jeromela and Conlan (UMAP 2024) argue scrutability is a precondition for safe delegation, not an ethical add-on. Akbar and Conlan (UMAP 2024) extend this to a user-controllable autonomy gradient — the system should learn how much personalisation the user wants in each context. Both Conlan-supervised. Ramos et al. 2024 show NL user-profile summaries achieve comparable personalisation to latent embeddings while satisfying scrutability criteria — which is precisely how local GraphRAG memory is designed to work.

## Graph backend decision: Cognee + FalkorDB

The three-way candidate decision (Cognee / FalkorDB / Neo4j) is now resolved. **FalkorDB** is the graph backend: it is 500× faster at p99 latency on neighbourhood expansion (the retrieval operation), uses Sparse Matrix Algebra rather than node-crawling, and has native LangChain/LlamaIndex integration. **Cognee** is the orchestration layer on top: it provides LLM-native KG construction, a unified memory abstraction that can switch backends without a code rewrite, and direct LlamaIndex integration for GraphRAG pipelines. The combination (Cognee as the write/construct layer, FalkorDB as the query layer) matches the MCP-local privacy requirement — the entire graph can live on-device behind a local MCP server. Neo4j is deferred for enterprise/multi-user scenarios where transactional guarantees matter more than inference latency.

## User state update: Mem0g pattern (write pipeline only)

The write pipeline for keeping the 5W+H graph current follows the Mem0g architecture (arXiv:2504.19413, production system achieving 91% lower p95 latency vs full-context injection): (1) entity extractor — LLM call extracting typed 5W+H entities from the user turn; (2) relation generator — LLM call inferring typed edges given the entities and existing graph context; (3) conflict detector — rule-based check for contradictions with existing edges; (4) conditional write — only after conflict resolution, using `:DEPRECATED_BY` edges rather than deletion so the audit trail remains intact. This four-stage pipeline is the basis for [[queries/grpo-and-personalisation-master-plan]] Layer 2. **Important:** Mem0's conflict resolution is entirely silent — the LLM Update Resolver decides ADD/UPDATE/DELETE/NOOP with no user notification. The thesis diverges from Mem0 precisely here: contradictions must be surfaced to the user before resolution, not resolved silently.

## The Scrutability Gap — the thesis's named contribution

No current production memory system provides what the UMAP tradition calls scrutability in the full sense. Mem0 is a pure developer API — users cannot inspect, correct, or contest any memory. ChatGPT's "Manage Memories" UI covers only explicitly saved memories; auto-learned chat-history memories are behind an all-or-nothing toggle, not individually auditable. Letta (the most transparent framework) allows agents to edit white-box memory blocks, but is a developer tool. The research literature (2024–2026) documents a "transparency asymmetry" (2025 AI Agent Index, arXiv:2602.17753) but has no formal definition, no design patterns, and no user studies on user-centric memory scrutability as a distinct concept from developer-API transparency. The thesis can claim the first formal definition of user-centric AI memory scrutability for conversational agents, operationalised as five constraints: (1) inspect — the user can read all beliefs the system holds; (2) contest — the user can flag a belief as wrong before the system acts on it; (3) correct — the user can provide the accurate belief; (4) deprecate (not delete) — corrected beliefs are marked `:USER_CORRECTED` and de-prioritised but kept for audit; (5) audit trail — the history of what changed, when, and why is always readable. This is Layer 5 of [[queries/grpo-and-personalisation-master-plan]] and the design response to the gap documented across Mem0, ChatGPT, and Gemini.

## Related

- [[topics/empathy]] — empathy grounds on user state; depends on this layer
- [[topics/tool-use-and-verification]] — memory lookup as a tool call
- [[topics/explainability]] — user-visible memory is part of scrutability
- [[topics/security-and-privacy]] — privacy-by-architecture and the GDPR angle
- [[entities/5w-h]] · [[entities/rag]] · [[entities/mcp]]
- [[entities/graph-rag]]

## Sources (ingested)

- [[sources/papers/rag-original]] — non-parametric memory foundation
- [[sources/dissertation/overpersonalisation-paper]] — over-personalisation failure modes + UMAP scrutability tradition

## Sources (to acquire — see [[questions/2026-04-30-asset-acquisition-todo]])

- Mem0 (arXiv:2504.19413) — production graph memory; entity extractor + conflict detector write pipeline; 91% latency reduction
- PersonalAI (arXiv:2506.17001) — flexible KG for personalised LLM agents; hybrid graph with hyper-edges for 5W+H
- Avoiding Over-Personalisation (arXiv:2509.07133) — rule-guided KG adaptation; validates the per-query relevance gate (Layer 3)
- Graph-based Agent Memory survey (arXiv:2602.05665) — taxonomy placing this architecture in the literature

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §3.2, §5.2
- `researchplan.tex`
- `docs/overpersonalisation/paper.tex`
