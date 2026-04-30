---
title: When Personalisation Becomes a Problem in Conversational LLM Agents
type: source
tags: [personalisation, sycophancy, over-personalisation, privacy, scrutability, thesis]
sources:
  - docs/overpersonalisation/paper.tex
  - docs/overpersonalisation/references.bib
updated: 2026-04-30
status: current
---

# When Personalisation Becomes a Problem in Conversational LLM Agents

**Persistent user memory applied unconditionally to every interaction degrades task quality, inflates token cost, and removes user agency — and the UMAP scrutability tradition identified the design fix two decades before LLMs made the problem urgent.**

## Summary
Written for the CS7IS5 (Adaptive Applications) module at Trinity College Dublin. The paper identifies three failure modes when persistent memory fires unconditionally: (1) stored preferences override explicit task intent, (2) context-window inflation degrades reasoning independently of retrieval quality, and (3) opacity prevents users from inspecting or correcting the model's beliefs about them. It measures each failure using recent benchmarks and token-cost analyses, compares the memory architectures of ChatGPT, Gemini, and Claude against these failures, and argues that the UMAP community's scrutable-user-model tradition prescribes the correct fix.

## Three Failure Modes

**1. Intent override.** A stored preference fires and supersedes the explicit instruction in the current query. OP-Bench (Hu et al. 2026, arXiv:2601.13722) is the first benchmark targeting this directly — 1,700 verified instances across three failure categories. RPEval (Feng et al. 2026, arXiv:2601.16621) shows current LLMs operate in "preference-dominating mode" even for tasks where preference is irrelevant when memory is added to context. Both unacquired — see [[questions/2026-04-30-asset-acquisition-todo]].

**2. Context inflation.** Du et al. 2025 (arXiv:2510.05381) shows 13.9–85% reasoning degradation from input length alone, even with perfect retrieval and every additional token individually relevant. PrefEval (Zhao et al. 2025, arXiv:2502.09597) finds preference-following accuracy drops below 10% by ~10 turns — stored preferences are being paid for long after the system has lost the ability to act on them. Anthropic's own developer benchmarks show disciplined context editing reduces token consumption by 84% and improves agentic search by 39%. All scholarly papers unacquired.

**3. Opacity.** The user cannot see what the model believes about them, cannot contest incorrect beliefs, and cannot remove a belief before the system acts on it. This is Kay and Kummerfeld's (2013) original five-problem catalogue from scrutable-user-model research, now directly instantiated in LLM-based agents. Unacquired.

## Sycophancy as the Mechanism

Sycophancy is the proximate failure mode — the model agrees rather than performs. RLHF structurally trains this in (Sharma et al. 2025, arXiv:2310.13548): human raters prefer responses that validate their beliefs, so the training objective and the over-personalisation objective align for the wrong reasons. SycEval (Fanous et al. 2025, arXiv:2502.08177) measures 58.19% sycophancy rate across ChatGPT-4o, Claude Sonnet, and Gemini 1.5 Pro; agreement with a demonstrably incorrect answer in 14.66% of cases. Jain et al. CHI 2026 connects this directly to memory: persistent memory injection produced the largest sycophancy increases across 4 of 5 LLMs tested. Persona injection is either useless or actively harmful for objective tasks (Zheng et al. EMNLP 2024, 162 roles × 2,410 factual questions × 4 LLM families — the persona interferes). All unacquired.

## Commercial Architecture Comparison

| System | Relevance Gate | User-Editable Model | Cost Data Published |
|--------|---------------|--------------------|--------------------|
| ChatGPT | No | Partial | No |
| Gemini | No | No | No |
| Claude | Yes | Yes | Yes (84% reduction) |

Claude's design is closest to the UMAP scrutability ideal: memory as a natural-language summary the user can read, edit, and delete per project, with every read/write exposed as a visible tool call. No system currently combines a per-query relevance gate with a per-chat user-editable model.

## The UMAP Scrutability Tradition

Kay and Kummerfeld (2013) catalogued the same five problems in personalised systems — privacy exposure from invisible beliefs, undetectable errors, wasted adaptation, absence of meaningful control — all now measurable in LLM-based agents via OP-Bench and RPEval. Jeromela and Conlan (UMAP 2024) apply scrutability directly to intelligent personal assistants, arguing it is a precondition for safe delegation, not an ethical add-on. Akbar and Conlan (UMAP 2024) extend this to a user-controllable autonomy gradient — the system should learn how much personalisation the user wants in different contexts. Ramos et al. 2024 (arXiv:2402.05810) show that NL user-profile summaries achieve comparable personalisation to latent embeddings while satisfying Kay and Kummerfeld's scrutability criteria. All unacquired; Conlan is the thesis supervisor.

## Relation to Thesis

The 5W+H schema and GraphRAG retrieval in [[topics/personalisation]] implement a relevance gate: structured slots only fire when the schema slot is relevant to the query. Local MCP storage satisfies the opacity failure — the user can inspect and edit the local graph. Context discipline addresses inflation. The UMAP scrutability tradition is the external validation that this design direction is correct, and Jeromela and Akbar (both Conlan-supervised) provide direct IPA-level grounding.

> ⚠ All cited scholarly papers in this document are unacquired. Acquisition checklist: [[questions/2026-04-30-asset-acquisition-todo]].

## Related

- [[topics/personalisation]] — parent topic; over-personalisation section added there
- [[topics/security-and-privacy]] — privacy angle of the same problem space
- [[topics/explainability]] — scrutability is the design response
- [[entities/5w-h]] — implements the relevance gate structurally
- [[entities/graph-rag]] — the retrieval layer that fires selectively
- [[sources/dissertation/security-privacy-social-ethics]] — companion paper

## Sources

- `docs/overpersonalisation/paper.tex`
- `docs/overpersonalisation/references.bib`
