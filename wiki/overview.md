---
title: Thesis Overview
type: meta
tags: [thesis, synthesis]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
  - researchplan.tex
  - README.md
updated: 2026-04-19
status: current
---

# Thesis Overview

**Can we architect a conversational LLM that is trustworthy in its reasoning,
accurate through appropriate task delegation, privacy-preserving in its
personalisation, and capable of genuine contextual empathy through systematic
user modelling?**

## Central hypothesis
Monolithic LLMs cannot simultaneously be empathetic, logically rigorous, and
personalised without catastrophic forgetting and hallucination. A **modular
architecture** — reasoning engine, user model, empathy layer, verification
layer — is required.

## Five pillars

| # | Pillar | Core claim |
| - | ------ | ---------- |
| 0 | [[topics/llm-foundations]] | Tokenisation, causal attention, contextualised embeddings — the architectural reasons monolithic LLMs fail at arithmetic, backtracking, and auditable explanation |
| 1 | [[topics/reasoning]] | Trustworthy reasoning comes from process rewards, honest refusal, and tool delegation — not from longer prose |
| 2 | [[topics/personalisation]] | Keep user state out of weights; 5W+H + GraphRAG retrieval preserves personalisation and privacy |
| 3 | [[topics/empathy]] | Appraisal-theoretic conditioning gives empathy an auditable signal |
| 4 | [[topics/tool-use-and-verification]] | MCP as the tool substrate; ontology verification of LLM claims |
| 5 | [[topics/explainability]] | Scrutability via citations, honest tool reports, translated latent state — not self-rationalisation |
| ★ | [[topics/ontology-integration]] | Cross-cutting topic — the flagship experimental lens that binds pillars 1, 4, 5 together |

## Current direction shift
On 2025-11-10 the advisor reframed the project around **ontology-LLM
integration** as the flagship experiment. Pure process-reward RL becomes
supporting infrastructure. See
[[decisions/2025-11-10-ontology-focus-shift]] and
[[experiments/experiment-catalog]].

## Experimental map

| Priority | Experiment | Status |
| -------- | ---------- | ------ |
| 🔵 Primary | Experiment 6 — Ontology-LLM (A + B) | Planning |
| 🔵 High | Experiment 3 — Proactive questioning | Planning |
| 🟡 Medium | Experiment 4 — Hybrid architecture | Planning |
| 🟡 Medium | Experiment 2 — Appraisal-based empathy | Planning |
| 🔴 Lower | Experiment 1 — Process rewards / GRPO | Active work; SFT v2 + benchmark on `main`, GRPO trainer on a separate branch. Ablation A/B/C/D. Kept as supporting infrastructure for Experiment 6 comparisons. |
| 🔴 Lower | Experiment 5 — Dynamic user modelling | Future work |

## Code scaffolding

- [[sources/code/sft-v2-pipeline]] — constitution-driven data pipeline
- [[sources/code/constitution-document]] — the 19-principle source
- [[sources/code/training-and-benchmark]] — LoRA + GRPO trainer, benchmark, context-degradation eval

## Key entities

- [[entities/constitution]] — v2 SFT principles
- [[entities/grpo]] — RL algorithm choice
- [[entities/qwen3-0.6b]] — base model
- [[entities/mcp]] · [[entities/rag]]
- [[entities/5w-h]] · [[entities/appraisal-theory]]

## Open questions
See [[questions/2026-04-19-initial-questions]] — grouped by exploration
TODOs, ontology-LLM (advisor prep), reasoning/RL design,
personalisation/empathy, infrastructure, and literature tensions.

## Sources

- [[sources/dissertation/research-plan]] — formal plan with thesis title, 5 objectives, 7 phases, 2 pivots
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — long-form literature review
- [[sources/dissertation/experimental-planning-document]] — experiments catalog + 2025-11-10 pivot
- [[sources/dissertation/personal-notes]]
- `README.md`
