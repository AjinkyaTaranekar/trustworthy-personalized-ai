---
title: Thesis Overview
type: meta
tags: [thesis, synthesis]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
  - docs/Dissertation/Experimental Planning Document.md
  - researchplan.tex
  - README.md
updated: 2026-05-03
status: current
---

# Thesis Overview

## Official Research Question (researchplan.tex §1.2)

> **"How can we architect a modular conversational AI system that prioritises transparency and is capable of genuine contextual empathy through systematic User Modelling and appropriate tool delegation, rather than relying solely on end-to-end neural generation?"**

This is the binding research question. All experiments, design decisions, and implementation choices must be anchored to it. The key terms are: **modular** (not monolithic), **transparent** (scrutable, not post-hoc rationalisation), **systematic User Modelling** (5W+H + KG, not implicit embeddings), **tool delegation** (MCP + logging, not internal hallucination).

## Central hypothesis
Monolithic LLMs cannot simultaneously be empathetic, logically rigorous, and personalised without catastrophic forgetting and hallucination. A **four-module architecture** is required — see [[decisions/2025-10-01-four-module-architecture]] for the binding design decision (Pivot 1, October 2025, Professor Conlan feedback).

## Operational hypothesis (added 2026-05-03)
A 0.6B model fine-tuned with constitution-guided SFT and GRPO achieves trust and empathy ratings **comparable to frontier models** (within 0.5 points on a 5-point human evaluation scale) on targeted trust-relevant behaviours, while running entirely on-device with local user memory and no persistent internet access — offering a privacy guarantee that API-only frontier models (Claude Sonnet 4.6, Minimax M2.7, Kimi K2.6) cannot match by architectural necessity. See [[decisions/2026-05-03-research-question-reframe]] for the full rationale. The comparison study is specified in [[experiments/frontier-model-comparison]]; the human evaluation instrument in [[experiments/human-evaluation-rubric]]; the psychological grounding of the constitution in [[topics/constitution-psychological-grounding]].

## Four-module architecture (Pivot 1 — binding)

| Module | Responsibility | Key tech |
|---|---|---|
| **Reasoning Module** | Constitutional trustworthiness, problem decomposition | Qwen3-0.6B + SFT + GRPO |
| **User Modelling Module** | 5W+H persistent profiles — NOT a neural network | FalkorDB + Cognee + local MCP server |
| **Tool Integration Layer** | Delegation, MCP invocation, explainability logs | Orchestration logic; every call logged |
| **Generator Module** | Final NL synthesis | Base LLM + prompting + RAG |

The User Modelling Module is a **structured database**, not trained on user data. This is what makes personalisation scrutable — users read graph nodes, not weight matrices. The Tool Integration Layer produces **real logs of actual decisions**, not post-hoc rationalisation.

## Why small models (Qwen3-0.6B, Gemma 4)

Small models enable **local deployment on the user's own device** — the architectural basis for the privacy guarantee. A model that physically cannot exfiltrate data does not need to be trusted not to. The thesis goal is not to beat frontier models on capability benchmarks; it is to demonstrate that the right modular architecture produces more trustworthy, scrutable, and empathetic outputs than a larger monolithic model with no structure. Gemma 4 is the secondary comparison model because Google's on-device Gemma story directly parallels this local-first argument.

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
On 2025-11-10 the advisor reframed the project around **ontology-LLM integration** as the flagship experiment. Pure process-reward RL becomes supporting infrastructure. See [[decisions/2025-11-10-ontology-focus-shift]] and [[experiments/experiment-catalog]].

## Binding constraint: small models only

All experiments are constrained to small models — **Qwen3-0.6B** (primary) and **Gemma 4** (secondary comparison). The central thesis claim is not that a small model beats frontier models at raw capability; it is that a small model with the right modular architecture (GRPO-trained reasoning + graph-gated personalisation + appraisal-conditioned empathy) is more trustworthy, more scrutable, and more empathetic than a larger monolithic model with no structure. The Gemma 4 comparison is included because Google's on-device inference story for Gemma directly mirrors the local-first privacy argument. This constraint is binding across all six experiments. See [[queries/grpo-and-personalisation-master-plan]] for the full implementation plan.

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
- [[sources/code/constitution-document]] — the 23-principle source
- [[sources/code/training-and-benchmark]] — LoRA + GRPO trainer, benchmark, context-degradation eval

## Key entities

- [[entities/constitution]] — v2 SFT principles
- [[entities/grpo]] — RL algorithm choice
- [[entities/qwen3-0.6b]] — base model
- [[entities/mcp]] · [[entities/rag]]
- [[entities/5w-h]] · [[entities/appraisal-theory]]

## Open questions
See [[questions/2026-04-19-initial-questions]] — grouped by exploration TODOs, ontology-LLM (advisor prep), reasoning/RL design, personalisation/empathy, infrastructure, and literature tensions.

## Sources

- [[sources/dissertation/research-plan]] — formal plan with thesis title, 5 objectives, 7 phases, 2 pivots
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — long-form literature review
- [[sources/dissertation/experimental-planning-document]] — experiments catalog + 2025-11-10 pivot
- [[sources/dissertation/personal-notes]]
- `README.md`
