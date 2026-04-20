---
title: Research Plan (researchplan.tex)
type: source
kind: dissertation-draft
author: the user (for CS7CS6 submission to Prof Owen Conlan)
tags: [thesis, plan, formal, objectives, timeline]
sources:
  - researchplan.tex
  - TCD_SCSS_CS7CS6_Research_Plan_Ajinkya_Taranekar.pdf
updated: 2026-04-19
status: current
---

# Research Plan

**The CS7CS6 formal research-plan submission. Establishes the thesis title, research question, five SMART objectives, a seven-phase timeline (Oct 2025 – Aug 2026), two documented pivots, ethics considerations, and the keyword- vs AI-search literature strategy.**

## Thesis identity

- **Title:** *Architecting Trust and Empathy in Conversational AI: Towards A Modular Approach Integrating 5W+H User Modelling and Explainable Reasoning*
- **Degree:** MSc Computer Science, TCD
- **Supervisor:** Prof. Owen Conlan
- **Keywords:** LLMs, Explainable AI, User Modelling, Empathetic AI, Reasoning, Trustworthiness, Conversational AI

## Formal research question

> How can we architect a modular conversational AI system that prioritises transparency and is capable of genuine contextual empathy through systematic User Modelling and appropriate tool delegation, rather than relying solely on end-to-end neural generation?

*(Narrower than the exploratory question in [[sources/dissertation/road-towards-trustworthy-empathetic-ai]]; the Research Plan Edits file flagged the original as "too huge" — this is the narrowed version the advisor agreed to.)*

## Five objectives

1. **Comprehensive Literature Review & Ethical Framework** (by Dec 2025)
2. **Baseline Implementation & Failure Analysis** via Boolean/Math GPT-2 (by Jan 2026) — tokenisation failure mode documentation.
3. **Design & Prototype Modular Reasoning Architecture** (months 3–5) — Reasoning Module (ToT + interleaved), User Modelling Module (5W+H), Tool Integration Layer (MCP).
4. **Privacy-Preserving User Modelling** (months 5–7) — AppraisePLM integration + local MCP storage, cold start via 5W+H.
5. **Multi-Dimensional Evaluation** (final months) — quantitative + qualitative + ablation.

## Seven phases

| Phase | Dates | Focus |
| ----- | ----- | ----- |
| 1 | Oct–Nov 2025 | Foundational experimentation + AppraisePLM reading |
| 2 | Nov–Dec 2025 | Modular architecture + interfaces |
| 3 | Dec 2025 – Mar 2026 | Reasoning engine implementation + comparative analysis (CoT, ToT, interleaved, latent) |
| 4 | Apr–May 2026 | User Modelling + empathy via [[entities/appraisal-theory]] + [[entities/graph-rag]] |
| 5 | May 2026 | Tool integration via [[entities/mcp]] |
| 6 | Jun–Jul 2026 | Explainability mechanism development |
| 7 | Jun–Aug 2026 | Comprehensive evaluation + write-up |

> ⚠ The tex source says "Phase 3: Dec'25 - Mar'25" (line ~244) — almost certainly a typo for "Mar'26". Flag for the user to correct on next edit pass.

## Two documented pivots

- **Pivot 1 (Oct 2025) — monolithic → modular.** Originally planned to fine-tune a single LLM with reasoning / empathy / user-modelling heads. Prof. Conlan argued catastrophic forgetting at every level; switched to specialised components communicating through interfaces.
- **Pivot 2 (Nov 2025) — emergent questioning → explicit 5W+H.** Originally planned to fine-tune on therapy transcripts hoping the model would emerge good questioning. Critique: "hoping for emergent behaviour rather than engineering for specific capabilities". Replaced with the systematic [[entities/5w-h]] framework.

The 2025-11-10 ontology shift ([[decisions/2025-11-10-ontology-focus-shift]]) post-dates this plan and is not yet reflected in it. A pivot-3 entry should be added when the plan is revised.

## Ethics

School Ethics Application **required**: user studies involve emotional conversations, personal-data capture for User Modelling, and qualitative interview debriefs. Covered: informed consent, data privacy, psychological safety, deception concerns (empathy simulation), vulnerable populations.

## Literature-search strategy

- **Keyword search** (Google Scholar + TCD Stella) — foundational canon (Attention, CoT, ToT, PAL). Query: `"large language model" AND (reasoning OR "chain of thought" OR "tree of thought" OR "latent space" OR "interleaved thinking")`.
- **AI search** (Perplexity Pro, Deep Research) — cutting-edge 2024-25 work (Coconut, interleaved reasoning, MCP, HRM, SEAL, structured templates). First-principles + adversarial prompt structure.
- **Reflection:** keyword = canonical, AI = frontier; combine both, verify AI-cited papers before citing.

## Papers cited in the plan — ingestion status

| Cited work | In `docs/Assets/`? | Wiki status |
| ---------- | ------------------- | ----------- |
| Attention Is All You Need | ✓ | [[sources/papers/attention-is-all-you-need]] |
| BERT | ✓ | [[sources/papers/bert]] |
| CoT | ✓ | [[sources/papers/chain-of-thought-prompting]] |
| ToT | ✓ | [[sources/papers/tree-of-thoughts]] |
| PAL | ✓ | [[sources/papers/pal]] |
| Coconut | ✓ | [[sources/papers/coconut-continuous-latent]] |
| DeepSeek-R1 | ✓ | [[sources/papers/deepseek-r1]] |
| MCP spec (Anthropic) | Indirect (multi-agent paper) | [[sources/papers/mcp-multi-agent]] |
| Interleaved Reasoning | ✓ | [[sources/papers/interleaved-reasoning]] |
| HRM | ✓ | [[sources/papers/hierarchical-reasoning-model]] |
| **Long 2023 — LLM-guided ToT** | ✗ | **Not ingested** |
| **Sun 2024 — Think-on-Graph** | ✗ | **Not ingested** — directly informs [[entities/graph-rag]] |
| **Zweiger 2025 — SEAL** | ✗ | Not ingested |
| **Yang 2025 — Structured Templates** | ✗ | Not ingested |
| **Debnath 2025 — AppraisePLM** | ✗ | **Not ingested** — blocker for Experiment 2 empathy work |

See [[questions/2026-04-19-initial-questions]] for the AppraisePLM / Think-on-Graph ingestion TODO.

## Related

- [[overview]]
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — long-form literature review
- [[sources/dissertation/experimental-planning-document]] — experiments catalog + 2025-11-10 pivot
- [[decisions/2025-11-10-ontology-focus-shift]]
- [[experiments/experiment-catalog]]
- [[entities/5w-h]] · [[entities/appraisal-theory]] · [[entities/mcp]] · [[entities/graph-rag]]
- [[topics/ontology-integration]] · [[topics/reasoning]] · [[topics/personalisation]] · [[topics/empathy]] · [[topics/explainability]]

## Raw

- `researchplan.tex`
- `TCD_SCSS_CS7CS6_Research_Plan_Ajinkya_Taranekar.pdf`
