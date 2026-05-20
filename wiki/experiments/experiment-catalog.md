---
title: Experiment Catalog
type: experiment
tags: [experiments, planning, ablation, small-model, qwen, gemma]
sources:
  - docs/Dissertation/Experimental Planning Document.md
  - README.md
updated: 2026-05-01
status: current
---

# Experiment Catalog

**All experiments in scope for the dissertation plus the current repo pipeline's ablation. Sourced from the 2025-11-10 meeting + post-meeting revisions.**

> **Binding constraint (2026-05-01):** All experiments use small models only — Qwen3-0.6B as the primary model and Gemma 4 as the secondary comparison. Small models enable **local on-device deployment**, which is the architectural basis for the privacy guarantee (a model that physically cannot exfiltrate data does not need to be trusted not to). The thesis does not claim to beat frontier models at capability; it claims that the right four-module architecture (see [[decisions/2025-10-01-four-module-architecture]]) produces more trustworthy, scrutable, and empathetic outputs. See [[queries/grpo-and-personalisation-master-plan]] and [[overview]] for full rationale.

## Formal evaluation strategy (researchplan.tex §1.4)

All experiments share the following evaluation instruments. **This is the binding evaluation specification from the research plan — do not add metrics that cannot be measured with these instruments.**

**Quantitative benchmarks:**
- Reasoning accuracy: GSM8K and MATH dataset (subset appropriate for small-model scale — do not use full MATH competition set)
- Logic puzzles where baseline LLMs demonstrably fail (design custom; focus on backtracking and multi-step)
- Appraisal detection accuracy: Crowd-event dataset ground truth (AppraisePLM, 21 dimensions)
- Computational accuracy: 100% expected when delegating to Python tool (MATH=CODE principle)

**Qualitative user studies (subject to TCD ethics approval):**
- Perceived empathy ratings: validated HCI instruments
- Trustworthiness assessment: scenario-based evaluation
- Explanation comprehension: can users understand *why* the system made a decision? (critical distinction: "satisfying" vs genuinely causal — post-hoc rationalisation can score high on the former)

**Ablation structure (per researchplan.tex):**
- With vs without ToT reasoning
- With vs without User Modelling
- With vs without tool augmentation
- Each ablation measures the individual contribution of that component

---

## Dissertation experiments (by priority)

### 🔵 Experiment 6 — Ontology-LLM Integration (PRIMARY FOCUS)
After the 2025-11-10 advisor meeting this became the flagship experiment. Two parallel approaches:

- **Approach A** — ontology as core knowledge base; LLM as NL interface.
- **Approach B** — ontology as post-hoc verifier of LLM claims.

Metrics: accuracy vs ontology ground truth, routing precision (A), verification accuracy (B), hallucination reduction, user trust, cross-cultural consistency.

Full treatment: [[topics/ontology-integration]]. Driving decision: [[decisions/2025-11-10-ontology-focus-shift]].

### 🔵 Experiment 3 — Proactive Questioning vs Assumption-Based Responses
Scenario-based within-subject study (N=50–100). Compare 5W+H-driven inquiry against generic advice. Metrics: empathy scale, usefulness, trust, interaction-style preference.

### 🟡 Experiment 4 — Hybrid Reasoning Architecture
Latent reasoner ([[sources/papers/hierarchical-reasoning-model|HRM]] / [[sources/papers/coconut-continuous-latent|Coconut]]) + language generator + tool augmentation ([[sources/papers/pal|PAL]]) + RAG. Metrics: task accuracy, explainability quality, token efficiency, computation transparency.

### 🟡 Experiment 2 — Empathetic Response via Appraisal Theory
Two-phase: appraisal detection (AppraisePLM tagger) → conditioned generation. Compare (a) standard LLM, (b) appraisal-conditioned, (c) human gold. Metrics: human empathy rating, appraisal F1, satisfaction.

### 🔴 Experiment 1 — Reasoning Process Reward System
Process-reward RL vs outcome-only RL on math, logic, and novel problems. Metrics: accuracy, intermediate-step quality, robustness to rephrasing (see [[sources/papers/none-of-the-others]]).

> ⚠ The current repo pipeline **implements this experiment** (SFT + GRPO + behavioural rewards) despite Experiment 1 being ranked "Lower Priority" in the planning doc. Either infrastructure-kept-for-ablation or priorities drifted — ask user.

### 🔴 Experiment 5 — Real-Time User Modelling with Dynamic Adaptation
5W+H-driven user model, updated per turn. Control: no modelling; static profile; dynamic (proposed). Metrics: satisfaction over turns, personalisation accuracy, adaptation responsiveness, privacy comfort.

### 🟡 Experiment 0 — Reasoning Paradigm Comparative Analysis (Phase 3, researchplan.tex)

**This experiment is specified in researchplan.tex Phase 3 (Dec 2025 – Mar 2026) and is absent from the current pipeline.** It must be completed before GRPO begins because its results determine which reasoning approach the Reasoning Module uses. From the plan: "implementing multiple reasoning approaches in parallel and running controlled comparisons."

Four conditions to compare on the same benchmark set (GSM8K subset + logic puzzles):

| Condition | Approach | Papers |
|---|---|---|
| R0 | Standard LLM with system prompt engineering (baseline) | — |
| R1 | Chain-of-Thought prompting | [[sources/papers/chain-of-thought-prompting]] |
| R2 | Tree-of-Thoughts (branching + backtracking) | [[sources/papers/tree-of-thoughts]] |
| R3 | Interleaved thinking (RL-trained alternating reasoning + action) | [[sources/papers/interleaved-reasoning]] |
| R4 | Latent reasoning (CoCoNut — continuous thought vectors) | [[sources/papers/coconut-continuous-latent]] |

Metrics: accuracy, intermediate-step quality (where visible), token efficiency, transparency to the user (can a user follow the reasoning?). The winner feeds into the Reasoning Module design. Results also inform the SFT training data format (which style of reasoning trace to teach).

> ⚠ **Not yet implemented.** Phase 3 deadline was Mar 2026. Either implement now as a pre-GRPO prerequisite, or document why it was superseded.

## Current repo ablation — Conditions A / B / C / D

Sits inside Experiment 1's scope. From the pipeline:

| Condition | Model | Question answered |
| --------- | ----- | ----------------- |
| **A** | Base Qwen3-0.6B, no training | Baseline |
| **B** | SFT only (`checkpoint_sft`) | Does SFT format matter? |
| **C** | SFT → GRPO, format + accuracy only | Does RL correctness signal matter? |
| **D** | SFT → GRPO + tool_integrity + behavioural (full) | Full thesis contribution |

Measured via [[sources/code/training-and-benchmark|`4_benchmark.py`]] and [[sources/code/training-and-benchmark|`5_context_degradation.py`]].

## Related

- [[sources/dissertation/experimental-planning-document]]
- [[decisions/2025-11-10-ontology-focus-shift]]
- [[questions/2026-04-19-initial-questions]]
- [[sources/code/sft-v2-pipeline]] · [[sources/code/training-and-benchmark]]
- [[entities/5w-h]] · [[entities/appraisal-theory]]

## Raw

- `docs/Dissertation/Experimental Planning Document.md`
- `README.md`
