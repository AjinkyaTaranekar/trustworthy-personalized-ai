---
title: Experiment Catalog
type: experiment
tags: [experiments, planning, ablation]
sources:
  - docs/Dissertation/Experimental Planning Document.md
  - README.md
updated: 2026-04-19
status: current
---

# Experiment Catalog

**All experiments in scope for the dissertation plus the current repo pipeline's ablation. Sourced from the 2025-11-10 meeting + post-meeting revisions.**

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
