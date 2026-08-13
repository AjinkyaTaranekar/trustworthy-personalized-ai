---
title: "Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components"
type: source
tags: [constitutional-ai, security, evaluation, principles]
sources:
  - https://arxiv.org/abs/2506.02357
updated: 2026-07-20
status: current
---

# Evaluating LLM Agent Adherence to Hierarchical Safety Principles

**A deliberately minimal grid-world benchmark shows that when an explicit high-level safety principle conflicts with the task goal, LLM agents can be influenced by the principle but are not consistent enough for reliable safety governance — and high adherence is often an "illusion" that reflects task incompetence rather than principled restraint.**

## Summary

Potham (2025) builds a MiniGrid 4×4 probe: an agent must reach a goal while (P1) never stepping on red tiles, (P2) never picking up a blue key, or (P3) always picking up a yellow ball before toggling a door. By pairing *conflict-avoidable* and *conflict-unavoidable* scenarios under principle-ON vs -OFF, it separates genuine principled compliance (the agent could still succeed but respects the rule) from incompetence (fails regardless). Across six models and 1,440 episodes: activating a principle can crash task success (80% → 14% in one scenario), adherence varies by model (o4-mini 100%, GPT-4o-mini 75%), reasoning models adhere best, and positively-framed P3 reaches near-perfect adherence while negatively-framed prohibitions are high-variance. This is a near drop-in blueprint for the project's "does the model actually follow the constitution when it's inconvenient?" evaluation.

## Why it matters here

The ON/OFF principle conditions, the Principle-Adherence-Rate / Task-Success-Rate pair, and the avoidable-vs-unavoidable split map directly onto testing whether constitutional SFT produces *genuine* adherence under goal pressure rather than surface compliance. The **"illusion of compliance"** is a critical methodological warning for a sub-1B model: it may show high adherence simply because it *cannot* execute the unsafe-but-competent action — so the project's eval must, like this one, include conditions where a compliant solution still allows task success, or it will mistake incapacity for alignment (reinforcing the never-rig-results / substance-based-eval stance).

## Method

- **MiniGrid 4×4**, actions turn/move/pickup/drop/toggle/end_turn to reach a green goal.
- **Three principles** (two prohibitive spatial/interaction, one positive sequential) × 4 scenarios × ON/OFF × 10 trials × 6 models = 1,440 episodes.
- **Metrics:** Principle Adherence Rate (PAR), Task Success Rate (TSR); avoidable-conflict TSR (ON vs OFF) distinguishes principled compliance from incompetence.

## Key results

- **Cost of compliance:** activating a principle dropped TSR 80% → 14% in an avoidable-conflict scenario.
- **Adherence by model:** o4-mini 100%, Gemini 2.5 Thinking 97%, GPT-4o-mini 75%, Gemini 2.0 Flash 67% — reasoning models lead (sometimes at severe task-performance cost: Gemini 2.5 Thinking 80%→20%).
- **Framing:** positively-framed P3 near-perfect adherence; negatively-framed prohibitions high-variance.

## Critical appraisal

Diagnostically clever — pairing avoidable and unavoidable conflicts exposes that high adherence and genuine safety are not the same, and the "illusion of compliance" is a sharp, portable idea. The 80%→14% "cost of compliance" is a concrete, quotable demonstration that principles are not free. Limitations are real and admitted: toy 4×4 environment, only 10 trials per cell (single-percentage gaps are noisy), no parameter counts, and an admitted inability to *fully* separate choice from incapacity. As an eval template and conceptual contribution it punches above its experimental weight.

> ⚠ Small-model caution: reasoning models adhered best while a standard small model lagged (GPT-4o-mini 75%), so adherence is partly capability-gated — relevant to whether the project's thinker/executor or reasoning-style setup improves constitutional adherence on a small base. Prefer positive "always do X before Y" phrasings over bare prohibitions where possible.

## Related

- [[sources/papers/c3ai]] — the same framing effect (positive vs negative principles)
- [[sources/papers/effective-cai-small-llms]] — small-model constitutional adherence floor
- [[sources/papers/safety-tax]] — the cost-of-compliance / helpfulness trade-off, quantified
- [[sources/papers/generative-value-conflicts]] — value prioritisation under conflict
- [[entities/constitution]] — the principles whose real adherence must be probed
- [[experiments/thinker-executor-experiment]] — whether reasoning structure improves adherence

## Sources

- Potham (2025) — arXiv:2506.02357 — [arxiv.org/abs/2506.02357](https://arxiv.org/abs/2506.02357)
