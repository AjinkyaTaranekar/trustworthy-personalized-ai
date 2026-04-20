---
title: "Seed1.5-Thinking: Advancing Reasoning Models with RL"
type: source
arxiv_id: 2504.13914v3
authors: ByteDance Seed
year: 2025
tags: [reasoning, rl, process-rewards, moe]
sources:
  - docs/Assets/Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3).pdf
  - docs/Literature Notes/Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3).md
updated: 2026-04-19
status: current
---

# Seed1.5-Thinking

**A 20B-active / 200B-total MoE "thinking-before-responding" model that beats DeepSeek-R1 by 8% on non-reasoning win rate while matching it on reasoning.**

## What it does
Scales reasoning RL with a relatively small active footprint (MoE). Reports 86.7 AIME 2024, 55.0 Codeforces, 77.3 GPQA. Releases two new internal benchmarks (BeyondAIME, Codeforces) for generalised reasoning.

## Why it matters for this thesis
The dissertation draft cites this as the canonical **process-reward** exemplar: reward signal spans process correctness (steps are sound), efficiency (shortest valid path), and verifiability (each step checkable independently). This is the axis the thesis wants to push: reward *how* the model thinks, not only *what* it concludes. Operationally it underwrites the "behavioural reward" design in the repo's RL pipeline — tool_integrity plus capability-honesty signals layered on top of outcome correctness.

## Related

- [[topics/reasoning]]
- [[sources/papers/deepseek-r1]] — the baseline this improves on
- [[entities/constitution]] — behavioural-reward counterpart in our pipeline
- [[decisions/2025-11-10-ontology-focus-shift]] — why this line became secondary

## Sources

- `docs/Assets/Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3).pdf`
- `docs/Literature Notes/Seed1.5-Thinking Advancing Superb Reasoning Models with Reinforcement Learning (2504.13914v3).md`
