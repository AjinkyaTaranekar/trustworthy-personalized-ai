---
title: "Token-Hungry, Yet Precise: DeepSeek R1 on MATH"
type: source
arxiv_id: 2501.18576v1
authors: Evstafev
year: 2025
tags: [reasoning, evaluation, latency, trade-off]
sources:
  - docs/Assets/Token-Hungry, Yet Precise DeepSeek R1 Highlights the Need for Multi-Step Reasoning Over Speed in MATH (2501.18576v1).pdf
  - docs/Literature Notes/Token-Hungry, Yet Precise DeepSeek R1 Highlights the Need for Multi-Step Reasoning Over Speed in MATH (2501.18576v1).md
updated: 2026-04-19
status: current
---

# Token-Hungry, Yet Precise

**DeepSeek R1 solves 30 previously-unsolvable MATH problems when time limits
are removed — but generates far more tokens than rivals. Documents the
accuracy-vs-efficiency trade-off explicitly across 11 temperatures.**

## What it does
Compares R1 to GPT-4o-mini, Gemini-1.5-flash-8b, Llama3.1-8b, Mistral-8b
without latency constraints. Shows R1's long reasoning is *load-bearing* for
correctness, not wasteful decoration.

## Why it matters for this thesis
Quantifies the cost side of "think longer → answer better". Important for
[[topics/empathy]] too: a trustworthy system that always thinks R1-long will
feel unresponsive. Motivates the case for
[[sources/papers/interleaved-reasoning|interleaved reasoning]] and
[[sources/papers/dual-head-reasoning-distillation|dual-head distillation]]
as cost-aware alternatives. Feeds directly into benchmark design —
tool_discipline shouldn't be optimised without a latency counter-metric.

## Related

- [[topics/reasoning]] · [[topics/empathy]]
- [[sources/papers/deepseek-r1]]
- [[sources/papers/prompting-science-report-2]]
- [[sources/papers/interleaved-reasoning]]

## Sources

- `docs/Assets/Token-Hungry, Yet Precise DeepSeek R1 Highlights the Need for Multi-Step Reasoning Over Speed in MATH (2501.18576v1).pdf`
- `docs/Literature Notes/Token-Hungry, Yet Precise DeepSeek R1 Highlights the Need for Multi-Step Reasoning Over Speed in MATH (2501.18576v1).md`
