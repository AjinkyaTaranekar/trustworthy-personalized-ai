---
title: "Diffusion of Thoughts (DoT)"
type: source
arxiv_id: 2402.07754v3
authors: Ye et al.
year: 2024
venue: NeurIPS 2024
tags: [reasoning, diffusion, cot, self-correction]
sources:
  - docs/Assets/Diffusion of Thoughts Chain-of-Thought Reasoning in Diffusion Language Models (2402.07754v3).pdf
  - docs/Literature Notes/Diffusion of Thoughts Chain-of-Thought Reasoning in Diffusion Language Models (2402.07754v3).md
updated: 2026-04-19
status: current
---

# Diffusion of Thoughts (DoT)

**Marries CoT with diffusion language models so reasoning steps diffuse
over time rather than being generated strictly left-to-right. Trades
compute for reasoning quality more smoothly than autoregressive CoT.**

## What it does
On multi-digit multiplication, boolean logic, and GSM-style problems, a
small diffusion model beats a much larger autoregressive model, with
emergent self-correction and compatibility with self-consistency decoding.

## Why it matters for this thesis
Supports the architectural-alternatives narrative around
[[sources/papers/attention-is-all-you-need|Attention Is All You Need]]'s
no-backtracking limitation: diffusion *does* backtrack. Less immediately
actionable for the current pipeline (diffusion LMs are not the base
model), but relevant as evidence that self-correction capability can be an
architectural property rather than a trained behaviour.

## Related

- [[topics/reasoning]] · [[topics/llm-foundations]]
- [[sources/papers/ladir]]
- [[sources/papers/coconut-continuous-latent]]

## Sources

- `docs/Assets/Diffusion of Thoughts Chain-of-Thought Reasoning in Diffusion Language Models (2402.07754v3).pdf`
- `docs/Literature Notes/Diffusion of Thoughts Chain-of-Thought Reasoning in Diffusion Language Models (2402.07754v3).md`
