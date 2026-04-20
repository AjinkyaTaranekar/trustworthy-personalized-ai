---
title: "Language Models are Hidden Reasoners (LaTRO)"
type: source
arxiv_id: 2411.04282v2
authors: Chen et al. (Salesforce AI Research)
year: 2024
tags: [reasoning, latent, self-reward, variational]
sources:
  - docs/Assets/Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2).pdf
  - docs/Literature Notes/Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2).md
updated: 2026-04-19
status: current
---

# LaTRO — Hidden Reasoners

**Frames reasoning as sampling from a latent distribution and optimises it via variational self-rewarding — no external reward model needed.**

## What it does
Jointly improves the reasoning process and the model's self-evaluation of reasoning quality. Gains 12.5% zero-shot on GSM8K over base, 9.6% over SFT across Phi-3.5-mini, Mistral-7B, Llama-3.1-8B.

## Why it matters for this thesis
LaTRO is philosophically aligned with the dissertation's claim that "reasoning is latent, not built" (the *Road Towards…* draft's emergence debate). It also suggests a **no-reward-model** route that could sidestep one of the costliest RL components. Combined with [[sources/papers/self-enhanced-reasoning|SERT]], hints at a recipe where the Qwen3-0.6B base self-generates + self-scores + self-trains before any teacher involvement.

## Related

- [[topics/reasoning]]
- [[sources/papers/self-enhanced-reasoning]]
- [[sources/papers/seed15-thinking]]
- [[entities/qwen3-0.6b]]

## Sources

- `docs/Assets/Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2).pdf`
- `docs/Literature Notes/Language Models are Hidden Reasoners Unlocking Latent Reasoning Capabilities via Self-Rewarding (2411.04282v2).md`
