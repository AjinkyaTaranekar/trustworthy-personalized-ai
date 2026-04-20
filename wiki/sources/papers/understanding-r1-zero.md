---
title: "Understanding R1-Zero-Like Training: A Critical Perspective"
type: source
arxiv_id: 2503.20783v2
authors: Liu et al. (Sea AI Lab)
year: 2025
tags:
  - rl
  - grpo
  - critique
  - dr-grpo
sources:
  - docs/Assets/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).pdf
  - docs/Literature Notes/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).md
updated: 2026-04-19
status: current
---

# Understanding R1-Zero-Like Training

**Critically analyses R1-Zero's two components (base model + RL) and finds: Qwen2.5 base already reasons without prompt templates (pretraining bias), and GRPO has an optimisation bias that artificially lengthens incorrect outputs. Proposes Dr. GRPO — unbiased, more token-efficient.**

## What it does
Attributes "aha moments" partly to pretraining, not only RL. Introduces Dr. GRPO and a minimalist recipe: 43.3% AIME 2024 with a 7B base.

## Why it matters for this thesis
Immediately relevant to this pipeline's [[entities/grpo|GRPO]] trainer. Two consequences: (1) the repo's use of [[entities/qwen3-0.6b|Qwen3-0.6B]] as base means some of the "emergent" behaviour in ablation Conditions C/D may be attributable to pretraining rather than RL — **experiment design should control for this**. (2) Consider switching to Dr. GRPO to avoid length-reward gaming. File as a follow-up in `wiki/questions/`.

## Related

- [[topics/reasoning]]
- [[sources/papers/deepseek-r1]]
- [[sources/papers/vapo]]
- [[entities/grpo]] · [[entities/qwen3-0.6b]]

## Sources

- `docs/Assets/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).pdf`
- `docs/Literature Notes/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).md`
