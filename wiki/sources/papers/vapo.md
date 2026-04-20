---
title: "VAPO: Value-based Augmented PPO for Reasoning"
type: source
arxiv_id: 2504.05118v3
authors: Yue et al. (ByteDance Seed)
year: 2025
tags: [rl, ppo, reasoning, value-based]
sources:
  - docs/Assets/VAPO Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (2504.05118v3).pdf
  - docs/Literature Notes/VAPO Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (2504.05118v3).md
updated: 2026-04-19
status: current
---

# VAPO

**Value-based PPO tuned for long-CoT reasoning: fixes value-model bias, heterogeneous sequence lengths, and sparse reward — reaches SOTA on AIME 2024 (60.4) in 5,000 steps with no training crashes.**

## What it does
Identifies three pathologies of value-based RL on long-chain reasoning and designs targeted fixes. Beats DeepSeek-R1-Zero-Qwen-32B and DAPO by 10+ points under identical settings.

## Why it matters for this thesis
VAPO is a direct alternative to the [[entities/grpo|GRPO]] used in this repo's pipeline. The repo's GRPO choice is pragmatic (memory-efficient, no value model) but VAPO's results argue that value-based approaches, when stabilised, can be more sample-efficient. Worth citing as a design alternative in the ablation study design, and as evidence that **RL instability is the bottleneck** — not reward shape.

## Related

- [[topics/reasoning]]
- [[entities/grpo]]
- [[sources/papers/understanding-r1-zero]] — critical review of GRPO
- [[sources/papers/deepseek-r1]]

## Sources

- `docs/Assets/VAPO Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (2504.05118v3).pdf`
- `docs/Literature Notes/VAPO Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks (2504.05118v3).md`
