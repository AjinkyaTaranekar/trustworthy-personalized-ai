---
title: Group Relative Policy Optimization (GRPO)
type: entity
tags: [rl, grpo, training, deepseek]
sources:
  - docs/Assets/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).pdf
  - docs/Assets/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).pdf
  - pipeline/2_model_trainer.py
updated: 2026-04-19
status: current
---

# GRPO

**A memory-efficient policy-gradient RL algorithm (introduced in the
DeepSeek family) that replaces PPO's critic/value model with a per-group
advantage computed from multiple sampled completions per prompt.**

## Why it's used in this repo
GRPO is the core RL algorithm of the pipeline's RL trainer. It sidesteps the
value-model cost of PPO, which matters on a small base model
([[entities/qwen3-0.6b|Qwen3-0.6B]]) where every GB of VRAM counts. Rewards
are composite: format, correctness, tool_integrity, and behavioural
(constitution-adherence).

## Known pitfalls (flagged by literature)
- **Length bias:** [[sources/papers/understanding-r1-zero]] identifies a
  systematic reward toward longer incorrect outputs. Remedy: Dr. GRPO.
- **Reward hacking:** [[sources/papers/vlm-r1]] documents OD-specific
  hacking that generalises to any rule-based reward.
- **Value-based alternatives:** [[sources/papers/vapo]] argues a stabilised
  value-based PPO can outperform GRPO on AIME.

## Related

- [[topics/reasoning]]
- [[sources/papers/deepseek-r1]] · [[sources/papers/understanding-r1-zero]]
- [[sources/papers/vapo]] · [[sources/papers/ui-r1]] · [[sources/papers/vlm-r1]]
- [[entities/qwen3-0.6b]] · [[entities/constitution]]

## Sources

- Pipeline: `pipeline/2_model_trainer.py` (LoRA + GRPO setup)
- Papers listed above.
