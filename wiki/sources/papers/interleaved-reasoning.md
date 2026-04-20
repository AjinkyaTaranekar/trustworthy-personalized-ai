---
title: Interleaved Reasoning for LLMs via RL
type: source
arxiv_id: 2505.19640v1
authors: Xie et al. (Apple)
year: 2025
tags: [reasoning, rl, interleaved, ttft]
sources:
  - docs/Assets/Interleaved Reasoning for Large Language Models via Reinforcement Learning (2505.19640v1).pdf
  - docs/Literature Notes/Interleaved Reasoning for Large Language Models via Reinforcement Learning (2505.19640v1).md
updated: 2026-04-19
status: current
---

# Interleaved Reasoning

**Uses RL (PPO / GRPO / REINFORCE++) with a rule-based intermediate-step reward to make LLMs *interleave* thinking and answering for multi-hop questions — slashes time-to-first-token by 80%+, lifts Pass@1 by up to 19.3%.**

## What it does
Incentivises correct intermediate steps via conditional rewards. Shows that interleaved reasoning is latent in pretrained models and can be *activated* by RL. Generalises from QA + logic training to MATH, GPQA, MMLU.

## Why it matters for this thesis
This is the **direct theoretical grounding** for the "interleaved thinking" approach the user flagged in `docs/Dissertation/Rough Notes.md`. It answers a pivotal design question: should reasoning happen up-front (long CoT) or woven into the answer? The TTFT win also matters for the empathy pillar — users perceive empathetic latency as important.

## Related

- [[topics/reasoning]] · [[topics/empathy]]
- [[sources/papers/react]] — tool-interleaved counterpart
- [[sources/papers/seed15-thinking]]
- [[entities/grpo]]

## Sources

- `docs/Assets/Interleaved Reasoning for Large Language Models via Reinforcement Learning (2505.19640v1).pdf`
- `docs/Literature Notes/Interleaved Reasoning for Large Language Models via Reinforcement Learning (2505.19640v1).md`
