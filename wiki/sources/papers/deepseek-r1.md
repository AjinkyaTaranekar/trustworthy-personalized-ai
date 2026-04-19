---
title: "DeepSeek-R1: Incentivizing Reasoning Capability via RL"
type: source
arxiv_id: 2501.12948v1
authors: DeepSeek-AI
year: 2025
tags: [reasoning, rl, grpo, distillation]
sources:
  - docs/Assets/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).pdf
  - docs/Literature Notes/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).md
updated: 2026-04-19
status: current
---

# DeepSeek-R1

**Incentivises long-chain reasoning via large-scale RL on verifiable tasks —
R1-Zero shows reasoning can emerge from RL alone, R1 adds cold-start SFT and
multi-stage training to fix readability and language mixing.**

## What it does
Two-track recipe. R1-Zero: pure RL from a base model, no SFT — rewards
correctness on verifiable problems; reasoning chains lengthen naturally.
R1: cold-start SFT on curated traces → reasoning RL → rejection-sampled SFT
→ alignment RL. Matches o1-1217 on reasoning benchmarks. Distilled into
Qwen/Llama 1.5B–70B.

## Why it matters for this thesis
R1 is the blueprint the pipeline in this repo follows: **SFT foundation → RL
for reasoning**, with verifiable rewards. The distilled Qwen variants directly
justify the choice of [[entities/qwen3-0.6b|Qwen3-0.6B]] as a small-model base.
The R1-Zero result is also philosophically important for
[[topics/reasoning]]: it suggests reasoning is more "unlocked" by RL than
"taught" by SFT, supporting the process-over-outcome framing the thesis
pushes further with [[sources/papers/seed15-thinking]].

## Related

- [[topics/reasoning]]
- [[sources/papers/seed15-thinking]] — explicit process-reward refinement
- [[entities/grpo]]
- [[entities/qwen3-0.6b]]
- [[entities/constitution]] — the SFT layer that feeds RL here

## Sources

- `docs/Assets/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).pdf`
- `docs/Literature Notes/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).md`
