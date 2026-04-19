---
title: "UI-R1: RL for GUI-Agent Action Prediction"
type: source
arxiv_id: 2503.21620v5
authors: Lu et al.
year: 2025
tags: [rl, grpo, agents, multimodal, gui]
sources:
  - docs/Assets/UI-R1 Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning (2503.21620v5).pdf
  - docs/Literature Notes/UI-R1 Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning (2503.21620v5).md
updated: 2026-04-19
status: current
---

# UI-R1

**Extends rule-based RL (R1-style) to multimodal GUI agents via a novel
rule-based *action* reward; 3B model beats much larger 7B SFT baselines
across ScreenSpot, ScreenSpot-Pro, ANDROIDCONTROL with only 136 curated
tasks.**

## What it does
Demonstrates [[entities/grpo|GRPO]] works outside text-only reasoning.
Rule-based rewards for discrete action types (click, type, scroll…)
generalise to out-of-domain tasks.

## Why it matters for this thesis
Out-of-thesis-scope for direct use, but extremely relevant as **evidence
that rule-based behavioural rewards generalise** — i.e. the repo's
tool_discipline reward (also rule-based, also action-type-aware) is part of
a working recipe. Also shows Qwen2.5-VL-3B is strong for multimodal RL,
hinting at future extensions beyond the current text-only pipeline.

## Related

- [[topics/tool-use-and-verification]] · [[topics/reasoning]]
- [[entities/grpo]]
- [[sources/papers/vlm-r1]]
- [[sources/papers/search-r1]]

## Sources

- `docs/Assets/UI-R1 Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning (2503.21620v5).pdf`
- `docs/Literature Notes/UI-R1 Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning (2503.21620v5).md`
