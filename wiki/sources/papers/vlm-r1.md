---
title: "VLM-R1: Stable R1-style RL for Vision-Language Models"
type: source
arxiv_id: 2504.07615v2
authors: Shen et al.
year: 2025
tags: [rl, grpo, multimodal, vision-language]
sources:
  - docs/Assets/VLM-R1 A Stable and Generalizable R1-style Large Vision-Language Model (2504.07615v2).pdf
  - docs/Literature Notes/VLM-R1 A Stable and Generalizable R1-style Large Vision-Language Model (2504.07615v2).md
updated: 2026-04-19
status: current
---

# VLM-R1

**Brings R1-style rule-based RL to vision-language models. Shows RL beats
SFT on generalisation; documents reward hacking in object detection, an
"OD aha moment", and scaling behaviour across model sizes.**

## What it does
RL framework for VLMs on tasks with deterministic ground truth. Open-source
codebase + ablations on data quality and scaling.

## Why it matters for this thesis
Reinforces the **rule-based reward pattern** the repo adopts — now with
evidence it works across modalities. The documented reward-hacking in OD is
important cautionary reading: any rule-based behavioural reward
(tool_discipline, refusal_honesty) has an analogous hacking surface. Plan
the ablation to watch for gamed outputs.

## Related

- [[topics/reasoning]] · [[topics/tool-use-and-verification]]
- [[entities/grpo]]
- [[sources/papers/ui-r1]]
- [[sources/papers/search-r1]]

## Sources

- `docs/Assets/VLM-R1 A Stable and Generalizable R1-style Large Vision-Language Model (2504.07615v2).pdf`
- `docs/Literature Notes/VLM-R1 A Stable and Generalizable R1-style Large Vision-Language Model (2504.07615v2).md`
