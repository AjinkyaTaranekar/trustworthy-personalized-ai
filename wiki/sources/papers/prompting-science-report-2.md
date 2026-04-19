---
title: "Prompting Science Report 2: The Decreasing Value of CoT"
type: source
arxiv_id: 2506.07142v1
authors: Meincke, Mollick, Mollick, Shapiro (Wharton)
year: 2025
tags: [prompting, cot, evaluation, caveat]
sources:
  - docs/Assets/Prompting Science Report 2 The Decreasing Value of Chain of Thought in Prompting (2506.07142v1).pdf
  - docs/Literature Notes/Prompting Science Report 2 The Decreasing Value of Chain of Thought in Prompting (2506.07142v1).md
updated: 2026-04-19
status: current
---

# Prompting Science Report 2

**CoT prompting is *not* universally helpful. For non-reasoning models it
adds small average gains with higher variance; for reasoning-tuned models
it adds tokens with near-zero accuracy gain. Many modern models already
CoT-reason implicitly.**

## What it does
Empirical report from Wharton/Mollick. Tests CoT across task types and
model tiers. Documents the flip from "CoT helps everywhere" to "CoT is
sometimes harmful and always expensive."

## Why it matters for this thesis
Important counterweight to the Batch 1 CoT-positive narrative. The repo's
SFT v2 constitution explicitly formats `<think>` blocks — but this paper
suggests that on Qwen3-0.6B (a small reasoning-trained model) the format
itself may already be baked in. Design question: does the SFT v2
constitution improve over the base Qwen's implicit CoT, or just regularise
its style? The Condition A vs Condition B ablation answers this directly.

## Related

- [[topics/reasoning]]
- [[sources/papers/chain-of-thought-prompting]]
- [[sources/papers/dual-head-reasoning-distillation]]
- [[entities/constitution]]

## Sources

- `docs/Assets/Prompting Science Report 2 The Decreasing Value of Chain of Thought in Prompting (2506.07142v1).pdf`
- `docs/Literature Notes/Prompting Science Report 2 The Decreasing Value of Chain of Thought in Prompting (2506.07142v1).md`
