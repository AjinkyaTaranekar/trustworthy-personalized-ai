---
title: "Self-Enhanced Reasoning Training (SERT)"
type: source
arxiv_id: 2502.12744v1
authors: Zhang et al.
year: 2025
venue: ICASSP 2025
tags: [reasoning, distillation, small-model, self-training]
sources:
  - docs/Assets/Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1).pdf
  - docs/Literature Notes/Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1).md
updated: 2026-04-19
status: current
---

# SERT — Self-Enhanced Reasoning Training

**Small models (GPT-2 scale) can already generate high-quality reasoning paths during sampling — those paths are just low-probability. SERT filters and self-trains on them, activating latent reasoning without needing a stronger teacher.**

## What it does
Zero-shot sampling produces latent good-reasoning paths; SERT selects the correct ones and fine-tunes the student on its own output. Improves reasoning distillation from GPT-3.5 teacher to GPT-2 student.

## Why it matters for this thesis
Directly supports the repo's small-model bet on [[entities/qwen3-0.6b|Qwen3-0.6B]]. If reasoning is **already latent** in a small model, the SFT+GRPO pipeline is essentially an unlocking procedure — reinforcing [[sources/papers/hidden-reasoners|Hidden Reasoners]]'s claim. Also suggests a cheap data-augmentation step: rejection-sample the base model's own reasoning before involving a teacher.

## Related

- [[topics/reasoning]]
- [[sources/papers/hidden-reasoners]]
- [[sources/papers/deepseek-r1]]
- [[entities/qwen3-0.6b]]

## Sources

- `docs/Assets/Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1).pdf`
- `docs/Literature Notes/Self-Enhanced Reasoning Training Activating Latent Reasoning in Small Models for Enhanced Reasoning Distillation (2502.12744v1).md`
