---
title: Hierarchical Reasoning Model (HRM)
type: source
arxiv_id: 2506.21734v3
authors: Wang et al.
year: 2025
tags: [reasoning, architecture, hierarchical, small-model]
sources:
  - docs/Assets/Hierarchical Reasoning Model (2506.21734v3).pdf
  - docs/Literature Notes/Hierarchical Reasoning Model (2506.21734v3).md
updated: 2026-04-19
status: current
---

# Hierarchical Reasoning Model (HRM)

**A 27M-parameter recurrent architecture with a high-level "slow planner" and low-level "fast executor" module that solves complex Sudoku, mazes, and ARC with only 1000 training samples — no pretraining, no CoT data.**

## What it does
Two interdependent recurrent modules process at different timescales (brain- inspired). Runs sequential reasoning in a single forward pass without supervising the intermediate process. Outperforms much larger long-context models on ARC.

## Why it matters for this thesis
HRM is the strongest existence proof that **reasoning is an architectural problem, not a scale problem** — a claim central to [[topics/reasoning]]. It also maps directly onto the dissertation's "human brain frequency modes" motif: slow abstract planning, fast rapid computation. A potential alternative lens to process-reward RL: change the substrate instead of the training signal.

## Related

- [[topics/reasoning]] · [[topics/llm-foundations]]
- [[sources/papers/looped-transformers-reasoning]] — sibling depth-over-params claim
- [[sources/papers/coconut-continuous-latent]] — latent reasoning family

## Sources

- `docs/Assets/Hierarchical Reasoning Model (2506.21734v3).pdf`
- `docs/Literature Notes/Hierarchical Reasoning Model (2506.21734v3).md`
