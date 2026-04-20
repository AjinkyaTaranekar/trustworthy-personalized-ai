---
title: "Reasoning with Latent Thoughts: Looped Transformers"
type: source
arxiv_id: 2502.17416v1
authors: Saunshi, Dikkala, Li, Kumar, Reddi (Google)
year: 2025
venue: ICLR 2025
tags: [reasoning, architecture, looped, depth]
sources:
  - docs/Assets/Reasoning with Latent Thoughts On the Power of Looped Transformers (2502.17416v1).pdf
  - docs/Literature Notes/Reasoning with Latent Thoughts On the Power of Looped Transformers (2502.17416v1).md
updated: 2026-04-19
status: current
---

# Looped Transformers for Reasoning

**Depth, not parameters, drives reasoning. A k-layer Transformer looped L times matches a kL-layer model on many reasoning problems — and implicitly generates latent thoughts equivalent to T CoT steps with T loops.**

## What it does
Shows theoretically and empirically that iterative algorithms suit looped architectures. Bridges loop count to CoT step count. Presents a dichotomy: reasoning benefits from depth; memorisation benefits from width/params.

## Why it matters for this thesis
Another "reasoning is architectural" argument ([[sources/papers/hierarchical-reasoning-model|HRM]] is its cousin). For a small model ([[entities/qwen3-0.6b|Qwen3-0.6B]]) this is actionable: loops at inference may recover depth without training a larger model. Also clean theory for the CoT step count → latent compute correspondence.

## Related

- [[topics/reasoning]] · [[topics/llm-foundations]]
- [[sources/papers/hierarchical-reasoning-model]]
- [[sources/papers/coconut-continuous-latent]]
- [[sources/papers/state-stream-transformer]]

## Sources

- `docs/Assets/Reasoning with Latent Thoughts On the Power of Looped Transformers (2502.17416v1).pdf`
- `docs/Literature Notes/Reasoning with Latent Thoughts On the Power of Looped Transformers (2502.17416v1).md`
