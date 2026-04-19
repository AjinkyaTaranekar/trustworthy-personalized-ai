---
title: "LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning"
type: source
arxiv_id: 2510.04573v3
authors: Kang et al.
year: 2025
tags: [reasoning, diffusion, latent, refinement]
sources:
  - docs/Assets/LaDiR Latent Diffusion Enhances LLMs for Text Reasoning (2510.04573v3).pdf
  - docs/Literature Notes/LaDiR Latent Diffusion Enhances LLMs for Text Reasoning (2510.04573v3).md
updated: 2026-04-19
status: current
---

# LaDiR

**Encodes reasoning steps into VAE latent blocks and runs a latent diffusion
model over them with bidirectional attention, enabling iterative refinement
of whole reasoning trajectories — parallel diverse paths, holistic revision.**

## What it does
VAE builds a structured latent reasoning space; latent diffusion denoises
blocks of thought tokens; adaptive test-time compute. Gains on maths +
planning benchmarks vs autoregressive, diffusion, and latent-reasoning
baselines.

## Why it matters for this thesis
Combines the themes of [[sources/papers/coconut-continuous-latent|Coconut]]
(latent) and [[sources/papers/diffusion-of-thoughts|DoT]] (diffusion).
Relevant as the ceiling reference for reasoning quality when scrutability
is abandoned — useful as a comparison point when arguing that the
ontology-verification direction
([[decisions/2025-11-10-ontology-focus-shift]]) accepts a performance gap
in exchange for trust.

## Related

- [[topics/reasoning]]
- [[sources/papers/coconut-continuous-latent]]
- [[sources/papers/diffusion-of-thoughts]]

## Sources

- `docs/Assets/LaDiR Latent Diffusion Enhances LLMs for Text Reasoning (2510.04573v3).pdf`
- `docs/Literature Notes/LaDiR Latent Diffusion Enhances LLMs for Text Reasoning (2510.04573v3).md`
