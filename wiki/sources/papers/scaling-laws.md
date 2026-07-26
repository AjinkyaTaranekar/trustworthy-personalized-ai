---
title: "Scaling Laws for Neural Language Models"
type: source
tags: [foundations, scaling]
sources:
  - https://arxiv.org/abs/2001.08361
updated: 2026-07-20
status: current
---

# Scaling Laws for Neural Language Models

**Language-model test loss falls as a smooth, predictable power law in three quantities — non-embedding parameters N, dataset size D, and training compute C — holding across many orders of magnitude, and implying that for a fixed compute budget the optimal move is to train very large models on modest data and stop before convergence.**

## Summary

Kaplan et al. (OpenAI / JHU, 2020) reframe "how good will a bigger model be?" as an engineering forecast. Fitting power laws to hundreds of Transformer LMs (up to 1.5B non-embedding params on WebText2), they find `L(N) ∝ N^−0.076`, `L(D) ∝ D^−0.095`, `L(C_min) ∝ C_min^−0.050`, that architecture *shape* (depth vs width) barely matters (~3% at fixed N), that large models are markedly more sample-efficient, and that compute-optimal training puts almost all extra compute into a bigger model (`N ∝ C^0.73`). This is the "capability needs scale" backdrop the sub-1B on-device thesis positions itself against — but its compute-optimal exponents were materially revised by Chinchilla (Hoffmann et al. 2022) toward more, cleaner data, which is the bridge argument the thesis actually uses.

## Why it matters here

BACKGROUND, not load-bearing. Two useful angles: (1) the shape-insensitivity and sample-efficiency findings imply that at a *fixed small N* the remaining lever is data quality and training procedure, not architecture — which is exactly where data-quality and dedup arguments become the real story for small models. (2) The Chinchilla correction (train smaller models on more, cleaner tokens) reframes "scale" away from raw parameter count toward data, giving the sub-1B thesis a scaling-laws-*aware* justification rather than a contrarian one.

## Key results

- **Power laws:** `L(N)=(8.8e13/N)^0.076`, `L(D)=(5.4e13/D)^0.095`, `L(C_min)=(3.1e8/C_min)^0.050` — valid across ~6 orders in N, ~2 in D, ~8 in C.
- **Compute-optimal allocation:** `N ∝ C^0.73`, `B ∝ C^0.24`, `S ∝ C^0.03` — big model, modest data, early stopping (train to ~10% above converged loss).
- **Anti-overfit data rule:** `D ≳ 5e3·N^0.74` (sublinear).
- **Shape near-irrelevant:** ~40× aspect-ratio change → ~3% loss change.

## Critical appraisal

Foundational and unusually influential — it legitimised the scale-first programme and directly motivated GPT-3. Its qualitative claims (smooth power laws, architecture near-irrelevance, sample-efficiency of large models) have held up. Its main empirical weakness — the compute-optimal allocation — was corrected by **Chinchilla**, so the specific exponents should be cited with that caveat. Scope: LM loss only (no downstream/safety metric), monolingual English, one architecture family, ≤1.5B non-embedding params; the constants are tokenisation-dependent and not fundamental.

## Related

- [[sources/papers/gpt3-few-shot]] — the 175B model these laws motivated
- [[sources/papers/qwen3-tr]] — the family whose 0.6B member the thesis uses
- [[sources/papers/lima]] — data quality > quantity, the small-model corollary
- [[sources/papers/qlora]] — "dataset suitability > size" echoes the Chinchilla reframing
- [[entities/qwen3-0.6b]] — the sub-1B target below this frontier
- [[topics/llm-foundations]] — scaling, capability, and their limits

## Sources

- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (2020) — arXiv:2001.08361 — [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
