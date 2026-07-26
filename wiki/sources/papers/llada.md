---
title: "Large Language Diffusion Models (LLaDA)"
type: source
tags: [diffusion, foundations, reasoning]
sources:
  - https://arxiv.org/abs/2502.09992
updated: 2026-07-22
status: current
---

# Large Language Diffusion Models (LLaDA)

**Core LLM capabilities — scalability, in-context learning, instruction-following — are properties of generative modelling, not of autoregression specifically: a masked-diffusion (non-autoregressive) model, LLaDA 8B, rivals LLaMA3 8B across standard benchmarks and beats GPT-4o on reversal-reasoning tasks.**

## Summary

Nie et al. (Renmin University / Ant Group, 2025) train a Transformer by masked diffusion — masking each token independently with probability `t∈[0,1]`, then predicting masked tokens with a `1/t`-weighted cross-entropy that gives a principled maximum-likelihood (NLL upper-bound) objective — generating text by iterative unmasking rather than left-to-right decoding. LLaDA 8B (2.3T tokens, SFT only) matches LLaMA3 8B on many benchmarks and is nearly symmetric on the "reversal curse" (poem completion 51.8 forward / 45.6 reversal, beating GPT-4o's 34.3 reversal). It decouples "LLM capabilities" from "autoregression" — a genuinely important paradigm probe. But for an on-device thesis the headline is double-edged: the bidirectionality that fixes reversal comes with multi-step, KV-cache-unfriendly inference. This is a contrast case, not a drop-in alternative to the AR base.

## Why it matters here

The canonical citation for "a non-AR LLM that rivals AR models" — a concrete foil to the project's autoregressive small base, useful for a background section motivating why the AR paradigm was chosen (or where diffusion could fit). Cite for the argument that some AR failure modes (the reversal curse) are *architectural*, and that bidirectional modelling can improve certain reasoning. Background/contrast material, not load-bearing to an AR-based pipeline.

## Method

- **Forward:** mask each token independently with probability `t` sampled uniformly from `[0,1]` (a proper generative model, unlike BERT's fixed ratio).
- **Objective:** cross-entropy on masked tokens weighted by `1/t` — the theoretical link to maximum likelihood.
- **Architecture/training:** Transformer mask-predictor with *no causal mask*; 2.3T tokens (~0.13M H800-hours), 1B and 8B, SFT on 4.5M pairs, no RL.
- **Inference:** reverse process from fully masked to unmasked with low-confidence remasking; supports diffusion / AR / block-diffusion sampling without retraining.

## Key results

- **Pre-trained 8B:** GSM8K (4-shot) 70.3 vs LLaMA3 8B 48.7 (Qwen2 7B leads at 80.2); MMLU 65.9 ≈ LLaMA3 65.4; beats LLaMA2 7B on nearly all 15 tasks.
- **Reversal curse:** LLaDA 51.8/45.6 (forward/reversal) — near-symmetric, beats GPT-4o (82.7/34.3) on the reversal direction.
- Reached with only SFT (no RL) and less data (2.3T vs LLaMA3's ~15T) — favourable but not apples-to-apples.

## Critical appraisal

A genuinely important paradigm probe. But the strongest claims are narrow: Qwen2/2.5 7B clearly outscore LLaDA on GSM8K/Math/HumanEval, so "rivals AR" holds against LLaMA-class, not the best AR peers; the reversal-curse win is on a somewhat bespoke task; and absolute post-SFT scores lag partly because there is no RL stage.

> ⚠ On-device caution (directly relevant): iterative unmasking *without KV-caching* is a real deployment obstacle — diffusion's trust/reasoning upsides currently trade against inference efficiency, exactly the wrong trade for a small on-device model today. The smallest LLaDA is 1B (headline 8B), so there is no 0.6B diffusion result to lean on.

## Related

- [[sources/papers/diffusion-of-thoughts]] — diffusion-LM chain-of-thought
- [[sources/papers/ladir]] — latent diffusion reasoning
- [[sources/papers/planner-executor-diffusion]] — diffusion planner + AR executor
- [[sources/papers/coconut-continuous-latent]] — non-token reasoning paradigms
- [[sources/papers/attention-is-all-you-need]] — the AR Transformer this contrasts with
- [[topics/llm-foundations]] — the autoregression-vs-diffusion design space
- [[topics/reasoning]] — architectural failure modes (reversal curse)

## Sources

- Nie, Zhu, You, Zhang, Ou, Hu, Zhou, Lin, Wen, Li (2025) — arXiv:2502.09992 — [arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)
