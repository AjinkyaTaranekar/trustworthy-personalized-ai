---
title: "Planner and Executor: Collaboration between Discrete Diffusion and Autoregressive Models in Reasoning"
type: source
tags: [reasoning, architecture, diffusion, latent]
sources:
  - https://arxiv.org/abs/2510.15244
updated: 2026-07-19
status: current
---

# Planner and Executor: Diffusion Planner + Autoregressive Executor

**A discrete diffusion language model acts as a global planner and an autoregressive model as the executor; the planner produces a short plan — in text, or in latent space via a learned projection into the executor's embedding space — that the executor consumes, and a tiny 64-token latent plan plus ~5 executor tokens can beat much larger reasoning models while using ~44× fewer tokens.**

## Summary

Berrayana et al. (EPFL / MBZUAI / ETH / MSRA, 2025) separate *global planning* (bidirectional, parallel — diffusion's strength) from *local execution* (sequential, precise — autoregression's strength). A diffusion planner (LLaDA-8B / Dream-7B) emits a plan that an AR executor (Qwen2.5/Llama 3B–8B) consumes, over either a text channel or a latent channel bridged by a small Linear-GELU-Linear projector (both models frozen, only the projector trained on 35k samples). Latent plans dramatically outperform text plans on hard reasoning (DART-5 54.0% vs 27.0%; AIME24 14.0% vs 0.0%), and ~69 total tokens beat Qwen3/DeepSeek-R1 (which averages 3,068 tokens on DART-5) — a ~44× token reduction. This is the closest published analogue to a thinker/executor split, with a sharp trust-vs-performance tension baked in.

## Why it matters here

It empirically studies exactly the design axis of a thinker/executor split — who plans, who executes, and how the plan is passed — and shows a decomposition can beat a monolith while being far more token-efficient (relevant to on-device budgets). The **text-vs-latent plan channel** is a concrete decision the project's thinker/executor must make, and it surfaces the core tension: latent is more accurate but destroys interpretability, so for a *trustworthy, auditable* on-device agent the text channel is likely the right trade — a contrast worth citing explicitly.

## Method

- **Text-space collaboration:** the diffusion planner emits an explicit textual plan appended to the executor's prompt (interpretable, costs plan tokens).
- **Latent-space collaboration:** the planner emits a latent plan; a learned **Linear-GELU-Linear** projector maps diffusion states into the AR executor's embedding space (both large models frozen — only the projector trained, 35k samples, plan lengths 64/128/256).
- **Design space:** four pairings (ARM→ARM, ARM→DDLM, DDLM→ARM, DDLM→DDLM) × text/latent channels; benchmarks ARC, DART 1–5, AIME24.

## Key results

- **Latent > text on hard tasks:** DART-5 54.0% vs 27.0%; AIME24 14.0% vs 0.0%.
- **Token efficiency:** ~64-token plan + ~5 executor tokens (~69 total) beats Qwen3/R1 on DART-5 and AIME24 despite ~44× more tokens (R1 avg 3,068 on DART-5).
- **Cheap coupling:** only a small projector is trained; a 64-token plan suffices — most of the reasoning "work" is global planning that compresses well.

## Critical appraisal

The most directly *architectural* study of a thinker/executor split, with a striking token-efficiency result and a valuable conceptual takeaway (separate global planning from local execution; consider a non-text plan channel). Big tensions for this thesis: the best-performing channel is the least interpretable; benchmarks are pure reasoning (ARC/DART/AIME) with **no tool-use**; the wins are task-specific (DART-5, AIME), not universal.

> ⚠ 0.6B caution: components are 3–8B (diffusion planners 7–8B), so running a diffusion planner + AR executor is memory-heavy on-device, and "44× fewer *tokens*" is not "44× cheaper on a phone" (no wall-clock/memory comparison). A 0.6B single-model thinker-executor may be a more realistic on-device target, using this paper as motivation for the *split* — with the **text (interpretable) plan channel** chosen over the latent one for trust.

## Related

- [[experiments/thinker-executor-experiment]] — the project's thinker/executor split this analogises
- [[sources/papers/coconut-continuous-latent]] — reasoning in latent/vector space
- [[sources/papers/diffusion-of-thoughts]] — diffusion-LM chain-of-thought
- [[sources/papers/ladir]] — latent diffusion reasoning
- [[topics/reasoning]] — global-plan vs local-execute decomposition
- [[topics/explainability]] — text-plan auditability vs latent-plan opacity

## Sources

- Berrayana, Heakl, Sohail, Hofmann, Khan, Chen (2025) — arXiv:2510.15244 — [arxiv.org/abs/2510.15244](https://arxiv.org/abs/2510.15244)
