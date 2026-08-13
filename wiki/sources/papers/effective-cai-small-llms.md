---
title: "How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers"
type: source
tags: [constitutional-ai, small-model, security, deepseek, evaluation]
sources:
  - https://arxiv.org/abs/2503.17365
updated: 2026-07-18
status: current
---

# How Effective Is Constitutional AI in Small LLMs?

**Constitutional AI's self-critique-and-revise loop can meaningfully improve safety in small (7–9B) uncensored models, but its effectiveness is strongly architecture-dependent — reasoning-capable Llama-family models (including DeepSeek-R1-Distill-Llama-8B) benefit, while Gemma-2 and Qwen2.5 mostly fail at the critique step, unable to detect harm in their own outputs.**

## Summary

Chacón Menke and Tan (2025) run the closest published test of "does CAI work at small scale." They *abliterate* four 7–9B models (removing the single activation direction responsible for refusal) to strip residual RLHF, then apply an inference-time generate → critique → revise loop, so any safety gain is attributable to CAI itself. DeepSeek-R1-Distill-Llama-8B is the standout — eliminating harmful content in several categories with lower variance (±31.60% vs Llama-3.1's ±49.02%) — while Gemma-2-9B and Qwen2.5-7B show limited improvement because harm *detection during critique* breaks down. A recognition-versus-application gap emerges: all models retain high SafetyBench scores (they *know* what is harmful) yet fail to apply that knowledge in open-ended self-critique. For a sub-1B constitutional thesis this is a yellow flag and a design argument at once.

## Why it matters here

This is the empirical case that a tiny model cannot be trusted to self-detect harm unaided — which is exactly why the project uses a written [[entities/constitution|constitution]] plus a teacher/critic scaffold rather than expecting the 0.6B student to police itself. It complements [[sources/papers/constitution-or-collapse]] (CAI at 8B: real ASR reduction at a helpfulness/collapse cost) and the original [[sources/papers/constitutional-ai-bai]], extending the small-model evidence downward. It also foreshadows the [[sources/papers/reducing-safety-tax|safety tax]]: abliteration + intervention cost Qwen2.5 up to −20% on commonsense morality.

## Method

- **Models (all 7–9B):** DeepSeek-R1-Distill-Llama-8B (the only reasoning-distilled one — the key contrast axis), Gemma-2-9B-it, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct.
- **Abliteration** removes refusal behaviour, forcing CAI to do the safety work — but note this is a confound (it may also damage the self-reflective circuitry CAI needs).
- **CAI loop (inference-time, no training):** two generic prompts — a critique prompt ("identify if the answer is harmful… recommend refusing if necessary") and a revise prompt ("rewrite… if the question itself is harmful, respond politely with a refusal"). This is a two-prompt operationalisation, not a full multi-principle constitution.
- **Judging:** a binary harm classifier plus GPT-4o-mini, over 90 HarmBench prompts (15 each across six harm categories). Capability/ethics tracked on MMLU, ETHICS, SafetyBench.

## Key results

- **Safety:** R1-Llama strongest (harm eliminated in several categories); Llama-3.1 strong; Gemma-2 and Qwen2.5 weak — failing at the *critique* step, not the revise step.
- **Failure texture:** Llama-3.1 often adds warnings while keeping harmful content (cosmetic refusal); weak models simply fail to flag harm.
- **Abliteration collateral (Δ vs original):** Llama-family lose ~nothing on MMLU/ethics; Qwen2.5 collapses on ETHICS CommonsenseMoral (**−20.0%**), −3.6% MMLU. All keep high SafetyBench scores — the recognition-vs-application gap.

## Critical appraisal

Clean isolation design and a genuinely useful negative result (CAI is not architecture-agnostic at small scale). But the "constitution" is just two generic prompts, so "CAI fails on Gemma/Qwen" may partly be "this minimal prompt fails"; abliteration confounds "small model can't self-critique" with "abliterated model can't"; the classifier+GPT-4o-mini judge has no reported human calibration; and N=90 with ±31–49% variance yields very wide intervals.

> ⚠ Conflict / caution: the models here are **7–9B and already show a critique bottleneck**, so a 0.6B model is likely to struggle even more with unaided self-critique. Cite this to justify *architecture/scaffold* choices (written constitution + teacher/critic + reasoning-capable base), not as proof CAI works at 0.6B.

## Related

- [[sources/papers/constitution-or-collapse]] — CAI at 8B; ASR reduction, helpfulness cost, model collapse
- [[sources/papers/constitutional-ai-bai]] — the original generate–critique–revise / RLAIF loop
- [[sources/papers/reducing-safety-tax]] — activates latent safety on Qwen3-0.6B without the capability cost
- [[sources/papers/deepseek-r1]] — the reasoning-distillation lineage that helps self-critique here
- [[entities/constitution]] — the project's 23-principle written constitution
- [[topics/security-and-privacy]] — alignment regression and safety framing
- [[topics/constitution-psychological-grounding]] — why principles need external scaffolding

## Sources

- Chacón Menke, Tan (2025) — arXiv:2503.17365 — [arxiv.org/abs/2503.17365](https://arxiv.org/abs/2503.17365)
