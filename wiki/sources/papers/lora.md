---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
type: source
tags: [sft, lora, training, small-model]
sources:
  - https://arxiv.org/abs/2106.09685
updated: 2026-07-18
status: current
---

# LoRA: Low-Rank Adaptation of Large Language Models

**Freeze the pre-trained weights and learn only a low-rank update ΔW = BA per weight matrix — cutting trainable parameters by up to ~10,000× and GPU memory by ~3×, with no added inference latency and accuracy matching or beating full fine-tuning.**

## Summary

Hu et al. (Microsoft, ICLR 2022) make parameter-efficient fine-tuning the default for the LLM era. Building on the observation that the *intrinsic dimension* of adaptation is low, they hypothesise the weight *update* itself is low-rank and capture it with a small factored matrix `ΔW = BA` (`B` zero-initialised, `A` Gaussian, output scaled by `α/r`) placed on the attention query/value projections while the backbone stays frozen. On GPT-3 175B, ~4.7M trainable params (≈0.003%) meet or exceed full fine-tuning, and because `BA` merges into `W0` at deployment there is zero inference-latency penalty — unlike adapters (+20–30% latency) or prefix-tuning (which eats context). This is the mechanism that makes single-GPU SFT of the project's 0.6B model feasible.

## Why it matters here

Directly load-bearing for the pipeline: a trustworthy on-device ~0.6B constitutional model is fine-tuned under tight compute, and LoRA is what makes SFT (and any RL) practical while keeping checkpoints MB-scale. It also enables the project's comparisons cleanly — one frozen Qwen3-0.6B backbone with **swappable adapters** maps onto the constitution/template/thinker-executor variants, allowing cheap A/B at inference. Merge-to-base gives zero-latency deployment, which matters for an edge target.

## Method

- **Reparameterisation:** `h = W0·x + BA·x`; only `A (r×k)`, `B (d×r)` trained, `r ≪ min(d,k)`. `B=0` at start so training begins exactly at the pre-trained function; scale by `α/r` so `r` can change without re-tuning the LR.
- **Placement:** on `Wq`/`Wv` (MLP/LayerNorm frozen for efficiency).
- **Zero-latency:** `BA` folds into `W0` before serving; task switching subtracts/adds adapter pairs.

## Key results

- **RoBERTa-large GLUE:** 89.0 (0.8M params) vs 88.9 full fine-tuning. **DeBERTa-XXL:** 91.3 vs 91.1.
- **GPT-2 Medium E2E:** 70.4 BLEU vs 68.2. **GPT-3 175B:** MNLI-m 91.7 vs 89.5; WikiSQL 73.4 vs 73.8 — with ~4.7M params.
- **Efficiency:** ~10,000× fewer trainable params; checkpoint ~350 GB → ~35 MB; ~3× less GPU memory, ~25% faster training.
- **Rank ablation:** competitive at **r=1**, optimal ~r=2–4, negligible gains to r=64 — the update is genuinely low-rank. Spreading a fixed budget across `Wq`+`Wv` beats concentrating on one.

## Critical appraisal

Simple, theoretically motivated, strong across families/scales, and immediately practical — the reason single-GPU fine-tuning of large models became routine. Numbers are credible and reproduced thousands of times. Scope: adaptation quality/efficiency only — it says nothing about alignment, safety, or data quality.

> ⚠ Caution: the low-rank evidence comes from large models on narrow tasks. At 0.6B with broad constitutional/instruction data, rank may need to be higher and adapting MLP may matter, so the pipeline should treat rank as a *tuned* hyperparameter rather than assume r=4. LoRA supports the *feasibility* half of the thesis (you can align a small model cheaply); it makes no claim about whether the behaviour is trustworthy — that is the dissertation's contribution.

## Related

- [[sources/papers/flan]] — instruction-tuning data that LoRA would fine-tune on
- [[sources/papers/instructgpt]] — the SFT stage LoRA makes cheap
- [[sources/papers/reducing-safety-tax]] — note: OPSA there uses *full-parameter* tuning; contrast with PEFT
- [[entities/qwen3-0.6b]] — the base model LoRA adapts in the pipeline
- [[sources/code/training-and-benchmark]] — where LoRA SFT is applied
- [[topics/reasoning]] — training-time concerns for trustworthy reasoning

## Sources

- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen (2021/2022) — arXiv:2106.09685 (ICLR 2022) — [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
