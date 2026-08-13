---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
type: source
tags: [foundations, attention, transformers]
sources:
  - https://arxiv.org/abs/2104.09864
updated: 2026-07-22
status: current
---

# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

**Encode token position by rotating the query/key vectors by an angle proportional to their absolute position, so the attention inner product depends only on the content and the relative offset (m−n) — giving relative-position behaviour for free, with a natural long-range decay, and full compatibility with linear attention.**

## Summary

Su et al. (Zhuiyi Technology, 2021) observe that additive relative-position schemes inject position as a bias term inside the attention logits, which breaks the factorisation efficient/linear attention needs. RoPE sidesteps this: split the query/key into 2D blocks and rotate block *i* by angle `m·θ_i` (`θ_i = 10000^(−2(i−1)/d)`); the dot product of two rotated vectors then depends on `m−n` through a block-diagonal rotation — absolute in *application*, relative in *effect*, and norm-preserving so linear attention still works. It also yields an automatic long-range decay (distant tokens interact less). The 2021 experiments are deliberately modest (+0.2 BLEU on WMT En-De; a clear edge only when longer sequences are used); RoPE's importance was confirmed *later* by adoption in LLaMA, Qwen, and most modern decoder-only LLMs. BACKGROUND/infrastructural — the architectural provenance of the project's base model.

## Why it matters here

RoPE is the positional encoding used by **Qwen** — the model family at the centre of the project's small-model pipeline — and by LLaMA/most modern decoder-only LLMs, so it is the correct citation when the background chapter explains *why* the chosen sub-1B model handles its context window and length behaviour. Its length-flexibility and efficiency properties are part of what makes small on-device models viable at usable context lengths. No trust/empathy/personalisation content — cite purely as base-model provenance, ideally in a sentence or two alongside [[sources/papers/t5|T5]] when situating the transformer stack.

## Method

- **Requirement:** find `f_q, f_k` such that `⟨f_q(x_m,m), f_k(x_n,n)⟩ = g(x_m, x_n, m−n)` — attention depends on relative offset only.
- **2D solution:** treat a feature pair as a complex number and multiply by `e^{imθ}`; the conjugate inner product then depends on `e^{i(m−n)θ}`.
- **General d-dim:** a block-diagonal rotation `R^d_{Θ,m}` rotates each of the d/2 coordinate pairs at its own frequency (the sinusoidal frequency schedule), realised with element-wise multiplies + sign-flips, not a matrix multiply.
- **Long-term decay** (summation-by-parts argument) and **linear-attention compatibility** (rotations are orthogonal/norm-preserving) are the two structural wins.

## Key findings

- **WMT14 En-De:** RoFormer 27.5 BLEU vs Transformer-base 27.3 (+0.2, modest).
- **Faster MLM pre-training convergence** vs vanilla BERT (qualitative curve, no exact loss).
- **Chinese long-document (CAIL2019-SCM):** RoFormer-1024 66.07% val / 69.79% test — best when extended to 1024 tokens, illustrating the length-flexibility advantage.

> Note: the digesting pass returned an internally inconsistent GLUE table for this paper, so this page deliberately does *not* quote RoFormer GLUE cell values (the qualitative claim is competitiveness with BERT). Confirm any GLUE numbers against the rendered source before use.

## Critical appraisal

A high-impact idea carried by later adoption more than by its own numbers: the derivation is clean and the norm-preserving / linear-attention property is the genuinely important insight, but the standalone 2021 evidence is thin (+0.2 BLEU, a narrow legal dataset, qualitative loss curves) and the authors admit they "have not come up with a faithful explanation" for the superior long-text behaviour despite proving the decay property. The mechanism is now foundational LLM infrastructure — which is the real reason to read it.

## Related

- [[sources/papers/attention-is-all-you-need]] — the attention mechanism RoPE modifies
- [[sources/papers/qwen3-tr]] — the base family that uses RoPE (with QK-Norm)
- [[sources/papers/t5]] — an alternative (relative-bucket) positional scheme in the same stack
- [[entities/qwen3-0.6b]] — the on-device base whose length behaviour RoPE governs
- [[sources/papers/context-length-hurts]] — length degradation despite positional encoding
- [[topics/llm-foundations]] — attention, positional encoding, context length

## Sources

- Su, Lu, Pan, Murtadha, Wen, Liu (2021) — arXiv:2104.09864 (Neurocomputing 2024) — [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
