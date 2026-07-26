---
title: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"
type: source
tags: [small-model, on-device, foundations]
sources:
  - https://arxiv.org/abs/2404.14219
updated: 2026-07-20
status: current
---

# Phi-3 Technical Report

**A 3.8B model (phi-3-mini) trained on a heavily filtered, "data-optimal" mix of web and synthetic data rivals GPT-3.5 / Mixtral 8x7B on standard benchmarks while being small enough to run locally on a phone — 4-bit quantised, ≈1.8 GB, >12 tokens/second on an iPhone 14.**

## Summary

Abdin et al. (Microsoft, 2024) argue recent gains are driven largely by *data quality*, not only scale: filtering web data by "educational level" and adding curated synthetic data lets a small model absorb far more reasoning per parameter. phi-3-mini (3.8B, 3.3T tokens, Llama-2-compatible tokenizer/blocks) hits MMLU 68.8% and MT-Bench 8.38 — GPT-3.5-class — and at 4-bit occupies ≈1.8 GB and runs >12 tok/s natively/offline on an A16 Bionic. The deliberate cost is factual recall (TriviaQA 64.0%), since low-value facts are stripped to preserve reasoning capacity. This is the anchor citation for "capable small models can run on-device", and its data-quality recipe and factual-weakness caveat both bear directly on a 0.6B target.

## Why it matters here

Direct precedent and a headline number for on-phone feasibility — a 3.8B model at 4-bit fits in ≈1.8 GB at >12 tok/s on consumer silicon, and the project's 0.6B target is ~6× smaller (more comfortable on memory/throughput). The "data-optimal regime" (educational-quality filtering + synthetic data, stripping low-value facts) is a template for squeezing capability into a 0.6B budget. But it also *warns* a 0.6B will be even weaker on factual recall — arguing for retrieval/grounding and a clarify-before-assert trust design over parametric knowledge.

## Key results

- **On-device:** phi-3-mini 4-bit ≈1.8 GB, >12 tok/s on iPhone 14 (A16), fully offline.
- **Benchmarks (mini):** MMLU 68.8% (GPT-3.5 71.4%), MT-Bench 8.38 (GPT-3.5 8.35), GSM8K 82.5%, HumanEval 58.5%, MATH 41.3%; TriviaQA 64.0% (weak — by design).
- Size ladder (small 7B, medium 14B) + later phi-3.5 (MoE ≈42B/6.6B-active ≈ GPT-4o-mini; Vision 4.2B). Safety post-training (SFT+DPO, RAI harm-category eval).

## Critical appraisal

The strongest reusable claim is quantitative and concrete (3.8B → 4-bit → ≈1.8 GB → >12 tok/s on a phone). Cautions: the on-phone number is a single-device feasibility demo with **no energy, thermal, or sustained-throughput data**; the data-quality recipe is directionally important but proprietary/under-specified; MMLU-vs-GPT-3.5 parity is on static benchmarks sensitive to contamination. The honest, important caveat is the factual-recall weakness — shrinking the model shifts the burden onto reasoning + external knowledge, which matters for any trust/empathy app that must not hallucinate.

> ⚠ For a 0.6B thesis: supports feasibility but don't over-claim latency (no sustained-load data here — pair with the edge-inference throttling evidence); expect even weaker parametric factual recall, so lean on grounding and honest abstention.

## Related

- [[sources/papers/mobillama]] — 0.5B on-phone at 4-bit; the sub-1B counterpart
- [[sources/papers/llm-inference-edge]] — the sustained-load/thermal counterweight to the single-shot number
- [[sources/papers/qlora]] — 4-bit as the on-device operating point; data quality > quantity
- [[sources/papers/qwen3-tr]] — the project's base model family
- [[entities/qwen3-0.6b]] — the ~6× smaller on-device target
- [[topics/security-and-privacy]] — on-device deployment as the privacy mechanism

## Sources

- Abdin et al. (Microsoft, 2024) — arXiv:2404.14219 — [arxiv.org/abs/2404.14219](https://arxiv.org/abs/2404.14219)
