---
title: "MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT"
type: source
tags: [small-model, on-device, foundations]
sources:
  - https://arxiv.org/abs/2402.16840
updated: 2026-07-18
status: current
---

# MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT

**A 0.5B/0.8B small language model built by sharing a single FFN block across all transformer layers matches or beats same-class baselines while cutting parameters ~60% and training GPU-hours 42% — and it is fully transparent (open weights, code, and a fully itemised open pre-training corpus), running on a real phone SoC at ~7 tokens/s in under 1 GB.**

## Summary

Thawakar et al. (MBZUAI / LLM360, 2024) target two problems at once: naive SLM scaling wastes the parameter budget on redundant per-layer FFNs (which are ~65% of trainable params), and most "open" models hide their training data. MobiLlama shares one MLP block across all layers, freeing budget for more depth/width, and releases the full stack — weights, code, checkpoints, and the itemised 1.2T-token Amber corpus. On a mid-range Snapdragon-685 the 4-bit 0.5B runs at 7.021 tok/s using ~770 MB. This is direct feasibility evidence for the "sub-1B model on a phone for privacy" thesis, and its transparency angle maps onto the trustworthy pillar.

## Why it matters here

Concrete support that a 0.5B model, 4-bit-quantised, fits in <1 GB and runs on a real phone — the exact claim underpinning the on-device privacy argument, with citable phone numbers (Snapdragon-685: 7.021 tok/s, 770 MB, 5.32 mAh/1k tokens) architecturally adjacent to a [[entities/qwen3-0.6b|Qwen3-0.6B]] student. The **full transparency** (open data) is a precondition for a trustworthy assistant and a citable exemplar that transparency and on-device efficiency are jointly achievable.

## Method

- **Shared-FFN parameter sharing:** one MLP block reused across all layers (FFNs are ~65% of params), −60% parameters vs a same-config large base, enabling more layers/width at the same cost.
- **0.5B:** hidden 2048, MLP 5632, 32 heads, 22 layers, context 2048. **0.8B:** hidden 2532, MLP 11,080.
- **Data — Amber (fully open, 1.2T tokens):** RefinedWeb 665B, StarCoder 292B, C4 198B, arXiv 30B, Books 29B, Wikipedia 24B, StackExchange 22B.

## Key results

- **Accuracy (9-benchmark avg):** 0.5B 46.00, 0.8B 46.67; 0.5B beats Pythia-410m by ~2.4% (HellaSwag 52.52 vs 40.85, PIQA 72.03 vs 67.19). For context, ~1.1–1.2B models score higher (TinyLlama 48.74) — the claim is efficiency at the sub-1B tier, not beating 2× models.
- **On-device (0.5B):** Snapdragon-685 (4-bit GGUF) 7.021 tok/s / 770 MB / 5.32 mAh per 1k tokens; laptop i7 CPU 36.32 tok/s / 799 MB; RTX 2080Ti 63.38 tok/s.
- **Training:** 0.5B in 7 days / 26.6K GPU-h (vs 12 days / 46.1K for the unshared 1.2B) — 42% fewer GPU-hours.

## Critical appraisal

A clean, well-motivated architectural idea backed by real edge measurements and — rare — genuinely open data/checkpoints, making it reproducible and auditable. Cautions: sub-1B accuracy is modest (MMLU ~26.45, near chance on knowledge); aggressive FFN sharing likely constrains capacity (not deeply probed).

> ⚠ Caution: ~7 tok/s is usable but slow for interactive chat, especially for a constitutional model emitting extra "thinking" tokens; and it is a **single-shot peak** with no thermal/sustained-load analysis — which [[sources/papers/llm-inference-edge]] shows can fall 15–41% under throttling. Budget latency against the sustained rate, not this peak.

## Related

- [[sources/papers/llm-inference-edge]] — the sustained-load/thermal counterweight to this single-shot number
- [[sources/papers/qlora]] — the cheap-training complement to this cheap-serving evidence
- [[entities/qwen3-0.6b]] — the comparably-sized student model
- [[entities/tml-interaction-small]] — frontier-scale contrast to the on-device small-model case
- [[topics/security-and-privacy]] — on-device deployment as the privacy guarantee
- [[topics/llm-foundations]] — SLM architecture and the FFN parameter budget

## Sources

- Thawakar, Vayani, Khan, Cholakkal, Anwer, Felsberg, Baldwin, Xing, Khan (2024) — arXiv:2402.16840 — [arxiv.org/abs/2402.16840](https://arxiv.org/abs/2402.16840)
