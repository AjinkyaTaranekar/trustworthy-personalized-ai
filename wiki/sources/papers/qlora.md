---
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
type: source
tags: [sft, lora, quantisation, training, small-model]
sources:
  - https://arxiv.org/abs/2305.14314
updated: 2026-07-18
status: current
---

# QLoRA: Efficient Finetuning of Quantized LLMs

**Finetune a frozen 4-bit-quantised LLM by backpropagating through it into small 16-bit LoRA adapters — cutting the memory to finetune a 65B model from >780 GB to <48 GB (single GPU) with no measurable loss versus 16-bit full finetuning.**

## Summary

Dettmers et al. (UW, NeurIPS 2023) make instruction-tuning of large models a single-GPU operation via three ideas: **4-bit NormalFloat (NF4)** storage (information-theoretically near-optimal for the roughly-normal distribution of pretrained weights), **Double Quantisation** of the quantisation constants (−0.373 bits/param, ~3 GB on 65B), and **Paged Optimizers** to survive memory spikes. With LoRA on *all* linear layers, 4-bit base + 16-bit adapters matches 16-bit full finetuning (NF4+DQ MMLU 53.1% vs BFloat16 53.0%). The resulting Guanaco-65B reaches 99.3% of ChatGPT on the Vicuna benchmark. For this project, QLoRA is the practical training enabler — but note it cuts *training* memory, not *inference* latency.

## Why it matters here

QLoRA is what makes it cheap and reproducible to SFT a 0.6B constitutional student on accessible hardware — at that scale the footprint is trivial (7B already fits in ~5–6 GB), so full multi-experiment ablation ladders are affordable. Its "data quality beats quantity" finding (9k OASST1 > a 450k FLAN-v2 subsample) is strong justification for a small, carefully-curated constitutional dataset over a large scraped one. Complements [[sources/papers/lora]] (the adapter it quantises) and the on-device serving story carried by [[sources/papers/mobillama]] and [[sources/papers/llm-inference-edge]].

## Method

- **NF4:** a 4-bit type built from the quantiles of `N(0,1)`, with an exact zero; beats FP4/Int4 at 4 bits (Pile perplexity 27.41 vs FP4 29.48 vs Int4 34.34).
- **Double Quantisation:** quantise the first-level FP32 scaling constants to 8-bit (blocksize 256) — 0.5 → 0.127 bits/param overhead.
- **Paged Optimizers:** NVIDIA unified memory pages optimiser states to CPU on spikes, preventing OOM.
- **LoRA on all layers** (rank 64, α 16) — required to match full finetuning; NF4 weights dequantise to BFloat16 only for the matmul.

## Key results

- **Memory:** 65B QLoRA <48 GB / 24 h (vs >780 GB full 16-bit); 33B 21 GB; 7B ~5–6 GB. Study spans 1,000+ models, 80M–65B.
- **Quality:** NF4+DQ 53.1% MMLU ≈ BFloat16 53.0%. Guanaco-65B 99.3% of ChatGPT (Vicuna); human-rater Elo 1023 (above ChatGPT's 916).
- **Data quality > quantity:** 9k OASST1 beats a 450k FLAN-v2 subsample; dataset size moved MMLU only 0.0–0.5.

## Critical appraisal

Rigorous, large-scale, with a genuinely novel data type; the memory numbers are dramatic and independently reproduced across the bitsandbytes/PEFT ecosystem. Cautions: the "beats ChatGPT" framing rests on a noisy GPT-4/human judge (κ=0.25) over a small Vicuna set; Guanaco is not safety-tuned; most tables are single-run.

> ⚠ Conflict / caution: QLoRA proves cheap *training*, not cheap *serving* — the 4-bit-then-dequant-to-bf16 compute path adds inference latency and is not itself an on-device speed win. The on-device *inference* argument must be carried separately by genuine Int4/GGUF serving (see the MobiLlama and edge-inference numbers).

## Related

- [[sources/papers/lora]] — the low-rank adapter QLoRA quantises the base around
- [[sources/papers/mobillama]] — genuine on-device 4-bit *inference* (the serving side)
- [[sources/papers/llm-inference-edge]] — sustained-load edge inference numbers
- [[sources/papers/flan]] — the FLAN-v2 data QLoRA shows is beaten by smaller curated sets
- [[entities/qwen3-0.6b]] — the small student QLoRA would finetune
- [[sources/code/training-and-benchmark]] — the pipeline's SFT setup

## Sources

- Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023) — arXiv:2305.14314 (NeurIPS 2023) — [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
