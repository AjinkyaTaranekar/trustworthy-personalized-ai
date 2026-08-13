---
title: "A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources"
type: source
tags: [on-device, quantisation, qwen]
sources:
  - https://arxiv.org/abs/2505.15030
updated: 2026-07-22
status: current
---

# A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources

**On CPU-only edge hardware, "quantise a bigger model" beats "run a smaller model at high precision" down to a floor of ~3.5 effective bits-per-weight, with 4-bit as the sweet spot (30–50% faster, near-lossless for large models), and the binding constraint shifts from compute (tiny models) to memory bandwidth (>1B models).**

## Summary

Song et al. (Xiamen / CUHK / Huawei, 2026) sweep 11 modern small/mid LLMs (incl. Qwen2.5-0.5B and Qwen3-0.6B) across 5 quantisation levels (2–8 bit) on two CPU platforms via llama.cpp, measuring capability (GSM8K/HellaSwag/MMLU/HumanEval/TruthfulQA) and system cost (memory, power, throughput), and decomposing throughput loss into communication vs computation. The headline "bigger-quantised beats smaller-precise / 4-bit near-lossless" is a *large-model* result — large models barely move under aggressive quantisation while **small models collapse** (Qwen2.5-1.5B GSM8K 60.80 FP16 → 21.46 q2_k). The genuinely useful design rules for a sub-1B target are the ~3.5-bit floor, the 4-bit sweet spot, and the compute-vs-communication crossover. Load-bearing BACKGROUND/support, complementing [[sources/papers/sustainable-edge-inference]] (real Joules) with capability curves and the *why*.

## Why it matters here

It explicitly includes **Qwen2.5-0.5B and Qwen3-0.6B**, so it anchors claims about a 0.6B model's footprint (0.5B q2_k ≈392 MB, q8_0 ≈576 MB) and its quantisation fragility. Hooks: (1) the ~3.5-bit floor and 4-bit sweet spot justify choosing ~4-bit K-quant for the deployed model; (2) the compute-vs-communication finding says a 0.6B model is **compute-bound**, so on-device latency is set by arithmetic throughput, not memory bandwidth — relevant when the harness adds reasoning/tool-call tokens.

## Key results

- **Large-model resilience vs small-model collapse:** Qwen2.5-14B GSM8K moves only ~4.6 pts q5_k→q2_k; 1.5B GSM8K 60.80 → 21.46 (fp16→q2_k); 0.5B GSM8K q8_0 33.28 → q2_k 21.99.
- **Memory:** 0.5B q2_k 392 MB / q8_0 576 MB; 7B q2_k 2411 MB / q8_0 7297 MB.
- **Power:** 7.9–9.5 W range (1.5B).
- **Bottleneck decomposition:** decoding communication impact grows with size (0.5B 52.9% → 3B 68.1%); computation impact larger for 0.5B — the compute/communication crossover.

*(Throughput tokens/s are figure-only and not transcribed to avoid fabrication.)*

## Critical appraisal

The more systematic of the two edge papers on the capability + system axes — the bits-per-weight threshold and communication/computation decomposition are genuinely useful design rules, and coverage of current Qwen2.5/Qwen3/Llama3 families makes it timely. But it meters power (W) + throughput, not true energy (J); it is CPU/llama.cpp-bound (not phone NPUs); the q4_0 7B memory (6910 MB) is flagged anomalous; and throughput/power results are figure-only (weakening speed-claim reproducibility).

> ⚠ Sub-1B caution: the reassuring "near-lossless 4-bit / bigger-is-better" narrative is a *large-model* result — the same tables show small models degrading badly below 4-bit (0.5B/1.5B GSM8K ~21–22% at q2_k). Use it to argue *for 4-bit and against 2–3-bit* at the project scale, not that quantisation is free. An efficiency/systems citation, not safety/personalisation.

## Related

- [[sources/papers/sustainable-edge-inference]] — real-Joules companion (Raspberry Pi)
- [[sources/papers/llm-inference-edge]] — sustained-load/thermal on phones
- [[sources/papers/mobillama]] — 0.5B on-phone deployment
- [[sources/papers/qwen3-tr]] — the Qwen3-0.6B base evaluated here
- [[entities/qwen3-0.6b]] — the on-device target
- [[topics/security-and-privacy]] — on-device feasibility for privacy

## Sources

- Song, Liu, Lin, Liao, Zhao, Wang, Hu, Jiang, Long, Zhen, Jiang, Yuan, Xiang, Xu (2026) — arXiv:2505.15030 — [arxiv.org/abs/2505.15030](https://arxiv.org/abs/2505.15030)
