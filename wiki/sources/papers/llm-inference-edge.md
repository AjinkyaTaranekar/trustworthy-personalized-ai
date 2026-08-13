---
title: "LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load"
type: source
tags: [on-device, latency, small-model, ttft]
sources:
  - https://arxiv.org/abs/2603.23640
updated: 2026-07-18
status: current
---

# LLM Inference at the Edge: Sustained-Load Trade-offs

**Measured on real edge hardware under sustained rather than single-burst load, mobile devices lose 15–41.5% of peak throughput to thermal throttling, while a dedicated low-power NPU and a laptop GPU stay thermally flat — so the right edge platform depends entirely on whether the binding constraint is peak throughput, sustained throughput, battery, or latency; no single device wins.**

## Summary

Tummalapalli et al. (2026) run the same model (Qwen 2.5 1.5B, 4-bit) over 20 back-to-back iterations on four edge platforms to expose sustained-load behaviour that single-shot benchmarks hide. The headline: the iPhone 16 Pro drops ~27% within three iterations and 41.5% peak-to-plateau (40.49 → 23.67 tok/s), the S24 Ultra a milder 15% (12.21 → 10.38), while the RTX 4050 laptop GPU (131.7 tok/s) and the Raspberry Pi + Hailo-10H NPU (6.914 tok/s, CV 0.04%, <2 W) never throttle. This is the sustained-load honesty check the on-device thesis needs: real phone throughput is the throttled plateau, not the marketing peak.

## Why it matters here

It is the missing counterweight to single-shot phone numbers like [[sources/papers/mobillama|MobiLlama's]] 7 tok/s. For a constitutional 0.6B model that emits extra "thinking" tokens, **sustained** decode throughput is the binding constraint, so the thesis should budget latency/energy against the *plateau* and acknowledge thermal ceilings. The sub-2 W throttle-free NPU result also supports a privacy story: background/asynchronous constitutional processing can run indefinitely on a small battery.

## Method

- **Model:** Qwen 2.5 1.5B, 4-bit, capped at 2,048-token context; platform-native runtimes (GGUF Q4_0 on Pi+Hailo, MLC q4f16_2 on S24, MLX Q4_0 on iPhone, GPTQ Int4 on RTX).
- **Platforms:** Raspberry Pi 5 + Hailo-10H (40 TOPS, <5 W); Galaxy S24 Ultra (Snapdragon 8 Gen 3); iPhone 16 Pro (A18 Pro); RTX 4050 laptop.
- **Protocol:** 20 consecutive greedy-decode iterations; log tokens/s, power, energy/token, TTFT, decode latency, and thermal state (Normal → Warm → Hot).

## Key results

- **Throughput (mean, sustained):** RTX 4050 131.70 (CV 2.2%, no throttle); iPhone 16 Pro 23.67 (peak 40.49, −41.5%); S24 Ultra 10.38 (peak 12.21, −15%); Hailo-10H 6.914 (CV 0.04%, no throttle).
- **Thermal:** iPhone Normal→Warm→Hot with severe degradation; S24 mild DVFS step-down (1000→720–770 MHz); RTX and Hailo flat.
- **Energy/token:** RTX 297.3 mJ, Hailo 270.5 mJ, S24 146.4 mJ (fuel-gauge, caveated) — energy proportionality near-identical across RTX and Hailo despite ~19× throughput gap.
- **Power:** Hailo whole-system 1.870 W; S24 1.486 W (display off).

## Critical appraisal

Fills a genuine gap with real hardware and honest instrumentation; the throttling numbers are exactly the data missing from single-shot edge claims. Trust the qualitative thermal story and relative ordering.

> ⚠ Caution: only one model (1.5B) and one bit-width (4-bit), and each platform uses a *different* runtime (GGUF/MLC/MLX/GPTQ), so cross-platform comparisons conflate silicon with software-stack maturity (the iPhone's severe throttle may partly reflect MLX). Only 20 iterations — true minutes-long steady state may throttle further. As a 2026 preprint, not yet peer-reviewed. The tested 1.5B is close to the 0.6B target, so a 0.6B student would plausibly land *above* these figures — strengthening practicality.

## Related

- [[sources/papers/mobillama]] — the single-shot on-device number this contextualises
- [[sources/papers/qlora]] — training-side efficiency (this is the serving side)
- [[entities/qwen3-0.6b]] — the sub-1B target below the 1.5B tested here
- [[entities/tml-interaction-small]] — frontier-scale real-time contrast
- [[topics/security-and-privacy]] — on-device inference as the privacy mechanism
- [[experiments/frontier-model-comparison]] — where on-device latency budgets matter

## Sources

- Tummalapalli et al. (2026) — arXiv:2603.23640 — [arxiv.org/abs/2603.23640](https://arxiv.org/abs/2603.23640)
