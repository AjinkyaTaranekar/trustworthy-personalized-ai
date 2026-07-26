---
title: "Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency"
type: source
tags: [on-device, quantisation, latency]
sources:
  - https://arxiv.org/abs/2504.03360
updated: 2026-07-22
status: current
---

# Sustainable LLM Inference for Edge AI

**Post-training weight-only quantisation of sub-3B LLMs on a Raspberry Pi 4 delivers large energy savings (up to ~79% J/token vs FP16) but with a non-linear, model- and task-dependent accuracy/energy/latency trade-off — so the "right" quantisation level is a Pareto choice, not "always go lower".**

## Summary

Husom et al. (SINTEF et al., 2025) run a hardware-metered edge benchmark (Joulescope on a Raspberry Pi 4) across 28 quantised variants of four small bases over five task datasets. The energy axis is rigorous — real Joules with idle power subtracted — and the findings are directional: aggressive quantisation gives the biggest wins (llama3.2-1B 17.60 → 3.75 J/token, ~79%; qwen2.5-0.5B 4.42 → 1.78 J/token, ~60%) but **3-bit is not reliably cheaper than 4-bit** (non-monotonic), quantisation sensitivity is model-specific, and per-token energy only means anything alongside response length (Pearson 0.54). Load-bearing BACKGROUND/support for the on-device efficiency half of the thesis — and it includes qwen2.5-0.5B and llama3.2-1B, the closest public analogues to a 0.6B target.

## Why it matters here

Concrete, hardware-measured evidence for the quantisation energy/latency/accuracy trade-off at exactly the target scale. Hooks: (1) justify a ~4-bit K-quant on-device operating point by citing the Pareto-optimal qwen0.5b Q4 points (~14–16 J/response) and llama1b_q4_K_S; (2) argue per-token energy must be reported *with* response length — relevant if the harness adds reasoning/tool tokens that inflate output length and thus energy; (3) caution that 3-bit is not automatically cheaper or safer than 4-bit. Complements the on-device serving evidence in [[sources/papers/llm-inference-edge]] and [[sources/papers/mobillama]].

## Key results

- **Footprint (qwen2.5-0.5b):** Q4_0 ≈336 MB, Q3_K_S ≈323 MB, Q8_0 506 MB.
- **Energy/token:** qwen0.5b 4.42 (FP16) → 1.78 (q3_K_M) J/tok; llama3.2-1b 17.60 → 3.75 (~79%); FP16→Q8_0 ≈52–54%. Non-monotonic (qwen1.5b Q4 107 J vs Q3 113.72 J per response).
- **Task drives energy:** HumanEval 1.94 J/tok (long outputs, cheapest) vs BIG-Bench Hard 11.92.
- **Pareto-optimal points:** gemma2-2b q3_K_M (acc 0.45), llama1b q4_K_S (0.39), qwen0.5b Q4/Q8.

*(Latency tables didn't render — the "up to 69% latency reduction" is the authors' prose claim, not a table-verified value.)*

## Critical appraisal

Methodologically strong on the *energy* axis — real hardware metering with idle subtraction is the rigour edge-energy claims usually lack, and the Pareto framing is the right lens. Weaknesses are on the *capability* axis: a strict delimiter-only prompt with no CoT drives accuracy so low (overall 0.30; GSM8K 0.06) that quant-vs-accuracy conclusions are muddied by harness effects (near a floor). Single device (one Pi 4, 4GB CPU-only ARM), one runtime (Ollama/llama.cpp); FP16/Q8 baselines missing for the 1.5B/2.6B models, so the "79%" is anchored on the 1B model.

> ⚠ When citing: trust the *directional* energy findings and the non-monotonicity insight; treat the specific accuracy numbers as harness-conditioned (don't use to claim capability ceilings); results are Pi-CPU-specific and may not match phone NPUs; the 79%/69% figures are best-case. An efficiency citation, not a safety/personalisation source.

## Related

- [[sources/papers/on-device-llm-eval]] — capability-vs-quant curves + the compute/bandwidth "why" (companion)
- [[sources/papers/llm-inference-edge]] — sustained-load throughput/thermal on phones
- [[sources/papers/mobillama]] — 0.5B on-phone footprint/throughput
- [[sources/papers/qlora]] / [[sources/papers/phi3-tr]] — 4-bit as the on-device operating point
- [[entities/qwen3-0.6b]] — the 0.6B target these small bases proxy
- [[topics/security-and-privacy]] — on-device inference as the privacy mechanism

## Sources

- Husom, Goknil, Astekin, Sen, Mithassel, Shar, Kåsen, Soylu (2025) — arXiv:2504.03360 — [arxiv.org/abs/2504.03360](https://arxiv.org/abs/2504.03360)
