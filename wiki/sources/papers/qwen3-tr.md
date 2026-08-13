---
title: "Qwen3 Technical Report"
type: source
tags: [qwen, small-model, reasoning, distillation]
sources:
  - https://arxiv.org/abs/2505.09388
updated: 2026-07-18
status: current
---

# Qwen3 Technical Report

**Qwen3 is an open-weight family (six dense: 0.6B/1.7B/4B/8B/14B/32B; two MoE: 30B-A3B and flagship 235B-A22B) that unifies "thinking" and "non-thinking" modes in a single model with a user-controllable thinking budget, trained on 36T tokens across 119 languages, with small models produced cheaply via strong-to-weak distillation.**

## Summary

The Qwen Team's report (Alibaba, 2025) is the primary reference for the dissertation's base model, Qwen3-0.6B. Its two design moves: fold reasoning and fast-chat into one model (selected by `/think` and `/no_think` flags, with a budget that trades latency for accuracy), and build the small models by strong-to-weak *distillation* from larger siblings at ~1/10 the GPU hours rather than training from scratch. The 0.6B base retains non-trivial capability (MMLU 52.81, GSM8K 59.59) and beats Qwen2.5-0.5B, but reasoning drops steeply with size (MATH 32.44 vs 60.80 at 8B). The report is capability-first and notably thin on safety — which is precisely the gap this project fills.

## Why it matters here

This defines what the [[entities/qwen3-0.6b|0.6B base]] *is* and can do, so it sets the capability floor against which any safety-alignment regression (the [[sources/papers/reducing-safety-tax|safety tax]]) must be measured. The built-in thinking/non-thinking modes are directly usable by a constitutional harness: thinking mode for a critique/reasoning step, non-thinking for fast responses, with the budget knob mattering for on-device latency. The `/think` `/no_think` chat-template flags define the exact interface the pipeline targets.

## Method

- **Architecture:** GQA, SwiGLU, RoPE, QK-Norm; MoE models use 128 experts, 8 activated per token; context extended to 32,768.
- **Sizes:** dense 0.6B → 32B; MoE 30B-A3B and 235B-A22B.
- **Unified modes:** one model does both; a *thinking budget* caps reasoning compute and injects a stop instruction when exhausted, giving a smooth latency/accuracy trade-off.
- **Pre-training (36T tokens, 3 stages):** general (~30T) → reasoning (~5T STEM/code) → long-context (32K). Data synthesised with Qwen2.5-VL/Math/Coder.
- **Post-training (4 stages):** long-CoT cold start → reasoning RL via [[entities/grpo|GRPO]] on ~4K verifier pairs (AIME'24 70.1→85.1) → thinking-mode fusion → general-domain RL. Strong-to-weak distillation produces the small models.

## Key numbers — base-model capability floor

| Model | MMLU | MATH | GSM8K | EvalPlus |
|---|---|---|---|---|
| Qwen3-0.6B-Base | 52.81 | 32.44 | 59.59 | 36.23 |
| Qwen3-1.7B-Base | 62.63 | 43.50 | 75.44 | 52.70 |
| Qwen3-8B-Base | 76.89 | 60.80 | 89.84 | 67.65 |
| Qwen3-235B-A22B-Base | 87.81 | 71.84 | 94.39 | 77.60 |

Reasoning RL lifted the flagship's AIME'24 from 70.1 → 85.1; strong-to-weak distillation gives small models better Pass@1 than the full pipeline at ~1/10 the GPU cost.

## Critical appraisal

Genuinely open weights across a wide size range, a clean unified-mode design with a practical budget knob, and a compute-efficient distillation story. Cautions: benchmark numbers are first-party and self-reported; the report is thin on safety/harmlessness; thinking mode adds latency/token cost (the budget trade-off is not free); and distillation-heavy small models inherit the teacher's blind spots.

> ⚠ Caution: strongly supportive on *feasibility* of the base (open, small, reasoning-capable, edge-deployable) but *silent on safety* — a "trustworthy" thesis must supply its own alignment and evaluation on top of Qwen3. The steep capability drop-off at 0.6B is the honest constraint the design must work within.

## Related

- [[entities/qwen3-0.6b]] — the base model entity page
- [[sources/papers/reducing-safety-tax]] — activates latent safety on Qwen3-0.6B; measures the tax against this floor
- [[sources/papers/deepseek-r1]] — the RL-for-reasoning lineage Qwen3's post-training draws on
- [[entities/grpo]] — the RL algorithm used in Qwen3's reasoning stage
- [[sources/papers/small-agents-collaborate]] — uses Qwen3 1.7B–32B in a multi-agent setting
- [[topics/reasoning]] — thinking/non-thinking modes as a reasoning substrate

## Sources

- Qwen Team, Alibaba (2025) — arXiv:2505.09388 — [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
