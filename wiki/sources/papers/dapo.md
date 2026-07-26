---
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
type: source
tags: [rl, dapo, grpo, reasoning]
sources:
  - https://arxiv.org/abs/2503.14476
updated: 2026-07-18
status: current
---

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**Reproducible, SOTA-class reasoning RL is achievable by fixing four concrete failure modes of naive GRPO — entropy collapse, zero-gradient batches, length bias, and truncation noise — yielding DAPO, which beats DeepSeek-R1-Zero-Qwen-32B on AIME 2024 (50 vs 47) at half the training steps, released fully open (algorithm + code + data).**

## Summary

Yu et al. (ByteDance Seed + Tsinghua AIR, 2025) observe that frontier reasoning-RL results were reported without reproducible detail, and that naive [[entities/grpo|GRPO]] on a strong base actually stalls (~30 AIME) due to diagnosable pathologies. DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) adds four techniques over GRPO, **removes the KL penalty**, and ships the verl code and the DAPO-Math-17K dataset. The ablation ladder (30 → 50 AIME) is the paper's best asset: it isolates *why* naive GRPO fails, with dynamic sampling (+8) and overlong filtering (+6) the biggest contributors. This is the second pillar of the project's RL family, and its KL-removal decision is a design flag for a constitutional model.

## Why it matters here

The four techniques are directly reusable in the pipeline's GRPO run — especially dynamic sampling, which fixes the exact zero-variance failure mode flagged in [[sources/papers/deepseekmath]]. But the **KL-removal is a decision to resist for a trustworthy thesis**: DAPO drops the KL-to-reference term to maximise reasoning, whereas for an on-device constitutional model the KL anchor is often what preserves aligned behaviour — so the project may want to *retain* GRPO's KL, trading some reasoning headroom for safety stability.

## The four techniques (precise)

1. **Clip-Higher** (fixes entropy collapse): decouple the clip into `ε_low=0.2`, `ε_high=0.28` so low-probability exploration tokens are not throttled — restoring exploration.
2. **Dynamic Sampling** (fixes zero-gradient batches): oversample and keep only prompts with `0 < accuracy < 1`, so every retained group yields a real gradient; sample until the batch is full.
3. **Token-Level PG Loss** (fixes length bias): normalise by total tokens across the batch (`1/Σ|o_i|`), weighting every token equally rather than every sample equally.
4. **Overlong Reward Shaping** (fixes truncation noise): mask loss on truncated samples (Overlong Filtering) plus a soft linear length penalty in a cache band before the hard limit.

Setup: Qwen2.5-32B, DAPO-Math-17K (integer answers for clean rule-based reward), LR 1e-6, 512 prompts × 16 responses, 20,480-token generations.

## Key results — AIME 2024 ablation ladder

| Configuration | AIME | Δ |
|---|---|---|
| Naive GRPO | 30 | — |
| + Overlong Filtering | 36 | +6 |
| + Clip-Higher | 38 | +2 |
| + Soft Overlong Punishment | 41 | +3 |
| + Token-level Loss | 42 | +1 |
| + Dynamic Sampling | **50** | +8 |

DAPO (50) beats DeepSeek-R1-Zero-Qwen-32B (47) at ~50% of the steps.

## Critical appraisal

Genuinely practical and reproducible — open code + data is a real contribution, and the ablation is convincing. The authors stress system interdependence (the pipeline is brittle and tuning-sensitive). Cautions: math-only (AIME), 32B with 20k-token generations and heavy sampling (a large-compute regime), and the small AIME point scale means +1/+2 deltas carry variance.

> ⚠ 0.6B caution: every technique was validated at 32B. Dynamic sampling's oversampling assumes ample rollout budget the on-device regime lacks, and a 0.6B model produces fewer solvable-with-variance groups — so the zero-gradient problem is *worse* at small scale while the fix is *more* expensive. Overlong shaping matters less for short on-device generations.

## Related

- [[sources/papers/deepseekmath]] — the GRPO origin DAPO extends; DAPO fixes its zero-variance failure mode
- [[entities/grpo]] — the base algorithm; the project's RL method
- [[sources/papers/understanding-r1-zero]] — Dr. GRPO, an alternative length-bias fix
- [[sources/papers/vapo]] — value-based alternative for long-CoT RL
- [[sources/papers/beyond-react]] — GRPO instability at 0.6B, relevant to applying DAPO small
- [[sources/code/training-and-benchmark]] — the pipeline's DAPO/GRPO run
- [[topics/reasoning]] — RL for trustworthy reasoning

## Sources

- Yu et al. (2025) — arXiv:2503.14476 — [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
