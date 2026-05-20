---
title: Group Relative Policy Optimization (GRPO)
type: entity
tags: [rl, grpo, dapo, training, deepseek, small-model]
sources:
  - docs/Assets/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948v1).pdf
  - docs/Assets/Understanding R1-Zero-Like Training A Critical Perspective (2503.20783v2).pdf
  - pipeline/2_model_trainer.py
updated: 2026-05-01
status: current
---

# GRPO

**A memory-efficient policy-gradient RL algorithm (introduced in the DeepSeek family) that replaces PPO's critic/value model with a per-group advantage computed from multiple sampled completions per prompt; in this repo, implemented as DAPO — the ByteDance improvement that fixes entropy collapse and length bias, critical for sub-1B models.**

## Why it's used in this repo
GRPO is the core RL algorithm for the pipeline's Phase 2 RL trainer (not yet implemented on `main` — see [[queries/grpo-and-personalisation-master-plan]]). It sidesteps the value-model cost of PPO, which matters on small models ([[entities/qwen3-0.6b|Qwen3-0.6B]], Gemma 4) where every GB of VRAM counts. Rewards are composite: format, correctness, tool_integrity, and constitution_score (all verifiable without a reward model).

## Recommended implementation: DAPO over vanilla GRPO

For sub-1B models, vanilla GRPO risks entropy collapse — all G completions within a group converge to identical outputs, producing zero-gradient batches. DAPO (ByteDance, arXiv:2503.14476) fixes this with three changes that should be implemented from day one: (1) Clip-Higher asymmetric clipping (ε_low=0.2, ε_high=0.28) lets high-reward completions update more freely; (2) dynamic sampling drops prompts with zero within-group variance, eliminating wasted steps; (3) token-level policy gradient loss (= Dr.GRPO) normalises by completion length rather than sequence count, removing the verbosity bias that rewards longer wrong answers. DAPO scored 50 AIME 2024 points on Qwen-32B; NVIDIA's NeMo-RL achieved competitive AIME scores on a 1.5B model in 400 steps using the same principles.

## Hyperparameters for Qwen3-0.6B

| Parameter | Value | Rationale |
|---|---|---|
| Group size G | 8 | Memory-constrained; DeepSeek uses 16 on 7B+ |
| KL coefficient β | 0.001 | R1 stage-1 value; increase if constitution drift detected |
| Learning rate | 1e-6 | Lower than SFT — fine-tuning a fine-tuned model |
| Rollout temperature | 1.0 | Standard; enables within-group diversity |
| Reference policy | `checkpoint_sft` | Not base model — anchors constitution |
| ε_low / ε_high | 0.2 / 0.28 | DAPO Clip-Higher values |

## Composite reward (all verifiable — no reward model needed)

```
reward = 0.30 × format_score       (think-tag regex)
       + 0.40 × accuracy_score     (tool execution / math check)
       + 0.15 × tool_integrity     (no hallucinated tool calls)
       + 0.15 × constitution_score  (pre-computed in SFT data; frozen critic rated)
```

## Known pitfalls (flagged by literature)
- **Length bias:** [[sources/papers/understanding-r1-zero]] identifies a systematic reward toward longer incorrect outputs. Remedy: DAPO token-level loss normalisation (= Dr.GRPO).
- **Entropy collapse:** within-group completions become identical → zero gradient → wasted steps. Remedy: DAPO Clip-Higher + dynamic sampling.
- **Constitutional drift:** RL reward signal can erode SFT-learned behaviours if constitution adherence is not explicitly rewarded. Remedy: `constitution_score` in composite reward + `4_benchmark.py --probe_only` drift monitoring after every checkpoint.
- **Reward hacking:** [[sources/papers/vlm-r1]] documents rule-based reward hacking that generalises across domains.
- **Value-based alternative:** [[sources/papers/vapo]] argues a stabilised value-based PPO outperforms GRPO on AIME; read before committing to GRPO if compute allows a critic.

## Related

- [[topics/reasoning]]
- [[queries/grpo-and-personalisation-master-plan]] — full implementation plan + sequencing
- [[sources/papers/deepseek-r1]] · [[sources/papers/understanding-r1-zero]]
- [[sources/papers/vapo]] · [[sources/papers/ui-r1]] · [[sources/papers/vlm-r1]]
- [[entities/qwen3-0.6b]] · [[entities/constitution]]

## Sources

- Pipeline: `pipeline/2_model_trainer.py` (SFT; GRPO to be added)
- Papers listed above.
- DAPO: arXiv:2503.14476 (to acquire — see [[questions/2026-04-30-asset-acquisition-todo]])
- DeepSeekMath (GRPO origin): arXiv:2402.03300 (to acquire)
