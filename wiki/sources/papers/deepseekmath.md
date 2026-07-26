---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
type: source
tags: [rl, grpo, reasoning, deepseek]
sources:
  - https://arxiv.org/abs/2402.03300
updated: 2026-07-18
status: current
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning

**A domain-specialised 7B open model reaches near-frontier maths by combining a 120B-token mined maths corpus with a new, memory-efficient RL algorithm — GRPO — that drops PPO's critic and estimates the baseline from group-relative sample rewards. This is the origin paper of the project's RL algorithm family.**

## Summary

Shao et al. (DeepSeek-AI, 2024) attack two bottlenecks in open maths reasoning: scarce/noisy web maths data and PPO's expensive value network. They mine a 120B-token corpus (~7–9× prior work) via an iteratively-trained fastText classifier, continue-train a 7B code base model on it, SFT into an Instruct model, then apply **GRPO** to reach GSM8K 88.2% and MATH 51.7% — beating all open 7B–70B models at the time on these benchmarks. GRPO is the load-bearing contribution here and the algorithm the whole dissertation RL story builds on ([[entities/grpo]]). A sober framing to carry: RL improved Maj@K but *not* Pass@K — it sharpens the existing distribution rather than creating new capability.

## Why it matters here

This is where [[entities/grpo|GRPO]] — the project's RL algorithm — comes from, so the mechanics matter precisely. It also supplies the thesis motif that "careful data + efficient RL beats scale". The Maj@K-not-Pass@K finding is a direct **0.6B caution**: RL only surfaces reasoning the base already latently holds, so a weak small base bounds what GRPO can deliver — while the critic-free memory saving is exactly what makes RL attractive on small compute.

## GRPO — the algorithm (capture precisely)

- **Critic-free:** for each question sample a group of G outputs from `π_old`, score them, and use group statistics as the baseline instead of a learned value network `V(s)` — a large memory/compute saving, and naturally aligned with comparison-trained reward models.
- **Group-relative advantage:** outcome supervision uses `Â_{i,t} = (r_i − mean(r)) / std(r)` broadcast to every token; process supervision sums normalised future step rewards.
- **KL in the loss, not the reward:** GRPO adds `−β·D_KL[π_θ‖π_ref]` directly to the objective (β=0.04) using the unbiased k3 estimator (guaranteed positive). Contrast [[sources/papers/dapo|DAPO]], which removes it.
- **Setup:** LR 1e-6, β=0.04, 64 samples/question, batch 1024, on GSM8K+MATH CoT.

## Key results

- **Base 7B:** GSM8K 64.2%, MATH 36.2%, MMLU 54.9%.
- **Instruct 7B:** GSM8K 82.9% (CoT) / 83.7% (tool-use); MATH 46.8% / 57.4%.
- **RL 7B (GRPO):** GSM8K **88.2%** (↑5.3), MATH **51.7%** (↑4.9), CMATH 88.8%.
- **Pre-training ablations:** code data helps program-aided reasoning; arXiv-only data was *ineffective*.
- **RL raises Maj@K but not Pass@K** — elicitation, not new capability.

## Critical appraisal

GRPO is elegant and genuinely cheaper than PPO; the wins are large on hard benchmarks and later corroborated by the whole DeepSeek-R1 line. Cautions: decontamination is 10-gram exact-match only; the "beats 7B–70B" claim is maths-scoped; the 120B corpus is not released (limiting exact reproduction). Scope: single domain (maths), single scale (7B), no sub-1B evidence.

> ⚠ Failure mode to design around: if all G group samples get identical reward, `std→0` and the advantage is undefined/zero — no learning signal. This is worse at small scale (sparser reward distribution) and is exactly what [[sources/papers/dapo|DAPO's]] dynamic sampling later fixes.

## Related

- [[entities/grpo]] — the algorithm this paper introduced; the project's RL method
- [[sources/papers/dapo]] — the practical DAPO fixes over naive GRPO
- [[sources/papers/understanding-r1-zero]] — GRPO length-bias critique (Dr. GRPO)
- [[sources/papers/deepseek-r1]] — the reasoning-RL line this seeds
- [[sources/papers/beyond-react]] — confirms GRPO instability at 0.6B
- [[topics/reasoning]] — RL for trustworthy reasoning
- [[sources/code/training-and-benchmark]] — where the project's GRPO/DAPO run lives

## Sources

- Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo (2024) — arXiv:2402.03300 — [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
