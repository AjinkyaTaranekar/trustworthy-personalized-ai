---
title: "Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable"
type: source
tags: [safety-tax, security, reasoning, trade-off]
sources:
  - https://arxiv.org/abs/2503.00555
updated: 2026-07-18
status: current
---

# Safety Tax: Safety Alignment Makes Your LRMs Less Reasonable

**Applying safety alignment to a Large Reasoning Model restores safety but measurably degrades its reasoning, so the standard sequential pipeline (reasoning-train, then safety-align) forces a direct trade-off — and the form of the safety data matters as much as its presence: long safety chains-of-thought preserve reasoning far better than blunt direct refusals.**

## Summary

Huang et al. (Georgia Tech, 2025) quantify the "safety tax" across three 32B reasoning models. Reasoning training raises accuracy but *also* raises harmfulness (s1.1-32B: harmful score 16.70 → 60.40, +43.7); re-aligning claws back safety at a reasoning cost. Crucially, the safety-data *form* changes the cost: their **DirectRefusal** (short canned refusals) cuts harm 59.6 points but destroys ~31% of reasoning, whereas **SafeChain** (long safety CoT) cuts harm 29.1 points for only ~7% reasoning loss. This is the empirical backbone for the dissertation's safety-versus-capability discussion, and it gives quantitative support for building *reasoned* refusals over hardcoded ones.

## Why it matters here

It names the project's central tension (safety-vs-capability). The actionable lesson maps straight onto the constitution/SFT approach and the project's substance-based-evaluation stance: prefer **reasoned/constitutional refusals** (SafeChain-style — refuse *and explain why*) over blunt hardcoded refusals (DirectRefusal-style), because the hardcoded form costs the most reasoning. It is the "defines the tax" companion to [[sources/papers/reducing-safety-tax]] (which offers a fix, OPSA, on Qwen3-0.6B).

## Method

- **Safety metric — Harmful Score:** fraction of harmful BeaverTails prompts (1,000) answered harmfully, per the BeaverTails moderation model.
- **Reasoning metric:** AIME24 / GPQA / MATH500 (averaged) via LM Eval Harness.
- **Safety datasets:** DirectRefusal (introduced here — short thinking + direct refusal, 1,000 samples) vs SafeChain (long safety CoT, 1,000-sample subset).
- **Models:** s1.1-32B (primary), DeepSeek-R1-Distill-Qwen-32B, LIMO-32B. SFT alignment (LR 5e-5, 5 epochs).

## Key results (s1.1-32B trajectory)

| Stage | Avg Reasoning | Harmful Score |
|---|---|---|
| Base | 40.76 | 16.70 |
| + reasoning training (LRM) | 63.40 | 60.40 |
| LRM + DirectRefusal | 32.49 (−31%) | 0.80 |
| LRM + SafeChain | 56.31 (−7%) | 30.80 |

The pattern (DirectRefusal = safer but big reasoning loss; SafeChain = milder on both) replicates across all three 32B models. Alignment itself is cheap (Table 3): the cost is capability, not compute.

## Critical appraisal

Clean, reproducible design; the cross-model replication is convincing and the DirectRefusal-vs-SafeChain contrast is genuinely actionable. Cautions: harmful score is moderation-model- and benchmark-dependent (single model, 1,000 prompts); reasoning is maths/science only; the tax is demonstrated but not formalised into a Pareto frontier or mechanistic account.

> ⚠ Conflict: this **contradicts Jiang et al. (2025)**, who claimed SafeChain can *improve* reasoning; here SafeChain still costs ~7%. Keep both claims per the wiki convention.
>
> ⚠ Scope/caveat: the study is **SFT-only and 32B**; the authors explicitly disclaim generalisation to RL alignment (GRPO/RLHF). So it does *not* measure the tax under the project's GRPO/DAPO family — an open question the dissertation could position itself to answer. And on a 0.6B model whose base reasoning budget is tiny, a ~31% DirectRefusal-style tax could be catastrophic — making reasoned refusals *more* important at small scale (an extrapolation to flag, not a proven result).

## Related

- [[sources/papers/reducing-safety-tax]] — OPSA reduces/reverses this tax on Qwen3-0.6B
- [[sources/papers/effective-cai-small-llms]] — small-model CAI safety; recognition-vs-application gap
- [[sources/papers/constitution-or-collapse]] — helpfulness cost / collapse from CAI
- [[entities/constitution]] — reasoned constitutional refusals over hardcoded ones
- [[topics/security-and-privacy]] — safety alignment and its costs
- [[topics/reasoning]] — the reasoning capability the tax erodes

## Sources

- Huang et al. (2025) — arXiv:2503.00555 — [arxiv.org/abs/2503.00555](https://arxiv.org/abs/2503.00555)
