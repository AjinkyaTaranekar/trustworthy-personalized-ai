---
title: "Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting"
type: source
tags: [sft, catastrophic-forgetting]
sources:
  - https://arxiv.org/abs/2601.02151
updated: 2026-07-21
status: current
---

# Entropy-Adaptive Fine-Tuning (EAFT): Resolving Confident Conflicts

**Catastrophic forgetting during SFT is caused mainly by "Confident Conflicts" — tokens the model predicts confidently (low entropy) but which disagree with the ground-truth label (low probability); gating the SFT loss by token-level entropy suppresses those destructive gradients and preserves general ability while matching downstream SFT performance.**

## Summary

Diao et al. (BUPT / Zhongguancun Academy, 2026) explain the SFT-vs-RL forgetting gap from first principles: SFT fits *external* supervision, on-policy RL aligns with the model's *internal* beliefs, and the destructive updates concentrate in the low-entropy/low-probability quadrant — where the model holds a stubborn confident prior that contradicts the label. Standard SFT drives huge gradients there, overwriting the priors that encode general capability. EAFT is a one-line loss change: multiply each token's cross-entropy by a normalised entropy gate (top-20 logits, 0.999 Pearson with full-vocab entropy, <0.4 KB overhead), so confident-conflict tokens get near-zero weight and uncertain tokens recover full SFT. Across Math/Medical/Agent domains and Qwen/GLM at 4B–32B, it cuts general-capability loss (e.g. Qwen3-4B −4.6 → −1.0) at near-equal target performance. Attacks forgetting on the **optimisation side**; complements [[sources/papers/improved-sft-forgetting]] (data side).

## Why it matters here

A gradient/loss-level answer to the project's core problem — SFT-ing a small model on a constitution while keeping general ability. The entropy gate is a minimal change to the standard SFT objective, implementable on a 0.6B constitution-SFT run at negligible cost and streamable into a JSONL-logged loop. The Confident-Conflicts quadrant can be measured on the constitution dataset to *predict* forgetting and report a general-vs-constitutional Pareto curve. Its whole thesis — you can close most of the SFT-vs-RL forgetting gap *within SFT* — is strong evidence for the SFT-only pivot (no GRPO needed to preserve general ability).

## Method

- **Confident Conflicts:** low-probability, low-entropy tokens (confident but wrong-per-label); a pilot hard-mask of the bottom 15% by both rankings reduced forgetting.
- **EAFT loss:** `L = −Σ_t H̃_t · log P_θ(y_t|·)`, with the entropy gate `H̃_t = H_t^{top-20}/ln(20)` in [0,1] — a soft, continuous version of the mask.

## Key results

- **Qwen3-4B Math:** general avg SFT −4.6 vs EAFT −1.0; target (math) essentially unchanged (69.4 vs 69.3).
- **Consistent across domains:** Medical general SFT −4.8 vs EAFT −1.6; Agent/BFCL SFT −6.3 vs EAFT −3.6; Qwen2.5-32B −3.2 vs −1.1; GLM4-9B −6.0 vs −3.9.
- **Soft gate ≫ hard mask** on target (69.27 vs 65.60); all entropy-aware variants beat SFT (entropy-awareness, not the exact formula, is the active ingredient).

## Critical appraisal

Strong, mechanistic, and cheap — the Confident-Conflicts framing is a crisp first-principles explanation, the fix is a one-line loss change, and top-20 makes it essentially free. Consistent across three domains and two families. Reservations: the 10-epoch regime may inflate the forgetting baseline; AIME24/25 are 30-question sets where single-item swings are noise; no sub-1B evidence.

> ⚠ Critical caveat for a *constitution*: EAFT protects confident priors — but a constitution deliberately tries to *overwrite* some confident (unsafe/unhelpful) default behaviours, and the authors flag knowledge-editing/counterfactual override as out of scope. So EAFT is right for the "keep general competence" half but may *resist* the "install new refusal/empathy behaviour" half — treat that tension as a first-class experimental question. Cite the mechanism, not the 4B–32B magnitudes, for 0.6B claims.

## Related

- [[sources/papers/improved-sft-forgetting]] — the data-side complement; they stack
- [[sources/papers/reducing-safety-tax]] — on-policy distillation; the SFT-vs-RL distributional-gap theme
- [[sources/papers/luspo]] — the RL-side instability partner (length bias) vs this SFT-side fix
- [[decisions/2026-05-03-research-question-reframe]] — the SFT-only pivot this supports
- [[entities/constitution]] — the behaviours SFT must install vs the priors EAFT protects
- [[sources/code/training-and-benchmark]] — where the entropy-gated loss would apply

## Sources

- Diao, Yang, Gong, Zhang, Yan, Han, Liang, Xu, Ma (2026) — arXiv:2601.02151 — [arxiv.org/abs/2601.02151](https://arxiv.org/abs/2601.02151)
