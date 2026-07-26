---
title: "A General Language Assistant as a Laboratory for Alignment"
type: source
tags: [alignment, rlhf, evaluation, anthropic]
sources:
  - https://arxiv.org/abs/2112.00861
updated: 2026-07-18
status: current
---

# A General Language Assistant as a Laboratory for Alignment

**Anthropic frames alignment around three concrete criteria — helpful, honest, harmless (HHH) — and uses a general assistant as an experimental laboratory to show that even simple interventions (a natural-language prompt, preference modelling, context distillation) improve alignment at little or no capability cost, with gains that tend to grow with model scale.**

## Summary

Askell et al. (Anthropic, 2021) is the foundational HHH paper and the seed of both RLHF assistants and Constitutional AI. Across seven model sizes (13M–52B) they test simple alignment techniques: a ~4,600-word, 14-conversation HHH prompt improves alignment (and can beat fine-tuning when alignment data is scarce); **context distillation** folds that long prompt into the weights via a KL objective; ranked preference modelling beats imitation on ranked tasks; and **preference-model pre-training (PMP)** on public preference-like data (Stack Exchange, Reddit) makes downstream reward modelling far more sample-efficient (+5–10% at 500 pairs). The alignment tax is small or negative at scale (a +3–5% code *bonus* at 13B–52B) — but small models (13M–810M) were sometimes *hurt* by the HHH prompt.

## Why it matters here

This is the canonical source for defining "trustworthy" as **helpful + honest + harmless** — the exact axes a trust-and-empathy thesis should adopt (empathy/warmth positioned as an extension of helpful and harmless). It is also directly load-bearing for the constitution-plus-SFT design: the paper shows a natural-language HHH *prompt* already moves alignment, and **context distillation** can bake it into the weights — essentially the project's mechanism (constitution in the prompt → distilled/SFT'd into a 0.6B model, no long constitution in context at inference).

## Method

- **HHH:** helpful (attempt the task, ask clarifying questions), honest (accurate, calibrated, transparent about limitations), harmless (avoid toxic/dangerous content, recognise sensitive contexts) — acknowledged to involve trade-offs and subjectivity.
- **Models:** 13M–52B, 400B tokens, 8,192 context.
- **Interventions:** a 14-conversation HHH prompt (prompting vs fine-tuning); context distillation `L(θ)=D_KL(p₀(X|C)‖p_θ(X))`; three preference objectives (imitation vs binary vs ranked PM, `L_PM=log(1+e^{r_bad−r_good})`); PMP as a transfer stage.

## Key results

- **HHH benchmark (~200 comparisons):** prompting improves alignment across scales; ranked PM ≫ imitation on ranked tasks, ≈ on binary.
- **Alignment tax/bonus:** code +3–5% *bonus* at 13B/52B; Lambada ~1–2% tax; prompting reduced toxicity at scale.
- **PMP:** +5–10% at 500 pairs (52B); binary PMP beats ranked by ~+5% at 500 pairs; transfers across dissimilar domains.
- **Human eval (~6,000 comparisons):** Elo roughly linear in log(model size).
- **Context distillation** performs comparably to direct prompting for large models.

## Critical appraisal

Defines the HHH vocabulary the field adopted; broad, honest, scale-aware; introduces context distillation and PMP (both influential); candid about trade-offs. Cautions: the HHH benchmark is small (~200 items) and Anthropic-authored (encodes the authors' values); "honest" is measured only by proxy; alignment-tax numbers rest on a limited benchmark set; the study is deliberately exploratory (RLHF left to successors).

> ⚠ 0.6B caution — the sharpest warning in this batch: the favourable "alignment gains scale with size / tax is small-or-negative" finding held at **13B–52B**, while **small models (13M–810M) were sometimes *hurt* by the HHH prompt** and the alignment bonus appeared only at scale. So a sub-1B constitutional model may pay a real alignment tax rather than a bonus — a hypothesis to test explicitly at 0.6B (measuring capability retention alongside HHH adherence, as [[sources/papers/c3ai|C3AI]] did), not to assume away. Making constitutional HHH work at sub-1B is precisely the gap this paper leaves open.

## Related

- [[sources/papers/constitutional-ai-bai]] — the CAI successor to this HHH programme
- [[sources/papers/instructgpt]] — the parallel RLHF line; alignment beats scale
- [[sources/papers/reducing-safety-tax]] — the alignment/safety tax revisited for small reasoning models
- [[sources/papers/flan]] — instruction tuning; the ≤8B degradation echoes the small-model caution
- [[entities/constitution]] — the HHH-derived written principles
- [[topics/security-and-privacy]] — HHH as the trust target

## Sources

- Askell, Bai, Chen, Drain, Ganguli, Henighan, Jones, Joseph, Mann, DasSarma, et al. (Anthropic, 2021) — arXiv:2112.00861 — [arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861)
