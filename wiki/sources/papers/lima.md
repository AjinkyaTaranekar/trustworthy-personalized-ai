---
title: "LIMA: Less Is More for Alignment"
type: source
tags: [sft, instruction-tuning, alignment]
sources:
  - https://arxiv.org/abs/2305.11206
updated: 2026-07-18
status: current
---

# LIMA: Less Is More for Alignment

**Almost all of a model's knowledge is acquired in pretraining, so alignment can be achieved with a tiny set (1,000 examples) of carefully curated, stylistically consistent SFT demonstrations — no RLHF, no massive instruction datasets — because alignment mostly teaches which subdistribution of response formats to surface (the Superficial Alignment Hypothesis).**

## Summary

Zhou et al. (Meta / CMU / USC / TAU, NeurIPS 2023) fine-tune LLaMa 65B on 1,000 hand-curated prompt-response pairs with pure supervised loss and no preference optimisation, yet it is rated equal-or-better than GPT-4 in 43% of blind comparisons and than DaVinci003 in 65%. Their ablations are the point: doubling (up to ~16×) the data does not help, whereas quality and prompt diversity do, and 30 targeted multi-turn examples lift "excellent" dialogue turns from 45.2% to 76.1%. For a constitution-plus-SFT pipeline this validates curating for quality and diversity over volume — with a real caveat that "less is more" leans on a very strong 65B base.

## Why it matters here

Direct support for investing the project's teacher-generated, constitution-grounded SFT data in *quality, diversity, and stylistic uniformity* rather than sheer count. LIMA also clarifies what SFT reliably teaches — *format and stance* (acknowledge-then-answer, refusal style, calibrated tone), the "subdistribution of formats" — a strong prior for what a written [[entities/constitution|constitution]] can instil through SFT alone. It validates the project's LLM-judge stance (GPT-4 judge agreed with humans 78–79%), consistent with substance-based evaluation.

## Method

- **Superficial Alignment Hypothesis:** knowledge is learnt in pretraining; alignment teaches which response formats to use when talking to a user.
- **Data (1,000 examples):** 200 Stack Exchange STEM + 200 SE Other + 200 wikiHow + 150 r/WritingPrompts + 50 Natural Instructions + 200 manually authored — quality-filtered, written in a uniform helpful-assistant tone (acknowledge, then answer).
- **Training:** LLaMa 65B, 15 epochs, pure SFT (no RLHF), residual dropout rising 0.0→0.3 across layers; checkpoint chosen by manual inspection on a 50-example dev set.

## Key results

- **Human preference (equal-or-better):** 65% vs DaVinci003, 58% vs Bard, 46% vs Claude, 43% vs GPT-4.
- **Absolute:** ~50% Excellent, 88% meet requirements.
- **Ablations (7B, Likert):** filtered > unfiltered data (~+0.5); diverse > homogeneous prompts; **quantity scaling (up to ~16×) gives no gain**.
- **Multi-turn:** +30 dialogue examples → excellent turns 45.2% → 76.1%.

## Critical appraisal

A clean, provocative hypothesis backed by controlled ablations that reframed the field's thinking on data quantity. Cautions: the evaluation is single-turn preference on 300 prompts (favours fluent formatting, not tested for factual reliability or adversarial safety); "equal or better in 43% vs GPT-4" bundles ties with wins; the hypothesis speaks to *helpfulness style*, not the harder honesty/harmlessness axes; ablations are on 7B, not the 65B.

> ⚠ 0.6B caution: "less is more" presupposes a *massive* pretraining base (65B) whose capabilities SFT merely surfaces. A sub-1B model has far less latent capability, so at 0.6B SFT may need to do more than surface style, and more/more-targeted data may be needed — treat LIMA's small-data optimism as an upper bound and run the quality/diversity/quantity ablation at the project's own scale.

## Related

- [[sources/papers/flan]] — the "scale the instruction data" view LIMA undercuts; contrast the ≤8B degradation
- [[sources/papers/instructgpt]] — the RLHF recipe LIMA argues is not always necessary
- [[sources/papers/qlora]] — independently finds data quality > quantity for instruction tuning
- [[sources/papers/c3ai]] — "less is more" for constitution size too (15 principles ≈ 58)
- [[entities/constitution]] — the curated principle set SFT would instil
- [[sources/code/sft-v2-pipeline]] — the project's curated SFT data generation
- [[experiments/human-evaluation-rubric]] — GPT-4-as-judge validation LIMA supports

## Sources

- Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, Lewis, Zettlemoyer, Levy (2023) — arXiv:2305.11206 (NeurIPS 2023) — [arxiv.org/abs/2305.11206](https://arxiv.org/abs/2305.11206)
