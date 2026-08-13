---
title: "Textbooks Are All You Need (phi-1)"
type: source
tags: [small-model, distillation, sft, foundations]
sources:
  - https://arxiv.org/abs/2306.11644
updated: 2026-07-22
status: current
---

# Textbooks Are All You Need (phi-1)

**A 1.3B-parameter code model trained on ~7B tokens of deliberately "textbook-quality" data — filtered real code plus GPT-3.5-synthesised textbooks and exercises — reaches 50.6% pass@1 on HumanEval, rivalling models 10× larger trained on 100× more data: data quality can substitute for scale.**

## Summary

Gunasekar et al. (Microsoft Research, 2023) bet that the bottleneck for a small code model is the *educational value* of its tokens, not their count. A GPT-4-labelled classifier keeps instructive, self-contained code; GPT-3.5 synthesises textbook prose and exercises. phi-1 (1.3B, 8×A100 × 4 days) hits 50.6% HumanEval / 55.5% MBPP — beating StarCoder (15.5B, ~1T tokens; 33.6%) and competitive with 16B WizardCoder. The standout is the **emergent jump**: finetuning the same base on <200M tokens of exercises lifts HumanEval from 29% → 50.6%, echoing LIMA's "alignment surfaces latent ability". This is *the* flagship "quality data beats scale for small models" precedent — with the honest caveat that the quality signal is teacher-distilled.

## Why it matters here

LOAD-BEARING flagship: the headline citation for "a curated, high-quality corpus lets a small model punch far above its weight" — directly underwriting the project's data-quality > quantity thesis and the choice to build a small, deliberately-curated constitutional/SFT dataset over a large scraped one. Concrete hooks: (1) the 29% → 50.6% jump from <200M curated tokens is a strong analogue for expecting a small constitutional SFT set to move a 0.6B student substantially; (2) the GPT-4-labelled "educational value" filter is a template for defining and enforcing example quality; (3) the decontamination protocol (n-gram + embedding/AST prune-and-retrain) is a reusable method for defending the project's own eval numbers.

## Method

- **Three datasets:** filtered code (~6B tokens, kept by a random-forest classifier on GPT-4 "educational value" labels); synthetic CodeTextbook (<1B, GPT-3.5); CodeExercises (~180M, GPT-3.5, docstring→function) for finetuning.
- **Models:** phi-1 1.3B (pretrain then finetune), phi-1-base (pretrain only), phi-1-small 350M.

## Key results

- **phi-1:** 50.6% HumanEval, 55.5% MBPP; **phi-1-base 29%**; **phi-1-small (350M) 45%**.
- **Comparisons:** StarCoder-15.5B 33.6%, GPT-3.5 ~47%, WizardCoder-16B 57.3% (lower MBPP than phi-1).
- **Emergent jump:** 29% → 50.6% from <200M tokens of high-quality exercises.
- **Decontamination:** aggressively pruning ~42.5% of CodeExercises (embedding+AST similarity to HumanEval) and retraining still beat StarCoder.

## Critical appraisal

The single most citable "quality beats scale for small models" result, with a clean, memorable emergent-finetuning data point. Cautions: the "quality" signal is defined by **GPT-4/GPT-3.5 — this is distillation from a much larger teacher**, so "textbooks are all you need" is partly "a big teacher's judgement is all you need"; benchmarks (HumanEval/MBPP) are narrow and leak-prone; the classifier-filtered corpus is not public; and CodeExercises share HumanEval's docstring→function format, so some of the jump is task-format alignment, not pure latent-ability unlocking.

> ⚠ 0.6B / honest framing: phi-1 is 1.3B and Python-specialist; the project's student is 0.6B and aimed at trustworthy/empathetic conversation (much harder to benchmark than code). Its quality signal is teacher-distilled — cite as precedent but be explicit the claim is "small model + curated (teacher-distilled) data", not "small model alone". Note prompt-fragility and narrow-benchmark caveats as realistic expectation-setters.

## Related

- [[sources/papers/lima]] — "alignment surfaces latent ability"; pair to note the distillation dependency
- [[sources/papers/phi3-tr]] / [[sources/papers/phi4-tr]] — the recipe extended to on-device and general reasoning
- [[sources/papers/fineweb]] — score-for-quality data curation (FineWeb-Edu)
- [[sources/papers/qlora]] — data quality > quantity for instruction tuning
- [[entities/qwen3-0.6b]] — the sub-1B student the recipe would seed
- [[sources/code/sft-v2-pipeline]] — the project's curated SFT data generation
- [[topics/reasoning]] — curated data for small-model capability

## Sources

- Gunasekar, Zhang, Aneja, Mendes, Del Giorno, Gopi, Javaheripi, Kauffmann, de Rosa, Saarikivi, Salim, Shah, Behl, Wang, Bubeck, Eldan, Kalai, Lee, Li (2023) — arXiv:2306.11644 — [arxiv.org/abs/2306.11644](https://arxiv.org/abs/2306.11644)
