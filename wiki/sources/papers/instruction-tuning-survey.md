---
title: "Instruction Tuning for Large Language Models: A Survey"
type: source
tags: [instruction-tuning, sft, evaluation]
sources:
  - https://arxiv.org/abs/2308.10792
  - https://github.com/xiaoya-li/Instruction-Tuning-Survey
updated: 2026-07-22
status: current
---

# Instruction Tuning for Large Language Models: A Survey

**A structured map of instruction tuning (SFT): datasets, methods/models, multi-modal and domain-specific variants, efficient-tuning techniques, and evaluation — repeatedly surfacing the finding that carefully curated, diverse instruction data matters more than raw volume.**

## Summary

Zhang et al. (2023, living survey through 2025) organise the SFT field around five axes and foreground the quality-vs-quantity thread. The central "why" is the objective gap: pretraining optimises next-token prediction, misaligned with "follow this instruction", and SFT teaches the *format and behaviour* of instruction-following over knowledge already latent from pretraining. Its most useful contribution for this project is the **dataset-construction taxonomy** — human-crafted vs synthetic-via-distillation (Alpaca, WizardLM, Orca) vs synthetic-via-self-improvement (Self-Instruct) — plus a catalogue of efficient-tuning techniques (LoRA/QLoRA) and evaluation methods (AlpacaEval, MT-Bench, HELM). BACKGROUND/SFT-pillar scaffolding — a map, not a measurement.

## Why it matters here

Frames the SFT background chapter: definitions, the pretraining-vs-instruction objective gap, the dataset taxonomy, and efficient-tuning options relevant to fitting training on limited GPUs. The project's constitutional/teacher-student data sits squarely in the **synthetic-via-distillation** category (alongside Alpaca/Orca) — useful for positioning and for anticipating the "inherits teacher bias" critique. Its evaluation section (LLM-as-judge vs closed-ended benchmarks) supports the project's substance-based judging stance. Centralises the quality > quantity literature ([[sources/papers/lima|LIMA]], Self-Instruct, Evol-Instruct) the project needs to justify a small curated constitutional dataset — but cite the *primary* papers for the claim.

## Taxonomy

1. **Datasets** by construction: human-crafted (Natural Instructions, Super-Natural Instructions ~5M, Dolly, OpenAssistant, LIMA 1K); distillation (Alpaca 52K, WizardLM/Evol-Instruct 70K, Orca 1M); self-improvement (Self-Instruct, Instruction Back-translation).
2. **Methods/models:** InstructGPT, FLAN-T5 (SFT ~0.2% of pretraining compute), Alpaca, Vicuna, WizardLM, LIMA (superficial alignment hypothesis).
3. **Multi-modal & domain-specific** (medical, legal, code).
4. **Efficient tuning:** LoRA, QLoRA, LOMO, delta-tuning.
5. **Evaluation:** MMLU/GSM8K/HumanEval/IFEval, AlpacaEval/MT-Bench, HELM.

## Key observations

- Dataset *construction method* is a primary axis; distillation-from-a-stronger-model dominates the low-cost end.
- Quality/diversity/complexity beats quantity (LIMA 1K, Evol-Instruct complexity, Self-Instruct diversity).
- SFT is cheap vs pretraining (FLAN-T5 ~0.2%) yet delivers large controllability gains — but tends to improve tasks *well-represented* in the tuning data and can capture surface patterns rather than deep comprehension.

## Critical appraisal

Ideal background scaffolding — a defensible, citable structure for the SFT pillar and a ready map of datasets/methods. But it is a *map, not a measurement*: many methods are catalogued without a common-benchmark comparison, and a living survey means figures drift across versions. Distilled datasets inherit the teacher's biases; evaluation leans on LLM-as-judge with its reliability caveats.

> ⚠ 0.6B caution: almost every representative model is 6B–176B, so the quality-over-quantity evidence *motivates* but does not *prove* the same at 0.6B — flag this gap and treat the sub-1B constitutional-SFT setting as the novel contribution the survey does not cover. For any load-bearing claim, lean on the primary sources it points to.

## Related

- [[sources/papers/lima]] — the superficial-alignment / quality>quantity anchor
- [[sources/papers/phi1-textbooks]] — curated-data small-model precedent
- [[sources/papers/instructgpt]] / [[sources/papers/flan]] — the SFT/RLHF lineage catalogued
- [[sources/papers/qlora]] / [[sources/papers/lora]] — efficient-tuning options
- [[sources/papers/mt-bench]] — the LLM-as-judge evaluation this reviews
- [[sources/code/sft-v2-pipeline]] — the project's distillation-category data recipe
- [[topics/reasoning]] — SFT as a route to trustworthy behaviour

## Sources

- Zhang, Dong, Li, Zhang, Sun, Wang, Li, Hu, Zhang, Wu, Wang (2023/2025) — arXiv:2308.10792 — [arxiv.org/abs/2308.10792](https://arxiv.org/abs/2308.10792)
