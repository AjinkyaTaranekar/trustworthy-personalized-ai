---
title: "Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting"
type: source
tags: [sft, catastrophic-forgetting]
sources:
  - https://arxiv.org/abs/2506.09428
updated: 2026-07-21
status: current
---

# Improved SFT to Mitigate Catastrophic Forgetting

**When you cannot see a base model's original SFT data, you can reconstruct a proxy of its instruction distribution by sampling the model itself, synthesise a high-quality general-purpose dataset from a committee of models, and mix that into domain fine-tuning so the model gains task skill without catastrophically forgetting general ability.**

## Summary

Ding and Wang (Nanchang University, 2025) frame catastrophic forgetting as *distribution shift* between the fine-tuning data and the data that shaped the base model. Because vendors' alignment corpora are private, public rehearsal sets (ShareGPT, UltraChat) are distributionally mismatched and can *worsen* drift. Their fix: prompt the base model with its own chat template to emit ~100k instructions (sampling its implicit distribution), generate 9 candidate responses per instruction from a 3-model committee (base + GPT-4 + Qwen2.5-72B), select the best by committee scoring, and mix this distribution-aligned rehearsal set with domain data. On Llama-3-70B the method preserves general ability (average 39.00 base → 39.21) where public-dataset baselines all *degrade* (35.91–38.35). The conceptual takeaway — rehearsal data should match the base model's own distribution — is more valuable than the thin numbers.

## Why it matters here

Directly on the project's central pain point: **catastrophic forgetting when SFT-ing a small model on a constitution**, where the model must retain general ability while acquiring constitutional behaviour. A constitution SFT set is narrow, stylised, and unlike the base's pretraining/alignment mix — exactly the mismatched data that triggers forgetting. Concrete hook: for a 0.6B model, sample the base itself to build a distribution-aligned general-ability replay set and mix it with constitution data, rather than borrowing a public SFT corpus. Attacks forgetting on the **data side**; pairs with [[sources/papers/entropy-adaptive-ft]] (optimisation side) — they stack.

## Method

1. **Reconstruct the instruction distribution:** prompt the base model with its own chat template → ~100k instructions.
2. **Multi-response generation:** committee of 3 models × 3 responses each = 9 candidates per instruction.
3. **Committee-as-judge filtering:** 5-point scoring, averaged, argmax response selected.
4. **Mix + train:** combine reconstructed general data with domain data (~17% domain), 1 epoch full fine-tuning.

## Key results

- **General-ability preservation (Llama-3-70B, benchmark avg):** base 39.00, proposed 39.21; all public-rehearsal baselines degraded (35.91–38.35).
- **Ablation:** single-model-single-response 38.81 → single-model-multi-response-filtered 39.06 → full committee 39.21 (small increments).

## Critical appraisal

The distribution-shift framing is intuitive and well-motivated; the execution is a standard synthetic-data + LLM-judge pipeline and the evidence is thin — one 70B base, a ~0.2-point average margin (the story is "avoid the drop others suffer", not "improve"), an anomalous GPQA=4.88 cell (likely a parsing artefact), dependence on GPT-4 as a committee member (limiting the "data-free" claim), and no variance reporting. Best read as a proof-of-concept for distribution-aligned rehearsal.

> ⚠ 0.6B: all evidence is at 70B — cite the *method*, not the magnitudes. The full committee (GPT-4 + 72B judge) is too heavy for a 0.6B budget; the lighter single-teacher rung (38.81/39.06) is the realistic transfer. Supports the **SFT-only pivot**: forgetting is materially mitigated purely within SFT via data curation, no RL required.

## Related

- [[sources/papers/entropy-adaptive-ft]] — the optimisation-side complement (entropy-gated loss); they stack
- [[sources/papers/lima]] / [[sources/papers/qlora]] — data quality/curation for SFT
- [[sources/papers/flan]] — instruction-tuning distribution effects
- [[decisions/2026-05-03-research-question-reframe]] — the SFT-only pivot this supports
- [[entities/constitution]] — the narrow, stylised data that risks forgetting
- [[sources/code/sft-v2-pipeline]] — where distribution-aligned rehearsal would slot in

## Sources

- Ding, Wang (2025) — arXiv:2506.09428 — [arxiv.org/abs/2506.09428](https://arxiv.org/abs/2506.09428)
