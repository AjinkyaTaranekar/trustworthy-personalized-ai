---
title: "Finetuned Language Models Are Zero-Shot Learners (FLAN)"
type: source
tags: [sft, instruction-tuning, small-model, trade-off]
sources:
  - https://arxiv.org/abs/2109.01652
updated: 2026-07-18
status: current
---

# Finetuned Language Models Are Zero-Shot Learners (FLAN)

**Fine-tuning a large LM on a diverse mix of NLP datasets rephrased as natural-language instructions teaches it to follow instructions in general, so it performs unseen task types zero-shot — the instruction-tuned 137B FLAN beats zero-shot GPT-3 175B on 20 of 25 datasets — but the recipe *hurts* generalisation below ~8B.**

## Summary

Wei et al. (Google, ICLR 2022) coin "instruction tuning": wrap 62 datasets (grouped into 12 task clusters, ~10 templates each) in natural-language instructions and fine-tune LaMDA-PT on the union, then evaluate zero-shot on a task cluster held *entirely* out of training. FLAN outperforms zero-shot GPT-3 and even few-shot GPT-3 on several benchmarks. Ablations pin down the causal ingredients — natural-language instructions (not dataset tags or bare examples), and task *diversity* (held-out performance keeps rising with more clusters, unsaturated at 7). The pivotal finding for this project is the **scale threshold**: instruction tuning helps 68B/137B models but *degrades* generalisation for 8B, 2B, and 422M models. FLAN is thus both the intellectual ancestor of the SFT/constitution pillar and the sharpest statement of the "will it work at 0.6B?" risk.

## Why it matters here

The dissertation's whole SFT premise *is* FLAN's premise — supervised instruction data is what makes a model follow novel instructions, underwriting the constitution/SFT approach and the template-vs-constitution comparisons. The OPTIONS trick and the "more clusters, no saturation" finding inform how the constitution and prompts should be phrased and how broad the question-category coverage should be. But the ≤8B degradation is a **direct challenge**, not just a hook.

## Method

- **Data:** 62 datasets → 12 task clusters; ~10 manual instruction templates per dataset (some task-inverting). Classification prompts append an `OPTIONS: [...]` list to keep the answer space explicit.
- **Training:** LaMDA-PT 137B, 30k gradient steps, batch 8,192 tokens, LR 3e-5, Adafactor, ~60 h on 128 TPU-v3.
- **Held-out-cluster protocol (the key move):** to test NLI, no NLI dataset is in training — the model must generalise to the whole cluster from instructions alone.

## Key results

- **Zero-shot FLAN beats zero-shot GPT-3 175B on 20/25 datasets**; beats few-shot GPT-3 on ANLI, RTE, BoolQ, ARC, OpenbookQA, StoryCloze.
- **Cluster ablation:** held-out performance rises monotonically 1→7 clusters, unsaturated — task diversity keeps helping.
- **Scale ablation (critical):** instruction tuning helps 137B/68B but *hurts* 8B/2B/422M — small models spend all capacity fitting the training tasks, leaving none to generalise.
- **Instruction ablation:** natural-language instructions beat bare input→output and dataset-name tags substantially — instructions, not mere multi-task exposure, drive the gains.

## Critical appraisal

Clean causal design (held-out clusters, instruction ablation) and a striking headline. Cautions: "best dev-template" reporting flatters results; heavy English bias; it measures held-out *task-type* generalisation on curated benchmarks, not open-ended real-user instructions or honesty/harmlessness (InstructGPT's territory).

> ⚠ Conflict / caution: FLAN found instruction tuning *hurts* generalisation below ~8B, whereas the thesis targets **0.6B**. The dissertation's contribution partly rests on showing that a *tighter, curated constitutional* objective (not 40 broad academic clusters competing for capacity) plus a far stronger modern base than LaMDA-PT lets a sub-1B model be instruction/constitution-tuned productively. This is the risk the experiments must answer.

## Related

- [[sources/papers/gpt3-few-shot]] — the zero-shot weakness FLAN fixes
- [[sources/papers/instructgpt]] — adds human-preference alignment on top of instruction SFT
- [[sources/papers/lora]] — the PEFT mechanism that makes instruction SFT cheap at small scale
- [[entities/constitution]] — the project's curated instruction/principle set
- [[sources/code/sft-v2-pipeline]] — the SFT data-generation pipeline this premise underpins
- [[topics/reasoning]] — SFT as a route to trustworthy behaviour

## Sources

- Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai, Le (2021/2022) — arXiv:2109.01652 (ICLR 2022) — [arxiv.org/abs/2109.01652](https://arxiv.org/abs/2109.01652)
