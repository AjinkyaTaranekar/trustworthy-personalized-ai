---
title: "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)"
type: source
tags: [rlhf, sft, rl, instruction-tuning]
sources:
  - https://arxiv.org/abs/2203.02155
updated: 2026-07-18
status: current
---

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**Aligning a language model to human intent via RLHF — SFT on human demonstrations, a reward model learned from preference rankings, then PPO against that reward — makes a 1.3B model's outputs preferred over the raw 175B GPT-3, while improving truthfulness and reducing toxicity: alignment beats scale.**

## Summary

Ouyang et al. (OpenAI, NeurIPS 2022) give the canonical RLHF pipeline and the direct precursor to ChatGPT. The pre-training objective (next-token prediction) is misaligned with what users want (follow my instruction helpfully, honestly, harmlessly), and bigger models are not automatically better-behaved. Their three-stage recipe — SFT on ~13k demonstrations → a 6B reward model from ~33k preference rankings → PPO (with a per-token KL penalty and a "PPO-ptx" pretraining-mix term) on ~31k prompts — makes the **1.3B InstructGPT preferred over the 175B GPT-3** (85±3% at matched scale), roughly doubles TruthfulQA truthfulness, and cuts closed-domain hallucination 41%→21%. This is the strongest single citation for why a small model can be made trustworthy without frontier scale.

## Why it matters here

"Alignment beats scale" (1.3B preferred over 175B) is the best external support for the on-device thesis that a **0.6B** model can be made trustworthy and useful. Even though the project's pivot dropped GRPO for SFT-only, InstructGPT's *SFT stage* and its "helpful, honest, harmless" objective map almost directly onto the [[entities/constitution|constitution]]'s principles and the pipeline's gold-response generation. The **alignment tax** finding is a caution the project should cite when reporting whether constitutional SFT degrades general capability — the benchmark comparisons are effectively measuring that tax at 0.6B.

## Method

- **Step 1 — SFT:** ~40 screened contractors write demonstrations; GPT-3 fine-tuned on ~13k prompts.
- **Step 2 — Reward model:** the SFT model samples K=4–9 outputs per prompt; labelers rank them; a 6B RM is trained with pairwise cross-entropy over all `C(K,2)` comparisons (~33k prompts). Inter-annotator agreement ~72–77%.
- **Step 3 — PPO:** optimise the SFT policy against the RM with a per-token KL penalty toward the SFT model (prevents reward hacking); **PPO-ptx** mixes in pretraining gradients to counter regressions (~31k prompts).

## Key results

- **Preference:** 1.3B InstructGPT preferred over 175B GPT-3; at matched 175B, preferred 85±3% (vs few-shot GPT-3, 71±4%).
- **Truthfulness:** ~2× more truthful+informative on TruthfulQA.
- **Hallucination:** 41% → 21% on closed-domain tasks.
- **Toxicity:** ~25% fewer toxic generations when prompted to be respectful; **no** significant bias improvement (Winogender, CrowS-Pairs).
- **Alignment tax:** plain PPO regressed on SQuAD/DROP/HellaSwag/WMT; PPO-ptx largely fixes it with little preference cost.

## Critical appraisal

A rigorous end-to-end demonstration that alignment > scale for user-facing quality, with careful human-eval methodology and honest accounting of the tax and value-representativeness problem. Cautions: headline numbers are *human-preference* judgements from a narrow ~40-labeler pool, not objective correctness; toxicity gains are instruction-conditional; bias benchmarks did not move; the pipeline is proprietary-data-dependent.

> ⚠ Tension to carry: InstructGPT's values come from opaque preference labels, whereas the dissertation encodes values via an explicit *written constitution* — arguably more transparent and auditable, a point the thesis can make in its favour. Models still follow explicitly harmful instructions — the exact risks the trustworthiness evaluation must measure.

## Related

- [[sources/papers/gpt3-few-shot]] — the unaligned base InstructGPT improves on
- [[sources/papers/flan]] — instruction SFT without the preference stage
- [[sources/papers/constitutional-ai-bai]] — replaces human preference labels with AI feedback (RLAIF)
- [[sources/papers/reducing-safety-tax]] — the alignment/safety tax, revisited for small reasoning models
- [[entities/constitution]] — the written-principle alternative to preference labels
- [[decisions/2026-05-03-research-question-reframe]] — "alignment beats scale" underwrites the 0.6B-vs-frontier framing
- [[topics/security-and-privacy]] — helpful/honest/harmless as a trust target

## Sources

- Ouyang et al. (2022) — arXiv:2203.02155 (NeurIPS 2022) — [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
