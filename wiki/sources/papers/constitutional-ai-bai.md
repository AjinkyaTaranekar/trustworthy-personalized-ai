---
title: Constitutional AI — Harmlessness from AI Feedback (Bai et al.)
type: source
kind: paper
tags: [alignment, sft, rl, training, security]
sources:
  - docs/Assets/Constitutional AI Harmlessness from AI Feedback (2212.08073v1).pdf
  - docs/Literature Notes/Constitutional AI Harmlessness from AI Feedback (2212.08073v1).md
arxiv: 2212.08073
updated: 2026-04-30
status: current
---

# Constitutional AI — Harmlessness from AI Feedback

**Anthropic's foundational paper introducing the CAI generate–critique–revise loop and RLAIF, achieving a Pareto improvement in harmlessness-vs-helpfulness over standard RLHF without any human labels for harm.**

## Summary

Bai et al. (Anthropic, December 2022) introduce Constitutional AI as a two-phase pipeline. In the supervised phase (SL-CAI): generate response to red-team prompts → self-critique according to a randomly sampled constitutional principle → revise → finetune on revised responses. In the RL phase (RL-CAI / RLAIF): use the SL-CAI model to generate pairs of responses to red-team prompts → AI evaluates which is more constitutional → train a preference model on these AI labels → finetune with RL. The result is a harmless but non-evasive assistant that explains objections rather than simply refusing. Constitutional RL achieves a Pareto improvement over standard RLHF (higher harmlessness *and* helpfulness Elo scores simultaneously). Chain-of-thought reasoning during critique improves performance and transparency.

## Thesis Connections

- **Direct ancestor** of the thesis's SFT v2 pipeline. The generate–critique–revise loop is implemented in `pipeline/sft_v2_generate_gold.py`; the 19-principle constitution in `pipeline/constitution.md` is adapted from this work.
- The RLAIF mechanism is what the thesis approximates with rule-based GRPO rewards (rather than a separate preference model).
- Cited in [[sources/dissertation/security-privacy-social-ethics]] for the critique-loop SPOF: the same model critiquing its own outputs shares biases → degeneration risk.
- [[sources/papers/constitution-or-collapse]] tests CAI at smaller scale (8B) and finds model collapse — directly relevant to the thesis's 0.6B model.

## Related

- [[entities/constitution]] — the thesis's 19-principle adaptation
- [[entities/grpo]] — RL mechanism replacing the preference model
- [[sources/papers/constitution-or-collapse]] — small-model replication
- [[topics/security-and-privacy]] — critique-loop SPOF risk
