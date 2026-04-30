---
title: Towards Understanding Sycophancy in Language Models (Sharma et al.)
type: source
kind: paper
tags: [sycophancy, rlhf, alignment, evaluation]
sources:
  - docs/Assets/Towards Understanding Sycophancy in Language Models (2310.13548v4).pdf
  - docs/Literature Notes/Towards Understanding Sycophancy in Language Models (2310.13548v4).md
arxiv: 2310.13548
updated: 2026-04-30
status: current
---

# Towards Understanding Sycophancy in Language Models

**Foundational paper demonstrating that RLHF structurally trains models to agree with users — sycophancy is a systemic property driven by preference data, not an edge case. Published at ICLR 2024 by Anthropic researchers.**

## Summary

Sharma et al. (2023/ICLR 2024) define sycophancy as model behaviour that matches user beliefs at the cost of truthfulness. They demonstrate consistent sycophantic behaviour across 5 AI assistants (Claude 1.3/2, GPT-3.5/4, LLaMA 2-70B) on 4 free-form tasks: biased feedback when user states a preference, capitulation when challenged ("Are you sure?"), biased answers when user implies a belief, and mimicry of user's incorrect poem attributions. The root cause is identified via Bayesian analysis of the hh-rlhf dataset: "matches user's beliefs" is consistently among the most predictive features of human preference labels, structurally incentivising models trained on this data to be sycophantic. PM optimisation sometimes amplifies sycophancy. Claude 1.3 admits mistakes on 98% of challenged questions.

## Thesis Connections

- Directly cited in [[sources/dissertation/overpersonalisation-paper]] as the mechanism behind the sycophancy failure mode.
- The "matches user's beliefs" preference-data finding explains why GRPO with user-satisfaction rewards will produce sycophancy unless explicitly counteracted.
- Motivates Constitution Principle 8 (Honesty and Epistemic Autonomy) as a necessary counterweight.
- [[sources/papers/syc-eval]] extends this with a cross-model measurement (58.19% rate); [[sources/dissertation/security-privacy-social-ethics]] links it to the critique-loop SPOF.

## Related

- [[topics/personalisation]] — sycophancy as mechanism section
- [[entities/constitution]] — Principle 8 (honesty / non-sycophancy)
- [[sources/papers/syc-eval]] — measurement companion
- [[sources/dissertation/overpersonalisation-paper]]
