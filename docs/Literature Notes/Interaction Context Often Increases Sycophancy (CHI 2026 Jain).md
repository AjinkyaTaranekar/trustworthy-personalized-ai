---
paper id: CHI-2026-3772318.3791915
title: "Interaction Context Often Increases Sycophancy in LLMs"
authors: [Shomik Jain, Charlotte Park, Matt Viana, Ashia Wilson, Dana Calacci]
publication date: 2026-04-13
abstract: "We investigate how the presence and type of interaction context shapes sycophancy in LLMs using two weeks of real interaction data from 38 users. We evaluate two forms of sycophancy: (1) agreement sycophancy — the tendency of models to produce overly affirmative responses, and (2) perspective sycophancy — the extent to which models reflect a user's viewpoint. User memory profiles are associated with the largest increases in agreement sycophancy. Perspective sycophancy increases only when models can accurately infer user viewpoints from interaction context."
comments: "CHI '26, April 13–17, 2026, Barcelona, Spain. Open access CC-BY."
pdf: "[[Assets/3772318.3791915.pdf]]"
url: https://doi.org/10.1145/3772318.3791915
tags: [sycophancy, personalisation, evaluation, alignment]
---

## Key Claims

- **38 participants × 2 weeks** of real interaction data (avg 90 queries, 34,416 tokens per user context) using GPT-4.1-Mini in a persistent context window.
- Two sycophancy forms: **Agreement sycophancy** (overly affirmative/flattering) and **Perspective sycophancy** (mirrors user's worldview/political views without explicit endorsement).
- Agreement sycophancy increases significantly with the *presence* of user context (p<0.05), but the *type* matters: **user memory profiles produce the largest increases** (Gemini 2.5 Pro: +45%, Claude Sonnet 4: +33%, GPT-4.1 Mini: +16%).
- Some models show increased sycophancy even with non-user synthetic contexts (Llama 4 Scout: +15%, Gemini 2.5 Pro: +9%) — context length effect, not just personalisation.
- Perspective sycophancy only increases when models accurately infer the user's viewpoint — Claude Sonnet 4 and GPT-4.1 Mini show this effect; measured on political explanations.
- GPT-5.1 shows no significant change with user interactions or memory profiles — an outlier.

## Thesis Relevance

The direct empirical link connecting memory architecture to sycophancy amplification. Memory profiles are the worst offender (+33–45%) — this is the central mechanism the thesis must address. The personalisation failure mode (over-personalisation → sycophancy) is now grounded in two weeks of *real* user data rather than synthetic rebuttals. Motivates the thesis's behavioural GRPO reward for sycophancy resistance at the memory-injection level, not just single-turn prompt level. Also directly relevant to Experiment 4 (over-personalisation evaluation under memory).

## Questions / Open Issues

- The study uses GPT-4.1-Mini as the interaction model, not a locally fine-tuned small model — how does memory-induced sycophancy manifest in a 0.6B model?
- No intervention tested — the paper identifies the problem but does not propose a remedy. Self-ReCheck (OP-Bench) or RP-Reasoner (RPEval) are candidate mitigations.
- Perspective sycophancy requires accurate user-viewpoint inference — at what context length does the 0.6B model stop inferring viewpoints accurately?
