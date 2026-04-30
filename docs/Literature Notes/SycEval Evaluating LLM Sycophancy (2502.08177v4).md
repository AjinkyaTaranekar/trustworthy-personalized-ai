---
paper id: 2502.08177v4
title: "SycEval: Evaluating LLM Sycophancy"
authors: [Aaron Fanous, Jacob Goldberg, Ank A. Agarwal, Joanna Lin, Anson Zhou, Roxana Daneshjou, Sanmi Koyejo]
publication date: 2025-02-12T07:32
abstract: "Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications."
comments: AIES 2025
pdf: "[[Assets/SycEval Evaluating LLM Sycophancy (2502.08177v4).pdf]]"
url: https://arxiv.org/abs/2502.08177v4
tags: [sycophancy, evaluation, benchmark, alignment]
---

## Key Claims

- **58.19% overall sycophancy rate** across ChatGPT-4o, Claude-Sonnet, Gemini-1.5-Pro on AMPS (maths) and MedQuad (medical advice).
- Two sycophancy directions: **progressive** (converges to correct answer, 43.52%) vs **regressive** (converges to incorrect, 14.66%) — both are problematic in different ways.
- **High persistence**: 78.5% (95% CI: 77.2–79.8%) — once sycophantic, the model maintains the stance across rebuttal chains regardless of context or model.
- Preemptive rebuttals cause more sycophancy than in-context (61.75% vs 56.52%); citation rebuttals cause the most *regressive* sycophancy.
- Gemini highest rate (62.47%), ChatGPT lowest (56.71%) — variation is narrow, confirming sycophancy is cross-model systemic behaviour.

## Thesis Relevance

Provides the key statistic (58.19%) cited in the LLNCS paper and confirms that sycophancy is persistent and cross-model — not corrected by simple conversational repair. The progressive/regressive distinction matters for the thesis: the model should exhibit progressive sycophancy correction (converge toward truth) not regressive.

## Questions / Open Issues

- Tested on maths and medical domains only; extension to the conversational/empathy domain of the thesis would be valuable.
- No test of locally-fine-tuned small models — sycophancy rate on Qwen3-0.6B after GRPO training unknown.
- Persistence metric (78.5%) suggests the behavioural reward in GRPO needs to penalise mid-conversation capitulation, not just single-turn responses.
