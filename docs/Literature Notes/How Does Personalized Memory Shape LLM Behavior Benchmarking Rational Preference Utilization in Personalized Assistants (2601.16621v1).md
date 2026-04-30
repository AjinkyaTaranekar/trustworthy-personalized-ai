---
paper id: 2601.16621v1
title: "How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants"
authors: [Xueyang Feng, Weinan Gan, Xu Chen, Quanyu Dai, Yong Liu]
publication date: 2026-01-23T10:19
abstract: "Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses. However, irrelevant personalized memories are often introduced into the context, interfering with the LLM's intent understanding. To comprehensively investigate the dual effects of personalization, we develop RPEval, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol. RPEval reveals the widespread phenomenon of irrational personalization in existing LLMs and, through error pattern analysis, illustrates its negative impact on user experience. Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on RPEval, and resolves 80% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at https://github.com/XueyangFeng/RPEval."
comments: ""
pdf: "[[Assets/How Does Personalized Memory Shape LLM Behavior Benchmarking Rational Preference Utilization in Personalized Assistants (2601.16621v1).pdf]]"
url: https://arxiv.org/abs/2601.16621v1
tags: [personalisation, over-personalisation, evaluation, benchmark, reasoning]
---

## Key Claims

- Proposes **Rational Personalisation** (L2): memory should be treated as a pragmatic Bayesian inference about user intent, not mechanically concatenated (L1) or ignored (L0).
- **RPEval**: 953 gold-standard samples, 91.86% human inter-annotator agreement; reveals a 40–90% accuracy gap between LLMs and humans on rational personalisation tasks.
- **RP-Reasoner**: ~35% improvement in intent prediction accuracy, ~26% reduction in error severity; resolves ~80% of bad cases in a large-scale commercial personalised assistant.
- Inverse scaling effect: irrational personalisation failure modes become *more pronounced* as model capability increases — larger models over-commit to stored preferences.
- Error taxonomy: Filter Bubble (FB), Redundant Information (RII), Under-Personalisation (UPB), Low Feasibility (LF), Verbose Generation (VB).

## Thesis Relevance

Provides the theoretical grounding (Rational Speech Acts) for why literal memory injection (L1) produces over-personalisation, and why intent-aware utilisation (L2) is needed. The inverse-scaling finding directly supports the claim that over-personalisation is not a capability issue solved by larger models. The error taxonomy maps onto the three failure modes in the LLNCS paper.

## Questions / Open Issues

- RP-Reasoner adds reasoning overhead — latency impact on an on-device 0.6B model?
- RPEval tested only on large commercial models; behaviour on small fine-tuned models (Qwen3-0.6B) is unknown.
- The "attraction bias" root cause (Niu et al. 2025) is cited but not acquired.
