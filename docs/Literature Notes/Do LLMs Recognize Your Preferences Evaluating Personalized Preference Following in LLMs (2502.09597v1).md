---
paper id: 2502.09597v1
title: "Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs"
authors: [Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, Kaixiang Lin]
publication date: 2025-02-13T18:52
abstract: "Large Language Models (LLMs) are increasingly used as chatbots, yet their ability to personalize responses to user preferences remains limited. We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in a long-context conversational setting. PrefEval comprises 3,000 manually curated user preference and query pairs spanning 20 topics. PrefEval contains user personalization or preference information in both explicit and implicit forms, and evaluates LLM performance using a generation and a classification task. With PrefEval, we evaluated the aforementioned preference following capabilities of 10 open-source and proprietary LLMs in multi-session conversations with varying context lengths up to 100k tokens. We benchmark with various prompting, iterative feedback, and retrieval-augmented generation methods. Our benchmarking effort reveals that state-of-the-art LLMs face significant challenges in proactively following users' preferences during conversations. In particular, in zero-shot settings, preference following accuracy falls below 10% at merely 10 turns (~3k tokens) across most evaluated models. Even with advanced prompting and retrieval methods, preference following still deteriorates in long-context conversations. Furthermore, we show that fine-tuning on PrefEval significantly improves performance. We believe PrefEval serves as a valuable resource for measuring, understanding, and enhancing LLMs' preference following abilities, paving the way for personalized conversational agents. Our code and dataset are available at https://prefeval.github.io/."
comments: "Accepted at ICLR 2025 as oral presentation. Code and data at: https://prefeval.github.io/"
pdf: "[[Assets/Do LLMs Recognize Your Preferences Evaluating Personalized Preference Following in LLMs (2502.09597v1).pdf]]"
url: https://arxiv.org/abs/2502.09597v1
tags: [personalisation, evaluation, benchmark, context-degradation]
---

## Key Claims

- **PrefEval**: 3,000 manually curated preference-query pairs, 20 topics, 3 preference forms (explicit, implicit choice-based, implicit persona-driven); ICLR 2025 oral.
- Zero-shot preference-following accuracy falls **below 10% at 10 turns (~3k tokens)** across most evaluated models — the preference stored earlier becomes functionally ignored.
- Even with advanced prompting and RAG baselines, preference following deteriorates with longer context; retrieval helps but does not solve the problem.
- Four error types: Preference-Unaware Violation (most common), Preference Hallucination Violation, Inconsistency Violation, Unhelpful Response.
- Fine-tuning on PrefEval significantly improves performance and generalises to longer contexts.

## Thesis Relevance

Cited in the LLNCS paper to support the claim that stored preferences become wasteful noise as conversation length grows. Directly motivates the thesis's selective memory injection argument (only inject preferences relevant to the current query, consistent with Self-ReCheck and RPEval findings). Also relevant to Experiment 5 (context degradation evaluation).

## Questions / Open Issues

- 10% accuracy at 3k tokens is strikingly low — does this generalise to the thesis's 0.6B model, which has a shorter effective context?
- Counterfactual question: would a scrutable user model (user-inspected NL profile) outperform the implicit preference-injection design used in PrefEval?
- Fine-tuning on PrefEval is promising but requires labelled preference data — how does it interact with the thesis's SFT constitution-driven pipeline?
