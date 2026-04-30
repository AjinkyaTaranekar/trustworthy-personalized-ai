---
paper id: 2601.13722v1
title: "OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents"
authors: [Yulin Hu, Zimo Long, Jiahe Guo, Xingyu Sui, Xing Fu, Weixiang Zhao, Yanyan Zhao, Bing Qin]
publication date: 2026-01-20T08:27
abstract: "Memory-augmented conversational agents enable personalized interactions using long-term user memory and have gained substantial traction. However, existing benchmarks primarily focus on whether agents can recall and apply user information, while overlooking whether such personalization is used appropriately. In fact, agents may overuse personal information, producing responses that feel forced, intrusive, or socially inappropriate to users. We refer to this issue as \\emph{over-personalization}. In this work, we formalize over-personalization into three types: Irrelevance, Repetition, and Sycophancy, and introduce \\textbf{OP-Bench} a benchmark of 1,700 verified instances constructed from long-horizon dialogue histories. Using \\textbf{OP-Bench}, we evaluate multiple large language models and memory-augmentation methods, and find that over-personalization is widespread when memory is introduced. Further analysis reveals that agents tend to retrieve and over-attend to user memories even when unnecessary. To address this issue, we propose \\textbf{Self-ReCheck}, a lightweight, model-agnostic memory filtering mechanism that mitigates over-personalization while preserving personalization performance. Our work takes an initial step toward more controllable and appropriate personalization in memory-augmented dialogue systems."
comments: ""
pdf: "[[Assets/OP-Bench Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents (2601.13722v1).pdf]]"
url: https://arxiv.org/abs/2601.13722v1
tags: [personalisation, over-personalisation, evaluation, benchmark, sycophancy]
---

## Key Claims

- Over-personalisation formalised into three types: **Irrelevance** (injecting off-topic memory), **Sycophancy** (over-accommodating user beliefs), **Repetition** (near-identical responses to distinct queries).
- OP-Bench: 1,700 human-verified instances across 20 users, 3 categories, 6 subcategories; constructed from long-horizon LoCoMo dialogues.
- All memory-augmentation methods show substantial performance degradation (26.2–61.1%) vs memory-free BASE; more sophisticated memory systems (MemU, MEMOS) exhibit worse over-personalisation than simple RAG.
- Root cause: retrieved memories receive >2× more attention than the user query — "memory hijacking" overrides response reasoning.
- **Self-ReCheck**: lightweight, model-agnostic memory filter reduces over-personalisation by 29% while preserving personalisation performance.

## Thesis Relevance

Primary empirical anchor for the over-personalisation chapter. The three-type taxonomy maps directly to the failure modes named in the LLNCS paper; the 26–61% degradation statistic and Self-ReCheck mitigation are both directly cited. The attention-attribution root-cause finding motivates the thesis argument that memory must be filtered before injection, not just retrieved.

## Questions / Open Issues

- Self-ReCheck is model-agnostic but adds latency — how much? Is it compatible with on-device deployment?
- OP-Bench uses LoCoMo (human-to-human dialogues); does the benchmark generalise to human-AI conversation patterns?
- No evaluation of local/on-device memory systems — gap for the thesis's local-first privacy argument.
