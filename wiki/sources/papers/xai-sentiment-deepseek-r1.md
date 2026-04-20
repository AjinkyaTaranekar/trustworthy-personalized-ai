---
title: Explainable Sentiment Analysis with DeepSeek-R1
type: source
arxiv_id: 2503.11655v4
authors: Huang, Wang
year: 2025
venue: IEEE Intelligent Systems
tags: [empathy, sentiment, explainability, deepseek]
sources:
  - docs/Assets/Explainable Sentiment Analysis with DeepSeek-R1 Performance, Efficiency, and Few-Shot Learning (2503.11655v4).pdf
  - docs/Literature Notes/Explainable Sentiment Analysis with DeepSeek-R1 Performance, Efficiency, and Few-Shot Learning (2503.11655v4).md
updated: 2026-04-19
status: current
---

# Explainable Sentiment Analysis with DeepSeek-R1

**DeepSeek-R1 reasoning traces provide transparent, step-by-step explanations for sentiment classification — 91.39% F1 on 5-class sentiment with just 5 shots, 8× more few-shot-efficient than GPT-4o.**

## What it does
Benchmarks full 671B DeepSeek-R1 and distilled variants vs GPT-4o on sentiment tasks. Documents distillation-architecture effects: 32B Qwen2.5-distilled beats 70B Llama-distilled by 6.69pp.

## Why it matters for this thesis
**The first empathy-adjacent source in the wiki.** Shows that reasoning-trace transparency transfers from maths/code to affective classification — directly relevant to the [[topics/empathy|appraisal detection]] pipeline. Also, the Qwen-vs-Llama distillation finding supports keeping the Qwen base for the repo's pipeline. Cite as the bridge between reasoning-RL literature and the empathy experimental track (Experiment 2 in the planning doc).

## Related

- [[topics/empathy]] · [[topics/reasoning]]
- [[sources/papers/deepseek-r1]]
- [[entities/qwen3-0.6b]]
- [[decisions/2025-11-10-ontology-focus-shift]] — sentiment classification is a natural ontology-verifier test case

## Sources

- `docs/Assets/Explainable Sentiment Analysis with DeepSeek-R1 Performance, Efficiency, and Few-Shot Learning (2503.11655v4).pdf`
- `docs/Literature Notes/Explainable Sentiment Analysis with DeepSeek-R1 Performance, Efficiency, and Few-Shot Learning (2503.11655v4).md`
