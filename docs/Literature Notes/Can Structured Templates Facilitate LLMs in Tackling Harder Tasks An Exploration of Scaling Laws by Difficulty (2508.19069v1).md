---
paper id: 2508.19069v1
title: "Can Structured Templates Facilitate LLMs in Tackling Harder Tasks? : An Exploration of Scaling Laws by Difficulty"
authors: [Zhichao Yang, Zhaoxin Fan, Gen Li, Yuanze Hu, Xinyu Wang, Ye Qiu, Xin Wang, Yifan Sun, Wenjun Wu]
publication date: 2025-08-26T14:26
abstract: "Structured, procedural reasoning is essential for Large Language Models (LLMs), especially in mathematics. While post-training methods have improved LLM performance, they still fall short in capturing deep procedural logic on complex tasks. To tackle the issue, in this paper, we first investigate this limitation and uncover a novel finding: a Scaling Law by Difficulty, which reveals that model performance follows a U-shaped curve with respect to training data complexity -- excessive low-difficulty data impedes abstraction, while high-difficulty data significantly enhances reasoning ability. Motivated by this, we propose the Structured Solution Template (SST) framework, which uses solution templates and a curriculum of varied difficulty to explicitly teach procedural reasoning. Specifically, SST comprises (1) fine-tuning with structured solution-template chains and dynamically weighted loss to prioritize procedural logic, (2) prompt-time injection of solution templates as cognitive scaffolds to guide inference, and (3) integrated curriculum fine-tuning that explicitly teaches the model to self-plan - execute - self-correct. Experiments on GSM8K, AIME24, and new Dynamic En benchmark show that SST significantly improves both accuracy and efficiency, especially on harder problems."
comments: "9 pages"
pdf: "[[Assets/Can Structured Templates Facilitate LLMs in Tackling Harder Tasks An Exploration of Scaling Laws by Difficulty (2508.19069v1).pdf]]"
url: https://arxiv.org/abs/2508.19069v1
tags: [reasoning, training, sft]
---

## Key Claims

- **Scaling Law by Difficulty**: model performance on hard tasks follows a U-shaped curve with respect to training data difficulty — more low-difficulty synthetic data *hurts* reasoning on hard problems by encouraging surface-pattern matching rather than procedural abstraction.
- Evidence: DeepSeek-R1-Distill-Qwen on AIME24 accuracy consistently *drops* as synthetic dataset size grows from 0→100k; small Open-R1 dataset (97k curated hard problems) achieves 30.48 vs synthetic methods.
- **SST Framework**: 3 stages — (1) SFT with difficulty-weighted structured solution chains; (2) prompt-time LoRA chain-generator that injects solution templates at inference; (3) curriculum fine-tuning with GRPO for plan-execute-self-correct.
- Results: +6.2 pts on GSM8K, +2.2 pts on AIME24 vs prior methods.
- The chain-generator (Qwen2.5-1.5B LoRA) produces `<chain>…</chain>` templates that scaffold the main solver without altering its weights.

## Thesis Relevance

Directly relevant to the thesis's SFT v2 data pipeline and GRPO reward design. The Scaling Law by Difficulty finding implies that the thesis's constitution-driven data generation must prioritise *quality and difficulty* over *quantity* — a principle that should inform the rejection-sampling threshold in `pipeline/sft_v2_generate_gold.py`. The SST chain-generator concept is analogous to the thesis's reasoning scaffold template approach. The GRPO self-correction stage mirrors the thesis's Experiment 1 (process-reward RL).

## Questions / Open Issues

- SST tested on mathematical reasoning (GSM8K, AIME) — does the difficulty-scaling law generalise to conversational and empathy tasks?
- The chain-generator adds a second model at inference; can this be collapsed into the 0.6B model's thinking-mode output?
- The GRPO stage in SST uses format + correctness rewards; the thesis needs behavioural rewards too — how do they interact with difficulty-weighted SFT?
