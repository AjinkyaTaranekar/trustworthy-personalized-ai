---
paper id: 2504.04918v1
title: "Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B"
authors: [Xue Zhang]
publication date: 2025-04-07T11:01
abstract: "As language models continue to grow larger, the cost of acquiring high-quality training data has increased significantly. Collecting human feedback is both expensive and time-consuming, and manual labels can be noisy, leading to an imbalance between helpfulness and harmfulness. Constitutional AI, introduced by Anthropic in December 2022, uses AI to provide feedback to another AI, greatly reducing the need for human labeling. However, the original implementation was designed for a model with around 52 billion parameters, and there is limited information on how well Constitutional AI performs with smaller models, such as LLaMA 3-8B. In this paper, we replicated the Constitutional AI workflow using the smaller LLaMA 3-8B model. Our results show that Constitutional AI can effectively increase the harmlessness of the model, reducing the Attack Success Rate in MT-Bench by 40.8%. However, similar to the original study, increasing harmlessness comes at the cost of helpfulness. The helpfulness metrics, which are an average of the Turn 1 and Turn 2 scores, dropped by 9.8% compared to the baseline. Additionally, we observed clear signs of model collapse in the final DPO-CAI model, indicating that smaller models may struggle with self-improvement due to insufficient output quality, making effective fine-tuning more challenging. Our study suggests that, like reasoning and math ability, self-improvement is an emergent property."
comments: "6 pages, 2 figures. Conducted as part of research on alignment techniques for language models"
pdf: "[[Assets/Constitution or Collapse Exploring Constitutional AI with Llama 3-8B (2504.04918v1).pdf]]"
url: https://arxiv.org/abs/2504.04918v1
tags: [alignment, sft, training, security, evaluation]
---

## Key Claims

- CAI replicated on Llama 3-8B using DPO (instead of PPO): **Attack Success Rate reduced by 40.8%** (71% → 42% on HeX-PHI red-team benchmark).
- Harmlessness gain comes at a cost: **helpfulness drops 9.8%** on MTBench (avg: 6.05 → 5.46).
- **Model collapse observed** in the final DPO-CAI model: repeated output phrases (e.g., closing politeness sentence looped indefinitely) caused by overfitting to repeated emojis in the SFT revision data.
- Root cause of collapse: Llama 3-8B could not reliably distinguish meaningful content from formatting noise in revision outputs — a quality-threshold issue for small models.
- Conclusion: "self-improvement is an emergent property" — small models lack the generation quality needed for reliable self-critique.

## Thesis Relevance

Directly cited in the security analysis paper as evidence for the critique-loop SPOF risk (constitution entity). Confirms that smaller models (8B) show model collapse during CAI; the thesis uses a 0.6B model, making this risk even more acute. Motivates the thesis's choice to use a separate, rule-based GRPO reward rather than relying on the same model to critique itself.

## Questions / Open Issues

- DPO was used rather than PPO — how much of the collapse is DPO-specific vs size-specific?
- The thesis could use a larger teacher model (e.g., Qwen3-7B or 32B) for the critique phase while training only the 0.6B student — this decouples quality from size.
- 40.8% ASR reduction at 9.8% helpfulness cost — is this trade-off acceptable for the thesis's use case?
