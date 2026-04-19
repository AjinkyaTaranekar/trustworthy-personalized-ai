---
paper id: 2509.21487v2
title: "Dual-Head Reasoning Distillation: Improving Classifier Accuracy with
  Train-Time-Only Reasoning"
authors: Jillian Xu, Dylan Zhou, Vinay Shukla, Yang Yang, Junrui Ruan, Shuhuai Lin, Wenfei Zou, Yinxiao Liu, Karthik Lakshmanan
publication date: 2025-09-25T19:47:19Z
abstract: "Chain-of-Thought (CoT) prompting often improves classification accuracy, but it introduces a significant throughput penalty with rationale generation (Wei et al., 2022; Cheng and Van Durme, 2024). To resolve this trade-off, we introduce Dual-Head Reasoning Distillation (DHRD), a simple training method for decoder-only language models (LMs) that adds (i) a pooled classification head used during training and inference and (ii) a reasoning head supervised by teacher rationales used only in training. We train with a loss function that is a weighted sum of label cross-entropy and token-level LM loss over input-plus-rationale sequences. On seven SuperGLUE tasks, DHRD yields relative gains of 0.65-5.47% over pooled baselines, with notably larger gains on entailment/causal tasks. Since we disable the reasoning head at test time, inference throughput matches pooled classifiers and exceeds CoT decoding on the same backbones by 96-142 times in QPS."
comments: "39th Conference on Neural Information Processing Systems (NeurIPS
  2025) Efficient Reasoning Workshop"
pdf: "[[Assets/Dual-Head Reasoning Distillation Improving Classifier Accuracy with Train-Time-Only Reasoning (2509.21487v2).pdf]]"
url: https://arxiv.org/abs/2509.21487v2
tags: []
---
