---
paper id: 2506.10943v2
title: "Self-Adapting Language Models"
authors: [Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, Pulkit Agrawal]
publication date: 2025-06-12T17:48
abstract: "Large language models (LLMs) are powerful but static; they lack mechanisms to adapt their weights in response to new tasks, knowledge, or examples. We introduce Self-Adapting LLMs (SEAL), a framework that enables LLMs to self-adapt by generating their own finetuning data and update directives. Given a new input, the model produces a self-edit-a generation that may restructure the information in different ways, specify optimization hyperparameters, or invoke tools for data augmentation and gradient-based updates. Through supervised finetuning (SFT), these self-edits result in persistent weight updates, enabling lasting adaptation. To train the model to produce effective self-edits, we use a reinforcement learning loop with the downstream performance of the updated model as the reward signal. Unlike prior approaches that rely on separate adaptation modules or auxiliary networks, SEAL directly uses the model's own generation to control its adaptation process. Experiments on knowledge incorporation and few-shot generalization show that SEAL is a promising step toward language models capable of self-directed adaptation. Our website and code is available at https://jyopari.github.io/posts/seal."
comments: ""
pdf: "[[Assets/Self-Adapting Language Models (2506.10943v2).pdf]]"
url: https://arxiv.org/abs/2506.10943v2
tags: [training, reasoning, rl]
---

## Key Claims

- **SEAL**: LLM generates natural-language "self-edits" — directives specifying how to restructure input data, set optimisation hyperparameters, or invoke data-augmentation tools — then applies them via gradient-based SFT to persistently update its own weights.
- RL outer loop: reward = downstream task performance after applying the self-edit; inner loop: SFT on the self-edit. Trained with ReST^EM (filtered behaviour cloning) rather than GRPO (found unstable).
- Knowledge incorporation: no-passage ICL on SQuAD improves from 33.5% → 47.0% with SEAL-generated self-edits.
- Few-shot generalisation on ARC-AGI subset: SEAL autonomously selects augmentations and hyperparameters, outperforming standard ICL and self-editing without RL training.
- Decoupling possible: teacher model generates self-edits, student model applies them (teacher-student formulation).

## Thesis Relevance

Cited in the original acquisition checklist as addressing self-adaptation — relevant to the thesis's personalisation chapter, where the system must adapt to individual users through interaction. SEAL's RL loop for learning *how to generate training data* (meta-learning) is conceptually related to the thesis's GRPO reward design for behavioural adaptation. The teacher-student decoupling is directly applicable: a larger critique model generates revision data; the 0.6B student model trains on it.

## Questions / Open Issues

- SEAL requires access to the model's weights for gradient updates during inference — impractical for consumer deployment; how does on-device weight-update feasibility work for 0.6B?
- ReST^EM was more stable than GRPO for SEAL's outer loop — should the thesis consider ReST^EM instead of GRPO for the behavioural RL stage?
- Privacy: on-device weight updates from personal conversations is the ideal SEAL application for the thesis, but requires careful handling of what is encoded in the weights.
