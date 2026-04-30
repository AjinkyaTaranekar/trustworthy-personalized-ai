---
paper id: 1610.05820v2
title: Membership Inference Attacks against Machine Learning Models
authors: [Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov]
publication date: 2016-10-18T22:38
abstract: |
  We quantitatively investigate how machine learning models leak information about the individual data records on which they were trained. We focus on the basic membership inference attack: given a data record and black-box access to a model, determine if the record was in the model's training dataset. To perform membership inference against a target model, we make adversarial use of machine learning and train our own inference model to recognize differences in the target model's predictions on the inputs that it trained on versus the inputs that it did not train on.
    We empirically evaluate our inference techniques on classification models trained by commercial "machine learning as a service" providers such as Google and Amazon. Using realistic datasets and classification tasks, including a hospital discharge dataset whose membership is sensitive from the privacy perspective, we show that these models can be vulnerable to membership inference attacks. We then investigate the factors that influence this leakage and evaluate mitigation strategies.
comments: "In the proceedings of the IEEE Symposium on Security and Privacy, 2017"
pdf: "[[Assets/Membership Inference Attacks against Machine Learning Models (1610.05820v2).pdf]]"
url: https://arxiv.org/abs/1610.05820v2
tags: [security, privacy, evaluation]
---

## Key Claims

- **Membership inference**: given black-box access to a model, determine whether a data record was in the training set — using an *attack model* trained on "shadow models" with known membership.
- **94% median accuracy** against Google Prediction API; **74%** against Amazon ML on realistic classification tasks; **>70%** on hospital discharge data (highly sensitive health records).
- Root cause: ML models behave differently on training data vs unseen data (overfitting) — this differential confidence signal is exploitable.
- Attack is "spooky action at a distance": the correlations that make valid generalising models possible also make them vulnerable — no generalising model can fully prevent leakage.
- Mitigation strategies (confidence rounding, limiting top-k outputs, regularisation) reduce but do not eliminate the attack.

## Thesis Relevance

Foundational paper for the re-identification risk claim in the security analysis. In the thesis's conversational AI setting, aggregated user queries and preferences stored in persistent memory represent a rich membership-inference surface: an adversary could determine whether a specific conversation (e.g., about a medical condition) was used to fine-tune or update the local model. Supports the local-first privacy argument — keeping model updates on-device reduces the attack surface.

## Questions / Open Issues

- The original experiments are on ML-as-a-service classification models (2017), not LLMs — does the attack transfer to transformer fine-tuning scenarios?
- For LoRA-fine-tuned models (the thesis uses LoRA), membership inference may exploit adapter weight divergence rather than prediction confidence — a different attack surface.
- Mitigation: differential privacy in training is the principled defence, but not yet in the thesis pipeline.
