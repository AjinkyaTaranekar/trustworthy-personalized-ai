---
title: Qwen3-0.6B
type: entity
tags: [qwen, small-model]
sources:
  - pipeline/2_model_trainer.py
  - README.md
updated: 2026-04-19
status: current
---

# Qwen3-0.6B

**The base model used by the pipeline. Provided via the `unsloth/Qwen3-0.6B` HF checkpoint. A small (~0.6B parameter) Qwen-family model used as the substrate for SFT and GRPO RL fine-tuning.**

## Why this choice
- Small enough to train with LoRA on a single GPU.
- Qwen bases show strong reasoning priors even without templates (per [[sources/papers/understanding-r1-zero|Understanding R1-Zero-Like Training]]), so some of the "unlocking" the pipeline aims for is essentially already available.
- Qwen-distilled variants of DeepSeek-R1 outperform Llama-distilled ones on reasoning and affective-classification tasks ([[sources/papers/xai-sentiment-deepseek-r1]]).
- Consistent with [[sources/papers/self-enhanced-reasoning|SERT]] / [[sources/papers/hidden-reasoners|LaTRO]]'s findings that small models have latent reasoning ability activatable without a larger teacher.

## Where it's used in the repo
- `pipeline/2_model_trainer.py` — LoRA fine-tuning base.
- `pipeline/3_infererence.py` — default `--base_model` for comparison.
- `pipeline/4_benchmark.py` — Condition A (no training) is this model uninstrumented.

## Related

- [[topics/reasoning]] · [[topics/llm-foundations]]
- [[entities/grpo]] · [[entities/constitution]]
- [[sources/papers/understanding-r1-zero]] · [[sources/papers/hidden-reasoners]] · [[sources/papers/self-enhanced-reasoning]]

## Sources

- `pipeline/2_model_trainer.py`
- `README.md`
