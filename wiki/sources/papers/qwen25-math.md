---
title: "Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement"
type: source
tags: [reasoning, qwen, distillation, small-model, rl]
sources:
  - https://arxiv.org/abs/2409.12122
updated: 2026-07-22
status: current
---

# Qwen2.5-Math: Toward Mathematical Expert Model via Self-Improvement

**A pipeline-wide "self-improvement" philosophy — a strong instruct model synthesises data, a reward model curates and ranks it, and RL closes the loop — lets math-specialised models (1.5B/7B/72B) reach or beat GPT-4o on math, with the 7B matching the previous 72B and inference-time reward-model selection (RM@N) consistently beating majority voting.**

## Summary

The Qwen Team (Alibaba, 2024) treat data quality and inference-time selection, not parameter count, as the lever for math. A model good enough to solve problems is also good enough to generate and filter more, so improvement is recursive across pre-training (>1T-token synthesised math corpus), post-training (RM-guided rejection-sampled CoT/TIR SFT then GRPO), and inference (RM@N selection). Qwen2.5-Math-72B-Instruct beats GPT-4o by ~2.5 avg; the 7B-Instruct (MATH 83.6) roughly matches the prior-gen 72B; RM@8 beats Maj@8 across nearly all benchmarks, and 7B + RM@256 reaches 21/30 AIME24 vs 9/30 greedy for the 72B. Mostly background/supporting evidence for the self-improvement and verifiable-reward narrative, with a genuinely transferable idea: RM@N > Maj@N buys accuracy at inference, model-agnostically.

## Why it matters here

Its GRPO fuses a **rule-based binary verifier with an RM signal** (`r = σ(α·r_m) + (r_v − 1)`) — exactly the outcome/verifiable-reward pattern the pipeline leans on, and evidence that verifier + RM together beat either alone. The "7B matches 72B / 1.5B beats prior open models" results support that small models can be pushed via data curation + inference-time selection rather than scale, and RM@N is a candidate inference-time scaffold for a weaker model (prior art that inference compute substitutes for parameters).

## Method

- **Self-improvement loop:** instruct model synthesises data → RM curates SFT data / stronger SFT improves the RM → RM steers inference sampling.
- **Data pipeline:** Qwen Math Corpus v2 (>1T tokens) via FastText recall, MinHash dedup, LM quality filtering, and synthetic generation from Qwen2-Math-72B-Instruct; 13-gram + LCS-ratio decontamination.
- **Post-training:** iterative rejection-sampled CoT (~2M EN) + TIR (Python-interpreter) SFT, then GRPO on 66K queries with 2–5 correct of 8. RM trained listwise on 361K EN problems.

## Key results

- **CoT (English avg):** 72B-Instruct 68.2 (GPT-4o 62.0); 7B 62.9; 1.5B 56.9. Base 72B MATH 66.8 (SOTA).
- **TIR beats CoT at every size:** 72B MATH 88.1; 7B-TIR approaches 72B-CoT.
- **Inference selection:** RM@8 > Maj@8 (72B MATH 89.8 vs 88.6); 7B + RM@256 = 21/30 AIME24.

## Critical appraisal

A strong, reproducible demonstration that data-centric self-improvement + inference-time RM selection can substitute for parameters on math; the most transferable idea is RM@N > Maj@N. Weaknesses are those of technical reports: thin controlled ablations (hard to attribute gains among corpus quality, RM curation, GRPO), opaque synthetic-data provenance (13-gram filter only partly addresses contamination), and competition scores show even a 72B is weak on the hardest problems without heavy sampling + RM.

> ⚠ 0.6B caution: the smallest model is 1.5B (2.5× a 0.6B target) — encouraging for small-model math but not a clean 0.6B result. Treat as BACKGROUND unless the pipeline adopts RM@N or the data-synthesis recipe.

## Related

- [[sources/papers/gsm8k]] — the verifier + outcome-reward ancestor
- [[sources/papers/deepseekmath]] / [[sources/papers/dapo]] — the GRPO family used here
- [[sources/papers/qwen3-tr]] — the successor base family (the project's 0.6B)
- [[sources/papers/structured-templates]] / [[sources/papers/thinker]] — small-model scaffolding, convergent
- [[sources/papers/phi4-tr]] — synthetic-data-quality small-model recipe
- [[topics/reasoning]] — verifiable rewards; inference-time selection
- [[entities/grpo]] — the RL algorithm

## Sources

- Qwen Team, Alibaba (2024) — arXiv:2409.12122 — [arxiv.org/abs/2409.12122](https://arxiv.org/abs/2409.12122)
