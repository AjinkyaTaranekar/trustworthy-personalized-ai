---
title: Constitution or Collapse? — CAI with Llama 3-8B (Zhang)
type: source
kind: paper
tags: [alignment, sft, training, security, evaluation]
sources:
  - docs/Assets/Constitution or Collapse Exploring Constitutional AI with Llama 3-8B (2504.04918v1).pdf
  - docs/Literature Notes/Constitution or Collapse Exploring Constitutional AI with Llama 3-8B (2504.04918v1).md
arxiv: 2504.04918
updated: 2026-04-30
status: current
---

# Constitution or Collapse? — CAI with Llama 3-8B

**Replicates Constitutional AI on Llama 3-8B using DPO, achieving 40.8% ASR reduction but at a 9.8% helpfulness cost, and observing model collapse — suggesting self-improvement is an emergent capability absent in small models.**

## Summary

Zhang (Stanford, April 2025) replicates the CAI pipeline (Bai et al. 2022) on the smaller Llama 3-8B model, replacing PPO with DPO. Results: Attack Success Rate reduced by 40.8% (71% → 42% on HeX-PHI red-team benchmark). Helpfulness cost: 9.8% drop on MTBench (avg 6.05 → 5.46). Model collapse observed in the final DPO-CAI model — the model repetitively outputs closing phrases, caused by overfitting to noisy revision data (repeated emojis in training responses). Root cause: Llama 3-8B lacks the generation quality to reliably produce useful self-critiques, making the supervised fine-tuning stage self-defeating. Conclusion: "self-improvement is an emergent property" — it does not generalise downward to small models.

## Thesis Connections

- Directly cited in [[sources/dissertation/security-privacy-social-ethics]] as evidence for the critique-loop SPOF risk in [[entities/constitution]].
- If 8B shows model collapse, the thesis's 0.6B model is at significantly higher risk — motivates using a larger teacher model for the critique phase rather than self-critique.
- The 40.8% ASR reduction at 9.8% helpfulness cost establishes a reference for what the thesis's training can expect.
- Motivates the thesis's GRPO approach (rule-based rewards rather than self-critique) as a collapse-resistant alternative.

## Related

- [[entities/constitution]] — security risks section
- [[sources/papers/constitutional-ai-bai]] — original CAI paper
- [[sources/code/sft-v2-pipeline]] — the thesis's SFT implementation
- [[topics/security-and-privacy]] — alignment regression / critique-loop SPOF
