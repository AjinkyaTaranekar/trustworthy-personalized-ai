---
title: "Embarrassingly Simple Self-Distillation Improves Code Generation"
type: source
arxiv_id: 2604.01193
authors: "Zhang, Bai, Zheng, Jaitly, Collobert, Yizhe Zhang"
year: 2026
venue: arXiv preprint
tags: [distillation, sft, small-model, self-training, qwen, reasoning]
sources:
  - https://arxiv.org/abs/2604.01193
updated: 2026-05-13
status: current
---

# Simple Self-Distillation (SSD)

**A language model can improve its own output quality by sampling from itself with temperature/truncation, then fine-tuning on those samples via standard SFT — no verifier, no teacher model, no RL required.**

## What it does

SSD has two stages: (1) sample N responses from the model using a specific temperature and nucleus-truncation configuration; (2) fine-tune the model on those samples with standard supervised fine-tuning. The key insight is that the model's own sampling distribution already contains high-quality outputs — they are just low-probability events. SSD shifts the distribution towards them without any external signal. Applied to Qwen3-30B-Instruct on LiveCodeBench v6, it raises pass@1 from 42.4% to 55.3%. Results generalise across Qwen and Llama families at 4B, 8B, and 30B scale, including instruction-tuned and reasoning variants.

## Theoretical framing: precision-exploration conflict

The paper identifies a *precision-exploration conflict* in LLM decoding: greedy/low-temperature decoding is precise but misses diverse valid solutions; high-temperature decoding explores but introduces noise. SSD resolves this by reshaping the token distribution contextually — suppressing unhelpful tail mass while preserving beneficial diversity. This is not learned from external data; it emerges from the model's own posterior.

## Why it matters for this thesis

### Constitutional SSD (direct adaptation)
The dissertation's [[entities/constitution|23-principle constitution]] and [[sources/code/training-and-benchmark|Constitutional Harness]] together act as an explicit quality filter — analogous to code execution in the original paper. A "Constitutional SSD" variant would: sample N conversational responses from [[entities/qwen3-0.6b|Qwen3-0.6B]] → score each against the rule-based harness → fine-tune on passing responses. This substitutes a constitutional compliance signal for a code-execution signal, which is a genuine novel contribution. The harness already does the filtering work; SSD tells you what to do with the passing outputs.

### Constitution drift angle
SSD provides a self-corrective mechanism that does not require RL. For the research paper on [[questions/2026-04-19-initial-questions|constitution drift]], SSD could serve as a low-cost baseline: does periodic constitutional SSD reduce drift rate more cheaply than full GRPO retraining? If yes, that is a practically significant finding.

### Interim step before GRPO
Since GRPO is not yet implemented (as of May 2026), constitutional SSD can generate preliminary training data and preliminary results on the SFT → harness-filtered SFT improvement curve before RL is online. Positions the GRPO contribution as a subsequent improvement over the SSD baseline.

### Complementarity with [[sources/papers/self-enhanced-reasoning|SERT]]
Both SSD and SERT exploit the idea that good outputs are already latent in the model's sampling distribution. SERT uses rejection sampling on a reasoning task; SSD uses temperature/truncation on code. Applied to constitutional conversational AI, the two converge: filter samples by constitution, not code execution or correctness labels.

## Caveats

- The paper's quality filter is *implicit* (temperature + truncation), not *explicit* (pass/fail on a test). Constitutional SSD would need an explicit filter — the harness — which is stronger and potentially noisier (false positives from the rule-based checker).
- Code generation has a crisp oracle (execution). Constitutional compliance is softer and partly subjective; the 23 principles are rule-based but some rely on regex heuristics that can misfire.
- The paper tests models at 4B–30B. Qwen3-0.6B is significantly smaller; scaling behaviour is extrapolated, not measured.

## Related

- [[sources/papers/self-enhanced-reasoning]] — SERT: same intuition (latent good outputs), different filter (correct reasoning steps)
- [[entities/constitution]] — the 23-principle filter that would replace code execution
- [[entities/qwen3-0.6b]] — the project's base model; same Qwen family as paper's experiments
- [[sources/code/training-and-benchmark]] — Constitutional Harness lives here; the filtering component
- [[topics/reasoning]] — self-improvement loop is relevant to trustworthy reasoning
- [[entities/grpo]] — SSD as a lighter baseline before GRPO is implemented

## Sources

- https://arxiv.org/abs/2604.01193 (web fetch; no local PDF in docs/Assets/)
