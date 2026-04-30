---
paper id: 2412.08905v1
title: "Phi-4 Technical Report"
authors: [Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang, Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, Yi Zhang]
publication date: 2024-12-12T03:37
abstract: "We present phi-4, a 14-billion parameter language model developed with a training recipe that is centrally focused on data quality. Unlike most language models, where pre-training is based primarily on organic data sources such as web content or code, phi-4 strategically incorporates synthetic data throughout the training process. While previous models in the Phi family largely distill the capabilities of a teacher model (specifically GPT-4), phi-4 substantially surpasses its teacher model on STEM-focused QA capabilities, giving evidence that our data-generation and post-training techniques go beyond distillation. Despite minimal changes to the phi-3 architecture, phi-4 achieves strong performance relative to its size -- especially on reasoning-focused benchmarks -- due to improved data, training curriculum, and innovations in the post-training scheme."
comments: ""
pdf: "[[Assets/Phi-4 Technical Report (2412.08905v1).pdf]]"
url: https://arxiv.org/abs/2412.08905v1
tags: [foundations, training, privacy]
---

## Key Claims

- **phi-4** (14B): Microsoft's data-quality-focused LM; synthetic data used throughout training (pre-training, mid-training, post-training).
- **Surpasses its teacher model** (GPT-4o) on STEM-focused QA (GPQA 56.1 vs GPT-4o 50.6; MATH 80.4 vs GPT-4o 74.6) — strong evidence that data quality beats parameter count.
- Achieves Llama-3.1-405B-level performance on many reasoning benchmarks at 14B parameters — candidate on-device small reasoning model.
- Three pillars: synthetic data for pretraining/midtraining, curation/filtering of organic data, post-training innovations (pivotal token search for DPO).
- Apache-2.0 licence; designed for deployment scenarios with compute constraints.

## Thesis Relevance

Cited in the security analysis as industry context for the local-first privacy argument: capable on-device models (phi-4 at 14B, Qwen3 at 0.6B) exist and are competitive with much larger cloud models. Demonstrates that the privacy–capability trade-off is not binary: small local models can provide real utility. Used in the thesis's comparison table of commercial memory architectures as a small-model deployment reference.

## Questions / Open Issues

- phi-4 is 14B — still much larger than the thesis's 0.6B target; the 23× size gap may mean the data-quality thesis does not generalise to 0.6B.
- Synthetic data is Microsoft-generated and not publicly available — the thesis's SFT v2 pipeline must generate its own, following different methods.
- No safety/alignment evaluation published alongside the TR — how does phi-4 perform on sycophancy and prompt injection benchmarks?
