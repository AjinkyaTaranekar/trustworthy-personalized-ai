---
title: LLM Foundations
type: topic
tags: [foundations, tokenisation, attention, embeddings]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
updated: 2026-04-19
status: current
---

# LLM Foundations

**The architectural mechanics that explain why monolithic LLMs fail at reasoning and arithmetic — and why the thesis therefore proposes a modular architecture.**

## Summary
Modern LLMs stand on four mechanical commitments: **subword tokenisation** (numbers become meaningless fragments), **self-attention** (autoregressive left-to-right, no backtracking), **contextualised embeddings** (token meaning depends on context, unlike static Word2Vec), and **bidirectional vs causal attention** (BERT understands, GPT generates). Each commitment has a trust implication that the rest of the thesis must work around.

## Key consequences for trustworthiness

- **Tokenisation breaks arithmetic.** BPE splits `183491` into `["183", "491"]`, destroying mathematical properties — a structural reason computation must be delegated to [[topics/tool-use-and-verification|tools]].
- **Autoregressive decoding can't backtrack.** A wrong early token corrupts everything after; this motivates [[sources/papers/tree-of-thoughts]] and process-reward RL.
- **Contextualised attention is opaque.** Attribution back to input tokens is ambiguous when every layer re-mixes them — the root of the "sociopath yapper" explainability problem in [[topics/reasoning]].

## Related

- [[topics/reasoning]] — architectural failures this topic explains
- [[topics/tool-use-and-verification]] — the structural response

## Sources (papers ingested)

- [[sources/papers/attention-is-all-you-need]]
- [[sources/papers/bert]]
- [[sources/papers/word2vec]]
- [[sources/papers/bpe-subword-units]]
- [[sources/papers/measuring-word-significance]]
- [[sources/papers/diffusion-of-thoughts]] — alternative to the autoregressive commitment

## Raw

- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] §2 "Foundational Mechanics"
