---
title: "Extracting Training Data from Large Language Models"
type: source
tags: [privacy, security, memorisation]
sources:
  - https://arxiv.org/abs/2012.07805
updated: 2026-07-18
status: current
---

# Extracting Training Data from Large Language Models

**An LLM with only black-box generation access will, when sampled and filtered cleverly, emit verbatim chunks of its training data — including personally identifiable information that appeared in a single document — demonstrating that LLMs memorise and can leak individual training records without any overfitting.**

## Summary

Carlini et al. (USENIX Security 2021) give the defining empirical demonstration of LLM memorisation. From 1,800 candidate generations of GPT-2, they confirm **604 unique verbatim memorised training examples** (33.5% true-positive rate; best configuration 67%), including the full name, address, email, phone and fax of a real individual and six users of an IRC conversation that appeared in exactly one training document. Crucially, memorisation is decoupled from overfitting (GPT-2's train/test loss gap is small) and grows with model size and with duplication: a controlled canary experiment showed the 1.5B model emitted every string inserted ≥33 times while the 117M model memorised none. This is the strongest single argument for the dissertation's privacy sub-theme.

## Why it matters here

It is the canonical proof that an LLM will regurgitate a real person's PII from its weights — the exact harm a trustworthy, privacy-preserving assistant must prevent. Direct warning for memory-augmented personalisation: fine-tuning on a user's history (especially anything repeated across sessions) makes that content extractable, arguing strongly for keeping personal context in an external, deletable store ([[entities/graph-rag]]) rather than in parameters. Its comparative-perplexity ranking idea is the methodological ancestor of [[sources/papers/what-should-llms-forget|WikiMem's]] calibrated scoring.

## Method

- **Threat model:** black-box sample + likelihood access; *untargeted* extraction (find any memorised example). Memorisation formalised as *k-eidetic* — recoverable and present in ≤k training documents (k=1 is the dangerous regime).
- **Stage 1 — Generation:** top-n (n=40) sampling, a high→low temperature-decay schedule to escape repetitive loops, and Common-Crawl-conditioned prompts.
- **Stage 2 — Ranking by comparison, not raw perplexity:** ratios of the target model's perplexity to a *smaller* GPT-2, to zlib compression entropy, to the lowercased text, plus a sliding-window minimum — each isolates "abnormally low loss for this content". De-duplicate, then verify top candidates against GPT-2's (unreleased) corpus.

## Key results

- **604 verbatim memorised examples** from 1,800 inspected (33.5% TPR); best config 67%. Internet-conditioned prompting: 273 vs 191 for plain sampling.
- **Content:** news, logs, licences, named individuals, contact info (32 cases), UUIDs/hashes, code, private conversations.
- **Scaling / duplication:** 1.5B memorised all canaries at ≥33 duplicates; 345M ~half; 117M none. **Bigger models memorise more; duplication is the dominant driver.**
- Authors stress 604 is a **lower bound**, not a census.

## Critical appraisal

Rigorous, real-world, honest about being a lower bound; the comparative-metric insight is genuinely clever and widely reused; the canary experiment cleanly establishes causality. Trust the existence proof and the scaling/duplication findings; treat 33.5%/67% as properties of the shortlist-ranking regime. Scope: GPT-2-scale, English web text, cloud, non-instruction-tuned.

> ⚠ Scale nuance: "117M memorised none, 1.5B memorised all at ≥33 duplicates" supports the claim that smaller on-device models memorise less verbatim PII — a defensible pro-privacy point, tempered by the fact that sufficiently-duplicated PII is still memorised, so the same local data must not be repeatedly fine-tuned in.

## Related

- [[sources/papers/membership-inference]] — the *membership*-leakage counterpart (who was in training)
- [[sources/papers/what-should-llms-forget]] — quantifies which memorised facts to forget; descends from this ranking idea
- [[sources/papers/federated-unlearning]] — efficient influence removal
- [[entities/graph-rag]] — external deletable memory as the alternative to memorisation in weights
- [[topics/security-and-privacy]] — Log-To-Leak, local-first privacy argument
- [[topics/personalisation]] — why user facts should live outside the weights

## Sources

- Carlini, Tramèr, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea, Raffel (2021) — arXiv:2012.07805 (USENIX Security 2021) — [arxiv.org/abs/2012.07805](https://arxiv.org/abs/2012.07805)
