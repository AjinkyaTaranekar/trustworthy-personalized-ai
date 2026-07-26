---
title: "Deduplicating Training Data Makes Language Models Better"
type: source
tags: [foundations, privacy, memorisation]
sources:
  - https://arxiv.org/abs/2107.06499
  - https://github.com/google-research/deduplicate-text-datasets
updated: 2026-07-22
status: current
---

# Deduplicating Training Data Makes Language Models Better

**Standard LM corpora contain pervasive exact and near-duplicate text; removing it shrinks the data, cuts memorised-output emission roughly 10×, eliminates train-test overlap that inflates evaluations, and leaves validation perplexity as good or better — so deduplication is a near-free improvement with a direct privacy benefit.**

## Summary

Lee et al. (Google / Penn, ACL 2022 — the lab behind the extraction/membership-inference line) show that web corpora are far more duplicated than curated ones (RealNews 13.6%, C4 3.0% of examples vs Wiki-40B 0.4%), and that a model's likelihood objective over-weights duplicated sequences into verbatim memorisation. Two scalable methods (ExactSubstr via suffix arrays over ≥50-token matches; NearDup via MinHash-LSH at ~0.8 similarity) cut unprompted memorised-token emission from ~1.6% to ~0.17% (≈10×) with flat-or-better perplexity, and remove train-test overlap (~4.6% in C4) that had silently inflated evaluations. This is the load-bearing empirical link in the "data hygiene → memorisation → privacy" chain, and honest about what dedup *cannot* do.

## Why it matters here

Directly supports the privacy sub-theme: this ties data deduplication to reduced memorisation and reduced train-test leakage — the mechanism behind the [[sources/papers/extracting-training-data|extraction]]/[[sources/papers/membership-inference|membership-inference]] papers. Cite for "duplication amplifies memorisation; removing duplicates cuts regurgitation ~10× without hurting perplexity" — the empirical backbone linking how LLMs learn from web-scale data to what personal data they leak. It also reinforces data-quality-beats-quantity (smaller deduplicated data trains as well or better), underwriting a small curated constitutional dataset.

## Key results

- **Memorisation (XL, unprompted):** Original ~1.571% → NearDup ~0.264% → ExactSubstr ~0.168% memorised tokens (~10× reduction).
- **Prompted:** the Original model reproduced ground-truth continuations >40% of the time from duplicated prompts; dedup much lower.
- **Contamination:** C4 train-test near-dup overlap ~4.6%; duplicates roughly halve apparent perplexity (Transformer-XL LM1B: 10.11 on duplicated vs 23.58 on unique examples).
- **No accuracy cost:** deduplicated models are no worse (up to ~10% perplexity improvement on Wiki-40B), and train ~3–19% faster.

*(Numbers via automated ar5iv extraction — confirm decimals against the source before quoting.)*

## Critical appraisal

Unusually clean — a ~10× memorisation reduction with flat-or-better perplexity is a rare free lunch, and the train-test-overlap finding forced the field to take contamination seriously (dedup is now standard in C4/RefinedWeb/FineWeb pipelines). Its honesty about limits is what makes it a good privacy citation rather than an overclaim.

> ⚠ The load-bearing caveat: dedup reduces the *amplification* of memorisation but **does not bound worst-case leakage** — a single-occurrence secret (medical record, password) can still be memorised and should never be in the corpus at all. So it complements, not replaces, differential privacy / data filtering. English-only, perplexity-centric; no sub-1B deployment result.

## Related

- [[sources/papers/extracting-training-data]] — the memorisation/regurgitation dedup reduces
- [[sources/papers/membership-inference]] — leakage this hygiene mitigates
- [[sources/papers/what-should-llms-forget]] / [[sources/papers/unlearning-at-scale]] — the erase end of the arc
- [[sources/papers/refinedweb]] / [[sources/papers/fineweb]] — pipelines where aggressive dedup is standard
- [[topics/security-and-privacy]] — data hygiene as a privacy lever
- [[sources/papers/lima]] — data quality > quantity, the curation corollary

## Sources

- Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini (2021) — arXiv:2107.06499 (ACL 2022) — [arxiv.org/abs/2107.06499](https://arxiv.org/abs/2107.06499)
