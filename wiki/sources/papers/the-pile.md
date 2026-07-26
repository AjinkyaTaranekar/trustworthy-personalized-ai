---
title: "The Pile: An 800GB Dataset of Diverse Text for Language Modeling"
type: source
tags: [foundations]
sources:
  - https://arxiv.org/abs/2101.00027
updated: 2026-07-22
status: current
---

# The Pile: An 800GB Dataset of Diverse Text for Language Modeling

**Mixing 22 smaller, high-quality, topically diverse datasets into a single 825 GiB English corpus improves cross-domain knowledge and downstream generalisation beyond training on Common Crawl alone.**

## Summary

Gao et al. (EleutherAI, 2021) assemble a deliberately diverse pretraining corpus (14 of 22 components newly built — PubMed Central, arXiv, GitHub, FreeLaw, Stack Exchange, etc.) on the premise that model quality is bounded by data *diversity*, not just volume. Models trained on the Pile beat both raw and filtered Common Crawl at matched scale, with the largest gains on academic writing, code, and specialised domains. It was the default open pretraining corpus for years (GPT-Neo/J/NeoX) and set a transparency norm (documented provenance, licensing, bias). Mostly BACKGROUND — the canonical "curate a diverse mixture" pole that [[sources/papers/refinedweb|RefinedWeb]] and [[sources/papers/fineweb|FineWeb]] later push back against.

## Why it matters here

Anchors the data-quality-vs-quantity narrative that motivates a small, high-signal constitutional dataset (quality and domain-fit over sheer volume). Its bias, consent, and PII admissions (Enron Emails, Books3) give the privacy sub-theme a concrete hook: web corpora ship with memorisable personal and copyrighted content by default — exactly what the [[sources/papers/deduplicating-training-data|dedup]]/leakage papers address.

## Key results

- **Scale:** 825.18 GiB raw → 1,254 GiB effective (higher-quality components up-weighted, e.g. Wikipedia 3 epochs).
- **Size-controlled training (Pile vs CC-100 vs raw CC):** Pile reaches val/test bpb 0.928/0.943, LAMBADA acc 50.1% — beating CC-100 and raw CC, with large margins on specialised domains (arXiv 0.79 vs 1.82 bpb; GitHub 0.56 vs 1.65).
- Uses bits-per-byte (tokenisation-invariant) over perplexity.

*(Numbers via ar5iv extraction — verify decimals against source.)*

## Critical appraisal

Historically pivotal — it made open LLM reproduction possible and its documentation set a lasting norm. But its methodological weaknesses are the flip side of its ambition: heuristic per-component weighting with no ablation isolating each contribution, evaluation confined to bpb/perplexity, and "diversity helps" not disentangled from the confound that curated domains are simply cleaner than raw CC.

> ⚠ Consent/legal: **Books3** (pirated books) has since become legally radioactive, so the corpus is not cleanly reusable today; Enron Emails carry PII. No on-device/sub-1B angle (models up to NeoX-20B) — cite for provenance and diversity framing.

## Related

- [[sources/papers/refinedweb]] — "web-only can rival curation" (the counter-thesis)
- [[sources/papers/fineweb]] — the ablation-disciplined successor
- [[sources/papers/deduplicating-training-data]] — the PII/memorisation concern it under-examines
- [[sources/papers/extracting-training-data]] — memorisable content shipped by default
- [[sources/papers/gpt3-few-shot]] — the scale moment the Pile responded to
- [[topics/llm-foundations]] — corpus construction and provenance

## Sources

- Gao, Biderman, Black, Golding, Hoppe, Foster, Phang, He, Thite, Nabeshima, Presser, Leahy (2021) — arXiv:2101.00027 — [arxiv.org/abs/2101.00027](https://arxiv.org/abs/2101.00027)
