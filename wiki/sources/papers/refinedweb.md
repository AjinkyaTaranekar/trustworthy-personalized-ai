---
title: "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only"
type: source
tags: [foundations]
sources:
  - https://arxiv.org/abs/2306.01116
updated: 2026-07-22
status: current
---

# The RefinedWeb Dataset for Falcon LLM

**Common Crawl, if filtered and deduplicated aggressively enough, yields a web-only English corpus (RefinedWeb, ~5T tokens) that trains models matching or beating those trained on curated mixtures like The Pile — so large-scale curation may be unnecessary.**

## Summary

Penedo et al. (TII, NeurIPS 2023) argue curation is labour-intensive, legally fraught, and unscalable, so the bottleneck is *quality* not *source*. Their MacroData Refinement pipeline (trafilatura extraction, URL/language filtering, MassiveText-style heuristics, then very aggressive fuzzy MinHash + exact-substring + URL dedup removing ~half the documents) produces web-only data that beats The Pile at matched compute *despite containing no Wikipedia, arXiv, books, or code*. It flipped the field's default from "curate a diverse mixture" to "filter and deduplicate the web hard enough and you don't need curation", backed by the real Falcon model line. The durable lesson: deduplication aggressiveness, not exotic filtering, drives most of the gain. BACKGROUND on data quality/provenance.

## Why it matters here

The clearest statement of data-quality-beats-quantity / curation-may-be-unnecessary — support for the framing that what a model becomes is set by data *hygiene* (filtering + dedup) more than hand-curated diversity, i.e. build a small, rigorously filtered constitutional dataset rather than a large messy one. Its dedup machinery is the same mechanism [[sources/papers/deduplicating-training-data|Lee et al.]] tie to reduced memorisation, so RefinedWeb doubles as evidence that strong dedup is now standard practice — indirectly serving the privacy/leakage sub-theme.

## Key results

- **Scale:** ~5T tokens (600B public extract); ~50% of documents removed by dedup; ~12–14% net retention of Common Crawl.
- **small-agg zero-shot:** RefinedWeb 3B@60GT ~59.8% vs C4 59.6%, OSCAR 59.1%, The Pile 57.9% — edges curated corpora at matched scale.
- **Falcon-RW-7B (350GT)** matches GPT-3 on the curated-benchmark aggregate.
- **Dedup ablation:** +0.6% to +2.9% accuracy across datasets from deduplication alone.

*(Numbers via ar5iv extraction — verify decimals against source.)*

## Critical appraisal

The paper that flipped the field's default, backed by a frontier model line — its most durable technical lesson is that dedup aggressiveness drives the gain. Caveats matter for honest citation: the full 5T set is private (limiting verification), the strongest evidence is ≤7B on modest token budgets (so "web-only rivals curated" is best supported at small-to-mid scale), and heuristic filtering choices make it a strong *existence proof* rather than a controlled study — the gap [[sources/papers/fineweb|FineWeb]] closed a year later with systematic ablations.

> Note: no on-device/sub-1B focus (models 1B–7B+, argument about corpus construction) — cite for provenance/quality, not the small-model deployment thesis.

## Related

- [[sources/papers/fineweb]] — the systematic-ablation successor (same team)
- [[sources/papers/the-pile]] — the curated-mixture pole this rebuts
- [[sources/papers/deduplicating-training-data]] — dedup → less memorisation (privacy link)
- [[sources/papers/lima]] — quality > quantity, the SFT corollary
- [[topics/llm-foundations]] — data hygiene as the real lever
- [[topics/security-and-privacy]] — dedup as standard hygiene

## Sources

- Penedo, Malartic, Hesslow, Cojocaru, Cappelli, Alobeidli, Pannier, Almazrouei, Launay (2023) — arXiv:2306.01116 (NeurIPS 2023 D&B) — [arxiv.org/abs/2306.01116](https://arxiv.org/abs/2306.01116)
