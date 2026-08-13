---
title: "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale"
type: source
tags: [foundations]
sources:
  - https://arxiv.org/abs/2406.17557
updated: 2026-07-22
status: current
---

# The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

**By treating dataset design as an experiment — hundreds of small controlled training ablations to justify each pipeline choice — Hugging Face builds FineWeb (15T tokens) that outperforms other open web datasets, plus FineWeb-Edu (1.3T tokens), an educational subset dramatically better on knowledge/reasoning benchmarks and matching a ~10× larger corpus.**

## Summary

Penedo et al. (Hugging Face, NeurIPS 2024) make dataset curation reproducible science: each filtering decision (trafilatura extraction, base quality/language filters, per-snapshot MinHash dedup, three ablation-discovered heuristic filters) is validated by a 1.71B-proxy training run on 8 benchmarks, with the pipeline, ablation models, and data all released. FineWeb-Edu is distilled by a classifier trained on Llama-3-70B educational-quality scores. This is the most methodologically disciplined entry in the data-curation lineage and the most directly useful of the pretraining-data papers for the project's methodology — mostly BACKGROUND, but its "score-for-quality" method and ablation-per-decision discipline are transferable templates.

## Why it matters here

FineWeb-Edu is the sharpest published evidence for **data-quality-beats-quantity**: an LLM-scored quality classifier lets a small dataset match a ~10× larger one on reasoning tasks — a near-perfect analogy for building a small, high-signal constitutional dataset by *scoring for the traits you want* rather than scaling raw volume. The "validate every data decision with a small training run" methodology is a template the project can echo to justify constitutional-data choices, and the LLM-as-annotator classifier connects to the [[sources/papers/prometheus|LLM-judge]] line the project already uses.

## Key results

- **Scale:** FineWeb 15T, FineWeb-Edu 1.3T tokens (96 Common Crawl snapshots, ~36T after base filtering — *approximate*).
- **Edu gains (350B-token training):** MMLU ~33% → ~37%, ARC ~46% → ~57%; FineWeb-Edu matches a much larger corpus at ~10× fewer tokens.
- **Counter-intuitive:** *per-snapshot* dedup beats *global* dedup (global disproportionately strips high-quality recurring content).
- Ablation setup: 1.71B Llama-arch proxy, 28B–350B tokens/run, ~80k H100-hours across 70+ models.

*(Numbers via HTML extraction; intermediate dedup token counts flagged approximate — verify against source.)*

## Critical appraisal

Where The Pile *asserted* diversity and RefinedWeb *asserted* dedup, FineWeb makes dataset design falsifiable by tying each choice to a controlled ablation and open-sourcing the evidence — FineWeb-Edu operationalises "quality beats quantity" more convincingly than any predecessor. Honest limitation: proxy-scale ablations on multiple-choice benchmarks are a weak stand-in for frontier-scale, instruction-tuned, agentic behaviour (the authors say so), so rankings are strong hypotheses, not settled law. FineWeb-Edu also inherits Llama-3-70B's notion of "educational" (a provenance/bias issue).

> Note: BACKGROUND. The 1.71B ablation scale is closer to sub-1B than the other pretraining papers, but it is a corpus-construction paper — cite for quality/provenance and the score-for-quality method, not as a small-model deployment result.

## Related

- [[sources/papers/refinedweb]] — the web-only predecessor FineWeb re-ablates more rigorously
- [[sources/papers/the-pile]] — the curated-mixture pole
- [[sources/papers/deduplicating-training-data]] — the dedup hygiene it standardises (per-snapshot)
- [[sources/papers/phi1-textbooks]] — the extreme of the quality-over-quantity thesis
- [[sources/papers/prometheus]] — LLM-as-annotator quality scoring
- [[topics/llm-foundations]] — data quality and provenance

## Sources

- Penedo, Kydlíček, Ben Allal, Lozhkov, Mitchell, Raffel, Von Werra, Wolf (2024) — arXiv:2406.17557 (NeurIPS 2024 D&B) — [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)
