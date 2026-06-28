---
title: Reasoning principles moved to the Tier-2 substrate in the purpose weighting
type: decision
tags: [eval, reasoning, constitution]
sources:
  - pipeline/principle_families.py
  - docs/TCD_Dissertation/methodology-results/methodology-results.tex
updated: 2026-06-28
status: current
---

**P1 (decompose) and P20 (first-principles) moved from Tier 3 (×1, instrumental) to Tier 2 (×2, substrate) in the constitution's a-priori purpose weighting; Tier 3 is now purely tool principles.**

## Summary

The purpose-weighted "trustworthiness" headline weights the 21 scored constitution items into three a-priori priority tiers (Tier 1 ×3, Tier 2 ×2, Tier 3 ×1), fixed from the model's purpose and never tuned to results. An audit on 2026-06-28 found the **"reasoning" family was split across all three tiers** — P21→T1, P15→T2, P1/P20→T3 — which is hard to defend. P1 (decompose) and P20 (first-principles) were moved to Tier 2 so the reasoning *mechanism* {P1, P15, P20} sits together in the substrate tier, and Tier 3 becomes a clean, purely-instrumental **tool** tier. The structure is now: **trust OUTCOMES (T1) / cognitive + relational SUBSTRATE (T2) / instrumental TOOL mechanism (T3)**.

## Rationale

- **Coherence.** The old tiering put two reasoning principles (decompose, first-principles) in the "tool/reasoning mechanism" tier alongside tool-discipline probes, while a third (self-correction) was already in Tier 2 and a fourth (5W+H) in Tier 1. The family was incoherently scattered.
- **Outcome vs substrate vs mechanism.** Tier 1 keeps its sharp meaning — the OUTCOMES whose failure most directly damages trust (fabrication, false confidence, not asking, faking tools). Rigorous reasoning does not *directly* harm the user the way a fabrication does, so it is not Tier 1; but it is the durable SUBSTRATE that sustains those outcomes, which is exactly what Tier 2 holds (robustness under pressure, personalisation/memory). P21 (5W+H) legitimately stays in Tier 1 because it is framed as an outcome ("ask the right question"), not a mechanism.
- **Thesis fit.** This is a trustworthy-*reasoning* thesis; weighting reasoning as merely ×1 instrumental under-sold the protagonist. ×2 elevates it above pure tool use without over-claiming it is as trust-critical as not-fabricating.

## Alternative considered: Tier 1

The original suggestion was to move P1/P20 to Tier 1. Defensible under a *scrutability* notion of trust (on-device, the user reads the `<think>`, so visible rigorous reasoning is itself a trust artifact). Rejected because it dilutes Tier 1's "what most damages trust when wrong" definition by mixing process with outcomes, and would have required moving P15 up too to avoid re-splitting the family. Tier 2 fixes the coherence problem with a single rule and a cleaner outcome/substrate/mechanism story.

## Integrity audit (a-priori, not score-tuned)

The move was score-audited **before** committing, on the GLM-5.1-judged scores of all five conditions:

- **Ranking unchanged** under the move: `sft_constitution > vanilla_tools > thinker_executor > vanilla_base > sft_template` (identical to the current map and to the unweighted flat mean).
- **Deltas tiny and uniform** (+0.0018 to +0.0076 weighted), so no model is differentially flattered. T-E is worst on decompose (P1=0.17) but best on first-principles (P20=0.67), so the two cancel for it.
- This satisfies the standing safeguard: weights are fixed from the purpose statement and never tuned to the scores.

Under the corrected weighting the per-tier breakdown also surfaces a real finding — sft_constitution's Tier-2 substrate score (0.606) is well above its Tier-1 (0.407) and Tier-3 (0.483): the constitutional model's strength is concentrated in the reasoning/pressure/personalisation substrate, which the old tiering hid inside the tool tier.

## Files touched

- `pipeline/principle_families.py` — `PRINCIPLE_TIER` (P1, P20 → 2), tier rationale comment, `TIER_LABELS`.
- `docs/TCD_Dissertation/methodology-results/methodology-results.tex` — `tab:tiers` table rows + the EXPLAIN lead-in (commented draft).

## Related

- [[sources/code/training-and-benchmark]]
- [[topics/reasoning]]
