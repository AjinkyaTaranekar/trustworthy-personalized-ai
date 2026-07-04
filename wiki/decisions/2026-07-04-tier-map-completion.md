---
title: Tier-map completion — H2b and P22_scratchpad added to the canonical principle map
type: decision
tags: [constitution, evaluation, judge, personalisation]
sources:
  - pipeline/principle_families.py
  - pipeline/constitution.md
updated: 2026-07-04
status: current
---

**The canonical principle map (`principle_families.py`) was completed with the two probed items it was missing — `H2b_memory_retention_multiturn` (tier 2, personalisation) and `P22_scratchpad_multistep` (tier 3, tool) — after a score audit showed the change is mechanical completeness, not weight tuning.**

## Summary

An audit of `principle_families.py` against `constitution.md` and the actual probe suite found that two items scored by `4_benchmark.py` were absent from both `PRINCIPLES` and `PRINCIPLE_TIER`. The absence had two silent effects: `family_of()` returned "unknown" so both items dropped out of every per-family aggregate, and `tier_of()` fell back to tier 3 (weight ×1), which was wrong for H2b — multi-turn memory retention is the same personalisation substrate as its sibling `H2_memory_persistence` (tier 2). The map now carries all 23 scored items, `PRINCIPLE_COUNT_NOTE` states the precise 25-defined / 23-scored / 21-covered accounting, and P24 was removed from `UNPROBED_PRINCIPLES` because the probe named "P22_scratchpad_multistep" actually tests constitution P24 SCRATCHPAD-FIRST (a probe-suite naming collision, now documented in the map).

## Decision

- `H2b_memory_retention_multiturn` → tier 2 (×2), family personalisation — matches H2; multi-turn retention after a distractor is memory substrate, per the tier rationale fixed on 2026-06-25.
- `P22_scratchpad_multistep` → tier 3 (×1), family tool — scratchpad usage is instrumental mechanism; this formalises the tier it already received via fallback, so only its family placement (and therefore family aggregates) changes.
- No other tier moved. The one arguable assignment identified in the audit — `P14_hold_pressure` at tier 2 although its failure mode is producing a false answer (a tier-1 concern) — was deliberately left as decided on 2026-06-28; moving it would need its own decision.
- A 5-tier scheme and alternative weightings were considered and rejected: with 23 scored items, five tiers give ~4–5 items per stratum (per-tier means become noise) and every extra boundary adds a researcher degree of freedom that weakens the a-priori defence. The planned robustness move is a weight sensitivity analysis (×3/×2/×1 vs ×2/×1.5/×1 vs unweighted) in the dissertation instead.

## Score audit (integrity check, run before adopting)

Purpose-weighted constitution score per condition, old map (H2b and P22 at fallback tier 3) vs new map, on the current judged reports. Only H2b's weight changes (×1 → ×2):

| condition | old weighted | new weighted | delta |
|---|---|---|---|
| vanilla_base | 0.3560 | 0.3555 | −0.0005 |
| vanilla_tools | 0.4202 | 0.4184 | −0.0018 |
| sft_template | 0.1610 | 0.1632 | +0.0022 |
| sft_constitution | 0.4699 | 0.4618 | −0.0081 |
| thinker_executor | 0.3617 | 0.3611 | −0.0006 |

All deltas ≤ 0.008, the condition ordering is unchanged (sft_constitution > vanilla_tools > thinker_executor > vanilla_base > sft_template), and the largest single effect is *negative* for sft_constitution (its H2b score is low, and H2b now weighs more) — the change cannot be read as tuning the weights toward the trained model. Family aggregates gain the two previously dropped items (21 → 23).

## Related

- [[decisions/2026-06-28-reasoning-tier-weighting]] — the prior tier move this follows procedurally
- [[sources/code/training-and-benchmark]] — comparative judge + weighted rank aggregates that consume the map
- [[entities/constitution]] — the constitution document itself

## Sources

- pipeline/principle_families.py (the map)
- pipeline/constitution.md (the 25 defined principles)
