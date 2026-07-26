---
title: "CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification"
type: source
tags: [tool-use, verification, sft, small-model]
sources:
  - https://arxiv.org/abs/2603.01940
updated: 2026-07-19
status: current
---

# CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification

**Replace expensive, noisy LLM-judge evaluation of tool-use trajectories with a rule-based verifier grounded in explicit task constraints sampled directly from the sandbox database — constraints then "fuzzified" into realistic ambiguous requests — yielding deterministically-verifiable, solvable-by-construction trajectories that let a 4B model match models ~17× its size on τ²-bench using only ~12k trajectories.**

## Summary

Chen et al. (Huawei Research, 2026) tackle two coupled problems in interactive tool-use data: *solvability* (synthetic tasks may have no valid solution) and *verifiability* (LLM judges are costly and unreliable). CoVe samples ground-truth constraints from *real database records* (so a solution provably exists), fuzzifies precise identifiers into human-like ambiguous descriptions (an Order ID becomes a subset of items; a User ID becomes an email — forcing interactive clarification), then verifies trajectories with a deterministic rule `V(τ, C)` that checks all constraints are satisfied *and* free of redundancy. CoVe-4B reaches 51.2% pass¹ on τ²-bench — matching xLAM-2-70b (51.5%) — and its 5k verified trajectories beat 90k Simia-derived data. This is arguably the strongest single reference for the project's tool-call verification pillar.

## Why it matters here

CoVe makes tool-call correctness *provable* by grounding it in DB-sampled constraints and checking with deterministic rules — the gold standard for a trustworthy thesis, sidestepping the LLM-judge fragility [[sources/papers/toolmind]] still relies on. The rule verifier `V(τ,C)` (all constraints satisfied AND redundancy-free) is a template for verifying a small [[experiments/thinker-executor-experiment|Executor's]] tool calls without a large on-device judge, and fuzzification forces exactly the clarify-before-assume, multi-turn interaction the project values. The "redundancy-free" criterion doubles as an on-device efficiency axis (fewer calls = less compute/battery).

## Method

- **Constraint sampling:** constraints read from the sandbox DB → tasks solvable by construction.
- **Constraint fuzzification:** precise IDs → ambiguous descriptions with a deterministic mapping back, so the agent must interactively de-fuzzify.
- **Verification:** rule-based `V(τ,C)` scores whether tool executions satisfy all constraints and are redundancy-free → score `Sτ`.
- **Training:** SFT on score-1 trajectories only; RL feeds `Sτ` directly as reward. 12k trajectories released. Base Qwen3-4B → CoVe-4B.

## Key results

- **τ²-bench pass¹:** Airline 43.0, Retail 59.4, avg **51.2** (+18.6 over base) — matches xLAM-2-70b (51.5), beats xLAM-2-32b (49.5).
- **Data efficiency:** CoVe-5K 44.7 > Simia-5K 39.7; CoVe beats Simia-90K using ~5.5% of the volume.
- **Honest negative result:** SFT+RL *underperforms* pure SFT, attributed to a weak open-weight user simulator.

## Critical appraisal

The most trustworthy-verification-flavoured of its cluster: correctness is provable, deterministic and cheap, and the 4B≈70B / 5K>90K results are strong evidence for verified-data sample efficiency. Honest limitations: it needs a *structured sandbox DB* to sample constraints from (may not transfer to open-web unstructured tools); rule verifiers are hand-designed per domain; only two τ²-bench domains tested.

> ⚠ 0.6B caution & pivot support: floor is 4B, so 4B≈70B may not hold at 0.6B; the verification-*at-training* recipe transfers on-device but verification-*at-inference* over real unstructured tools does not directly (no ground-truth DB). The SFT+RL regression from a weak simulator directly **supports the SFT-only / no-GRPO pivot** ([[decisions/2026-05-03-research-question-reframe]]).

## Related

- [[sources/papers/toolmind]] — LLM-judge verification; CoVe's deterministic verifier is stronger
- [[sources/papers/t1]] — sandbox execution-grounded verification
- [[experiments/thinker-executor-experiment]] — the Executor whose tool calls need verifying
- [[decisions/2026-05-03-research-question-reframe]] — the SFT-only pivot CoVe's RL regression supports
- [[topics/tool-use-and-verification]] — constraint-guided, redundancy-aware verification
- [[topics/reasoning]] — verifiable rewards over learned reward models

## Sources

- Chen, Gong, Li, Liu, Tian, Fu, Wu, Zhang, Zhang, Zhang, Tu, Liu (2026) — arXiv:2603.01940 — [arxiv.org/abs/2603.01940](https://arxiv.org/abs/2603.01940)
