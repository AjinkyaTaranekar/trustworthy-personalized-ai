---
title: "Training Verifiers to Solve Math Word Problems (GSM8K)"
type: source
tags: [evaluation, reasoning, rl]
sources:
  - https://arxiv.org/abs/2110.14168
  - https://github.com/openai/grade-school-math
updated: 2026-07-22
status: current
---

# Training Verifiers to Solve Math Word Problems (GSM8K)

**Introduces GSM8K — an 8.5K-problem grade-school math word-problem benchmark — and shows that training a separate verifier to rank many sampled solutions scales far better with data than plain finetuning: a 6B model with verification roughly matches a finetuned 175B, an ~30× effective parameter saving.**

## Summary

Cobbe et al. (OpenAI, 2021) contribute two things the field still uses daily: the GSM8K dataset (7.5K train / 1K test, human-written, 2–8 steps, natural-language solutions) and the generate-and-verify method — sample many candidate solutions, then pick the one a trained verifier scores highest. Autoregressive generation cannot undo a wrong step, but *judging* a finished solution is easier than producing a flawless one, so allocating test-time compute to sampling-plus-ranking beats one greedy chain. The headline "6B verification ≈ finetuned 175B" is the origin of the outcome-reward / solution-ranking idea that later underpins RLHF-style verifiable rewards and [[entities/grpo|GRPO]]. Its enduring caution: outcome-only labels credit correct answers reached by *unfaithful* reasoning — the exact failure the trust/verifiability agenda must guard against.

## Why it matters here

Load-bearing as the **definitional citation for GSM8K** — the verifiable-reward benchmark the pipeline uses — and as the **origin of outcome-reward verification** that the project's RL/SFT lineage builds on. The trust angle is explicit: outcome-only rewards credit lucky-but-wrong reasoning, so a verifier is not a genuine reasoning checker. "6B verification ≈ finetuned 175B" is a clean precedent that inference-time selection substitutes for parameters — directional support for the on-device small-model thesis.

## Method

- **Dataset:** high-quality, high-diversity, moderate-difficulty problems with natural-language (not bare-equation) solutions; calculator annotations for inference.
- **Verifier:** generator samples 100 solutions/problem, labelled by final-answer correctness; verifier predicts correctness. Token-level verification + a joint LM objective + 20% dropout beat solution-level; generator and verifier kept separate.
- **Test-time:** sample N candidates, rank by verifier, return the top (majority-voting the top-ranked adds gains).

## Key results

- **6B:** finetuning ~40% (100 samples) → **verification ~51%**; **175B verification ~58%**; 6B-verification ≳ finetuned-175B (~30× parameter saving). *(Several figures are read from plots, not tables — treat as approximate.)*
- **Reasoning is load-bearing:** removing natural-language reasoning drops 6B from 20.6% → 5.2%.
- **Optimal sample budget:** verifier accuracy peaks ~400 samples then degrades as search "fools" it.

## Critical appraisal

Foundational — it created the benchmark used everywhere for math reasoning and articulated the generate-and-verify idea seeding later verifiable-reward RL. Its weakness is exactly the trust-relevant one: outcome-only labels reward correct answers regardless of faithful reasoning (an early reward-hacking signature), and small verifiers rely on coarse heuristics rather than checking each step.

> ⚠ 0.6B caution: absolute scores are low by modern standards (2021 GPT-3-scale), and the smallest model is 6B — cite the ~30× parameter-saving multiplier as *directional*, not a 0.6B guarantee.

## Related

- [[sources/papers/deepseekmath]] — GRPO's group-baseline verifiable reward descends from this
- [[sources/papers/dapo]] — integer-answer rule-based rewards, same verifiability lineage
- [[sources/papers/qwen25-math]] — RM@N inference selection; verifier + RM fused in GRPO
- [[sources/papers/instructgpt]] — the RLHF reward-model line this seeds
- [[topics/reasoning]] — outcome vs process reward; faithful reasoning
- [[sources/code/training-and-benchmark]] — where GSM8K is scored in the pipeline

## Sources

- Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, Hilton, Nakano, Hesse, Schulman (2021) — arXiv:2110.14168 — [arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
