---
title: "Universal and Transferable Adversarial Attacks on Aligned Language Models"
type: source
tags: [security]
sources:
  - https://arxiv.org/abs/2307.15043
  - https://llm-attacks.org
updated: 2026-07-20
status: current
---

# Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)

**A single adversarial suffix, found by greedy coordinate gradient (GCG) optimisation against a few open-source chat models, can be appended to almost any harmful request to jailbreak them — and the same suffix transfers black-box to ChatGPT, PaLM-2/Bard, and (weakly) Claude, showing RLHF alignment does not remove the adversarial attack surface.**

## Summary

Zou et al. (CMU / CAIS / DeepMind / Bosch, 2023) turn jailbreaking from artisanal prompt-craft into gradient optimisation. GCG combines an *affirmative-response* target (make the model begin "Sure, here is how to…"), a discrete greedy-coordinate-gradient token search over the suffix, and multi-prompt/multi-model joint optimisation to produce one universal, transferable suffix. White-box ASR reaches 99% on Vicuna-7B behaviours (56% on LLaMA-2-7B-Chat); the ensemble suffix transfers to GPT-3.5 (86.6%), GPT-4 (46.9%), PaLM-2 (~66%), but only ~2.1% to Claude-2. The deep result is *transfer*: a suffix built on small open models breaks much larger closed ones — alignment is a breakable outer layer, not a fix. This is the canonical automated-jailbreak reference and one half of the project's adversarial suite.

## Why it matters here

GCG is precisely the automated, transferable jailbreak class the project's adversarial/jailbreak suite should include. Because suffixes transfer *from small open models*, a 0.6B on-device model is a plausible both-target-and-source (attackable directly, and a cheap surrogate to craft suffixes against larger models). Claude-2's ~2.1% ASR (input filtering + strong alignment) is the key data point for the project's central design argument — **layered defence (constitution baked in + serve-time filtering) beats naive RLHF** — and motivates testing whether an in-model constitutional harness on a 0.6B measurably lowers GCG ASR versus a vanilla-aligned baseline.

## Method

- **Affirmative-response target:** optimise the suffix so the model *starts* with a compliant prefix; once committed, it tends to continue harmfully.
- **Greedy Coordinate Gradient:** rank top-k token substitutions at every suffix position via one-hot gradients, batch-evaluate candidate swaps, greedily take the best (searches all coordinates, unlike AutoPrompt).
- **Multi-prompt/multi-model:** jointly optimise over 25 behaviours and multiple source models (Vicuna-7B/13B, Guanaco) → behaviour-general, transferable suffix. Benchmark: AdvBench (500 harmful strings + 500 behaviours).

## Key results

- **White-box ASR:** Vicuna-7B 99% (behaviours), LLaMA-2-7B-Chat 56%; vs AutoPrompt 25%, PEZ/GBDA 0%.
- **Transfer (ensemble):** GPT-3.5 86.6%, GPT-4 46.9%, PaLM-2 ~66%, **Claude-2 ~2.1%** (the robustness outlier).
- Over-optimising the suffix (too many steps) *reduces* transfer; ~30 s of manual tweaking pushed GPT-3.5 toward ~100%, so automated numbers are a lower bound.

## Critical appraisal

The canonical automated-jailbreak citation — its limits are practical (needs white-box source models, suffixes are patchable, Claude resists) not conceptual. The "reasonable-attempt" success criterion for behaviours is softer than exact-match (may inflate ASR), and transfer numbers are time-stamped as vendors add filters — but the *existence proof* (alignment as a breakable outer layer) is durable. For a defence-oriented thesis the value is the threat framing more than any 2023 percentage.

## Related

- [[sources/papers/ignore-previous-prompt]] — the low-tech natural-language injection counterpart
- [[sources/papers/trustllm]] — GCG-ASR operationalises its safety/robustness dimension
- [[sources/papers/reducing-safety-tax]] — adaptive-jailbreak robustness (PAIR) and its limits
- [[entities/constitution]] — in-model defence vs post-hoc guardrail
- [[topics/security-and-privacy]] — the adversarial threat model
- [[sources/code/training-and-benchmark]] — the project's adversarial suite

## Sources

- Zou, Wang, Carlini, Nasr, Kolter, Fredrikson (2023) — arXiv:2307.15043 — [arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043)
- Project — [llm-attacks.org](https://llm-attacks.org)
