---
title: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
type: source
tags: [evaluation, llm-as-judge]
sources:
  - https://arxiv.org/abs/2306.05685
updated: 2026-07-18
status: current
---

# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**A strong LLM (GPT-4) can serve as a scalable, cheap, explainable stand-in for human judges on open-ended multi-turn chat quality — reaching ~85% agreement with humans, on par with human–human agreement (~81%) — provided its known biases (position, verbosity, self-enhancement, weak math grading) are recognised and mitigated.**

## Summary

Zheng et al. (LMSYS / Berkeley, NeurIPS 2023) provide the founding empirical justification for "LLM-as-a-judge." Because capability benchmarks (MMLU, HELM) miss the open-ended, instruction-following qualities users care about, and human evaluation does not scale, they measure how closely a frontier judge tracks human preference and dissect where it fails. GPT-4 as judge exceeds 80% agreement with human experts (~85% without ties) — the same level as human–human agreement — and they release two durable benchmarks (MT-Bench: 80 curated multi-turn questions; Chatbot Arena: crowdsourced pairwise battles). Crucially they quantify and give mitigations for four judge biases. This is the core methodological foundation for the project's automated evaluation harness.

## Why it matters here

The dissertation grades trust/empathy/safety/helpfulness with an LLM judge, and this is the canonical evidence that such a judge is human-aligned enough to justify automated grading — while mandating bias controls. It underwrites the project's preference for a substance-based LLM judge over single-word regex ([[experiments/human-evaluation-rubric]], [[experiments/frontier-model-comparison]]), and every bias here maps to a concrete safeguard the harness needs.

## Method

- **Three judge protocols:** pairwise comparison, single-answer grading, reference-guided grading (a reference solution supplied for math).
- **MT-Bench:** 80 multi-turn questions across eight categories (writing, roleplay, extraction, reasoning, math, coding, STEM, humanities).
- **Chatbot Arena:** anonymous head-to-head battles, ~30K votes in month one.
- **Agreement metric:** probability two random non-identical judges agree on a random question, reported with/without ties; ~3K expert pairwise votes collected.
- **Bias probes:** position (answer-swap), verbosity (a "repetitive list" padding attack), self-enhancement (does a model prefer its own outputs), limited reasoning (math grading with/without CoT and references).

## Key results (and the safeguards they imply)

- **Agreement:** GPT-4 ~85% with humans (≈ human–human 81%); highest when quality gap is large (~100%), softest on close calls (~70%).
- **Position bias:** GPT-4 65% consistent (30% biased to first answer), 77.5% with few-shot; Claude-v1 only 23.8%. → **swap answer order and average.**
- **Verbosity bias:** GPT-4 fooled 8.7% by padding vs 91.3% for Claude-v1/GPT-3.5. → a safety-tuned small model that adds caveats/warnings inflates length; **use a verbosity-robust judge and/or penalise padding.**
- **Self-enhancement:** GPT-4 +10% own-answer preference, Claude-v1 +25%. → **avoid a judge sharing the student's model family** (e.g. a Qwen judge grading Qwen3-0.6B).
- **Math grading:** default fails 14/20; reference-guided cuts to 3/20 (15%). → **use reference-guided grading for correctness.**

## Critical appraisal

The most-cited, rigorously-probed justification for LLM-as-a-judge, with released benchmarks and vote data and honest failure-mode accounting. Cautions: it is 2023-era (GPT-4 is the gold judge; MT-Bench's 80 questions saturate for modern models); reliability of *small* judges is not the focus; and open-ended judging has a soft ceiling (humans agree only ~81%).

> ⚠ Caution: do not port the exact 85% agreement to a small or same-family judge. The project should report its judge model, apply the bias mitigations above, and ideally include a human- or inter-judge-agreement calibration for its own setup.

## Related

- [[experiments/human-evaluation-rubric]] — the external human ground truth the LLM judge is calibrated against
- [[experiments/frontier-model-comparison]] — where the automated judge scores Qwen3-0.6B vs frontier models
- [[experiments/sft-benchmark-analysis-20260525]] — benchmark analysis that consumes judge scores
- [[sources/papers/reducing-safety-tax]] — uses Llama-Guard-style classifier scoring; complementary eval machinery
- [[sources/papers/none-of-the-others]] — reasoning-vs-memorisation evaluation caveat
- [[topics/explainability]] — judge explanations and scrutability

## Sources

- Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica (2023) — arXiv:2306.05685 (NeurIPS 2023) — [arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
