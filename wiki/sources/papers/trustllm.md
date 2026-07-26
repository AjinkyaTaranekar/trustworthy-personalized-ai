---
title: "TrustLLM: Trustworthiness in Large Language Models"
type: source
tags: [security, evaluation, alignment, privacy]
sources:
  - https://arxiv.org/abs/2401.05561
  - https://trustllmbenchmark.github.io/TrustLLM-Website/
updated: 2026-07-20
status: current
---

# TrustLLM: Trustworthiness in Large Language Models

**Trustworthiness is not a single axis but an eight-dimension construct, and a unified benchmark over 16 models and 30+ datasets shows capability and trustworthiness tend to rise together, proprietary models generally lead open-source, and over-alignment (refusing benign prompts) is a pervasive failure mode.**

## Summary

Huang, Sun et al. (Lehigh + 70-author consortium, 2024) derive an eight-dimension taxonomy from a ~600-paper review — truthfulness, safety, fairness, robustness, privacy, machine ethics (the six operationalised), plus transparency and accountability (qualitative) — and build the first integrated benchmark across 16 mainstream LLMs. Headline findings: capability roughly tracks trustworthiness, proprietary models generally win but Llama2 is a strong open exception, and over-alignment is systemic (Llama2-7B refuses **57%** of benign prompts). The durable contribution is the *taxonomy and vocabulary*, not any single 2024 leaderboard number — it's a ready backbone for framing "trustworthy personalised on-device AI".

## Why it matters here

The six operationalised dimensions are a ready-made scaffold for the dissertation's trustworthiness framing — adopt/adapt them, then argue which matter most for a *personalised on-device assistant* (privacy and safety rise in weight; truthfulness/robustness under user-specific context). The **57% benign-refusal** over-alignment figure is a direct warning for a 0.6B constitutional assistant whose in-model constitution could over-trigger refusal — motivating the project's clarify-before-assume / substance-based-judge stance (don't reward blanket refusal). Its transparency critique (proprietary models hide *how* trust is achieved) is an argument *for* an auditable, published constitution baked in via SFT.

## Method

- **Eight dimensions** (six operationalised): truthfulness (hallucination, sycophancy, adversarial factuality), safety (jailbreak, toxicity, misuse), fairness (stereotype, disparagement, preference), robustness (perturbation, OOD), privacy (awareness + PII leakage on Enron), machine ethics.
- 16 models (GPT-4, ChatGPT, Claude, Llama2 7/13/70B, Vicuna, …), 30+ datasets, 18+ subtasks; scored via exact-match, classification, refusal-rate, semantic similarity, correlation-with-human.

## Key results

- **Over-alignment:** Llama2-7B refuses 57% of benign prompts.
- **Fairness:** best model GPT-4 only ~65% stereotype-recognition accuracy.
- **Privacy:** ChatGPT best awareness (Pearson r≈0.665 vs human); nearly all models leak some PII.
- **Safety:** open-source lags proprietary on jailbreak/toxicity/misuse.
- Capability ≈ trustworthiness; proprietary lead, with Llama2 the open-weight exception.

## Critical appraisal

The lasting value is the orthogonal, literature-grounded taxonomy — excellent as an evaluation scaffold. Weaknesses are those of all mega-benchmarks: a Jan-2024 snapshot that dates fast (no Gemini/Claude-3/GPT-4o), contamination risk, and metric proxies that can reward pathologies (a high *refusal* rate scores "safe" yet *is* the over-alignment the paper criticises). Lean on the framework and the over-alignment finding, not individual decimals (the digest flagged some as machine-extracted — confirm before quoting verbatim).

## Related

- [[sources/papers/adversarial-attacks-zou]] — operationalises the safety/robustness dimension (GCG jailbreak)
- [[sources/papers/ignore-previous-prompt]] — safety + privacy (prompt/PII leakage) dimension
- [[sources/papers/general-language-assistant]] — the HHH framing TrustLLM decomposes further
- [[sources/papers/hallucination-survey]] — the truthfulness dimension
- [[entities/constitution]] — the in-model trust mechanism vs opaque post-hoc filtering
- [[topics/security-and-privacy]] — the dissertation's trust/threat framing
- [[experiments/human-evaluation-rubric]] — trust dimensions as scored axes

## Sources

- Huang, Sun et al. (2024) — arXiv:2401.05561 (ICML 2024 position) — [arxiv.org/abs/2401.05561](https://arxiv.org/abs/2401.05561)
