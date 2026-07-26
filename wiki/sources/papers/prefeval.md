---
title: "Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs (PrefEval)"
type: source
tags: [personalisation, evaluation]
sources:
  - https://arxiv.org/abs/2502.09597
  - https://prefeval.github.io
updated: 2026-07-20
status: current
---

# PrefEval: Evaluating Personalized Preference Following in LLMs

**State-of-the-art LLMs largely fail to proactively infer, remember, and follow explicitly and implicitly stated user preferences across multi-turn conversations — zero-shot preference-following accuracy collapses below 10% by ten turns and toward zero at long context — unless externally reminded, retrieved, or fine-tuned.**

## Summary

Zhao et al. (UCLA / Amazon AGI, ICLR 2025 oral) build PrefEval: 3,000 preference-query pairs (20 topics × three preference forms — explicit, implicit choice-based, implicit persona-driven) embedded in multi-session contexts up to ~100k tokens, with a preference stated at turn *p* and an unrelated query at turn *q* (5/10/30/70/300 turns later). A preference stated once should govern all later relevant responses; PrefEval isolates that proactive binding that long-context and instruction-following benchmarks never test. The crisis is proactivity, not retrieval: models can retrieve a preference when reminded yet fail to apply it unprompted. This is the canonical evidence that preference following degrades over turns — the empirical spine of the personalisation-evaluation strand. **Distinct from Feng's [[sources/papers/rpeval|RPEval]] (2601.16621)** — cite carefully.

## Why it matters here

Its four-error taxonomy (Preference-Unaware / Hallucination / Inconsistent / Unhelpful) maps onto constitution principles about honouring stated preferences and is a ready LLM-judge rubric (substance-based, not regex). The SFT-generalises-across-lengths result is the strongest argument a sub-1B model could be fine-tuned to hold preferences, and the GPT-o1 reminder-recovery even at 103k tokens reframes the failure as proactivity/attention-allocation — motivating a reminder/harness mechanism at serve time. Any 0.6B result must be framed relative to this near-floor baseline (frontier models fail *zero-shot*), competing only under reminder/RAG/SFT.

## Method

- **Two tasks:** generation (free-form answer, judged by Claude 3 Sonnet over four binary error types; 5% human-checked error rate) and classification (4-option MCQ; correlates 0.73 with generation).
- **Five interventions:** zero-shot, Reminder, Self-Critic, Few-Shot CoT, RAG (top-5 retrieved exchanges). Filler turns from LMSYS-Chat-1M.

## Key results

- **Zero-shot collapse:** ~80% at low turns → below 10% at 10 turns (~3k tokens) → ~0% at 30–300 turns.
- **Reminder recovers some:** at 10 turns Claude 3.5 Sonnet 7%→45%, Gemini 1.5 Pro 7%→91%, GPT-o1-preview 50%→98%; at 300 turns only the reasoning model recovers (GPT-o1 14%→98%).
- **Intervention order (counterintuitive):** RAG best; Reminder > Self-Critic ≈ CoT. Implicit preferences harder than explicit.
- **SFT:** fine-tuning Mistral-7B on PrefEval beats RAG zero-shot and generalises 10-turn → 70-turn (attention to preference regions +4.97%).

## Critical appraisal

The single most citable result for personalisation evaluation: a first-class model can claim it "understands" your preference and ignore it two turns later. Well-constructed, with a genuine dual-task design. Cautions: the Claude judge evaluating Claude models is a mild circularity (partly defended by the human check + classification cross-check); "multiple preferences help" may be a prompt-salience artefact; LMSYS filler may not resemble a coherent real session.

## Related

- [[sources/papers/rpeval]] — Feng's RPEval (rational *use* of memory); distinct benchmark, cite separately
- [[sources/papers/op-bench]] — over-personalisation from memory augmentation
- [[sources/papers/context-length-hurts]] — the length-degradation that compounds this
- [[sources/papers/mem0]] — RAG/memory as the retrieval-side fix
- [[entities/5w-h]] — the user-preference model this evaluates
- [[topics/personalisation]] — preference following as the core capability
- [[experiments/human-evaluation-rubric]] — the four-error taxonomy as a judged rubric

## Sources

- Zhao, Hong, Liu, Hazarika, Lin (2025) — arXiv:2502.09597 (ICLR 2025 oral) — [arxiv.org/abs/2502.09597](https://arxiv.org/abs/2502.09597)
