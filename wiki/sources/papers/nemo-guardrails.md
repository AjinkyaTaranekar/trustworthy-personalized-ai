---
title: "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails"
type: source
tags: [guardrails, security]
sources:
  - https://arxiv.org/abs/2310.10501
updated: 2026-07-21
status: current
---

# NeMo Guardrails: Controllable and Safe LLM Applications with Programmable Rails

**Programmable, runtime "rails" defined in a purpose-built dialogue language (Colang) let developers add interpretable, model-agnostic safety and topical controls to any LLM application as a proxy layer — without retraining the model.**

## Summary

Rebedea et al. (NVIDIA, EMNLP 2023 demo) contrast *embedded* rails (safety trained into weights via RLHF — opaque, fixed, model-specific) with *programmable* rails (defined at runtime, LLM-agnostic, editable without retraining). Colang models dialogue as canonical forms (LLM-generated intents indexed in a vector store), flows (dialogue policy), and execution rails (Python safety actions). A three-step CoT loop maps utterance → canonical form → next step → bot message. Combined input+output moderation blocks 99% of harmful prompts at 2% false-positive on GPT-3.5; a self-consistency hallucination rail lifts false-premise deflection 65%→95%. The honest catch: it is a *complement* to (not replacement for) in-weights alignment, and the sequential CoT costs ~3× latency. This is the canonical external-control counterpoint to an in-model constitution.

## Why it matters here

NeMo is the reference *external* safety approach; the project's constitutional harness is the *internal/in-context* alternative — this paper supplies the explicit contrast (embedded vs programmable rails) and the trade-offs to argue in the safety chapter: interpretability + editability vs 3× latency, bypassability, and dependence on the base model. A constitution-in-prompt on a small model aims to get the editability of programmable rails *without* a separate 3× proxy pipeline. Its moderation protocol (balanced harmful/helpful set; report both harmful-blocked and helpful-blocked) is a reusable template that mirrors the [[sources/papers/abstention-survey|abstention]] URUP/ARSP trade-off.

## Method

- **Colang:** canonical forms (dynamic LLM-generated intents), flows (branching dialogue policy), execution rails (Python actions), context state; canonical forms embedded in a vector DB (Annoy/FAISS) for similarity few-shot.
- **Runtime three-step CoT:** generate user canonical form → decide next step (Colang flow or LLM generalisation) → generate bot message under rail constraints. Rail types: input/output moderation, topical, fact-checking, hallucination, retrieval.

## Key results

- **Topical rails (Banking77, 77 intents):** text-davinci-003 0.77/0.83 intent/next-step; Falcon-7B 0.70/0.75; needs ≥k=3 samples per canonical form.
- **Moderation (200 balanced):** GPT-3.5 with input+output moderation blocks 99% harmful at 2% false-positive (vs 93%/0% without).
- **Hallucination rail:** GPT-3.5 false-premise deflection 65% → 95%.
- **Cost:** ~3× latency and cost from three sequential LLM calls.

## Critical appraisal

The reference implementation of "safety as a separable, programmable runtime layer", and Colang's canonical-form-over-vector-store design is a clean, inspectable alternative to fixed intent classifiers. Its honesty about being a complement to in-weights alignment is a strength. But the evaluation is demo-grade — small bespoke/synthetic sets (200 moderation samples, 20 hallucination questions), no strong adversarial jailbreak testing despite claiming jailbreak detection — so numbers are feasibility evidence, not guarantees. The 2–5% helpful-blocked false-positive rate is an untuned over-refusal cost, and an external proxy can be bypassed if the app path around it isn't controlled.

> ⚠ 0.6B / on-device: the ~3× latency + vector store + multiple sequential LLM calls make the full NeMo pattern heavy for a sub-1B setting — motivating a lighter single-pass constitutional prompt over an external multi-call proxy. That it worked with Falcon-7B (0.70–0.75) shows rail-style control degrades gracefully at small scale, but per-turn cost argues for folding the policy into the model's own context.

## Related

- [[sources/papers/trustllm]] — the safety dimension; over-alignment / over-refusal
- [[sources/papers/abstention-survey]] — inference-time refusal (cheap but bypassable); ARSP/URUP
- [[sources/papers/ignore-previous-prompt]] — jailbreak/injection the rails aim to catch
- [[entities/constitution]] — in-model constitution vs external rails
- [[topics/security-and-privacy]] — layered defence; post-hoc vs in-weights
- [[sources/dissertation/security-privacy-social-ethics]] — the critique-loop SPOF discussion

## Sources

- Rebedea, Dinu, Sreedhar, Parisien, Cohen (NVIDIA, 2023) — arXiv:2310.10501 (EMNLP 2023 demo) — [arxiv.org/abs/2310.10501](https://arxiv.org/abs/2310.10501)
