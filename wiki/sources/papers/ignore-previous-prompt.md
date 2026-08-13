---
title: "Ignore Previous Prompt: Attack Techniques For Language Models"
type: source
tags: [security, privacy]
sources:
  - https://arxiv.org/abs/2211.09527
  - https://github.com/agencyenterprise/PromptInject
updated: 2026-07-20
status: current
---

# Ignore Previous Prompt: Attack Techniques for Language Models (PromptInject)

**LLM applications that build their prompt by string-substituting untrusted user input are vulnerable to simple, human-crafted "ignore previous instructions" injections that either hijack the app's goal or leak its hidden prompt — and, counter-intuitively, the larger/more instruction-tuned the model, the more susceptible it is.**

## Summary

Perez and Ribeiro (AE Studio, NeurIPS 2022 ML Safety Workshop) give the foundational prompt-injection paper. Because an app concatenates a trusted developer instruction with untrusted user text into one flat string with no privilege boundary, the model cannot tell "instruction" from "data" — a confused-deputy flaw inherent to prompting. Their PromptInject framework composes two attacks: **goal hijacking** (make the model emit an attacker-chosen rogue string) at 58.6% success, and **prompt leaking** (reveal the hidden base prompt) at 23.6%, on text-davinci-002. The **inverse-scaling** finding is the durable one: the smaller text-ada-001 succumbs only ~13.8% — instruction-following capability is itself the attack surface. This is the natural-language half of the project's adversarial suite, and prompt leaking maps directly onto constitution-leak resistance.

## Why it matters here

Alongside [[sources/papers/adversarial-attacks-zou|GCG]] (optimised suffixes), PromptInject supplies the *natural-language injection* threat — "ignore previous instructions", goal hijacking, and **prompt/constitution leaking** — a trustworthy on-device assistant must resist. Prompt leaking is directly on-point: if the assistant's constitution/system prompt is the trust mechanism, leaking it is both a confidentiality breach and a roadmap for further attack, so the adversarial suite should test constitution-leak resistance explicitly. Its mitigation — *separate instructions from data* — argues that trust rules embedded in weights (constitutional SFT) are harder to override than rules in an easily-overridden system-prompt string.

## Method

- **Threat model:** black-box, input-only; attacker submits crafted text into the user-input slot.
- **Attacks:** goal hijacking (emit a rogue string) and prompt leaking (reveal the base prompt).
- **Framework:** base prompt (35 official OpenAI examples), attack prompt, rogue string, delimiters, escape characters, private value, model settings — assembled combinatorially, scored by exact/substring match, 4 repeats each.

## Key results

- **Goal hijacking 58.6% ± 1.6; prompt leaking 23.6% ± 2.7** (text-davinci-002) — leaking is harder than hijacking.
- **Inverse scaling:** text-ada-001 only ~13.8% — bigger/more instruction-tuned ⇒ more injectable.
- **Formatting matters:** delimiters raised hijacking ~43.6% → 58.6%.
- **Defences help but don't close it:** a stop sequence cut success ~60.0% → 47.5%; appending text after user input → ~51.8%.

## Critical appraisal

The foundational prompt-injection paper; its lasting value is conceptual (naming goal hijacking vs prompt leaking, the confused-deputy framing, inverse scaling) rather than its 2022 numbers. The sample is small (35 prompts × 4 repeats) and exact/substring matching under-counts *semantic* successes (a paraphrased leak still succeeds), so real ASR is likely higher. The model set is dated, but the architectural vulnerability (no instruction/data boundary) persists in every modern chat model — and grows with tool-using agents.

> ⚠ Inverse-scaling caveat for personalisation: the very instruction-following that enables personalisation *widens* the injection surface, so the constitution must counter-balance it. A sub-1B prompt-wrapper deployment is exactly PromptInject's target — testing goal-hijack and constitution-leak ASR on the 0.6B, with and without the harness, is a clean on-device experiment.

## Related

- [[sources/papers/adversarial-attacks-zou]] — the optimised (gradient) injection counterpart
- [[sources/papers/trustllm]] — maps to its safety (jailbreak) + privacy (leakage) dimensions
- [[entities/constitution]] — constitution-leak as a confidentiality + attack-surface risk
- [[topics/security-and-privacy]] — prompt injection, Log-To-Leak, threat taxonomy
- [[sources/dissertation/security-privacy-social-ethics]] — the project's security analysis
- [[sources/code/training-and-benchmark]] — the adversarial suite

## Sources

- Perez, Ribeiro (2022) — arXiv:2211.09527 (NeurIPS 2022 ML Safety Workshop) — [arxiv.org/abs/2211.09527](https://arxiv.org/abs/2211.09527)
- Code — [github.com/agencyenterprise/PromptInject](https://github.com/agencyenterprise/PromptInject)
