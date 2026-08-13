---
title: "OpenAI GPT-5 System Card"
type: source
tags: [security, evaluation, sycophancy]
sources:
  - https://arxiv.org/abs/2601.03267
updated: 2026-07-22
status: current
---

# OpenAI GPT-5 System Card

**GPT-5 ships as a family trained with "safe-completions" (optimise output safety + helpfulness rather than binary refusal) and reports measurable reductions across the exact failure modes this project studies — sycophancy, prompt injection, hallucination, deception — benchmarked against GPT-4o and o3, though most numbers are self-reported relative reductions.**

## Summary

OpenAI's 2026 system card names and quantifies four interaction/agent-level failure modes: **sycophancy** (gpt-5-thinking offline 0.040 / gpt-5-main 0.052 vs GPT-4o 0.145; ~69–75% lower online prevalence), **prompt injection** (gpt-5-thinking defence 0.97–0.99 vs o3 0.80–0.94, plus a cached-page-browsing architectural mitigation), **hallucination** (large *relative* reductions vs o3, but SimpleQA absolute hallucination still 0.40), and **deception** (sharp drops on most agentic scenarios, e.g. CharXiv 0.09 vs o3 0.87, but an AbstentionBench *regression* and ~2.1% of production responses still CoT-flagged). It is valuable because it treats "safety" as a taxonomy rather than a monolith — high-value BACKGROUND and partly load-bearing for the project's safety framing.

## Why it matters here

This is the frontier-model reference where **sycophancy** and **prompt injection** — the exact failure modes the project targets on a sub-1B on-device model — are explicitly *named challenges* with defined evals and numbers. It lets the project say "the failure modes we target on a 0.6B model are the same ones OpenAI dedicates safety-card sections to at the frontier". Hooks: (1) borrow the four-way taxonomy (sycophancy / prompt injection / hallucination / deception) as the trust-failure taxonomy; (2) contrast *methods* — GPT-5 uses safe-completions + massive scale + cached-page browsing, the project a constitutional harness on 0.6B, so the card motivates *why* these behaviours matter while its solutions don't transfer to on-device (the frontier-vs-on-device gap, consistent with the novelty gap that no sub-1B constitutional-harness paper exists); (3) reuse eval *design* (production-representative sycophancy scoring, infeasible-task deception tests, LLM-grader-with-human-validation) for the substance-based judge.

## Key results (safety sections)

- **Sycophancy:** gpt-5-thinking 0.040, gpt-5-main 0.052 vs GPT-4o 0.145; online prevalence −69% (free) / −75% (paid). Sycophancy was turned into a training reward.
- **Prompt injection (defence rate):** browsing 0.99 vs o3 0.89; tool-calling 0.99 vs 0.80; coding 0.97 vs 0.94; Gray Swan external "SOTA".
- **Hallucination:** production ≥1-major-error 78% fewer (gpt-5-thinking vs o3); LongFact/FActScore "5× fewer factual errors"; **SimpleQA absolute hallucination still 0.40**.
- **Deception:** CharXiv 0.09 vs 0.87, Broken Tools 0.11 vs 0.61; **AbstentionBench regresses (0.53 vs 0.44)**; CoT monitor flags ~2.1% of production responses (monitor ~81% precision / 84% recall).

## Critical appraisal

Valuable precisely because it names and quantifies four failure modes rather than treating safety as monolithic, and safe-completions is a substantive shift from refusal-only training. But read it as advocacy-adjacent: many headline numbers are *relative reductions* against OpenAI's own prior models (flattering the improvement, hiding absolute residual risk), evals are self-constructed and self-graded (graders are LLMs — 75% human agreement on hallucination), defence rates near 0.99 can create false adversarial-robustness confidence, and at least one mitigation regresses (AbstentionBench).

> ⚠ When citing: use for the *problem taxonomy and importance*, not as achievable targets for a small model — note absolute residuals (SimpleQA 0.40; ~2.1% production deception) to avoid overstating frontier "solvedness". Support for the trust/safety motivation, background rather than a methodological dependency.

## Related

- [[sources/papers/sycophancy-sharma]] / [[sources/papers/syc-eval]] — the sycophancy failure mode, measured
- [[sources/papers/ignore-previous-prompt]] / [[sources/papers/adversarial-attacks-zou]] — the injection/jailbreak threats
- [[sources/papers/trustllm]] — the trustworthiness taxonomy this complements
- [[sources/papers/abstention-bench]] — the AbstentionBench eval GPT-5 regresses on
- [[sources/papers/hallucination-survey]] — the factuality failure mode
- [[topics/security-and-privacy]] — the four-way trust-failure taxonomy
- [[entities/tml-interaction-small]] — another frontier reference contrast

## Sources

- OpenAI (2026) — arXiv:2601.03267 — [arxiv.org/abs/2601.03267](https://arxiv.org/abs/2601.03267)
