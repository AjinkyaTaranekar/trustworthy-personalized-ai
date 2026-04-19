---
title: Explainability (XAI)
type: topic
tags: [xai, scrutability, interpretability]
sources:
  - docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md
updated: 2026-04-19
status: current
---

# Explainability

**Treating "explain yourself" as an architectural property — citations,
tool-call transparency, and translated internal state — rather than as
prose self-rationalisation, which is fabricated by construction.**

## Why this is a separate topic from [[topics/reasoning]]
Reasoning is about *arriving at a correct answer*; explainability is
about *being auditable*. An answer can be correct and unexplainable
(Coconut) or plausible-sounding and unauditable (post-hoc CoT). The
thesis treats these as orthogonal and designs for both.

## Four scrutability routes (from the draft)

1. **Source attribution** — RAG-style citations back to raw documents.
2. **Honest tool reporting** — PAL-style "I used a calculator" traces.
3. **Translating internal reasoning state** — HRM-style latent reasoner
   exposed via a translator module.
4. **Ontology-based post-hoc verification** — the 2025-11-10 direction
   shift. Claims extracted, checked, flagged, or corrected.

## Why attention visualisations don't count

Attention weights are not causally faithful — high attention can coexist
with zero influence on the output (residual + MLP paths dominate).
Adversarial attacks can manipulate attention without changing outputs.
Intuition tool only.

## Calibration as an explainability property

A well-calibrated model's confidence matches its accuracy. LLMs
routinely overconfidently assert wrong answers, which is an
explainability failure even when the answer is correct — the user has
no honest signal of how much to trust a specific response.

## Related

- [[topics/reasoning]] — sibling topic; they share the sociopath-yapper root cause
- [[topics/tool-use-and-verification]] — tool reporting is one scrutability route
- [[decisions/2025-11-10-ontology-focus-shift]] — ontology verification is the chosen route
- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent reasoning traces for classification
- [[sources/papers/coconut-continuous-latent]] · [[sources/papers/hierarchical-reasoning-model]] — architectures that sacrifice scrutability for reasoning quality
- [[entities/rag]] · [[entities/mcp]]

## Raw

- `docs/Dissertation/Road Towards Trustworthy and Empathetic AI.md` §5.1 "Scrutability & Explainability"
