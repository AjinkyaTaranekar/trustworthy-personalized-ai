---
title: "Talking About Large Language Models"
type: source
tags: [foundations, psychology]
sources:
  - https://arxiv.org/abs/2212.03551
updated: 2026-07-22
status: current
---

# Talking About Large Language Models

**As LLMs get better at mimicking human language we are tempted to describe them with mentalistic words — "knows", "believes", "thinks", "reasons" — but doing so is a category mistake: a bare LLM only performs next-token prediction over the statistical distribution of human text, so we must repeatedly step back and re-describe how they actually work to keep our claims honest.**

## Summary

Shanahan (Imperial / DeepMind, 2022; CACM 2024) argues that mentalistic vocabulary carries entailments — a knower has truth-access, a believer can be right or wrong against the world — that human language-users earn by being embedded in a shared world, and that a sequence predictor has not earned. His central distinctions became standard reference points: the *bare model* (a deterministic sequence-prediction function) vs the *dialogue system* wrapped around it; "knowing that word X follows word Y" ≠ "knowing that fact X is true"; and reasoning-as-pattern-completion, where chain-of-thought *resembles* valid argument without any guarantee its causal structure mirrors truth-preserving inference. Grounding (retrieval, vision, embodiment) narrows but does not close the gap, because correlation with the world is not causal, truth-tracking reference. A load-bearing *conceptual* citation for the background chapter, though it contributes no method.

## Why it matters here

For a thesis on architecting trust and empathy, Shanahan supplies the crucial guardrail: an empathetic-*sounding* small model is not an empathetic agent, and a model that answers warmly is not a person who cares — felt trust must be engineered and evaluated, never assumed from fluent human-like output. This underwrites the project's insistence on **substance-based evaluation over surface mimicry** (the "a model that answers is not a person who answers" framing that opens the background), and the design stance that trust/empathy are constructed properties of the *system* (constitution, retrieval, guardrails), not emergent mental states of the bare model. Pairs with [[sources/papers/theoretical-impediments-ml|Pearl]] as the two philosophical pillars: Shanahan for "language ≠ mind", Pearl for "seeing ≠ doing".

## Argument structure

- **Next-token framing:** an LLM prompted with "The first person to walk on the Moon was" is answering "given the statistical distribution of human text, what likely follows?" — treating factual, fictional, and nursery-rhyme continuations indifferently.
- **Bare model vs embedded system:** keep the precise sequence-prediction function distinct from the prompt/guardrail/interface wrapper.
- **Humans vs LLMs on answering:** a human informant has communicative intent, a model of the interlocutor, and external truth-access (observation, consultation, argument); a bare LLM has none.
- **Reasoning:** CoT output resembles argument but is not guaranteed logically faithful; trust in a deductive system requires a causal structure mirroring valid inference.
- **RLHF** reshapes which sequences are likely (preference-weighted data) but does not change the fundamental operation.

## Critical appraisal

An influential, clearly written intervention whose distinctions have become standard in the "what do LLMs understand" debate; its strength is conceptual hygiene. Its limitation: it draws the human/machine line at a demanding embodiment-and-community standard and stays deliberately agnostic about hybrid tool-using/embodied systems — exactly where current engineering lives — and it predates the strongest agentic systems, so its concessions are the part most in tension with 2024–2026 practice. Best read as a disciplined caution, not a verdict; it says bare prediction does not suffice but cannot say how much grounding does.

> Note: BACKGROUND/motivation, not method. No on-device/small-model angle — its weight is philosophical (anti-anthropomorphism as a design principle; CoT-as-pattern-completion caveat for small models).

## Related

- [[sources/papers/theoretical-impediments-ml]] — Pearl; the second philosophical pillar (correlation vs causation)
- [[topics/llm-foundations]] — what LLMs are and are not
- [[topics/constitution-psychological-grounding]] — engineering trust behaviours vs assuming mental states
- [[sources/papers/hallucination-survey]] — the faithfulness gap this frames philosophically
- [[experiments/human-evaluation-rubric]] — substance-based evaluation over surface mimicry
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — the background argument

## Sources

- Shanahan (2022) — arXiv:2212.03551 (Communications of the ACM 67(2), 2024) — [arxiv.org/abs/2212.03551](https://arxiv.org/abs/2212.03551)
