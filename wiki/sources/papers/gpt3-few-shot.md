---
title: "Language Models are Few-Shot Learners (GPT-3)"
type: source
tags: [foundations, prompting]
sources:
  - https://arxiv.org/abs/2005.14165
updated: 2026-07-18
status: current
---

# Language Models are Few-Shot Learners (GPT-3)

**Scaling an autoregressive language model to 175B parameters makes it a competent in-context learner — performing many tasks from a natural-language description plus a few in-prompt examples, with no gradient updates — often approaching fine-tuned SOTA, while several abilities emerge only at scale.**

## Summary

Brown et al. (OpenAI, NeurIPS 2020, Best Paper) establish task-agnostic in-context few-shot learning. Eight models (125M→175B) are trained on ~300B tokens and evaluated zero/one/few-shot with weights frozen. GPT-3 approaches or matches fine-tuned systems on many tasks (LAMBADA 86.4%, TriviaQA 71.2% closed-book), generates news near-indistinguishable from human writing (~52% human detection), but is near-chance on careful entailment (ANLI, WiC ~49.4%). Crucially, the few-shot advantage grows with scale and some skills (arithmetic, unscrambling) emerge sharply — the empirical basis for scaling laws, and, read alongside FLAN and InstructGPT, the "before" picture the on-device thesis reacts to.

## Why it matters here

Foundational context and mostly a *contrast* case. The dissertation's arc is essentially: GPT-3 showed capability needs scale → [[sources/papers/flan|FLAN]] showed instructions can be *taught* → [[sources/papers/instructgpt|InstructGPT]] showed *alignment beats scale* → therefore a small, constitution-tuned 0.6B model can be trustworthy and useful. GPT-3 also motivates baking behaviour in via SFT rather than runtime few-shot: a 0.6B on-device model has little scale *and* little context budget, and GPT-3-style in-context learning is weakest exactly at small scale.

## Method

- **Architecture:** GPT-2-style decoder-only with alternating dense/locally-banded sparse attention; 8 sizes to 175B (96 layers, d_model 12288), all on ~300B tokens.
- **Data (quality-reweighted, not raw size):** filtered Common Crawl (410B tokens, 60% weight, 0.44 epochs), WebText2 (22%), Books1/Books2 (8%/8%), Wikipedia (3%, 3.4 epochs); higher-quality sets sampled more than once.
- **In-context settings:** zero-shot (description only), one-shot, few-shot (K examples within the 2,048-token window) — no weight updates in any setting.

## Key results

- **LAMBADA** few-shot 86.4% (prior SOTA ~68%); **TriviaQA** 71.2% closed-book (matches retrieval-augmented fine-tuned systems); **SAT analogies** 65.2% (above average human applicant); 3-digit addition ~80–94%.
- **News generation:** humans detect 175B-generated articles at ~52% (chance).
- **Scaling:** few-shot improves dramatically with size and the few-shot-over-zero-shot gap widens — larger models are better meta-learners; loss follows a power law in compute.
- **Weak spots:** NLI/ANLI near chance, WiC ~49.4%, some reading-comprehension lags.

## Critical appraisal

Paradigm-defining scope and rigour, an 8-model scaling sweep, and an unusually candid limitations/impacts section (misuse, bias, energy). Cautions: a filtering **bug left some train/test contamination** (post-hoc analysis judged impact mostly negligible, PIQA/Winograd asterisked); headline numbers use favourable prompt formats and best K; the artifact is proprietary. It is a capability paper — **no claim to being aligned, honest, or safe** (the direct motivation for InstructGPT).

> ⚠ Scale-dependence caution: GPT-3 shows abilities *emerge only at scale* and few-shot benefit grows with size; with FLAN's ≤8B degradation, this is the central risk the dissertation must empirically refute at 0.6B — that a sub-1B model may lack capacity for robust instruction/tool-use behaviour. Its documented bias/misuse is the "before" the constitution and substance-based evaluation aim to reduce.

## Related

- [[sources/papers/flan]] — teaches the zero-shot instruction following GPT-3 lacks
- [[sources/papers/instructgpt]] — aligns GPT-3 to intent; "alignment beats scale"
- [[sources/papers/attention-is-all-you-need]] — the Transformer GPT-3 scales
- [[entities/qwen3-0.6b]] — the small on-device model this contrasts with
- [[topics/llm-foundations]] — scaling, in-context learning, and their limits
- [[topics/reasoning]] — emergence and the small-scale capacity question

## Sources

- Brown et al. (2020) — arXiv:2005.14165 (NeurIPS 2020) — [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
