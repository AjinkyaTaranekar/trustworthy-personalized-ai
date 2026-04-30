---
paper id: 2510.05381v1
title: Context Length Alone Hurts LLM Performance Despite Perfect Retrieval
authors: [Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, Hao Peng]
publication date: 2025-10-06T21:17
abstract: "Large language models (LLMs) often fail to scale their performance on long-context tasks performance in line with the context lengths they support. This gap is commonly attributed to retrieval failures -- the models' inability to identify relevant information in the long inputs. Accordingly, recent efforts often focus on evaluating and improving LLMs' retrieval performance: if retrieval is perfect, a model should, in principle, perform just as well on a long input as it does on a short one -- or should it? This paper presents findings that the answer to this question may be negative. Our systematic experiments across 5 open- and closed-source LLMs on math, question answering, and coding tasks reveal that, even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases but remains well within the models' claimed lengths. This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked and the models are forced to attend only to the relevant tokens. A similar performance drop is observed when all relevant evidence is placed immediately before the question. Our findings reveal a previously-unrealized limitation: the sheer length of the input alone can hurt LLM performance, independent of retrieval quality and without any distraction. They motivate our simple, model-agnostic mitigation strategy that transforms a long-context task into a short-context one by prompting the model to recite the retrieved evidence before attempting to solve the problem. On RULER, we observe a consistent improvement of GPT-4o up to 4% on an already strong baseline."
comments: "18 pages (9 pages of main content), 5 figures, accepted at the Findings of EMNLP 2025"
pdf: "[[Assets/Context Length Alone Hurts LLM Performance Despite Perfect Retrieval (2510.05381v1).pdf]]"
url: https://arxiv.org/abs/2510.05381v1
tags: [personalisation, context-degradation, evaluation, rag]
---

## Key Claims

- Performance degrades 13.9–85% as context length grows even when the model can perfectly retrieve all evidence (100% exact match) — falsifying the retrieval-centric view of long-context failure.
- The degradation occurs even when irrelevant tokens are replaced with whitespace, and even when all distractor tokens are masked (model attends *only* to evidence + question) — sheer length is the cause.
- Tested on GSM8K (maths), MMLU (QA), HumanEval (code) across Llama-3.1-8B and Mistral-0.3-7B; consistent pattern across tasks and models.
- **Mitigation**: "recite-then-reason" — prompt model to recite retrieved evidence before answering; consistent +4% on RULER for GPT-4o.
- Calls into question benchmarks that evaluate retrieval separately from reasoning — improvements in retrieval do not translate to improvements in task performance.

## Thesis Relevance

Directly cited in the LLNCS paper as empirical evidence for the context-inflation failure mode (injecting large user-memory context degrades response quality independent of relevance). Motivates the thesis's Self-ReCheck and selective memory injection design: inject only what is needed, not the full memory store.

## Questions / Open Issues

- The recite-then-reason mitigation adds tokens to the context itself — does it compound for very long inputs?
- The effect is measured on structured benchmarks; how does it apply to open-ended conversational AI where quality is harder to measure?
- The 5B–128B model range was not tested; unclear if the degradation scales differently for the thesis's 0.6B model.
