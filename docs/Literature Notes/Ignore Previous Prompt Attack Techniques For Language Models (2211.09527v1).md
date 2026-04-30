---
paper id: 2211.09527v1
title: "Ignore Previous Prompt: Attack Techniques For Language Models"
authors: [Fábio Perez, Ian Ribeiro]
publication date: 2022-11-17T13:43
abstract: "Transformer-based large language models (LLMs) provide a powerful foundation for natural language tasks in large-scale customer-facing applications. However, studies that explore their vulnerabilities emerging from malicious user interaction are scarce. By proposing PromptInject, a prosaic alignment framework for mask-based iterative adversarial prompt composition, we examine how GPT-3, the most widely deployed language model in production, can be easily misaligned by simple handcrafted inputs. In particular, we investigate two types of attacks -- goal hijacking and prompt leaking -- and demonstrate that even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject is available at https://github.com/agencyenterprise/PromptInject."
comments: ML Safety Workshop NeurIPS 2022
pdf: "[[Assets/Ignore Previous Prompt Attack Techniques For Language Models (2211.09527v1).pdf]]"
url: https://arxiv.org/abs/2211.09527v1
tags: [security, tool-use, evaluation]
---

## Key Claims

- **PromptInject framework**: quantitative analysis of two prompt injection attacks against GPT-3: (1) **goal hijacking** — redirect model to print a target adversarial string; (2) **prompt leaking** — extract the original system prompt.
- Goal hijacking: **58.6% ± 1.6 mean success rate** across 35 base prompts; prompt leaking: **23.6% ± 2.7**.
- More capable models (text-davinci-002) are **more vulnerable** than weaker ones — higher instruction-following ability enables adversaries to exploit the same capability.
- Using `print` rather than `say`, and adding `instead`, significantly boosts attack success — small wording changes have outsized effects.
- Stop sequences, post-processing constraints, and limiting output tokens are the most effective defences; temperature and frequency penalties have minimal effect.

## Thesis Relevance

Foundational OWASP LLM01 (Prompt Injection) paper cited in the security analysis. In the thesis's MCP architecture, `read_url` and `web_search` tools return live web content that could contain adversarial instructions. Constitution Principle 10 (follow tool-returned content) creates the amplified attack surface described here — the thesis must implement an extraction layer converting raw tool output to structured data before it enters the model's reasoning.

## Questions / Open Issues

- Tested on GPT-3 (text-davinci-002, 2022) — how do modern aligned models (Claude, GPT-4) respond? Suggests stronger alignment helps but does not eliminate the risk.
- The Log-To-Leak attack (Hu et al. 2026) extends this beyond goal hijacking to data exfiltration — the more dangerous threat for the thesis's use case.
- Structural defence (output extraction layer) is the right response; this needs to be specified as a required pre-deployment task in the thesis.
