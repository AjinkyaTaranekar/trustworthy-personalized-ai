---
title: Asset Acquisition TODO — Overpersonalisation + Security Papers (2026-04-30)
type: question
tags: [todos, personalisation, security, privacy, scrutability, sycophancy]
sources:
  - docs/overpersonalisation/references.bib
  - docs/security-analysis/references.bib
updated: 2026-04-30
status: current
---

# Asset Acquisition TODO

**Status as of 2026-05-01 (third pass):** 7 more PDFs added manually by user. All previously [ ] papers except Budzyń 2025 (Lancet, paywalled) and Gemma 4 TR (no PDF) are now acquired with full Literature Notes. AppraisePLM identified as CoNLL 2025 paper co-authored by supervisor Conlan — Experiment 2 unblocked. Wiki source page created for AppraisePLM.

---

## From: Overpersonalisation Paper (`docs/overpersonalisation/paper.tex`)

- [x] **OP-Bench** — Hu et al. 2026 — arXiv:[2601.13722](https://arxiv.org/abs/2601.13722) — first benchmark targeting over-personalisation directly; central empirical anchor of the paper
- [x] **RPEval** — Feng et al. 2026 — arXiv:[2601.16621](https://arxiv.org/abs/2601.16621) — grounds irrational personalisation in Rational Speech Acts theory; evidence for preference-dominating mode
- [x] **Sycophancy in Language Models** — Sharma et al. 2025 — arXiv:[2310.13548](https://arxiv.org/abs/2310.13548) — RLHF structurally trains models to agree; foundational sycophancy claim
- [x] **SycEval** — Fanous et al. 2025 — arXiv:[2502.08177](https://arxiv.org/abs/2502.08177) — 58.19% sycophancy rate measured on ChatGPT-4o, Claude Sonnet, Gemini 1.5 Pro
- [x] **Interaction Context Often Increases Sycophancy** — Jain et al. CHI 2026 — DOI:[10.1145/3772318.3791915](https://doi.org/10.1145/3772318.3791915) — user memory profiles +33–45% agreement sycophancy; real 38-user × 2-week study
- [x] **Context Length Alone Hurts LLM Performance** — Du et al. 2025 — arXiv:[2510.05381](https://arxiv.org/abs/2510.05381) — 13.9–85% reasoning degradation from input length even with perfect retrieval; token-cost argument
- [x] **PrefEval** — Zhao et al. 2025 — arXiv:[2502.09597](https://arxiv.org/abs/2502.09597) — preference-following accuracy drops below 10% by ~10 turns; stored preferences become wasteful
- [x] **Personas in System Prompts** — Zheng et al. EMNLP 2024 — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.888/) — persona injection useless or actively harmful for objective tasks; 162 roles × 2,410 questions
- [x] **Scrutable User Models (framework)** — Kay & Kummerfeld 2013 — DOI:[10.1145/2395123.2395129](https://doi.org/10.1145/2395123.2395129) — 5 personalisation problems; 4 principles; Personis framework; foundational 20-year scrutability work
- [x] **Scrutable UMs for Time Management** — Jeromela & Conlan UMAP 2024 — DOI:[10.1145/3631700.3665182](https://doi.org/10.1145/3631700.3665182) — 6 IPA scrutability challenges; Stages concept for data stewardship; Conlan co-author
- [x] **HITL Control in PIPAs** — Akbar & Conlan UMAP 2024 — DOI:[10.1145/3631700.3664903](https://doi.org/10.1145/3631700.3664903) — HITL activation by autonomy preferences, cost, uncertainty; travel assistance use case; Conlan co-author
- [x] **Transparent and Scrutable Recommendations** — Ramos et al. 2024 — arXiv:[2402.05810](https://arxiv.org/abs/2402.05810) — NL user-profile summaries replace latent embeddings; matching perf + scrutability
- [x] **TEARS** — Penaloza et al. 2025 — arXiv:[2410.19302](https://arxiv.org/abs/2410.19302) — textual representations for scrutable recommendations

## From: Security Analysis Paper (`docs/security-analysis/security-review.tex`)

- [x] **Qwen3 Technical Report** — 2025 — arXiv:[2505.09388](https://arxiv.org/abs/2505.09388) — base model technical report (entity page [[entities/qwen3-0.6b]] exists but no source/paper page for the TR itself)
- [x] **Phi-4 Technical Report** — Microsoft 2024 — arXiv:[2412.08905](https://arxiv.org/abs/2412.08905) — on-device small model; industry context for local-first argument
- [ ] **Gemma 4 Technical Report** — Google DeepMind 2026 — no arXiv yet; retrieve from [Google DeepMind blog](https://blog.google/technology/developers/gemma-4/) — on-device inference peer to Qwen3
- [x] **Membership Inference Attacks** — Shokri et al. 2017 — arXiv:[1610.05820](https://arxiv.org/abs/1610.05820) — re-identification risk from aggregated conversational data; underpins re-identification claim
- [x] **Ignore Previous Prompt** — Perez & Ribeiro 2022 — arXiv:[2211.09527](https://arxiv.org/abs/2211.09527) — prompt injection attack techniques; OWASP LLM01 foundational paper
- [x] **Universal and Transferable Adversarial Attacks** — Zou et al. 2023 — arXiv:[2307.15043](https://arxiv.org/abs/2307.15043) — transferable jailbreak attacks on aligned LLMs; cited in red-team benchmark requirement
- [x] **Log-To-Leak** — Hu et al. 2026 — [OpenReview:UVgbFuXPaO](https://openreview.net/forum?id=UVgbFuXPaO) — MCP prompt injection → silent user-query exfiltration via logging tool; highest-priority open gap
- [x] **Constitutional AI: Harmlessness from AI Feedback** — Bai et al. 2022 — arXiv:[2212.08073](https://arxiv.org/abs/2212.08073) — original CAI paper; source for generate–critique–revise loop and its single-point-of-failure risk
- [x] **Constitution or Collapse?** — Zhang 2025 — arXiv:[2504.04918](https://arxiv.org/abs/2504.04918) — CAI degeneration with Llama 3-8B; small-model critique loop failure mode
- [x] **Synthetic Attachment** — Lipin 2025 — DOI:[10.13140/RG.2.2.10944.03843](https://doi.org/10.13140/RG.2.2.10944.03843) — parasocial bonds, SARRS clinical tool; theoretical paper (un-peer-reviewed); emotional dependency claim
- [ ] **Endoscopist Deskilling** — Budzyń et al. 2025 — DOI:[10.1016/S2468-1253(25)00133-5](https://doi.org/10.1016/S2468-1253(25)00133-5) — documented AI-induced deskilling in clinical setting; empirical evidence for OWASP LLM09
- [x] **The Extended Hollowed Mind** — Klein & Klein 2025 — DOI:[10.3389/frai.2025.1719019](https://doi.org/10.3389/frai.2025.1719019) — frictionless AI access bypasses effortful processes that build durable understanding
- [x] **GPT-5 System Card** — OpenAI 2025 — arXiv:[2601.03267](https://arxiv.org/abs/2601.03267) — frontier model reference for privacy-crisis framing
- [x] **Claude Sonnet 4.6 System Card** — Anthropic 2026 — [CDN PDF](https://www-cdn.anthropic.com/bbd8ef16d70b7a1665f14f306ee88b53f686aa75.pdf) — frontier model reference

---

## Still outstanding from previous lint (2026-04-19)

These five were flagged before and remain unacquired.

- [x] **AppraisePLM** — Debnath, Graham, Conlan 2025 — CoNLL 2025 (2025.conll-1.16); supervisor Conlan co-author; **Experiment 2 now unblocked**
- [x] **Think-on-Graph** — Sun et al. 2024 — informs [[entities/graph-rag]]
- [x] **LLM-guided Tree-of-Thought** — Long 2023
- [x] **SEAL** — Zweiger 2025 — Self-Adapting Language Models
- [x] **Structured Solution Templates** — Yang 2025

---

## Web / Blog / Regulation refs — no PDF asset needed

These are cited but are not academic papers. No Obsidian PDF required. Listed for completeness.

- OWASP LLM Top 10 2025 — https://genai.owasp.org/llm-top-10/
- GDPR Regulation 2016 — https://eur-lex.europa.eu/eli/reg/2016/679/oj
- OpenAI Memory blog 2025 — https://openai.com/index/memory-and-new-controls-for-chatgpt/
- OpenAI Custom Instructions blog 2023 — https://openai.com/index/custom-instructions-for-chatgpt/
- Gemini Personalization blog 2025 — https://blog.google/products/gemini/gemini-personalization/
- Gemini Temporary Chats blog 2025 — https://blog.google/products/gemini/temporary-chats-privacy-controls/
- Anthropic Memory blog 2025 — https://www.anthropic.com/news/memory
- Anthropic Context Management blog 2025 — https://www.anthropic.com/news/context-management
- Gemini 3 blog 2026 — https://blog.google/products-and-platforms/products/gemini/gemini-3/
