---
title: Wiki Index
type: meta
updated: 2026-04-30
---

# Index

Catalog of everything in the wiki. Regenerated on every ingest. One line per entry. For chronological activity, see [[log]].

## Meta

- [[overview]] — thesis synthesis and research-question anchor
- [[log]] — append-only chronological journal
- [[tags]] — canonical tag vocabulary (check before adding new tags)
- `../CLAUDE.md` — schema and rules governing this wiki

## Topics

- [[topics/llm-foundations]] — tokenisation, attention, embeddings; why monolithic LLMs fail
- [[topics/reasoning]] — trustworthy reasoning across SFT, RL, architecture, evaluation
- [[topics/personalisation]] — 5W+H, GraphRAG, cold start; over-personalisation failure modes; scrutability
- [[topics/empathy]] — appraisal theory, Gricean grounding, conversation design; dependency/deskilling risks
- [[topics/tool-use-and-verification]] — PAL/ReAct delegation, MCP, ontology verification; prompt injection risk
- [[topics/explainability]] — citations, honest tool reports, translated latent state
- [[topics/ontology-integration]] — flagship: ontology as KB (A) or post-hoc verifier (B)
- [[topics/security-and-privacy]] — local-first privacy argument, OWASP threat taxonomy, Log-To-Leak, alignment regression

## Entities

- [[entities/constitution]] — 19-principle SFT v2 constitution
- [[entities/grpo]] — group relative policy optimisation (the repo's RL algorithm)
- [[entities/mcp]] — Model Context Protocol — "USB for AI"
- [[entities/rag]] — retrieval-augmented generation pattern
- [[entities/qwen3-0.6b]] — the pipeline's base model
- [[entities/graph-rag]] — KG-backed RAG for user-state memory
- [[entities/5w-h]] — who/what/when/where/why/how user-modelling schema
- [[entities/appraisal-theory]] — structured empathy substrate

## Sources

### Papers — Foundations
- [[sources/papers/attention-is-all-you-need]] — the Transformer
- [[sources/papers/bert]] — bidirectional pre-training
- [[sources/papers/word2vec]] — static word embeddings
- [[sources/papers/bpe-subword-units]] — subword tokenisation (root of arithmetic failure)
- [[sources/papers/measuring-word-significance]] — vector length as importance

### Papers — Prompted reasoning
- [[sources/papers/chain-of-thought-prompting]] — CoT as prompted reasoning
- [[sources/papers/auto-cot]] — automates CoT exemplar creation
- [[sources/papers/prompting-science-report-2]] — diminishing returns on modern models
- [[sources/papers/tree-of-thoughts]] — search-based deliberation

### Papers — RL for reasoning
- [[sources/papers/deepseek-r1]] — R1-Zero + multi-stage R1
- [[sources/papers/seed15-thinking]] — process-reward RL exemplar
- [[sources/papers/vapo]] — value-based PPO for long-CoT
- [[sources/papers/understanding-r1-zero]] — GRPO length-bias critique, Dr. GRPO
- [[sources/papers/interleaved-reasoning]] — RL-trained interleaved thinking
- [[sources/papers/hidden-reasoners]] — LaTRO self-rewarding latent reasoning

### Papers — Architectural / latent reasoning
- [[sources/papers/hierarchical-reasoning-model]] — HRM: slow + fast modules
- [[sources/papers/looped-transformers-reasoning]] — depth > parameters
- [[sources/papers/coconut-continuous-latent]] — reason in vector space
- [[sources/papers/ladir]] — latent diffusion reasoning
- [[sources/papers/state-stream-transformer]] — persistent latent state
- [[sources/papers/diffusion-of-thoughts]] — diffusion-LM CoT

### Papers — Small-model / distillation
- [[sources/papers/self-enhanced-reasoning]] — SERT small-model self-training
- [[sources/papers/dual-head-reasoning-distillation]] — train-time-only reasoning
- [[sources/papers/token-hungry-deepseek-r1]] — accuracy-vs-efficiency trade-off

### Papers — Tool use
- [[sources/papers/pal]] — code-as-reasoning delegation
- [[sources/papers/react]] — interleaved reason-act loop
- [[sources/papers/mcp-multi-agent]] — MCP coordination protocol
- [[sources/papers/search-r1]] — RL-trained search-tool use
- [[sources/papers/rag-original]] — retrieval-augmented generation

### Papers — Multimodal / agent RL
- [[sources/papers/ui-r1]] — RL for GUI-agent action prediction
- [[sources/papers/vlm-r1]] — R1-style RL for vision-language

### Papers — Evaluation
- [[sources/papers/none-of-the-others]] — reasoning-vs-memorisation variation

### Papers — Empathy / affect
- [[sources/papers/xai-sentiment-deepseek-r1]] — transparent sentiment classification
- [[sources/papers/appraise-plm]] — AppraisePLM: 21-dim appraisal regression + emotion classification; CoNLL 2025; Debnath, Graham, Conlan (TCD); unblocks Experiment 2

### Papers — Personalisation / over-personalisation
- [[sources/papers/op-bench]] — first benchmark for over-personalisation; 26–61% degradation from memory augmentation
- [[sources/papers/rpeval]] — Rational Personalisation framework; 40–90% human-LLM accuracy gap; inverse scaling
- [[sources/papers/sycophancy-sharma]] — RLHF structurally incentivises sycophancy; foundational mechanism paper (ICLR 2024)
- [[sources/papers/syc-eval]] — 58.19% sycophancy rate; 78.5% persistence; cross-model measurement (AIES 2025)

### Papers — Security / alignment
- [[sources/papers/constitutional-ai-bai]] — original CAI paper; generate–critique–revise loop; RLAIF (Anthropic 2022)
- [[sources/papers/constitution-or-collapse]] — CAI at 8B: 40.8% ASR reduction, 9.8% helpfulness cost, model collapse

### Papers — Context / preference tracking
- [[sources/papers/context-length-hurts]] (Literature Note only) — 13.9–85% degradation from context length alone despite perfect retrieval
- [[sources/papers/prefeval]] (Literature Note only) — preference following <10% at 10 turns; ICLR 2025 oral
- [[sources/papers/transparent-scrutable-recs]] (Literature Note only) — NL user profiles: scrutable + competitive accuracy (UCL/Sheffield, ACL 2024)
- [[sources/papers/tears]] (Literature Note only) — TEARS: textual representations + optimal transport for scrutable recommenders

### Papers — Knowledge graphs / tool use
- [[sources/papers/think-on-graph]] — LLM ⊗ KG tight coupling; beam search on KG; SOTA on 6/9 datasets (ICLR 2024)
- [[sources/papers/llm-guided-tot]] (Literature Note only) — ToT software system with checker module and backtracking controller

### Papers — Adaptation / training
- [[sources/papers/seal]] (Literature Note only) — SEAL: LLMs generate self-edits for weight updates via RL outer loop (NeurIPS 2025)
- [[sources/papers/structured-templates]] (Literature Note only) — Scaling Law by Difficulty; SST framework; +6.2 GSM8K pts

### Papers — Security threats
- [[sources/papers/membership-inference]] (Literature Note only) — shadow-model attack; 94% accuracy on Google ML; re-identification risk
- [[sources/papers/ignore-previous-prompt]] (Literature Note only) — PromptInject: goal hijacking 58.6%, prompt leaking 23.6%
- [[sources/papers/adversarial-attacks-zou]] (Literature Note only) — GCG suffix; transfers to ChatGPT/Claude/Bard; black-box jailbreak

### Papers — Frontier model references
- [[sources/papers/qwen3-tr]] (Literature Note only) — Qwen3 family 0.6B–235B; unified thinking/non-thinking; base model technical report
- [[sources/papers/phi4-tr]] (Literature Note only) — Phi-4 14B; data quality > scale; surpasses GPT-4o on STEM-QA
- [[sources/papers/gpt5-system-card]] (Literature Note only) — GPT-5 system card; sycophancy + prompt injection as safety challenges

### Dissertation drafts (user-authored raw)
- [[sources/dissertation/research-plan]] — formal CS7CS6 plan: title, 5 objectives, 7 phases, 2 pivots
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — main thesis argument + literature review
- [[sources/dissertation/experimental-planning-document]] — 6 experiments, 2025-11-10 meeting, timeline
- [[sources/dissertation/personal-notes]] — Experiment.md + Rough Notes.md + Research Plan Edits.md
- [[sources/dissertation/overpersonalisation-paper]] — LLNCS paper: three failure modes, sycophancy mechanism, UMAP scrutability tradition, commercial memory architecture comparison
- [[sources/dissertation/security-privacy-social-ethics]] — security analysis: local-first privacy argument, Log-To-Leak, alignment regression, critique-loop SPOF, dependency/deskilling ethics

### Code (pipeline summaries)
- [[sources/code/sft-v2-pipeline]] — constitution-driven data generation
- [[sources/code/constitution-document]] — full 19-principle source
- [[sources/code/training-and-benchmark]] — v1 scripts + LoRA + GRPO + benchmark + context degradation

## Experiments

- [[experiments/experiment-catalog]] — all six experiments + ablation A/B/C/D

## Decisions

- [[decisions/2025-11-10-ontology-focus-shift]] — primary focus moves to ontology-LLM integration

## Questions

- [[questions/2026-04-19-initial-questions]] — consolidated TODOs, advisor-prep questions, literature tensions
- [[questions/2026-04-30-asset-acquisition-todo]] — 27-paper acquisition checklist from overpersonalisation + security papers (+ 5 carry-over from prior lint)

## Queries

_None yet. Ask me a durable question and I will offer to file the answer here._

---

## Not yet ingested

- `IMPROVEMENT_ROADMAP.md` — 54KB roadmap at repo root. Ingest if still authoritative.
- Per-file deep-dives of individual `pipeline/sft_*.py` scripts — summarised together in [[sources/code/sft-v2-pipeline]] for now.

## Cited but not in `docs/Assets/`

Papers with stub Literature Notes but no PDF. Full checklist: [[questions/2026-04-30-asset-acquisition-todo]].

- Budzyń et al. 2025 — endoscopist deskilling (Lancet, paywalled; institution access needed)
- Google DeepMind 2026 — Gemma 4 Technical Report (no arXiv; blog only)
