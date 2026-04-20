---
title: Wiki Index
type: meta
updated: 2026-04-19
---

# Index

Catalog of everything in the wiki. Regenerated on every ingest. One line per
entry. For chronological activity, see [[log]].

## Meta

- [[overview]] — thesis synthesis and research-question anchor
- [[log]] — append-only chronological journal
- [[tags]] — canonical tag vocabulary (check before adding new tags)
- `../CLAUDE.md` — schema and rules governing this wiki

## Topics

- [[topics/llm-foundations]] — tokenisation, attention, embeddings; why monolithic LLMs fail
- [[topics/reasoning]] — trustworthy reasoning across SFT, RL, architecture, evaluation
- [[topics/personalisation]] — 5W+H, GraphRAG, cold start, privacy paradox
- [[topics/empathy]] — appraisal theory, Gricean grounding, conversation design
- [[topics/tool-use-and-verification]] — PAL/ReAct delegation, MCP, ontology verification
- [[topics/explainability]] — citations, honest tool reports, translated latent state
- [[topics/ontology-integration]] — flagship: ontology as KB (A) or post-hoc verifier (B)

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

### Dissertation drafts (user-authored raw)
- [[sources/dissertation/research-plan]] — formal CS7CS6 plan: title, 5 objectives, 7 phases, 2 pivots
- [[sources/dissertation/road-towards-trustworthy-empathetic-ai]] — main thesis argument + literature review
- [[sources/dissertation/experimental-planning-document]] — 6 experiments, 2025-11-10 meeting, timeline
- [[sources/dissertation/personal-notes]] — Experiment.md + Rough Notes.md + Research Plan Edits.md

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

## Queries

_None yet. Ask me a durable question and I will offer to file the answer here._

---

## Not yet ingested

- `IMPROVEMENT_ROADMAP.md` — 54KB roadmap at repo root. Ingest if still authoritative.
- Per-file deep-dives of individual `pipeline/sft_*.py` scripts — summarised together in
  [[sources/code/sft-v2-pipeline]] for now.

## Cited but not in `docs/Assets/`

Papers referenced in the research plan that we don't hold PDFs for
(flagged in [[questions/2026-04-19-initial-questions]]):

- Debnath et al. 2025 — **AppraisePLM** (blocks Experiment 2)
- Sun et al. 2024 — **Think-on-Graph** (informs [[entities/graph-rag]])
- Long 2023 — LLM-guided Tree-of-Thought
- Zweiger 2025 — **SEAL** (Self-Adapting Language Models)
- Yang 2025 — Structured Solution Templates
