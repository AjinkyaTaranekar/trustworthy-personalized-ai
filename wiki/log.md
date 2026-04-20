# Log

Append-only chronological journal. Format: `## [YYYY-MM-DD] <kind> | <title>`. Greppable via `grep "^## \[" wiki/log.md`.

`<kind>` ∈ `bootstrap | ingest | query | lint | decision | refactor`.

---

## [2026-04-19] bootstrap | wiki initialised
- Created `CLAUDE.md` schema at repo root.
- Created `wiki/` scaffolding: `index.md`, `log.md`, `overview.md`, and dirs `topics/`, `entities/`, `sources/{papers,dissertation,code}/`, `experiments/`, `decisions/`, `questions/`, `queries/`.
- Seeded topic stubs: `reasoning`, `personalisation`, `empathy`, `tool-use-and-verification`.
- Seeded entity: `constitution` (pointer to `pipeline/constitution.md`).
- Seeded decision: `2025-11-10-ontology-focus-shift` (drawn from `docs/Dissertation/Experimental Planning Document.md` meeting summary).
- **Not ingested yet:** 33 PDFs in `docs/Assets/`, 32 existing per-paper notes in `docs/Literature Notes/`, four dissertation drafts in `docs/Dissertation/`, `researchplan.tex`, `pipeline/` scripts. User drives ingestion one-at-a-time via §4.1.
- **Raw layer left untouched** to preserve Obsidian wikilinks and git history.

## [2026-04-20] refactor | tag vocabulary unified
- Audited tags across all 57 wiki files — found multiple duplicates/typos that made tag-based pivoting unreliable.
- Created `wiki/tags.md` — canonical tag registry organised by category (themes, techniques, entities, modalities, evaluation, document types, workflow) with per-tag counts + meanings and a "Deprecated — do not use" section.
- Updated `CLAUDE.md` §3 with a new §3.1 "Tag discipline" subsection requiring future writers to consult `tags.md` first and add any new tag there in the same edit.
- Normalisations applied across 14 files:
  - `foundation` → `foundations` (6 papers)
  - `tokenization` → `tokenisation` (bpe + llm-foundations; British to match user's prose)
  - `tools` → `tool-use` (tool-use-and-verification)
  - `vectors` → dropped as redundant with `embeddings` (word2vec)
  - `experiment-6` → dropped (ontology-integration) — tagging by experiment number is fragile
  - `affect` / `emotion` / `psychology` / `framework` / `model` / `base` → dropped where redundant (empathy, appraisal-theory, 5w-h, qwen3-0.6b, personalisation, graph-rag)
  - Added proper entity tags: `5w-h`, `appraisal-theory`, `graph-rag` on pages representing or closely referring to those entities
- Linked `tags.md` from `index.md` Meta section.
- Convention preserved: flow-style YAML (one line). One file (`understanding-r1-zero.md`) remained in block style after a prior linter pass; tags there are already canonical (no change needed).

## [2026-04-19] lint | post-batch-2 health check
- **Checked clean:** no contradictions; no orphan pages (every file has ≥1 inbound link via `index.md` or a topic).
- **Stale annotations found + fixed:** 5 `_(not yet …)_` strings pointing at pages that now exist (`sft-v2-pipeline`, `mcp`, `grpo`, `qwen3-0.6b` × 2).
- **Stale conflict marker** in `index.md` (constitution 18/19) — removed.
- **Dangling link flagged:** `[[entities/graph-rag]]` referenced from 6 pages with no target → resolved by filing the entity (below).
- **Concept-density signal:** "ontology" appears 66× across 19 files without a dedicated page → resolved by filing `topics/ontology-integration` (below).
- **Sources not yet held:** research plan cites 5 papers not in `docs/Assets/` (AppraisePLM, Think-on-Graph, LLM-guided ToT, SEAL, structured templates) — now tracked at the bottom of `index.md` and in `questions/2026-04-19-initial-questions`.
- **Flagged in tex:** `researchplan.tex` Phase 3 dates say "Dec'25 - Mar'25", almost certainly a Mar'26 typo. Raw not edited.

## [2026-04-19] ingest | research plan + graph-rag + ontology-integration
- Created [[sources/dissertation/research-plan]] summarising `researchplan.tex` — thesis title, formal research question (narrower than the exploratory one), 5 SMART objectives, 7-phase timeline (Oct '25 – Aug '26), two documented pivots (Oct monolithic→modular, Nov emergent-questioning→5W+H), ethics requirements, keyword+AI search strategy.
- Created [[entities/graph-rag]] — un-dangles 6 inbound references; pulls in the Cognee/FalkorDB/Neo4J backend decision + Think-on-Graph literature pointer.
- Created [[topics/ontology-integration]] — flagship topic anchoring Experiment 6; covers Approach A (KB) and B (verifier) with design-decision list + scrutability-vs-performance tension.
- Cross-linked: `topics/tool-use-and-verification`, `topics/personalisation`, `entities/5w-h`, `entities/rag`, `decisions/2025-11-10-ontology-focus-shift`, `experiments/experiment-catalog`, `overview`, `index` — all updated.
- `index.md` now has a "Cited but not in docs/Assets/" subsection listing the five missing-PDF papers to acquire.

## [2026-04-19] decision | resolved batch-2 conflicts
- **Conflict 1 — constitution 18 vs 19.** User confirmed 19. Edited `pipeline/constitution.md` intro line from "18 principles" to "19". Removed the ⚠ flag from `wiki/entities/constitution.md` and `wiki/sources/code/constitution-document.md`.
- **Conflict 2 — GRPO trainer missing from pipeline.** User confirmed: the GRPO trainer lives on a separate branch, not `main`. Updated `wiki/sources/code/training-and-benchmark.md` with a branch note (removed the "memory stale" caveat). Updated `wiki/overview.md` to remove the Experiment-1-priority tension callout — Experiment 1 is active supporting infrastructure for Experiment 6 comparisons. Marked both items resolved in `wiki/questions/2026-04-19-initial-questions.md`.
- One residual: the `project_state` memory still references the RL-branch file paths from `main`. Not corrected — left as a ToDo for next code-editing turn.

## [2026-04-19] ingest | papers batch 2 + dissertation drafts + code (21 papers + 3 drafts + 3 code-summary pages)
- Ingested remaining 21 papers → `wiki/sources/papers/`:
  - Architectural: `hierarchical-reasoning-model`, `looped-transformers-reasoning`, `coconut-continuous-latent`, `ladir`, `state-stream-transformer`, `diffusion-of-thoughts`
  - RL: `vapo`, `interleaved-reasoning`, `understanding-r1-zero`, `self-enhanced-reasoning`, `hidden-reasoners`
  - Evaluation / caveats: `token-hungry-deepseek-r1`, `none-of-the-others`, `prompting-science-report-2`, `auto-cot`
  - Distillation: `dual-head-reasoning-distillation`
  - Retrieval: `rag-original`
  - Foundations: `measuring-word-significance`
  - Multimodal RL: `ui-r1`, `vlm-r1`
  - Empathy: `xai-sentiment-deepseek-r1` (seeds the empathy topic's first real source)
- Ingested dissertation drafts → `wiki/sources/dissertation/`:
  - `road-towards-trustworthy-empathetic-ai`, `experimental-planning-document`, `personal-notes` (Experiment.md + Rough Notes.md + Research Plan Edits.md)
- Ingested code → `wiki/sources/code/`:
  - `sft-v2-pipeline` (question gen → gold response → rejection sample → assemble)
  - `constitution-document` (full 19-principle source summary with the 18/19 conflict flagged)
  - `training-and-benchmark` (v1 scripts, LoRA + GRPO, benchmark, context degradation)
- Created new entities: `grpo`, `mcp`, `rag`, `qwen3-0.6b`, `5w-h`, `appraisal-theory`.
- Created new topic: `explainability`.
- Created `experiments/experiment-catalog` consolidating all 6 experiments + ablation A/B/C/D.
- Created `questions/2026-04-19-initial-questions` consolidating user TODOs + advisor-prep questions + literature tensions.
- Updated every existing topic page and `overview.md` to cross-link new content; regenerated `index.md`.
- **Conflicts flagged:**
  - `pipeline/constitution.md` intro line says "18 principles" while body + summary + README say "19". Flagged in `entities/constitution.md` and `sources/code/constitution-document.md`; left raw source untouched.
  - Experiment 1 (process-reward RL) is ranked "Lower Priority" in the planning doc yet fully implemented in `pipeline/` — flagged in `overview.md`, `experiments/experiment-catalog`, and `questions/2026-04-19-initial-questions`.
- **Stale memory risk:** `project_state` memory (dated 2026-03-27, now 22+ days old) references a `pipeline/2c_rl_trainer.py` not present in current repo. Flagged in `sources/code/training-and-benchmark`.

## [2026-04-19] ingest | papers batch 1 — foundations, reasoning, tool-use (12 papers)
- Ingested 12 papers from `docs/Literature Notes/` to `wiki/sources/papers/`:
  - Foundations: `attention-is-all-you-need`, `bert`, `word2vec`, `bpe-subword-units`
  - Reasoning: `chain-of-thought-prompting`, `tree-of-thoughts`, `deepseek-r1`, `seed15-thinking`
  - Tool-use: `pal`, `react`, `mcp-multi-agent`, `search-r1`
- Created new topic page `topics/llm-foundations.md` (tokenisation, attention, embeddings as the architectural reason the thesis needs modularity).
- Updated `topics/reasoning.md` and `topics/tool-use-and-verification.md` with ingested-paper bullets.
- Updated `overview.md` to add the foundations pillar (pillar 0).
- Regenerated `wiki/index.md` with the new structure.
- **Conflicts flagged:** none.
- **Questions opened:** none filed yet; several `entities/*` pages referenced but deferred (`grpo`, `mcp`, `qwen3-0.6b`, `graph-rag`, `rag`) — create when next ingest touches them.
- **Batch 2 candidates** listed at the end of `index.md` § "Papers (not yet ingested)".
