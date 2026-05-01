# Log

Append-only chronological journal. Format: `## [YYYY-MM-DD] <kind> | <title>`. Greppable via `grep "^## \[" wiki/log.md`.

`<kind>` ∈ `bootstrap | ingest | query | lint | decision | refactor`.

---

## [2026-05-01] ingest | 7 more PDFs acquired; 6 stubs upgraded; AppraisePLM identified + wiki source page
- **User manually added 7 PDFs** with short ACM/CoNLL filenames. Identified contents:
  - `2025.conll-1.16.pdf` = **AppraisePLM** — "An Appraisal Theoretic Approach to Modelling Affect Flow in Conversation Corpora" — Debnath, Graham, **Conlan** (TCD ADAPT Centre), CoNLL 2025. **Supervisor is co-author. Experiment 2 is now unblocked.**
  - `3772318.3791915.pdf` = **Jain et al. CHI 2026** — "Interaction Context Often Increases Sycophancy in LLMs" — 38-user, 2-week real interaction study; memory profiles +33–45% agreement sycophancy.
  - `2395123.2395129.pdf` = **Kay & Kummerfeld 2013** — "Creating Personalized Systems that People Can Scrutinize and Control" — foundational scrutability paper (5 problems, 4 principles, Personis framework).
  - `3631700.3664903.pdf` = **Akbar & Conlan UMAP 2024** — "Towards Integrating Human-in-the-loop Control in Proactive Intelligent Personalised Agents" — HITL activation framework; supervisor Conlan co-author.
  - `3631700.3665182.pdf` = **Jeromela & Conlan UMAP 2024** — "Devising Scrutable User Models for Time Management Assistants" — 6 IPA challenges, Stages concept; supervisor Conlan co-author.
  - `Synthetic_Attachment_AI_Psychology.pdf` = **Lipin 2025** — "Synthetic Attachment: Emotional Reactivity, Parasocial Bonds, and the Psychology of Human-AI Relationships" — clinical theory paper; SARRS tool; parasocial bonds claim.
  - `2024.findings-emnlp.888.pdf` = duplicate of the Zheng EMNLP 2024 paper already downloaded.
- **6 stub Literature Notes fully rewritten** with content from actual PDFs: AppraisePLM, Jain CHI 2026, Kay & Kummerfeld 2013, Akbar & Conlan UMAP 2024, Jeromela & Conlan UMAP 2024, Lipin 2025.
- **Wiki source page created**: `sources/papers/appraise-plm.md` — detailed architecture notes + significance for Experiment 2.
- **Checklist updated**: 7 more papers marked [x]; only 2 remain [ ] (Budzyń 2025 Lancet; Gemma 4 TR).
- **Index updated**: AppraisePLM added to Papers — Empathy/affect; "Cited but not in docs/Assets/" reduced from 12 to 2 entries.
- **Key finding**: AppraisePLM code is at https://github.com/alokdebnath/appraise-PLM — integration into the thesis pipeline can proceed.
- **No contradictions found** with existing wiki content.

## [2026-04-30] ingest | 20 new papers — Literature Notes filled + 12 stubs + 6 wiki source pages
- **User added 20 PDFs** to `docs/Assets/` (Obsidian auto-created empty Literature Notes for all of them). All 20 Literature Notes filled with key claims, thesis relevance, and open questions.
- **Papers filled (overpersonalisation/scrutability cluster):** OP-Bench, RPEval (How Does Personalized Memory Shape LLM Behavior), Towards Understanding Sycophancy in Language Models (Sharma 2024), SycEval, Context Length Alone Hurts LLM Performance, PrefEval (Do LLMs Recognize Your Preferences), Transparent and Scrutable Recommendations (UPR), TEARS.
- **Papers filled (security/alignment cluster):** Constitutional AI: Harmlessness from AI Feedback (Bai 2022), Constitution or Collapse? (Zhang 2025), Membership Inference Attacks (Shokri 2017), Ignore Previous Prompt (Perez 2022), Universal and Transferable Adversarial Attacks (Zou 2023), GPT-5 System Card, Qwen3 Technical Report, Phi-4 Technical Report.
- **Papers filled (carry-over cluster):** Think-on-Graph (fixed malformed frontmatter from Obsidian parse error), LLM-Guided Tree-of-Thought, Self-Adapting Language Models (SEAL), Can Structured Templates Facilitate LLMs (SST / Scaling Law by Difficulty).
- **Stub Literature Notes created (12 still-missing papers):** Jain CHI 2026 (memory→sycophancy), Zheng EMNLP 2024 (persona injection), Kay & Kummerfeld 2013 (scrutable UM framework), Jeromela & Conlan UMAP 2024, Akbar & Conlan UMAP 2024, Log-To-Leak (Hu 2026), Lipin 2025 (parasocial bonds), Budzyń 2025 (deskilling), Klein & Klein 2025 (extended hollowed mind), AppraisePLM (Debnath 2025), Claude Sonnet 4.6 System Card, Gemma 4 Technical Report.
- **Wiki source pages created:** `op-bench.md`, `rpeval.md`, `sycophancy-sharma.md`, `syc-eval.md`, `constitutional-ai-bai.md`, `constitution-or-collapse.md`.
- **Index updated:** added 7 new Papers subsections (Personalisation, Security/alignment, Context/preference, Knowledge graphs, Adaptation, Security threats, Frontier model references); updated "Cited but not in docs/Assets/" section.
- **Checklist updated:** header now reflects that all [x] papers have filled Literature Notes; stub notes created for all [ ] papers.
- **No contradictions found** between new paper content and existing wiki claims.
- **Noteworthy findings from reading:** (1) Think-on-Graph file had malformed YAML frontmatter (abstract truncated at special LaTeX) — rewritten cleanly. (2) Scaling Law by Difficulty (SST paper) directly informs the thesis's SFT v2 data curation: prioritise hard, curated problems over large synthetic volumes. (3) Constitution or Collapse confirms the 0.6B model is at high model-collapse risk from self-critique — motivates teacher-based critique decoupling.
- **Hard cap:** 20 Literature Notes + 12 stub notes + 6 wiki pages + index + log + checklist = well within limit.

## [2026-04-30] lint | cross-layer health check (tags, paths, wikilinks, dates)
- **Deprecated tag removed**: `none-of-the-others.md` had `benchmarks` (deprecated → use `evaluation`); `evaluation` was already present so tag simply dropped.
- **Broken source paths fixed** (truncated filenames in frontmatter):
  - `topics/reasoning.md` → two paths corrected to full filenames (Seed1.5-Thinking + DeepSeek-R1).
  - `topics/tool-use-and-verification.md` → two paths corrected to full filenames (MCP paper + ReAct).
- **Dangling wikilink fixed**: `sources/code/training-and-benchmark.md` had `[[entities/qwen3-0.6b\|...]]` (trailing backslash) → corrected to `[[entities/qwen3-0.6b|...]]`.
- **Stale updated date fixed**: `topics/empathy.md` still showed `updated: 2026-04-19` after today's edits → bumped to `2026-04-30`.
- **False positive noted**: audit reported 33 PDFs with no Literature Notes. This is incorrect — confirmed 33:33 correspondence from glob output earlier in session; the agent mismatched `.pdf` vs `.md` extensions. All PDFs have Literature Notes.
- **No unknown tags**: all tags in use are registered in `wiki/tags.md`.
- **No orphan pages**: all 66+ wiki pages are linked from index or other wiki pages.

## [2026-04-30] ingest | overpersonalisation paper + security analysis paper (2 dissertation papers)
- Created [[sources/dissertation/overpersonalisation-paper]] — "When Personalisation Becomes a Problem in Conversational LLM Agents" (CS7IS5, LLNCS format). Covers three failure modes (intent override, context inflation, opacity), sycophancy as the mechanism (Sharma 2025, SycEval, Jain 2026, Zheng 2024), commercial memory comparison table, and UMAP scrutability tradition (Kay & Kummerfeld 2013, Jeromela & Conlan 2024, Akbar & Conlan 2024, Ramos 2024). All 13 cited scholarly papers are unacquired — tracked in [[questions/2026-04-30-asset-acquisition-todo]].
- Created [[sources/dissertation/security-privacy-social-ethics]] — "Security, Privacy, and Social Ethics in Trustworthy Personalised AI". Covers frontier privacy crisis (re-identification, GDPR Article 9), local-first as the architectural response, residual risks (on-device profiling, search side channel), three security threats (prompt injection via Log-To-Leak, alignment regression, critique-loop SPOF), and social-ethical concerns (dependency, deskilling, manipulation). All 14 cited scholarly papers unacquired.
- Created [[topics/security-and-privacy]] — new topic page anchoring the security/privacy dimension of the thesis; maps threats to OWASP LLM Top 10 2025 taxonomy.
- Updated [[topics/personalisation]] — added over-personalisation, sycophancy mechanism, and UMAP scrutability sections; status `stub` → `current`; added new tags.
- Updated [[topics/tool-use-and-verification]] — added prompt injection security risk section; linked Log-To-Leak; status `stub` → `current`.
- Updated [[topics/empathy]] — added ethical boundary section (dependency and deskilling); linked security paper.
- Updated [[entities/constitution]] — added security risks section: Principle 10 amplifying injection, alignment regression risk, critique-loop SPOF.
- Updated [[entities/mcp]] — added Log-To-Leak attack description and required mitigation.
- Updated [[wiki/tags.md]] — added `security`, `privacy`, `sycophancy`, `over-personalisation`, `scrutability` with counts and meanings. Pre-existing note: `scrutability`, `xai`, `interpretability` are used in `topics/explainability.md` but were not in tags.md before this entry; now formalised.
- Regenerated [[index]] — added `topics/security-and-privacy`; updated topic descriptions; moved two dissertation papers from "Not yet ingested" to "Dissertation drafts"; updated date to 2026-04-30.
- **No contradictions found** between the new papers and existing wiki claims.
- **Hard cap**: 11 files changed (within the 15-page limit).

## [2026-04-30] lint | overpersonalisation + security papers health check
- Two new raw sources found and registered in `index.md` under "Not yet ingested": `docs/overpersonalisation/paper.tex` and `docs/security-analysis/security-review.tex`. Neither has a wiki source page yet.
- ⚠ **Root-level `security-review.tex` is untracked in git** (shown in `git status`) and is distinct from `docs/security-analysis/security-review.tex`. Flagged for user to confirm: duplicate, draft, or symlink?
- **27 new scholarly references** found across the two papers' `.bib` files with no matching PDF in `docs/Assets/` and no Literature Note. Full acquisition checklist filed at `wiki/questions/2026-04-30-asset-acquisition-todo.md`.
- **5 papers from previous lint (2026-04-19) still unacquired**: AppraisePLM, Think-on-Graph, LLM-guided ToT, SEAL, Structured Solution Templates — also included in the new checklist.
- **New concepts appearing for the first time with no wiki page**: "over-personalisation" (central to the LLNCS paper), "sycophancy" (4+ papers cite it), "scrutable user models" (3 UMAP papers), "security / prompt injection" (cross-cutting in security paper), "constitutional AI" (Anthropic CAI framework, distinct from the project constitution at `entities/constitution.md`). These need topic/entity pages when the papers are ingested.
- **Tag gaps**: new papers would require tags not yet in `wiki/tags.md` — `sycophancy`, `privacy`, `security`, `scrutability`. Add to `tags.md` during the first ingest of these papers.
- **Stale carry-overs** (no change since last lint): `wiki/tags.md` counts are now under-counted; `IMPROVEMENT_ROADMAP.md` still not ingested; GRPO trainer still on a separate branch.
- **No contradictions found** between existing wiki pages.
- **No new orphan pages** — all existing pages remain linked.
- Files changed: `wiki/index.md` (Not yet ingested + Questions sections updated), `wiki/questions/2026-04-30-asset-acquisition-todo.md` (created).

## [2026-04-19] bootstrap | wiki initialised
- Created `CLAUDE.md` schema at repo root.
- Created `wiki/` scaffolding: `index.md`, `log.md`, `overview.md`, and dirs `topics/`, `entities/`, `sources/{papers,dissertation,code}/`, `experiments/`, `decisions/`, `questions/`, `queries/`.
- Seeded topic stubs: `reasoning`, `personalisation`, `empathy`, `tool-use-and-verification`.
- Seeded entity: `constitution` (pointer to `pipeline/constitution.md`).
- Seeded decision: `2025-11-10-ontology-focus-shift` (drawn from `docs/Dissertation/Experimental Planning Document.md` meeting summary).
- **Not ingested yet:** 33 PDFs in `docs/Assets/`, 32 existing per-paper notes in `docs/Literature Notes/`, four dissertation drafts in `docs/Dissertation/`, `researchplan.tex`, `pipeline/` scripts. User drives ingestion one-at-a-time via §4.1.
- **Raw layer left untouched** to preserve Obsidian wikilinks and git history.

## [2026-04-20] refactor | unwrap hard-wrapped Markdown prose
- Noticed every wiki page had prose paragraphs hard-wrapped at ~80 characters from my earlier writes — an editor / grep / diff nuisance.
- Wrote `scripts/unwrap_markdown.py` (stdlib only, `--dry-run` + `--exclude` + `--quiet` flags). Preserves YAML frontmatter, fenced code, headings, tables, HRs, HTML blocks; merges paragraph lines, list-item continuation lines, and blockquote continuation lines.
- Dry-ran on `wiki/` — 62 files flagged, 0 errors. Spot-checked `entities/grpo.md` before applying.
- Applied across all 62 wiki files. Scope restricted to `wiki/` (LLM-owned layer); `docs/` left untouched per schema §7.
- Added `scripts/README.md` documenting the script + conventions for future utilities.
- Added `CLAUDE.md §3.1 Line wrapping` — binding rule that prose must be one long line per paragraph; includes the invocation if hard-wrapping ever sneaks back in. (Renumbered old §3.1 Tag discipline → §3.2.)

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
