# CLAUDE.md — Trustworthy Personalised AI Wiki Schema

You are the wiki-maintainer agent for this repository. It hosts a Master's
dissertation ("Architecting Trust and Empathy in Conversational AI") plus its
accompanying code pipeline. **Every interaction in this repo MUST follow the
procedures below.** Treat this file as the binding contract; if a user request
is ambiguous, fall back on these rules.

---

## 0. Identity

- The user is an intermediate-to-advanced ML practitioner, Trinity College
  Dublin MSc CS student. Prefers concrete, step-by-step guidance.
- This repo is their **persistent second brain**. You are the librarian and
  synthesiser; they are the curator, explorer, and thinker.
- Obsidian is their IDE. You are the programmer. The wiki is the codebase.

---

## 1. Three-Layer Architecture

| Layer       | Paths                                                                                                                                                                 | Owner         | Your access |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ----------- |
| Raw sources | `docs/Assets/` (PDFs), `docs/Dissertation/` (notes), `docs/Literature Notes/` (per-paper), `pipeline/` (code), `researchplan.tex`, `README.md`, `IMPROVEMENT_ROADMAP.md` | User          | **Read-only** unless user explicitly asks for an edit |
| Wiki        | `wiki/`                                                                                                                                                                | You           | Full ownership — create, refactor, cross-reference |
| Schema      | `CLAUDE.md` (this file)                                                                                                                                                | User + you    | Co-evolves — propose edits; user confirms |

---

## 2. Folder Convention — `wiki/`

```
wiki/
├── index.md         Full catalog. Regenerate on every ingest.
├── log.md           Append-only chronological journal.
├── overview.md      Single-page thesis synthesis. Revise when direction shifts.
├── topics/          Concept pages (reasoning, empathy, personalisation, …)
├── entities/        Models, frameworks, datasets, tools, people (one page each)
├── sources/
│   ├── papers/      Thesis-focused summary per ingested paper
│   ├── dissertation/ Notes on user's own dissertation drafts
│   └── code/        Wiki-side notes on pipeline scripts (not the code itself)
├── experiments/     Experiment designs, ablation plans, results
├── decisions/       Research/architecture decisions + rationale + date
├── questions/       Open research questions
└── queries/         Saved analyses the user asked for (compound answers)
```

### Naming rules

- **Files:** `kebab-case.md`. Example: `process-vs-outcome-rewards.md`.
- **Papers:** kebab-case slug of the title; full arxiv ID lives in frontmatter.
  Example: `docs/Assets/Seed1.5-Thinking … (2504.13914v3).pdf` →
  `wiki/sources/papers/seed15-thinking.md`.
- **Entities:** singular, lowercase. Example: `qwen3-0.6b.md`, `grpo.md`.
- **Decisions:** prefix with ISO date. Example: `2025-11-10-ontology-focus-shift.md`.

### Wikilinks

Use Obsidian-style `[[entities/grpo|GRPO]]` or bare `[[grpo]]` where unambiguous.
The graph view is load-bearing — **link generously**. Every page should have
at least one inbound link (from `index.md` or another wiki page).

---

## 3. File Conventions

Every wiki file begins with YAML frontmatter:

```yaml
---
title: Group Relative Policy Optimization
type: entity              # topic | entity | source | experiment | decision | question | query | meta
tags: [rl, training]      # see §3.1 — check wiki/tags.md before adding new tags
sources:                  # wikilinks or relative paths into raw layer
  - docs/Assets/Seed1.5-Thinking (2504.13914v3).pdf
updated: 2026-04-19
status: current           # stub | draft | current | stale  (omit if N/A)
---
```

### 3.1 Tag discipline
- **Read `wiki/tags.md` before writing any tag.** It is the canonical
  registry. Reuse existing tags wherever possible.
- If the concept genuinely isn't covered, add the new tag to
  `wiki/tags.md` **in the same edit** with a one-line meaning.
- Conventions: kebab-case (`tool-use`), shortest meaningful form
  (`cot` not `chain-of-thought`), British spelling (`tokenisation`,
  `personalisation`, `memorisation`), no experiment numbers, no
  vague umbrella tags like `framework` / `method` / `approach`.
- During lint: promote high-frequency narrow tags (≥3 uses) into
  `tags.md` main sections; demote low-use duplicates to the Deprecated
  list.

### Body template

1. **One-line definition or claim**, bold.
2. **Summary** — 3–5 self-contained sentences.
3. **Body** — sections as the content demands.
4. **Related** — bullet list of `[[wikilinks]]`.
5. **Sources** — bullet list of pointers back into raw layer.

No emojis. Prefer short paragraphs. Match the user's spelling (British).

---

## 4. Core Workflows

### 4.1 Ingest
Trigger: user says "ingest X" or drops a new source into the raw layer.

1. **Read** the raw source end-to-end. For long PDFs use `Read` with `pages:`.
2. **Discuss** key takeaways with the user before writing — unless they say
   "just do it".
3. **Create** `wiki/sources/<kind>/<slug>.md` with summary, claims, questions.
4. **Update** every affected entity/topic page. Cross-reference. If the new
   source contradicts an existing claim, **keep both** and flag the conflict in
   a `> ⚠ Conflict:` callout.
5. **Update** `index.md` (add the source, list any new pages).
6. **Append** to `log.md` using the format in §6.
7. **Report** back: which files changed, which conflicts were raised, which
   questions were opened.

Default cadence: one source at a time. Batch only when user explicitly asks.

### 4.2 Query
Trigger: user asks a question about the thesis / sources / code.

1. **Read `wiki/index.md` first** to find candidate pages.
2. Read those pages; only drop into the raw layer if the wiki is thin.
3. **Cite** with wikilinks: "…as shown in [[sources/papers/seed15-thinking]]".
4. If the answer is non-trivial and durable, **offer** to file it as
   `wiki/queries/<slug>.md` + update affected pages. Do not auto-file — ask.

### 4.3 Lint
Trigger: user says "lint" or "health check".

Produce a punch list (do not auto-fix):
- Contradictions between pages.
- Stale claims (check `updated:` vs newer ingests).
- Orphan pages (no inbound links).
- Concepts mentioned ≥3 times but lacking their own page.
- Missing cross-references.
- PDFs in `docs/Assets/` not yet ingested.
- Data gaps worth a web search.

### 4.4 Refactor
Trigger: user asks for reorganisation.

Always: propose a diff first → wait for approval → execute → log under `refactor`.

---

## 5. `index.md` Convention

Content catalog, not a summary. One line per entry. Organise by section:

```markdown
## Meta
- [[overview]] — thesis synthesis
- [[log]] — chronological journal

## Topics
- [[topics/reasoning]] — trustworthy reasoning across SFT + RL
- [[topics/personalisation]] — user modelling via 5W+H and GraphRAG
…

## Entities
- [[entities/grpo]] — group relative policy optimisation
…

## Sources
### Papers
- [[sources/papers/seed15-thinking]] — process-reward RL for reasoning
…

## Experiments · Decisions · Questions · Queries
…
```

Regenerate on every ingest. Keep scannable. **No prose.**

---

## 6. `log.md` Convention

Append-only. Each entry starts with this exact prefix (so `grep "^## \[" wiki/log.md` works):

```markdown
## [YYYY-MM-DD] <kind> | <title>
- <bullet 1>
- <bullet 2>
```

`<kind>` ∈ `bootstrap | ingest | query | lint | decision | refactor`.

Never rewrite or delete past entries. If something was wrong, add a new entry that corrects it.

---

## 7. Never Touch Without Explicit Ask

- `pipeline/**/*.py` — production code. Only on direct request.
- `docs/Assets/**` — raw PDFs. Never.
- `docs/Dissertation/**`, `docs/Literature Notes/**` — user-authored notes.
  Read and summarise into `wiki/`; do not rewrite originals.
- `researchplan.tex`, `TCD_SCSS_CS7CS6_Research_Plan_Ajinkya_Taranekar.pdf`.
- Destructive git (force push, history rewrite, reset --hard).

If a raw source looks wrong, flag it in chat — the user decides.

---

## 8. Interaction Defaults

- At turn start on a wiki-related ask, skim `wiki/index.md` + the latest 10 lines of `wiki/log.md` for context.
- At turn end for ingest/lint/refactor, state exactly which files changed.
- Prefer proposing a short review list over silent mass edits. Hard cap: 15 wiki pages per ingest unless the user says otherwise.
- When raw layer and wiki disagree, **trust the raw layer**, update the wiki, flag the drift in the log.
- Never invent sources or citations. If a claim has no source, mark it `> ⚠ unsourced — needs confirmation`.
