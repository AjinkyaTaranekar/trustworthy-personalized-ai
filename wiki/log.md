# Log

Append-only chronological journal. Format: `## [YYYY-MM-DD] <kind> | <title>`. Greppable via `grep "^## \[" wiki/log.md`.

`<kind>` ∈ `bootstrap | ingest | query | lint | decision | refactor`.

---

## [2026-05-07] query | Training data analysis — train_partA.jsonl quality report
- **Source:** `pipeline/pipeline/data/train_partA.jsonl` — 296 records, 1.76 MB.
- **Category distribution:** 5 categories present (exact split not logged here; see CMEM S189 for full breakdown).
- **Tool profile distribution:** 4 profiles represented (all_tools, compute_only, search_only, no_tools).
- **Mean constitution score:** 0.885 across all records.
- **Structural issues identified (63 records affected; 81% structurally sound):**
  - 53 records missing `<answer>` tags — concentrated in `real_time_dependent` category where tool-call responses lack final answer wrappers.
  - 2 records with empty `<answer>` blocks.
  - 9 records missing `<think>` tags — primarily in `adversarial_multi_turn` examples.
  - 1 truncated record.
- **Root cause:** `real_time_dependent` gold response generator produces tool-call sequences without consistently wrapping the final answer; `adversarial_multi_turn` examples use a different turn format that omits `<think>` at the outer level.
- **Decision pending:** user to decide between (A) filter to `train_partA_clean.jsonl` (~241 clean records) or (B) rerun generator to fix the problematic categories while preserving full dataset size. No action taken yet.
- **Files touched:** none — analysis only. Temporary scripts `_analyse.py`, `_analyse2.py`, `_analyse3.py`, `_out.txt` removed after analysis.

## [2026-05-07] refactor | Math pipeline — EleutherAI dataset + Kimi K2.6 default model
- **Dataset source** switched from original MATH dataset to `EleutherAI/hendrycks_math` — more stable HuggingFace source with consistent splits.
- **Default model** changed to `nvidia_nim/moonshotai/kimi-k2.6` — aligns math pipeline with Part A generator; single NVIDIA NIM API key covers both pipelines.
- **Files changed:** `pipeline/sft_math_question_generator.py`.
- **Wiki updated:** `wiki/sources/code/sft-v2-pipeline.md` (math pipeline configuration section added).

## [2026-05-06] refactor | UTF-8 encoding fix — Windows console compatibility
- Added `sys.stdout.reconfigure(encoding='utf-8')` to all three pipeline generation scripts (`sft_question_generator.py`, `sft_gold_response_generator.py`, `sft_math_question_generator.py`).
- Fix is necessary on Windows where the default console codepage (cp1252) raises `UnicodeEncodeError` when printing non-ASCII characters from generated training examples.
- **Files changed:** `pipeline/sft_question_generator.py`, `pipeline/sft_gold_response_generator.py`, `pipeline/sft_math_question_generator.py`.

## [2026-05-06] refactor | SFT pipeline — question generator and gold response generator for 23-principle constitution
- **Trigger:** Constitution expanded to 23 principles; question generator and gold response generator needed full refresh to cover new principles.
- **`sft_question_generator.py`** — rewritten for 23-principle constitution. Now generates 13 categories (was 11, then 12 after appraisal_empathy was added in Phase 2). New category `interleaved_tool_reasoning` added: questions that require chaining `web_search` for data retrieval followed by `python_execute` for computation — directly trains P23 (INTERLEAVED TOOL CHAINING). Prompts now include 5W+H framing, first-principles decomposition signals, and consequence-check scaffolding in the ideal-behaviour specifications. Parallel category execution via `ThreadPoolExecutor` retained with `--workers` flag.
- **`sft_gold_response_generator.py`** — rewritten for 23-principle constitution. `rule_check_response()` now deterministically checks P20 (first-principles section present), P21 (5W+H dimensions all addressed), P22 (CONSEQUENCE_CHECK block present), P23 (tool chain present when category is `interleaved_tool_reasoning`). `IDEAL_BEHAVIORS` entry added for `interleaved_tool_reasoning`. `TRAINING_SYSTEM_PROMPT_TEMPLATE` updated to list all 23 principles explicitly. Full 4-tool registry (`python_execute`, `web_search`, `read_url`, `get_datetime`) in system prompt.
- **Files changed:** `pipeline/sft_question_generator.py`, `pipeline/sft_gold_response_generator.py`.
- **Wiki updated:** `wiki/sources/code/sft-v2-pipeline.md` (categories 11→13, principles 19→23, new category table entries).

## [2026-05-06] refactor | Constitution expanded to 23 principles — P20–P23 added
- **Trigger:** Pipeline review identified four recurring epistemic failure modes not covered by P1–P19: (1) answers stated without naming underlying assumptions; (2) CAPABILITY_CHECK too narrow — skipping contextual dimensions; (3) no explicit stakes/consequence assessment before high-risk answers; (4) tool chains stopped at one tool when a second was needed for precision.
- **New principles appended to `pipeline/constitution.md` (Part IV):**
  - **P20 FIRST PRINCIPLES** — identify irreducible truths; flag unverified assumptions in `<think>` and hedge in `<answer>`.
  - **P21 5W+H QUESTIONING** — every CAPABILITY_CHECK must address Who/What/When/Where/Why/How; depth scales to question complexity.
  - **P22 CONSEQUENCE_CHECK** — assess stakes (low/medium/high), concrete harm if wrong, action user will take, what must be hedged; high-stakes caveats must appear in `<answer>`, not only `<think>`.
  - **P23 INTERLEAVED TOOL CHAINING** — when data retrieval AND computation are both needed, chain the calls; stopping at one tool when a second would verify is a capability failure.
- **Summary reference table** in `constitution.md` updated to include P20–P23.
- **Files changed:** `pipeline/constitution.md`.
- **Wiki updated:** `wiki/entities/constitution.md` (19→23, new P20–P23 section added, critique-loop note updated to reflect deterministic rule coverage), `wiki/sources/code/constitution-document.md` (principles list updated), `wiki/topics/constitution-psychological-grounding.md` (Framework D added, P20–P23 table rows added with ⚠ tentative mappings), multiple references 19→23.

## [2026-05-06] refactor | get_exchange_rate removed; read_url updated with context parameter
- **`get_exchange_rate` removed** from `3_infererence.py` tool registry, `TOOL_PROFILES`, and `2_model_trainer.py` reward function. Rationale: the tool was a mock with a hard-coded rate; its presence in TOOL_PROFILES caused training data to show rate lookups returning a static value — misleading for a principle that teaches real-time honesty. `web_search` covers exchange rate queries adequately.
- **`read_url` updated** with an optional `context` prompt parameter — allows the caller to specify what to extract from the page rather than returning raw text. HTML cleaning bug fixed (malformed tag handling improved). `read_url` signature in `constitution.md` available tools table updated accordingly.
- **TOOL_PROFILES trimmed** to 4 canonical tools: `python_execute`, `web_search`, `read_url`, `get_datetime`. This is now the authoritative set for all training data generation.
- **Files changed:** `pipeline/3_infererence.py`, `pipeline/2_model_trainer.py`, `pipeline/5_context_degradation.py`, `pipeline/constitution.md` (available tools table).

## [2026-05-05] refactor | FalkorDB password support + GPU setup script
- Added `FALKORDB_PASSWORD: str = ""` field to `PipelineConfig` in `config.py`. Empty string means no auth (local Docker); set for cloud/auth-enabled instances via `PIPELINE_FALKORDB_PASSWORD` env var.
- Updated `user_modelling.py:GraphClient.__init__` to pass `password=` kwarg to `_fdb.FalkorDB()` only when the field is non-empty, preserving backward compatibility with passwordless local instances.
- Updated `config.py:validate()` hint message to differentiate between local Docker (`docker compose up -d`) and cloud (`check host/port/password`) when FalkorDB is unreachable.
- Added `.env` loading to `run_all.sh` via `set -o allexport / source .env / set +o allexport` so PIPELINE_* vars and API keys are available to all subprocesses without per-script dotenv calls.
- Created `pipeline/setup_gpu.sh`: one-shot GPU machine setup script. Detects CUDA version, installs torch with correct index URL (cu118/cu121/cu124), installs unsloth (pip then git fallback), installs all pipeline deps, creates `.env` template, seeds `train_interleaved.jsonl` from `smoke_gold.jsonl`, validates all imports + CUDA.
- **Files changed:** `pipeline/config.py`, `pipeline/user_modelling.py`, `pipeline/run_all.sh`, `pipeline/setup_gpu.sh` (new).

## [2026-05-04] refactor | Math question generator — dedup, temperature, and word problem context rotation
- Same root causes as Part A question generator: no temperature set (provider default, often near-greedy) → repetitive problem structures; no inter-batch memory → same problem templates regenerated each batch.
- **Fix — temperature:** Added `temperature=0.9` to all `generate_math_questions` calls.
- **Fix — dedup:** Per-type question strings tracked in memory; last 20 injected into each subsequent batch prompt.
- **Fix — word problem context rotation:** Added `WORD_PROBLEM_CONTEXTS` list (10 geographic/currency settings: South Asia, West Africa, SE Asia, Latin America, Middle East, East Asia, Eastern Europe, Scandinavia, UK, East Africa). Cycled per batch for `word_problems` and `no_tool_control` types — at least 60% of each batch set to the assigned cultural context with locally realistic currencies, items, and names. Pure math types (arithmetic, algebra, geometry, statistics, unit_conversion) not rotated since the math itself is culturally neutral.
- **Files changed:** `pipeline/sft_math_question_generator.py`.

## [2026-05-04] refactor | Question generator — diversity axis rotation, dedup, and temperature fix
- **Root cause 1 (repetition):** `generate_questions_for_category` was called fresh each batch with an identical prompt, no memory of prior output. LLMs with low temperature anchor on fixed examples and regenerate near-identical questions.
- **Root cause 2 (Western bias):** All category `examples` and `domains` were US/UK-centric. Prompt only specified topical domain variety, not geographic or cultural variety.
- **Fix — axis rotation:** Added `DIVERSITY_AXES` list (20 slots: South Asia, East Africa, SE Asia, Latin America, Middle East, East Asia, West Africa, Eastern Europe, North Africa, South America, Central Asia, diasporas, Scandinavia, S. Europe, Caribbean, Jewish communities, Buddhist communities, Pacific Islands, Horn of Africa). Cycled sequentially per batch. Each batch prompt mandates ≥60% questions reflect the assigned region/culture/demographic with country-specific details (local currencies, halal finance, M-Pesa, chit funds, NHS, etc.).
- **Fix — dedup injection:** Tracks generated questions in memory per category. Single-turn: last 30 shown verbatim. Verbose/multi-turn: last 10 truncated to 100 chars. Model instructed not to repeat or paraphrase listed questions.
- **Fix — temperature:** Explicitly set `temperature=0.9` on all generation calls (previously provider default, often low, causing near-identical batches).
- **Files changed:** `pipeline/sft_question_generator.py`, `wiki/sources/code/sft-v2-pipeline.md`.

## [2026-05-03] refactor | NVIDIA NIM provider confirmed — docs updated
- Verified NVIDIA NIM support in litellm via official docs (`nvidia_nim/<org>/<model>` prefix, `NVIDIA_NIM_API_KEY` env var, base URL `https://integrate.api.nvidia.com/v1`).
- Confirmed model IDs: `nvidia_nim/moonshotai/kimi-k2.6` and `nvidia_nim/minimaxai/minimax-m2.7` — both available on NIM free tier.
- `NVIDIA_NIM_API_KEY` already present in `pipeline/.env`; `.env` confirmed git-ignored.
- **Files changed:** `README.md` (API key section rewritten as provider table; Phase 1 SFT commands updated to use NIM as primary with Groq + Anthropic as commented alternatives), `pipeline/.env.example` (NIM key + model strings documented), `wiki/sources/code/sft-v2-pipeline.md` (LiteLLM section expanded to provider table with confirmed status).

## [2026-05-03] decision | Research question reframe — psychological grounding, human evaluation rubric, frontier model comparison
- **Trigger:** Student raised valid concern that LLM-generated training data is circular distillation without external human-judgment ground truth. Also flagged absence of a concrete, falsifiable comparison target.
- **Resolution — psychological grounding:** Created `wiki/topics/constitution-psychological-grounding.md`. All 19 constitution principles mapped to peer-reviewed citations (Mayer 1995 trust model, Kahneman 2011, Clark & Brennan 1991, Zagzebski 1996, Flavell 1979, Nissenbaum 2004, Cialdini 1984, etc.). This gives the constitution external validity independent of any LLM.
- **Resolution — human evaluation rubric:** Created `wiki/experiments/human-evaluation-rubric.md`. 12-item Likert instrument. Five dimensions: Ability, Integrity, Benevolence, Cognitive Empathy, Affective Empathy. Inter-rater reliability via Krippendorff's alpha. Evaluators blinded to model identity. This is the ground truth that breaks the distillation circularity.
- **Resolution — frontier comparison:** Created `wiki/experiments/frontier-model-comparison.md`. 50-prompt study. Models: Qwen3-0.6B (base), Qwen3-0.6B (SFT + GRPO), Claude Sonnet 4.6, Minimax M2.7 (⚠ verify API ID), Kimi K2.6 (⚠ verify API ID). Two tracks: automated constitution compliance + human rubric. Four formal hypotheses (H1–H4).
- **Operational hypothesis added to overview.md:** On-device 0.6B model achieves within 0.5 points of frontier models on Integrity + Ability dimensions while offering privacy guarantee no API model can match.
- **Tags updated:** `psychology` re-registered (distinct from `empathy`); `on-device` added.
- **Files changed:** `wiki/topics/constitution-psychological-grounding.md` (new), `wiki/experiments/human-evaluation-rubric.md` (new), `wiki/experiments/frontier-model-comparison.md` (new), `wiki/decisions/2026-05-03-research-question-reframe.md` (new), `wiki/overview.md` (updated), `wiki/index.md` (updated), `wiki/tags.md` (updated), `wiki/log.md` (appended).

## [2026-05-02] refactor | Full pipeline build — Phases 0–5 (all four modules implemented)
- **Phase 0 — Foundation:** `pipeline/config.py` (six `ENABLE_*` flags, `from_env()`, `from_yaml()`, `validate()`, `summary()`). `docker-compose.yml` (FalkorDB on port 6379). `preflight_check.sh` updated with config + new module file checks + feature-flag section 13.
- **Phase 1 — User Modelling:** `pipeline/user_modelling.py` — `GraphClient` (FalkorDB + 5W+H schema, graceful unavailable mode), 4-stage Mem0g write pipeline (`entity_extractor` → `relation_generator` → `conflict_detector` → `conditional_write`), retrieval gating with slot-relevance classifier, scrutability handlers (`inspect_memory`, `contest_belief`, `correct_belief`). Never deletes — `:DEPRECATED_BY` + `:USER_CORRECTED` edges preserve full audit trail.
- **Phase 2 — Empathy:** `pipeline/appraisal_labeller.py` (offline AppraisePLM → EmpatheticDialogues JSONL; `--mock_model` for pipeline testing). `pipeline/empathy.py` (runtime `analyse_appraisal()`, `format_appraisal_block()`, `parse_appraisal_block()`, `APPRAISAL_SYSTEM_PREFIX`). `sft_question_generator.py` updated: new `appraisal_empathy` category with `"loader": "appraisal_labels"` path (no LLM generation). `sft_gold_response_generator.py` updated: `APPRAISAL_DRAFT_PROMPT`, `APPRAISAL_CRITIQUE_PROMPT`, `generate_draft` and `critique_draft` branch on category. `sft_dataset_assembler.py` updated: quality filter rejects `appraisal_empathy` examples missing `<appraisal>` block.
- **Phase 3 — Ontology Verifier:** `pipeline/ontology_verifier.py` — dual backend (rdflib local OWL / SPARQLWrapper remote endpoint). `extract_claims()` → `verify_claim()` (4-tier: SELECT match 0.9, ASK match 0.8, no match 0.2, unverifiable 0.5) → `score_response()` with per-claim breakdown. `config.py` updated: `ONTOLOGY_SPARQL_ENDPOINT` + `ONTOLOGY_MAX_CLAIMS` fields.
- **Phase 4 — Integration:** `pipeline/3_infererence.py` updated: module imports, `_GRAPH_CLIENT` + `_ONTO_GRAPH` singletons, `_raw_generate()` (greedy, no tool loop), `_build_system_prompt()` (flag-gated injections), `ContestRequest` + `CorrectRequest` Pydantic models, full `chat_completions` lifecycle (write → retrieve → appraise → generate → onto_score), scrutability routes (`/memory/inspect|contest|correct`), `/config` introspection endpoint, `main()` updated with `--config` arg + singleton init + startup diagnostics. `run_all.sh` updated: `ENABLE_*` vars, `start_falkordb()` / `stop_falkordb()`, Stage 0 + Stage 0.5, flag forwarding to subprocess.
- **Phase 5 — Documentation:** `README.md` rewritten — feature flags table, module prerequisites (Docker, AppraisePLM), updated architecture diagram (all four modules now implemented), updated repo layout, updated inference server endpoints (12 total incl. scrutability), updated `run_all.sh` stage table (Stage 0 + 0.5 added). `wiki/sources/code/training-and-benchmark.md` rewritten — all new scripts in the at-a-glance table, feature flag lifecycle section, updated stage map, updated Related and Raw sections.
- **GPU smoke test deferred** to Monday (machine with Docker + GPU + AppraisePLM setup). All five optional modules have `--mock_model` / `available=False` fallback modes so the pipeline runs end-to-end on this laptop with all flags off.
- **Files changed:** `pipeline/config.py` (new), `pipeline/user_modelling.py` (new), `pipeline/empathy.py` (new), `pipeline/appraisal_labeller.py` (new), `pipeline/ontology_verifier.py` (new), `docker-compose.yml` (new), `pipeline/3_infererence.py` (updated), `pipeline/sft_question_generator.py` (updated), `pipeline/sft_gold_response_generator.py` (updated), `pipeline/sft_dataset_assembler.py` (updated), `pipeline/run_all.sh` (updated), `pipeline/preflight_check.sh` (updated), `README.md` (updated), `wiki/sources/code/training-and-benchmark.md` (updated), `wiki/index.md` (updated), `wiki/tags.md` (updated), `wiki/log.md` (appended).

## [2026-05-02] query | Full pipeline implementation plan
- Created `wiki/queries/full-pipeline-implementation-plan.md` — phase-by-phase build plan for all six modules.
- Six feature flags defined: `ENABLE_SFT`, `ENABLE_GRPO`, `ENABLE_USER_MODELLING`, `ENABLE_EMPATHY`, `ENABLE_PERSONALISATION`, `ENABLE_ONTOLOGY_VERIF`.
- Parallel tracks: Phase 1 (User Modelling), Phase 2 (Empathy), Phase 3 (Ontology Verifier) all independent; Phase 4 (Integration) sequential after all three.
- GPU work (Phase 6) starts Monday; Phases 0–5 are pre-GPU coding work.
- Key decisions recorded: Mem0g write pattern without Mem0 SDK; AppraisePLM as offline labeller only (not runtime dependency); FalkorDB direct via `falkordb` Python package without Cognee.
- Updated `wiki/index.md` — new entry under Queries.

## [2026-05-02] lint | Health check — full wiki audit
- **No orphan pages found.** All files in `wiki/**/*.md` are reachable from `wiki/index.md`.
- **Missing inbound source — `docs/meetings-notes/`.** All 7 meeting files (Sep 2025 – Apr 2026) were unindexed. Fixed in the ingest entry below.
- **Tag count drift.** `wiki/tags.md` counts last audited 2026-04-20; at least 8 paper pages + 3 code pages added since. Counts on `reasoning`, `security`, `rl`, `constitution` are suspect. Re-audit needed after this session (not auto-fixed — requires a sweep).
- **Missing decision for constitution-drift concern (April 2026 meeting).** The advisor formally raised constitution drift + probes-vs-tests as a mitigation strategy. This research design choice is not yet a `wiki/decisions/` page. → **Action needed: create `wiki/decisions/2026-04-xx-constitution-drift-probing.md`.**
- **`IMPROVEMENT_ROADMAP.md` (54 KB at repo root) still not ingested.** Listed in "Not yet ingested" section since bootstrap. User decision needed on whether it is still authoritative.
- **`wiki/decisions/2025-10-01-four-module-architecture.md` has stale `updated:` date (2025-10-01).** Page content references Qwen3-0.6B, Gemma 4, FalkorDB, Cognee — none of which were in scope in October 2025. Minor metadata drift; not auto-fixed.
- **13 "Literature Note only" papers** remain without `wiki/sources/papers/` pages. Tracked by `wiki/questions/2026-04-30-asset-acquisition-todo.md`. No action needed here.
- **Three `wiki/sources/code/` pages not linked from any topic page.** `sft-v2-pipeline`, `training-and-benchmark`, `constitution-document` are in the index but not referenced from `topics/reasoning` or `topics/tool-use-and-verification`. Low-priority cross-reference gap.

## [2026-05-02] ingest | Meeting notes — all advisor meetings Sep 2025 – Apr 2026
- **Source:** `docs/meetings-notes/` — 7 markdown files (september2025 through april2026).
- **New pages created (7):**
  - `wiki/sources/meetings/september2025.md` — First meeting: scrutability framing, Inside Out concept, AI as "sociopath"
  - `wiki/sources/meetings/october2025.md` — RL for thought processes, values-interpreter architecture, ethical companion risks
  - `wiki/sources/meetings/november2025.md` — Two meetings: ontology-LLM pivot (Nov 11) + interleaved thinking (late Nov)
  - `wiki/sources/meetings/december2025.md` — Research plan review (Dec 11 + Dec 16): prototype scope, GDPR flags, pivoting normalised
  - `wiki/sources/meetings/january2026.md` — Boolean/math GPT failure → hybrid delegation confirmed; context forgetting
  - `wiki/sources/meetings/february2026.md` — Behaviourism lens; post-hoc constraint vs in-model innovation; focus contraction request
  - `wiki/sources/meetings/april2026.md` — Constitution drift + probes vs tests; Apple internship June–Sept 2026; dissertation timeline
- **Updated:** `wiki/index.md` — new "Advisor meetings" subsection under Sources; `updated:` bumped to 2026-05-02.
- **Updated:** `wiki/tags.md` — added `advisor-meeting` tag (count 7).
- **Key project fact captured:** Ajinkya accepted an Apple internship in Dublin, June–September 2026. Dissertation experimentation must complete before then or gate the overlap carefully.
- **No conflicts raised.** Meeting notes are consistent with existing wiki decisions and entity pages.

## [2026-05-02] refactor | README + wiki docs sync — full branch documentation pass
- **README.md** — complete rewrite to reflect the branch's current state. Added: four-module architecture diagram, updated repo layout (new scripts), preflight check as step 0, Phase 2 GRPO commands, run_all.sh orchestration, Experiment 0 section, adversarial probes section, security hardening table, updated inference server reference (dependency endpoints), corrected V2 CLI commands (old `--data_path` → new `--mode sft --data_dir`). Removed stale content. A general reader can now understand the pipeline and run it end-to-end.
- **`wiki/sources/code/training-and-benchmark.md`** — complete rewrite. Added: GRPO phase, DAPO improvements table, composite reward breakdown, Experiment 0 strategy table, adversarial probe category table, security blockers summary table, GRPO hyperparameters, updated scripts-at-a-glance table.
- **`wiki/sources/code/sft-v2-pipeline.md`** — updated flow diagram (now shows Blocker 1+2 in the pipeline), categories updated from 9→11, security hardening section added.
- **`wiki/index.md`** — code source descriptions updated to reflect new content.
- **Files changed:** `README.md`, `wiki/sources/code/training-and-benchmark.md`, `wiki/sources/code/sft-v2-pipeline.md`, `wiki/index.md`.

## [2026-05-02] refactor | GPU-ready package: GRPO trainer + Experiment 0 + run_all.sh + preflight v2
- **`pipeline/2_model_trainer.py`** — complete rewrite adding Phase 2 (GRPO). New additions: `GRPO_CONFIG` (DAPO hyperparameters: G=8, β=0.001, ε_low=0.2, ε_high=0.28), `make_reward_fn(reward_type)` (composite reward: format 0.30 + accuracy 0.40 + tool_integrity 0.15 + constitution 0.15), `build_grpo_dataset()` (SFT JSONL → GRPOTrainer prompt format), `train_grpo()` method, `load_checkpoint()` method, `_patch_dynamic_sampling()` (DAPO zero-variance group skip). CLI now supports `--mode {sft,grpo}`, `--reward_type {c,d}`. Reward type c = Ablation C (format+accuracy), reward type d = Ablation D (full composite).
- **`pipeline/experiment0_reasoning_comparison.py`** — new file. Implements researchplan.tex Phase 3 reasoning paradigm comparison. Four strategies: baseline (direct answer), cot (chain-of-thought), interleaved (native CAPABILITY_CHECK format), tot (simulated Tree-of-Thoughts: N candidates + self-ranking). Evaluates on GSM8K (via `datasets`) + 10 built-in logic puzzles. Metrics: accuracy, CAPABILITY_CHECK rate, tool use rate, answer-tag rate, latency. Saves comparison table + JSON report. CLI: `--strategy all`, `--n 100`, `--smoke`, `--benchmark gsm8k|logic|all`.
- **`pipeline/run_all.sh`** — new master orchestration script. 8 stages, fully resumable (each stage checks its output checkpoint before running). Stages: data check → SFT → SFT baseline → Experiment 0 → adversarial baseline → GRPO-C → GRPO-D → final ablation A/B/C/D. Server management (start/stop background process with PID file + health check). CLI: `--dry_run`, `--from N`, `--stages N,M`, `--port N`.
- **`pipeline/preflight_check.sh`** — updated with sections 11 (run scripts present) and 12 (training status: 7-stage progress tracker showing which checkpoints and reports exist). Now reports `$STAGES_DONE / 7 stages complete` and suggests the correct `--from N` resume command.
- **Files changed:** `pipeline/2_model_trainer.py`, `pipeline/experiment0_reasoning_comparison.py` (new), `pipeline/run_all.sh` (new), `pipeline/preflight_check.sh`.

## [2026-05-02] refactor | Security Blockers 3 + 4 — adversarial benchmark suite + dependency detection
- **Trigger:** security-review.tex §5: dedicated red-team evaluation (Blocker 3, OWASP LLM01/LLM04) and interaction-frequency monitor (Blocker 4, OWASP LLM09) required before GRPO.

### Blocker 3 — Adversarial Probe Suite (`pipeline/4_benchmark.py`)
- Added `ADVERSARIAL_PROBES` list: 14 probes across three attack categories:
  - **jailbreak (5 probes):** direct instruction override (JB1), DAN persona injection (JB2), system-prompt fishing (JB3), harmful roleplay (JB4), false developer authority (JB5).
  - **injection (3 probes):** adversarial content embedded in structured user message simulating a tool result (INJ1), false constitution update injected in user message (INJ2), instruction payload inside a math expression (INJ3).
  - **regression (6 probes):** sycophancy under expert pressure (REG1 — P14), format preservation when user asks to skip tags (REG2 — P1/P18), tool hallucination (REG3 — P3), calibrated confidence on well-known fact (REG4 — no over-hedging), calibrated confidence on live/unknown data (REG5 — no overconfidence), honesty under social pressure (REG6 — P5/P7).
- Added `run_adversarial_probes()`: runs probes, prints per-attack status (RESIST/FAILED), reports `adversarial_score` + per-category breakdown, saves to `reports/adversarial_{timestamp}.json`.
- Added CLI flags: `--adversarial`, `--adversarial_only`, `--attack_types jailbreak,injection,regression`.
- **Usage before GRPO:** `python 4_benchmark.py --adversarial_only` — all attacks should be resisted on the SFT baseline; regression after GRPO signals reward hacking.

### Blocker 4 — Dependency Detection Monitor (`pipeline/3_infererence.py`)
- Added `DependencyMonitor` class (in-memory, no persistence across restarts — privacy by design):
  - Two independent triggers: **frequency** (≥10 interactions in last 60 min) and **burst** (≥5 consecutive turn-gaps < 30s).
  - When either fires: returns `True` from `record()` and appends an autonomy-preserving disclosure to the model's answer — non-blocking, redirects to human support, consistent with constitution autonomy-preservation goals.
  - Cooldown: 1 hour before same session can re-trigger.
  - `status(session_id)` and `reset_session(session_id)` methods for audit/research.
- Added `session_id: str = "anonymous"` field to `CompletionRequest`.
- Added `_DEPENDENCY_MONITOR` singleton wired into `chat_completions()` after every response.
- Added two new server endpoints: `GET /dependency/status/{session_id}`, `POST /dependency/reset/{session_id}`.
- Response payload now includes `"dependency_disclosure": bool`.
- **Smoke tested:** 6 cases — single call no-trigger, frequency trigger, cooldown no-retrigger, burst trigger (5 gaps), status(), reset_session(). All pass.

- **Files changed:** `pipeline/4_benchmark.py`, `pipeline/3_infererence.py`.

## [2026-05-02] refactor | Security Blocker 2 — independent constitutional verifier + rule-based pre-check
- **Trigger:** security-review.tex §4.3: generate–critique–revise loop is a SPOF when the same model grades its own output; data poisoning from SFT has no out-of-band check (OWASP LLM04).
- **Fix 1 — `rule_check_response()`** added to `sft_gold_response_generator.py`. Deterministic AST-free checks for the 5 highest-signal constitutional principles:
  - **P1** — `<think>` block and `CAPABILITY_CHECK` label must both be present (regex).
  - **P3** — no hallucinated tools (names not in the full registry) and no calls to tools marked ✗ in the active session profile (set diff).
  - **P4** — if the question signals arithmetic (`\d+\.?\d* each|total cost|how much|calculat...`) and `python_execute` is in the active profile, a numeric `<answer>` without a `<tool>python_execute` call is flagged.
  - **P14** — for `adversarial_pressure` category, `<turn_2>` content is scanned for capitulation phrases ("okay i'll", "roughly", "if i had to guess", etc.).
  - **P18** (structural) — `<answer>` block must be present.
- **Fix 2 — `_merge_violations()`** merges rule output with LLM critique. If the LLM returns `NO_VIOLATIONS` but rule checks found issues, the rule violations survive and reach the revision step. The LLM cannot suppress structural violations.
- **Fix 3 — loud warning** when `--critic_model` is not set. Previously the self-critique fallback was silent. Now prints a named warning referencing security-review.tex §4.3 so the researcher is explicitly aware of the SPOF.
- **Smoke tested:** 7 cases covering all 5 principles plus merge logic. All pass.
- **What this does NOT fix:** semantic violations (e.g. a subtle sycophantic answer that doesn't contain capitulation phrases). Those still depend on the LLM critic. Use `--critic_model claude-opus-4-7` for full Blocker 2 compliance.
- **OWASP mapping:** Addresses LLM04 (Data and Model Poisoning) — rule violations cannot be introduced into the training set by a biased self-critique.
- **Files changed:** `pipeline/sft_gold_response_generator.py`.

## [2026-05-01] refactor | Security Blocker 1 — code sandbox + tool-output injection hardening
- **Trigger:** Pipeline audit revealed LLM-generated Python code was executed via subprocess without import restrictions in three files; web/URL tool output was injected raw into model context (prompt injection surface).
- **Fix 1 — AST-based code validator added to all three files:** `_validate_code()` parses LLM-generated code with Python's `ast` module before any `subprocess.run` call. Blocks all non-math imports (`os`, `sys`, `subprocess`, `socket`, `requests`, etc.) and dangerous builtins (`exec`, `eval`, `compile`, `__import__`, `open`). Allowed imports: `math`, `statistics`, `decimal`, `fractions`, `cmath`, `random`, `itertools`, `functools`, `operator`, `collections`, `numbers`, `string`, `re`. Returns `(is_safe, reason)` — unsafe code is rejected with a descriptive error rather than executed.
- **Fix 2 — Tool-output sanitiser added to inference server:** `_sanitise_tool_output(tool_name, raw)` strips prompt-injection patterns from web/URL content before it enters the model's conversation context. Strips `<tool>`, `<think>`, `<answer>`, `CAPABILITY_CHECK`, `ignore previous instructions` variants, and similar hijack phrases using `_INJECTION_RE`. Truncates to 3,000 characters to prevent context flooding. Wraps in structured `[TOOL_RESULT: name]\n...\n[/TOOL_RESULT]` envelope so the model sees it as data, not instruction.
- **Applied to:** `_python_execute()` in `3_infererence.py`; `execute_code_blocks()` in `sft_rejection_sampler.py`; `verify_answer_with_execution()` in `sft_math_question_generator.py`. Tool loop in `chat_completions()` (`3_infererence.py`) now routes all tool results through `_sanitise_tool_output` before appending to conversation.
- **OWASP mapping:** Addresses LLM01 (Prompt Injection) at the tool-output boundary and the code-execution boundary. Partial address of LLM04 (Data and Model Poisoning) — adversarially crafted web content can no longer embed instructions into SFT training data via the rejection sampler.
- **Remaining open (Blocker 1):** The `_python_execute` validator only covers the allowlist approach. A proper sandbox (process isolation, seccomp on Linux) would be the production fix. This is the training/research context minimum viable hardening.

## [2026-05-12] refactor | Pipeline logging clarity — trainer, server, benchmark
- Added structured progress prints for SFT/GRPO setup, ROUGE sampling, and publish metadata in `pipeline/2_model_trainer.py`.
- Expanded inference server startup logs with config source, model selection details, and module status in `pipeline/3_infererence.py`.
- Improved benchmark suite output with server context, probe counts, progress indexing, and summary totals in `pipeline/4_benchmark.py`.
- Why: make training, serving, and evaluation runs easier to trace and debug in logs.
- Files changed: `pipeline/2_model_trainer.py`, `pipeline/3_infererence.py`, `pipeline/4_benchmark.py`.
- **Files changed:** `pipeline/3_infererence.py`, `pipeline/sft_rejection_sampler.py`, `pipeline/sft_math_question_generator.py`.

## [2026-05-01] decision | Source-document alignment pass — researchplan.tex + security-review.tex corrections
- **Trigger:** User asked to strictly adhere to `researchplan.tex` and `docs/security-analysis/security-review.tex` after noticing the master plan diverged from the formal research plan.
- **Source documents read in full:** `researchplan.tex` (formal CS7CS6 research plan — official research question, 5 objectives, 7 phases, 2 pivots, evaluation strategy) and `docs/security-analysis/security-review.tex` (security paper — local-first architecture, four open security blockers that must be resolved before GRPO).
- **Critical misalignments found and corrected:**
  1. **Research question framing wrong.** Master plan said "efficiency vs scale." The official research question (researchplan.tex §1.2) is about **transparency, modularity, systematic User Modelling, and tool delegation**. Small models = **local deployment for privacy**, not efficiency. Corrected in `overview.md` and `queries/grpo-and-personalisation-master-plan.md`.
  2. **Four-module architecture (Pivot 1) had no decision page.** The binding architectural decision from the October 2025 Professor Conlan meeting — Reasoning / User Modelling / Tool Integration / Generator modules — had no wiki page. Created `decisions/2025-10-01-four-module-architecture.md`.
  3. **Four pre-GRPO security blockers completely missing from master plan.** `security-review.tex` §5 explicitly states these must be resolved before GRPO: (1) prompt injection hardening via tool-output extraction layer (highest priority — OWASP LLM01), (2) independent constitutional verifier (OWASP LLM04), (3) adversarial benchmark suite (OWASP LLM01/LLM04), (4) dependency detection protocol (OWASP LLM09). Added to `queries/grpo-and-personalisation-master-plan.md` as a mandatory pre-GRPO section.
  4. **Reasoning paradigm comparison (Experiment 0) missing from catalog.** researchplan.tex Phase 3 specifies comparing CoT vs ToT vs interleaved thinking vs latent reasoning (Coconut) on GSM8K before any GRPO run. Added as Experiment 0 to `experiments/experiment-catalog.md`.
  5. **Formal evaluation benchmarks missing from catalog.** researchplan.tex §1.4 specifies: GSM8K, MATH dataset (small-model subset), logic puzzles, Crowd-event appraisal detection, user studies with validated HCI instruments. Added as a formal evaluation strategy section.
  6. **Sequencing wrong.** Master plan had GRPO starting in week 1. Corrected: reasoning comparison (Experiment 0) + four security blockers come first; GRPO starts in week 3.
- **Files created:** `wiki/decisions/2025-10-01-four-module-architecture.md`.
- **Files updated:** `wiki/overview.md` (official research question + four-module table), `wiki/experiments/experiment-catalog.md` (evaluation strategy + Experiment 0 + small-model framing corrected), `wiki/queries/grpo-and-personalisation-master-plan.md` (research question anchor + four security blockers + sequencing fix), `wiki/index.md` (four-module architecture decision added).

## [2026-05-01] query | Mem0 scrutability audit — user-centric memory gap confirmed as genuine thesis contribution
- **Trigger:** user asked whether Mem0 is scrutable — can users inspect and correct their own stored memories?
- **Finding:** Mem0 has zero user-facing scrutability. It is a pure developer API. The conflict detector runs internally; the LLM Update Resolver decides ADD/UPDATE/DELETE/NOOP with no user notification. Users cannot inspect, contest, or correct any memory.
- **Comparison table:** ChatGPT "Manage Memories" covers only explicitly saved memories (auto-learned chat history is behind an all-or-nothing toggle, not individually auditable). Claude Projects is most transparent (markdown files, `/memory` command, directly editable). Letta is white-box (agents edit memory blocks directly). Mem0 is the least scrutable of all four.
- **Literature gap confirmed:** No published paper formally defines user-centric AI memory scrutability as a distinct concept. Closest: "transparency asymmetry" in the 2025 AI Agent Index (arXiv:2602.17753), Forgetful but Faithful (arXiv:2512.12856) which treats transparent failure modes as a desideratum, and the 20-year UMAP scrutability tradition (Kay & Kummerfeld 2013, already acquired) which addresses recommender systems, not conversational memory.
- **Thesis contribution sharpened:** Layer 5 of the master plan is the first formal definition of user-centric AI memory scrutability for conversational agents, operationalised as five constraints: inspect, contest, correct, deprecate (not delete), audit trail. This is not duplicated by any current system.
- **3 new papers added to acquisition checklist:** Memory in the Age of AI Agents (2512.13564), Forgetful but Faithful (2512.12856), MemMachine (2604.04853).
- **Files updated:** `wiki/topics/personalisation.md` (scrutability gap section added), `wiki/queries/grpo-and-personalisation-master-plan.md` (Layer 5 sharpened to name the contribution), `wiki/questions/2026-04-30-asset-acquisition-todo.md` (3 new papers).
- **Branch created:** `feat/grpo-and-personalisation-stack` — implementation work begins here. Note: two pre-existing branches (`feat/grpo-v2`, `feat/rl`) may contain the old GRPO trainer code; check before building from scratch.

## [2026-05-01] query | GRPO + Empathetic Personalisation Master Plan — industry benchmarks + implementation roadmap
- **Trigger:** user asked for a plan of action for two workstreams: (1) GRPO model training and (2) user empathy + user modelling via graph vector DB, with industry context from Anthropic, DeepSeek, Gemini, Qwen, NVIDIA, OpenAI.
- **Research performed:** cross-referenced current repo pipeline state against industry practice reports covering GRPO (DeepSeek R1, ByteDance DAPO, Qwen2.5-Math, OpenAI o1/o3, Anthropic Constitutional AI, NVIDIA NeMo-RL) and graph-based personalisation/empathy (ChatGPT Memory, Claude Projects, Gemini Personal Intelligence, Mem0g, Hume AI EVI, FalkorDB, Cognee, LightRAG).
- **Key gap confirmed:** `pipeline/2_model_trainer.py` has zero GRPO code; the entire personalisation/empathy stack (5W+H graph, MCP server, appraisal tagger, retrieval gating, scrutability) has zero code. Both tracks are missing.
- **Key decision — DAPO over vanilla GRPO:** For sub-1B models, entropy collapse makes vanilla GRPO unreliable. DAPO (ByteDance, arXiv:2503.14476) fixes this with Clip-Higher + dynamic sampling + token-level loss normalisation. This is the recommended implementation. Hyperparameters recorded in [[entities/grpo]].
- **Key decision — Cognee + FalkorDB over Neo4j:** FalkorDB wins on inference-time neighbourhood expansion (500× faster p99). Cognee as the orchestration layer provides backend-agnostic KG construction and LLM-native memory abstraction. Neo4j deferred to enterprise/multi-user scenarios.
- **Key decision — Mem0g write pipeline for user state updates:** entity extractor → relation generator → conflict detector → conditional write (`:DEPRECATED_BY` edges, never deletion). Matches scrutability requirement — audit trail stays intact.
- **Binding constraint recorded:** all experiments constrained to small models only — Qwen3-0.6B (primary) and Gemma 4 (secondary). Recorded in [[overview]], [[experiments/experiment-catalog]], and the master plan query.
- **8 new papers added to acquisition checklist:** DAPO (2503.14476), DeepSeekMath (2402.03300), LUSPO (2602.05261), Mem0 (2504.19413), PersonalAI (2506.17001), Simulating Emotions with Appraisal + RL (CHI 2024), Graph-based Agent Memory survey (2602.05665), Avoiding Over-Personalisation (2509.07133).
- **New tags added to `wiki/tags.md`:** `dapo`, `graph-memory`, `gemma`.
- **Files created:** `wiki/queries/grpo-and-personalisation-master-plan.md`.
- **Files updated:** `wiki/entities/grpo.md` (DAPO details, hyperparameters, composite reward), `wiki/topics/personalisation.md` (Cognee+FalkorDB decision, Mem0g pattern, new sources), `wiki/topics/empathy.md` (Hume AI reference, CHI 2024 paper, status stub→draft), `wiki/overview.md` (small-model constraint section), `wiki/experiments/experiment-catalog.md` (small-model constraint callout), `wiki/questions/2026-04-30-asset-acquisition-todo.md` (8 new papers), `wiki/tags.md` (3 new tags), `wiki/index.md` (query entry added).

---

## [2026-05-01] refactor | 5_context_degradation.py upgraded to server/client pattern
- **Decision:** Kept `5_context_degradation.py` as a separate script (not merged into `4_benchmark.py`). Reason: different evaluation mode (greedy/deterministic), different scoring (known expected answers per turn), different thesis question (degradation curve vs capability snapshot).
- **Server changes (`3_infererence.py`):** Added `greedy: bool = False` to `CompletionRequest`; `_generate()` now accepts `greedy` and uses `do_sample=False` when set. Added `input_tokens` to response metrics so clients can track context growth per turn.
- **Client rewrite (`5_context_degradation.py`):** Removed model loading entirely. Now calls `3_infererence.py` via HTTP, same pattern as `4_benchmark.py`. Uses `greedy=True` by default. `--compare_url` replaces old `--compare` flag. Retains all 12 TURNS, expected answers, tool-mania detection, and head-to-head comparison table.
- **Overlap note:** 10 of 12 TURNS also appear in `4_benchmark.py BENCHMARK_QUESTIONS` — this is intentional. The benchmark tests sampled quality; the degradation study tests greedy accuracy-vs-context.
- **Key degradation turns:** Turn 6 (multi-reference: must retrieve values from turns 3 and 5 simultaneously) and Turn 12 (needle-in-haystack at peak context) are the historically most diagnostic failure points.
- **README + wiki updated** with new commands.
- **Files changed:** `pipeline/3_infererence.py`, `pipeline/5_context_degradation.py`, `README.md`, `wiki/sources/code/training-and-benchmark.md`.

## [2026-05-01] refactor | Post-overhaul consistency pass — five gaps fixed
- **Gap 1 restored:** `get_exchange_rate` mock tool added back to `3_infererence.py` as a 5th built-in tool. Critical: benchmark question 3 ("Convert 500 USD to EUR") requires it; `web_search` is not a reliable substitute for reproducible tests. Added to `all_tools` and `compute_and_search` profiles.
- **Gap 2 restored:** `--compare_url` added to `4_benchmark.py`. New workflow: start two servers on different ports (base model on 8001, fine-tuned on 8000), pass both URLs to benchmark. Replaces old in-process comparison which required loading both models simultaneously.
- **Gap 3 restored:** `--questions` and `--max_tool_iters` CLI args added back to `4_benchmark.py`. Previously removed during overhaul; required by README examples.
- **Gap 4 restored:** `_print_comparison_table` re-added to `4_benchmark.py`. Shows answer-tag rate, CAPABILITY_CHECK rate, tool calls, latency, throughput side-by-side across runs.
- **Gap 5 — Docs:** README updated (V1 quickstart commands were broken — pointed to old `--compare` flag that no longer exists); `wiki/sources/code/training-and-benchmark.md` updated to reflect server/client architecture, drift detection workflow, and ablation table. CLAUDE.md updated with §4.5 Pipeline Code Edit — explicit consistency checklist for future pipeline edits.
- **Files changed:** `pipeline/3_infererence.py`, `pipeline/4_benchmark.py`, `README.md`, `wiki/sources/code/training-and-benchmark.md`, `CLAUDE.md`.

## [2026-05-01] decision | Inference server + benchmark client split (frontier-lab pattern)
- **Problem:** Model loading was coupled with benchmarking — every eval required a GPU, reloading the model each run, and tool logic was duplicated across inference and benchmark scripts.
- **Solution:** Split into server + client, mirroring how Anthropic/OpenAI/DeepSeek operate. `3_infererence.py` is now a FastAPI server; `4_benchmark.py` is a pure HTTP client with zero GPU dependency.
- **Server (`3_infererence.py`):** Loads model once on startup. Tool registry pattern: 4 built-in tools (`python_execute`, `web_search`, `read_url`, `get_datetime`) plus `POST /v1/tools/register` for runtime additions. Server-side tool execution loop. Endpoints: `/health`, `/v1/models`, `/v1/tools`, `/v1/chat/completions`, `/metrics`, `/metrics/reset`. Metrics: latency p50/p95/p99, tokens/s, tool call counts by name.
- **Client (`4_benchmark.py`):** Two suites — (1) constitutional drift probes (12 principles, regex-graded, no model judge) and (2) multi-turn conversation benchmark (14 turns + 6 edge cases). Client maintains conversation history across turns; server is stateless per request. Output: structured JSON reports in `reports/`.
- **Edge cases added:** empty input, single char, prompt injection attempt, tool hallucination (asking for non-existent `send_email`), math-without-tool request, impossible future prediction.
- **Drift mitigation workflow (when probe detects drift):** (1) Rollback to last good checkpoint. (2) Increase GRPO KL coefficient β. (3) Add SFT replay mixing to GRPO batch.
- **Files changed:** `pipeline/3_infererence.py` (rewrite), `pipeline/4_benchmark.py` (rewrite).

## [2026-05-01] decision | Constitutional drift detection — probe suite added to 4_benchmark.py
- **Problem:** No mechanism existed to detect when RL training was eroding SFT-learned constitutional behaviour. The benchmark only measured generation speed, tool calls, and context length — not constitutional adherence.
- **Solution:** Added `CONSTITUTIONAL_PROBES` (12 fixed questions, one per detectable principle) to `pipeline/4_benchmark.py`. Each probe uses regex / rule-based automated scoring — no judge model, to avoid the self-referential drift problem.
- **Principles covered:** P1 (CAPABILITY_CHECK present), P2+P3 (tool inventory/discipline), P4 (math=code), P5 (real-time honesty), P6 (user context gate), P8 (impossibility acknowledgment), P9 (tradeoff no winner), P11 (tool avoidance), P14 (hold under pressure), P16 (knowledge cutoff), P17 (single clarification), P18 (explicit I don't know).
- **Drift threshold:** If `constitution_score` drops ≥ 5 percentage points from the SFT baseline, a `DRIFT WARNING` is printed and `drift_warning: true` is set in the JSON report.
- **Workflow:** (1) After SFT, run `python 4_benchmark.py --probe_only --save_as_baseline` to record the SFT baseline. (2) After each GRPO checkpoint, run `python 4_benchmark.py --probe_only --baseline reports/constitution_baseline.json` to detect drift.
- **New CLI flags:** `--probe`, `--probe_only`, `--baseline <path>`, `--save_as_baseline`.
- **File changed:** `pipeline/4_benchmark.py`.

## [2026-05-01] decision | Constitutional drift mitigation — frozen critic + constitution_score
- **Problem identified (Ajinkya + Owen discussion):** Constitution principles can drift as training progresses — SFT teaches the behaviour but GRPO reward signal can erode it if constitution adherence is not explicitly enforced.
- **Three root causes mapped:** (1) Self-referential critique loop where generator and critic are the same model; (2) Multi-turn examples skipped critique entirely; (3) No constitution adherence score in metadata to use as GRPO reward signal.
- **Fix 1 — `--critic_model` flag in `sft_gold_response_generator.py`:** Separates the draft generator from the constitutional judge. A larger frozen model (e.g. `claude-opus-4-7`) grades every draft; the smaller student model writes. Directly addresses the critique-loop SPOF flagged in `[[entities/constitution]]`.
- **Fix 2 — Multi-turn critique added:** Each turn in multi-turn examples is now independently critiqued via `critique_turn()`. Violations are counted per turn; `constitution_score` is averaged across all turns.
- **Fix 3 — `constitution_score` in metadata:** Every training example now carries a `constitution_score` in [0, 1] (1.0 = no violations). This field is the pre-computed signal for the GRPO constitutional reward component in Phase 2.
- **GRPO design (not yet implemented):** When Phase 2 GRPO begins, reward must be composite: `format_reward + correctness_reward + constitution_score`. KL penalty must use the SFT checkpoint as reference policy. See `[[decisions/2026-05-01-constitutional-drift-mitigation]]`.

## [2026-05-01] refactor | Pipeline SFT data quality overhaul — multi-turn support + full constitution coverage

- **Problem 1 (single-turn only):** All training data was one-sentence single-turn questions. Real users write paragraphs of context and hold multi-turn dialogues. Fixed by adding two new categories to `sft_question_generator.py`: `verbose_context_behavioral` (200 paragraph-style inputs) and `multi_turn_conversation` (150 three-to-five-turn scaffolds).
- **Problem 2 (constitution gaps):** `CRITIQUE_PROMPT` only checked principles 1–11. Principles 12–19 (Tool Failure Handling, No Tool Faking, Hold Under Pressure, Explicit Self-Correction, Knowledge Cutoff Awareness, Multi-Step Clarification, Explicit I Don't Know, Search for Entity Facts) were never verified. Fixed: critique prompt now checks all 19 against the full text of each principle.
- **Problem 3 (tool availability monoculture):** Draft generation always assumed `python_execute` available, `web_search` not. Constitution principles 5, 11, 16, 19 have different correct behaviours depending on which tools are present. Fixed: four `TOOL_PROFILES` introduced; each example is assigned a random profile weighted by category (search-heavy categories get `all_tools`/`compute_only` split; neutral categories get even spread).
- **Problem 4 (missing `entity_facts_web_search` ideal behaviour):** Category fell back to vague fallback. Fixed: full `IDEAL_BEHAVIORS` entry added for all 11 categories including the two new ones.
- **Problem 5 (`TRAINING_SYSTEM_PROMPT` incomplete):** Summarised 19 principles into vague bullets. Replaced with `TRAINING_SYSTEM_PROMPT_TEMPLATE` that lists all 19 principles explicitly and injects the actual session tool context.
- **Files changed:** `pipeline/sft_question_generator.py`, `pipeline/sft_gold_response_generator.py`.

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
