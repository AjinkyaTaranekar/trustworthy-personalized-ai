# Plan: Metrics + dissertation graphs for the ablation ladder

> Status: **Phase 1 BUILT & validated (2026-06-21).** `experiment_metrics.py`, `experiment_figures.py` (9 figures), `analyze_experiments.py --figures`, and a `5_judgement_day.py` truncation fix are in. Validated on all five conditions (judge-free; T=0.7, n=3). **Phase 2 (judge-based) pending §4 run.** Remaining judge/analysis improvements listed at the bottom.

## Context
The latest committed benchmark run gives a clean 4-condition ladder (vanilla_base → vanilla_tools → sft_template → sft_constitution; Thinker–Executor still to come). Manual inspection surfaced findings the current tooling does **not** capture:

- `analyze_experiments.py` consolidates aggregate suite scores + adjacent-rung deltas (bootstrap CIs) but has **no depth/answer-quality or tool-behaviour metrics and emits no figures**.
- `export_assets.py` renders figures/LaTeX but is hardwired to a **2-condition (vanilla vs SFT) flat-glob**, so it cannot draw the 5-rung ladder or read the per-label `reports/<label>/` subdirs.
- The signature finding — reasoning **relocates** from `<think>` (91% empty in sft_constitution) into the answer body + tools — and the constitution model's tool-trigger-happiness (172 calls) are invisible to both scripts.

Goal: a judge-free metric + figure layer over the existing reports that (a) quantifies depth/relocation, tool behaviour, compliance and drift across all rungs, and (b) renders dissertation-ready vector figures. Judge-based quality (persona, win-rate) is deferred to Phase 2 because persona is currently unjudged (`personas_judged=0`).

**Decisions:** Phase 1 = all four judge-free figure families now; judge-based metrics later.

## Architecture
Single source of truth for metrics, reused by both the numeric ladder and the figures — no logic duplicated between tables and graphs.

- **NEW `pipeline/experiment_metrics.py`** — pure, GPU-free extractor functions over any `4_benchmark.py` report. Returns per-condition aggregates **and** per-item vectors keyed `group::q_idx` (so the existing `bootstrap_delta` works unchanged). Reuses `principle_families.py` for family grouping; sources the decoy-tool list from `4_benchmark.py` (do not hardcode).
- **EDIT `pipeline/analyze_experiments.py`** — register the new judge-free metrics as ladder rows via the existing `SUITE_METRICS` pattern (they flow into the printed ladder, `experiment_ladder_*.csv/.tex`, and bootstrap deltas for free). Add a `--figures` flag that calls the new figures module. Reuse its subdir-aware `find_report` and `bootstrap_delta`.
- **NEW `pipeline/experiment_figures.py`** — matplotlib renderers (guarded import + `matplotlib.use("Agg")`, like `export_assets.py:350-354`). Writes vector PDFs to `reports/dissertation_assets/` (same OUTDIR as `export_assets.py`) using **distinct `fig_ladder_*` names** to avoid clobbering export_assets' `fig_per_family.pdf`/`fig_think_collapse.pdf`. Runnable standalone or via `analyze_experiments.py --figures`.
- `export_assets.py` left untouched (keeps existing 2-condition methodology assets working).

## Phase 1 — metric catalogue (judge-free, on existing data)

**A. Compliance & capability** (mostly present; ensure per-condition + per-family + deltas)
- constitution rule score (overall + per family via `principle_families.py`)
- category coverage (overall + per category from `scores_by_category`)
- adversarial (overall + per attack type: injection/jailbreak/regression)
- drift (overall + `adherence_curve` per turn, `first_drift_at`)

**B. Reasoning depth & relocation** (NEW — the original contribution)
- `think_length` mean/median; `think_empty` rate
- answer length (chars/words) from `answer_content`
- **reasoning-externalisation ratio** = `think_chars / (think_chars + answer_chars)` per response → the relocation signal (low = reasoning moved to the answer body)
- clarification rate (proxy: `response_type=='ask'` / non-empty `ask_content` — ties to P21)
- depth-vs-cost = total reasoning length vs `metrics.latency_s` / `tokens_generated`
- *Honesty note in code + caption:* these are **structural proxies**; semantic depth needs the Phase-2 judge.

**C. Tool behaviour** (NEW)
- tool calls / response; distinct tools used
- tool-failure rate (`metrics.tool_failures` / calls)
- **decoy-bait rate** (calls to `send_email`/`write`/`check_exchange_rate`/`check_fact` — benchmark traps)
- over-use proxy (P11 score vs tool calls on `no_tools` probes); latency/token cost

## Phase 1 — figure catalogue (metric → graph)
All to `reports/dissertation_assets/`, vector PDF, colourblind-safe palette with TCD-blue accent, print-legible fonts, British-spelling single-line captions.

1. **Compliance ladder + deltas**
   - `fig_ladder_compliance.pdf` — grouped bars, conditions × {constitution-rule, category, adversarial, drift}
   - `fig_ladder_per_family.pdf` — grouped bars, conditions × 5 principle families
   - `fig_ladder_deltas.pdf` — forest plot of adjacent-rung deltas with bootstrap 95% CI
2. **Reasoning depth & relocation**
   - `fig_think_distribution.pdf` — box/violin of `think_length` per condition (the collapse) + %-empty annotation
   - `fig_reasoning_location.pdf` — stacked bar: share of reasoning in `<think>` vs answer body (headline relocation figure)
   - `fig_depth_vs_cost.pdf` — scatter, reasoning length vs latency, coloured by condition
3. **Tool behaviour & failures**
   - `fig_tool_usage.pdf` — grouped bars per condition: calls/response, failure rate, decoy-bait rate
4. **Drift over turns + category coverage**
   - `fig_drift_curve.pdf` — adherence vs turn, one line/condition, vertical marker at `first_drift_at`
   - `fig_category_heatmap.pdf` — condition × category heatmap of math capability

LaTeX tables (depth + tool rows) come for free via the extended `analyze_experiments.py` CSV/`.tex` output.

## Phase 2 — judge-based (deferred; after running the judge)
No new infra, just run + add figures:
- Run `5_judgement_day.py` (fills `llm_score`/`combined_score` + persona dimensions) and `compare_report.py --judge` (head-to-head win-rate leaderboard).
- New figures: persona radar (6 dims, condition overlays), win-rate leaderboard bar, rule-vs-judge agreement scatter + Cohen's κ, and render the 6×6 persona-dimension correlation (already computed in `analyze_experiments.py`) as a redundancy heatmap.

## Judge & analysis script improvements (review)

From reading `5_judgement_day.py` and `analyze_experiments.py` against the data:

**Done now (cheap, high-value, judge-free):**
- *New metric* `hollow_pass_rate` — rule passes but answer body empty (catches "passed P1 via a long `<think>` but delivered nothing"). Currently 0 on the re-run vanillas (they now answer), but it is the right validity guard.
- *New metric* `externalisation_ratio` + `clarification_rate` — quantify the two real rule-vs-quality gaps (reasoning moved to the answer; only C4 asks). These ARE the meaningful figures.
- *Judge fix* — response truncation 2000→4000 chars so relocated reasoning isn't cut before scoring.
- *Decoy detection* sourced from the live tool registry (can't drift).

**Proposed (do during write-up / Phase 2):**
1. **Report rule and judge separately as the primary view; demote `combined` (50/50 blend).** The rule probe rewards form (long `<think>`) and penalises substance (relocated reasoning, `<ask>`), so averaging it 50/50 with the more reliable judge drags the judge down. `analyze_experiments.py` already keeps the rows separate — lead with `constitution (judge)` and `constitution (rule)`, treat `combined` as secondary.
2. **Rule-vs-judge agreement (Cohen's κ) per condition** — a validity figure/number once §4 runs; shows *where* the deterministic probe and the judge disagree (expected: the reasoning family in C3/C4). Strong methodological evidence that the judge is needed.
3. **Fix the P21 `<ask>` rule probe in `4_benchmark.py`** — it scores a correct clarifying question as 0, i.e. it punishes the exact behaviour P21 is meant to reward. Until fixed, footnote it; the judge will partially correct it at the combined level if the P21 rubric credits asking.
4. **Double-escaped-code leak** in `sft_template` P4 tool calls (`"2 + 2 = 4\""`) — the `_repair_double_escaped_code` path doesn't catch non-`python_execute` calls; tighten or note.
5. **Judge robustness:** persona transcript truncation (9000) and per-item judge retries/confidence are untracked — fine for now, revisit if judge-failure counts are high.

## Constraints / caveats to bake in
- **Sampling:** run used temperature 0.7, n=3/principle → structural metrics are single samples; report bootstrap CIs and recommend a greedy re-run before freezing final dissertation numbers. Treat current as directional.
- **matplotlib** must be present in the analysis env (guarded; skips with a message if absent) — confirm/install.
- **No filename clashes** with `export_assets.py` outputs (use `fig_ladder_*` prefix).
- **Decoy tool list** sourced from `4_benchmark.py`, not duplicated.
- Missing conditions (thinker_executor not yet run, persona unjudged) must degrade gracefully — `find_report` already returns `None`; figures skip absent series.

## Files
- NEW `pipeline/experiment_metrics.py`
- NEW `pipeline/experiment_figures.py`
- EDIT `pipeline/analyze_experiments.py` (metric rows + `--figures`)
- EDIT `pipeline/ABLATION_LADDER_RUNBOOK.md` (document the analysis+figures step) and `wiki/sources/code/training-and-benchmark.md` (CLAUDE.md §4.5 pipeline-edit sync) + append `wiki/log.md`
- Phase 2: run (no edit) `5_judgement_day.py`, `compare_report.py`; extend `experiment_figures.py`

## Verification
1. Dry-run on existing data (omit thinker_executor + persona):
   `python analyze_experiments.py --labels vanilla_base vanilla_tools sft_template sft_constitution --suites constitution categories drift adversarial --figures`
2. Confirm: new depth/tool rows in the printed ladder + `experiment_ladder_*.csv`; PDFs land in `reports/dissertation_assets/`; no crash with thinker_executor/persona absent.
3. Sanity-check against the manual analysis: think 1354→150 chars & 91% empty; tool calls 0/1/18/172; constitution 0.51/0.57/0.42/0.58; drift adherence collapse after turn ~5.
4. Eyeball each PDF for print legibility (font size, colourblind palette).
5. When Thinker–Executor lands: append `thinker_executor` to `--labels`; figures auto-include the 5th rung. Then run Phase 2 judge + figures.
