# Final Dataset Generation Pipeline — Design Spec
**Date:** 2026-05-06
**Author:** Ajinkya Taranekar + Claude Code
**Status:** Approved — ready for implementation planning

---

## 1. Scope

Full replacement of two scripts:
- `pipeline/sft_question_generator.py` — generates question bank per category
- `pipeline/sft_gold_response_generator.py` — generates gold responses via teacher model

Both scripts are replaced in-place with no dead-code residue. `pipeline/sft_math_pipeline.py` and `pipeline/sft_dataset_assembler.py` are untouched. `pipeline/run_all.sh` orchestration is unchanged. `pipeline/3_infererence.py` receives targeted updates to the tool registry and system prompt only.

---

## 2. Tool Registry

### 2.1 Final Tool Set (4 tools)

| Tool | Signature | Purpose |
|---|---|---|
| `python_execute` | `python_execute(code: str)` | All computation and arithmetic |
| `web_search` | `web_search(query: str)` | All external data — rates, prices, tax, weather, facts, versions |
| `read_url` | `read_url(url: str, prompt: str = "")` | Follow a specific URL; `prompt` states what to extract so the LLM retains context after reading |
| `get_datetime` | `get_datetime()` | Current UTC datetime — prevents date hallucination |

### 2.2 Removed

`get_exchange_rate` is removed from the tool registry, all TOOL_PROFILES, and the system prompt. It used hardcoded static rates which is misleading. `web_search` covers this use case generically and correctly.

### 2.3 Design Rationale

The model must be a generalist, not a specialist-tool caller. Everything retrievable from the world is accessed via `web_search`. The model learns: *need a fact from the world → web_search; need arithmetic → python_execute; need to read a specific page → read_url; need the current time → get_datetime.*

### 2.4 TOOL_PROFILES

```python
TOOL_PROFILES = {
    "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime"},
    "compute_only":       {"python_execute"},
    "compute_and_search": {"python_execute", "web_search", "read_url"},
    "no_tools":           set(),
}
```

---

## 3. Extended CAPABILITY_CHECK Format

The `<think>` block is the model's mandatory reasoning scaffold. It is extended with three new named sub-sections: **5W+H**, **First Principles**, and **CONSEQUENCE_CHECK**. All sub-sections scale to question complexity — a trivial question gets one short line per sub-section; a complex one gets a full breakdown.

### 3.1 Full Format

```
<think>
CAPABILITY_CHECK:

  5W+H:
    WHO is affected: [the user / third parties / institutions involved]
    WHAT is required: [list of requirements to answer correctly]
    WHEN does this apply: [time-sensitivity — live data needed, training cutoff relevant, dated context]
    WHERE does this apply: [jurisdiction, region, domain, platform]
    WHY are they asking: [inferred intent and underlying goal]
    HOW to approach: [tool selection and method]

  First Principles:
    Core truth: [the irreducible fact(s) this answer rests on]
    Assumptions I am making: [what I am taking for granted — flag if unverified]

  Session tools: [exact inventory as given in system prompt]
  Gap: [what I cannot obtain with available tools or knowledge]
  Strategy: [how I proceed given the gap — tool chain or honest refusal]

  CONSEQUENCE_CHECK:
    Stakes: [low / medium / high + one-line reason]
    If I am wrong: [concrete harm to the user]
    User will likely: [the action the user will take with this answer]
    Accountability: [what I must hedge, verify, or explicitly flag in the answer]
</think>
<answer>
[response to the user — high-stakes answers include an explicit caveat tied to CONSEQUENCE_CHECK]
</answer>
```

### 3.2 Scaling Rules

- **Trivial questions** (stable definitions, capitals, unit conversions): Each sub-section is one line. CONSEQUENCE_CHECK is one line noting stakes are low.
- **Moderate questions** (entity facts, calculations, advice): Full 5W+H, one-sentence First Principles, two-line CONSEQUENCE_CHECK.
- **Complex/high-stakes questions** (financial decisions, medical context, legal context, interleaved tool chains): Full expansion of all sub-sections. CONSEQUENCE_CHECK expands to name the specific failure mode and the explicit caveat that must appear in `<answer>`.

---

## 4. New Constitution Principles (P20–P23)

Four new principles are appended to `pipeline/constitution.md` and added to the critique and revision prompts in the gold response generator.

### P20 — FIRST PRINCIPLES
Before answering any non-trivial question, break it to its irreducible truths. Name the core fact the answer rests on. Name the assumptions being made. If an assumption is unverified, flag it in `<think>` and hedge it in `<answer>`.

### P21 — 5W+H QUESTIONING
Every CAPABILITY_CHECK must explicitly address Who is affected, What is required, When this applies, Where it applies, Why the user is asking, and How to approach it. For simple questions a single line per dimension suffices. Never skip the framework — skipping it is how unexamined assumptions become confident wrong answers.

### P22 — CONSEQUENCE_CHECK
Every response must include a CONSEQUENCE_CHECK inside `<think>`. It must assess:
1. **Stakes** — low / medium / high, and why.
2. **Failure mode** — concrete harm if the answer is wrong.
3. **User action** — what the user will likely do with this answer.
4. **Accountability** — what must be hedged or flagged in `<answer>`. High-stakes answers must surface this caveat explicitly in the answer text, not bury it in `<think>`.

### P23 — INTERLEAVED TOOL CHAINING
When a question requires both external data retrieval AND computation, chain the tool calls. Never stop after one tool if a second tool would make the answer verifiable or precise. The pattern is: web_search to retrieve a value → python_execute to compute on it → optionally web_search again to verify or enrich. Calling only one tool when two are needed is a capability failure, not a conservative choice.

### Updated Rule-Based Checker

The deterministic `rule_check_response` function gains three new structural checks (run before the LLM critic, same PRINCIPLE_N: format):

- **P21 check:** CAPABILITY_CHECK present but `5W+H` label missing → violation.
- **P22 check:** CAPABILITY_CHECK present but `CONSEQUENCE_CHECK` label missing → violation.
- **P23 check:** Category is `interleaved_tool_reasoning` and fewer than two distinct tool calls appear in the response → violation.

---

## 5. Categories

### 5.1 Existing 12 Categories — Updates

All 12 existing categories are updated in three ways:

1. **Ideal behaviour descriptions** are rewritten to explicitly reward 5W+H, First Principles, CONSEQUENCE_CHECK, and interleaved chaining wherever natural.
2. **Example questions** are updated to include at least one chained-tool scenario per category where the domain supports it.
3. **Question generation prompt** is updated to instruct the LLM generator to produce questions that sometimes require chaining (not only single-tool scenarios).

Category-specific chaining additions:

| Category | Chaining opportunity added |
|---|---|
| `real_time_dependent` | "What is the EUR/INR rate today, and how much would €300 cost me?" → search → compute |
| `entity_facts_web_search` | "What Python version is latest, and is my code compatible?" → search version → read changelog URL |
| `knowledge_boundary` | "What did the latest IPCC report say, and how does that compare to the 1.5°C target?" → search → synthesise |
| `user_context_behavioral` | 5W+H makes explicit what context is missing before any tool use |
| `impossible_tasks` | CONSEQUENCE_CHECK formalises why the task is impossible (failure mode is the point) |
| `subjective_tradeoffs` | First Principles — what is the irreducible decision criterion — shapes the tradeoff enumeration |
| `adversarial_pressure` | CONSEQUENCE_CHECK formalises why capitulation is harmful (quantified risk appears in think block) |
| `multi_step_clarification` | 5W+H drives which clarifying question is most critical |
| `ambiguous_underspecified` | First Principles surfaces what is fundamentally unknown |
| `verbose_context_behavioral` | 5W+H organises the rich context the user provided before identifying the gap |
| `multi_turn_conversation` | CONSEQUENCE_CHECK updates each turn as context fills in |
| `appraisal_empathy` | CONSEQUENCE_CHECK flags emotional stakes; First Principles grounds the empathetic response |

### 5.2 New Category: `interleaved_tool_reasoning` (150 examples)

**Description:** Questions that inherently require chaining at least two different tools to answer correctly. A single tool call is insufficient. The model must retrieve external data with `web_search` or `read_url`, then act on that data with `python_execute`, and optionally loop back to search again.

**Ideal behaviour:** Chain the tools. State in CAPABILITY_CHECK that chaining is required. Retrieve first, compute second. Show the extracted value from the search result before passing it to python_execute. If the search returns a range, compute both bounds. Never approximate when the chain is available.

**Example question domains and chain patterns:**

| Domain | Chain pattern |
|---|---|
| Tax / GST calculation | web_search(rate) → python_execute(cost × rate) |
| Foreign currency cost | web_search(exchange rate today) → python_execute(amount × rate) |
| Compound interest with live rate | web_search(current ECB/RBI rate) → python_execute(compound formula) |
| Nutritional calculation | web_search(calories in X) → python_execute(sum across items) |
| Software compatibility | web_search(latest version) → read_url(changelog URL) → reason |
| Live event timing | web_search(event time) → get_datetime() → python_execute(difference) |
| Regulatory compliance | web_search(current regulation) → read_url(official source) → apply to user's situation |
| Investment return | web_search(current index performance) → python_execute(return on principal) |

**Diversity axes:** Same 20 geographic/cultural rotation used by all categories. Tax questions must use local tax systems (GST India, VAT EU, HST Canada, etc.), not default to US.

**Format:** Single-turn. No follow-up pressure or multi-turn scaffold needed — the complexity comes from the tool chain, not the conversation structure.

---

## 6. Changes to `3_infererence.py`

These are targeted, minimal changes — no structural refactor:

1. **Remove** `_get_exchange_rate` function and its `register_tool` call.
2. **Remove** `get_exchange_rate` from all `TOOL_PROFILES` entries.
3. **Update** `_system_prompt_for_profile` to reflect the 4-tool inventory and include the new CAPABILITY_CHECK format (5W+H + First Principles + CONSEQUENCE_CHECK sub-sections).
4. **Update** `rule_check_response` with the three new structural checks (P21, P22, P23) described in §4.
5. **Update** the `TOOL_PROFILES` constant — remove `get_exchange_rate` entry.

### 6.1 `read_url` changes (detail)

**Signature:** `read_url(url: str, prompt: str = "") -> str`

**HTML cleaning — current (broken):** `re.sub(r"<[^>]+>", " ", raw)` strips tag brackets but leaves `<script>` and `<style>` block *content* in the output, polluting the text with JavaScript source and CSS rules.

**HTML cleaning — new (correct):**
1. Strip `<script>...</script>` blocks entirely (content included).
2. Strip `<style>...</style>` blocks entirely (content included).
3. Strip all remaining HTML tags with the existing regex.
4. Collapse whitespace.
5. Truncate to 4000 characters (unchanged).

**Prompt echo — output format:**
```
[TOOL_RESULT: read_url]
Extraction goal: {prompt}

{clean_text}
[/TOOL_RESULT]
```
When `prompt` is empty the "Extraction goal:" line is omitted. The goal line reminds the LLM what it was looking for before it reads the full page text, preventing context drift on long pages.

**Training data implication:** The gold response generator must teach the model to always pass a descriptive `prompt` when calling `read_url`. The ideal-behaviour description for `entity_facts_web_search` and `interleaved_tool_reasoning` categories explicitly requires this.

No changes to `FastAPI` routes, `DependencyMonitor`, `OntologyGraph`, or model loading.

---

## 7. Changes to `pipeline/constitution.md`

Append P20–P23 in the same format as existing principles (correct/wrong examples, one-line summary table entry). Update the summary reference table at the bottom to include all 23 principles.

---

## 8. Data Flow

```
sft_question_generator.py
  → 13 categories (12 updated + 1 new)
  → diversity axis rotation (20 axes)
  → outputs: pipeline/data/questions_partA.jsonl

sft_gold_response_generator.py
  → reads questions_partA.jsonl
  → draft → rule_check (P1–P23) → LLM critique → revise
  → outputs: pipeline/data/train_partA.jsonl
  → format: {messages, metadata} same as before (sft_dataset_assembler.py compatible)
```

---

## 9. Backwards Compatibility

- `sft_dataset_assembler.py` reads `train_partA.jsonl` — the `{messages, metadata}` format is unchanged, so no changes needed there.
- `2_model_trainer.py` reads assembler output — unaffected.
- `4_benchmark.py` uses `TOOL_PROFILES` from `3_infererence.py` — updated profiles are compatible (only an entry removed).
- All CLI flags of the two replaced scripts are preserved. New flags may be added but none removed.

---

## 10. Success Criteria

1. The model, after training on this dataset, produces a `CONSEQUENCE_CHECK` block in every response.
2. The model, given a question like "Calculate GST on caramelised popcorn costing ₹200", calls `web_search` for the rate and then `python_execute` for the arithmetic — without being told to chain.
3. The model's 5W+H in CAPABILITY_CHECK matches the actual question requirements (verified by LLM critique against P21).
4. The rule-based checker catches missing CONSEQUENCE_CHECK and missing 5W+H with zero false negatives on the training set.
5. No `get_exchange_rate` calls appear anywhere in the generated training data.
