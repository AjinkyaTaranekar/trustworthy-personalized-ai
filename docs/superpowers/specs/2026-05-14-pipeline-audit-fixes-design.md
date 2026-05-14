# Pipeline Audit & Fix Design — 2026-05-14

## Scope

Three files. Approach A: targeted bug fixes + structured tool schemas. No API surface changes.

Files touched: `pipeline/sft_question_generator.py`, `pipeline/sft_v3_generator.py`, `pipeline/3_infererence.py`.

---

## Section 1 — sft_question_generator.py

**Problem:** 12 out of 17 question categories are commented out, leaving only:
- `interleaved_tool_reasoning`
- `scratchpad_decomposition`
- `partial_capability_honest`
- `inventory_constraint`
- `environment_timeout`

**Missing active categories:**
- `user_context_behavioral`
- `real_time_dependent`
- `impossible_tasks`
- `subjective_tradeoffs`
- `adversarial_pressure`
- `knowledge_boundary`
- `multi_step_clarification`
- `ambiguous_underspecified`
- `entity_facts_web_search`
- `verbose_context_behavioral`
- `multi_turn_conversation`
- `appraisal_empathy`

**Fix:** Uncomment all 12 category definitions. No logic changes required — the generator already has full format handling for all these category types (`multi_turn`, `two_turn`, `verbose_single_turn`, `appraisal_labels`).

---

## Section 2 — sft_v3_generator.py

### 2a. LiteLLM reasoning_content fallback

**Problem:** `_call_with_stop()` only reads `resp.choices[0].message.content`. For models that return extended thinking separately (`reasoning_content` field — Claude extended thinking, DeepSeek R1), the `<think>` block is silently lost.

**Fix:** In `_call_with_stop()`, after reading `content`, check:
```python
reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
if reasoning and "<think>" not in content.lower():
    content = f"<think>{reasoning}</think>\n{content}"
```
This is a no-op for Kimi/MiniMax (they embed `<think>` inline) and safe for all other models.

### 2b. Multi-turn format support

**Problem:** `_process_one_v3()` reads `item.get("question", "").strip()` which silently skips `multi_turn_conversation` items that use `{"turns": [...], "format": "multi_turn"}` structure. These items have no `"question"` key, so they're silently skipped with `return "error"`.

**Fix:** Add a `_build_multiturn_messages(item, teacher_system)` helper:
- For `format == "multi_turn"`: the `turns` list is all user messages. Build initial_messages as `[system] + [user=turn[0], assistant="Understood.", user=turn[1], assistant="Understood.", ..., user=turn[-1]]`. The lightweight `"Understood."` placeholder assistant turns provide conversation structure without needing generation — only the final assistant response is generated via `_generate_with_intercept`. The resulting JSONL training example contains the full interleaved conversation.
- For `format == "two_turn"`: generate once for `turn_1` (model should refuse), then append `turn_2` as user, generate once more (model should hold). Both generations go through `_generate_with_intercept`.
- In `_process_one_v3()`: detect format via `item.get("format", "single_turn")` and use the helper when `format in {"multi_turn", "two_turn"}`.
- Extract question from `turns[-1]` (multi_turn) or `turn_1` (two_turn) for logging and dedup tracking.

### 2c. partial_capability_honest ideal behavior

**Problem:** `_IDEAL_BEHAVIORS_V3` has no entry for `partial_capability_honest`, so it gets the generic default that doesn't mention `[YES]`/`[BLOCKED]` decomposition or scratchpad usage.

**Fix:** Add:
```python
"partial_capability_honest": (
    "Use scratchpad_read() first to decompose all sub-tasks. "
    "Tag each task [YES] or [BLOCKED: reason]. "
    "Answer [YES] parts fully and assertively — equal confidence as if unblocked. "
    "For [BLOCKED] parts: name exactly what cannot be done, why (professional expertise / "
    "missing context / unavailable tool / unknowable), and the precise redirect. "
    "Never apply uniform caution to all parts — capability is calibrated, not all-or-nothing."
),
```

### 2d. Structured tool schemas in prompts

**Problem:** `STUDENT_PROMPTS` only names tools; teacher prompt lists them as a pipe-separated string. No parameter names, types, or call examples. Industry-standard distillation prompts include typed schemas.

**Fix for student prompts:** Replace bare tool name lists with structured typed docs per profile:

```
Tools available this session:

python_execute(code: str)
  Execute Python. Only math/stats/stdlib imports. Returns stdout/stderr.
  Example: <tool>python_execute(code='print(2**10)')</tool>

web_search(query: str)
  Search the web for current information. Returns text summary.
  Example: <tool>web_search(query='current EUR/USD rate')</tool>
...
```

Each profile's student prompt includes only the tools available for that profile.

**Fix for teacher prompt:** Replace the `"Session tools available: {tool_profile['context']}"` line in `_make_teacher_prompt()` with the same structured schema block built dynamically from `tool_profile["label"]`.

Add a shared `_build_tool_schema_block(available_tools: set[str]) -> str` helper that both student and teacher prompts call — single source of truth.

### 2e. Scratchpad in teacher tool listing

**Problem:** `_make_teacher_prompt()` doesn't mention `scratchpad_read` / `scratchpad_update` in the tool listing, but `_IDEAL_BEHAVIORS_V3["scratchpad_decomposition"]` instructs the teacher to use them.

**Fix:** Include scratchpad tools in `_build_tool_schema_block()` output always (they are always available). Teacher prompt and student prompt both show them.

---

## Section 3 — 3_infererence.py

### 3a. Thread-unsafe _CURRENT_SESSION_ID

**Problem:** `_CURRENT_SESSION_ID` is a module-level global set at request time in `chat_completions()`. With concurrent FastAPI requests, two requests racing through this path would clobber each other's session ID, causing the wrong scratchpad to be read/written.

**Fix:** Remove `_CURRENT_SESSION_ID` as a module-level global. Bind `session_id` locally at request entry inside `chat_completions()`. Pass it explicitly to scratchpad functions via kwarg injection at dispatch time.

**Concrete approach:** In the tool dispatch block in `chat_completions()`, bind `session_id` in a local variable at request entry. For `fn_name in {"scratchpad_read", "scratchpad_update"}`, inject `session_id` into `tc["kwargs"]` before calling `fn(**tc["kwargs"])`. Update the module-level `_scratchpad_read` and `_scratchpad_update` functions to accept `session_id: str = ""` as a parameter rather than reading the global. Remove the `_CURRENT_SESSION_ID` module global and the `global _CURRENT_SESSION_ID` assignment.

### 3b. Remove debug print

**Problem:** Line 280, `print(data)` is left in `_web_search()`. Prints the raw DuckDuckGo JSON to stdout on every web search at inference time.

**Fix:** Delete that line.

### 3c. Structured tool docs in system prompt

**Problem:** `_system_prompt_for_profile()` generates minimal call-syntax examples (`<tool>python_execute(code='...')</tool>`) with no parameter descriptions.

**Fix:** Replace the `call_lines` block with the same `_build_tool_schema_block()` helper used in Section 2. This ensures training (teacher prompt) and inference (system prompt) tool descriptions are identical — critical for training-serving consistency.

Since `sft_v3_generator.py` and `3_infererence.py` are separate files, the helper will be defined in each separately (not shared via import — avoids cross-file coupling). The function is short (under 30 lines) and can be kept in sync by the tests.

---

## Data flow summary

```
sft_question_generator.py
  → generates questions for ALL 17 categories
  → JSONL: {question, category, format} or {turns, category, format}

sft_v3_generator.py
  → reads JSONL
  → builds teacher prompt with structured tool schemas
  → _generate_with_intercept (captures reasoning_content)
  → multi-turn items: _build_multiturn_messages helper
  → saves with student prompt (structured tool schemas, profile-scoped)

3_infererence.py
  → _system_prompt_for_profile uses same structured tool schemas
  → scratchpad session_id passed per-request (no global)
  → _web_search has no debug print
```

---

## What is NOT changing

- The custom XML `<tool>` format (training-serving consistent, unchanged)
- The stop-sequence intercept loop architecture
- The 5-stage constitutional harness
- The `TOOL_PROFILES` definitions (already consistent across files)
- The `_IDEAL_BEHAVIORS_V3` for the 4 existing categories
- The curriculum learning split, GRPO reward functions, or trainer

---

## Testing

Existing test suites cover:
- `tests/test_sft_v3_generator.py` — 16 tests (all should still pass; add tests for multi-turn helper and reasoning_content fallback)
- `tests/test_constitutional_harness.py` — unchanged
- `tests/test_scratchpad.py` — unchanged (scratchpad store logic unchanged)

New tests to add:
- Multi-turn `_build_multiturn_messages` helper (turns → conversation)
- `reasoning_content` fallback in `_call_with_stop`
- `_build_tool_schema_block` output includes correct tools per profile
