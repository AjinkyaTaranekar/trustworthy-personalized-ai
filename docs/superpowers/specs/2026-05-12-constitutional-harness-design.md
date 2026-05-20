# Constitutional Harness — Design Spec
**Date:** 2026-05-12
**Branch:** feat/grpo-and-personalisation-stack
**Status:** Approved for implementation

---

## Problem

Constitutional compliance is enforced at training time (via `rule_check_response()` in `sft_gold_response_generator.py`) but not at inference time. Once the model is deployed, there is no mechanism to catch or correct violations before the response reaches the user. The Meta-Harness paper (Lee et al., 2026) proposes an inference-time orchestration layer that validates outputs and steers on failure — this spec adapts that concept to the constitutional AI pipeline.

---

## What We Are Building

An inference-time constitutional validation and correction loop — `ConstitutionalHarness` — that:

1. Intercepts every final response from the generation loop
2. Runs deterministic rule-based checks for key constitutional principles
3. On violation: injects a corrective prompt and retries generation (up to 2 times)
4. Tracks per-principle failure rates and adaptively reinforces weak principles in the system prompt
5. Exposes violation data in the response envelope and via a metrics endpoint
6. Enables `4_benchmark.py` to run probes with and without harness for side-by-side comparison

---

## Architecture

```
User query
    → system prompt (base + optional harness adaptation suffix)
    → _generate() + tool loop  (existing, unchanged)
    → final response text
         ↓
  ConstitutionalHarness.check_and_steer()
  ├── run_checks(response, question, tool_profile) → violations list
  │     No violations  → pass through
  │     Violations     → build_corrective_prompt(violations)
  │                      → append corrective user turn to conv
  │                      → re-run _generate() [up to 2 retries]
  │                      → re-check after each retry
  └── HarnessMetrics.record(violations)
       → update reports/harness_metrics.json
       → if principle fail_rate > 0.30 (rolling 50 requests)
            → return adaptation_suffix → prepended to next system prompt
                ↓
  Response envelope:
    response, harness_violations, harness_retries, ...existing fields
```

---

## Components

### 1. `pipeline/constitutional_harness.py` (new file)

**`run_checks(response, question, tool_profile_label) → list[str]`**

Deterministic checks ported and adapted from `sft_gold_response_generator.rule_check_response()`. Checks run at inference time:

| Principle | Check | Signal |
|---|---|---|
| P1 DECOMPOSE FIRST | `<think>` block present + `CAPABILITY_CHECK` label inside | Regex |
| P3 TOOL DISCIPLINE | No hallucinated tools; no unavailable tools called | Set diff against active profile |
| P4 MATH = CODE | Numeric answer in `<answer>` without `python_execute` when tool available | Regex + tool list |
| P18 ANSWER PRESENT | `<answer>` block exists | Regex |

P14 (adversarial capitulation) is excluded: it requires multi-turn `<turn_2>` tagging that does not apply to live inference. P20–P23 are excluded: they require semantic understanding beyond deterministic regex.

Returns a list of violation strings in `PRINCIPLE_N: description` format, or empty list if compliant.

**`build_corrective_prompt(violations: list[str]) → str`**

Formats violations into a targeted corrective user turn:
```
[HARNESS] Your previous response had constitutional violations that must be corrected before I can show it to the user:
- PRINCIPLE_1: <think> block present but CAPABILITY_CHECK label is missing.
- PRINCIPLE_18: <answer> block is absent.
Please rewrite your response fully, fixing each violation listed above.
```

**`class HarnessMetrics`**

Tracks per-principle pass/fail over a rolling window of 50 requests.

- `record(violations: list[str])` — updates in-memory counters
- `get_adaptation_suffix() → str` — returns a bolded reminder block for any principle with fail_rate > 0.30. Empty string if all principles healthy. This is the **meta-adaptation** mechanism: the harness learns which principles the current model is weakest on and dynamically reinforces them in the live system prompt.
- `save(path)` — writes `reports/harness_metrics.json` with per-principle rates + rolling window state
- `load(path)` — restores state across server restarts

Metrics JSON schema:
```json
{
  "window_size": 50,
  "request_count": 142,
  "principles": {
    "P1": {"checks": 142, "failures": 12, "fail_rate": 0.085},
    "P3": {"checks": 142, "failures": 3,  "fail_rate": 0.021},
    "P4": {"checks": 89,  "failures": 31, "fail_rate": 0.348},
    "P18": {"checks": 142, "failures": 5, "fail_rate": 0.035}
  },
  "adaptation_active": ["P4"],
  "total_retries": 38,
  "retry_success_rate": 0.71
}
```

### 2. `pipeline/3_infererence.py` (changes)

**Imports (guarded):**
```python
try:
    from constitutional_harness import ConstitutionalHarness
    _harness_available = True
except ImportError as _e:
    _harness_available = False
    print(f"[INFO] constitutional_harness not importable ({_e}) — ENABLE_HARNESS disabled")
```

**Startup:** Instantiate `ConstitutionalHarness()` when `cfg.ENABLE_HARNESS=true` and `_harness_available`.

**`CompletionRequest` model — new field:**
```python
harness_enabled: Optional[bool] = None
# If not None, overrides cfg.ENABLE_HARNESS for this request.
# Allows 4_benchmark.py to toggle harness per-request without restarting server.
```

**`chat_completions` endpoint — harness section (after tool loop, before return):**
```python
harness_violations = []
harness_retries = 0
effective_harness = (
    req.harness_enabled if req.harness_enabled is not None else cfg.ENABLE_HARNESS
)
if effective_harness and _HARNESS is not None:
    response, harness_violations, harness_retries = _HARNESS.check_and_steer(
        response=response,
        conv=conv,
        question=user_turn,
        tool_profile_label=req.tool_profile,
        generate_fn=lambda c: _generate(c, req.max_new_tokens, req.temperature, req.greedy)[0],
        max_retries=2,
    )
```

**Response envelope additions:**
```python
"harness_violations": harness_violations,  # [] when no violations or harness off
"harness_retries":    harness_retries,      # 0 when no retries needed
```

Both fields are always present (empty/zero when harness off) so clients need no version check.

**New endpoint:**
```
GET /harness/metrics   → HarnessMetrics JSON (404 if harness not enabled)
POST /harness/reset    → reset rolling counters
```

**System prompt adaptation:** In `_build_system_prompt()`, append `_HARNESS.metrics.get_adaptation_suffix()` when harness is active and suffix is non-empty.

### 3. `pipeline/config.py` (change)

Add:
```python
ENABLE_HARNESS: bool = Field(default=False, description="Inference-time constitutional validation and correction loop")
```

### 4. `pipeline/4_benchmark.py` (changes)

**New flag:** `--with_harness` (boolean, default false)

When `--with_harness` is set, `run_constitution_probes()` runs twice:
- Pass 1: all probes with `harness_enabled=False` → baseline scores
- Pass 2: all probes with `harness_enabled=True` → harness scores

Saves `constitution_probe_harness_comparison_{timestamp}.json`:
```json
{
  "timestamp": "...",
  "without_harness": { "constitution_score": 0.72, "scores_by_principle": {...} },
  "with_harness":    { "constitution_score": 0.89, "scores_by_principle": {...} },
  "delta":           { "constitution_score": +0.17, "scores_by_principle": {...} },
  "harness_stats":   { "total_retries": 5, "retry_success_rate": 0.80, "principles_triggered": ["P1", "P4"] }
}
```

Print a diff table to stdout showing per-principle improvement.

**Usage:**
```bash
python pipeline/4_benchmark.py --probe_only --with_harness
```

---

## Feature Flag Behaviour

| `PIPELINE_ENABLE_HARNESS` | `harness_enabled` in request | Effective behaviour |
|---|---|---|
| `false` (default) | not set | No harness — identical to current behaviour |
| `false` | `true` | Harness runs for this request only |
| `true` | not set | Harness runs for all requests |
| `true` | `false` | Harness skipped for this request (benchmark baseline pass) |

---

## Logging / Print Statements

All harness and tool activity must be visible in the server terminal so training runs and benchmark sessions can be diagnosed without attaching a debugger. Follow the existing pipeline pattern (`[TAG] message`).

### Harness prints (in `constitutional_harness.py` and `3_infererence.py`)

```
[HARNESS] Checking response for constitutional violations...
[HARNESS] ✓ No violations — response passed (P1, P3, P4, P18)
```
```
[HARNESS] ✗ Violations found (2):
[HARNESS]   · PRINCIPLE_1: <think> block present but CAPABILITY_CHECK label missing
[HARNESS]   · PRINCIPLE_18: <answer> block absent
[HARNESS] Injecting corrective prompt → retry 1/2...
[HARNESS] ✓ Retry 1 passed — violations cleared
```
```
[HARNESS] ✗ Retry 2 still violated (P1) — returning best response with violation flags
```
```
[HARNESS] Adaptation active: P4 fail_rate=0.38 — reinforcing in system prompt
```
```
[HARNESS] Metrics saved → reports/harness_metrics.json (142 requests, 38 retries)
```

### Tool prints (in `3_infererence.py` — existing tool loop, fill gaps)

These should already exist but must be consistent. Required lines:

```
[TOOL] Calling: python_execute(code='print(2+2)')
[TOOL] Result (45 chars): 4\n
[TOOL] Error: tool 'web_search' not available in profile 'compute_only'
[TOOL] Error: tool 'fly_to_moon' is not registered on this server
[TOOL] Execution error in python_execute: NameError: name 'x' is not defined
```

Print format: `[TOOL] <verb>: <detail>` — never silently swallow tool outcomes.

### Benchmark harness-comparison prints (in `4_benchmark.py`)

```
  [HARNESS COMPARISON] Running probes without harness...
  [HARNESS COMPARISON] Without harness: constitution_score=0.72 (9/12 passed)
  [HARNESS COMPARISON] Running probes with harness...
  [HARNESS COMPARISON] With harness:    constitution_score=0.89 (11/12 passed)  [+0.17]
  [HARNESS COMPARISON] Retries triggered: 5  |  Retry success rate: 80.0%
  [HARNESS COMPARISON] Per-principle delta:
    P1_decompose_first      : FAIL → PASS  (+1)
    P4_math_code            : FAIL → PASS  (+1)
    P2P3_tool_discipline    : PASS → PASS  ( 0)
    ...
```

---

## Files Changed

| File | Change type |
|---|---|
| `pipeline/constitutional_harness.py` | New |
| `pipeline/3_infererence.py` | Modified — harness hook, new endpoint, CompletionRequest field |
| `pipeline/config.py` | Modified — ENABLE_HARNESS flag |
| `pipeline/4_benchmark.py` | Modified — --with_harness flag, comparison report |
| `README.md` | Modified — new flag, new endpoints, benchmarking usage |

---

## What This Proves (Dissertation Evidence)

1. **Inference-time constitutional enforcement** is feasible without retraining and without an LLM judge — purely deterministic, auditable, explainable.
2. **Meta-adaptation**: the harness learns which principles the current model is weakest on from live traffic and dynamically reinforces them — no human intervention required.
3. **Measurable improvement**: the `--with_harness` benchmark comparison directly quantifies the harness's contribution to constitutional compliance, giving a clean ablation result (Condition B: SFT only vs Condition B + Harness).
4. Closes the training-inference gap: principles enforced during data generation (`sft_gold_response_generator.py`) are now also enforced at serving time.
