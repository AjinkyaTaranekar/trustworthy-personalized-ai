"""Unit tests for constitutional_harness.py — run with: pytest pipeline/tests/test_constitutional_harness.py -v"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from constitutional_harness import run_checks, build_corrective_prompt, HarnessMetrics, ConstitutionalHarness


# ── Fixtures ────────────────────────────────────────────────────────────────

PROFILE_COMPUTE = "compute_only"
PROFILE_NO_TOOLS = "no_tools"

GOOD_RESPONSE = (
    "<think>CAPABILITY_CHECK: I have python_execute available.\n"
    "The user wants 7+3. I will use python_execute.\n</think>"
    "<tool>python_execute(code='print(7+3)')</tool>"
    "<answer>The answer is 10.</answer>"
)

BAD_NO_THINK = "The answer is 10."
BAD_NO_CAPCHECK = "<think>I will compute this.</think><answer>10.</answer>"
BAD_NO_ANSWER = "<think>CAPABILITY_CHECK: ok.</think>I think it is 10."
BAD_HALLUCINATED_TOOL = (
    "<think>CAPABILITY_CHECK: ok.</think>"
    "<tool>fly_to_moon(destination='Mars')</tool>"
    "<answer>Done.</answer>"
)
BAD_UNAVAILABLE_TOOL = (
    "<think>CAPABILITY_CHECK: web_search not available.</think>"
    "<tool>web_search(query='weather')</tool>"
    "<answer>Here is the weather.</answer>"
)
BAD_MATH_NO_CODE = (
    "<think>CAPABILITY_CHECK: python_execute available.</think>"
    "<answer>7823 multiplied by 349 minus 1024 is 2,730,603.</answer>"
)

MATH_QUESTION = "What is 7823 multiplied by 349, then subtract 1024?"
PLAIN_QUESTION = "What is the capital of France?"


# ── run_checks ───────────────────────────────────────────────────────────────

def test_run_checks_clean_response():
    violations = run_checks(GOOD_RESPONSE, MATH_QUESTION, PROFILE_COMPUTE)
    assert violations == [], f"Expected no violations, got: {violations}"

def test_run_checks_missing_think_block():
    violations = run_checks(BAD_NO_THINK, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_1" in v for v in violations)
    assert any("think" in v.lower() for v in violations)

def test_run_checks_missing_capability_check():
    violations = run_checks(BAD_NO_CAPCHECK, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_1" in v for v in violations)
    assert any("CAPABILITY_CHECK" in v for v in violations)

def test_run_checks_missing_answer_block():
    violations = run_checks(BAD_NO_ANSWER, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_18" in v for v in violations)

def test_run_checks_hallucinated_tool():
    violations = run_checks(BAD_HALLUCINATED_TOOL, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_3" in v for v in violations)
    assert any("fly_to_moon" in v for v in violations)

def test_run_checks_unavailable_tool():
    violations = run_checks(BAD_UNAVAILABLE_TOOL, PLAIN_QUESTION, PROFILE_NO_TOOLS)
    assert any("PRINCIPLE_3" in v for v in violations)
    assert any("web_search" in v for v in violations)

def test_run_checks_math_no_code():
    violations = run_checks(BAD_MATH_NO_CODE, MATH_QUESTION, PROFILE_COMPUTE)
    assert any("PRINCIPLE_4" in v for v in violations)

def test_run_checks_math_no_code_skipped_when_no_tool():
    violations = run_checks(BAD_MATH_NO_CODE, MATH_QUESTION, PROFILE_NO_TOOLS)
    assert not any("PRINCIPLE_4" in v for v in violations)

def test_run_checks_multiple_violations():
    violations = run_checks(BAD_NO_THINK, MATH_QUESTION, PROFILE_COMPUTE)
    assert len(violations) >= 2


# ── build_corrective_prompt ──────────────────────────────────────────────────

def test_build_corrective_prompt_contains_violations():
    violations = ["PRINCIPLE_1: missing CAPABILITY_CHECK", "PRINCIPLE_18: no answer block"]
    prompt = build_corrective_prompt(violations)
    assert "PRINCIPLE_1" in prompt
    assert "PRINCIPLE_18" in prompt
    assert "[HARNESS]" in prompt

def test_build_corrective_prompt_empty():
    prompt = build_corrective_prompt([])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# ── HarnessMetrics ───────────────────────────────────────────────────────────

def test_harness_metrics_record_violations():
    m = HarnessMetrics()
    m.record(["PRINCIPLE_1: missing capcheck"])
    snap = m.snapshot()
    assert snap["principles"]["P1"]["failures"] == 1
    assert snap["principles"]["P1"]["checks"] == 1

def test_harness_metrics_record_pass():
    m = HarnessMetrics()
    m.record([])
    snap = m.snapshot()
    for p in snap["principles"].values():
        assert p["failures"] == 0
        assert p["checks"] == 1

def test_harness_metrics_adaptation_suffix_empty_when_healthy():
    m = HarnessMetrics()
    for _ in range(10):
        m.record([])
    assert m.get_adaptation_suffix() == ""

def test_harness_metrics_adaptation_suffix_present_when_failing():
    m = HarnessMetrics(window_size=10)
    for _ in range(4):
        m.record(["PRINCIPLE_4: math without code"])
    suffix = m.get_adaptation_suffix()
    assert "P4" in suffix or "MATH" in suffix

def test_harness_metrics_save_load(tmp_path):
    m = HarnessMetrics()
    m.record(["PRINCIPLE_1: missing capcheck"])
    path = tmp_path / "harness_metrics.json"
    m.save(str(path))
    m2 = HarnessMetrics()
    m2.load(str(path))
    snap = m2.snapshot()
    assert snap["principles"]["P1"]["failures"] == 1


# ── ConstitutionalHarness.check_and_steer ───────────────────────────────────

def test_harness_passes_clean_response():
    harness = ConstitutionalHarness()
    calls = []
    def fake_generate(conv, temp_scale=1.0):
        calls.append(conv)
        return GOOD_RESPONSE
    result, violations, retries = harness.check_and_steer(
        response=GOOD_RESPONSE,
        conv=[{"role": "user", "content": MATH_QUESTION}],
        question=MATH_QUESTION,
        tool_profile_label=PROFILE_COMPUTE,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert violations == []
    assert retries == 0
    assert len(calls) == 0

def test_harness_retries_on_violation():
    harness = ConstitutionalHarness()
    calls = []
    def fake_generate(conv, temp_scale=1.0):
        calls.append(conv)
        return GOOD_RESPONSE
    result, violations, retries = harness.check_and_steer(
        response=BAD_NO_ANSWER,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert retries == 1
    assert violations == []
    assert len(calls) == 1

def test_harness_exhausts_retries_returns_best():
    harness = ConstitutionalHarness()
    def always_bad(conv, temp_scale=1.0):
        return BAD_NO_THINK
    result, violations, retries = harness.check_and_steer(
        response=BAD_NO_THINK,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=always_bad,
        max_retries=2,
    )
    assert retries == 2
    assert len(violations) > 0


# ── SSD data flywheel ────────────────────────────────────────────────────────

def test_harness_logs_ssd_candidate_on_first_pass(tmp_path):
    ssd_log = tmp_path / "ssd_candidates.jsonl"
    harness = ConstitutionalHarness(ssd_log_path=str(ssd_log))
    def fake_generate(conv, temp_scale=1.0):
        return GOOD_RESPONSE
    harness.check_and_steer(
        response=GOOD_RESPONSE,
        conv=[{"role": "user", "content": MATH_QUESTION}],
        question=MATH_QUESTION,
        tool_profile_label=PROFILE_COMPUTE,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert ssd_log.exists(), "SSD log should be created on a passing response"
    entries = [json.loads(line) for line in ssd_log.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["question"] == MATH_QUESTION
    assert entries[0]["response"] == GOOD_RESPONSE
    assert entries[0]["retried"] is False

def test_harness_logs_ssd_candidate_on_retry_cleared(tmp_path):
    ssd_log = tmp_path / "ssd_candidates.jsonl"
    harness = ConstitutionalHarness(ssd_log_path=str(ssd_log))
    def fake_generate(conv, temp_scale=1.0):
        return GOOD_RESPONSE
    harness.check_and_steer(
        response=BAD_NO_ANSWER,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert ssd_log.exists()
    entries = [json.loads(line) for line in ssd_log.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["retried"] is True
    assert entries[0]["response"] == GOOD_RESPONSE

def test_harness_does_not_log_ssd_candidate_when_retries_exhausted(tmp_path):
    ssd_log = tmp_path / "ssd_candidates.jsonl"
    harness = ConstitutionalHarness(ssd_log_path=str(ssd_log))
    def always_bad(conv, temp_scale=1.0):
        return BAD_NO_THINK
    harness.check_and_steer(
        response=BAD_NO_THINK,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=always_bad,
        max_retries=2,
    )
    assert not ssd_log.exists(), "SSD log must NOT be written when all retries fail"

def test_harness_ssd_log_contains_required_fields(tmp_path):
    ssd_log = tmp_path / "ssd_candidates.jsonl"
    harness = ConstitutionalHarness(ssd_log_path=str(ssd_log))
    def fake_generate(conv, temp_scale=1.0):
        return GOOD_RESPONSE
    harness.check_and_steer(
        response=GOOD_RESPONSE,
        conv=[{"role": "user", "content": MATH_QUESTION}],
        question=MATH_QUESTION,
        tool_profile_label=PROFILE_COMPUTE,
        generate_fn=fake_generate,
        max_retries=2,
    )
    entry = json.loads(ssd_log.read_text().splitlines()[0])
    for field in ("question", "response", "retried", "principles_checked", "timestamp"):
        assert field in entry, f"SSD log entry missing required field: {field}"

def test_harness_ssd_logging_disabled_when_path_is_none():
    harness = ConstitutionalHarness(ssd_log_path=None)
    def fake_generate(conv, temp_scale=1.0):
        return GOOD_RESPONSE
    result, violations, retries = harness.check_and_steer(
        response=GOOD_RESPONSE,
        conv=[{"role": "user", "content": MATH_QUESTION}],
        question=MATH_QUESTION,
        tool_profile_label=PROFILE_COMPUTE,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert violations == []


# ── Temperature escalation on retry ──────────────────────────────────────────

def test_harness_retry1_passes_elevated_temp_scale():
    harness = ConstitutionalHarness()
    temp_scales_seen = []
    def fake_generate(conv, temp_scale=1.0):
        temp_scales_seen.append(temp_scale)
        return GOOD_RESPONSE
    harness.check_and_steer(
        response=BAD_NO_ANSWER,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=fake_generate,
        max_retries=2,
    )
    assert len(temp_scales_seen) == 1, "Should call generate_fn exactly once on first retry"
    assert temp_scales_seen[0] > 1.0, (
        f"Retry 1 must use temp_scale > 1.0 to explore the distribution; got {temp_scales_seen[0]}"
    )

def test_harness_retry2_uses_higher_temp_scale_than_retry1():
    harness = ConstitutionalHarness()
    temp_scales_seen = []
    attempt = [0]
    def bad_then_good(conv, temp_scale=1.0):
        temp_scales_seen.append(temp_scale)
        attempt[0] += 1
        return BAD_NO_THINK if attempt[0] < 2 else GOOD_RESPONSE
    harness.check_and_steer(
        response=BAD_NO_THINK,
        conv=[{"role": "user", "content": PLAIN_QUESTION}],
        question=PLAIN_QUESTION,
        tool_profile_label=PROFILE_NO_TOOLS,
        generate_fn=bad_then_good,
        max_retries=2,
    )
    assert len(temp_scales_seen) == 2, "Should call generate_fn twice (two retries)"
    assert temp_scales_seen[1] > temp_scales_seen[0], (
        f"Retry 2 temp_scale ({temp_scales_seen[1]}) must exceed retry 1 ({temp_scales_seen[0]})"
    )
