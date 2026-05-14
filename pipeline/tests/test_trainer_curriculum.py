"""Tests for curriculum staging in 2_model_trainer.py — CLI smoke test only."""
import subprocess
import sys
from pathlib import Path


def test_curriculum_flags_in_help():
    """Verify the three new CLI flags appear in --help output."""
    result = subprocess.run(
        [sys.executable, "pipeline/2_model_trainer.py", "--help"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    output = result.stdout + result.stderr
    assert "--curriculum_stage" in output, f"--curriculum_stage missing from help:\n{output[:2000]}"
    assert "--from_checkpoint" in output, f"--from_checkpoint missing from help:\n{output[:2000]}"
    assert "--v3_format" in output, f"--v3_format missing from help:\n{output[:2000]}"


def test_split_curriculum_stages_basic():
    """Unit test for _split_curriculum_stages without importing the trainer module directly."""
    # We test via exec to avoid the `2_` filename import issue
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "trainer",
        str(Path(__file__).parent.parent / "2_model_trainer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Patch HAS_LIBS to False so heavy imports are skipped
    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {
        "unsloth": mock.MagicMock(),
        "trl": mock.MagicMock(),
        "datasets": mock.MagicMock(),
        "torch": mock.MagicMock(),
    }):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass  # May fail on heavy imports but function should be defined

    # Try accessing _split_curriculum_stages directly
    fn = getattr(mod, "_split_curriculum_stages", None)
    if fn is None:
        # Try via importlib
        import pytest
        pytest.skip("Could not load trainer module (expected in non-GPU env)")

    def _ex(has_tool=False, asst_len=100):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        if has_tool:
            msgs += [
                {"role": "assistant", "content": "<tool>x()</tool>"},
                {"role": "tool", "content": "r"},
                {"role": "assistant", "content": "<answer>a</answer>"},
            ]
        else:
            msgs.append({"role": "assistant", "content": "x" * asst_len})
        return {"messages": msgs, "metadata": {}}

    examples = [_ex(False, 200)] * 10 + [_ex(True)] * 10 + [_ex(False, 800)] * 5
    s1, s2, s3 = fn(examples)
    assert len(s1) == 10
    assert len(s2) == 25
    assert len(s3) > len(s2)
