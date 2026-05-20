"""Smoke test that new negative-trajectory categories are registered."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sft_question_generator import CATEGORIES

def test_inventory_constraint_registered():
    assert "inventory_constraint" in CATEGORIES
    cat = CATEGORIES["inventory_constraint"]
    assert cat["count"] >= 50
    assert "required_profile" in cat
    assert "constrained_tool" in cat

def test_environment_timeout_registered():
    assert "environment_timeout" in CATEGORIES
    cat = CATEGORIES["environment_timeout"]
    assert cat["count"] >= 50

def test_both_categories_have_examples():
    for name in ("inventory_constraint", "environment_timeout"):
        assert len(CATEGORIES[name]["examples"]) >= 3
