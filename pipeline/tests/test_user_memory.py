import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_read_empty_user_returns_all_sections():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        result = store.read("user_001")
    assert "USER MEMORY" in result
    for section in ("WHO", "WHAT", "WHERE", "WHY", "HOW", "FACTS", "CONSTRAINTS"):
        assert section in result


def test_update_persists_value():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_001", "who", "Senior ML engineer at a fintech startup")
        result = store.read("user_001")
    assert "Senior ML engineer" in result


def test_update_creates_file_on_disk():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_abc", "facts", "User prefers metric units")
        assert (Path(tmp) / "user_abc.json").exists()


def test_update_survives_reload():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store1 = UserMemoryStore(store_dir=Path(tmp))
        store1.update("user_001", "what", "Working on fine-tuning sub-1B models")
        store2 = UserMemoryStore(store_dir=Path(tmp))
        result = store2.read("user_001")
    assert "fine-tuning" in result


def test_update_invalid_section_returns_error():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        result = store.update("user_001", "illegal_key", "content")
    assert "Error" in result
    assert "illegal_key" in result


def test_read_with_prompt_surfaces_relevant_section():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_001", "who", "ML researcher working on language models and NLP")
        store.update("user_001", "where", "Based in Dublin, Ireland, EU jurisdiction applies")
        store.update("user_001", "constraints", "Limited GPU budget, 40 hours per week available")
        result = store.read("user_001", prompt="GPU compute budget for training language models")
    assert "ML researcher" in result or "GPU budget" in result or "language model" in result


def test_read_with_prompt_always_includes_who():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_001", "who", "Postgraduate student at Trinity College Dublin")
        result = store.read("user_001", prompt="weather forecast")
    assert "Postgraduate student" in result


def test_relevance_score_full_overlap():
    from user_memory import _relevance_score
    score = _relevance_score("python machine learning language model", "python machine learning")
    assert score > 0.8


def test_relevance_score_empty_content():
    from user_memory import _relevance_score
    assert _relevance_score("", "python") == 0.0


def test_relevance_score_empty_section():
    from user_memory import _relevance_score
    assert _relevance_score("(empty — some description)", "python") == 0.0


def test_read_returns_top_three_sections():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        for section, content in [
            ("who", "Python developer with 5 years experience"),
            ("what", "Building a recommendation system"),
            ("where", "Remote worker in Berlin"),
            ("why", "Career growth and interesting problems"),
            ("how", "Prefers concise code examples"),
            ("facts", "Uses Linux, vim, prefers Python 3.11+"),
            ("constraints", "No cloud budget, runs locally only"),
        ]:
            store.update("user_001", section, content)
        result = store.read("user_001", prompt="python code examples for developers")
    assert result.count("[") >= 3


def test_update_overwrites_existing_value():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_001", "who", "Junior developer")
        store.update("user_001", "who", "Senior developer with 10 years experience")
        result = store.read("user_001")
    assert "Senior developer" in result
    assert "Junior developer" not in result
