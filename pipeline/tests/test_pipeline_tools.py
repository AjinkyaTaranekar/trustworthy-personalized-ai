import sys
import re
import urllib.request as _urllib_req
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Standalone function tests
# ---------------------------------------------------------------------------

def test_python_execute_valid():
    from pipeline_tools import python_execute
    assert python_execute("print(2 + 2)") == "4"


def test_python_execute_blocked_import():
    from pipeline_tools import python_execute
    result = python_execute("import os; print(os.getcwd())")
    assert "blocked_import" in result


def test_python_execute_blocked_builtin():
    from pipeline_tools import python_execute
    result = python_execute("eval('1+1')")
    assert "blocked_builtin" in result


def test_python_execute_syntax_error():
    from pipeline_tools import python_execute
    result = python_execute("def foo(:\n    pass")
    assert "syntax_error" in result


def test_python_execute_no_output():
    from pipeline_tools import python_execute
    result = python_execute("x = 1 + 1")
    assert "no output" in result.lower()


def test_get_datetime_format():
    from pipeline_tools import get_datetime
    result = get_datetime()
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)


def test_read_url_strips_html(monkeypatch):
    from pipeline_tools import read_url
    html = (
        b"<html><head><style>body{margin:0}</style></head>"
        b"<nav>Menu items go here</nav>"
        b"<body><p>Hello world text here. More content follows about the topic.</p></body>"
        b"<footer>Copyright 2024</footer></html>"
    )

    class FakeResp:
        def read(self, n):
            return html

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(_urllib_req, "urlopen", lambda *a, **kw: FakeResp())
    result = read_url("http://example.com")
    assert "[Fetched: http://example.com]" in result
    assert "<html>" not in result
    assert "<style>" not in result
    assert "Menu items" not in result  # nav stripped


def test_read_url_with_prompt_returns_relevant(monkeypatch):
    from pipeline_tools import read_url
    html = (
        b"<html><body>"
        b"<p>Python is a programming language used for data science and machine learning.  "
        b"It has libraries like numpy and pandas for statistics.</p>"
        b"<p>The weather in Dublin is often rainy and mild throughout the year.  "
        b"Temperatures rarely exceed 20 degrees Celsius in summer.</p>"
        b"</body></html>"
    )

    class FakeResp:
        def read(self, n):
            return html

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(_urllib_req, "urlopen", lambda *a, **kw: FakeResp())
    result = read_url("http://example.com", prompt="python programming libraries")
    assert "Prompt: python programming libraries" in result
    assert "Python" in result


def test_read_url_prompt_scoring():
    from pipeline_tools import _score_paragraphs
    paragraphs = [
        "Python is great for machine learning tasks.",
        "The weather in Dublin is often rainy.",
        "Deep learning requires significant compute resources for training models.",
    ]
    result = _score_paragraphs(paragraphs, "python machine learning")
    assert "Python" in result[0]


def test_web_search_missing_key(monkeypatch):
    from pipeline_tools import web_search
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    result = web_search("test query")
    assert "EXA_API_KEY not set" in result


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

def test_registry_call_python_execute():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    result = reg.call("python_execute", {"code": "print(3 * 7)"}, {"python_execute"})
    assert result == "21"


def test_registry_call_unavailable_tool():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    result = reg.call("python_execute", {"code": "print(1)"}, set())
    assert "not available" in result.lower()


def test_registry_call_unknown_tool():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    result = reg.call("nonexistent_tool", {}, {"nonexistent_tool"})
    assert "not registered" in result.lower()


def test_registry_execute_xml_python():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    result = reg.execute("python_execute(code='print(100)')", {"python_execute"})
    assert result == "100"


def test_registry_execute_xml_datetime():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    result = reg.execute("get_datetime()", {"get_datetime"})
    assert re.match(r"\d{4}-\d{2}-\d{2}", result)


def test_registry_to_openai_schemas_filters_by_profile():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    schemas = reg.to_openai_schemas({"python_execute", "web_search"})
    names = {s["function"]["name"] for s in schemas}
    assert "python_execute" in names
    assert "web_search" in names
    assert "get_datetime" not in names
    assert "read_url" not in names


def test_registry_schemas_list_includes_all_builtins():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    names = {s["name"] for s in reg.schemas_list()}
    expected = {
        "python_execute", "web_search", "read_url", "get_datetime",
        "scratchpad_read", "scratchpad_update",
        "user_memory_read", "user_memory_update",
    }
    assert expected.issubset(names)


def test_registry_register_custom_tool():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    reg.register(
        "echo_tool", "Echo the input.",
        {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        lambda msg="", **_: f"ECHO: {msg}",
    )
    result = reg.call("echo_tool", {"msg": "hello"}, {"echo_tool"})
    assert result == "ECHO: hello"


def test_registry_deregister():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    reg.register(
        "tmp_tool", "", {"type": "object", "properties": {}, "required": []}, lambda: "x"
    )
    reg.deregister("tmp_tool")
    result = reg.call("tmp_tool", {}, {"tmp_tool"})
    assert "not registered" in result.lower()


def test_registry_scratchpad_no_store():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    reg.session_id = "sess1"
    result = reg.execute("scratchpad_read()", {"scratchpad_read"})
    assert "empty" in result.lower() or "initialisation" in result.lower()


def test_registry_failure_config_503():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    failure_config = {"inject_503": True}
    result = reg.execute(
        "web_search(query='test query')", {"web_search"}, failure_config
    )
    assert "503" in result


def test_registry_call_check_profile_false():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    # get_datetime is NOT in active_tools, but check_profile=False bypasses the guard
    result = reg.call("get_datetime", {}, active_tools=set(), check_profile=False)
    assert re.match(r"\d{4}-\d{2}-\d{2}", result)


def test_registry_user_memory_no_store():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    reg.session_id = "sess1"
    result = reg._user_memory_read("test prompt")
    assert "not available" in result.lower()


def test_registry_scratchpad_with_store():
    from pipeline_tools import ToolRegistry
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scratchpad import ScratchpadStore
    store = ScratchpadStore()
    reg = ToolRegistry(scratchpad_store=store)
    reg.session_id = "test_sess"
    result = reg._scratchpad_read()
    assert "SCRATCHPAD" in result
    update_result = reg._scratchpad_update("notes", "test note content")
    assert "updated" in update_result.lower() or "✓" in update_result
