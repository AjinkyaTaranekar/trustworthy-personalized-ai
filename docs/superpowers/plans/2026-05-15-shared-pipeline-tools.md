# Shared Pipeline Tools, User Memory & Read-URL Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract all tool implementations into a shared `pipeline_tools.py` module (ending code duplication between generator and inference), add `UserMemoryStore` with 5W+H ontology sections, improve `read_url` to extract prompt-relevant content, switch inference `web_search` from DuckDuckGo to exa.ai, and enforce `<think>` blocks in training data.

**Architecture:** `pipeline_tools.py` holds every tool function and a `ToolRegistry` class with two call paths — `execute(xml_inner)` for the generator's intercept loop and `call(name, kwargs)` for the inference server. `user_memory.py` holds a JSON-on-disk `UserMemoryStore` with 7 ontology sections. Both files are imported by `sft_v3_generator.py` and `3_infererence.py`, which shed their local copies.

**Tech Stack:** Python 3.11+, pytest, exa-py, existing litellm / FastAPI / unsloth stack.

---

## File Map

| Action | File |
|--------|------|
| Create | `pipeline/pipeline_tools.py` |
| Create | `pipeline/user_memory.py` |
| Create | `pipeline/tests/test_pipeline_tools.py` |
| Create | `pipeline/tests/test_user_memory.py` |
| Modify | `pipeline/sft_v3_generator.py` |
| Modify | `pipeline/3_infererence.py` |

---

### Task 1: Create `pipeline/pipeline_tools.py` — standalone tool functions

**Files:**
- Create: `pipeline/pipeline_tools.py`
- Create: `pipeline/tests/test_pipeline_tools.py`

- [ ] **Step 1: Write failing tests for standalone tool functions**

```python
# pipeline/tests/test_pipeline_tools.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import urllib.request as _urllib_req
from unittest.mock import patch, MagicMock


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
    import re
    result = get_datetime()
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)


def test_read_url_strips_html(monkeypatch):
    from pipeline_tools import read_url
    html = (
        b"<html><head><style>body{margin:0}</style></head>"
        b"<nav>Menu</nav>"
        b"<body><p>Hello world text here. More content follows about the topic.</p></body>"
        b"<footer>Copyright</footer></html>"
    )
    class FakeResp:
        def read(self, n): return html
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(_urllib_req, "urlopen", lambda *a, **kw: FakeResp())
    result = read_url("http://example.com")
    assert "[Fetched: http://example.com]" in result
    assert "<html>" not in result
    assert "<style>" not in result
    assert "Menu" not in result  # nav stripped


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
        def read(self, n): return html
        def __enter__(self): return self
        def __exit__(self, *a): pass
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
    # Python paragraph should rank first
    assert "Python" in result[0]


def test_web_search_missing_key(monkeypatch):
    from pipeline_tools import web_search
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    result = web_search("test query")
    assert "EXA_API_KEY not set" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd C:\Users\admin\Documents\trustworthy-personalized-ai
python -m pytest pipeline/tests/test_pipeline_tools.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'pipeline_tools'`

- [ ] **Step 3: Create `pipeline/pipeline_tools.py` with standalone functions**

```python
# pipeline/pipeline_tools.py
"""
Shared tool implementations for the trustworthy-AI pipeline.

Imported by sft_v3_generator.py (training data generation) and
3_infererence.py (inference server) to eliminate diverged copies.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})
_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


# ---------------------------------------------------------------------------
# python_execute
# ---------------------------------------------------------------------------

def python_execute(code: str) -> str:
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Error: syntax_error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    return f"Error: blocked_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS:
                return f"Error: blocked_import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                return f"Error: blocked_builtin: {node.func.id}"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        out = (proc.stdout or proc.stderr).strip()
        return out if out else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (15s limit)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# web_search — exa.ai
# ---------------------------------------------------------------------------

def web_search(query: str, num_results: int = 3) -> str:
    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return (
            f"web_search unavailable: EXA_API_KEY not set. "
            f"Cannot retrieve live data for: {query}"
        )
    try:
        from exa_py import Exa
        exa = Exa(api_key=api_key)
        result = exa.search_and_contents(
            query,
            num_results=num_results,
            text={"max_characters": 400},
        )
        snippets = []
        for r in result.results:
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
            text = getattr(r, "text", "") or ""
            snippets.append(f"**{title}** ({url})\n{text[:500]}")
        return "\n\n".join(snippets) if snippets else f"No results found for: {query}"
    except ImportError:
        return "web_search unavailable: exa_py not installed — run: pip install exa-py"
    except Exception as e:
        return f"web_search error: {e}"


# ---------------------------------------------------------------------------
# read_url — HTML cleaning + prompt-aware extraction
# ---------------------------------------------------------------------------

_URL_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "be", "it", "i", "my", "me", "we", "you", "with",
    "that", "this", "from", "by", "as", "do", "not", "no",
})


def _score_paragraphs(paragraphs: list[str], prompt: str) -> list[str]:
    """Return top-5 paragraphs scored by keyword overlap with prompt."""
    if not prompt:
        return paragraphs[:8]
    prompt_words = set(re.findall(r'\w+', prompt.lower())) - _URL_STOP_WORDS
    if not prompt_words:
        return paragraphs[:8]
    scored = []
    for p in paragraphs:
        p_words = set(re.findall(r'\w+', p.lower()))
        score = len(prompt_words & p_words) / len(prompt_words)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:5]]
    return top if top else paragraphs[:3]


def read_url(url: str, prompt: str = "") -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            body = r.read(50_000).decode("utf-8", errors="replace")
        # Remove noisy structural blocks entirely (content + tags)
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            body = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", " ", body,
                flags=re.DOTALL | re.IGNORECASE,
            )
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        # Split on sentence-terminal double-space (paragraph boundaries in stripped HTML)
        paragraphs = [
            p.strip()
            for p in re.split(r'(?<=[.!?])\s{2,}', body)
            if len(p.strip()) > 60
        ]
        relevant = _score_paragraphs(paragraphs, prompt)
        content = "\n\n".join(relevant)
        if len(content) > 2000:
            content = content[:2000] + " … [truncated]"
        prefix = f"[Fetched: {url}]"
        if prompt:
            prefix += f"\nPrompt: {prompt}"
        return f"{prefix}\n\n{content}"
    except Exception as e:
        return f"read_url failed: {e}"


# ---------------------------------------------------------------------------
# get_datetime
# ---------------------------------------------------------------------------

def get_datetime() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest pipeline/tests/test_pipeline_tools.py -v -k "not web_search_missing"
```

Expected: 9 tests PASSED (skip web_search missing-key test until env is confirmed).

Run the missing-key test separately:
```
python -m pytest pipeline/tests/test_pipeline_tools.py::test_web_search_missing_key -v
```

Expected: PASSED (EXA_API_KEY should not be set in test env).

- [ ] **Step 5: Commit**

```
git add pipeline/pipeline_tools.py pipeline/tests/test_pipeline_tools.py
git commit -m "feat: add pipeline_tools.py with shared tool functions (python_execute, web_search, read_url, get_datetime)"
```

---

### Task 2: Add `ToolRegistry` class to `pipeline_tools.py`

**Files:**
- Modify: `pipeline/pipeline_tools.py` (append ToolRegistry)
- Modify: `pipeline/tests/test_pipeline_tools.py` (append registry tests)

- [ ] **Step 1: Append failing registry tests**

Add to the bottom of `pipeline/tests/test_pipeline_tools.py`:

```python
# --- ToolRegistry tests ---

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
    import re
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
    assert {"python_execute", "web_search", "read_url", "get_datetime",
            "scratchpad_read", "scratchpad_update",
            "user_memory_read", "user_memory_update"}.issubset(names)


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
    reg.register("tmp_tool", "", {"type": "object", "properties": {}, "required": []}, lambda: "x")
    reg.deregister("tmp_tool")
    result = reg.call("tmp_tool", {}, {"tmp_tool"})
    assert "not registered" in result.lower()


def test_registry_scratchpad_no_store():
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    reg.session_id = "sess1"
    # No scratchpad_store bound — returns static training string
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
    """check_profile=False lets native mode bypass profile restrictions."""
    from pipeline_tools import ToolRegistry
    reg = ToolRegistry()
    # tool is registered but NOT in active_tools — should still succeed with check_profile=False
    result = reg.call("get_datetime", {}, active_tools=set(), check_profile=False)
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}", result)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest pipeline/tests/test_pipeline_tools.py -k "registry" -v 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError: module 'pipeline_tools' has no attribute 'ToolRegistry'`

- [ ] **Step 3: Append ToolRegistry to `pipeline/pipeline_tools.py`**

Append after the `get_datetime` function (after line `return datetime.now(timezone.utc).strftime(...)`):

```python


# ---------------------------------------------------------------------------
# ToolRegistry — unified registry for generator (XML) and inference (REST)
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., str]

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _parse_python_code(s: str) -> Optional[str]:
    """Extract code= argument from a python_execute(code='...') call string."""
    for pat in (
        r'python_execute\s*\(\s*code\s*=\s*"""(.*?)"""\s*\)',
        r"python_execute\s*\(\s*code\s*=\s*'(.*?)'\s*\)",
        r'python_execute\s*\(\s*code\s*=\s*"(.*?)"\s*\)',
    ):
        m = re.search(pat, s, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return None


class ToolRegistry:
    """
    Unified tool registry used by both sft_v3_generator.py and 3_infererence.py.

    Two execution paths:
      execute(xml_inner, active_tools, failure_config)  — generator intercept loop
      call(fn_name, kwargs, active_tools, check_profile) — inference server

    Session-scoped tools (scratchpad, user_memory) read self.session_id at call
    time so the registry can be reused across requests by updating session_id.
    """

    def __init__(
        self,
        scratchpad_store=None,
        user_memory_store=None,
    ) -> None:
        self._scratchpad = scratchpad_store
        self._user_memory = user_memory_store
        self._session_id: Optional[str] = None
        self._specs: Dict[str, ToolSpec] = {}
        self._register_builtins()

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    def _register_builtins(self) -> None:
        self._specs["python_execute"] = ToolSpec(
            "python_execute",
            "Execute Python code and return stdout/stderr.",
            {"type": "object", "properties": {"code": {"type": "string", "description": "Python source code to run"}}, "required": ["code"]},
            lambda code="", **_: python_execute(code),
        )
        self._specs["web_search"] = ToolSpec(
            "web_search",
            "Search the web using exa.ai and return a summary.",
            {"type": "object", "properties": {"query": {"type": "string", "description": "Search query string"}}, "required": ["query"]},
            lambda query="", **_: web_search(query),
        )
        self._specs["read_url"] = ToolSpec(
            "read_url",
            "Fetch the text content of a URL. Pass prompt= to extract relevant content.",
            {
                "type": "object",
                "properties": {
                    "url":    {"type": "string", "description": "URL to fetch"},
                    "prompt": {"type": "string", "description": "What you are trying to extract from this page"},
                },
                "required": ["url"],
            },
            lambda url="", prompt="", **_: read_url(url, prompt),
        )
        self._specs["get_datetime"] = ToolSpec(
            "get_datetime",
            "Return the current UTC date and time.",
            {"type": "object", "properties": {}, "required": []},
            lambda **_: get_datetime(),
        )
        # Scratchpad tools — closures capture self so session_id is resolved at call time
        self._specs["scratchpad_read"] = ToolSpec(
            "scratchpad_read",
            "Read the full session scratchpad — constitution TLDR, context, tasks, notes.",
            {"type": "object", "properties": {}, "required": []},
            lambda **_: self._scratchpad_read(),
        )
        self._specs["scratchpad_update"] = ToolSpec(
            "scratchpad_update",
            "Update one section of the session scratchpad.",
            {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["context", "tasks", "notes"],
                        "description": "Section to update. 'constitution_tldr' is read-only.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the section — overwrites the previous value.",
                    },
                },
                "required": ["section", "content"],
            },
            lambda section="", content="", **_: self._scratchpad_update(section, content),
        )
        # User memory tools
        self._specs["user_memory_read"] = ToolSpec(
            "user_memory_read",
            "Read relevant user memory sections based on a prompt about the user.",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What aspect of the user you want to look up"},
                },
                "required": [],
            },
            lambda prompt="", **_: self._user_memory_read(prompt),
        )
        self._specs["user_memory_update"] = ToolSpec(
            "user_memory_update",
            "Update a section of the user's persistent memory when you learn new facts.",
            {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["who", "what", "where", "why", "how", "facts", "constraints"],
                        "description": "Ontology section to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content for the section — overwrites the previous value.",
                    },
                },
                "required": ["section", "content"],
            },
            lambda section="", content="", **_: self._user_memory_update(section, content),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, name: str, description: str, parameters: Dict[str, Any], fn: Callable) -> None:
        self._specs[name] = ToolSpec(
            name=name, description=description, parameters=parameters, fn=fn
        )

    def deregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def schemas_list(self) -> list[Dict[str, Any]]:
        """All tool schemas — used by /v1/tools endpoint."""
        return [s.schema() for s in self._specs.values()]

    def to_openai_schemas(self, active_tools: set[str]) -> list[Dict[str, Any]]:
        """OpenAI-format schemas for tools in active_tools — used by native tool-call mode."""
        return [
            {"type": "function", "function": spec.schema()}
            for name, spec in self._specs.items()
            if name in active_tools
        ]

    def call(
        self,
        fn_name: str,
        kwargs: Dict[str, Any],
        active_tools: set[str],
        check_profile: bool = True,
    ) -> str:
        """Execute a tool by name + kwargs. Used by inference server.

        check_profile=False bypasses the active_tools guard — used in native mode
        where the model selects tools from its pre-training knowledge, not the profile.
        """
        if fn_name not in self._specs:
            return f"Error: tool '{fn_name}' is not registered on this server."
        if check_profile and fn_name not in active_tools:
            return f"Error: tool '{fn_name}' is not available in this session."
        try:
            return self._specs[fn_name].fn(**kwargs)
        except Exception as e:
            return f"Tool execution error: {e}"

    def execute(
        self,
        tool_inner: str,
        active_tools: set[str],
        failure_config: Optional[Dict] = None,
    ) -> str:
        """Parse XML inner text and execute the tool. Used by generator intercept loop."""
        s = tool_inner.strip()

        if s.startswith("python_execute"):
            if "python_execute" not in active_tools:
                return "Error: python_execute is not available in this session."
            code = _parse_python_code(s)
            if code is None:
                return "Error: could not parse python_execute arguments."
            return python_execute(code)

        if s.startswith("web_search"):
            if "web_search" not in active_tools:
                return "Error: web_search is not available in this session."
            m = re.search(r"query\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)
            query = m.group(1) if m else s
            if failure_config and failure_config.get("inject_503"):
                failure_config.setdefault("web_search_count", 0)
                failure_config["web_search_count"] += 1
                if failure_config["web_search_count"] == 1:
                    return "HTTP 503 Service Unavailable. The search service is temporarily down. Please retry with a different query."
            return web_search(query)

        if s.startswith("read_url"):
            if "read_url" not in active_tools:
                return "Error: read_url is not available in this session."
            url_m = re.search(r"url\s*=\s*['\"](.+?)['\"]", s)
            prompt_m = re.search(r"prompt\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)
            return read_url(
                url_m.group(1) if url_m else "",
                prompt_m.group(1) if prompt_m else "",
            )

        if s.startswith("get_datetime"):
            return get_datetime()

        if s.startswith("scratchpad_read"):
            if "scratchpad_read" not in active_tools:
                return "Error: scratchpad_read is not available in this session."
            return self._scratchpad_read()

        if s.startswith("scratchpad_update"):
            section_m = re.search(r"section\s*=\s*['\"](\w+)['\"]", s)
            content_m = re.search(r'content\s*=\s*["\'](.+?)["\']', s, re.DOTALL)
            return self._scratchpad_update(
                section_m.group(1) if section_m else "",
                content_m.group(1) if content_m else "",
            )

        if s.startswith("user_memory_read"):
            prompt_m = re.search(r"prompt\s*=\s*['\"](.+?)['\"]", s, re.DOTALL)
            return self._user_memory_read(prompt_m.group(1) if prompt_m else "")

        if s.startswith("user_memory_update"):
            section_m = re.search(r"section\s*=\s*['\"](\w+)['\"]", s)
            content_m = re.search(r'content\s*=\s*["\'](.+?)["\']', s, re.DOTALL)
            return self._user_memory_update(
                section_m.group(1) if section_m else "",
                content_m.group(1) if content_m else "",
            )

        tool_name = s.split("(")[0].strip() if "(" in s else s[:40]
        return f"Error: unknown tool '{tool_name}' — only registered tools are callable."

    # ── Session-scoped backends ───────────────────────────────────────────────

    def _scratchpad_read(self) -> str:
        if self._scratchpad is None or self._session_id is None:
            return "(scratchpad is empty — training example initialisation)"
        return self._scratchpad.read(self._session_id)

    def _scratchpad_update(self, section: str, content: str) -> str:
        if self._scratchpad is None or self._session_id is None:
            return "(scratchpad updated)"
        return self._scratchpad.update(self._session_id, section, content)

    def _user_memory_read(self, prompt: str = "") -> str:
        if self._user_memory is None or self._session_id is None:
            return "(user memory not available in this session)"
        return self._user_memory.read(self._session_id, prompt)

    def _user_memory_update(self, section: str, content: str) -> str:
        if self._user_memory is None or self._session_id is None:
            return "(user memory update skipped — training context)"
        return self._user_memory.update(self._session_id, section, content)
```

- [ ] **Step 4: Run registry tests**

```
python -m pytest pipeline/tests/test_pipeline_tools.py -v
```

Expected: All tests PASSED. If `test_registry_failure_config_503` fails because web_search runs (EXA_API_KEY is set), check that the mock returns 503 before any real API call — the failure_config check is first in `execute()`.

- [ ] **Step 5: Commit**

```
git add pipeline/pipeline_tools.py pipeline/tests/test_pipeline_tools.py
git commit -m "feat: add ToolRegistry to pipeline_tools — unified execute/call/schemas for generator and inference"
```

---

### Task 3: Create `pipeline/user_memory.py`

**Files:**
- Create: `pipeline/user_memory.py`
- Create: `pipeline/tests/test_user_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# pipeline/tests/test_user_memory.py
import sys
import tempfile
import json
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
    # Should include at least one of the relevant sections
    assert "ML researcher" in result or "GPU budget" in result or "language model" in result


def test_read_with_prompt_always_includes_who():
    from user_memory import UserMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = UserMemoryStore(store_dir=Path(tmp))
        store.update("user_001", "who", "Postgraduate student at Trinity College Dublin")
        result = store.read("user_001", prompt="weather forecast")
    # 'who' is always included even with low relevance
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
    # Result should contain exactly 3 sections (WHO + 2 most relevant)
    assert result.count("[") >= 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest pipeline/tests/test_user_memory.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'user_memory'`

- [ ] **Step 3: Create `pipeline/user_memory.py`**

```python
# pipeline/user_memory.py
"""
Per-user persistent memory with 5W+H ontology sections.

Stores one JSON file per user at pipeline/data/user_memory/<user_id>.json.
The UserMemoryStore is wire-compatible with the GraphRAG integration planned
for a later milestone — the read/update interface is the same.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

_SECTIONS: Dict[str, str] = {
    "who":         "Role, expertise, background, identity",
    "what":        "Current goals, projects, active topics",
    "where":       "Geographic/domain context, jurisdiction",
    "why":         "Underlying motivations, constraints, reasons",
    "how":         "Preferred formats, communication style, tool preferences",
    "facts":       "Confirmed factual assertions about the user",
    "constraints": "Hard limits: budget, time, access restrictions",
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "be", "it", "i", "my", "me", "we", "you", "with",
    "that", "this", "from", "by", "as", "do", "not", "no",
})

_DEFAULT_STORE_DIR = Path(__file__).parent / "data" / "user_memory"


def _relevance_score(content: str, prompt: str) -> float:
    """Keyword overlap fraction: |prompt_words ∩ content_words| / |prompt_words|."""
    if not content or "(empty" in content:
        return 0.0
    prompt_words = set(re.findall(r'\w+', prompt.lower())) - _STOP_WORDS
    if not prompt_words:
        return 0.5
    content_words = set(re.findall(r'\w+', content.lower()))
    return len(prompt_words & content_words) / len(prompt_words)


class UserMemoryStore:
    """JSON-on-disk per-user memory. In-memory LRU cache; saves on every update."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self._dir = Path(store_dir) if store_dir else _DEFAULT_STORE_DIR
        self._cache: Dict[str, Dict[str, str]] = {}

    def _path(self, user_id: str) -> Path:
        return self._dir / f"{user_id}.json"

    def _empty_record(self) -> Dict[str, str]:
        return {k: f"(empty — {v})" for k, v in _SECTIONS.items()}

    def _load(self, user_id: str) -> Dict[str, str]:
        if user_id in self._cache:
            return self._cache[user_id]
        p = self._path(user_id)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in _SECTIONS.items():
                    data.setdefault(k, f"(empty — {v})")
                self._cache[user_id] = data
                return data
            except (json.JSONDecodeError, OSError):
                pass
        data = self._empty_record()
        self._cache[user_id] = data
        return data

    def _save(self, user_id: str, data: Dict[str, str]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(user_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def read(self, user_id: str, prompt: str = "") -> str:
        """Return memory sections. Without prompt: all sections. With prompt: top-3 by relevance."""
        data = self._load(user_id)
        if not prompt:
            lines = [f"[{k.upper()}]\n{v}" for k, v in data.items()]
            return f"=== USER MEMORY (id: {user_id}) ===\n\n" + "\n\n".join(lines)

        scored = []
        for section, content in data.items():
            score = _relevance_score(content, prompt)
            if section == "who":
                score = max(score, 0.1)  # always surface identity as baseline context
            scored.append((score, section, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(s, c) for _, s, c in scored[:3]]
        lines = [f"[{s.upper()}]\n{c}" for s, c in top]
        return f"=== USER MEMORY (id: {user_id}) ===\n\n" + "\n\n".join(lines)

    def update(self, user_id: str, section: str, content: str) -> str:
        if section not in _SECTIONS:
            return (
                f"Error: '{section}' is not a valid section. "
                f"Valid sections: {sorted(_SECTIONS.keys())}."
            )
        data = self._load(user_id)
        data[section] = content
        self._cache[user_id] = data
        self._save(user_id, data)
        return f"✓ user_memory[{section}] updated"
```

- [ ] **Step 4: Run tests**

```
python -m pytest pipeline/tests/test_user_memory.py -v
```

Expected: All 12 tests PASSED.

- [ ] **Step 5: Commit**

```
git add pipeline/user_memory.py pipeline/tests/test_user_memory.py
git commit -m "feat: add UserMemoryStore with 5W+H ontology sections and JSON-on-disk persistence"
```

---

### Task 4: Update `pipeline/sft_v3_generator.py`

Remove local tool implementations and replace with `ToolRegistry`. Add `<think>` block quality gate (retry up to 2× if the first assistant message has no `<think>` block ≥ 150 chars).

**Files:**
- Modify: `pipeline/sft_v3_generator.py`
- Test: `pipeline/tests/test_sft_v3_generator.py` (add think-block check)

- [ ] **Step 1: Add failing think-block test**

Open `pipeline/tests/test_sft_v3_generator.py` and append:

```python
def test_think_block_length_min_threshold():
    """_think_block_length must return ≥150 for a valid training example."""
    from sft_v3_generator import _think_block_length
    short = "<think>Too short.</think><answer>Answer.</answer>"
    long_think = "<think>" + "x" * 200 + "</think><answer>Answer.</answer>"
    assert _think_block_length(short) < 150
    assert _think_block_length(long_think) >= 150


def test_build_v3_example_rejects_missing_think(monkeypatch):
    """_process_one_v3 must return 'error' when all generation attempts lack <think>."""
    import io
    import threading
    from sft_v3_generator import _process_one_v3, TOOL_PROFILES

    # Mock _generate_with_intercept to return a response without <think>
    no_think_conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Direct answer without thinking tags.<answer>Answer.</answer>"},
    ]
    monkeypatch.setattr(
        "sft_v3_generator._generate_with_intercept",
        lambda **kwargs: no_think_conv,
    )

    buf = io.StringIO()
    lock = threading.Lock()
    item = {"category": "impossible_tasks", "question": "What is 1 + 1?"}
    result = _process_one_v3(item, "test_model", None, buf, lock, 1, 1, 0.0)
    assert result == "error"
```

- [ ] **Step 2: Run to confirm failure**

```
python -m pytest pipeline/tests/test_sft_v3_generator.py::test_build_v3_example_rejects_missing_think -v
```

Expected: FAIL (function doesn't reject missing think yet, or AttributeError).

- [ ] **Step 3: Edit `sft_v3_generator.py` — remove local tool functions, import ToolRegistry**

Replace the following sections at the top of the file:

**Remove lines 46–50** (the `_MAX_RETRIES` / `_BASE_DELAY` block stays — only tool-related code is removed):

Remove `# ---------------------------------------------------------------------------` through `# ---------------------------------------------------------------------------` blocks that contain:
- `_ALLOWED_IMPORTS` (lines ~181–186)
- `_BLOCKED_BUILTINS` (line ~186)
- `_parse_python_code` function
- `_run_safe_python` function
- `_fetch_url` function
- `_execute_tool_v3` function
- `_exa_search` function

Add this import near the top (after `import litellm`):

```python
from pipeline_tools import ToolRegistry as _ToolRegistry

_TOOL_REGISTRY = _ToolRegistry()   # no scratchpad/user_memory — training context
```

- [ ] **Step 4: Edit `_generate_with_intercept` — replace `_execute_tool_v3` call**

Find this line in `_generate_with_intercept` (around line 420):

```python
        result = _execute_tool_v3(tool_inner, active_tools, failure_config)
```

Replace with:

```python
        result = _TOOL_REGISTRY.execute(tool_inner, active_tools, failure_config)
```

- [ ] **Step 5: Edit `_process_one_v3` — add `<think>` quality gate**

Find the block that calls `_generate_with_intercept` and saves the example (around line 569):

```python
    try:
        t0 = time.monotonic()
        conversation = _generate_with_intercept(
            messages=initial_messages,
            model=model,
            tool_profile=tool_profile,
            api_base=api_base,
            failure_config=failure_config,
        )
        n_tool_turns = sum(1 for m in conversation if m["role"] == "tool")
        print(f"  {tag} {len(conversation)} msgs ({n_tool_turns} tool turns) in {time.monotonic()-t0:.1f}s")

        example = _build_v3_example(conversation, question, category, tool_profile)
```

Replace with:

```python
    try:
        t0 = time.monotonic()
        conversation = None
        for _think_attempt in range(2):
            conversation = _generate_with_intercept(
                messages=initial_messages,
                model=model,
                tool_profile=tool_profile,
                api_base=api_base,
                failure_config=failure_config,
            )
            first_asst = next(
                (m["content"] for m in conversation if m["role"] == "assistant"), ""
            )
            if _think_block_length(first_asst) >= 150:
                break
            if _think_attempt == 0:
                print(f"  {tag} no <think> block (attempt 1) — retrying")
        else:
            print(f"  {tag} skipped: no valid <think> block after 2 attempts")
            return "error"

        n_tool_turns = sum(1 for m in conversation if m["role"] == "tool")
        print(f"  {tag} {len(conversation)} msgs ({n_tool_turns} tool turns) in {time.monotonic()-t0:.1f}s")

        example = _build_v3_example(conversation, question, category, tool_profile)
```

- [ ] **Step 6: Run the full test suite**

```
python -m pytest pipeline/tests/test_sft_v3_generator.py -v
```

Expected: All tests PASSED including `test_build_v3_example_rejects_missing_think`.

- [ ] **Step 7: Smoke-test the import**

```
python -c "import sys; sys.path.insert(0, 'pipeline'); from sft_v3_generator import _TOOL_REGISTRY; print('ok', type(_TOOL_REGISTRY))"
```

Expected: `ok <class 'pipeline_tools.ToolRegistry'>`

- [ ] **Step 8: Commit**

```
git add pipeline/sft_v3_generator.py pipeline/tests/test_sft_v3_generator.py
git commit -m "refactor: sft_v3_generator uses shared ToolRegistry; add <think> block quality gate with 2-attempt retry"
```

---

### Task 5: Update `pipeline/3_infererence.py`

Replace all local tool implementations and `_REGISTRY` dict with `ToolRegistry`. Add `UserMemoryStore`. Update tool profiles to include user_memory tools. Update the system prompt note.

**Files:**
- Modify: `pipeline/3_infererence.py`

The changes are surgical — do them one section at a time to avoid breaking the server.

- [ ] **Step 1: Replace tool implementations and registry at the top of the file**

Remove these exact named blocks (leave `_sanitise_tool_output`, `_is_tool_error`, `_is_non_retryable_tool_error`, and `_tool_failure_prompt` completely untouched — they contain no `_REGISTRY` references and must stay):

1. The `# Code safety validation` comment block + `_ALLOWED_IMPORTS`, `_BLOCKED_BUILTINS`, `_validate_code` function (~lines 114–145)
2. `_python_execute` function
3. `_get_datetime` function
4. `_web_search` function
5. `_read_url` function
6. `_scratchpad_read` function
7. `_scratchpad_update` function
8. `ToolSpec` dataclass
9. `_REGISTRY: Dict[str, ToolSpec] = {}` module variable
10. `register_tool()` function
11. All `register_tool("python_execute", ...)`, `register_tool("get_datetime", ...)`, `register_tool("web_search", ...)`, `register_tool("read_url", ...)`, `register_tool("scratchpad_read", ...)`, `register_tool("scratchpad_update", ...)` calls
12. `_SCRATCHPAD_TOOLS = {"scratchpad_read", "scratchpad_update"}` line
13. The `TOOL_PROFILES: Dict[str, set] = {…}` dict

After removing all the above, insert the following replacement block at the location where `_ALLOWED_IMPORTS` used to begin (just above the Metrics class):

```python
# ---------------------------------------------------------------------------
# Shared tool registry — implementations live in pipeline_tools.py
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_tools import ToolRegistry   # noqa: E402

try:
    from user_memory import UserMemoryStore as _UserMemoryStoreClass
    _user_memory_importable = True
except ImportError as _e:
    _user_memory_importable = False
    _UserMemoryStoreClass = None
    print(f"[INFO] user_memory not importable ({_e}) — user_memory tools will be no-ops")

# Single global registry — session_id is updated per-request in chat_completions
_TOOL_REGISTRY: ToolRegistry = ToolRegistry()   # stores bound at main() startup

# Tool profiles — which tools are active per session
_ALWAYS_ON_TOOLS = frozenset({
    "scratchpad_read", "scratchpad_update",
    "user_memory_read", "user_memory_update",
})

TOOL_PROFILES: Dict[str, set] = {
    "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime"} | _ALWAYS_ON_TOOLS,
    "compute_only":       {"python_execute"} | _ALWAYS_ON_TOOLS,
    "compute_and_search": {"python_execute", "web_search", "read_url"} | _ALWAYS_ON_TOOLS,
    "no_tools":           set(_ALWAYS_ON_TOOLS),
}
```

- [ ] **Step 2: Remove the prompt-injection sanitiser dependency on `_REGISTRY`**

The `_sanitise_tool_output`, `_is_tool_error`, `_is_non_retryable_tool_error`, `_tool_failure_prompt` functions do NOT depend on `_REGISTRY` — they are unchanged and stay.

- [ ] **Step 3: Replace global `_CURRENT_SESSION_ID` with `_TOOL_REGISTRY.session_id`**

Find:
```python
_CURRENT_SESSION_ID: Optional[str] = None  # Set per-request, read by scratchpad tool handlers
```

Remove this line. The session_id is now on `_TOOL_REGISTRY`.

- [ ] **Step 4: Replace `_to_openai_schemas` function**

Find:
```python
def _to_openai_schemas(active_tools: set) -> List[Dict[str, Any]]:
    ...
    return [
        {"type": "function", "function": spec.schema()}
        for name, spec in _REGISTRY.items()
        if name in active_tools
    ]
```

Replace with:
```python
def _to_openai_schemas(active_tools: set) -> List[Dict[str, Any]]:
    return _TOOL_REGISTRY.to_openai_schemas(active_tools)
```

- [ ] **Step 5: Update `_build_system_prompt` scratchpad note**

Find `_SCRATCHPAD_NOTE = (` and replace the entire string with:

```python
    _SCRATCHPAD_NOTE = (
        "\n\nAlways-on tools (available in every session — not listed in tool inventory above):\n"
        "  scratchpad_read()                              → read your full scratchpad\n"
        "  scratchpad_update(section=..., content=...)    → update context / tasks / notes\n"
        "  user_memory_read(prompt='...')                 → read relevant user memory\n"
        "  user_memory_update(section=..., content=...)   → update user memory when you learn facts\n"
        "Use scratchpad for any query with 3+ requirements or 2+ tool calls (P24).\n"
        "Call user_memory_read at the start of conversations to retrieve user context; "
        "call user_memory_update whenever you learn a new fact about the user."
    )
```

- [ ] **Step 6: Update `chat_completions` — session binding and tool execution**

Find:
```python
    # ── Scratchpad session binding ────────────────────────────────────────
    global _CURRENT_SESSION_ID
    if _SCRATCHPAD_STORE is not None:
        _CURRENT_SESSION_ID = req.session_id or _SCRATCHPAD_STORE.new_session_id()
    else:
        _CURRENT_SESSION_ID = req.session_id
```

Replace with:
```python
    # ── Session binding (scratchpad + user memory) ────────────────────────
    session_id = req.session_id
    if _SCRATCHPAD_STORE is not None and not session_id:
        session_id = _SCRATCHPAD_STORE.new_session_id()
    _TOOL_REGISTRY.session_id = session_id
```

Find the tool-call execution block inside the `for iteration` loop:

```python
            if tc:
                fn_name = tc["function"]
                kwargs_preview = str(tc["kwargs"])[:120]
                print(f"[TOOL] Calling: {fn_name}({kwargs_preview})")
                if fn_name not in _REGISTRY:
                    raw_result = f"Error: tool '{fn_name}' is not registered on this server."
                    print(f"[TOOL] Error: tool '{fn_name}' is not registered on this server")
                elif not use_native and fn_name not in active_tools:
                    raw_result = f"Error: tool '{fn_name}' is not available in profile '{req.tool_profile}'."
                    print(f"[TOOL] Error: tool '{fn_name}' not available in profile '{req.tool_profile}'")
                else:
                    try:
                        raw_result = _REGISTRY[fn_name].fn(**tc["kwargs"])
                        result_preview = str(raw_result)[:80].replace("\n", "\\n")
                        print(f"[TOOL] Result ({len(str(raw_result))} chars): {result_preview}")
                    except Exception as e:
                        raw_result = f"Tool execution error: {e}"
                        print(f"[TOOL] Execution error in {fn_name}: {e}")
                tools_used[fn_name] = tools_used.get(fn_name, 0) + 1
```

Replace with:
```python
            if tc:
                fn_name = tc["function"]
                kwargs_preview = str(tc["kwargs"])[:120]
                print(f"[TOOL] Calling: {fn_name}({kwargs_preview})")
                raw_result = _TOOL_REGISTRY.call(
                    fn_name, tc["kwargs"], active_tools,
                    check_profile=not use_native,
                )
                result_preview = str(raw_result)[:80].replace("\n", "\\n")
                print(f"[TOOL] Result ({len(str(raw_result))} chars): {result_preview}")
                tools_used[fn_name] = tools_used.get(fn_name, 0) + 1
```

Find the scratchpad task_status injection block:

```python
                if (
                    _SCRATCHPAD_STORE is not None
                    and _CURRENT_SESSION_ID
                    and fn_name not in ("scratchpad_read", "scratchpad_update")
                ):
                    task_status = _SCRATCHPAD_STORE.get_task_status(_CURRENT_SESSION_ID)
```

Replace with:
```python
                if (
                    _SCRATCHPAD_STORE is not None
                    and _TOOL_REGISTRY.session_id
                    and fn_name not in ("scratchpad_read", "scratchpad_update")
                ):
                    task_status = _SCRATCHPAD_STORE.get_task_status(_TOOL_REGISTRY.session_id)
```

Find the harness `check_and_steer` call (inside the `if effective_harness and _HARNESS` block):

```python
        final, harness_violations, harness_retries = _HARNESS.check_and_steer(
            response=final,
            conv=conv,
            question=user_turn,
            tool_profile_label=req.tool_profile,
            generate_fn=lambda c, ts=1.0: _generate(
                c,
                req.max_new_tokens,
                max(req.temperature, 0.3) * ts if ts != 1.0 else req.temperature,
                req.greedy and ts == 1.0,
            )[0],
            session_id=_CURRENT_SESSION_ID,
            max_retries=2,
        )
```

Replace `session_id=_CURRENT_SESSION_ID` with `session_id=_TOOL_REGISTRY.session_id`:

```python
        final, harness_violations, harness_retries = _HARNESS.check_and_steer(
            response=final,
            conv=conv,
            question=user_turn,
            tool_profile_label=req.tool_profile,
            generate_fn=lambda c, ts=1.0: _generate(
                c,
                req.max_new_tokens,
                max(req.temperature, 0.3) * ts if ts != 1.0 else req.temperature,
                req.greedy and ts == 1.0,
            )[0],
            session_id=_TOOL_REGISTRY.session_id,
            max_retries=2,
        )
```

- [ ] **Step 7: Update REST endpoints that reference `_REGISTRY`**

Find:
```python
@app.get("/v1/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": [t.schema() for t in _REGISTRY.values()],
        "profiles": {k: sorted(v) for k, v in TOOL_PROFILES.items()},
    }
```

Replace with:
```python
@app.get("/v1/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": _TOOL_REGISTRY.schemas_list(),
        "profiles": {k: sorted(v) for k, v in TOOL_PROFILES.items()},
    }
```

Find:
```python
@app.post("/v1/tools/register")
def register_tool_endpoint(req: ToolRegistration) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    try:
        exec(req.python_code, ns)  # noqa: S102
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code compile error: {e}")
    fn = ns.get("tool_fn")
    if not callable(fn):
        raise HTTPException(status_code=400, detail="python_code must define a callable named 'tool_fn'")
    register_tool(req.name, req.description, req.parameters, fn)
    return {"registered": req.name, "total_tools": len(_REGISTRY)}
```

Replace with:
```python
@app.post("/v1/tools/register")
def register_tool_endpoint(req: ToolRegistration) -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    try:
        exec(req.python_code, ns)  # noqa: S102
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code compile error: {e}")
    fn = ns.get("tool_fn")
    if not callable(fn):
        raise HTTPException(status_code=400, detail="python_code must define a callable named 'tool_fn'")
    _TOOL_REGISTRY.register(req.name, req.description, req.parameters, fn)
    return {"registered": req.name, "total_tools": len(_TOOL_REGISTRY._specs)}
```

Find:
```python
@app.delete("/v1/tools/{name}")
def delete_tool(name: str) -> Dict[str, Any]:
    if name not in _REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    del _REGISTRY[name]
    for profile_set in TOOL_PROFILES.values():
        profile_set.discard(name)
    return {"removed": name}
```

Replace with:
```python
@app.delete("/v1/tools/{name}")
def delete_tool(name: str) -> Dict[str, Any]:
    if name not in _TOOL_REGISTRY._specs:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    _TOOL_REGISTRY.deregister(name)
    for profile_set in TOOL_PROFILES.values():
        profile_set.discard(name)
    return {"removed": name}
```

- [ ] **Step 8: Bind stores in `main()`**

Find the scratchpad initialisation block in `main()`:

```python
    # ── Scratchpad store ──────────────────────────────────────────────────
    global _SCRATCHPAD_STORE
    if _scratchpad_available:
        _SCRATCHPAD_STORE = ScratchpadStore()
        print("[SCRATCHPAD] Session scratchpad store initialised")
    else:
        print("[SCRATCHPAD] scratchpad module not available — scratchpad tools disabled")
```

Replace with:

```python
    # ── Scratchpad store ──────────────────────────────────────────────────
    global _SCRATCHPAD_STORE
    if _scratchpad_available:
        _SCRATCHPAD_STORE = ScratchpadStore()
        _TOOL_REGISTRY._scratchpad = _SCRATCHPAD_STORE
        print("[SCRATCHPAD] Session scratchpad store initialised")
    else:
        print("[SCRATCHPAD] scratchpad module not available — scratchpad tools disabled")

    # ── User memory store ─────────────────────────────────────────────────
    if _user_memory_importable and _UserMemoryStoreClass is not None:
        _TOOL_REGISTRY._user_memory = _UserMemoryStoreClass()
        print("[USER MEMORY] User memory store initialised (pipeline/data/user_memory/)")
    else:
        print("[USER MEMORY] user_memory module not available — user_memory tools disabled")
```

- [ ] **Step 9: Smoke-test the import**

```
python -c "
import sys
sys.path.insert(0, 'pipeline')
# Patch heavy imports so we can import without GPU
import unittest.mock as m
sys.modules['torch'] = m.MagicMock()
sys.modules['uvicorn'] = m.MagicMock()
sys.modules['fastapi'] = m.MagicMock()
sys.modules['pydantic'] = m.MagicMock()
from pipeline_tools import ToolRegistry
print('ToolRegistry import ok')
from user_memory import UserMemoryStore
print('UserMemoryStore import ok')
print('All shared imports OK')
"
```

Expected: three `ok` lines, no errors.

- [ ] **Step 10: Verify tool registry has all tools**

```
python -c "
import sys
sys.path.insert(0, 'pipeline')
from pipeline_tools import ToolRegistry
reg = ToolRegistry()
names = {s['name'] for s in reg.schemas_list()}
expected = {'python_execute','web_search','read_url','get_datetime','scratchpad_read','scratchpad_update','user_memory_read','user_memory_update'}
assert expected.issubset(names), f'Missing: {expected - names}'
print('All 8 tools registered:', sorted(names))
"
```

Expected: prints all 8 tool names.

- [ ] **Step 11: Run full test suite**

```
python -m pytest pipeline/tests/ -v --ignore=pipeline/tests/test_constitutional_harness.py 2>&1 | tail -20
```

Expected: All tests PASSED (or pre-existing failures only — not new ones from this change).

- [ ] **Step 12: Commit**

```
git add pipeline/3_infererence.py
git commit -m "refactor: 3_infererence uses shared ToolRegistry; switch web_search to exa.ai; add user_memory tools to all profiles"
```
