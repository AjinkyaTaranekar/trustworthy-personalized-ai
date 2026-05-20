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


# ---------------------------------------------------------------------------
# scratchpad_sections / user_memory_sections — static schema introspection
# ---------------------------------------------------------------------------

def scratchpad_sections() -> str:
    return (
        "Scratchpad sections — call scratchpad_update(section=<key>, content=<value>) to write:\n"
        "  context  — Background about the current task or conversation state (what is being solved, constraints noted).\n"
        "  tasks    — Ordered list of sub-steps or sub-goals to complete in this response.\n"
        "  notes    — Intermediate calculations, hypotheses, or scratch work that inform the final answer.\n\n"
        "Use scratchpad_read() to retrieve all sections at once. "
        "The scratchpad is session-scoped and not visible to the user."
    )


def user_memory_sections() -> str:
    return (
        "User memory sections (5W+H ontology) — call user_memory_update(section=<key>, content=<value>) to write:\n"
        "  who         — Who the user is: role, identity, background, demographics.\n"
        "  what        — What the user does or is currently working on.\n"
        "  where       — Where the user operates: location, platform, regulatory or cultural context.\n"
        "  why         — Why the user asks: motivation, goals, values, what they are trying to achieve.\n"
        "  how         — How the user prefers to work: learning style, tools, communication style, level of detail wanted.\n"
        "  facts       — Specific known facts: skill levels, past experiences, stated preferences, language.\n"
        "  constraints — Hard limits: budget, time, policy, legal restrictions, access limitations.\n\n"
        "Use user_memory_read(prompt=<topic>) to retrieve relevant sections. "
        "Memory persists across conversations and should be updated whenever new durable facts are learned."
    )


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
      execute(xml_inner, active_tools, failure_config)   — generator intercept loop
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
            (
                "Execute sandboxed Python code and return stdout or stderr. "
                "Use for: arithmetic, unit conversions, data transformations, algorithm prototyping. "
                "Allowed imports: math, statistics, decimal, fractions, random, itertools, "
                "functools, operator, collections, re, string. "
                "No file I/O, no network, no exec/eval. "
                "Usage: python_execute(code='print(2 ** 10)')"
            ),
            {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code to run. Use print() to produce output."}},
                "required": ["code"],
            },
            lambda code="", **_: python_execute(code),
        )
        self._specs["web_search"] = ToolSpec(
            "web_search",
            (
                "Search the web via exa.ai and return top-3 result summaries with titles and URLs. "
                "Use for: current events, live prices, named entities, facts with a recency requirement. "
                "Write a natural-language question or keyword phrase as the query. "
                "Usage: web_search(query='current USD/EUR exchange rate')"
            ),
            {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Natural-language search query — be specific to get precise results."}},
                "required": ["query"],
            },
            lambda query="", **_: web_search(query),
        )
        self._specs["read_url"] = ToolSpec(
            "read_url",
            (
                "Fetch and clean the text content of a URL, returning the most relevant paragraphs. "
                "Use after web_search to read a specific page in detail. "
                "Pass prompt= to bias extraction toward what you are looking for. "
                "Usage: read_url(url='https://example.com', prompt='annual revenue figures')"
            ),
            {
                "type": "object",
                "properties": {
                    "url":    {"type": "string", "description": "Full URL including https://"},
                    "prompt": {"type": "string", "description": "What you want to extract — guides paragraph selection."},
                },
                "required": ["url"],
            },
            lambda url="", prompt="", **_: read_url(url, prompt),
        )
        self._specs["get_datetime"] = ToolSpec(
            "get_datetime",
            (
                "Return the current UTC date and time as a string. "
                "Call this at the start of any time-sensitive response to anchor your answer in real time. "
                "No arguments needed. "
                "Usage: get_datetime()"
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda **_: get_datetime(),
        )
        self._specs["scratchpad_sections"] = ToolSpec(
            "scratchpad_sections",
            (
                "Return the list of available scratchpad section keys and what each section is for. "
                "Call this BEFORE scratchpad_update so you know which section to write to. "
                "No arguments needed. "
                "Usage: scratchpad_sections()"
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda **_: scratchpad_sections(),
        )
        self._specs["user_memory_sections"] = ToolSpec(
            "user_memory_sections",
            (
                "Return the user memory section keys and their meaning (5W+H ontology). "
                "Call this BEFORE user_memory_update so you know which section to write to. "
                "No arguments needed. "
                "Usage: user_memory_sections()"
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda **_: user_memory_sections(),
        )
        # Scratchpad tools — closures capture self so session_id is resolved at call time
        self._specs["scratchpad_read"] = ToolSpec(
            "scratchpad_read",
            (
                "Read the entire session scratchpad and return all sections (context, tasks, notes). "
                "Call this to resume multi-step reasoning or check what you have already recorded. "
                "No arguments needed. "
                "Usage: scratchpad_read()"
            ),
            {"type": "object", "properties": {}, "required": []},
            lambda **_: self._scratchpad_read(),
        )
        self._specs["scratchpad_update"] = ToolSpec(
            "scratchpad_update",
            (
                "Write to one section of the session scratchpad. Overwrites the previous value for that section. "
                "Call scratchpad_sections() first to see valid section keys and their purpose. "
                "Sections: context (task background), tasks (ordered steps), notes (intermediate work). "
                "Usage: scratchpad_update(section='notes', content='Step 1 result: 42')"
            ),
            {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["context", "tasks", "notes"],
                        "description": "Which section to write. Call scratchpad_sections() to understand each section's purpose.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content — overwrites the previous value for this section.",
                    },
                },
                "required": ["section", "content"],
            },
            lambda section="", content="", **_: self._scratchpad_update(section, content),
        )
        # User memory tools
        self._specs["user_memory_read"] = ToolSpec(
            "user_memory_read",
            (
                "Read the user's persistent memory and return sections relevant to your prompt. "
                "Call this at the start of every response to personalise your answer. "
                "Pass a prompt describing what aspect of the user you need (e.g. 'technical background'). "
                "Usage: user_memory_read(prompt='user goals and constraints')"
            ),
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What aspect of the user you want — guides which sections are returned."},
                },
                "required": [],
            },
            lambda prompt="", **_: self._user_memory_read(prompt),
        )
        self._specs["user_memory_update"] = ToolSpec(
            "user_memory_update",
            (
                "Write a new fact about the user to their persistent memory. "
                "Call this when you learn something durable about the user (role, goal, preference, constraint). "
                "Call user_memory_sections() first to choose the correct section key. "
                "Usage: user_memory_update(section='facts', content='Prefers metric units.')"
            ),
            {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["who", "what", "where", "why", "how", "facts", "constraints"],
                        "description": "5W+H section to write. Call user_memory_sections() to understand each section's meaning.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content — overwrites the previous value for this section.",
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
        """OpenAI-format schemas for active_tools — used by native tool-call mode."""
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

        if s.startswith("scratchpad_sections"):
            return scratchpad_sections()

        if s.startswith("user_memory_sections"):
            return user_memory_sections()

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
