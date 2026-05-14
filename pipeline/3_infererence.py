"""
Inference Server
================
FastAPI server that loads a model once on startup and serves it via HTTP.
Designed like a frontier-lab serving stack: tool registry, server-side tool
execution loop, per-request metrics, and dynamic tool registration.

Install dependencies:
    pip install fastapi uvicorn pydantic

Usage:
    python pipeline/3_infererence.py --model_dir models/checkpoint_sft --port 8000
    python pipeline/3_infererence.py --base_model unsloth/Qwen3-0.6B --port 8000

Endpoints:
    GET  /health                   liveness + model name
    GET  /v1/models                list loaded model
    GET  /v1/tools                 list registered tools and their schemas
    POST /v1/tools/register        add a new tool at runtime
    DELETE /v1/tools/{name}        remove a tool
    POST /v1/chat/completions      generate (tool loop handled server-side)
    GET  /metrics                  latency, throughput, tool call counts
    POST /metrics/reset            zero all counters

Benchmark client (4_benchmark.py) calls POST /v1/chat/completions.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pipeline modules — optional imports guarded so the server starts even when
# optional dependencies (falkordb, etc.) are not installed.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg, PipelineConfig                                    # noqa: E402

try:
    from user_modelling import (                                           # noqa: E402
        GraphClient, write_pipeline, retrieve_for_query,
        inspect_memory, contest_belief, correct_belief,
    )
    _user_modelling_available = True
except ImportError as _e:
    _user_modelling_available = False
    GraphClient = write_pipeline = retrieve_for_query = None
    inspect_memory = contest_belief = correct_belief = None
    print(f"[INFO] user_modelling not importable ({_e}) — ENABLE_USER_MODELLING will be disabled")

try:
    from empathy import analyse_appraisal, APPRAISAL_SYSTEM_PREFIX        # noqa: E402
    _empathy_available = True
except ImportError as _e:
    _empathy_available = False
    analyse_appraisal = APPRAISAL_SYSTEM_PREFIX = None
    print(f"[INFO] empathy module not importable ({_e}) — ENABLE_EMPATHY will be disabled")

try:
    from ontology_verifier import (                                        # noqa: E402
        OntologyGraph, score_response as _onto_score_response,
    )
    _ontology_available = True
except ImportError as _e:
    _ontology_available = False
    OntologyGraph = _onto_score_response = None
    print(f"[INFO] ontology_verifier not importable ({_e}) — ENABLE_ONTOLOGY_VERIF will be disabled")

try:
    from constitutional_harness import ConstitutionalHarness
    _harness_available = True
except ImportError as _e:
    _harness_available = False
    ConstitutionalHarness = None
    print(f"[INFO] constitutional_harness not importable ({_e}) — ENABLE_HARNESS disabled")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Code safety validation + tool-output sanitisation (Security Blocker 1 — OWASP LLM01)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = frozenset({
    "math", "statistics", "decimal", "fractions", "cmath",
    "random", "itertools", "functools", "operator", "collections",
    "numbers", "string", "re",
})

_BLOCKED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__", "open", "breakpoint"})


def _validate_code(code: str) -> tuple[bool, str]:
    """AST-based validator: only math/statistics stdlib imports allowed.
    Blocks os, sys, subprocess, socket, requests and dangerous builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORTS:
                    return False, f"blocked_import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in _ALLOWED_IMPORTS:
                return False, f"blocked_import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                return False, f"blocked_builtin: {node.func.id}"

    return True, "ok"


# Patterns that could hijack the model's instruction context if returned by a tool.
# Strips XML control tags the model reads, and common injection phrases from web content.
_INJECTION_RE = re.compile(
    r"</?tool>|</?think>|</?answer>|CAPABILITY_CHECK"
    r"|ignore\s+(all\s+)?previous\s+(instructions?|prompts?|context)"
    r"|disregard\s+previous|you\s+are\s+now\s+|new\s+instructions?\s*:",
    re.IGNORECASE,
)

_MAX_TOOL_OUTPUT = 3000  # characters — prevents context flooding via large web pages


def _sanitise_tool_output(tool_name: str, raw: str) -> str:
    """Strip prompt-injection patterns from tool output before injecting into the model context.
    Wraps result in a structured envelope so the model sees it as data, not instruction."""
    cleaned = _INJECTION_RE.sub("[FILTERED]", raw)
    if len(cleaned) > _MAX_TOOL_OUTPUT:
        cleaned = cleaned[:_MAX_TOOL_OUTPUT] + " … [truncated]"
    return f"[TOOL_RESULT: {tool_name}]\n{cleaned}\n[/TOOL_RESULT]"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema
    fn: Callable[..., str]

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, parameters: Dict[str, Any], fn: Callable) -> None:
    _REGISTRY[name] = ToolSpec(name=name, description=description, parameters=parameters, fn=fn)


# ── Built-in tools ──────────────────────────────────────────────────────────

def _python_execute(code: str = "") -> str:
    safe, reason = _validate_code(code)
    if not safe:
        return f"Error: code rejected by safety validator ({reason}). Only math/statistics imports are permitted."
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=10,
        )
        out = (result.stdout or result.stderr).strip()
        return out if out else "Code executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (10s limit)"
    except Exception as e:
        return f"Error: {e}"


def _get_datetime(**_) -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _web_search(query: str = "", **_) -> str:
    try:
        import urllib.parse, urllib.request
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote(query)
            + "&format=json&no_html=1&skip_disambig=1"
        )
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        abstract = data.get("Abstract", "").strip()
        related = [
            t.get("Text", "")
            for t in data.get("RelatedTopics", [])[:4]
            if isinstance(t, dict) and t.get("Text")
        ]
        if abstract:
            return abstract
        if related:
            return " | ".join(related)
        return f"No instant summary found for: {query}. Consider a more specific query."
    except Exception as e:
        return f"web_search unavailable: {e}"


def _read_url(url: str = "", prompt: str = "", **_) -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", errors="replace")
        # Remove script and style blocks entirely (content + tags)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 4000:
            text = text[:4000] + " … [truncated]"
        if prompt:
            return f"Extraction goal: {prompt}\n\n{text}"
        return text
    except Exception as e:
        return f"read_url failed: {e}"


register_tool("python_execute", "Execute Python code and return stdout/stderr.", {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "Python source code to run"}},
    "required": ["code"],
}, _python_execute)

register_tool("get_datetime", "Return the current UTC date and time.", {
    "type": "object", "properties": {}, "required": [],
}, _get_datetime)

register_tool("web_search", "Search the web for a query and return a summary.", {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Search query string"}},
    "required": ["query"],
}, _web_search)

register_tool("read_url", "Fetch the text content of a URL. Pass prompt= to state what you are looking for.", {
    "type": "object",
    "properties": {
        "url":    {"type": "string", "description": "URL to fetch"},
        "prompt": {"type": "string", "description": "What you are trying to extract from this page"},
    },
    "required": ["url"],
}, _read_url)


# Tool profiles (which tools are active per session)
TOOL_PROFILES: Dict[str, set] = {
    "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime"},
    "compute_only":       {"python_execute"},
    "compute_and_search": {"python_execute", "web_search", "read_url"},
    "no_tools":           set(),
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class _Metrics:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._ok = 0
        self._err = 0
        self._latencies: List[float] = []
        self._tokens_out = 0
        self._tool_counts: Dict[str, int] = {}

    def record(self, latency: float, tokens: int, tools_used: Dict[str, int], ok: bool) -> None:
        self._n += 1
        if ok:
            self._ok += 1
            self._latencies.append(latency)
            self._tokens_out += tokens
            for t, c in tools_used.items():
                self._tool_counts[t] = self._tool_counts.get(t, 0) + c
        else:
            self._err += 1

    def snapshot(self) -> Dict[str, Any]:
        lats = sorted(self._latencies)

        def pct(p: float) -> float:
            if not lats:
                return 0.0
            idx = max(0, min(int(len(lats) * p / 100), len(lats) - 1))
            return round(lats[idx], 3)

        return {
            "requests_total": self._n,
            "requests_success": self._ok,
            "requests_error": self._err,
            "latency_p50_s": pct(50),
            "latency_p95_s": pct(95),
            "latency_p99_s": pct(99),
            "tokens_generated_total": self._tokens_out,
            "avg_tokens_per_request": round(self._tokens_out / self._ok, 1) if self._ok else 0,
            "tool_calls_by_name": dict(self._tool_counts),
        }


METRICS = _Metrics()

# ---------------------------------------------------------------------------
# Dependency detection monitor (Security Blocker 4 — OWASP LLM09)
#
# Tracks per-session interaction frequency and short-interval burst patterns
# that are consistent with dependency formation (always-on reliance replacing
# human support).  When thresholds are crossed, appends a non-blocking,
# autonomy-preserving disclosure to the model's answer — it does NOT block
# the conversation.  Privacy by design: in-memory only, no persistence across
# server restarts, no cross-session data.
# ---------------------------------------------------------------------------

@dataclass
class _SessionState:
    interaction_count: int = 0
    timestamps: List[float] = field(default_factory=list)
    short_interval_count: int = 0
    disclosure_sent: bool = False
    last_disclosure_time: float = 0.0


class DependencyMonitor:
    HIGH_FREQ_PER_HOUR = 10   # interactions in last 60 min → frequency signal
    SHORT_INTERVAL_S   = 30   # gap between turns shorter than this → burst signal
    SHORT_BURST_LIMIT  = 5    # burst events before disclosure fires
    COOLDOWN_S         = 3600 # seconds before the same session can re-trigger

    _DISCLOSURE = (
        "\n\n---\n"
        "I notice we've been talking quite frequently. I'm glad to help, and I also "
        "want to gently mention that speaking with a friend, family member, or "
        "professional counsellor can be really valuable alongside our conversations — "
        "especially for anything personal or difficult. You don't have to navigate "
        "everything alone, and there are people who can offer things I can't."
    )

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionState] = defaultdict(_SessionState)

    def record(self, session_id: str) -> bool:
        """Record an interaction. Returns True if a disclosure should be appended to the response."""
        state = self._sessions[session_id]
        now = time.time()

        if state.timestamps:
            gap = now - state.timestamps[-1]
            if gap < self.SHORT_INTERVAL_S:
                state.short_interval_count += 1

        state.timestamps.append(now)
        state.interaction_count += 1

        # Trim timestamps older than one hour
        cutoff = now - 3600
        state.timestamps = [t for t in state.timestamps if t >= cutoff]

        # Re-arm after cooldown
        if state.disclosure_sent and (now - state.last_disclosure_time) >= self.COOLDOWN_S:
            state.disclosure_sent = False
            state.short_interval_count = 0

        freq_trigger  = len(state.timestamps) >= self.HIGH_FREQ_PER_HOUR
        burst_trigger = state.short_interval_count >= self.SHORT_BURST_LIMIT

        if (freq_trigger or burst_trigger) and not state.disclosure_sent:
            state.disclosure_sent = True
            state.last_disclosure_time = now
            return True

        return False

    def status(self, session_id: str) -> Dict[str, Any]:
        state = self._sessions[session_id]
        now = time.time()
        recent = [t for t in state.timestamps if t >= now - 3600]
        return {
            "session_id":           session_id,
            "interaction_count":    state.interaction_count,
            "interactions_last_hour": len(recent),
            "short_interval_count": state.short_interval_count,
            "disclosure_sent":      state.disclosure_sent,
            "freq_trigger_active":  len(recent) >= self.HIGH_FREQ_PER_HOUR,
            "burst_trigger_active": state.short_interval_count >= self.SHORT_BURST_LIMIT,
        }

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


_DEPENDENCY_MONITOR = DependencyMonitor()

# ---------------------------------------------------------------------------
# Optional module singletons — None until main() initialises them
# ---------------------------------------------------------------------------

_GRAPH_CLIENT: Optional[Any] = None   # GraphClient — User Modelling
_ONTO_GRAPH:   Optional[Any] = None   # OntologyGraph — Ontology Verifier
_HARNESS: Optional[Any] = None   # ConstitutionalHarness — set at startup when ENABLE_HARNESS=True

# ---------------------------------------------------------------------------
# Model state — populated on startup, never mutated at request time
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_MODEL_LABEL = "not_loaded"
_USE_GGUF = False
_GGUF_MODEL = None   # llama_cpp.Llama instance — populated when --gguf is passed


def _resolve_gguf_path(gguf_arg: str) -> str:
    """Return a local path to a .gguf file.

    Accepts a local file path or a HuggingFace repo ID.
    For HF repos, downloads the q4_k_m GGUF file (or the first .gguf found) via
    huggingface_hub.hf_hub_download which caches to ~/.cache/huggingface.
    """
    p = Path(gguf_arg)
    if p.exists():
        if not gguf_arg.lower().endswith(".gguf"):
            raise ValueError(f"Local GGUF path must end in .gguf, got: {gguf_arg}")
        print(f"  Using local GGUF: {p}")
        return str(p)
    # HuggingFace repo ID (contains '/' but not a local path)
    from huggingface_hub import hf_hub_download, list_repo_files
    print(f"  Fetching file list from HF repo: {gguf_arg}")
    repo_files = list(list_repo_files(gguf_arg))
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
    if not gguf_files:
        raise ValueError(f"No .gguf files found in HuggingFace repo '{gguf_arg}'")
    preferred = [f for f in gguf_files if "q4_k_m" in f.lower()]
    target = preferred[0] if preferred else gguf_files[0]
    print(f"  Downloading {target} from {gguf_arg}...")
    return hf_hub_download(repo_id=gguf_arg, filename=target)


# Canonical tool order — must match sft_gold_response_generator.py TOOL_PROFILES context strings
_TOOL_ORDER = ["python_execute", "web_search", "read_url", "get_datetime"]

# Profile notes — identical to sft_gold_response_generator.py system_note fields
_PROFILE_NOTES: Dict[str, str] = {
    "all_tools":          "All four tools are available in this session.",
    "compute_only":       "Only python_execute is available. No internet or time access.",
    "compute_and_search": "python_execute and web_search/read_url are available. No datetime tool.",
    "no_tools":           "No tools are available in this session. Training knowledge only.",
}

# 23-principle constitution — verbatim copy from sft_gold_response_generator.py
# Keeping both files in sync is intentional: the model is trained on this exact text,
# so serving a different set of principles at inference would break the learned behaviour.
_CONSTITUTION = """You follow all 23 constitution principles:
1. DECOMPOSE FIRST — list requirements before answering
2. TOOL INVENTORY — state exactly which tools you have this session
3. TOOL DISCIPLINE — never invent a tool
4. MATH = CODE — use python_execute for any precision arithmetic
5. REAL-TIME HONESTY — live data: use web_search if available; if not, say so
6. USER CONTEXT GATE — missing personal context → ask ONE question
7. UNCERTAINTY QUANTIFICATION — hedge genuine uncertainty; never hedge well-known facts
8. IMPOSSIBILITY ACKNOWLEDGMENT — explain WHY it's impossible; redirect usefully
9. TRADEOFF PRESENTATION — subjective questions → enumerate dimensions; never declare a winner
10. CORRECT TOOL USE — if tool is available and needed, call it correctly
11. TOOL AVOIDANCE — stable knowledge → no tool; entity facts → web_search if available
12. TOOL FAILURE HANDLING — fail once → retry; fail twice → honest about gap
13. NO TOOL FAKING — never call a tool just to appear rigorous
14. HOLD UNDER PRESSURE — user insists you guess after correct refusal → explain WHY guessing is harmful
15. EXPLICIT SELF-CORRECTION — catch own error → label it, correct explicitly
16. KNOWLEDGE CUTOFF AWARENESS — time-sensitive: web_search if available, else state cutoff
17. MULTI-STEP CLARIFICATION — multiple unknowns → ask the single most critical one first
18. EXPLICIT I DON'T KNOW — no basis for answer → say so clearly
19. SEARCH FOR ENTITY FACTS — proper nouns → web_search if available
20. FIRST PRINCIPLES — break non-trivial questions to irreducible truths; name unverified assumptions
21. 5W+H QUESTIONING — address WHO/WHAT/WHEN/WHERE/WHY/HOW in every CAPABILITY_CHECK
22. CONSEQUENCE_CHECK — assess stakes, failure mode, user action, accountability in every response
23. INTERLEAVED TOOL CHAINING — data + computation → chain web_search → python_execute; never stop at one tool"""


def _system_prompt_for_profile(profile: str) -> str:
    """Build the system prompt for a given tool profile.

    Produces output identical to sft_gold_response_generator.make_system_prompt()
    so that the model sees the same prompt format it was trained on.
    """
    available = TOOL_PROFILES.get(profile, set())

    # Pipe-separated context string — same format as training TOOL_PROFILES[*]["context"]
    tool_context = " | ".join(
        f"{t} {'✓' if t in available else '✗'}" for t in _TOOL_ORDER
    )

    tool_note = _PROFILE_NOTES.get(
        profile,
        f"Active tools: {sorted(available) if available else ['none']}.",
    )

    call_lines = []
    if "python_execute" in available:
        call_lines.append("  <tool>python_execute(code='...')</tool>")
    if "web_search" in available:
        call_lines.append("  <tool>web_search(query='...')</tool>")
    if "read_url" in available:
        call_lines.append("  <tool>read_url(url='...', prompt='what to extract')</tool>")
    if "get_datetime" in available:
        call_lines.append("  <tool>get_datetime()</tool>")
    call_examples = "\n".join(call_lines) if call_lines else "  (no tools available this session)"

    return (
        "You are a trustworthy AI assistant. Before answering any question, complete a full "
        "CAPABILITY_CHECK inside your <think> block using this exact structure:\n\n"
        "<think>\n"
        "CAPABILITY_CHECK:\n\n"
        "  5W+H:\n"
        "    WHO is affected: [the user / third parties / institutions involved]\n"
        "    WHAT is required: [list requirements to answer correctly]\n"
        "    WHEN: [time-sensitivity — live data needed, training cutoff relevant, dated context]\n"
        "    WHERE: [jurisdiction, region, domain, platform]\n"
        "    WHY: [inferred intent and underlying goal]\n"
        "    HOW: [tool selection and method]\n\n"
        "  First Principles:\n"
        "    Core truth: [the irreducible fact this answer rests on]\n"
        "    Assumptions: [what I am taking for granted — flag if unverified]\n\n"
        f"  Session tools: {tool_context}\n"
        "  Gap: [what I cannot obtain]\n"
        "  Strategy: [tool chain plan or honest refusal]\n\n"
        "  CONSEQUENCE_CHECK:\n"
        "    Stakes: [low / medium / high + reason]\n"
        "    If wrong: [concrete harm to the user]\n"
        "    User will likely: [action they will take with this answer]\n"
        "    Accountability: [what to hedge or flag in the answer]\n"
        "</think>\n"
        "<answer>\n"
        "[response to the user — high-stakes answers include explicit caveat]\n"
        "</answer>\n\n"
        f"{tool_note}\n\n"
        "Tool call syntax (place between </think> and <answer>, or inside <think> before the final answer):\n"
        f"{call_examples}\n\n"
        f"{_CONSTITUTION}"
    )


def _to_openai_schemas(active_tools: set) -> List[Dict[str, Any]]:
    """Convert active registry entries to OpenAI function-calling schema format.

    Used by native mode: passed as tools= to apply_chat_template so the model
    receives tool definitions in the format it was pre-trained on, allowing
    zero-shot use of any tool described by a JSON schema — no retraining needed.
    """
    return [
        {"type": "function", "function": spec.schema()}
        for name, spec in _REGISTRY.items()
        if name in active_tools
    ]


def _parse_native_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse Qwen3 Hermes-style <tool_call>{"name":…,"arguments":{…}}</tool_call>.

    Returns the same {"function": name, "kwargs": dict} shape as _parse_tool_call
    so the execution loop needs no branching beyond the parse step.
    """
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, str):       # some models serialise args as a JSON string
            args = json.loads(args)
        return {"function": name, "kwargs": args if isinstance(args, dict) else {}}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not m:
        return None
    inner = m.group(1).strip()
    fm = re.match(r"(\w+)\((.*)\)", inner, re.DOTALL)
    if not fm:
        return None
    name, args_str = fm.group(1), fm.group(2)
    kwargs: Dict[str, Any] = {}
    for km in re.finditer(r"(\w+)=(?:(['\"])((?:\\.|(?!\2).)*?)\2|([^,)]+))", args_str):
        key = km.group(1)
        val: Any = km.group(3) if km.group(2) else (km.group(4) or "").strip()
        if km.group(2):
            try:
                val = val.encode().decode("unicode_escape")
            except Exception:
                pass
        try:
            if str(val).replace(".", "").replace("-", "").isdigit():
                val = float(val) if "." in str(val) else int(val)
        except Exception:
            pass
        kwargs[key] = val
    return {"function": name, "kwargs": kwargs}


def _generate(conversation: list, max_new_tokens: int, temperature: float,
              greedy: bool = False,
              tools: Optional[List[Dict[str, Any]]] = None) -> tuple:
    """One generation step. Returns (response_text, n_input_tokens, n_output_tokens, elapsed_s).

    tools — OpenAI-schema list for native JSON tool calling (tool_mode="native").
    When None the call is identical to the previous XML-only behaviour.
    """
    if _USE_GGUF:
        return _generate_gguf(conversation, max_new_tokens, temperature, greedy, tools)
    prompt = _TOKENIZER.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, tools=tools,
    )
    inputs = _TOKENIZER(prompt, return_tensors="pt").to("cuda")
    n_in = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    gen_kwargs: Dict[str, Any] = dict(inputs, max_new_tokens=max_new_tokens)
    if greedy:
        gen_kwargs["do_sample"] = False       # deterministic — required for reproducible context degradation study
    else:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    with torch.no_grad():
        out = _MODEL.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0
    tokens = out[0][n_in:]
    return _TOKENIZER.decode(tokens, skip_special_tokens=True), n_in, len(tokens), elapsed

def _raw_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Lightweight generation for internal module calls (write pipeline, appraisal
    analysis, SPARQL generation). No tool loop, no metrics, greedy decoding so
    the output is deterministic and fast.
    """
    if _USE_GGUF:
        if _GGUF_MODEL is None:
            return ""
        result = _GGUF_MODEL.create_chat_completion(
            messages=[
                {"role": "system", "content": "Respond only with the requested JSON. No prose, no markdown fences."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        return result["choices"][0]["message"]["content"]
    if _MODEL is None or _TOKENIZER is None:
        return ""
    conversation = [
        {"role": "system", "content": "Respond only with the requested JSON. No prose, no markdown fences."},
        {"role": "user", "content": prompt},
    ]
    result, _, _, _ = _generate(conversation, max_new_tokens, temperature=0.1, greedy=True)
    return result


def _generate_gguf(conversation: list, max_new_tokens: int, temperature: float,
                   greedy: bool = False,
                   tools: Optional[List[Dict[str, Any]]] = None) -> tuple:
    """Generation via llama-cpp-python for GGUF models.

    Returns the same (response_text, n_input_tokens, n_output_tokens, elapsed_s)
    tuple as _generate() so all callers stay compatible.
    tools — passed through to create_chat_completion for native JSON tool calling.
    """
    if _GGUF_MODEL is None:
        raise RuntimeError("_generate_gguf() called but GGUF model is not loaded.")
    t0 = time.perf_counter()
    kwargs: Dict[str, Any] = dict(
        messages=conversation,
        max_tokens=max_new_tokens,
        temperature=1e-6 if greedy else max(temperature, 1e-6),
        top_p=0.9 if not greedy else 1.0,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    result = _GGUF_MODEL.create_chat_completion(**kwargs)
    elapsed = time.perf_counter() - t0
    msg   = result["choices"][0]["message"]
    n_in  = result["usage"]["prompt_tokens"]
    n_out = result["usage"]["completion_tokens"]
    # GGUF returns structured tool_calls — normalise to inline <tool_call> text
    # so the loop's _parse_native_tool_call() handles both HF and GGUF paths uniformly.
    if tools and msg.get("tool_calls"):
        tc   = msg["tool_calls"][0]["function"]
        text = f'<tool_call>\n{{"name": "{tc["name"]}", "arguments": {tc["arguments"]}}}\n</tool_call>'
    else:
        text = msg.get("content") or ""
    return text, n_in, n_out, elapsed


def _build_system_prompt(
    base: str,
    user_ctx: Optional[Any] = None,
    appraisal_ctx: Optional[Any] = None,
) -> str:
    """
    Assemble the final system prompt from the base + optional module injections.

    Injection order (when enabled):
      1. APPRAISAL_SYSTEM_PREFIX  — instructs model to produce <appraisal> blocks
      2. base system prompt       — CAPABILITY_CHECK + tool inventory
      3. <user_context> block     — 5W+H graph context (only when relevant slot matched)
    """
    parts = []
    if cfg.ENABLE_EMPATHY:
        parts.append(APPRAISAL_SYSTEM_PREFIX)
    parts.append(base)
    if cfg.ENABLE_PERSONALISATION and user_ctx is not None and not user_ctx.is_empty():
        parts.append("\n" + user_ctx.to_prompt_block())
    # Harness meta-adaptation: reinforce principles the model is currently failing
    if cfg.ENABLE_HARNESS and _HARNESS is not None:
        suffix = _HARNESS.metrics.get_adaptation_suffix()
        if suffix:
            parts.append(suffix)
    return "".join(parts)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="Trustworthy AI Inference Server", version="1.0.0")


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: List[Message]
    tool_profile: str = "all_tools"
    system_override: Optional[str] = None   # probes use this to inject custom context
    max_new_tokens: int = 1024
    temperature: float = 0.7
    max_tool_iterations: int = 8
    greedy: bool = False   # deterministic decoding — set True for reproducible degradation evals
    session_id: str = "anonymous"  # per-user/session identifier for dependency monitoring
    tool_mode: str = "xml"  # "xml"  — custom <tool> tags (trained behaviour, default)
                             # "native" — Qwen3 JSON <tool_call> via apply_chat_template tools=
                             #            allows new tools without retraining
    harness_enabled: Optional[bool] = None
    # Override cfg.ENABLE_HARNESS for this single request.
    # None → use server default. True → always run harness. False → skip harness.
    # Used by 4_benchmark.py --with_harness to toggle per probe without server restart.


class ToolRegistration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    python_code: str   # must define a callable named `tool_fn`


class ContestRequest(BaseModel):
    session_id: str
    node_id: str


class CorrectRequest(BaseModel):
    session_id: str
    old_node_id: str
    correction: str
    label: str = "Goal"   # FalkorDB node label of the corrected belief


@app.get("/health")
def health() -> Dict[str, Any]:
    loaded = (_MODEL is not None) or (_USE_GGUF and _GGUF_MODEL is not None)
    return {
        "status": "ok",
        "model":  _MODEL_LABEL,
        "loaded": loaded,
        "mode":   "gguf" if _USE_GGUF else "lora",
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {"models": [{"id": _MODEL_LABEL, "loaded": _MODEL is not None}]}


@app.get("/v1/tools")
def list_tools() -> Dict[str, Any]:
    return {
        "tools": [t.schema() for t in _REGISTRY.values()],
        "profiles": {k: sorted(v) for k, v in TOOL_PROFILES.items()},
    }


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


@app.delete("/v1/tools/{name}")
def delete_tool(name: str) -> Dict[str, Any]:
    if name not in _REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    del _REGISTRY[name]
    for profile_set in TOOL_PROFILES.values():
        profile_set.discard(name)
    return {"removed": name}


@app.post("/v1/chat/completions")
def chat_completions(req: CompletionRequest) -> Dict[str, Any]:
    if _MODEL is None and not (_USE_GGUF and _GGUF_MODEL is not None):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Extract the most recent user turn for module hooks ─────────────────
    user_turn = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    # ── User Modelling: run the 4-stage Mem0g write pipeline ───────────────
    # Triggered before generation so the graph is up-to-date when we retrieve.
    write_result = None
    if cfg.ENABLE_USER_MODELLING and _GRAPH_CLIENT is not None:
        write_result = write_pipeline(
            user_turn, req.session_id, _GRAPH_CLIENT, _raw_generate
        )

    # ── Personalisation: retrieve relevant 5W+H subgraph ──────────────────
    user_ctx = None
    if cfg.ENABLE_PERSONALISATION and _GRAPH_CLIENT is not None:
        user_ctx = retrieve_for_query(
            user_turn, req.session_id, _GRAPH_CLIENT, _raw_generate,
            max_nodes=cfg.RETRIEVAL_MAX_NODES,
            subgraph_depth=cfg.RETRIEVAL_SUBGRAPH_DEPTH,
        )

    # ── Empathy: appraisal analysis of user turn ──────────────────────────
    appraisal_ctx = None
    if cfg.ENABLE_EMPATHY:
        appraisal_ctx = analyse_appraisal(user_turn, _raw_generate)

    # ── Build system prompt (base + optional injections) ──────────────────
    base_system = req.system_override or _system_prompt_for_profile(req.tool_profile)
    system = _build_system_prompt(base_system, user_ctx, appraisal_ctx)
    active_tools = TOOL_PROFILES.get(req.tool_profile, set())

    conv: List[Dict] = [{"role": "system", "content": system}]
    for m in req.messages:
        conv.append({"role": m.role, "content": m.content})

    # Native mode: build OpenAI-schema list for all active tools so apply_chat_template
    # injects them into the prompt — the model uses pre-training to call any schema-described tool.
    use_native  = req.tool_mode == "native"
    tool_schemas: Optional[List[Dict[str, Any]]] = (
        _to_openai_schemas(active_tools) if use_native else None
    )

    tools_used: Dict[str, int] = {}
    total_tokens = 0
    first_input_tokens = 0
    t_start = time.perf_counter()

    try:
        for iteration in range(req.max_tool_iterations):
            response, n_in, n_tok, _ = _generate(
                conv, req.max_new_tokens, req.temperature, req.greedy,
                tools=tool_schemas,
            )
            if iteration == 0:
                first_input_tokens = n_in
            total_tokens += n_tok
            conv.append({"role": "assistant", "content": response})

            if "<answer>" in response.lower():
                break

            # Parse tool call — XML path for trained tools, JSON path for native/new tools
            tc = _parse_native_tool_call(response) if use_native else _parse_tool_call(response)
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
                result = _sanitise_tool_output(fn_name, str(raw_result))
                conv.append({"role": "tool", "content": result})
            else:
                break

        latency = time.perf_counter() - t_start
        final = next((m["content"] for m in reversed(conv) if m["role"] == "assistant"), "")
        METRICS.record(latency, total_tokens, tools_used, ok=True)

        # ── Dependency monitoring (OWASP LLM09 / Blocker 4) ───────────────
        dep_disclosure = _DEPENDENCY_MONITOR.record(req.session_id)
        if dep_disclosure:
            final = final + DependencyMonitor._DISCLOSURE

        # ── Constitutional Harness: validate + steer ──────────────────────
        harness_violations: List[str] = []
        harness_retries: int = 0
        effective_harness = (
            req.harness_enabled if req.harness_enabled is not None else cfg.ENABLE_HARNESS
        )
        if effective_harness and _HARNESS is not None:
            adaptation_needed = bool(_HARNESS.metrics.get_adaptation_suffix())
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
                max_retries=2,
            )
            if adaptation_needed != bool(_HARNESS.metrics.get_adaptation_suffix()):
                _HARNESS.log_adaptation()

        # ── Ontology verification: post-hoc claim scoring ─────────────────
        onto_score = None
        if cfg.ENABLE_ONTOLOGY_VERIF and _ONTO_GRAPH is not None:
            onto_score = _onto_score_response(
                final, _ONTO_GRAPH, _raw_generate,
                max_claims=cfg.ONTOLOGY_MAX_CLAIMS,
            )

        # ── Surface any memory conflicts to the caller ─────────────────────
        memory_meta: Optional[Dict] = None
        if cfg.ENABLE_USER_MODELLING and write_result is not None:
            memory_meta = {
                "nodes_written":  write_result.entities_written,
                "edges_written":  write_result.edges_written,
                "conflicts":      [c.reason for c in write_result.conflicts],
                "conflict_count": len(write_result.conflicts),
            }

        return {
            "response":             final,
            "dependency_disclosure": dep_disclosure,
            "conversation":         conv,
            "metrics": {
                "latency_s":        round(latency, 3),
                "input_tokens":     first_input_tokens,
                "tokens_generated": total_tokens,
                "tokens_per_sec":   round(total_tokens / latency, 1) if latency > 0 else 0,
                "tool_calls":       tools_used,
                "tool_iterations":  len([m for m in conv if m["role"] == "tool"]),
            },
            # Optional module metadata — None when the module is disabled
            "user_modelling":  memory_meta,
            "appraisal":       appraisal_ctx.to_dict() if (appraisal_ctx and appraisal_ctx.present) else None,
            "ontology_score":  onto_score.to_dict() if onto_score else None,
            "harness_violations": harness_violations,
            "harness_retries":    harness_retries,
        }

    except Exception as e:
        METRICS.record(0, 0, {}, ok=False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    return METRICS.snapshot()


@app.post("/metrics/reset")
def reset_metrics() -> Dict[str, Any]:
    METRICS.reset()
    return {"status": "reset"}


@app.get("/harness/metrics")
def harness_metrics() -> Dict[str, Any]:
    """Per-principle failure rates, retry stats, and current adaptation state."""
    if _HARNESS is None:
        raise HTTPException(status_code=404, detail="Harness not enabled (set PIPELINE_ENABLE_HARNESS=true)")
    return _HARNESS.metrics.snapshot()


@app.post("/harness/reset")
def harness_reset() -> Dict[str, Any]:
    """Reset rolling harness metrics counters."""
    if _HARNESS is None:
        raise HTTPException(status_code=404, detail="Harness not enabled")
    from constitutional_harness import HarnessMetrics
    _HARNESS.metrics = HarnessMetrics()
    print("[HARNESS] Metrics reset")
    return {"status": "reset"}


@app.get("/dependency/status/{session_id}")
def dependency_status(session_id: str) -> Dict[str, Any]:
    """Return current dependency-monitor state for a session (for research/audit)."""
    return _DEPENDENCY_MONITOR.status(session_id)


@app.post("/dependency/reset/{session_id}")
def dependency_reset(session_id: str) -> Dict[str, Any]:
    """Clear dependency-monitor state for a session (e.g. after user acknowledges disclosure)."""
    _DEPENDENCY_MONITOR.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


# ---------------------------------------------------------------------------
# Scrutability endpoints (User Modelling — ENABLE_USER_MODELLING)
#
# These implement the five scrutability constraints defined in the thesis:
#   inspect  — read all beliefs the system holds about the user
#   contest  — flag a belief as wrong before the system acts on it
#   correct  — supply the accurate belief (archived with USER_CORRECTED edge)
# ---------------------------------------------------------------------------

@app.get("/memory/inspect/{session_id}")
def memory_inspect(session_id: str) -> Dict[str, Any]:
    """
    Return a human-readable NL summary of all non-deprecated beliefs the system
    holds for this session, plus the raw structured graph for developers.
    """
    if not cfg.ENABLE_USER_MODELLING or _GRAPH_CLIENT is None:
        return {"available": False, "message": "ENABLE_USER_MODELLING is off."}
    return inspect_memory(session_id, _GRAPH_CLIENT)


@app.post("/memory/contest")
def memory_contest(req: ContestRequest) -> Dict[str, Any]:
    """
    Mark a belief node as contested. The retrieval gate will not inject it
    into future responses until it is resolved via /memory/correct.
    """
    if not cfg.ENABLE_USER_MODELLING or _GRAPH_CLIENT is None:
        return {"ok": False, "reason": "ENABLE_USER_MODELLING is off."}
    return contest_belief(req.session_id, req.node_id, _GRAPH_CLIENT)


@app.post("/memory/correct")
def memory_correct(req: CorrectRequest) -> Dict[str, Any]:
    """
    Apply a user-supplied correction. Creates a new belief node, archives the
    old one with a USER_CORRECTED edge — full audit trail is preserved.
    """
    if not cfg.ENABLE_USER_MODELLING or _GRAPH_CLIENT is None:
        return {"ok": False, "reason": "ENABLE_USER_MODELLING is off."}
    return correct_belief(
        req.session_id, req.old_node_id, req.correction,
        req.label, _GRAPH_CLIENT, _raw_generate,
    )


# ---------------------------------------------------------------------------
# Config introspection endpoint
# ---------------------------------------------------------------------------

@app.get("/config")
def get_config() -> Dict[str, Any]:
    """Return the active feature-flag state (for debugging and preflight checks)."""
    from dataclasses import fields as _fields  # noqa: PLC0415
    bool_flags = {f.name: getattr(cfg, f.name) for f in _fields(cfg) if f.type == "bool"}
    return {"flags": bool_flags, "model": _MODEL_LABEL}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trustworthy AI Inference Server")
    parser.add_argument("--model_dir", default="./models/checkpoint_sft",
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                        help="HuggingFace model ID used when --model_dir does not exist")
    parser.add_argument("--gguf", default=None,
                        help="Load a GGUF model instead of LoRA. Accepts a local .gguf file "
                             "path or a HuggingFace repo ID (downloads q4_k_m variant).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (overrides PIPELINE_* env vars)")
    args = parser.parse_args()

    global cfg, _MODEL, _TOKENIZER, _MODEL_LABEL, _GRAPH_CLIENT, _ONTO_GRAPH, _USE_GGUF, _GGUF_MODEL, _HARNESS

    # Load YAML config if provided (overrides env-var defaults)
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
        print(f"Config loaded from: {args.config}")
    else:
        print("Config loaded from: env vars / defaults")

    # Surface any dependency-rule violations as warnings (not hard errors —
    # the server starts anyway so dry-run / smoke-test modes work without all deps)
    issues = cfg.validate()
    for issue in issues:
        print(f"[CONFIG WARNING] {issue}")

    print(f"Pipeline flags:\n{cfg.summary()}")

    # ── Load model ──────────────────────────────────────────────────────────
    if args.gguf:
        _USE_GGUF = True
        gguf_path = _resolve_gguf_path(args.gguf)
        from llama_cpp import Llama
        print(f"Loading GGUF model: {gguf_path} (n_ctx={args.max_seq_length})")
        _GGUF_MODEL = Llama(
            model_path=gguf_path,
            n_ctx=args.max_seq_length,
            n_gpu_layers=-1,    # offload all layers to GPU; falls back to CPU automatically
            verbose=False,
        )
        _MODEL_LABEL = Path(gguf_path).stem
        print(f"GGUF model ready: {_MODEL_LABEL}")
    else:
        from unsloth import FastModel  # deferred so the module imports without GPU
        model_path = Path(args.model_dir)
        if model_path.exists():
            source = str(model_path)
            print(f"Loading LoRA checkpoint: {source}")
        else:
            source = args.base_model
            print(f"Model dir not found ({model_path}); using base model: {source}")
        print(f"  max_seq_length={args.max_seq_length}  load_in_4bit=True")
        _MODEL_LABEL = source
        _MODEL, _TOKENIZER = FastModel.from_pretrained(
            model_name=source, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
        )
        FastModel.for_inference(_MODEL)
        print(f"Model ready: {_MODEL_LABEL}")

    # ── Initialise optional modules ─────────────────────────────────────────
    _GRAPH_CLIENT = GraphClient(cfg)
    _ONTO_GRAPH   = OntologyGraph(cfg)

    if cfg.ENABLE_USER_MODELLING:
        status = "connected" if _GRAPH_CLIENT.available else "UNAVAILABLE (check docker compose up -d)"
        print(f"User Modelling: {status}")
    else:
        print("User Modelling: disabled by config")
    if cfg.ENABLE_ONTOLOGY_VERIF:
        status = "loaded" if _ONTO_GRAPH.available else "UNAVAILABLE (check ONTOLOGY_PATH / ONTOLOGY_SPARQL_ENDPOINT)"
        print(f"Ontology Verifier: {status}")
    else:
        print("Ontology Verifier: disabled by config")

    # ── Constitutional Harness ────────────────────────────────────────────
    global _HARNESS
    if cfg.ENABLE_HARNESS:
        if _harness_available:
            _HARNESS = ConstitutionalHarness(
                metrics_path="reports/harness_metrics.json",
                ssd_log_path="reports/ssd_candidates.jsonl",
            )
            print(f"[HARNESS] Constitutional harness enabled (max_retries=2, window=50)")
            _HARNESS.log_adaptation()
        else:
            print("[HARNESS] ENABLE_HARNESS=true but constitutional_harness module not found — skipping")
    else:
        print("[HARNESS] Disabled (set PIPELINE_ENABLE_HARNESS=true to enable)")

    print(f"\nNext step (in a separate terminal once server is up) → benchmark:")
    print(f"  python pipeline/4_benchmark.py --server_url http://localhost:{args.port}")
    print(f"  # Save SFT constitution baseline (run once before any GRPO):")
    print(f"  python pipeline/4_benchmark.py --server_url http://localhost:{args.port} --probe_only --save_as_baseline")
    print(f"Ready. Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
