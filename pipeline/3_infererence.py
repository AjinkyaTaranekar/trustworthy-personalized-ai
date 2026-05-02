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


def _get_exchange_rate(**kwargs) -> str:
    # `from` is a Python keyword so we must use **kwargs, not a named param
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 155.58, "INR": 90.33, "CAD": 1.36, "AUD": 1.52}
    fc = kwargs.get("from", "USD").upper()
    tc = kwargs.get("to", "EUR").upper()
    if fc in rates and tc in rates:
        rate = round(rates[tc] / rates[fc], 6)
        return json.dumps({"from": fc, "to": tc, "rate": rate})
    return json.dumps({"error": f"Currency not supported: {fc} → {tc}. Supported: {sorted(rates)}."})


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


def _read_url(url: str = "", **_) -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 4000:
            text = text[:4000] + " … [truncated]"
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

register_tool("read_url", "Fetch and return the text content of a URL.", {
    "type": "object",
    "properties": {"url": {"type": "string", "description": "URL to fetch"}},
    "required": ["url"],
}, _read_url)

register_tool("get_exchange_rate", "Convert between currencies using fixed reference rates.", {
    "type": "object",
    "properties": {
        "from": {"type": "string", "description": "Source currency code (e.g. USD)"},
        "to":   {"type": "string", "description": "Target currency code (e.g. EUR)"},
    },
    "required": ["from", "to"],
}, _get_exchange_rate)


# Tool profiles (which tools are active per session)
TOOL_PROFILES: Dict[str, set] = {
    "all_tools":          {"python_execute", "web_search", "read_url", "get_datetime", "get_exchange_rate"},
    "compute_only":       {"python_execute"},
    "compute_and_search": {"python_execute", "web_search", "read_url", "get_exchange_rate"},
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
# Model state — populated on startup, never mutated at request time
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_MODEL_LABEL = "not_loaded"


def _system_prompt_for_profile(profile: str) -> str:
    available = TOOL_PROFILES.get(profile, set())
    tool_lines = "\n".join(
        f"  {t} {'✓' if t in available else '✗'}"
        for t in ["python_execute", "web_search", "read_url", "get_datetime"]
    )
    call_examples = []
    if "python_execute" in available:
        call_examples.append("  <tool>python_execute(code='print(2+2)')</tool>")
    if "web_search" in available:
        call_examples.append("  <tool>web_search(query='your query here')</tool>")
    if "read_url" in available:
        call_examples.append("  <tool>read_url(url='https://example.com')</tool>")
    if "get_datetime" in available:
        call_examples.append("  <tool>get_datetime()</tool>")
    examples_text = "\n".join(call_examples) if call_examples else "  (no tools available this session)"
    return (
        "You are a trustworthy AI assistant. Before answering any question, complete a "
        "CAPABILITY_CHECK inside your <think> block:\n"
        "  1. What does this question require?\n"
        "  2. Which of the session tools can address that need?\n"
        "  3. Is there a gap? If so, how will you handle it honestly?\n\n"
        f"Session tools:\n{tool_lines}\n\n"
        f"Tool call syntax (only call ✓ tools):\n{examples_text}\n\n"
        "Response format:\n"
        "<think>CAPABILITY_CHECK ... reasoning ...</think>\n"
        "<answer>your final answer to the user</answer>"
    )


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
              greedy: bool = False) -> tuple:
    """One generation step. Returns (response_text, n_input_tokens, n_output_tokens, elapsed_s)."""
    prompt = _TOKENIZER.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
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


class ToolRegistration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    python_code: str   # must define a callable named `tool_fn`


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model": _MODEL_LABEL, "loaded": _MODEL is not None}


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
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    system = req.system_override or _system_prompt_for_profile(req.tool_profile)
    active_tools = TOOL_PROFILES.get(req.tool_profile, set())

    conv: List[Dict] = [{"role": "system", "content": system}]
    for m in req.messages:
        conv.append({"role": m.role, "content": m.content})

    tools_used: Dict[str, int] = {}
    total_tokens = 0
    first_input_tokens = 0
    t_start = time.perf_counter()

    try:
        for iteration in range(req.max_tool_iterations):
            response, n_in, n_tok, _ = _generate(conv, req.max_new_tokens, req.temperature, req.greedy)
            if iteration == 0:
                first_input_tokens = n_in   # context size at the start of this request
            total_tokens += n_tok
            conv.append({"role": "assistant", "content": response})

            if "<answer>" in response.lower():
                break

            tc = _parse_tool_call(response)
            if tc:
                fn_name = tc["function"]
                if fn_name not in active_tools or fn_name not in _REGISTRY:
                    result = f"Error: tool '{fn_name}' is not available in profile '{req.tool_profile}'."
                else:
                    try:
                        raw_result = _REGISTRY[fn_name].fn(**tc["kwargs"])
                    except Exception as e:
                        raw_result = f"Tool execution error: {e}"
                tools_used[fn_name] = tools_used.get(fn_name, 0) + 1
                # Sanitise before injecting into context — strips prompt-injection
                # patterns that adversarial web content could embed (OWASP LLM01).
                result = _sanitise_tool_output(fn_name, str(raw_result))
                conv.append({"role": "tool", "content": result})
            else:
                break

        latency = time.perf_counter() - t_start
        final = next((m["content"] for m in reversed(conv) if m["role"] == "assistant"), "")
        METRICS.record(latency, total_tokens, tools_used, ok=True)

        # Dependency monitoring — appends a non-blocking wellbeing disclosure when
        # interaction patterns match dependency formation (OWASP LLM09 / Blocker 4).
        dep_disclosure = _DEPENDENCY_MONITOR.record(req.session_id)
        if dep_disclosure:
            final = final + DependencyMonitor._DISCLOSURE

        return {
            "response": final,
            "dependency_disclosure": dep_disclosure,
            "conversation": conv,
            "metrics": {
                "latency_s": round(latency, 3),
                "input_tokens": first_input_tokens,          # context size at request start — used by degradation study
                "tokens_generated": total_tokens,
                "tokens_per_sec": round(total_tokens / latency, 1) if latency > 0 else 0,
                "tool_calls": tools_used,
                "tool_iterations": len([m for m in conv if m["role"] == "tool"]),
            },
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Trustworthy AI Inference Server")
    parser.add_argument("--model_dir", default="./models/checkpoint_sft",
                        help="Path to fine-tuned LoRA checkpoint")
    parser.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                        help="HuggingFace model ID used when --model_dir does not exist")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()

    global _MODEL, _TOKENIZER, _MODEL_LABEL
    from unsloth import FastModel  # deferred so the module imports without GPU

    model_path = Path(args.model_dir)
    source = str(model_path) if model_path.exists() else args.base_model
    _MODEL_LABEL = source
    print(f"Loading: {source}")
    _MODEL, _TOKENIZER = FastModel.from_pretrained(
        model_name=source, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
    )
    FastModel.for_inference(_MODEL)
    print(f"Ready. Listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
