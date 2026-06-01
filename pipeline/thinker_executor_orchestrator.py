#!/usr/bin/env python
"""thinker_executor_orchestrator.py — two-model inference loop for the Thinker–Executor experiment.

Loads two LoRA adapters (Thinker + Executor) on a shared Qwen3-0.6B base and runs the
orchestration loop that the dual-SFT training set was built for:

    Thinker  (prose <think> + EXACTLY ONE of <ask>/<act>/<answer>; never tool-call syntax)
        ├─ <answer> → return to the user (done)
        ├─ <ask>    → surface the question to the user and await their reply
        └─ <act>    → hand the plain-language instruction to the Executor
                          Executor → EXACTLY ONE native <tool_call>{json}</tool_call>
                          ToolRegistry executes it (python_execute / web_search / read_url / get_datetime)
                          the result is fed back into the Thinker context as a `tool` turn
                      → loop until the Thinker answers (or max_steps)

Single source of truth: the served system prompts (`THINKER_STUDENT_PROMPT`,
`EXECUTOR_STUDENT_PROMPT`), the `[USER MEMORY]` block (`prepend_memory`), and the tool
registry are imported from the same modules the training data was stamped with, so inference
and training are byte-identical.

Memory is the Thinker's alone (prose-only — it never calls a tool): the orchestrator injects
the user's 5W+H profile as a `[USER MEMORY]` block on the user turn, exactly as in training.
An empty profile becomes the explicit "(no profile stored)" block (cold-start regime).

Model loading uses PEFT multi-adapter on ONE base (not two unsloth instances): the base loads
once and the two LoRA adapters are switched per generation step (`set_adapter`). For 0.6B this
fits any GPU; pass --load_in_4bit to shrink further.

Entry points:
    python thinker_executor_orchestrator.py --self_test          # CPU, no GPU/model — validates the loop
    python thinker_executor_orchestrator.py --question "..."      # one-shot
    python thinker_executor_orchestrator.py --chat                # interactive REPL (handles <ask>)
    python thinker_executor_orchestrator.py --serve --port 8000   # FastAPI, /v1/chat/completions for 4_benchmark.py

Models (each is a local dir OR a HuggingFace repo id):
    --thinker  models/checkpoint_thinker   (or AjinkyaTaranekar/trustworthy-ai-thinker)
    --executor models/checkpoint_executor
    --base_model unsloth/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from sft_v3_generator import (
    THINKER_STUDENT_PROMPT,
    EXECUTOR_STUDENT_PROMPT,
    prepend_memory,
)
from pipeline_tools import ToolRegistry

# Executor-owned tools (delegated via <act>) + get_datetime, which is advertised in the
# Executor's schema list. check_profile=False at call time — the model picks the tool.
EXECUTOR_TOOLS = {"python_execute", "web_search", "read_url", "get_datetime"}
MAX_RESULT_CHARS = 2000   # mirror the assembler's MAX_RESULT_CHARS so the Thinker sees results as in training

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_THINK_RE    = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ASK_RE      = re.compile(r"<ask>(.*?)</ask>", re.DOTALL | re.IGNORECASE)
_ACT_RE      = re.compile(r"<act>(.*?)</act>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Capture the Hermes body up to the </tool_call> TAG (not \{.*?\}) and parse strict=False —
# python_execute `code` carries inner braces and raw newlines. Same fix as the splitter /
# 3_infererence.py _parse_native_tool_call (the brace-delimited+strict version dropped every
# python_execute call).
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_native_tool_call(text: str) -> Optional[dict]:
    """Parse Qwen3 Hermes <tool_call>{"name":…,"arguments":{…}}</tool_call> → {function, kwargs}."""
    m = _TOOLCALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1), strict=False)
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, str):           # some models serialise arguments as a JSON string
            args = json.loads(args, strict=False)
        return {"function": name, "kwargs": args if isinstance(args, dict) else {}}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def parse_thinker(text: str) -> tuple[Optional[str], str]:
    """Reduce a Thinker turn to (kind, content) where kind is the EARLIEST of ask/act/answer.
    Returns (None, full_text) if no recognised tag is present (defensive — treated as answer)."""
    best: Optional[tuple[str, str, int]] = None
    for kind, rx in (("ask", _ASK_RE), ("act", _ACT_RE), ("answer", _ANSWER_RE)):
        m = rx.search(text)
        if m and (best is None or m.start() < best[2]):
            best = (kind, m.group(1).strip(), m.start())
    if best:
        return best[0], best[1]
    return None, text.strip()


def extract_think(text: str) -> str:
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""


def sanitise_result(raw: str, tool_name: str = "") -> str:
    """Default sanitiser: length-truncation only. The server injects its own
    sanitiser (injection-stripping + per-tool budgets) via ThinkerExecutor(sanitiser=...)."""
    s = (raw or "").strip()
    if len(s) > MAX_RESULT_CHARS:
        s = s[:MAX_RESULT_CHARS] + f"\n…[truncated {len(s) - MAX_RESULT_CHARS} chars]"
    return s


def looks_like_error(raw: str) -> bool:
    """Heuristic: did a tool result come back as an error? Used to flag trace entries."""
    head = (raw or "")[:60].lower()
    return head.startswith("error") or "error:" in head or "traceback" in head


def openai_schemas(registry: ToolRegistry, names: set[str] = EXECUTOR_TOOLS) -> list[dict]:
    """Build the OpenAI-schema tool list the Executor was trained with (4 tools, same shape
    as the `native_tools` metadata stamped onto every Executor row)."""
    out = []
    for n in sorted(names):
        spec = registry._specs.get(n)
        if spec:
            out.append({"type": "function", "function": spec.schema()})
    return out


# ---------------------------------------------------------------------------
# Generation backends (injectable so the loop is testable without a GPU)
# ---------------------------------------------------------------------------
class Generator:
    """Abstract two-role generator. role ∈ {'thinker','executor'}."""

    def generate(self, role: str, conversation: list, tools: Optional[list],
                 max_new_tokens: int, temperature: float, greedy: bool) -> str:
        raise NotImplementedError


class ScriptedGenerator(Generator):
    """Returns queued canned outputs per role — used by --self_test to exercise the control
    flow, parsing, and real tool execution on CPU with no model loaded."""

    def __init__(self, thinker_outputs: list[str], executor_outputs: list[str]) -> None:
        self._t = list(thinker_outputs)
        self._e = list(executor_outputs)

    def generate(self, role, conversation, tools, max_new_tokens, temperature, greedy):
        q = self._t if role == "thinker" else self._e
        return q.pop(0) if q else "<answer>(scripted output exhausted)</answer>"


class HFGenerator(Generator):
    """Real backend: ONE Qwen3-0.6B base + two PEFT LoRA adapters switched per step."""

    def __init__(self, base_model: str, thinker: str, executor: str,
                 load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.torch = torch

        # The LoRA checkpoint dirs ship the tokenizer; fall back to base if not.
        try:
            self.tok = AutoTokenizer.from_pretrained(thinker)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(base_model)

        load_kwargs: dict[str, Any] = dict(torch_dtype="auto")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            )
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        print(f"[load] base {base_model} ready; attaching adapters…", flush=True)
        model = PeftModel.from_pretrained(base, thinker, adapter_name="thinker")
        model.load_adapter(executor, adapter_name="executor")
        if torch.cuda.is_available() and not load_in_4bit:
            model = model.to("cuda")
        model.eval()
        self.model = model
        print(f"[load] adapters ready: thinker={thinker}  executor={executor}", flush=True)

    def generate(self, role, conversation, tools, max_new_tokens, temperature, greedy):
        self.model.set_adapter("thinker" if role == "thinker" else "executor")
        prompt = self.tok.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            tools=tools, enable_thinking=True,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = dict(**inputs, max_new_tokens=max_new_tokens)
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        with self.torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class ThinkerExecutor:
    def __init__(self, generator: Generator, registry: Optional[ToolRegistry] = None,
                 max_steps: int = 6, thinker_max_tokens: int = 1536,
                 executor_max_tokens: int = 512, temperature: float = 0.7,
                 verbose: bool = False,
                 sanitiser: Optional[Callable[[str, str], str]] = None) -> None:
        self.gen = generator
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.tmax = thinker_max_tokens
        self.emax = executor_max_tokens
        self.temperature = temperature
        self.verbose = verbose
        # sanitiser(raw, tool_name) -> str. Defaults to length-only truncation. The inference
        # server passes its injection-stripping `_sanitise_tool_output` so the Thinker sees the
        # same filtered tool output the single-model server enforces (feature parity).
        self.sanitiser: Callable[[str, str], str] = sanitiser or sanitise_result
        self.schemas = openai_schemas(self.registry)

    def _log(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr, flush=True)

    def _run_executor(self, instruction: str) -> dict:
        """Delegate one <act> instruction to the Executor + tool registry.

        Returns a rich record: call (parsed tool call | None), executor_raw, tool (fn name),
        output_full (untruncated tool output), output_model (sanitised, what the Thinker sees),
        is_error, gen_ms (Executor generation), tool_ms (tool execution)."""
        exec_conv = [
            {"role": "system", "content": EXECUTOR_STUDENT_PROMPT},
            {"role": "user", "content": instruction},
        ]
        # Greedy — the Executor is a deterministic one-tool-call transducer.
        t_gen = time.perf_counter()
        exec_raw = self.gen.generate("executor", exec_conv, self.schemas, self.emax, 0.0, greedy=True)
        gen_ms = round((time.perf_counter() - t_gen) * 1000)
        call = parse_native_tool_call(exec_raw)
        tool_name = (call or {}).get("function", "")
        if call is None:
            raw_full = ("Error: the execution system did not return a valid tool call. "
                        f"Raw output: {exec_raw[:150]}")
            is_error, tool_ms = True, 0
        else:
            t_tool = time.perf_counter()
            raw_full = str(self.registry.call(call["function"], call["kwargs"],
                                              EXECUTOR_TOOLS, check_profile=False))
            tool_ms = round((time.perf_counter() - t_tool) * 1000)
            is_error = looks_like_error(raw_full)
        return {
            "call": call, "executor_raw": exec_raw, "tool": tool_name,
            "output_full": raw_full, "output_model": self.sanitiser(raw_full, tool_name),
            "is_error": is_error, "gen_ms": gen_ms, "tool_ms": tool_ms,
        }

    def run(self, question: str, memory_text: str = "", history: Optional[list] = None,
            session_id: str = "anonymous") -> dict:
        """Run one user turn through the loop.

        history — prior Thinker messages (user/assistant/tool dicts), WITHOUT the system turn,
                  to continue a conversation (e.g. resolving an <ask>). None for a fresh turn.

        Returns a dict with: response (final Thinker turn, full text incl <think>), type
        ('answer'|'ask'), answer (the tag content), think_content, tool_trace, steps, conversation.
        """
        self.registry.session_id = session_id
        msgs: list[dict] = [{"role": "system", "content": THINKER_STUDENT_PROMPT}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": prepend_memory(question, memory_text)})

        tool_trace: list[dict] = []
        final_text = ""
        final_kind = "answer"

        for step in range(1, self.max_steps + 1):
            out = self.gen.generate("thinker", msgs, None, self.tmax, self.temperature, greedy=False)
            kind, content = parse_thinker(out)
            msgs.append({"role": "assistant", "content": out})
            self._log(f"[thinker {step}] kind={kind} :: {content[:140]}")

            if kind == "act":
                ex = self._run_executor(content)
                result = ex["output_model"]
                self._log(f"[executor {step}] call={ex['call']} :: {result[:140]}")
                # Superset trace schema (mirrors 3_infererence.py tool_trace for uniform reporting);
                # `tool_call` + `result` are retained for backward compatibility / --self_test.
                tool_trace.append({
                    "step": step, "iteration": step - 1, "act": content,
                    "think_before_call": extract_think(out),
                    "executor_raw": ex["executor_raw"],
                    "tool_call": ex["call"], "tool": ex["tool"],
                    "input": (ex["call"] or {}).get("kwargs", {}),
                    "output_full": ex["output_full"], "output_model": result,
                    "result": result, "output_chars": len(ex["output_full"]),
                    "success": not ex["is_error"], "is_error": ex["is_error"],
                    "gen_ms": ex["gen_ms"], "tool_ms": ex["tool_ms"],
                })
                msgs.append({"role": "tool", "content": result})
                continue

            # answer / ask / unrecognised → terminal
            final_kind = kind or "answer"
            final_text = out
            break
        else:
            # Loop exhausted without an answer — force one final resolution.
            msgs.append({"role": "user",
                         "content": "Enough information has been gathered. Provide your final <answer> now."})
            final_text = self.gen.generate("thinker", msgs, None, self.tmax, self.temperature, greedy=False)
            msgs.append({"role": "assistant", "content": final_text})
            k, _ = parse_thinker(final_text)
            final_kind = k or "answer"

        kind, content = parse_thinker(final_text)
        answer_content = content if kind in ("answer", "ask") else final_text
        return {
            "response": final_text,
            "type": final_kind,
            "answer": answer_content,
            "think_content": extract_think(final_text),
            "tool_trace": tool_trace,
            "steps": len(tool_trace),
            "conversation": msgs,
        }


# ---------------------------------------------------------------------------
# Rendering / interactive / serve
# ---------------------------------------------------------------------------
def _render(res: dict) -> str:
    lines = []
    if res["think_content"]:
        lines.append(f"[think] {res['think_content']}")
    for t in res["tool_trace"]:
        fn = (t["tool_call"] or {}).get("function", "?")
        lines.append(f"[step {t['step']}] act: {t['act']}")
        lines.append(f"           → {fn}() → {t['result'][:200]}")
    lines.append(f"[{res['type']}] {res['answer']}")
    return "\n".join(lines)


def _chat(orch: ThinkerExecutor, memory_text: str) -> None:
    print("Thinker–Executor chat. Empty line or Ctrl-D to exit.\n")
    history: Optional[list] = None
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        res = orch.run(user, memory_text=memory_text, history=history)
        history = [m for m in res["conversation"] if m["role"] != "system"]
        print("\n" + _render(res) + "\n")


def _serve(orch: ThinkerExecutor, host: str, port: int, model_label: str) -> None:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Thinker–Executor Orchestrator", version="1.0.0")

    class Req(BaseModel):
        messages: list
        tool_profile: Optional[str] = None
        system_override: Optional[str] = None      # ignored — the Thinker has a fixed served prompt
        max_new_tokens: int = 1536
        temperature: float = 0.7
        session_id: Optional[str] = "anonymous"
        tool_mode: Optional[str] = "native"
        harness_enabled: Optional[bool] = None
        memory_text: Optional[str] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_label, "architecture": "thinker_executor"}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": model_label, "architecture": "thinker_executor"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: Req):
        msgs = req.messages or []
        last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        # Prior turns (everything before the final user message) continue the Thinker context.
        history = [m for m in msgs[:-1] if m.get("role") in ("user", "assistant", "tool")] if len(msgs) > 1 else None
        orch.temperature = req.temperature
        res = orch.run(last_user, memory_text=req.memory_text or "",
                       history=history, session_id=req.session_id or "anonymous")
        think = res["think_content"]
        return {
            "response": res["response"],
            "answer_content": res["answer"],
            "think_content": think,
            "think_length": len(think),
            "think_empty": not bool(think.strip()),
            "tool_trace": res["tool_trace"],
            "conversation": res["conversation"],
            "type": res["type"],
            # OpenAI-ish envelope for any client that expects choices[].message.content
            "choices": [{"message": {"role": "assistant", "content": res["response"]}}],
        }

    print(f"Ready. Listening on {host}:{port}")
    print(f"  benchmark: python 4_benchmark.py --server_url http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Offline self-test (no GPU, no model) — validates the orchestration control flow
# ---------------------------------------------------------------------------
def _self_test() -> None:
    print("=== Thinker–Executor orchestrator self-test (scripted, CPU) ===")

    # 1) act → real python_execute → answer
    thinker = [
        "<think>The user wants 17 times 23. This is deterministic arithmetic; I'll delegate it "
        "rather than risk a mental-math slip.</think>\n"
        "<act>Use python_execute to compute 17 * 23 and print the result.</act>",
        "<think>The execution system returned 391, which resolves the request exactly.</think>\n"
        "<answer>17 × 23 = 391. Want me to work through anything else?</answer>",
    ]
    executor = [
        '<think>\n</think>\n<tool_call>\n{"name": "python_execute", "arguments": {"code": "print(17 * 23)"}}\n</tool_call>',
    ]
    orch = ThinkerExecutor(ScriptedGenerator(thinker, executor), ToolRegistry(), verbose=True)
    res = orch.run("What is 17 times 23?", memory_text="")
    assert res["type"] == "answer", res["type"]
    assert res["steps"] == 1, res["steps"]
    assert res["tool_trace"][0]["tool_call"]["function"] == "python_execute"
    assert res["tool_trace"][0]["result"].strip() == "391", res["tool_trace"][0]["result"]
    assert "391" in res["answer"], res["answer"]
    print("  [pass] act → python_execute(391) → answer\n")

    # 2) ask path (genuine ambiguity, no tool)
    orch2 = ThinkerExecutor(ScriptedGenerator(
        ["<think>‘weather’ where? The location is the critical unknown.</think>\n"
         "<ask>Which city's weather would you like?</ask>"], []), ToolRegistry())
    r2 = orch2.run("What's the weather like?", memory_text="")
    assert r2["type"] == "ask" and "city" in r2["answer"].lower(), r2
    assert r2["steps"] == 0
    print("  [pass] ambiguous → ask (no tool call)\n")

    # 3) memory block injection (cold-start → explicit empty block)
    wrapped = prepend_memory("hello", "")
    assert "(no profile stored for this user yet)" in wrapped, wrapped
    assert "[USER MEMORY" in wrapped
    print("  [pass] empty memory → explicit cold-start block\n")

    # 4) executor parse failure is fed back as a recoverable tool error
    orch4 = ThinkerExecutor(ScriptedGenerator(
        ["<think>delegate</think>\n<act>Search the web for the EUR/USD rate.</act>",
         "<think>I could not get a clean result; I'll answer honestly.</think>\n"
         "<answer>I wasn't able to fetch a live rate just now.</answer>"],
        ["I will search now."]), ToolRegistry())   # executor emits no <tool_call>
    r4 = orch4.run("EUR to USD?", memory_text="")
    assert r4["tool_trace"][0]["tool_call"] is None
    assert "did not return a valid tool call" in r4["tool_trace"][0]["result"]
    assert r4["type"] == "answer"
    print("  [pass] malformed executor output → recoverable error → answer\n")

    print("ALL SELF-TESTS PASSED")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Thinker–Executor two-model inference orchestrator.")
    p.add_argument("--thinker", default="models/checkpoint_thinker",
                   help="Thinker LoRA checkpoint dir or HF repo id.")
    p.add_argument("--executor", default="models/checkpoint_executor",
                   help="Executor LoRA checkpoint dir or HF repo id.")
    p.add_argument("--base_model", default="unsloth/Qwen3-0.6B")
    p.add_argument("--load_in_4bit", action="store_true", help="4-bit base (bitsandbytes).")
    p.add_argument("--max_steps", type=int, default=6, help="Max Thinker↔Executor cycles per turn.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--memory_file", default=None,
                   help="Text file with a 5W+H profile to inject as the [USER MEMORY] block.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--question", default=None, help="One-shot question.")
    mode.add_argument("--chat", action="store_true", help="Interactive REPL.")
    mode.add_argument("--serve", action="store_true", help="FastAPI server (4_benchmark-compatible).")
    mode.add_argument("--self_test", action="store_true", help="Offline control-flow test (no GPU).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    memory_text = ""
    if args.memory_file:
        memory_text = Path(args.memory_file).read_text(encoding="utf-8").strip()

    gen = HFGenerator(args.base_model, args.thinker, args.executor, load_in_4bit=args.load_in_4bit)
    orch = ThinkerExecutor(gen, ToolRegistry(), max_steps=args.max_steps,
                           temperature=args.temperature, verbose=args.verbose)
    label = f"thinker={Path(args.thinker).name}|executor={Path(args.executor).name}"

    if args.serve:
        _serve(orch, args.host, args.port, label)
    elif args.chat:
        _chat(orch, memory_text)
    else:
        q = args.question or "What is 17 times 23?"
        print(_render(orch.run(q, memory_text=memory_text)))


if __name__ == "__main__":
    main()
