"""
Probe: does Qwen3-0.6B actually emit native <tool_call>s, and does it loop multi-turn?
======================================================================================
Standalone diagnostic — no inference server, no API keys. Loads the model directly and
runs the SAME intercept loop the real pipeline uses: generate, stop at </tool_call>,
execute the tool, feed the result back as a `tool` message, and generate again — until
the model produces an <answer> (or no tool call) or we hit the turn cap.

It runs each query under two system-prompt conditions so you can see what suppresses calls:
  [native]  no custom system prompt  -> pure Qwen3 native tool-calling
  [ours]    our STUDENT_PROMPTS["all_tools"] -> what training/inference actually uses

Usage:
    python probe_native_tool_calling.py
    python probe_native_tool_calling.py --model unsloth/Qwen3-0.6B --max_turns 6
    python probe_native_tool_calling.py --only ours        # just our prompt
    python probe_native_tool_calling.py --no_thinking      # enable_thinking=False

Read the output: each "[turn N]" block is one generation. If you see <tool_call> then a
[TOOL_RESULT] then another turn, multi-turn tool use is working. If a condition produces a
single turn with an answer and no <tool_call>, that condition suppresses tool calls.
"""

import argparse
import json
import re
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# OpenAI-schema tool definitions passed to apply_chat_template(tools=...).
TOOLS = [
    {"type": "function", "function": {
        "name": "python_execute",
        "description": "Execute Python code and return its stdout. Use for any precise arithmetic or computation.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python source; use print() for output."}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information (prices, news, live facts).",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query."}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_datetime",
        "description": "Return the current UTC date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

_ALLOWED = {"math", "statistics", "decimal", "fractions", "cmath", "random",
            "itertools", "functools", "operator", "collections", "re", "string"}


def _exec_tool(name: str, args: dict) -> str:
    """Stubbed/real tool execution so the loop can continue without API keys."""
    if name == "python_execute":
        code = args.get("code", "")
        try:
            import ast
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] not in _ALLOWED:
                            return f"Error: import '{a.name}' not allowed"
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").split(".")[0] not in _ALLOWED:
                        return f"Error: import '{node.module}' not allowed"
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
            return (p.stdout or p.stderr).strip() or "(no output)"
        except Exception as e:
            return f"Error: {e}"
    if name == "web_search":
        # canned result so the model has something to synthesise from
        return f"[web_search results for {args.get('query','')!r}] Top hit: BTC ≈ $94,250 USD (example/stubbed value)."
    if name == "get_datetime":
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"(no stub for tool {name})"


_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_call(text: str):
    m = _TOOLCALL_RE.search(text)
    if not m:
        # tokenizer sometimes drops the closing tag when we stop on it
        m2 = re.search(r"<tool_call>\s*(\{.*)", text, re.DOTALL)
        if not m2:
            return None
        raw = m2.group(1).strip()
    else:
        raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        return obj.get("name"), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None


def run_case(model, tok, query: str, system: str | None, enable_thinking: bool, max_turns: int):
    label = "ours" if system else "native"
    print(f"\n{'='*78}\nQUERY: {query}\nCONDITION: [{label}]  (system_prompt={'set' if system else 'none'}, enable_thinking={enable_thinking})\n{'='*78}")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": query})

    import torch
    for turn in range(1, max_turns + 1):
        text = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                       enable_thinking=enable_thinking, tokenize=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=512, do_sample=False)
        try:  # stop right after a tool call so we can execute it (mirrors the pipeline)
            gen_kwargs.update(stop_strings=["</tool_call>"], tokenizer=tok)
        except TypeError:
            pass
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[turn {turn}] model output:\n{gen}")

        parsed = _parse_tool_call(gen)
        if not parsed or not parsed[0]:
            print(f"\n--> no tool call this turn. Conversation ENDS at turn {turn}.")
            return
        name, args = parsed
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: args = {}
        result = _exec_tool(name, args)
        print(f"    [executed {name}({args})] -> {result[:200]}")
        # feed the tool call + result back so the model can continue (multi-turn)
        messages.append({"role": "assistant", "content": gen if gen.endswith("</tool_call>") else gen + "</tool_call>"})
        messages.append({"role": "tool", "name": name, "content": result})
    print(f"\n--> hit max_turns={max_turns} without a final answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-0.6B")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--no_thinking", action="store_true", help="enable_thinking=False")
    ap.add_argument("--only", choices=["native", "ours"], default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()

    # our actual student prompt — single source of truth
    try:
        from sft_v3_generator import STUDENT_PROMPTS
        ours = STUDENT_PROMPTS["all_tools"]
    except Exception as e:
        print(f"[warn] could not import STUDENT_PROMPTS ({e}); skipping 'ours' condition")
        ours = None

    queries = [
        "What is the current price of Bitcoin in USD?",          # expects web_search
        "Calculate 17.5% VAT on a purchase of 240 pounds.",      # expects python_execute
    ]
    conditions = []
    if args.only in (None, "native"):
        conditions.append(None)
    if args.only in (None, "ours") and ours:
        conditions.append(ours)

    for q in queries:
        for sys_prompt in conditions:
            run_case(model, tok, q, sys_prompt, not args.no_thinking, args.max_turns)


if __name__ == "__main__":
    main()
