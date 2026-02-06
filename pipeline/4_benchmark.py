import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from unsloth import FastModel

BENCHMARK_QUESTIONS: List[str] = [
    # 1 - Simple arithmetic (should use tool)
    "What is 15 + 27?",
    # 2 - Multi-step arithmetic (should use tool)
    "Now multiply that result by 3 and subtract 10.",
    # 3 - Currency conversion (requires tool + context recall)
    "Convert 500 USD to EUR for me.",
    # 4 - Context recall (tests memory of earlier turns)
    "What was the very first calculation I asked you to do, and what was the answer?",
    # 5 - Complex math (multiple operations)
    "Calculate (144 / 12) + (25 * 4) - 37.",
    # 6 - Referencing previous result
    "Add the result of that last calculation to the EUR amount you computed earlier.",
    # 7 - Multi-step reasoning with tool
    "If I invest that combined amount at 5% annual interest, how much will I have after 3 years? Use compound interest.",
    # 8 - Knowledge question (no tool needed ideally)
    "What is the capital of Ireland?",
    # 9 - Combining knowledge + calculation
    "What is the approximate population of that city, and what is 1% of it?",
    # 10 - Summarisation / reflection
    "Summarise everything we have discussed so far in bullet points.",
]


class Tools:
    @staticmethod
    def python_execute(code: str = "", **kw) -> str:
        try:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            exec(code)
            sys.stdout = old_stdout
            out = buf.getvalue()
            return out.strip() if out else "Code executed successfully (no output)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def get_exchange_rate(**kwargs) -> str:
        mock_rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 155.58, "INR": 90.33}
        fc = kwargs.get("from", "").upper()
        tc = kwargs.get("to", "").upper()
        if fc in mock_rates and tc in mock_rates:
            rate = mock_rates[tc] / mock_rates[fc]
            return json.dumps({"rate": round(rate, 4)})
        return json.dumps({"error": "Currency not supported"})


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not match:
        return None
    tool_text = match.group(1).strip()
    func_match = re.match(r"(\w+)\((.*)\)", tool_text, re.DOTALL)
    if not func_match:
        return None
    func_name = func_match.group(1)
    args_str = func_match.group(2)
    kwargs: Dict[str, Any] = {}
    if args_str:
        pattern = r"(\w+)=(?:(['\"])((?:\\.|(?!\2).)*?)\2|([^,\)]+))"
        for m in re.finditer(pattern, args_str):
            key = m.group(1)
            qchar = m.group(2)
            qval = m.group(3)
            uval = m.group(4)
            value = qval if qchar else (uval.strip() if uval else "")
            if qchar:
                try:
                    value = value.encode().decode("unicode_escape")
                except Exception:
                    pass
            try:
                if value.replace(".", "").replace("-", "").isdigit():
                    value = float(value) if "." in value else int(value)
            except Exception:
                pass
            kwargs[key] = value
    return {"function": func_name, "kwargs": kwargs}


def execute_tool(tool_call: Dict[str, Any], tools: Tools) -> str:
    fn = tool_call["function"]
    if not hasattr(tools, fn):
        return f"Error: Unknown tool '{fn}'"
    try:
        return getattr(tools, fn)(**tool_call["kwargs"])
    except Exception as e:
        return f"Error executing {fn}: {e}"


SYSTEM_WITH_TOOLS = (
    "You are a helpful assistant that thinks step by step. Use <think> for reasoning "
    "about what to do next, <tool> for tool calls, and <answer> for final responses. You may need "
    "multiple iterations of thinking and tool calls to reach the final answer.\n\n"
    "Available tools:\n"
    "1. python_execute(code='...') - Execute Python code and return the output.\n"
    "   Example: <tool>python_execute(code='print(15 + 27)')</tool>\n\n"
    "2. get_exchange_rate(from='USD', to='EUR') - Get currency exchange rates.\n"
    "   Example: <tool>get_exchange_rate(from='USD', to='EUR')</tool>\n\n"
    "To use a tool, wrap your call in <tool></tool> tags."
)

SYSTEM_NO_TOOLS = (
    "You are a helpful assistant. Provide clear and accurate responses. "
    "Use <think> for reasoning and <answer> for final responses."
)


def run_single_turn(
    model,
    tokenizer,
    conversation: list,
    max_new_tokens: int,
    max_tool_iters: int,
    temperature: float,
    tools: Tools,
) -> Dict[str, Any]:
    """
    Generate a response for the current conversation state.
    Handles tool loops internally.  Returns a metrics dict.
    """
    turn_metrics: Dict[str, Any] = {
        "tool_calls": 0,
        "tool_results": [],
        "sub_iterations": [],
    }
    total_output_tokens = 0
    total_gen_time = 0.0

    for tool_iter in range(max_tool_iters):
        try:
            prompt_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_token_count = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
            gen_time = time.perf_counter() - t0

            output_ids = outputs[0][input_token_count:]
            output_token_count = len(output_ids)
            response = tokenizer.decode(output_ids, skip_special_tokens=True)

        except Exception as e:
            # Generation failed — record error and break out of tool loop
            print(f"   ⚠️  Generation error (iter {tool_iter+1}): {e}")
            turn_metrics["error"] = f"Generation failed: {e}"
            conversation.append({"role": "assistant", "content": f"[generation error: {e}]"})
            break

        total_output_tokens += output_token_count
        total_gen_time += gen_time

        sub_iter = {
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "generation_time_s": round(gen_time, 3),
            "tokens_per_sec": round(output_token_count / gen_time, 2) if gen_time > 0 else 0,
        }
        turn_metrics["sub_iterations"].append(sub_iter)

        conversation.append({"role": "assistant", "content": response})

        # Final answer?
        if "<answer>" in response.lower():
            break

        # Tool call?
        tc = extract_tool_call(response)
        if tc:
            turn_metrics["tool_calls"] += 1
            result = execute_tool(tc, tools)
            turn_metrics["tool_results"].append({
                "tool": tc["function"],
                "kwargs": {k: str(v) for k, v in tc["kwargs"].items()},
                "result": result,
            })
            conversation.append({"role": "tool", "content": f"Tool result: {result}"})
        else:
            break  # No tool call, no answer → stop

    # Aggregate
    # Count the total context tokens at the end of this turn
    try:
        final_prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
        )
        total_context_tokens = len(tokenizer.encode(final_prompt))
    except Exception:
        total_context_tokens = 0

    turn_metrics.update({
        "total_input_tokens": int(sum(s["input_tokens"] for s in turn_metrics["sub_iterations"])),
        "total_output_tokens": int(total_output_tokens),
        "total_context_tokens": int(total_context_tokens),
        "total_generation_time_s": round(total_gen_time, 3),
        "overall_tokens_per_sec": round(total_output_tokens / total_gen_time, 2) if total_gen_time > 0 else 0,
        "has_answer": any("<answer>" in m.get("content", "").lower() for m in conversation if m["role"] == "assistant"),
    })

    return turn_metrics


def run_benchmark(
    model,
    tokenizer,
    questions: List[str],
    max_new_tokens: int,
    max_tool_iters: int,
    temperature: float,
    tools: Tools,
    model_label: str,
    include_tools: bool = True,
) -> Dict[str, Any]:
    """Run all questions sequentially, accumulating context."""

    system_prompt = SYSTEM_WITH_TOOLS if include_tools else SYSTEM_NO_TOOLS
    conversation: list = [{"role": "system", "content": system_prompt}]

    turns: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  {model_label}")
    print(f"{'='*60}")

    for idx, question in enumerate(questions, 1):
        print(f"\n── Turn {idx}/{len(questions)}: {question[:80]}")
        conversation.append({"role": "user", "content": question})

        try:
            metrics = run_single_turn(
                model, tokenizer, conversation,
                max_new_tokens, max_tool_iters, temperature, tools,
            )

            # Extract the assistant's final text for this turn
            assistant_msgs = [m["content"] for m in conversation if m["role"] == "assistant"]
            last_response = assistant_msgs[-1] if assistant_msgs else ""

            turn_data = {
                "turn_number": idx,
                "question": question,
                "response": last_response,
                "metrics": metrics,
            }
            turns.append(turn_data)

            print(f"   ↳ ctx={metrics['total_context_tokens']}  "
                  f"in={metrics['total_input_tokens']}  "
                  f"out={metrics['total_output_tokens']}  "
                  f"tools={metrics['tool_calls']}  "
                  f"time={metrics['total_generation_time_s']}s  "
                  f"tok/s={metrics['overall_tokens_per_sec']}")

        except Exception as e:
            print(f"   ❌ Turn {idx} failed: {e}")
            error_metrics = {
                "tool_calls": 0, "tool_results": [], "sub_iterations": [],
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_context_tokens": 0, "total_generation_time_s": 0,
                "overall_tokens_per_sec": 0, "has_answer": False,
                "error": str(e),
            }
            turns.append({
                "turn_number": idx,
                "question": question,
                "response": f"[turn failed: {e}]",
                "metrics": error_metrics,
            })
            # Add a placeholder so conversation history stays consistent
            conversation.append({"role": "assistant", "content": f"[turn failed: {e}]"})

    # Build summary statistics
    summary = _build_summary(turns)

    return {
        "model_label": model_label,
        "include_tools": include_tools,
        "system_prompt": system_prompt,
        "conversation": conversation,
        "turns": turns,
        "summary": summary,
    }


def _build_summary(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate stats across all turns."""
    ctx_lengths = [t["metrics"]["total_context_tokens"] for t in turns]
    out_tokens = [t["metrics"]["total_output_tokens"] for t in turns]
    in_tokens = [t["metrics"]["total_input_tokens"] for t in turns]
    gen_times = [t["metrics"]["total_generation_time_s"] for t in turns]
    tok_rates = [t["metrics"]["overall_tokens_per_sec"] for t in turns]
    tool_counts = [t["metrics"]["tool_calls"] for t in turns]
    answer_flags = [t["metrics"]["has_answer"] for t in turns]

    return {
        "total_turns": len(turns),
        "total_tool_calls": sum(tool_counts),
        "turns_with_answer_tag": sum(answer_flags),
        "total_input_tokens": sum(in_tokens),
        "total_output_tokens": sum(out_tokens),
        "total_generation_time_s": round(sum(gen_times), 3),
        "avg_tokens_per_sec": round(sum(tok_rates) / len(tok_rates), 2) if tok_rates else 0,
        "context_growth": {
            "start": ctx_lengths[0] if ctx_lengths else 0,
            "end": ctx_lengths[-1] if ctx_lengths else 0,
            "per_turn": ctx_lengths,
        },
        "output_tokens_per_turn": out_tokens,
        "input_tokens_per_turn": in_tokens,
        "generation_time_per_turn": gen_times,
        "tokens_per_sec_per_turn": tok_rates,
        "tool_calls_per_turn": tool_counts,
    }

def save_report(data: Dict[str, Any], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n📄  Report saved → {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Continuous-conversation benchmark for base vs fine-tuned model."
    )
    ap.add_argument("--model_dir", default="./models/checkpoint_sft",
                     help="Path to the fine-tuned model checkpoint")
    ap.add_argument("--base_model", default="unsloth/Qwen3-0.6B",
                     help="Base model name on HF Hub")
    ap.add_argument("--compare", action="store_true",
                     help="Run both base and custom models")
    ap.add_argument("--questions", default=None,
                     help="Comma-separated list of questions (overrides defaults)")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_tool_iters", type=int, default=10,
                     help="Max tool-call iterations per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output_dir", default="./reports")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    tools = Tools()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    questions = (
        [q.strip() for q in args.questions.split(",")]
        if args.questions
        else BENCHMARK_QUESTIONS
    )

    results: Dict[str, Any] = {
        "type": "benchmark",
        "timestamp": timestamp,
        "questions": questions,
        "config": {
            "base_model": args.base_model,
            "custom_model": str(model_dir),
            "max_new_tokens": args.max_new_tokens,
            "max_tool_iters": args.max_tool_iters,
            "temperature": args.temperature,
            "num_questions": len(questions),
        },
        "runs": {},
    }

    try:
        if args.compare:
            # ── Base model (no tools + with tools) ──
            try:
                print("\n🔄 Loading base model …")
                base_model, base_tok = FastModel.from_pretrained(
                    model_name=args.base_model,
                    max_seq_length=2048, load_in_4bit=True, dtype=None,
                )
                FastModel.for_inference(base_model)

                try:
                    results["runs"]["base_no_tools"] = run_benchmark(
                        base_model, base_tok, questions,
                        args.max_new_tokens, args.max_tool_iters, args.temperature,
                        tools, model_label="Base Model (No Tools)", include_tools=False,
                    )
                except Exception as e:
                    print(f"\n❌ Base model (no tools) run failed: {e}")
                    results["runs"]["base_no_tools"] = {"error": str(e), "model_label": "Base Model (No Tools)"}

                try:
                    results["runs"]["base_with_tools"] = run_benchmark(
                        base_model, base_tok, questions,
                        args.max_new_tokens, args.max_tool_iters, args.temperature,
                        tools, model_label="Base Model (With Tools)", include_tools=True,
                    )
                except Exception as e:
                    print(f"\n❌ Base model (with tools) run failed: {e}")
                    results["runs"]["base_with_tools"] = {"error": str(e), "model_label": "Base Model (With Tools)"}

                # Free base model memory
                del base_model, base_tok
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ Failed to load base model: {e}")
                results["errors"] = results.get("errors", []) + [f"Base model load failed: {e}"]

            # ── Custom model ──
            if model_dir.exists():
                try:
                    print("\n🔄 Loading custom model …")
                    custom_model, custom_tok = FastModel.from_pretrained(
                        model_name=str(model_dir),
                        max_seq_length=2048, load_in_4bit=True, dtype=None,
                    )
                    FastModel.for_inference(custom_model)

                    results["runs"]["custom"] = run_benchmark(
                        custom_model, custom_tok, questions,
                        args.max_new_tokens, args.max_tool_iters, args.temperature,
                        tools, model_label="Custom Model (Fine-tuned)", include_tools=True,
                    )
                except Exception as e:
                    print(f"\n❌ Custom model run failed: {e}")
                    results["runs"]["custom"] = {"error": str(e), "model_label": "Custom Model (Fine-tuned)"}
            else:
                print(f"⚠️  Custom model dir not found: {model_dir}")

        else:
            # Single model mode
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            print("\n🔄 Loading custom model …")
            model, tok = FastModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=2048, load_in_4bit=True, dtype=None,
            )
            FastModel.for_inference(model)

            results["runs"]["custom"] = run_benchmark(
                model, tok, questions,
                args.max_new_tokens, args.max_tool_iters, args.temperature,
                tools, model_label="Custom Model (Fine-tuned)", include_tools=True,
            )

        _print_comparison_table(results)

    except Exception as e:
        print(f"\n❌ Unexpected top-level error: {e}")
        results["fatal_error"] = str(e)

    finally:
        # ALWAYS save whatever we collected, even on crash
        save_report(results, output_dir, f"benchmark_{timestamp}.json")
        print("\n✅ Report saved (even if some runs failed).")


def _print_comparison_table(results: Dict[str, Any]):
    """Pretty-print a comparison table to stdout."""
    runs = results.get("runs", {})
    if not runs:
        return

    # Filter to only runs that completed (have a "summary" key)
    valid_labels = [k for k in runs if "summary" in runs[k]]
    errored_labels = [k for k in runs if "error" in runs[k] and "summary" not in runs[k]]

    if errored_labels:
        print(f"\n⚠️  The following runs failed and have no metrics:")
        for k in errored_labels:
            print(f"   • {runs[k].get('model_label', k)}: {runs[k].get('error', 'unknown')}")

    if not valid_labels:
        print("\n⚠️  No runs completed successfully — nothing to summarise.")
        return

    n_turns = results["config"]["num_questions"]

    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY  —  {n_turns} turns")
    print(f"{'='*90}")

    header = f"{'Metric':<35}"
    for label in valid_labels:
        header += f"  {runs[label]['model_label']:>22}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Total input tokens",   lambda s: f"{s['total_input_tokens']:,}"),
        ("Total output tokens",  lambda s: f"{s['total_output_tokens']:,}"),
        ("Total generation time", lambda s: f"{s['total_generation_time_s']:.1f}s"),
        ("Avg tokens/sec",        lambda s: f"{s['avg_tokens_per_sec']:.1f}"),
        ("Total tool calls",      lambda s: f"{s['total_tool_calls']}"),
        ("Turns with <answer>",   lambda s: f"{s['turns_with_answer_tag']}/{s['total_turns']}"),
        ("Final context length",  lambda s: f"{s['context_growth']['end']:,}"),
    ]

    for label_text, fmt_fn in rows:
        line = f"{label_text:<35}"
        for run_key in valid_labels:
            summary = runs[run_key]["summary"]
            try:
                line += f"  {fmt_fn(summary):>22}"
            except Exception:
                line += f"  {'N/A':>22}"
        print(line)

    # Per-turn context growth
    print(f"\n{'Context length per turn':}")
    for i in range(n_turns):
        line = f"  Turn {i+1:<3}"
        for run_key in valid_labels:
            ctx = runs[run_key]["summary"]["context_growth"]["per_turn"]
            val = ctx[i] if i < len(ctx) else "-"
            line += f"  {str(val):>22}"
        print(line)

    print()


if __name__ == "__main__":
    main()
