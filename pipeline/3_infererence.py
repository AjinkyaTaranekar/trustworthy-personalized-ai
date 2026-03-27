import argparse
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from unsloth import FastModel


# Tool implementations
class Tools:
    @staticmethod
    def python_execute(code: str) -> str:
        """Execute Python code in a subprocess with a 10-second timeout."""
        import subprocess
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip() or result.stderr.strip()
            print(f"Executed code:\n{code}\nOutput:\n{output}")
            return output if output else "Code executed successfully (no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (10s limit)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def get_exchange_rate(**kwargs) -> str:
        """Mock currency converter."""
        # Simplified mock rates
        mock_rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 155.58, "INR": 90.33}
        from_currency = kwargs.get("from", "").upper()
        to_currency = kwargs.get("to", "").upper()
        if from_currency in mock_rates and to_currency in mock_rates:
            rate = mock_rates[to_currency] / mock_rates[from_currency]
            return json.dumps({"rate": rate})
        return json.dumps({"error": "Currency not supported"})

def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool name and arguments from <tool>...</tool> tags."""
    tool_match = re.search(r'<tool>(.*?)</tool>', text, re.DOTALL)
    if not tool_match:
        return None
    
    tool_text = tool_match.group(1).strip()
    
    # Parse tool call: function_name(arg1='value1', arg2='value2')
    func_match = re.match(r'(\w+)\((.*)\)', tool_text)
    if not func_match:
        return None
    
    func_name = func_match.group(1)
    args_str = func_match.group(2)
    
    # Parse arguments
    kwargs = {}
    if args_str:
        # Match key=value pairs, capturing quoted strings properly
        pattern = r"(\w+)=(?:(['\"])((?:\\.|(?!\2).)*?)\2|([^,\)]+))"
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            quote_char = match.group(2)
            quoted_value = match.group(3)
            unquoted_value = match.group(4)
            
            # Use quoted value if present, otherwise unquoted
            value = quoted_value if quote_char else (unquoted_value.strip() if unquoted_value else "")
            
            # Decode escape sequences in quoted strings
            if quote_char:
                value = value.encode().decode('unicode_escape')
            
            # Try to convert to appropriate type
            try:
                if value.replace('.', '').replace('-', '').isdigit():
                    value = float(value) if '.' in value else int(value)
            except:
                pass
            kwargs[key] = value
    
    return {"function": func_name, "kwargs": kwargs}


def execute_tool(tool_call: Dict[str, Any], tools: Tools) -> str:
    """Execute a tool call and return the result."""
    func_name = tool_call["function"]
    kwargs = tool_call["kwargs"]
    
    if not hasattr(tools, func_name):
        return f"Error: Unknown tool '{func_name}'"
    
    try:
        func = getattr(tools, func_name)
        result = func(**kwargs)
        return result
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int, max_iterations: int, temperature: float, tools: Tools, model_name: str = "Model", include_tools: bool = True):
    """Run inference with a given model and return the full conversation."""
    if include_tools:
        system_prompt = (
            "You are a trustworthy assistant that prioritizes accuracy, transparency, and accountability. "
            "Think step by step using <think>, call tools using <tool>, and give your final answer in <answer>. "
            "You must never guess when tools are needed, always acknowledge uncertainty, ask for missing information "
            "rather than assuming, deny impossible tasks honestly, and present tradeoffs for subjective questions.\n\n"
            "Available tools:\n"
            "1. python_execute(code='...') - Execute Python code and return the output.\n"
            "   Example: <tool>python_execute(code='print(15 + 27)')</tool>\n\n"
            "2. get_exchange_rate(from='USD', to='EUR') - Get currency exchange rates between two currencies.\n"
            "   Example: <tool>get_exchange_rate(from='USD', to='EUR')</tool>\n\n"
            "To use a tool, wrap your call in <tool></tool> tags with the exact function syntax shown above."
        )
    else:
        system_prompt = (
            "You are a trustworthy assistant that prioritizes accuracy, transparency, and accountability. "
            "Think step by step using <think> and give your final answer in <answer>. "
            "Always acknowledge uncertainty and never guess facts you are not confident about."
        )
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    print(f"\n{model_name}")
    print(f"User: {prompt}\n")
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # Generate response
        prompt_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        print(f"Response (iteration {iteration}):\n{response}\n")
        
        # Check if we have a final answer
        if '<answer>' in response.lower():
            break
        
        # Check for tool call
        tool_call = extract_tool_call(response)
        if tool_call:
            print(f"[Executing tool: {tool_call['function']}]")
            tool_result = execute_tool(tool_call, tools)
            print(f"[Tool result: {tool_result}]\n")
            
            # Add tool result using the same role as training data
            conversation.append({
                "role": "tool",
                "content": f"Tool result: {tool_result}"
            })
        else:
            # No tool call and no answer - model might be done
            break
    
    return conversation


def save_report(report_data: Dict[str, Any], output_dir: Path, filename: str):
    """Save a report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"\n📄 Report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Inference with tool support for interleaved thinking")
    parser.add_argument(
        "--model_dir",
        default="./models/checkpoint_sft",
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--base_model",
        default="unsloth/Qwen3-0.6B",
        help="Base model from Hugging Face Hub (e.g., 'unsloth/Qwen3-0.6B')",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare custom model with base model",
    )
    parser.add_argument("--prompt", default="What is 15 + 27?", help="User prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max tool call iterations")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--output_dir",
        default="./reports",
        help="Directory to save reports",
    )
    parser.add_argument(
        "--no_tools",
        action="store_true",
        help="Run base model without tool instructions (only for comparison mode)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    if not model_dir.exists() and not args.compare:
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Initialize tools
    tools = Tools()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.compare:
        print(f"Loading base model: {args.base_model}...")
        base_model, base_tokenizer = FastModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        
        # Load custom model if exists
        custom_model, custom_tokenizer = None, None
        if model_dir.exists():
            print(f"Loading custom model: {model_dir}...")
            custom_model, custom_tokenizer = FastModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=4096,
                load_in_4bit=True,
                dtype=None,
            )
        
        # Run inference on base model WITHOUT tools (baseline)
        base_no_tools_conversation = run_inference(
            base_model, base_tokenizer, args.prompt, 
            args.max_new_tokens, args.max_iterations, args.temperature, 
            tools, model_name="BASE MODEL (No Tools)", include_tools=False
        )
        
        # Run inference on base model WITH tool instructions
        base_with_tools_conversation = run_inference(
            base_model, base_tokenizer, args.prompt, 
            args.max_new_tokens, args.max_iterations, args.temperature, 
            tools, model_name="BASE MODEL (With Tools)", include_tools=True
        )
        
        # Run inference on custom model if loaded
        custom_conversation = None
        if custom_model is not None:
            custom_conversation = run_inference(
                custom_model, custom_tokenizer, args.prompt,
                args.max_new_tokens, args.max_iterations, args.temperature,
                tools, model_name="CUSTOM MODEL", include_tools=True
            )
        
        # Create and save comparison report
        comparison_report = {
            "timestamp": timestamp,
            "prompt": args.prompt,
            "config": {
                "base_model": args.base_model,
                "custom_model": str(model_dir) if model_dir.exists() else "Not found",
                "max_new_tokens": args.max_new_tokens,
                "max_iterations": args.max_iterations,
                "temperature": args.temperature,
            },
            "base_model_no_tools": {
                "conversation": base_no_tools_conversation,
                "num_turns": len([m for m in base_no_tools_conversation if m["role"] == "assistant"]),
            },
            "base_model_with_tools": {
                "conversation": base_with_tools_conversation,
                "num_turns": len([m for m in base_with_tools_conversation if m["role"] == "assistant"]),
            },
            "custom_model_output": {
                "conversation": custom_conversation if custom_conversation else None,
                "num_turns": len([m for m in custom_conversation if m["role"] == "assistant"]) if custom_conversation else 0,
            } if custom_conversation else None,
        }
        
        # Save comparison report
        save_report(
            comparison_report,
            output_dir,
            f"comparison_{timestamp}.json"
        )
        
    else:
        print("Loading custom model...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=4096,
            load_in_4bit=True,
            dtype=None,
        )
        
        conversation = run_inference(
            model, tokenizer, args.prompt,
            args.max_new_tokens, args.max_iterations, args.temperature,
            tools, model_name="CUSTOM MODEL", include_tools=True
        )
        
        # Save single model report
        single_report = {
            "timestamp": timestamp,
            "prompt": args.prompt,
            "config": {
                "model": str(model_dir),
                "max_new_tokens": args.max_new_tokens,
                "max_iterations": args.max_iterations,
                "temperature": args.temperature,
            },
            "output": {
                "conversation": conversation,
                "num_turns": len([m for m in conversation if m["role"] == "assistant"]),
            },
        }
        
        save_report(
            single_report,
            output_dir,
            f"inference_{timestamp}.json"
        )


if __name__ == "__main__":
    main()
