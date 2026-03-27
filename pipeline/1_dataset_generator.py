import json
import random
import argparse
from typing import List, Dict
from pathlib import Path


def load_templates(templates_dir: Path) -> Dict:
    """Load templates from JSON files."""
    templates_path = templates_dir / "templates.json"
    
    with open(templates_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    return templates


class DatasetGenerator:
    """Generate synthetic datasets for training."""
    
    def __init__(self, templates: Dict, seed: int = 42):
        random.seed(seed)
        self.templates = templates
        self.generated_examples = set()
    
    def render_template(self, template: str, variables: Dict) -> str:
        """Replace {variable} placeholders with values."""
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result
    
    def sample_variables(self, variables: Dict) -> Dict:
        """Sample one value for each variable.
        
        For aligned variables (same length arrays that should be sampled together),
        we pick a random index and use that index for all variables of that length.
        This ensures e.g., country_a and search_result_a always come from the same index.
        """
        if not variables:
            return {}
        
        # Group variables by their array length
        length_groups = {}
        for k, v in variables.items():
            length = len(v)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(k)
        
        result = {}
        # For each group of same-length arrays, pick one random index
        for length, keys in length_groups.items():
            idx = random.randint(0, length - 1)
            for k in keys:
                result[k] = variables[k][idx]
        
        return result
    
    def generate_interleaved_example(self, template_config: Dict) -> Dict:
        """Generate example for interleaved thinking model with multi-turn tool calls."""
        template = random.choice(template_config["templates"])
        variables = self.sample_variables(template["variables"])
        
        conversation = [
            {"role": "system", "content": template_config["system_prompt"]},
            {"role": "user", "content": self.render_template(template["user"], variables)}
        ]
        
        # Handle new multi-turn format with "turns" array
        if "turns" in template:
            for turn in template["turns"]:
                assistant_parts = []
                
                # Add thinking
                if "think" in turn and turn["think"]:
                    think = self.render_template(turn["think"], variables)
                    assistant_parts.append(f"<think>{think}</think>")
                
                # Add tool call if present
                if "tool" in turn and turn["tool"]:
                    tool = self.render_template(turn["tool"], variables)
                    assistant_parts.append(f"<tool>{tool}</tool>")
                
                if assistant_parts:
                    conversation.append({"role": "assistant", "content": "\n".join(assistant_parts)})
                
                # Add tool result if present
                if "tool_result" in turn and turn["tool_result"]:
                    tool_result = self.render_template(turn["tool_result"], variables)
                    conversation.append({"role": "tool", "content": tool_result})
            
            # Add final answer
            if "answer" in template:
                answer = self.render_template(template["answer"], variables)
                conversation.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
        
        # Handle legacy single-turn format for backwards compatibility
        else:
            assistant_parts = []
            
            if "think" in template:
                think = self.render_template(template["think"], variables)
                assistant_parts.append(f"<think>{think}</think>")
            
            if "tool" in template:
                tool = self.render_template(template["tool"], variables)
                assistant_parts.append(f"<tool>{tool}</tool>")
            
            conversation.append({"role": "assistant", "content": "\n".join(assistant_parts)})
            
            if "tool" in template and "tool_result" in template:
                tool_result = self.render_template(template["tool_result"], variables)
                conversation.append({"role": "tool", "content": tool_result})
                
                answer = self.render_template(template["answer"], variables)
                conversation.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
            elif "answer" in template:
                answer = self.render_template(template["answer"], variables)
                conversation[-1]["content"] += f"\n<answer>{answer}</answer>"
        
        # Count tool calls for metadata
        num_tool_calls = 0
        if "turns" in template:
            num_tool_calls = sum(1 for t in template["turns"] if t.get("tool"))
        elif "tool" in template:
            num_tool_calls = 1
        
        return {
            "messages": conversation,
            "metadata": {
                "model_variant": "interleaved",
                "has_thinking": True,
                "has_tools": num_tool_calls > 0,
                "num_tool_calls": num_tool_calls,
                "is_multi_turn": "turns" in template and len(template.get("turns", [])) > 1
            }
        }
    
    def generate_interleaved_with_personalization_example(self, template_config: Dict) -> Dict:
        """Generate example for full system (interleaved + personalization + emotion).
        Supports both multi-turn (turns array) and legacy single-turn formats.
        """
        template = random.choice(template_config["templates"])
        variables = self.sample_variables(template["variables"])

        conversation = [
            {"role": "system", "content": template_config["system_prompt"]},
            {"role": "user", "content": self.render_template(template["user"], variables)},
        ]

        num_tool_calls = 0

        # Multi-turn format (same logic as generate_interleaved_example)
        if "turns" in template:
            for turn in template["turns"]:
                assistant_parts = []
                if turn.get("think"):
                    assistant_parts.append(f"<think>{self.render_template(turn['think'], variables)}</think>")
                if turn.get("tool"):
                    assistant_parts.append(f"<tool>{self.render_template(turn['tool'], variables)}</tool>")
                    num_tool_calls += 1
                if assistant_parts:
                    conversation.append({"role": "assistant", "content": "\n".join(assistant_parts)})
                if turn.get("tool_result"):
                    conversation.append({"role": "tool", "content": self.render_template(turn["tool_result"], variables)})

            if "answer" in template:
                answer = self.render_template(template["answer"], variables)
                scrutiny = template.get("scrutiny", {})
                scrutiny_text = ""
                if scrutiny:
                    rendered = {
                        k: [self.render_template(v, variables) for v in val] if isinstance(val, list)
                        else self.render_template(val, variables)
                        for k, val in scrutiny.items()
                    }
                    scrutiny_text = f"\n<scrutiny>{json.dumps(rendered, indent=2)}</scrutiny>"
                conversation.append({"role": "assistant", "content": f"<answer>{answer}</answer>{scrutiny_text}"})

        # Legacy single-turn format
        else:
            assistant_parts = []
            if "think" in template:
                assistant_parts.append(f"<think>{self.render_template(template['think'], variables)}</think>")
            if "tool" in template:
                assistant_parts.append(f"<tool>{self.render_template(template['tool'], variables)}</tool>")
                num_tool_calls = 1
            conversation.append({"role": "assistant", "content": "\n".join(assistant_parts)})

            if "tool" in template and "tool_result" in template:
                conversation.append({"role": "tool", "content": self.render_template(template["tool_result"], variables)})
                answer = self.render_template(template["answer"], variables)
                scrutiny = template.get("scrutiny", {})
                scrutiny_text = ""
                if scrutiny:
                    rendered = {
                        k: [self.render_template(v, variables) for v in val] if isinstance(val, list)
                        else self.render_template(val, variables)
                        for k, val in scrutiny.items()
                    }
                    scrutiny_text = f"\n<scrutiny>{json.dumps(rendered, indent=2)}</scrutiny>"
                conversation.append({"role": "assistant", "content": f"<answer>{answer}</answer>{scrutiny_text}"})
            elif "answer" in template:
                answer = self.render_template(template["answer"], variables)
                conversation[-1]["content"] += f"\n<answer>{answer}</answer>"

        return {
            "messages": conversation,
            "metadata": {
                "model_variant": "interleaved_with_personalization",
                "has_thinking": True,
                "has_tools": num_tool_calls > 0,
                "num_tool_calls": num_tool_calls,
                "has_personalization": "profile" in template.get("user", ""),
                "has_emotion": "emotion" in template.get("user", "") or "appraisal" in str(template),
            },
        }
    
    def generate_rl_verifiable_example(self) -> Dict:
        """Generate verifiable example for RL training with step-by-step reasoning verification."""
        if "rl_verifiable" not in self.templates:
            raise ValueError("rl_verifiable templates not found in templates configuration")
        
        template_config = self.templates["rl_verifiable"]
        template = random.choice(template_config["templates"])
        variables = self.sample_variables(template["variables"])
        
        # Render question
        question = self.render_template(template["question"], variables)
        
        # Render step-by-step reasoning
        steps = []
        for step_template in template["steps"]:
            step = {
                "step_number": step_template["step_number"],
                "description": self.render_template(step_template["description"], variables),
                "reasoning": self.render_template(step_template["reasoning"], variables),
                "verification": self.render_template(step_template["verification"], variables)
            }
            steps.append(step)
        
        # Render tool calls if present
        tool_calls = []
        if template.get("tool_calls"):
            for tool_template in template["tool_calls"]:
                tool_call = {
                    "tool_name": self.render_template(tool_template["tool_name"], variables),
                    "parameters": {
                        k: self.render_template(str(v), variables) 
                        for k, v in tool_template["parameters"].items()
                    },
                    "expected_result": self.render_template(str(tool_template["expected_result"]), variables),
                    "verification": self.render_template(tool_template["verification"], variables)
                }
                tool_calls.append(tool_call)
        
        # Render answer
        answer = self.render_template(template["answer"], variables)
        
        return {
            "question": question,
            "steps": steps,
            "tool_calls": tool_calls,
            "answer": answer,
            "verifiable": True,
            "metadata": {
                "model_variant": "rl_training",
                "reward_function": "step_by_step_verification",
                "has_tool_calls": len(tool_calls) > 0,
                "num_steps": len(steps)
            }
        }
    
    def generate_dataset(self, model_variant: str, num_examples: int) -> List[Dict]:
        """Generate dataset for a specific model variant."""
        dataset = []
        
        generators = {
            "interleaved": self.generate_interleaved_example,
            "interleaved_with_personalization": self.generate_interleaved_with_personalization_example
        }
        
        if model_variant not in generators:
            raise ValueError(f"Unknown model variant: {model_variant}")
        
        template_config = self.templates[model_variant]
        generator = generators[model_variant]
        
        for _ in range(num_examples):
            example = generator(template_config)
            dataset.append(example)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")
        
        print(f"Saved {len(dataset)} examples to {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Generate datasets for interleaved thinking models")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    parser.add_argument("--templates_dir", default=None, help="Templates directory (default: ./pipeline/templates)")
    parser.add_argument("--variant", required=True, choices=["interleaved", "interleaved_with_personalization", "rl_verifiable"], help="Training variant to generate (required)")
    parser.add_argument("--train_size", type=int, default=75000, help="Training examples for selected variant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Determine templates directory
    script_dir = Path(__file__).parent
    templates_dir = Path(args.templates_dir) if args.templates_dir else script_dir / "templates"
    
    # Load templates
    print("Loading templates...")
    templates = load_templates(templates_dir)
    
    # Initialize generator
    generator = DatasetGenerator(templates, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training dataset
    if args.variant == "interleaved":
        print(f"\nGenerating Interleaved Dataset ({args.train_size} examples)...")
        interleaved = generator.generate_dataset("interleaved", args.train_size)
        generator.save_dataset(interleaved, output_dir / "train_interleaved.jsonl")
    
    elif args.variant == "interleaved_with_personalization":
        print(f"\nGenerating Interleaved with Personalization Dataset ({args.train_size} examples)...")
        interleaved_with_personalization = generator.generate_dataset("interleaved_with_personalization", args.train_size)
        generator.save_dataset(interleaved_with_personalization, output_dir / "train_interleaved_with_personalization.jsonl")
    
    elif args.variant == "rl_verifiable":
        print(f"\nGenerating RL Verifiable Dataset ({args.train_size} examples)...")
        rl_dataset = [generator.generate_rl_verifiable_example() for _ in range(args.train_size)]
        generator.save_dataset(rl_dataset, output_dir / "train_rl_verifiable.jsonl")
    
if __name__ == "__main__":
    main()
