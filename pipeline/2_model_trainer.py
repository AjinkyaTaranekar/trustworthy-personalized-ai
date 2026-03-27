import json
import os
import argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from unsloth import FastModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False


MODEL_CONFIG = {
    "base_model": "unsloth/Qwen3-0.6B",
    # Increased from 2048 to handle longer think traces and multi-turn contexts
    "max_seq_length": 4096,
    "load_in_4bit": True,
    # Reduced rank — 0.6B model doesn't need r=64; r=16 trains faster and generalises better
    "lora_r": 16,
    "lora_alpha": 32,  # 2× rank is standard practice
}

SFT_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,       # One extra epoch for the expanded template set
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 100,           # Evaluate periodically to catch overfitting
    "bf16": True,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "packing": True,             # Pack short examples into full sequences — faster training
}


def messages_to_text(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


class ModelTrainer:
    """Trainer for interleaved thinking models using SFT."""
    
    def __init__(self, data_dir: str, output_dir: str, output_name: str = "checkpoint_sft"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name = output_name
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self):
        """Load the base model with Unsloth."""
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=MODEL_CONFIG["base_model"],
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            load_in_4bit=MODEL_CONFIG["load_in_4bit"],
            dtype=None,
        )
        return self
    
    def apply_lora(self):
        """Apply LoRA for efficient fine-tuning."""
        self.model = FastModel.get_peft_model(
            self.model,
            r=MODEL_CONFIG["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=MODEL_CONFIG["lora_alpha"],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return self
    
    def train_sft(self, dataset_path: str, output_name: str = "checkpoint_sft"):
        """Run supervised fine-tuning on the provided dataset."""
        raw = load_dataset("json", data_files=dataset_path)

        # Convert messages → text and split off a 5% eval set
        full_dataset = raw["train"].map(
            messages_to_text,
            fn_kwargs={"tokenizer": self.tokenizer},
        )
        split = full_dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset  = split["test"]
        print(f"  Train: {len(train_dataset)} examples | Eval: {len(eval_dataset)} examples")

        training_args = SFTConfig(
            output_dir=str(self.output_dir / output_name),
            per_device_train_batch_size=SFT_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
            num_train_epochs=SFT_CONFIG["num_train_epochs"],
            learning_rate=SFT_CONFIG["learning_rate"],
            warmup_steps=SFT_CONFIG["warmup_steps"],
            logging_steps=SFT_CONFIG["logging_steps"],
            save_steps=SFT_CONFIG["save_steps"],
            eval_steps=SFT_CONFIG["eval_steps"],
            eval_strategy="steps",
            bf16=SFT_CONFIG["bf16"],
            optim=SFT_CONFIG["optim"],
            weight_decay=SFT_CONFIG["weight_decay"],
            packing=SFT_CONFIG["packing"],
            max_seq_length=MODEL_CONFIG["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )

        train_result = trainer.train()

        # Save loss history for thesis charts
        log_history = trainer.state.log_history
        loss_log_path = self.output_dir / output_name / "loss_history.json"
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_log_path, "w") as f:
            json.dump(log_history, f, indent=2)
        print(f"  Loss history saved → {loss_log_path}")

        save_path = self.output_dir / output_name
        trainer.save_model(str(save_path))

        return self
    
    def train(self):
        """Run the full training pipeline."""
        self.load_base_model()
        self.apply_lora()
        
        dataset_path = self.data_dir / "train_interleaved.jsonl"
        self.train_sft(str(dataset_path), self.output_name)


def main():
    parser = argparse.ArgumentParser(description="Train interleaved thinking models")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--output_dir", default="./models", help="Output directory")
    parser.add_argument("--output_name", default="checkpoint_sft", help="Output model checkpoint directory name")
    parser.add_argument("--skip_if_exists", action="store_true",
                        help="Skip training if model already exists")
    
    args = parser.parse_args()
    
    if not HAS_LIBS:
        print("Error: Required libraries not installed.")
        print("Install with: pip install unsloth trl transformers datasets")
        return
    
    model_path = Path(args.output_dir)
    if args.skip_if_exists and model_path.exists() and any(model_path.iterdir()):
        print(f"Skipping training - model already exists at {model_path}")
        return
    
    trainer = ModelTrainer(args.data_dir, args.output_dir, args.output_name)
    trainer.train()


if __name__ == "__main__":
    main()
