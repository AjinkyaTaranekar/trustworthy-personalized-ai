# Trustworthy Personalized AI

Building a Trustworthy Personalized LLM with Interleaved Thinking and First Principles

## Setup

### Install Requirements

Navigate to the `pipeline` directory and install dependencies:

```bash
cd pipeline
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

Generate synthetic training data:

```bash
python 1_dataset_generator.py --variant interleaved --train_size 5
```

**Options:**
- `--variant`: Dataset variant (e.g., `interleaved`)
- `--train_size`: Number of training examples to generate

This will create a JSONL dataset in `data/train_interleaved.jsonl`

### 2. Train Model

Train the model on generated data:

```bash
python 2_model_trainer.py --data_dir data --output_dir models
```

**Options:**
- `--data_dir`: Directory containing training data
- `--output_dir`: Directory to save trained models

The trainer uses **Qwen3-4B** as the base model with LoRA fine-tuning.
