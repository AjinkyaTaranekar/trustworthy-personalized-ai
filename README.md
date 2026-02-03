# Trustworthy Personalized AI

Building a Trustworthy Personalized LLM with Interleaved Thinking and First Principles

## Setup

### Create and Activate Virtual Environment

Navigate to the `pipeline` directory and create a virtual environment:

```bash
cd pipeline
python3 -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Install Requirements

Install dependencies in the virtual environment:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

Generate synthetic training data:

```bash
python3 1_dataset_generator.py --variant interleaved --train_size 5
```

**Options:**
- `--variant`: Dataset variant (e.g., `interleaved`)
- `--train_size`: Number of training examples to generate

This will create a JSONL dataset in `data/train_interleaved.jsonl`

### 2. Train Model

Train the model on generated data:

```bash
python3 2_model_trainer.py --data_dir data --output_dir models --output_name qwen3-0.6b
```

**Options:**
- `--data_dir`: Directory containing training data
- `--output_dir`: Directory to save trained models
- `--output_name`: Name of the output checkpoint directory (default: `checkpoint_sft`)
- `--skip_if_exists`: Skip training if model already exists

The trainer uses **Qwen3-0.6B** as the base model with LoRA fine-tuning.
