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

### 3. Run Inference on the Trained Model

To run inference using the trained model, execute the following command:

```bash
python3 3_infererence.py --prompt "calculate 100*20-10+(50/12.5)"
```

Compare the model with default base model:
```bash
python3 3_infererence.py --compare --prompt "calculate 159*234/24-5"
```

Compare the model with a different base model:
```bash
python3 3_infererence.py --base_model "unsloth/Qwen3-4B" --compare --prompt "calculate 159*234/24-5"
```

**Options:**
- `--prompt`: User prompt for the model
- `--model_dir`: Path to the fine-tuned model (default: `./models/checkpoint_sft`)
- `--base_model`: Base model for comparison (default: `unsloth/Qwen3-0.6B`)
- `--compare`: Compare custom model with base model
- `--max_new_tokens`: Max tokens to generate (default: 2048)
- `--max_iterations`: Max tool call iterations (default: 10)
- `--temperature`: Sampling temperature (default: 0.7)
- `--output_dir`: Directory to save reports (default: `./reports`)

**Report Saving:**

Inference results are automatically saved as JSON reports in the `reports/` directory:
- Single model: `inference_YYYYMMDD_HHMMSS.json`
- Comparison: `comparison_YYYYMMDD_HHMMSS.json`

Each report includes:
- Timestamp and prompt
- Model configuration
- Complete conversation history
- Number of turns for each model (in comparison mode)

