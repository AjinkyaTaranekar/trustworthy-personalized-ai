---
title: SFT Training Crash Debugging Guide (Ubuntu + A4000)
type: query
tags: [training, debugging, sft, grpo, gpu]
updated: 2026-05-11
status: current
---

**A4000 (16 GB VRAM) training crash triage guide for the Qwen3-0.6B SFT and GRPO pipeline.**

## 1. First: Read the Crash

Run these immediately after a crash to find the root cause before the kernel ring buffer rolls over.

```bash
# OOM killer — most likely culprit for step-0 crash
dmesg | grep -iE "killed process|oom.killer|out of memory" | tail -20

# CUDA / NVIDIA driver errors
dmesg | grep -iE "nvidia|cuda|nvrm" | tail -30

# Systemd journal (captures Python tracebacks if run as a service)
journalctl -xe --no-pager | tail -50

# GPU state right now — look for ERR! in the processes section
nvidia-smi

# Detailed VRAM breakdown
nvidia-smi -q -d MEMORY

# System RAM (confirm it wasn't host RAM, not just VRAM)
free -h
```

---

## 2. Interpret What You See

| Signal | Meaning | Go to |
|--------|---------|-------|
| `Out of memory: Killed process … python` in dmesg | Host RAM OOM — model load ate all RAM | §3.1 |
| `CUDA out of memory` in Python traceback | VRAM OOM during forward/backward pass | §3.2 |
| `NVRM: Xid … error` in dmesg | GPU hardware / driver fault | §3.3 |
| `Bus error` or `Segmentation fault` | Driver or PCIe issue | §3.3 |
| Clean exit, no error | Dataset or config crash before first step | §3.4 |

---

## 3. Fix Recipes

### 3.1 Host RAM OOM (model load)

Qwen3-0.6B in 4-bit loads ~2 GB weights but the tokeniser + dataset prep can spike host RAM to 8–12 GB. If the machine has ≤16 GB RAM and is running a desktop/other processes:

```bash
# Check available RAM before starting
free -h

# Kill memory hogs
sudo systemctl stop <service>   # e.g. a Jupyter server you left running

# Add a swap file as emergency buffer (not a permanent fix)
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

### 3.2 VRAM OOM (forward/backward pass)

This is the most common step-0 crash. `packing=True` fills the entire batch to `max_seq_length=4096` on the very first step, causing a peak VRAM spike larger than steady-state training.

**Quick fix — apply in order until stable:**

**Step A:** Set this env var before running (prevents CUDA allocator fragmentation):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python pipeline/2_model_trainer.py --mode sft
```

**Step B:** If still OOM, reduce batch size in `pipeline/2_model_trainer.py`:
```python
SFT_CONFIG = {
    "per_device_train_batch_size": 1,   # was 2
    "gradient_accumulation_steps": 8,   # keep effective batch = 8
    ...
}
```

**Step C:** If still OOM, reduce sequence length:
```python
MODEL_CONFIG = {
    ...
    "max_seq_length": 2048,   # was 4096 — halves activation memory
}
```

**Step D:** If still OOM, disable packing temporarily (slower but more predictable memory):
```python
SFT_CONFIG = {
    ...
    "packing": False,
}
```

**Step E:** Nuclear option — reduce LoRA rank:
```python
MODEL_CONFIG = {
    ...
    "lora_r":     8,    # was 16
    "lora_alpha": 16,   # was 32 (keep ratio 2:1)
}
```

### 3.3 GPU Hardware / Driver Error

```bash
# Check driver version
nvidia-smi | grep "Driver Version"

# Run the built-in GPU diagnostic
sudo nvidia-smi -q | grep -iE "error|xid|ecc"

# Check PCIe bandwidth isn't throttled
sudo nvidia-smi -q -d PCIE | grep "Link Width"

# Stress test GPU memory (install first: pip install gpuburn)
# Run for 60s to verify hardware is stable
nvidia-smi dmon -s u -d 1   # watch utilisation live during a test run
```

If Xid errors appear (especially Xid 79 = GPU has fallen off the bus, Xid 13 = graphics exception), the driver or hardware is faulty — check PCIe slot, power connectors, and consider reinstalling the NVIDIA driver.

```bash
# Reinstall driver without removing CUDA
sudo apt-get install --reinstall nvidia-driver-<version>
```

### 3.4 Silent Crash (no GPU error)

If `dmesg` is clean and no CUDA error appears, the crash is in Python before the GPU is touched — usually a dataset or config issue.

```bash
# Run with full Python traceback visible
python -u pipeline/2_model_trainer.py --mode sft 2>&1 | tee training.log

# Check the log for the last lines before exit
tail -50 training.log
```

Common non-GPU causes:
- Missing or corrupt `train_interleaved.jsonl` (check it exists and has content)
- `tokenizer.apply_chat_template` failing on a malformed example
- `load_best_model_at_end=True` conflicting with eval setup (set to `False` to test)

---

## 4. Live Monitoring During Training

Run this in a second terminal to catch the exact moment VRAM spikes:

```bash
# Poll GPU every 2 seconds — log to file for post-mortem
nvidia-smi dmon -s mu -d 2 | tee gpu_monitor.log

# Or watch live with colour
watch -n 2 nvidia-smi
```

If you want to be alerted when VRAM exceeds a threshold:
```bash
# Notify when free VRAM drops below 1 GB
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free_mb" -lt 1000 ]; then
    echo "$(date): WARNING — only ${free_mb} MB free" | tee -a gpu_monitor.log
  fi
  sleep 5
done
```

---

## 5. Recommended Stable Config for A4000 (16 GB)

These are conservative settings confirmed safe for 16 GB. The default config was written for 24 GB.

```python
MODEL_CONFIG = {
    "max_seq_length": 2048,   # 4096 is marginal on 16 GB with packing
    "load_in_4bit":   True,
    "lora_r":         16,
    "lora_alpha":     32,
}

SFT_CONFIG = {
    "per_device_train_batch_size": 1,   # 2 risks OOM with packing
    "gradient_accumulation_steps": 8,   # effective batch = 8
    "packing":                     True,
    ...
}

GRPO_CONFIG = {
    "num_generations":             4,   # 8 requires 24 GB — halve for 16 GB
    "per_device_train_batch_size": 1,
    ...
}
```

---

## 6. Resuming After a Crash

Once the crash is fixed, resume from the last saved checkpoint (saves happen every 500 steps for SFT, 100 steps for GRPO):

```bash
# SFT — auto-detects latest checkpoint-N in models/checkpoint_sft/
python pipeline/2_model_trainer.py --mode sft --resume

# GRPO
python pipeline/2_model_trainer.py --mode grpo --resume
```

Check what checkpoints landed before the crash:
```bash
ls -la models/checkpoint_sft/
# Look for checkpoint-500/, checkpoint-1000/, etc.
# If only adapter_config.json exists, training never completed step 500
```

---

## Related

- [[entities/grpo]] — GRPO training configuration
- [[queries/grpo-and-personalisation-master-plan]] — full pipeline plan
- [[sources/code/training-and-benchmark]] — pipeline code overview
