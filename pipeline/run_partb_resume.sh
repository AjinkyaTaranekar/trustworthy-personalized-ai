#!/usr/bin/env bash
# One-shot launcher for sft_math_pipeline.py --resume
# Scheduled via cron — safe to kill this terminal after nohup kicks off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/nohup_generator.out"

echo "=== run_partb_resume.sh started at $(date) ===" >> "$LOG"

cd "$SCRIPT_DIR"
nohup .venv/bin/python -u sft_math_pipeline.py \
    --resume \
    --model nvidia_nim/minimaxai/minimax-m2.7 \
    >> "$LOG" 2>&1 &

echo "Launched PID $! — logging to $LOG" >> "$LOG"

# Remove this cron job after firing so it doesn't repeat
(crontab -l 2>/dev/null | grep -v "run_partb_resume.sh") | crontab -
