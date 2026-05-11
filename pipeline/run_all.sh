#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master orchestration script for the full training pipeline
# =============================================================================
# Runs every stage in the correct order (per researchplan.tex + security-review.tex).
# Each stage checks its output checkpoint before running — fully resumable.
#
# Usage (from repo root on a GPU machine):
#   bash pipeline/run_all.sh                   # run all stages
#   bash pipeline/run_all.sh --from stage3     # resume from a specific stage
#   bash pipeline/run_all.sh --dry_run         # print plan without executing
#   bash pipeline/run_all.sh --stages 2,3,4    # run specific stages only
#
# Stage map:
#   Stage 1  — SFT data quality check (skip if data exists)
#   Stage 2  — SFT training             → models/checkpoint_sft/
#   Stage 3  — SFT constitutional baseline → reports/constitution_baseline.json
#   Stage 4  — Experiment 0 reasoning comparison → reports/experiment0_*.json
#   Stage 5  — Adversarial baseline  → reports/adversarial_baseline.json
#   Stage 6  — GRPO Condition C (format+accuracy) → models/checkpoint_grpo_c/
#   Stage 6b — GRPO-C drift check    → reports/grpo_c_drift_*.json
#   Stage 7  — GRPO Condition D (full composite)  → models/checkpoint_grpo_d/
#   Stage 7b — GRPO-D drift check    → reports/grpo_d_drift_*.json
#   Stage 8  — Final ablation benchmark (A/B/C/D) → reports/final_ablation_*.json
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$SCRIPT_DIR"
DATA_DIR="$PIPELINE/data"
MODELS_DIR="$PIPELINE/models"
REPORTS_DIR="$PIPELINE/reports"
LOG_FILE="$PIPELINE/run_all.log"

# Load .env from repo root if present (exports PIPELINE_* and API keys)
if [ -f "$REPO_ROOT/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +o allexport
fi

# Default settings
DRY_RUN=0
FROM_STAGE=1
ONLY_STAGES=""
SERVER_PORT=8000
SERVER_PORT_BASE=8001  # for base-model comparison runs

# ── Feature flags — read from environment (PIPELINE_* prefix) ─────────────────
# These mirror pipeline/config.py defaults. Override with env vars, e.g.:
#   PIPELINE_ENABLE_USER_MODELLING=true bash pipeline/run_all.sh
ENABLE_USER_MODELLING="${PIPELINE_ENABLE_USER_MODELLING:-false}"
ENABLE_EMPATHY="${PIPELINE_ENABLE_EMPATHY:-false}"
ENABLE_GRPO="${PIPELINE_ENABLE_GRPO:-true}"
ENABLE_ONTOLOGY_VERIF="${PIPELINE_ENABLE_ONTOLOGY_VERIF:-false}"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED=""; GRN=""; YLW=""; CYN=""; RST=""
if [ -t 1 ] && command -v tput &>/dev/null 2>&1; then
  RED=$(tput setaf 1 2>/dev/null||true); GRN=$(tput setaf 2 2>/dev/null||true)
  YLW=$(tput setaf 3 2>/dev/null||true); CYN=$(tput setaf 6 2>/dev/null||true)
  RST=$(tput sgr0 2>/dev/null||true)
fi

log()     { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
section() { echo ""; echo "${CYN}$(printf '=%.0s' {1..70})${RST}"; echo "${CYN}  $1${RST}"; echo "${CYN}$(printf '=%.0s' {1..70})${RST}"; log "=== $1 ==="; }
ok()      { echo "  ${GRN}[DONE]${RST} $1"; log "DONE: $1"; }
skip()    { echo "  ${YLW}[SKIP]${RST} $1 (checkpoint exists)"; log "SKIP: $1"; }
fail()    { echo "  ${RED}[FAIL]${RST} $1"; log "FAIL: $1"; exit 1; }
info()    { echo "  ${CYN}[INFO]${RST} $1"; }
dry()     { echo "  ${YLW}[DRY]${RST}  Would run: $1"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry_run)      DRY_RUN=1; shift;;
    --from)         FROM_STAGE="$2"; shift 2;;
    --stages)       ONLY_STAGES="$2"; shift 2;;
    --port)         SERVER_PORT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

should_run() {
  local stage="$1"
  if [ -n "$ONLY_STAGES" ]; then
    echo "$ONLY_STAGES" | grep -qw "$stage"
    return $?
  fi
  [ "$stage" -ge "$FROM_STAGE" ]
}

run_or_dry() {
  if [ "$DRY_RUN" -eq 1 ]; then
    dry "$*"
  else
    eval "$@"
  fi
}

# ── Server management ─────────────────────────────────────────────────────────
SERVER_PID=""
SERVER_PID_FILE="$PIPELINE/.server.pid"

start_server() {
  local model_path="$1" port="${2:-$SERVER_PORT}"
  log "Starting inference server: $model_path on port $port"
  if [ -f "$SERVER_PID_FILE" ]; then
    stop_server
  fi
  # Forward feature flags as env vars so the server inherits the same config
  env PIPELINE_ENABLE_USER_MODELLING="$ENABLE_USER_MODELLING" \
      PIPELINE_ENABLE_EMPATHY="$ENABLE_EMPATHY" \
      PIPELINE_ENABLE_ONTOLOGY_VERIF="$ENABLE_ONTOLOGY_VERIF" \
  python "$PIPELINE/3_infererence.py" --model_dir "$model_path" --port "$port" \
    > "$PIPELINE/.server_${port}.log" 2>&1 &
  echo "$!" > "$SERVER_PID_FILE"
  SERVER_PID="$!"

  # Wait for health endpoint (up to 120s)
  local url="http://localhost:$port/health"
  local waited=0
  while [ $waited -lt 120 ]; do
    if python -c "import urllib.request; urllib.request.urlopen('$url', timeout=3)" 2>/dev/null; then
      log "Server healthy at port $port (PID=$SERVER_PID)"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  fail "Server did not start within 120s — check $PIPELINE/.server_${port}.log"
}

stop_server() {
  if [ -f "$SERVER_PID_FILE" ]; then
    local pid
    pid=$(cat "$SERVER_PID_FILE")
    kill "$pid" 2>/dev/null || true
    rm -f "$SERVER_PID_FILE"
    log "Server stopped (PID=$pid)"
  fi
  SERVER_PID=""
}

trap stop_server EXIT

# ── FalkorDB management ───────────────────────────────────────────────────────

start_falkordb() {
  if [ "$ENABLE_USER_MODELLING" != "true" ]; then
    return 0
  fi
  log "Starting FalkorDB (docker compose up -d)..."
  if ! command -v docker &>/dev/null; then
    fail "docker not found — install Docker to use ENABLE_USER_MODELLING"
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    dry "docker compose up -d  (from $REPO_ROOT)"
    return 0
  fi
  docker compose -f "$REPO_ROOT/docker-compose.yml" up -d || fail "docker compose up -d failed"
  # Wait for FalkorDB to accept connections (up to 30s)
  local waited=0
  while [ $waited -lt 30 ]; do
    if python -c "import socket; s=socket.create_connection(('localhost',6379),timeout=2); s.close()" 2>/dev/null; then
      log "FalkorDB ready on port 6379"
      return 0
    fi
    sleep 2; waited=$((waited + 2))
  done
  fail "FalkorDB did not start within 30s — check: docker compose logs falkordb"
}

stop_falkordb() {
  if [ "$ENABLE_USER_MODELLING" != "true" ]; then
    return 0
  fi
  log "Stopping FalkorDB..."
  docker compose -f "$REPO_ROOT/docker-compose.yml" stop 2>/dev/null || true
}

# ── Main pipeline ─────────────────────────────────────────────────────────────

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$REPORTS_DIR"
echo "" >> "$LOG_FILE"
log "====== run_all.sh started ======"
[ "$DRY_RUN" -eq 1 ] && info "DRY RUN — no commands will execute"
info "Feature flags: GRPO=$ENABLE_GRPO | USER_MODELLING=$ENABLE_USER_MODELLING | EMPATHY=$ENABLE_EMPATHY | ONTOLOGY=$ENABLE_ONTOLOGY_VERIF"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# =============================================================================
# Stage 0 — FalkorDB startup (only when ENABLE_USER_MODELLING=true)
# =============================================================================

section "Stage 0 — Infrastructure Setup"
start_falkordb

# =============================================================================
# Stage 0.5 — Appraisal labelling (only when ENABLE_EMPATHY=true)
# =============================================================================

APPRAISAL_LABELS="$DATA_DIR/appraisal_labels.jsonl"
if [ "$ENABLE_EMPATHY" = "true" ]; then
  if [ -f "$APPRAISAL_LABELS" ]; then
    NLABELS=$(wc -l < "$APPRAISAL_LABELS" | tr -d ' ')
    skip "Appraisal labels already exist ($NLABELS examples) — delete to regenerate"
  else
    section "Stage 0.5 — Appraisal Labelling (AppraisePLM on EmpatheticDialogues)"
    run_or_dry "python '$PIPELINE/appraisal_labeller.py' --output '$APPRAISAL_LABELS'" \
      && ok "Appraisal labels written to $APPRAISAL_LABELS" \
      || fail "appraisal_labeller.py failed — check AppraisePLM setup (see pipeline/appraisal_labeller.py)"
  fi
else
  info "ENABLE_EMPATHY=false — skipping appraisal labelling"
fi

# =============================================================================
# Stage 1 — SFT data check
# =============================================================================

if should_run 1; then
  section "Stage 1 — SFT Data Check"
  SFT_DATA="$DATA_DIR/train_interleaved.jsonl"
  if [ -f "$SFT_DATA" ]; then
    NLINES=$(wc -l < "$SFT_DATA" | tr -d ' ')
    skip "SFT data present ($NLINES examples) — $SFT_DATA"
  else
    info "SFT data not found. Generating via sft pipeline..."
    info "This requires ANTHROPIC_API_KEY and takes ~1–2 hours."
    run_or_dry "python '$PIPELINE/sft_question_generator.py' --output '$DATA_DIR/questions_partA.jsonl'"
    run_or_dry "python '$PIPELINE/sft_gold_response_generator.py' \
      --questions '$DATA_DIR/questions_partA.jsonl' \
      --output '$DATA_DIR/train_partA.jsonl' \
      --critic_model claude-opus-4-7"
    run_or_dry "python '$PIPELINE/sft_math_pipeline.py' \
      --gsm8k_count 500 \
      --math_count 500 \
      --math_max_level 3 \
      --output '$DATA_DIR/train_partB.jsonl'"
    run_or_dry "python '$PIPELINE/sft_dataset_assembler.py' \
      --input_a '$DATA_DIR/train_partA.jsonl' \
      --input_b '$DATA_DIR/train_partB.jsonl' \
      --output '$DATA_DIR/train_interleaved.jsonl'"
    ok "SFT data generated"
  fi
fi

# =============================================================================
# Stage 2 — SFT Training
# =============================================================================

if should_run 2; then
  section "Stage 2 — SFT Training"
  SFT_CKPT="$MODELS_DIR/checkpoint_sft"
  if [ -f "$SFT_CKPT/adapter_config.json" ]; then
    skip "SFT checkpoint exists: $SFT_CKPT"
  else
    info "Training SFT on Qwen3-0.6B (expected: 2–4 hours on A100)..."
    run_or_dry "python '$PIPELINE/2_model_trainer.py' \
      --mode sft \
      --data_dir '$DATA_DIR' \
      --output_dir '$MODELS_DIR' \
      --output_name checkpoint_sft"
    [ "$DRY_RUN" -eq 0 ] && [ ! -f "$SFT_CKPT/adapter_config.json" ] && \
      fail "SFT checkpoint not found after training — check logs"
    ok "SFT training complete"
  fi
fi

# =============================================================================
# Stage 3 — Constitutional Baseline (SFT)
# =============================================================================

if should_run 3; then
  section "Stage 3 — SFT Constitutional Baseline"
  BASELINE="$REPORTS_DIR/constitution_baseline.json"
  if [ -f "$BASELINE" ]; then
    SCORE=$(python -c "import json; d=json.load(open('$BASELINE')); print(d.get('constitution_score','?'))" 2>/dev/null || echo "?")
    skip "SFT baseline exists (score=$SCORE) — $BASELINE"
  else
    info "Starting server with SFT checkpoint..."
    start_server "$MODELS_DIR/checkpoint_sft" "$SERVER_PORT"
    run_or_dry "python '$PIPELINE/4_benchmark.py' \
      --server_url 'http://localhost:$SERVER_PORT' \
      --probe_only \
      --save_as_baseline \
      --output_dir '$REPORTS_DIR'"
    stop_server
    ok "Constitutional baseline saved: $BASELINE"
  fi
fi

# =============================================================================
# Stage 4 — Experiment 0: Reasoning Paradigm Comparison
# =============================================================================

if should_run 4; then
  section "Stage 4 — Experiment 0: Reasoning Paradigm Comparison"
  EXP0_DONE=$(ls "$REPORTS_DIR"/experiment0_*.json 2>/dev/null | head -1)
  if [ -n "$EXP0_DONE" ]; then
    skip "Experiment 0 results exist: $EXP0_DONE"
  else
    info "Starting server with BASE MODEL (no fine-tuning)..."
    start_server "$MODELS_DIR/checkpoint_sft" "$SERVER_PORT"
    info "Running reasoning comparison (baseline / cot / interleaved / tot)..."
    run_or_dry "python '$PIPELINE/experiment0_reasoning_comparison.py' \
      --server_url 'http://localhost:$SERVER_PORT' \
      --n 100 \
      --output_dir '$REPORTS_DIR'"
    stop_server
    ok "Experiment 0 complete"
  fi
fi

# =============================================================================
# Stage 5 — Adversarial Baseline (SFT)
# =============================================================================

if should_run 5; then
  section "Stage 5 — Adversarial Baseline (SFT)"
  ADV_BASELINE="$REPORTS_DIR/adversarial_baseline.json"
  if [ -f "$ADV_BASELINE" ]; then
    skip "Adversarial baseline exists: $ADV_BASELINE"
  else
    start_server "$MODELS_DIR/checkpoint_sft" "$SERVER_PORT"
    run_or_dry "python '$PIPELINE/4_benchmark.py' \
      --server_url 'http://localhost:$SERVER_PORT' \
      --adversarial_only \
      --output_dir '$REPORTS_DIR'"
    # Rename to fixed name for easy reference
    if [ "$DRY_RUN" -eq 0 ]; then
      LATEST_ADV=$(ls -t "$REPORTS_DIR"/adversarial_*.json 2>/dev/null | head -1)
      [ -n "$LATEST_ADV" ] && cp "$LATEST_ADV" "$ADV_BASELINE"
    fi
    stop_server
    ok "Adversarial baseline saved"
  fi
fi

# =============================================================================
# Stage 6 — GRPO Condition C (format + accuracy only)
# =============================================================================

if should_run 6; then
  section "Stage 6 — GRPO Training: Condition C (format+accuracy)"
  GRPO_C="$MODELS_DIR/checkpoint_grpo_c"
  if [ -f "$GRPO_C/adapter_config.json" ]; then
    skip "GRPO-C checkpoint exists: $GRPO_C"
  else
    info "GRPO Condition C: SFT → RL with format+accuracy rewards only"
    info "Expected: 1–2 hours on A100"
    run_or_dry "python '$PIPELINE/2_model_trainer.py' \
      --mode grpo \
      --sft_checkpoint '$MODELS_DIR/checkpoint_sft' \
      --data_dir '$DATA_DIR' \
      --output_dir '$MODELS_DIR' \
      --output_name checkpoint_grpo_c \
      --reward_type c"
    ok "GRPO-C training complete"
  fi
fi

# Stage 6b — Drift check after GRPO-C
if should_run 6; then
  section "Stage 6b — Constitutional Drift Check (GRPO-C)"
  DRIFT_C=$(ls "$REPORTS_DIR"/grpo_c_drift_*.json 2>/dev/null | head -1)
  if [ -n "$DRIFT_C" ]; then
    skip "GRPO-C drift report exists"
  else
    start_server "$MODELS_DIR/checkpoint_grpo_c" "$SERVER_PORT"
    run_or_dry "python '$PIPELINE/4_benchmark.py' \
      --server_url 'http://localhost:$SERVER_PORT' \
      --probe_only \
      --baseline '$REPORTS_DIR/constitution_baseline.json' \
      --output_dir '$REPORTS_DIR'"
    if [ "$DRY_RUN" -eq 0 ]; then
      LATEST=$(ls -t "$REPORTS_DIR"/constitution_probe_*.json 2>/dev/null | head -1)
      [ -n "$LATEST" ] && cp "$LATEST" "$REPORTS_DIR/grpo_c_drift_${TIMESTAMP}.json"
    fi
    stop_server
    ok "GRPO-C drift check complete"
  fi
fi

# =============================================================================
# Stage 7 — GRPO Condition D (full composite)
# =============================================================================

if should_run 7; then
  section "Stage 7 — GRPO Training: Condition D (full composite)"
  GRPO_D="$MODELS_DIR/checkpoint_grpo_d"
  if [ -f "$GRPO_D/adapter_config.json" ]; then
    skip "GRPO-D checkpoint exists: $GRPO_D"
  else
    info "GRPO Condition D: SFT → RL with format+accuracy+tool+constitution rewards"
    info "Expected: 1–2 hours on A100"
    run_or_dry "python '$PIPELINE/2_model_trainer.py' \
      --mode grpo \
      --sft_checkpoint '$MODELS_DIR/checkpoint_sft' \
      --data_dir '$DATA_DIR' \
      --output_dir '$MODELS_DIR' \
      --output_name checkpoint_grpo_d \
      --reward_type d"
    ok "GRPO-D training complete"
  fi
fi

# Stage 7b — Drift check after GRPO-D
if should_run 7; then
  section "Stage 7b — Constitutional Drift Check (GRPO-D)"
  DRIFT_D=$(ls "$REPORTS_DIR"/grpo_d_drift_*.json 2>/dev/null | head -1)
  if [ -n "$DRIFT_D" ]; then
    skip "GRPO-D drift report exists"
  else
    start_server "$MODELS_DIR/checkpoint_grpo_d" "$SERVER_PORT"
    run_or_dry "python '$PIPELINE/4_benchmark.py' \
      --server_url 'http://localhost:$SERVER_PORT' \
      --probe_only \
      --baseline '$REPORTS_DIR/constitution_baseline.json' \
      --output_dir '$REPORTS_DIR'"
    if [ "$DRY_RUN" -eq 0 ]; then
      LATEST=$(ls -t "$REPORTS_DIR"/constitution_probe_*.json 2>/dev/null | head -1)
      [ -n "$LATEST" ] && cp "$LATEST" "$REPORTS_DIR/grpo_d_drift_${TIMESTAMP}.json"
    fi
    stop_server
    ok "GRPO-D drift check complete"
  fi
fi

# =============================================================================
# Stage 8 — Final Ablation: A/B/C/D
# =============================================================================

if should_run 8; then
  section "Stage 8 — Final Ablation Benchmark: Conditions A / B / C / D"
  FINAL_DONE=$(ls "$REPORTS_DIR"/final_ablation_*.json 2>/dev/null | head -1)
  if [ -n "$FINAL_DONE" ]; then
    skip "Final ablation report exists: $FINAL_DONE"
  else
    info "Running full benchmark + adversarial on all four conditions..."
    ABLATION_REPORT="$REPORTS_DIR/final_ablation_${TIMESTAMP}.json"

    run_or_dry "python - <<'PYEOF'
import json, subprocess, requests
from pathlib import Path

server_url = 'http://localhost:$SERVER_PORT'
reports_dir = Path('$REPORTS_DIR')
conditions = {
    'A_base':   '$MODELS_DIR/checkpoint_sft/../..',   # placeholder — use base model
    'B_sft':    '$MODELS_DIR/checkpoint_sft',
    'C_grpo_c': '$MODELS_DIR/checkpoint_grpo_c',
    'D_grpo_d': '$MODELS_DIR/checkpoint_grpo_d',
}
print('Ablation A/B/C/D — run each condition through 4_benchmark.py manually')
print('See README for the full benchmark command for each checkpoint.')
PYEOF"

    # Run each condition sequentially
    for label_ckpt in \
      "A_base:unsloth/Qwen3-0.6B" \
      "B_sft:$MODELS_DIR/checkpoint_sft" \
      "C_grpo_c:$MODELS_DIR/checkpoint_grpo_c" \
      "D_grpo_d:$MODELS_DIR/checkpoint_grpo_d"
    do
      LABEL="${label_ckpt%%:*}"
      CKPT="${label_ckpt##*:}"
      info "Running condition $LABEL with checkpoint: $CKPT"

      if [ -d "$CKPT" ] || [[ "$CKPT" == unsloth/* ]]; then
        start_server "$CKPT" "$SERVER_PORT"
        run_or_dry "python '$PIPELINE/4_benchmark.py' \
          --server_url 'http://localhost:$SERVER_PORT' \
          --probe \
          --adversarial \
          --baseline '$REPORTS_DIR/constitution_baseline.json' \
          --output_dir '$REPORTS_DIR'"
        if [ "$DRY_RUN" -eq 0 ]; then
          LATEST=$(ls -t "$REPORTS_DIR"/benchmark_*.json 2>/dev/null | head -1)
          [ -n "$LATEST" ] && cp "$LATEST" "$REPORTS_DIR/ablation_${LABEL}_${TIMESTAMP}.json"
        fi
        stop_server
        ok "Condition $LABEL done"
      else
        info "Skipping $LABEL — checkpoint not found at $CKPT"
      fi
    done

    ok "Final ablation complete — see reports/ablation_*_${TIMESTAMP}.json"
  fi
fi

# =============================================================================
# Done
# =============================================================================

section "Pipeline Complete"
echo ""
echo "  All stages finished. Key outputs:"
echo "    models/checkpoint_sft/           — SFT Condition B"
echo "    models/checkpoint_grpo_c/        — GRPO Condition C"
echo "    models/checkpoint_grpo_d/        — GRPO Condition D"
echo "    reports/constitution_baseline.json"
echo "    reports/experiment0_*.json"
echo "    reports/adversarial_baseline.json"
echo "    reports/ablation_*_${TIMESTAMP}.json"
echo ""
echo "  Log: $LOG_FILE"
log "====== run_all.sh complete ======"
