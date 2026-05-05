#!/usr/bin/env bash
# =============================================================================
# preflight_check.sh — Pre-flight validation before Experiment 0
# =============================================================================
# Checks every hard requirement and soft dependency needed before running the
# reasoning paradigm comparison (Experiment 0 — researchplan.tex Phase 3).
#
# Run from the repo root or from pipeline/:
#   bash pipeline/preflight_check.sh
#
# Exit codes:
#   0  all hard requirements met (warnings may exist)
#   1  one or more hard requirements are MISSING
# =============================================================================

PASS=0; FAIL=0; WARN=0; HARD_FAIL=0

RED=""; GRN=""; YLW=""; CYN=""; RST=""
if [ -t 1 ] && command -v tput &>/dev/null 2>&1; then
  RED=$(tput setaf 1 2>/dev/null || true)
  GRN=$(tput setaf 2 2>/dev/null || true)
  YLW=$(tput setaf 3 2>/dev/null || true)
  CYN=$(tput setaf 6 2>/dev/null || true)
  RST=$(tput sgr0  2>/dev/null || true)
fi

pass()    { echo "  ${GRN}[PASS]${RST} $1"; ((PASS++));               }
fail()    { echo "  ${RED}[FAIL]${RST} $1"; ((FAIL++)); ((HARD_FAIL++)); }
warn()    { echo "  ${YLW}[WARN]${RST} $1"; ((WARN++));               }
info()    { echo "  ${CYN}[INFO]${RST} $1";                           }
section() { echo ""; echo "${CYN}=== $1 ===${RST}";                   }

# Resolve repo root — works whether called from repo root or pipeline/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE="$SCRIPT_DIR"

# py_check FILE  — syntax-check a Python file via argv (avoids path-in-string issues)
py_check() {
  python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$1" 2>/dev/null
}

# py_import PKG  — test whether a Python package is importable
py_import() {
  python -c "import sys; sys.exit(0 if __import__('$1') else 1)" 2>/dev/null
}

# py_ver PKG  — print a package version
py_ver() {
  python -c "import $1; print(getattr($1,'__version__','?'))" 2>/dev/null || echo "?"
}

# =============================================================================
section "1. Git State"
# =============================================================================

BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
EXPECTED="feat/grpo-and-personalisation-stack"
if [ "$BRANCH" = "$EXPECTED" ]; then
  pass "On correct branch: $BRANCH"
else
  warn "Expected branch '$EXPECTED', currently on '$BRANCH'"
fi

DIRTY=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$DIRTY" -eq 0 ]; then
  pass "Working tree clean"
else
  warn "$DIRTY uncommitted change(s) — consider committing before running experiments"
fi

# =============================================================================
section "2. Python Version"
# =============================================================================

PY_VER=$(python --version 2>&1 | awk '{print $2}')
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MINOR" -ge 10 ]; then
  pass "Python $PY_VER (>= 3.10 required)"
else
  fail "Python $PY_VER — 3.10+ required (pipeline uses modern type-union syntax)"
fi

# =============================================================================
section "3. GPU / CUDA"
# =============================================================================

CUDA_STATUS=$(python -c "
import torch
if torch.cuda.is_available():
    print('ok', torch.cuda.device_count(), torch.cuda.get_device_name(0))
else:
    print('none')
" 2>/dev/null || echo "none")

if echo "$CUDA_STATUS" | grep -q "^ok"; then
  GPU_INFO=$(echo "$CUDA_STATUS" | cut -d' ' -f2-)
  pass "CUDA available — $GPU_INFO"
else
  warn "No CUDA GPU — inference will be slow on CPU"
  warn "Experiment 0 works on CPU for small batches; GRPO training requires a GPU"
  info "Use Colab / Kaggle / HPC cluster for full training runs"
fi

# =============================================================================
section "4. Required Python Packages"
# =============================================================================

MISSING_HARD=(); MISSING_SOFT=()

check_pkg() {
  local imp="$1" install="$2" hardness="$3"
  if python -c "import $imp" 2>/dev/null; then
    VER=$(py_ver "$imp")
    pass "$imp ($VER)"
  else
    if [ "$hardness" = "hard" ]; then
      fail "MISSING: $imp  (pip install $install)"
      MISSING_HARD+=("$install")
    else
      warn "MISSING optional: $imp  (pip install $install)"
      MISSING_SOFT+=("$install")
    fi
  fi
}

check_pkg torch         torch          hard
check_pkg transformers  transformers   hard
check_pkg datasets      datasets       hard
check_pkg trl           trl            hard
check_pkg fastapi       fastapi        hard
check_pkg uvicorn       uvicorn        hard
check_pkg requests      requests       hard
check_pkg litellm       litellm        hard
check_pkg dotenv        python-dotenv  hard
check_pkg pydantic      pydantic       hard
check_pkg accelerate    accelerate     soft
check_pkg bitsandbytes  bitsandbytes   soft
check_pkg unsloth       unsloth        soft

# =============================================================================
section "5. Environment Variables and .env"
# =============================================================================

ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  pass ".env file present"
  # Check key using argv — avoids embedding path in string
  KEY_OK=$(python -c "
import sys
from dotenv import load_dotenv
import os
load_dotenv(sys.argv[1])
key = os.environ.get('ANTHROPIC_API_KEY', '')
print('ok' if key.startswith('sk-ant') else 'missing')
" "$ENV_FILE" 2>/dev/null || echo "missing")
  if [ "$KEY_OK" = "ok" ]; then
    pass "ANTHROPIC_API_KEY set and looks valid"
  else
    fail "ANTHROPIC_API_KEY missing or invalid in .env (needed for --critic_model in SFT)"
    info "Add to .env:  ANTHROPIC_API_KEY=sk-ant-api03-..."
  fi
else
  fail ".env not found — create at repo root with ANTHROPIC_API_KEY=sk-ant-..."
  fail "ANTHROPIC_API_KEY cannot be checked (no .env)"
fi

# =============================================================================
section "6. Pipeline Files"
# =============================================================================

check_file() {
  local path="$1" desc="$2"
  if [ -f "$path" ]; then
    pass "$desc"
  else
    fail "MISSING FILE: $desc ($(basename "$path"))"
  fi
}

check_file "$PIPELINE/config.py"                     "config.py — feature flags singleton"
check_file "$PIPELINE/user_modelling.py"             "user_modelling.py — 5W+H graph + Mem0g write pipeline"
check_file "$PIPELINE/empathy.py"                    "empathy.py — appraisal-conditioned generation"
check_file "$PIPELINE/appraisal_labeller.py"         "appraisal_labeller.py — offline AppraisePLM labeller"
check_file "$PIPELINE/ontology_verifier.py"          "ontology_verifier.py — post-hoc SPARQL claim scorer"
check_file "$REPO_ROOT/docker-compose.yml"           "docker-compose.yml — FalkorDB service"
check_file "$PIPELINE/constitution.md"               "constitution.md — 19 constitutional principles"
check_file "$PIPELINE/3_infererence.py"              "3_infererence.py — inference server"
check_file "$PIPELINE/4_benchmark.py"                "4_benchmark.py — benchmark + adversarial suite"
check_file "$PIPELINE/sft_gold_response_generator.py" "sft_gold_response_generator.py"
check_file "$PIPELINE/sft_question_generator.py"     "sft_question_generator.py"
check_file "$PIPELINE/sft_math_question_generator.py" "sft_math_question_generator.py"
check_file "$PIPELINE/sft_rejection_sampler.py"      "sft_rejection_sampler.py"
check_file "$PIPELINE/sft_dataset_assembler.py"      "sft_dataset_assembler.py"
check_file "$PIPELINE/2_model_trainer.py"            "2_model_trainer.py — SFT trainer"
check_file "$PIPELINE/1_dataset_generator.py"        "1_dataset_generator.py — pipeline orchestrator"
check_file "$PIPELINE/5_context_degradation.py"      "5_context_degradation.py — context degradation eval"

# =============================================================================
section "7. Directories (create if missing)"
# =============================================================================

for dir_entry in "data:training data" "models:model checkpoints" "reports:benchmark reports"; do
  dname="${dir_entry%%:*}"; ddesc="${dir_entry##*:}"
  dpath="$PIPELINE/$dname"
  if [ -d "$dpath" ]; then
    pass "Directory exists: $dname/ ($ddesc)"
  else
    mkdir -p "$dpath"
    warn "Created missing directory: $dname/ ($ddesc)"
  fi
done

SFT_DATA="$PIPELINE/data/train_interleaved.jsonl"
if [ -f "$SFT_DATA" ]; then
  NLINES=$(wc -l < "$SFT_DATA" | tr -d ' ')
  pass "SFT data present: train_interleaved.jsonl ($NLINES examples)"
else
  warn "SFT data not found — run: python pipeline/1_dataset_generator.py"
fi

# =============================================================================
section "8. Python Syntax Check — All Pipeline .py Files"
# =============================================================================

SYNTAX_OK=0; SYNTAX_FAIL=0
for pyfile in "$PIPELINE"/*.py; do
  fname=$(basename "$pyfile")
  if py_check "$pyfile"; then
    pass "Syntax OK: $fname"
    ((SYNTAX_OK++))
  else
    fail "Syntax ERROR: $fname"
    ((SYNTAX_FAIL++))
  fi
done
info "$SYNTAX_OK files clean, $SYNTAX_FAIL with errors"

# =============================================================================
section "9. Security Blockers — Symbol Presence"
# =============================================================================

check_sym() {
  local file="$1" sym="$2" label="$3"
  if grep -q "$sym" "$file" 2>/dev/null; then
    pass "Blocker $label: $sym  ($(basename "$file"))"
  else
    fail "Blocker $label MISSING: $sym not found in $(basename "$file")"
  fi
}

check_sym "$PIPELINE/3_infererence.py"               "_validate_code"         "1a code sandbox"
check_sym "$PIPELINE/3_infererence.py"               "_sanitise_tool_output"  "1b injection sanitiser"
check_sym "$PIPELINE/sft_rejection_sampler.py"       "_validate_code"         "1c sampler sandbox"
check_sym "$PIPELINE/sft_math_question_generator.py" "_validate_code"         "1d math-gen sandbox"
check_sym "$PIPELINE/sft_gold_response_generator.py" "rule_check_response"    "2a rule-based verifier"
check_sym "$PIPELINE/sft_gold_response_generator.py" "_merge_violations"      "2b violation merge"
check_sym "$PIPELINE/4_benchmark.py"                 "ADVERSARIAL_PROBES"     "3a adversarial suite"
check_sym "$PIPELINE/4_benchmark.py"                 "run_adversarial_probes" "3b adversarial runner"
check_sym "$PIPELINE/3_infererence.py"               "DependencyMonitor"      "4a dependency monitor"
check_sym "$PIPELINE/3_infererence.py"               "_DEPENDENCY_MONITOR"    "4b monitor singleton"

# =============================================================================
section "10. Network and Dataset Access"
# =============================================================================

HF_OK=$(python -c "
import urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=8)
    print('ok')
except Exception as e:
    print('fail')
" 2>/dev/null || echo "fail")

if [ "$HF_OK" = "ok" ]; then
  pass "HuggingFace Hub reachable"
else
  warn "HuggingFace Hub not reachable — model/dataset downloads will fail"
fi

if python -c "import datasets" 2>/dev/null; then
  GSM8K_OK=$(python -c "
from datasets import load_dataset
try:
    ds = load_dataset('openai/gsm8k', 'main', split='test[:3]', trust_remote_code=True)
    assert len(ds) == 3
    print('ok')
except Exception as e:
    print('fail:', e)
" 2>/dev/null || echo "fail")
  if echo "$GSM8K_OK" | grep -q "^ok"; then
    pass "GSM8K dataset accessible"
  else
    warn "GSM8K not downloadable — check network or run manually:"
    info "  python -c \"from datasets import load_dataset; load_dataset('openai/gsm8k','main')\""
  fi
else
  fail "Cannot check GSM8K — install datasets first: pip install datasets"
fi

# =============================================================================
section "11. Run Scripts"
# =============================================================================

for script_entry in \
  "experiment0_reasoning_comparison.py:Experiment 0 — reasoning comparison" \
  "run_all.sh:Master orchestration script"
do
  fname="${script_entry%%:*}"; desc="${script_entry##*:}"
  fpath="$PIPELINE/$fname"
  if [ -f "$fpath" ]; then
    pass "$desc  ($fname)"
  else
    warn "MISSING: $fname — $desc"
  fi
done

# =============================================================================
section "12. Training Status"
# =============================================================================

# Helper: check a checkpoint and extract a score field from the nearest report
check_ckpt() {
  local path="$1" label="$2"
  if [ -f "$path/adapter_config.json" ]; then
    pass "Checkpoint $label exists: $path/"
    return 0
  else
    warn "Checkpoint $label not yet trained: $path/"
    return 1
  fi
}

check_report() {
  local glob_pattern="$1" label="$2" score_key="$3"
  local found
  found=$(ls $glob_pattern 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    if [ -n "$score_key" ]; then
      SCORE=$(python -c "import json; d=json.load(open('$found')); print(round(d.get('$score_key',0),3))" 2>/dev/null || echo "?")
      pass "Report $label: $found  ($score_key=$SCORE)"
    else
      pass "Report $label: $found"
    fi
    return 0
  else
    warn "Report $label not yet generated"
    return 1
  fi
}

echo ""
echo "  Training pipeline — stages 2–8 of run_all.sh:"
echo ""

# Stage 2 — SFT
if check_ckpt "$PIPELINE/models/checkpoint_sft" "B (SFT)"; then
  SFT_DONE=1
else
  SFT_DONE=0
fi

# Stage 3 — Constitutional baseline
check_report "$PIPELINE/reports/constitution_baseline.json" "SFT constitutional baseline" "constitution_score"

# Stage 4 — Experiment 0
check_report "$PIPELINE/reports/experiment0_*.json" "Experiment 0 reasoning comparison" ""

# Stage 5 — Adversarial baseline
check_report "$PIPELINE/reports/adversarial_baseline.json" "Adversarial baseline (SFT)" "adversarial_score"

# Stage 6 — GRPO Condition C
if check_ckpt "$PIPELINE/models/checkpoint_grpo_c" "C (GRPO format+accuracy)"; then
  check_report "$PIPELINE/reports/grpo_c_drift_*.json" "GRPO-C drift report" "constitution_score"
fi

# Stage 7 — GRPO Condition D
if check_ckpt "$PIPELINE/models/checkpoint_grpo_d" "D (GRPO full composite)"; then
  check_report "$PIPELINE/reports/grpo_d_drift_*.json" "GRPO-D drift report" "constitution_score"
fi

# Stage 8 — Final ablation
check_report "$PIPELINE/reports/ablation_D_grpo_d_*.json" "Final ablation D (thesis contribution)" ""

# Progress summary
STAGES_DONE=0
[ -f "$PIPELINE/models/checkpoint_sft/adapter_config.json" ]  && ((STAGES_DONE++))
[ -f "$PIPELINE/reports/constitution_baseline.json" ]         && ((STAGES_DONE++))
[ -n "$(ls $PIPELINE/reports/experiment0_*.json 2>/dev/null)" ] && ((STAGES_DONE++))
[ -f "$PIPELINE/reports/adversarial_baseline.json" ]           && ((STAGES_DONE++))
[ -f "$PIPELINE/models/checkpoint_grpo_c/adapter_config.json" ] && ((STAGES_DONE++))
[ -f "$PIPELINE/models/checkpoint_grpo_d/adapter_config.json" ] && ((STAGES_DONE++))
[ -n "$(ls $PIPELINE/reports/ablation_D_grpo_d_*.json 2>/dev/null)" ] && ((STAGES_DONE++))

echo ""
info "Training pipeline: $STAGES_DONE / 7 stages complete"
if [ "$STAGES_DONE" -eq 0 ]; then
  info "To run the full pipeline:  bash pipeline/run_all.sh"
elif [ "$STAGES_DONE" -lt 7 ]; then
  NEXT_STAGE=$((STAGES_DONE + 1))
  info "To resume from next stage:  bash pipeline/run_all.sh --from $((STAGES_DONE + 1))"
else
  info "Full training pipeline complete."
fi

# =============================================================================
section "13. Feature Flag State"
# =============================================================================
# Read active flags from config.py via env overrides, then surface what each
# enabled flag still needs before it is runnable.

if python -c "from config import cfg; print(cfg.summary())" 2>/dev/null; then
  :
else
  warn "Could not import config.py — run from repo root or pipeline/"
fi

# Per-flag readiness checks (only fire when the flag is on)
FLAG_CHECK=$(python -c "
import sys, os
sys.path.insert(0, '$PIPELINE')
try:
    from config import cfg
    checks = []

    if cfg.ENABLE_GRPO:
        import pathlib
        if not pathlib.Path(cfg.SFT_CHECKPOINT_DIR).exists():
            checks.append('WARN|ENABLE_GRPO: checkpoint_sft not found at ' + cfg.SFT_CHECKPOINT_DIR + ' — run SFT first')
        else:
            checks.append('PASS|ENABLE_GRPO: checkpoint_sft found at ' + cfg.SFT_CHECKPOINT_DIR)

    if cfg.ENABLE_USER_MODELLING:
        import socket
        try:
            s = socket.create_connection((cfg.FALKORDB_HOST, cfg.FALKORDB_PORT), timeout=2)
            s.close()
            checks.append('PASS|ENABLE_USER_MODELLING: FalkorDB reachable at ' + cfg.FALKORDB_HOST + ':' + str(cfg.FALKORDB_PORT))
        except OSError:
            checks.append('FAIL|ENABLE_USER_MODELLING: FalkorDB not reachable at ' + cfg.FALKORDB_HOST + ':' + str(cfg.FALKORDB_PORT) + ' — run: docker compose up -d')

    if cfg.ENABLE_EMPATHY:
        import pathlib
        if pathlib.Path(cfg.APPRAISAL_LABELS_PATH).exists():
            n = sum(1 for _ in open(cfg.APPRAISAL_LABELS_PATH))
            checks.append('PASS|ENABLE_EMPATHY: appraisal_labels.jsonl found (' + str(n) + ' examples)')
        else:
            checks.append('FAIL|ENABLE_EMPATHY: ' + cfg.APPRAISAL_LABELS_PATH + ' not found — run: python pipeline/appraisal_labeller.py')

    if cfg.ENABLE_PERSONALISATION and not cfg.ENABLE_USER_MODELLING:
        checks.append('FAIL|ENABLE_PERSONALISATION: requires ENABLE_USER_MODELLING=True')

    if cfg.ENABLE_ONTOLOGY_VERIF:
        import pathlib
        if pathlib.Path(cfg.ONTOLOGY_PATH).exists():
            checks.append('PASS|ENABLE_ONTOLOGY_VERIF: ontology file found at ' + cfg.ONTOLOGY_PATH)
        else:
            checks.append('FAIL|ENABLE_ONTOLOGY_VERIF: ' + cfg.ONTOLOGY_PATH + ' not found')

    if not checks:
        checks.append('INFO|All optional modules disabled (flags default to False)')

    print('\n'.join(checks))
except Exception as e:
    print('WARN|Could not evaluate flags: ' + str(e))
" 2>/dev/null || echo "WARN|config.py not importable from current directory")

while IFS= read -r line; do
  level="${line%%|*}"; msg="${line#*|}"
  case "$level" in
    PASS) pass  "$msg" ;;
    FAIL) fail  "$msg" ;;
    WARN) warn  "$msg" ;;
    INFO) info  "$msg" ;;
  esac
done <<< "$FLAG_CHECK"

# =============================================================================
section "SUMMARY"
# =============================================================================

echo ""
echo "  Checks run : $((PASS + FAIL + WARN))"
printf "  ${GRN}%-12s${RST}: %d\n" "Passed"   "$PASS"
printf "  ${YLW}%-12s${RST}: %d\n" "Warnings" "$WARN"
printf "  ${RED}%-12s${RST}: %d\n" "Failed"   "$FAIL"
echo ""

if [ "${#MISSING_HARD[@]}" -gt 0 ]; then
  echo "${RED}  Hard installs required:${RST}"
  echo "    pip install ${MISSING_HARD[*]}"
  echo ""
fi

if [ "${#MISSING_SOFT[@]}" -gt 0 ]; then
  echo "${YLW}  Soft installs (needed for full training, optional for Experiment 0):${RST}"
  echo "    pip install ${MISSING_SOFT[*]}"
  echo ""
fi

if [ "$HARD_FAIL" -gt 0 ]; then
  echo "${RED}  RESULT: $HARD_FAIL hard requirement(s) unmet — fix before proceeding.${RST}"
  exit 1
else
  echo "${GRN}  RESULT: All hard requirements met.${RST}"
  [ "$WARN" -gt 0 ] && echo "${YLW}  ($WARN warning(s) above — review before GPU cluster runs).${RST}"
  exit 0
fi
