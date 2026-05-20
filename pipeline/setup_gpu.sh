#!/usr/bin/env bash
# =============================================================================
# setup_gpu.sh — One-shot setup for the pipeline on a GPU machine
# =============================================================================
# Run this once after cloning the repo. It installs Python deps, creates the
# .env template, seeds training data from smoke_gold.jsonl (so you can run
# Stage 2 without needing API keys), and validates that CUDA + the key
# imports work.
#
# Usage:
#   bash setup_gpu.sh                  # full setup
#   bash setup_gpu.sh --skip-torch      # if PyTorch already installed
#   bash setup_gpu.sh --smoke-only      # seed data + validate only
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ENV_FILE="$REPO_ROOT/.env"

SKIP_TORCH=0
SMOKE_ONLY=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-torch)  SKIP_TORCH=1; shift;;
    --smoke-only)  SMOKE_ONLY=1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED=""; GRN=""; YLW=""; CYN=""; RST=""
if [ -t 1 ] && command -v tput &>/dev/null 2>&1; then
  RED=$(tput setaf 1 2>/dev/null||true); GRN=$(tput setaf 2 2>/dev/null||true)
  YLW=$(tput setaf 3 2>/dev/null||true); CYN=$(tput setaf 6 2>/dev/null||true)
  RST=$(tput sgr0 2>/dev/null||true)
fi
ok()   { echo "  ${GRN}[OK]${RST}   $1"; }
info() { echo "  ${CYN}[INFO]${RST} $1"; }
warn() { echo "  ${YLW}[WARN]${RST} $1"; }
fail() { echo "  ${RED}[FAIL]${RST} $1"; exit 1; }
step() { echo ""; echo "${CYN}── $1 ──${RST}"; }

# =============================================================================
# 0. Preflight
# =============================================================================

step "Preflight"

python3 --version &>/dev/null || fail "python3 not found — install Python 3.10+"
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
[ "$PY_MINOR" -ge 10 ] || fail "Python 3.10+ required (got 3.$PY_MINOR)"
ok "Python 3.$PY_MINOR"

if ! command -v nvidia-smi &>/dev/null; then
  warn "nvidia-smi not found — continuing, but GPU training will fail without CUDA"
else
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  ok "GPU detected: $GPU_NAME"
fi

# Detect CUDA version for torch wheel selection
CUDA_VER=""
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
  ok "CUDA $CUDA_VER detected via nvcc"
elif nvidia-smi &>/dev/null 2>&1; then
  CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
  [ -n "$CUDA_VER" ] && ok "CUDA $CUDA_VER detected via nvidia-smi"
fi

if [ -z "$CUDA_VER" ]; then
  warn "Could not detect CUDA version — will install torch with default CUDA index"
fi

# =============================================================================
# 1. Install PyTorch (skip if --skip-torch)
# =============================================================================

if [ "$SMOKE_ONLY" -eq 0 ] && [ "$SKIP_TORCH" -eq 0 ]; then
  step "Installing PyTorch"

  # Check if torch is already present with CUDA support
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    ok "torch $TORCH_VER already installed with CUDA — skipping"
    SKIP_TORCH=1
  fi
fi

if [ "$SMOKE_ONLY" -eq 0 ] && [ "$SKIP_TORCH" -eq 0 ]; then
  CUDA_MAJOR="${CUDA_VER%%.*}"
  CUDA_MINOR_RAW="${CUDA_VER##*.}"
  CUDA_MINOR="${CUDA_MINOR_RAW%%.*}"

  # Pick the right torch index URL
  if [ "$CUDA_MAJOR" = "12" ]; then
    if [ "$CUDA_MINOR" -ge 4 ]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_MINOR" -ge 1 ]; then
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
      TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
  elif [ "$CUDA_MAJOR" = "11" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  else
    warn "CUDA $CUDA_VER — using default torch index (may not have CUDA support)"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  fi

  info "Installing torch from $TORCH_INDEX"
  pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" \
    || fail "torch install failed — check CUDA version and try manually"
  ok "torch installed"
fi

# =============================================================================
# 2. Install unsloth
# =============================================================================

if [ "$SMOKE_ONLY" -eq 0 ]; then
  step "Installing unsloth"

  if python3 -c "import unsloth" 2>/dev/null; then
    ok "unsloth already installed — skipping"
  else
    # Try pip first; fall back to git install
    pip install unsloth 2>/dev/null \
      || pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" \
      || fail "unsloth install failed — see https://github.com/unslothai/unsloth#installation"
    ok "unsloth installed"
  fi
fi

# =============================================================================
# 3. Install remaining pipeline deps
# =============================================================================

if [ "$SMOKE_ONLY" -eq 0 ]; then
  step "Installing pipeline dependencies"

  pip install \
    trl \
    transformers \
    datasets \
    litellm \
    python-dotenv \
    pyyaml \
    fastapi \
    uvicorn \
    pydantic \
    falkordb \
    || fail "pip install failed"

  ok "Pipeline deps installed"
fi

# =============================================================================
# 4. Create directory structure
# =============================================================================

step "Creating directories"
mkdir -p "$DATA_DIR" "$SCRIPT_DIR/models" "$SCRIPT_DIR/reports"
ok "data, models, reports"

# =============================================================================
# 5. Seed training data from smoke file (skips Stage 1 / API key requirement)
# =============================================================================

step "Seeding training data"

SMOKE="$DATA_DIR/smoke_gold.jsonl"
TRAIN="$DATA_DIR/train_interleaved.jsonl"

if [ -f "$TRAIN" ]; then
  N=$(wc -l < "$TRAIN" | tr -d ' ')
  ok "train_interleaved.jsonl already exists ($N examples) — skipping seed"
elif [ -f "$SMOKE" ]; then
  cp "$SMOKE" "$TRAIN"
  N=$(wc -l < "$TRAIN" | tr -d ' ')
  ok "Seeded train_interleaved.jsonl from smoke_gold.jsonl ($N examples)"
  warn "This is smoke data only. For a real run, generate full data via Stage 1 (needs API keys)."
else
  warn "smoke_gold.jsonl not found — train_interleaved.jsonl not created."
  warn "Stage 2 (SFT training) will fail. Run Stage 1 first or place training data manually."
fi

# =============================================================================
# 6. Create .env template
# =============================================================================

step "Environment file"

if [ -f "$ENV_FILE" ]; then
  ok ".env already exists at $ENV_FILE — not overwriting"
else
  cat > "$ENV_FILE" <<'EOF'
# =============================================================================
# Pipeline environment variables
# Copy this file to .env and fill in the values you need.
# =============================================================================

# ── Stage 1: Data generation (skip if using smoke data for Stage 2 testing) ──
# Required by sft_question_generator.py, sft_gold_response_generator.py,
# sft_math_question_generator.py when calling NVIDIA NIM.
NVIDIA_API_KEY=

# Required by sft_gold_response_generator.py for the critic model (claude-opus-4-7)
ANTHROPIC_API_KEY=

# ── FalkorDB (only if PIPELINE_ENABLE_USER_MODELLING=true) ───────────────────
# For local Docker: leave HOST=localhost, PORT=6379, PASSWORD empty.
# For FalkorDB Cloud: set HOST to your instance host, PASSWORD to your token.
PIPELINE_FALKORDB_HOST=localhost
PIPELINE_FALKORDB_PORT=6379
PIPELINE_FALKORDB_USER=
PIPELINE_FALKORDB_PASSWORD=
PIPELINE_FALKORDB_GRAPH_NAME=user_model

# ── Feature flags (all default to false; set true to enable) ─────────────────
PIPELINE_ENABLE_USER_MODELLING=false
PIPELINE_ENABLE_EMPATHY=false
PIPELINE_ENABLE_GRPO=false
PIPELINE_ENABLE_PERSONALISATION=false
PIPELINE_ENABLE_ONTOLOGY_VERIF=false
EOF
  ok ".env created at $ENV_FILE — fill in your API keys before Stage 1"
fi

# =============================================================================
# 7. Validate
# =============================================================================

step "Validation"

python3 - <<'PYEOF'
import sys, importlib

checks = [
    ("torch",        lambda: __import__("torch").__version__),
    ("transformers", lambda: __import__("transformers").__version__),
    ("datasets",     lambda: __import__("datasets").__version__),
    ("trl",          lambda: __import__("trl").__version__),
    ("unsloth",      lambda: __import__("unsloth").__version__ if hasattr(__import__("unsloth"), "__version__") else "ok"),
    ("litellm",      lambda: __import__("litellm").__version__),
    ("falkordb",     lambda: __import__("falkordb").__version__ if hasattr(__import__("falkordb"), "__version__") else "ok"),
    ("fastapi",      lambda: __import__("fastapi").__version__),
    ("uvicorn",      lambda: __import__("uvicorn").__version__),
]

ok = True
for name, version_fn in checks:
    try:
        ver = version_fn()
        print(f"  \033[32m[OK]\033[0m   {name} {ver}")
    except Exception as e:
        print(f"  \033[31m[MISS]\033[0m {name} — {e}")
        ok = False

import torch
if torch.cuda.is_available():
    print(f"  \033[32m[OK]\033[0m   CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
else:
    print(f"  \033[33m[WARN]\033[0m CUDA not available via torch — GPU training will fail")

sys.exit(0 if ok else 1)
PYEOF

VALIDATE_EXIT=$?

echo ""
if [ $VALIDATE_EXIT -eq 0 ]; then
  ok "All imports validated"
else
  warn "Some packages missing — re-run setup or install manually"
fi

# =============================================================================
# Done
# =============================================================================

echo ""
echo "${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
echo "${GRN}  Setup complete.${RST}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Fill in .env (at least ANTHROPIC_API_KEY if running Stage 1)"
echo ""
echo "  2. Quick GPU smoke test (Stage 2 only, uses seeded smoke data):"
echo "       bash run_all.sh --stages 2"
echo ""
echo "  3. Full run from SFT:"
echo "       bash run_all.sh --from 2"
echo ""
echo "  4. To enable FalkorDB Cloud (user modelling):"
echo "       Edit .env → set PIPELINE_FALKORDB_HOST, PIPELINE_FALKORDB_PASSWORD"
echo "       Set PIPELINE_ENABLE_USER_MODELLING=true"
echo "       bash run_all.sh --from 2"
echo ""
echo "  Log: run_all.log"
echo "${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
