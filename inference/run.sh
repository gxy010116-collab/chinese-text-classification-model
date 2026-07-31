#!/usr/bin/env bash
# ============================================================================
# One-click evaluation pipeline for Chinese text classification
# ============================================================================
# Usage:
#   bash run.sh              # FP32 on auto-detected device
#   bash run.sh --quantize   # INT8 dynamic quantization
#   bash run.sh --device cpu # Force CPU
#
# This script:
#   1. Creates a Python virtual environment (if not already present)
#   2. Installs dependencies from requirements.txt
#   3. Runs the evaluation script
#   4. Outputs classification report with accuracy / precision / recall / F1
# ============================================================================

set -euo pipefail

# ---- helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

step()  { printf "${CYAN}[STEP]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}   %s\n" "$*"; }
err()   { printf "${RED}[ERR]${NC}  %s\n" "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

# ---- 1. venv ----
if [ ! -d "$VENV_DIR" ]; then
    step "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR" || err "Failed to create venv (python3 not found?)"
    ok "venv created at $VENV_DIR"
else
    ok "Using existing venv at $VENV_DIR"
fi

# ---- 2. install dependencies ----
step "Installing dependencies from requirements.txt..."
"$PIP" install --upgrade pip -q
"$PIP" install -r requirements.txt -q
ok "Dependencies installed"

# ---- 3. run evaluation ----
step "Running evaluation..."
"$PYTHON" eval.py "$@" 2>&1

ok "Done."
