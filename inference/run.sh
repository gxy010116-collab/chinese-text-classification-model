#!/usr/bin/env bash
# ============================================================================
# One-click pipeline: train + evaluate Chinese text classification
# ============================================================================
# Usage:
#   bash run.sh                    # FP32 on auto-detected device
#   bash run.sh --quantize         # INT8 dynamic quantization
#   bash run.sh --device cpu       # Force CPU
#   bash run.sh --skip-train       # Skip training (use existing checkpoint)
#
# This script:
#   1. Creates a Python virtual environment (if not already present)
#   2. Installs dependencies from requirements.txt
#   3. Generates training data & fine-tunes the model (if no checkpoint exists)
#   4. Runs the evaluation script on the fine-tuned model
#   5. Outputs classification report with accuracy / precision / recall / F1
#   6. Saves evaluation results JSON + run log under results/
# ============================================================================

set -euo pipefail

# ---- helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

step()  { printf "${CYAN}[STEP]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}   %s\n" "$*"; }
warn()  { printf "${RED}[WARN]${NC} %s\n" "$*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"
CHECKPOINT="$SCRIPT_DIR/checkpoints/pytorch_model.bin"

SKIP_TRAIN=false
EVAL_ARGS=()

# Parse known flags; pass unknown args to eval.py
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train) SKIP_TRAIN=true; shift ;;
        *) EVAL_ARGS+=("$1"); shift ;;
    esac
done

# ---- 1. venv ----
if [ ! -d "$VENV_DIR" ]; then
    step "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR" || { echo "ERROR: python3 not found. Install Python 3.9+ first." >&2; exit 1; }
    ok "venv created at $VENV_DIR"
else
    ok "Using existing venv at $VENV_DIR"
fi

# ---- 2. install dependencies ----
step "Installing dependencies from requirements.txt..."
"$PIP" install --upgrade pip -q
"$PIP" install -r requirements.txt -q
ok "Dependencies installed"

# ---- 3. train (if needed) ----
if [ "$SKIP_TRAIN" = false ] && [ ! -f "$CHECKPOINT" ]; then
    mkdir -p "$SCRIPT_DIR/results"
    step "No checkpoint found. Starting fine-tuning..."
    step "  (This will generate ~4000 training samples and fine-tune for 3 epochs."
    step "   On CPU this may take 30-60 minutes. Use --skip-train to skip.)"
    echo ""
    # Training is pinned to CPU: GPU/MPS training is non-deterministic across
    # backends and would break the documented seed=42 reproducibility contract.
    "$PYTHON" train.py --epochs 3 --samples 400 --seed 42 --max-length 48 --device cpu 2>&1 | tee results/train_run.log
    ok "Training complete. Checkpoint saved to checkpoints/"
elif [ "$SKIP_TRAIN" = true ]; then
    step "Skipping training (--skip-train). Using existing checkpoint if available."
elif [ -f "$CHECKPOINT" ]; then
    ok "Checkpoint found at $CHECKPOINT — skipping training."
fi

# ---- 4. run evaluation ----
mkdir -p "$SCRIPT_DIR/results"
step "Running evaluation on fine-tuned model..."
echo ""
if [ ${#EVAL_ARGS[@]} -gt 0 ]; then
    "$PYTHON" eval.py --output "$SCRIPT_DIR/results/eval_results.json" "${EVAL_ARGS[@]}" 2>&1 | tee results/eval_run.log
else
    "$PYTHON" eval.py --output "$SCRIPT_DIR/results/eval_results.json" 2>&1 | tee results/eval_run.log
fi

ok "Done. Results saved to results/ (eval_results.json, eval_run.log)."
