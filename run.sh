#!/usr/bin/env bash
# run.sh — Activate venv and launch the evaluation pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh first."
    exit 1
fi

source "$VENV/bin/activate"

# Prefer the env-var key; fall back to the hardcoded placeholder check
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    echo "  Run:  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "Starting evaluation pipeline …"
echo "  Model  : gpt-4o-mini"
echo "  Samples: 50"
echo "  Output : $SCRIPT_DIR/results/"
echo ""

python "$SCRIPT_DIR/hallucination_eval.py" "$@"
