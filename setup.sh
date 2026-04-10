#!/usr/bin/env bash
# setup.sh — Create venv and install all dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

echo "────────────────────────────────────────────────────────────"
echo " Medical Hallucination Eval — Environment Setup"
echo "────────────────────────────────────────────────────────────"

# 1. Create virtual environment
if [ ! -d "$VENV" ]; then
    echo "[1/4] Creating virtual environment at $VENV …"
    python3 -m venv "$VENV"
else
    echo "[1/4] Virtual environment already exists — skipping."
fi

# 2. Activate
source "$VENV/bin/activate"
echo "[2/4] Virtual environment activated."

# 3. Upgrade pip, install requirements
echo "[3/4] Installing Python packages …"
pip install --upgrade pip --quiet
pip install "spacy==3.7.4"          # pin before installing the rest
pip install -r "$SCRIPT_DIR/requirements.txt"

# 4. Download spaCy model
echo "[4/4] Downloading spaCy model en_core_web_sm …"
python -m spacy download en_core_web_sm

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  1. Export your API key:  export OPENAI_API_KEY='sk-...'"
echo "  2. Run:                  source .venv/bin/activate && python hallucination_eval.py"
echo "     OR use the helper:    bash run.sh"
