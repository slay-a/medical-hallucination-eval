#!/usr/bin/env bash
# run.sh — Full reproduction chain.
#   bash run.sh            # generation (needs OPENAI_API_KEY) + evaluation + analysis
#   bash run.sh --offline  # skip generation; recompute metrics, ablations, calibration, plots from stored results
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
[ -d "$VENV" ] || { echo "ERROR: run setup.sh first"; exit 1; }
source "$VENV/bin/activate"
cd "$SCRIPT_DIR"

if [ "${1:-}" != "--offline" ]; then
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "ERROR: export OPENAI_API_KEY='sk-...' (or use --offline)"; exit 1; }
  python hallucination_eval.py          # E0 baseline + E1 RAG generation and NLI evaluation (50 docs)
  python e2_extractive_eval.py          # E2 extractive baseline (no API calls)
  python e1b_fullnote_rag_eval.py       # E1b: full note + excerpts (50 calls)
  python e3_cove_eval.py                # E3: RAG + Chain-of-Verification (about 500 calls; checkpointed)
fi
python recompute_metrics.py             # header filter, per-sample metrics, paired tests
python ablations.py                     # thresholds, top-k, cleaned evidence, coverage, taxonomy (local models only)
if [ -d "$HOME/Desktop/Thesis/ann-pt-summ-1.0.1-partial" ]; then
  python calibration_annptsumm.py       # judge vs. medical-expert labels (credentialed data, local only)
fi
python analyze.py                       # figures and tables for the thesis
echo "Done. Outputs in results/"
