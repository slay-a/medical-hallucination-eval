#!/usr/bin/env bash
# run.sh — Full reproduction chain.
#   bash run.sh               # pilot generation (needs OPENAI_API_KEY) + evaluation + analysis
#   bash run.sh --offline     # skip generation; recompute metrics, ablations, calibration, plots from stored results
#   bash run.sh --main-study  # judge selection, MedNLI fine-tuning, MIMIC-IV main study and QA pilot (credentialed
#                             # data on this machine, local models only; needs the mlx-lm server, see below)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
[ -d "$VENV" ] || { echo "ERROR: run setup.sh first"; exit 1; }
source "$VENV/bin/activate"
cd "$SCRIPT_DIR"
MODE="${1:-}"

if [ "$MODE" = "--main-study" ]; then
  # Requires: ann-pt-summ and MedNLI under ~/Desktop/Thesis (PhysioNet credentialed; never copied elsewhere) and a
  # separate Python 3.13 environment .venv-llm with mlx-lm, serving Qwen2.5-7B-Instruct-4bit:
  #   .venv-llm/bin/python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8080
  python judge_candidates.py                       # five candidate judges vs. expert labels -> results/judge_candidates.csv
  python finetune_mednli.py                        # MedNLI fine-tuning of the best general model (GPU; run alone)
  python judge_candidates.py --only mednli_deberta_large
  curl -sf http://127.0.0.1:8080/v1/models >/dev/null || { echo "ERROR: start the mlx-lm server first (see comment above)"; exit 1; }
  python mimic_generate.py --ablations             # 110 courses x (E0, E1 + 6 variants, E1b, E2, E3); resumable
  python mimic_qa.py generate                      # 3 questions x 2 conditions x 110 courses; resumable
  python mimic_evaluate.py                         # best judge by kappa; UFR, CR, coverage, citation accuracy, tests
  python mimic_qa.py evaluate
  python analyze.py
  echo "Done. Aggregates in results/mimic_*.csv; protected per-claim outputs in results_private/ (git-ignored)"
  exit 0
fi

if [ "$MODE" != "--offline" ]; then
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
  python calibration_variants.py
fi
python analyze.py                       # figures and tables for the thesis
echo "Done. Outputs in results/"
