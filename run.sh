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
  python judge_candidates.py                       # candidate judges vs. expert labels -> results/judge_candidates.csv
  python finetune_mednli.py                        # MedNLI fine-tuning of the best general model (GPU; run alone)
  python judge_candidates.py --only mednli_deberta_large
  python judge_selection_bias.py                   # nested cross-validation, patient-disjoint transfer, per-source agreement
  python data_overlap.py                           # patient and admission overlap (needs MIMIC-IV-Note discharge.csv.gz)
  curl -sf http://127.0.0.1:8080/v1/models >/dev/null || { echo "ERROR: start the mlx-lm server first (see comment above)"; exit 1; }
  python mimic_generate.py --ablations             # 110 courses x (E0, E1 + 6 variants, E1b, E2, E3); resumable
  python mimic_extra.py --length-matched --nocite-e1b
  python mimic_extra.py --replicates 2 --rep-docs 40
  python mimic_qa.py generate                      # 3 questions x 2 conditions x 110 courses; resumable
  python atomic_claims.py decompose --set expert
  python atomic_claims.py decompose --set main
  python mimic_evaluate.py --tau 0.5               # threshold-0.5 run, archived under results/mimic_tau05/
  mkdir -p results/mimic_tau05 && cp results/mimic_summary.csv results/mimic_pairwise_tests.csv results/mimic_ablations.csv results/mimic_reference.csv results/mimic_tau05/
  python mimic_evaluate.py --reuse-forward --reuse-reverse   # selected judge and threshold; UFR, CR, coverage, citations, tests
  python mimic_qa.py evaluate
  # second judge: serve Bespoke-MiniCheck-7B first (README), on port 8081
  curl -sf http://127.0.0.1:8081/health >/dev/null && python mimic_second_judge.py || echo "skip: MiniCheck server not running"
  python atomic_claims.py score
  python atomic_claims.py analyze
  python mimic_revision_analysis.py --windows
  [ -f results_private/audit_items.json ] || python audit_sample.py   # then label with audit_tool.py (Appendix F of the thesis)
  [ -f results_private/audit_labels.json ] && python audit_sample.py --refresh-judge && python audit_summary.py
  python analyze.py
  python check_no_protected_text.py
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
