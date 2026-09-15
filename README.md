# Hallucination Evaluation in LLM-Based Medical Summarization

Claim-level evaluation of hallucinations in patient-facing summaries generated from clinical notes,
and a controlled comparison of three summarization approaches (zero-context GPT-4o-mini, retrieval-augmented
GPT-4o-mini, and a centroid-based extractive baseline). Every generated claim is checked against the source
note with sentence-embedding retrieval and a natural-language-inference cross-encoder, yielding two metrics per
summary: the Unsupported Fact Rate (UFR) and the Contradiction Rate (CR).

M.S. thesis project, Department of Computer Science, California State University, Northridge
(COMP 696C Spring 2026 / COMP 698C Fall 2026). Author: Srilaya Ponangi. Advisor: Taehyung (George) Wang.

## Pipeline

```
source note ─┬─ E0  GPT-4o-mini, full note ──────────┐
             ├─ E1  GPT-4o-mini, top-3 retrieved chunks ─┤─ summary ─► claim segmentation (spaCy, header lines removed)
             └─ E2  centroid extractive, top-5 sentences ─┘             ─► evidence retrieval (all-MiniLM-L6-v2, top-3 sentences)
                                                                        ─► NLI cross-encoder (nli-MiniLM2-L6-H768)
                                                                        ─► UFR, CR per summary ─► Wilcoxon, bootstrap CI
```

## Scripts

| Script | Purpose | Needs API key |
|---|---|---|
| `hallucination_eval.py` | E0 and E1 generation on 50 MTSamples notes (seed 42) and NLI evaluation | yes |
| `e2_extractive_eval.py` | E2 extractive baseline and its evaluation | no |
| `e1b_fullnote_rag_eval.py` | E1b: RAG with the full note plus excerpts (implemented, not yet run) | yes |
| `e3_cove_eval.py` | E3: RAG + Chain-of-Verification (implemented, not yet run) | yes |
| `recompute_metrics.py` | Header filter, per-sample metrics, paired tests, bootstrap CIs | no |
| `ablations.py` | Threshold sweep, top-k ablation, cleaned-evidence ablation, coverage proxy, negation analysis, error taxonomy | no |
| `calibration_annptsumm.py` | Judge validation against medical-expert annotations (ann-pt-summ, credentialed; local models only) | no |
| `analyze.py` | Figures (`results/fig_*.png`) and example tables | no |
| `preprocessing.py` | Shared header filter and MTSamples line-break repair | — |
| `thesis/build_thesis.py` | Builds the thesis (.docx and .pdf, CSUN format) from the results | no |
| `deliverables/build_progress_slides.py`, `deliverables/build_source_data_description.py` | Course deliverables | no |

## Reproduce

```bash
bash setup.sh                          # venv + dependencies + spaCy model
# place mtsamples.csv one directory above this repository
export OPENAI_API_KEY='sk-...'
bash run.sh                            # generation + evaluation + analysis
bash run.sh --offline                  # everything except generation, from the stored results
```

Random seed 42 is fixed for document sampling; GPT-4o-mini is called with temperature 0.3, so regenerated
summaries differ slightly from the stored ones, but the stored `results/summaries.csv` reproduces every
number in the thesis exactly through `recompute_metrics.py`.

## Data

* **MTSamples** (public): `../mtsamples.csv` (17 MB, 4,999 transcriptions). 50 documents sampled from the
  624 Discharge Summary and Consult - History and Physical notes.
* **MIMIC-IV-Note v2.2** (PhysioNet, credentialed): staged locally for future work; never committed.
* **ann-pt-summ v1.0.1** (PhysioNet, credentialed): 210 expert-annotated patient summaries used only to
  validate the judge, processed with local models only; per-sentence outputs go to `results_private/`
  (git-ignored). Only aggregate statistics are in `results/calibration_*.csv`.

## Results files

`results/claims_all.csv` (one row per claim: condition, claim, evidence, NLI label, probabilities, is_header),
`results/summaries.csv`, `results/comparison_per_sample.csv`, `results/aggregate_statistics.csv`,
`results/pairwise_tests.csv`, ablation and calibration CSVs, and `results/fig_*.png`.
