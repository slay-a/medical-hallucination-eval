# Hallucination Evaluation in LLM-Based Medical Summarization

Claim-level evaluation of hallucinations in patient-facing summaries generated from clinical notes, and a
controlled comparison of five summarization conditions: full-note LLM summarization without retrieval (E0), retrieval-augmented
generation with excerpts only (E1) or with the full note (E1b), RAG with Chain-of-Verification (E3), and a
centroid-based extractive baseline (E2). Every generated claim is checked against the source note with
sentence-embedding retrieval and a natural-language-inference (NLI) judge, yielding two metrics per summary:
the Unsupported Fact Rate (UFR) and the Contradiction Rate (CR).

The work has three studies:

1. **Pilot study** on 50 public MTSamples notes with GPT-4o-mini and a small MiniLM NLI judge, including the
   validation of that judge against medical-expert annotations (ann-pt-summ).
2. **Judge selection study**: six candidate judges (MiniLM, two DeBERTa-v3-large NLI models, two MiniCheck
   encoders, Bespoke-MiniCheck-7B) plus a MedNLI fine-tuned DeBERTa, scored against 1,781 expert-annotated sentences.
3. **Main study** on 110 MIMIC-IV hospital courses (ann-pt-summ) with Qwen2.5-7B-Instruct running locally:
   the five conditions with citations, coverage against the clinician-written discharge instructions,
   citation accuracy, six retrieval ablations, and a question-answering pilot with abstention.

M.S. thesis project, Department of Computer Science, California State University, Northridge
(COMP 696C Spring 2026 / COMP 698C Fall 2026). Author: Srilaya Ponangi. Advisor: Taehyung (George) Wang.

## Pipeline

```
source note ─┬─ E0  LLM, full note ──────────────────────────┐
             ├─ E1  LLM, top-3 retrieved chunks only ─────────┤
             ├─ E1b LLM, full note + retrieved chunks ────────┤─ summary ─► claim segmentation (spaCy; headers and abstentions removed)
             ├─ E3  E1 draft ► per-claim verification ► revision ┤          ─► evidence: top-3 sentences (pilot) or whole course in windows (main study)
             └─ E2  centroid extractive, top-5 sentences ─────┘          ─► NLI judge: nli-MiniLM2-L6-H768 (pilot) / DeBERTa-v3-large NLI (main study)
                                                                          ─► UFR, CR per summary ─► Wilcoxon, bootstrap CI, effect sizes
```

## Scripts

| Script | Purpose | Needs |
|---|---|---|
| `hallucination_eval.py` | Pilot: E0 and E1 generation on 50 MTSamples notes (seed 42) and NLI evaluation | OpenAI key |
| `e2_extractive_eval.py` | Pilot: E2 extractive baseline and its evaluation | – |
| `e1b_fullnote_rag_eval.py` | Pilot: E1b, RAG with the full note plus excerpts | OpenAI key |
| `e3_cove_eval.py` | Pilot: E3, RAG + Chain-of-Verification (checkpointed; `--revise-only`) | OpenAI key |
| `recompute_metrics.py` | Header and abstention filters, per-sample metrics, paired tests, bootstrap CIs | – |
| `ablations.py` | Threshold sweep, top-k, cleaned evidence, coverage proxy, negation analysis, error taxonomy | – |
| `calibration_annptsumm.py`, `calibration_variants.py` | Pilot judge vs. medical-expert annotations; five aggregation variants | ann-pt-summ (credentialed) |
| `judge_candidates.py` | Judge selection study: candidates × evidence modes vs. expert labels → `results/judge_candidates.csv` | ann-pt-summ |
| `minicheck_server.py` | Local scoring server for Bespoke-MiniCheck-7B (MLX, 8-bit; Python 3.13 environment) used as a judge candidate | `.venv-llm`, converted model |
| `finetune_mednli.py` | MedNLI fine-tuning of the best general NLI model (saved outside the repository) | MedNLI (credentialed), GPU |
| `mimic_generate.py` | Main study generation with the local model server; `--ablations` adds the six E1 variants; resumable | mlx-lm server, ann-pt-summ |
| `mimic_qa.py generate` / `evaluate` | Question-answering pilot with abstention (QA-E0, QA-E1) | mlx-lm server, ann-pt-summ |
| `mimic_extra.py` | Main study variants: length-matched E0 (`--length-matched`), E1b without citations (`--nocite-e1b`), two repeated generations of E0, E1, E1b and E3 on 40 courses (`--replicates 2 --rep-docs 40`, seeds 1000r + k) | mlx-lm server, ann-pt-summ |
| `citations.py` | Shared claim splitter: removes citation markers before sentence segmentation, assigns each marker by position, strips merged section headers | – |
| `mimic_evaluate.py` | Main study evaluation with the selected judge: UFR, CR, supported and unsupported counts, coverage vs. clinician's instructions, citation coverage, accuracy and reliability, paired tests with Holm adjustment, ablations | ann-pt-summ |
| `mimic_second_judge.py` | Scores every base-condition claim with a second judge (Bespoke-MiniCheck-7B) and reports agreement with the selected judge → `results/mimic_judge_agreement.csv` | MiniCheck server |
| `atomic_claims.py decompose` / `score` / `analyze` | Splits expert-set sentences and main-study claims into atomic facts with the local model, scores every fact, compares agreement with the experts → `results/atomic_results.csv` | mlx-lm server, ann-pt-summ |
| `judge_selection_bias.py` | Nested cross-validation of the judge selection, patient-disjoint transfer, course-clustered intervals, per-source agreement | ann-pt-summ |
| `data_overlap.py` | Matches every course to its MIMIC-IV-Note discharge note; patient and admission overlap → `results/data_overlap.csv` (counts only) | MIMIC-IV-Note |
| `mimic_revision_analysis.py` | Variant tests, run-to-run variation, abstention, coverage equivalence check, question-answering utility, per-condition intervals, splitter comparison, overlapping-window check (`--windows`) | ann-pt-summ |
| `audit_sample.py`, `audit_tool.py`, `audit_summary.py` | Blind manual audit: course-clustered stratified sample (60 courses, seed 7), local labeling page on 127.0.0.1:8765, weighted estimates of the judge's precision, false omission rate and the manually confirmed unsupported share → `results/audit_*.csv` | ann-pt-summ |
| `check_no_protected_text.py` | Scans every tracked file for eight-word sequences of the credentialed texts and of the private outputs → `results/protected_text_scan*.csv` (counts only) | ann-pt-summ, MedNLI |
| `analyze.py` | Figures (`results/fig_*.png`) and example tables | – |
| `preprocessing.py` | Shared header filter, abstention filter, MTSamples line-break repair | – |
| `thesis/build_thesis.py` | Builds the thesis (.docx and .pdf, CSUN format) from the results | LibreOffice |
| `deliverables/build_progress_slides.py`, `deliverables/build_source_data_description.py` | Course deliverables | – |

## Reproduce

```bash
bash setup.sh                          # venv + pinned dependencies (requirements.txt) + spaCy model 3.7.1
bash setup.sh --llm                    # also the model-server environment .venv-llm (requirements-llm.txt; Apple silicon)
# place mtsamples.csv one directory above this repository
export OPENAI_API_KEY='sk-...'
bash run.sh                            # pilot: generation + evaluation + analysis
bash run.sh --offline                  # pilot without generation, from the stored results
bash run.sh --main-study               # judge study, MedNLI fine-tuning, MIMIC-IV main study, QA pilot (local models)
```

The main study needs the credentialed data under `~/Desktop/Thesis` and a second environment for the model
server (Python 3.13, `pip install mlx-lm`):

```bash
.venv-llm/bin/python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8080
```

The Bespoke-MiniCheck-7B judge candidate is converted once with `mlx_lm.convert --hf-path <HF snapshot dir> --mlx-path <dir> -q --q-bits 8 --trust-remote-code`;
copy `tokenizer.json`, `tokenizer.model` and `tokenizer_config.json` from the Hugging Face snapshot into the converted directory afterwards
(the converter re-saves a broken tokenizer under transformers 5), then serve it with `.venv-llm/bin/python minicheck_server.py --model <dir> --port 8081`.

`requirements-lock.txt` and `requirements-llm-lock.txt` list both environments exactly as used. Seeds: 42 for the MTSamples
sample, the MedNLI fine-tuning, the repeated-generation subset and the selection-bias analyses; 1000r + k for course k of
repeated generation r; 7 for the audit sample; 0 for the pilot bootstrap. The LLMs are called with temperature 0.3, so
regenerated summaries differ slightly from the stored ones, but the stored `results/summaries.csv` reproduces every pilot
number exactly through `recompute_metrics.py`.

| Result | Data needed | In this repository |
|---|---|---|
| Pilot study | MTSamples (public) | everything; `bash run.sh --offline` |
| Judge validation and selection | ann-pt-summ (PhysioNet, credentialed) | aggregate statistics only |
| MedNLI adaptation | MedNLI (PhysioNet, credentialed) | script and summary statistics |
| Main study and robustness checks | ann-pt-summ courses (from MIMIC-IV-Note) | aggregates in `results/mimic_*.csv` |
| Data overlap check | MIMIC-IV-Note (credentialed) | counts only |
| Manual audit | as the main study | aggregates only |

Main-study generations, claims, labels and audit items contain MIMIC-derived text and are stored only in the
git-ignored `results_private/`.

## Data

* **MTSamples** (public): `../mtsamples.csv` (17 MB, 4,999 transcriptions). 50 documents sampled from the
  624 Discharge Summary and Consult - History and Physical notes.
* **ann-pt-summ v1.0.1** (PhysioNet, credentialed): 210 expert-annotated patient summaries used to validate and
  select the judge, and the 110 MIMIC-IV hospital courses with clinician-written discharge instructions that form
  the main-study corpus. Processed with local models only.
* **MedNLI v1.0.0** (PhysioNet, credentialed): 14,049 clinician-written premise and hypothesis pairs used to
  fine-tune the judge.
* **MIMIC-IV-Note v2.2** (PhysioNet, credentialed): the source of the hospital courses; staged locally, never committed.

No credentialed text is committed, uploaded to shared storage, or sent to any external API.

## Manual audit of the judge

`python audit_sample.py` draws 60 courses (seed 7) and, within them, up to 30 claims flagged and 30 accepted by the judge for each
of E0, E1, E1b and E3, plus 30 cited claims of E1 and of E1b → `results_private/audit_items.json`. `python audit_tool.py` serves a
local labeling page at http://127.0.0.1:8765 that shows one course at a time and hides the condition and the judge's label; labels
(supported, partly supported, not supported, unsure, with an error type) are saved to `results_private/audit_labels.json`. The
protocol is Appendix F of the thesis. After the main-study evaluation, `python audit_sample.py --refresh-judge` stores the judge's
current label of every audited claim, and `python audit_summary.py` writes `results/audit_summary.csv`, `audit_types.csv` and
`audit_citations.csv` (Section 7.9).

## Results files

Pilot: `results/claims_all.csv` (one row per claim: condition, claim, evidence, NLI label, probabilities,
is_header, is_abstention), `results/summaries.csv`, `results/comparison_per_sample.csv`,
`results/aggregate_statistics.csv`, `results/pairwise_tests.csv`, ablation and calibration CSVs.
Judge study: `results/judge_candidates.csv`. Main study: `results/mimic_summary.csv`,
`results/mimic_pairwise_tests.csv`, `results/mimic_ablations.csv`, `results/mimic_reference.csv`,
`results/mimic_qa_summary.csv`, `results/mimic_qa_utility.csv`, `results/mimic_condition_ci.csv`, `results/mimic_variant_tests.csv`,
`results/mimic_replicates.csv`, `results/mimic_abstention.csv`, `results/mimic_equivalence.csv`, `results/mimic_window_check.csv`,
`results/mimic_splitting_check.csv`, `results/mimic_judge_agreement.csv`, `results/mimic_second_judge_tests.csv`,
`results/mimic_tau05/` (threshold 0.5), `results/atomic_results.csv`, `results/judge_selection_bias.csv`,
`results/judge_selection_frequency.csv`, `results/judge_per_source.csv`, `results/data_overlap.csv`, `results/audit_*.csv`,
`results/protected_text_scan*.csv`. Figures: `results/fig_*.png`.
