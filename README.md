# Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization

Code and results for my M.S. thesis in the Department of Computer Science at California State University,
Northridge (COMP 696C Spring 2026 and COMP 698C Fall 2026).
Author: Srilaya Ponangi. Advisor: Dr. Taehyung (George) Wang.

The thesis asks how often an LLM writes something the clinical note does not support when it summarizes the
note for a patient, and whether retrieval and verification bring that number down. Every claim in a summary is
checked against the source note. Sentence embeddings find the evidence and a natural language inference (NLI)
model acts as the judge. Each summary gets two numbers: the Unsupported Fact Rate (UFR) and the Contradiction
Rate (CR).

## The five conditions

| Condition | What it does |
|---|---|
| E0 | The LLM reads the whole note. No retrieval. |
| E1 | Retrieval augmented generation (RAG). The LLM sees only the top 3 retrieved chunks. |
| E1b | RAG with the whole note and the retrieved chunks. |
| E3 | The E1 draft, then each claim is verified and the summary is revised (Chain-of-Verification). |
| E2 | No LLM. An extractive baseline that takes the 5 sentences closest to the centroid. |

## The three studies

1. **Pilot study.** 50 public MTSamples notes with GPT-4o-mini and a small MiniLM NLI judge. It also validates
   that judge against annotations by medical experts (ann-pt-summ).
2. **Judge selection study.** Six candidate judges (MiniLM, two DeBERTa-v3-large NLI models, two MiniCheck
   encoders, Bespoke-MiniCheck-7B) and a DeBERTa fine tuned on MedNLI, scored against 1,781 sentences annotated
   by experts.
3. **Main study.** 110 MIMIC-IV hospital courses (ann-pt-summ) with Qwen2.5-7B-Instruct running locally. It has
   the five conditions with citations, coverage against the discharge instructions written by clinicians,
   citation accuracy, six retrieval ablations, and a question answering pilot with abstention.

## How a summary is scored

1. The summary is split into claims with spaCy. Headers and abstentions are removed.
2. Evidence is picked for each claim. The pilot takes the top 3 sentences of the note. The main study takes the
   whole hospital course in windows.
3. The NLI judge labels the claim against the evidence. The pilot uses nli-MiniLM2-L6-H768. The main study uses
   a DeBERTa-v3-large NLI model.
4. UFR and CR are computed for each summary.
5. The conditions are compared with Wilcoxon tests, bootstrap confidence intervals and effect sizes.

## Scripts

| Script | Purpose | Needs |
|---|---|---|
| `hallucination_eval.py` | Pilot: generates E0 and E1 for 50 MTSamples notes (seed 42) and runs the NLI evaluation | OpenAI key |
| `e2_extractive_eval.py` | Pilot: the E2 extractive baseline and its evaluation | none |
| `e1b_fullnote_rag_eval.py` | Pilot: E1b, RAG with the whole note plus excerpts | OpenAI key |
| `e3_cove_eval.py` | Pilot: E3, RAG with Chain-of-Verification (checkpointed, `--revise-only`) | OpenAI key |
| `recompute_metrics.py` | Header and abstention filters, metrics per sample, paired tests, bootstrap intervals | none |
| `ablations.py` | Threshold sweep, top k, cleaned evidence, coverage proxy, negation analysis, error taxonomy | none |
| `calibration_annptsumm.py`, `calibration_variants.py` | Pilot judge against the expert annotations, with five aggregation variants | ann-pt-summ (credentialed) |
| `judge_candidates.py` | Judge selection study: every candidate and evidence mode against the expert labels. Writes `results/judge_candidates.csv` | ann-pt-summ |
| `minicheck_server.py` | Local scoring server for Bespoke-MiniCheck-7B (MLX, 8 bit, Python 3.13 environment), one of the judge candidates | `.venv-llm`, converted model |
| `finetune_mednli.py` | Fine tunes the best general NLI model on MedNLI. The model is saved outside the repository | MedNLI (credentialed), GPU |
| `mimic_generate.py` | Main study generation with the local model server. `--ablations` adds the six E1 variants. Resumable | mlx-lm server, ann-pt-summ |
| `mimic_qa.py generate` / `evaluate` | Question answering pilot with abstention (QA-E0, QA-E1) | mlx-lm server, ann-pt-summ |
| `mimic_extra.py` | Main study variants: E0 matched in length (`--length-matched`), E1b without citations (`--nocite-e1b`), and two repeated generations of E0, E1, E1b and E3 on 40 courses (`--replicates 2 --rep-docs 40`, seeds 1000r + k) | mlx-lm server, ann-pt-summ |
| `citations.py` | Shared claim splitter. Removes citation markers before sentence splitting, assigns each marker by position, and strips merged section headers | none |
| `mimic_evaluate.py` | Main study evaluation with the selected judge: UFR, CR, supported and unsupported counts, coverage against the clinician's instructions, citation coverage, accuracy and reliability, paired tests with Holm adjustment, ablations | ann-pt-summ |
| `mimic_second_judge.py` | Scores every claim of the base conditions with a second judge (Bespoke-MiniCheck-7B) and reports its agreement with the selected judge. Writes `results/mimic_judge_agreement.csv` | MiniCheck server |
| `atomic_claims.py decompose` / `score` / `analyze` | Splits the expert sentences and the main study claims into atomic facts with the local model, scores every fact, and compares agreement with the experts. Writes `results/atomic_results.csv` | mlx-lm server, ann-pt-summ |
| `judge_selection_bias.py` | Nested cross validation of the judge selection, transfer to separate patients, intervals clustered by course, agreement per source | ann-pt-summ |
| `data_overlap.py` | Matches every course to its MIMIC-IV-Note discharge note and counts the patient and admission overlap. Writes `results/data_overlap.csv` (counts only) | MIMIC-IV-Note |
| `mimic_revision_analysis.py` | Variant tests, variation between runs, abstention, coverage equivalence check, question answering utility, intervals per condition, splitter comparison, overlapping window check (`--windows`) | ann-pt-summ |
| `audit_sample.py`, `audit_tool.py`, `audit_map_types.py`, `audit_summary.py`, `audit_agreement.py` | Blind manual audit. A stratified sample clustered by course (60 courses, seed 7), a local labeling page on 127.0.0.1:8765, and weighted estimates of the judge's precision, its false omission rate and the share of claims I confirmed as unsupported. Writes `results/audit_*.csv` | ann-pt-summ |
| `verifact_eval.py` | External check of the selected judge on MIMIC-III-Ext-VeriFact-BHC, which has 12,614 propositions with adjudicated clinician labels. `--score --premise admission` scores each proposition against the five best matching windows (400 tokens each) of the admission's notes, with the main study's windowing and threshold. `--summarize` computes the metrics against the clinician labels and checks the cap of five windows against scoring all windows. `--compare-retrievers` compares the window rankings. `--baseline` reports the published AI raters. Writes `results/verifact_judge_eval.csv`, `results/verifact_retriever_check.csv` and `results/verifact_ai_raters.csv` (metrics only) | VeriFact-BHC |
| `check_no_protected_text.py` | Scans every tracked file for sequences of eight words from the credentialed texts and the private outputs. Writes `results/protected_text_scan*.csv` (counts only) | ann-pt-summ, MedNLI |
| `analyze.py` | Figures (`results/fig_*.png`) and example tables | none |
| `preprocessing.py` | Shared header filter, abstention filter, and the MTSamples line break repair | none |
| `thesis/build_thesis.py` | Builds the thesis (.docx and .pdf, CSUN format) from the results | LibreOffice |
| `deliverables/build_progress_slides.py`, `deliverables/build_source_data_description.py` | Course deliverables | none |

## Reproduce

```bash
bash setup.sh                          # venv, pinned dependencies (requirements.txt), spaCy model 3.7.1
bash setup.sh --llm                    # also the model server environment .venv-llm (requirements-llm.txt, Apple silicon)
# place mtsamples.csv one directory above this repository
export OPENAI_API_KEY='sk-...'
bash run.sh                            # pilot: generation, evaluation, analysis
bash run.sh --offline                  # pilot without generation, from the stored results
bash run.sh --main-study               # judge study, MedNLI fine tuning, MIMIC-IV main study, QA pilot (local models)
```

The main study needs the credentialed data under `~/Desktop/Thesis` and a second environment for the model
server (Python 3.13, `pip install mlx-lm`):

```bash
.venv-llm/bin/python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8080
```

The runs with the two other generators use the same scripts. Serve `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
or `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, pass it with `--model`, and set `MIMIC_TAG=llama31` or
`MIMIC_TAG=mistral7b`. The outputs get that suffix, for example `results/mimic_summary_llama31.csv`.

The Bespoke-MiniCheck-7B judge candidate is converted once with
`mlx_lm.convert --hf-path <HF snapshot dir> --mlx-path <dir> -q --q-bits 8 --trust-remote-code`.
After that, copy `tokenizer.json`, `tokenizer.model` and `tokenizer_config.json` from the Hugging Face snapshot
into the converted directory, because the converter saves a broken tokenizer under transformers 5. Then serve it
with `.venv-llm/bin/python minicheck_server.py --model <dir> --port 8081`.

`requirements-lock.txt` and `requirements-llm-lock.txt` list both environments exactly as used.

Seeds:

* 42 for the MTSamples sample, the MedNLI fine tuning, the repeated generation subset and the selection bias analyses
* 1000r + k for course k of repeated generation r
* 7 for the audit sample and 8 for its second wave
* 0 for the pilot bootstrap

The LLMs are called with temperature 0.3, so regenerated summaries differ slightly from the stored ones. The
stored `results/summaries.csv` still reproduces every pilot number exactly through `recompute_metrics.py`.

| Result | Data needed | In this repository |
|---|---|---|
| Pilot study | MTSamples (public) | everything; `bash run.sh --offline` |
| Judge validation and selection | ann-pt-summ (PhysioNet, credentialed) | aggregate statistics only |
| MedNLI adaptation | MedNLI (PhysioNet, credentialed) | script and summary statistics |
| Main study and robustness checks | ann-pt-summ courses (from MIMIC-IV-Note) | aggregates in `results/mimic_*.csv` |
| Data overlap check | MIMIC-IV-Note (credentialed) | counts only |
| Manual audit | same as the main study | aggregates only |
| External check of the judge | MIMIC-III-Ext-VeriFact-BHC (PhysioNet, credentialed) | aggregates in `results/verifact_*.csv` |

The main study generations, claims, labels, audit items and the VeriFact scores per proposition contain text
derived from MIMIC. They are stored only in `results_private/`, which git ignores.

The external check runs separately from `run.sh`:

```bash
python verifact_eval.py --baseline
python verifact_eval.py --score --premise admission --batch 8 --chunk 256
python verifact_eval.py --summarize --premise admission
```

The scoring step is resumable and takes about a day on my laptop.
`python verifact_eval.py --compare-retrievers --premise admission` reproduces the choice of the window ranking.

## Data

* **MTSamples** (public): `../mtsamples.csv` (17 MB, 4,999 transcriptions). I sampled 50 documents from the
  624 Discharge Summary and Consult (History and Physical) notes.
* **ann-pt-summ v1.0.1** (PhysioNet, credentialed): 210 patient summaries annotated by experts, used to validate
  and select the judge, and the 110 MIMIC-IV hospital courses with discharge instructions written by clinicians,
  which are the main study corpus. Processed with local models only.
* **MedNLI v1.0.0** (PhysioNet, credentialed): 14,049 premise and hypothesis pairs written by clinicians, used to
  fine tune the judge.
* **MIMIC-IV-Note v2.2** (PhysioNet, credentialed): the source of the hospital courses. Staged locally, never
  committed.
* **MIMIC-III-Ext-VeriFact-BHC v1.0.0** (PhysioNet, credentialed): 12,614 valid propositions from Brief Hospital
  Course summaries of 100 MIMIC-III patients with adjudicated clinician labels, plus the patients' notes. Used
  only for the external check of the judge (thesis Section 6.7). Staged locally, never committed.

No credentialed text is committed, uploaded to shared storage, or sent to any external API.

## Manual audit of the judge

`python audit_sample.py` draws 60 courses (seed 7). Within them it draws up to 30 claims flagged and 30 accepted
by the judge for each of E0, E1, E1b and E3, plus 30 cited claims of E1 and of E1b. It writes
`results_private/audit_items.json`.

`python audit_tool.py` serves a local labeling page at http://127.0.0.1:8765. The page shows one course at a time
and hides the condition and the judge's label. The labels (supported, partly supported, not supported, unsure,
with an error type) are saved to `results_private/audit_labels.json`. The protocol is Appendix C of the thesis.

`python audit_sample.py --extend 15 --seed 8` adds a second wave of claims from the same 60 courses.
`audit_map_types.py` maps my notes on the partly and not supported labels to the five error types with keyword
rules. `audit_agreement.py` measures the agreement between my first labeling pass and my second blind pass, and
writes `results/audit_agreement.csv`.

After the main study evaluation, `python audit_sample.py --refresh-judge` stores the judge's current label of
every audited claim, and `python audit_summary.py` writes `results/audit_summary.csv`, `audit_types.csv` and
`audit_citations.csv` (Section 7.9).

## Result files

* **Pilot:** `results/claims_all.csv` (one row per claim: condition, claim, evidence, NLI label, probabilities,
  is_header, is_abstention), `results/summaries.csv`, `results/comparison_per_sample.csv`,
  `results/aggregate_statistics.csv`, `results/pairwise_tests.csv`, and the ablation and calibration CSVs.
* **Judge study:** `results/judge_candidates.csv`.
* **Main study:** `results/mimic_summary.csv`, `results/mimic_pairwise_tests.csv`, `results/mimic_ablations.csv`,
  `results/mimic_reference.csv`, `results/mimic_qa_summary.csv`, `results/mimic_qa_utility.csv`,
  `results/mimic_condition_ci.csv`, `results/mimic_variant_tests.csv`, `results/mimic_replicates.csv`,
  `results/mimic_abstention.csv`, `results/mimic_equivalence.csv`, `results/mimic_window_check.csv`,
  `results/mimic_splitting_check.csv`, `results/mimic_verifier_check.csv`, `results/mimic_judge_agreement.csv`,
  `results/mimic_second_judge_tests.csv`, `results/mimic_tau05/` (threshold 0.5), `results/atomic_results.csv`,
  `results/judge_selection_bias.csv`, `results/judge_selection_frequency.csv`, `results/judge_per_source.csv`,
  `results/data_overlap.csv`, `results/audit_*.csv`, `results/protected_text_scan*.csv`. The files that end in
  `_llama31` and `_mistral7b` hold the same tables for the two other generators.
* **External check of the judge:** `results/verifact_judge_eval.csv`, `results/verifact_retriever_check.csv`,
  `results/verifact_ai_raters.csv`.
* **Figures:** `results/fig_*.png`.
