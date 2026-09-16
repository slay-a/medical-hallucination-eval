# Hallucination Evaluation in LLM-Based Medical Summarization

Claim-level evaluation of hallucinations in patient-facing summaries generated from clinical notes, and a
controlled comparison of five summarization conditions: zero-context LLM summarization (E0), retrieval-augmented
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
| `mimic_evaluate.py` | Main study evaluation with the best judge: UFR, CR, coverage vs. clinician's instructions, citation accuracy, tests, ablations | ann-pt-summ |
| `mimic_second_judge.py` | Scores every base-condition claim with a second judge (Bespoke-MiniCheck-7B) and reports agreement with the selected judge → `results/mimic_judge_agreement.csv` | MiniCheck server |
| `audit_sample.py`, `audit_summary.py` | Manual precision audit: draws a stratified sample of claims for hand labeling (`results_private/audit_sample.csv`), then computes precision, kappa and a corrected unsupported rate → `results/audit_summary.csv` | – |
| `analyze.py` | Figures (`results/fig_*.png`) and example tables | – |
| `preprocessing.py` | Shared header filter, abstention filter, MTSamples line-break repair | – |
| `thesis/build_thesis.py` | Builds the thesis (.docx and .pdf, CSUN format) from the results | LibreOffice |
| `deliverables/build_progress_slides.py`, `deliverables/build_source_data_description.py` | Course deliverables | – |

## Reproduce

```bash
bash setup.sh                          # venv + dependencies + spaCy model
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

Random seed 42 is fixed for document sampling; the LLMs are called with temperature 0.3, so regenerated summaries
differ slightly from the stored ones, but the stored `results/summaries.csv` reproduces every pilot number exactly
through `recompute_metrics.py`. Main-study generations contain MIMIC-derived text and are stored only in the
git-ignored `results_private/`; the aggregate tables in `results/mimic_*.csv` are committed.

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

`python audit_sample.py` writes `results_private/audit_sample.csv` (320 claims: 40 flagged and 40 accepted by the judge from each of
E0, E1, E1b and E3, in random order, with the hospital course next to each claim). Sort by `order`, read the course, and fill
`human_label` with `S` (the claim is supported by the course), `U` (not stated or contradicted) or `?` (unsure); do not look at the
`judge_label` and `p_support` columns while labeling. `python audit_summary.py` then writes `results/audit_summary.csv`, and the thesis
picks it up as Section 7.8 on the next build.

## Results files

Pilot: `results/claims_all.csv` (one row per claim: condition, claim, evidence, NLI label, probabilities,
is_header, is_abstention), `results/summaries.csv`, `results/comparison_per_sample.csv`,
`results/aggregate_statistics.csv`, `results/pairwise_tests.csv`, ablation and calibration CSVs.
Judge study: `results/judge_candidates.csv`. Main study: `results/mimic_summary.csv`,
`results/mimic_pairwise_tests.csv`, `results/mimic_ablations.csv`, `results/mimic_reference.csv`,
`results/mimic_qa_summary.csv`, `results/mimic_judge_agreement.csv`, `results/audit_summary.csv` (when labeled). Figures: `results/fig_*.png`.
