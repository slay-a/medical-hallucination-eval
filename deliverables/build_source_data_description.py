#!/usr/bin/env python3
"""
build_source_data_description.py — Source Code and Data Description (Fall 2026), as .docx and .pdf.

Everything factual (file sizes, row counts, versions, commit hash) is read from the repository at build time.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "thesis"))
from render_docx import _font, _pf, _shade, _repeat_header, soffice_convert  # noqa: E402

OUT = HERE / "Source_Code_and_Data_Description_Fall2026"


def kb(p: Path):
    n = p.stat().st_size
    return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1024/1024:.1f} MB"


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "n/a"


def table(doc, columns, rows, widths, font=9.5):
    t = doc.add_table(rows=1 + len(rows), cols=len(columns)); t.style = "Table Grid"; t.autofit = False
    for j, c in enumerate(columns):
        cell = t.rows[0].cells[j]; cell.width = Inches(widths[j]); _shade(cell)
        p = cell.paragraphs[0]; _pf(p, spacing="single"); r = p.add_run(str(c)); _font(r, font, True)
    _repeat_header(t.rows[0])
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            cell = t.rows[i].cells[j]; cell.width = Inches(widths[j])
            p = cell.paragraphs[0]; _pf(p, spacing="single"); r = p.add_run(str(v)); _font(r, font)
    doc.add_paragraph()


def heading(doc, text):
    p = doc.add_paragraph(); _pf(p, spacing="single", before=12, after=6, keep_next=True); r = p.add_run(text); _font(r, 13, True)


def para(doc, text, size=11):
    p = doc.add_paragraph(); _pf(p, spacing="single", after=6); r = p.add_run(text); _font(r, size)
    return p


def main():
    scripts = [
        ("hallucination_eval.py", "Main pipeline: loads MTSamples, samples 50 notes (seed 42), generates E0 (full note) and E1 (top-3 retrieved chunks only) summaries with GPT-4o-mini, segments claims (spaCy; header lines removed), retrieves top-3 evidence sentences (all-MiniLM-L6-v2), labels claims with cross-encoder/nli-MiniLM2-L6-H768, writes per-claim and per-summary results."),
        ("e2_extractive_eval.py", "E2 extractive baseline: centroid ranking of source sentences by mean cosine similarity, top-5 kept in document order, no LLM call; evaluated with the same judge; appends E2 rows and columns to the result files."),
        ("e1b_fullnote_rag_eval.py", "E1b: RAG with the full note plus the retrieved excerpts, and its evaluation. Requires OPENAI_API_KEY; 50 calls."),
        ("e3_cove_eval.py", "E3: factored Chain-of-Verification on top of the E1 draft; per-claim verification against top-3 sentences from the full note, then revision (keep / replace / remove per claim); checkpointed with a --revise-only mode. Requires OPENAI_API_KEY; about 500 calls."),
        ("preprocessing.py", "Shared utilities: is_markdown_header() removes section-header lines from the claim set; clean_mtsamples_text() and sentencize_cleaned() repair the comma-encoded line breaks of MTSamples."),
        ("recompute_metrics.py", "Offline recomputation of every metric from results/claims_all.csv with headers excluded: per-sample UFR/CR, comparison table, aggregate statistics, Wilcoxon tests, bootstrap 95% CIs, effect sizes, header-filter effect tables."),
        ("ablations.py", "Offline robustness analyses: NLI threshold sweep, evidence top-k (1, 3, 5), cleaned-evidence re-scoring, coverage proxy, negation analysis, keyword-assisted error taxonomy."),
        ("calibration_annptsumm.py", "Judge validation against ann-pt-summ expert annotations (credentialed data, local models only): sentence-level precision/recall/F1/kappa, AUROC, threshold sweep, per-label recall, summary-level correlation."),
        ("calibration_variants.py", "Five evidence-aggregation variants of the judge evaluated on the expert-annotated set."),
        ("judge_candidates.py", "Judge selection study: six candidate judges (MiniLM, two DeBERTa-v3-large NLI models, MiniCheck DeBERTa and RoBERTa, Bespoke-MiniCheck-7B) plus the MedNLI-tuned model, each in two evidence modes (top-3 sentences; whole course in 400-token windows), scored against the 1,781 expert-annotated sentences with thresholds chosen on the other subset; writes results/judge_candidates.csv."),
        ("minicheck_server.py", "Local HTTP scoring server for Bespoke-MiniCheck-7B (InternLM2.5-7B grounding checker, 8-bit MLX conversion kept outside the repository); returns the probability of 'Yes' for (document, claim) pairs with per-document prompt caching; queried by judge_candidates.py and mimic_evaluate.py over localhost."),
        ("finetune_mednli.py", "Fine-tunes the strongest general NLI model on MedNLI (1 epoch, Adafactor, lr 2e-5, effective batch 16 as 4 x 4 accumulation, max length 128, embedding block frozen, MPS); saves the best-dev checkpoint outside the repository (~/Desktop/Thesis/models)."),
        ("mimic_generate.py", "Main-study generation against the local OpenAI-compatible mlx-lm server (Qwen2.5-7B-Instruct, 4-bit): E0, E1 (with citations) and its six retrieval variants, E1b, E2 and E3 for the 110 ann-pt-summ hospital courses; resumable JSONL output in results_private/."),
        ("mimic_evaluate.py", "Main-study evaluation with the judge of highest kappa: claim splitting, forward UFR/CR, citation accuracy, reverse coverage of the clinician's discharge instructions, scoring of the doctor-written instructions themselves, paired Wilcoxon tests and ablation tests; aggregates to results/mimic_*.csv, per-claim rows to results_private/."),
        ("mimic_qa.py", "Question-answering pilot: three fixed questions per course answered from the full course (QA-E0) or three retrieved chunks (QA-E1) with an abstention instruction; evaluation of abstention rate, UFR/CR and support by the clinician's instructions."),
        ("analyze.py", "Figures (results/fig_*.png, 200 dpi) and example tables (claims fixed by RAG, persistent unsupported claims)."),
        ("thesis/build_thesis.py", "Builds the thesis (.docx and .pdf) in CSUN format from the result files; content modules ch1_4.py, ch5_pilot.py, ch_judge.py, ch_mimic.py, ch8_9.py, appendices.py; two-pass table of contents, list of tables and list of figures via LibreOffice."),
        ("thesis/render_docx.py, thesis/results_loader.py, thesis/references.py", "Rendering engine (python-docx), results access layer, IEEE reference database with citation resolver."),
        ("deliverables/build_progress_slides.py", "Progress report slides (PDF and PPTX)."),
        ("deliverables/build_source_data_description.py", "This document."),
        ("setup.sh, run.sh, requirements.txt", "Environment bootstrap; reproduction chains (bash run.sh, bash run.sh --offline, bash run.sh --main-study); pinned dependencies."),
    ]
    rows = []
    for name, purpose in scripts:
        parts = [ROOT / n.strip() for n in name.split(",")]
        size = " + ".join(kb(p) for p in parts if p.exists())
        rows.append([name, size, purpose])

    res = ROOT / "results"
    claims = pd.read_csv(res / "claims_all.csv"); cmp = pd.read_csv(res / "comparison_per_sample.csv")
    n_hdr = int(claims.is_header.sum()); nh = claims[~claims.is_header]
    result_files = [
        ("claims_all.csv", f"{len(claims):,} rows: one per claim and condition; columns doc_id, condition, specialty, description, claim, evidence (top-3 sentences, '|'-separated), evidence_scores, label, p_entailment, p_contradiction, is_header. {n_hdr} rows are header lines (is_header = True) and are excluded from all metrics; {len(nh):,} claims remain ({nh.groupby('condition').size().to_dict()})."),
        ("summaries.csv", f"{len(cmp)} rows: source note metadata, the E0/E1/E2 summary texts and word counts, and per-condition UFR, CR and claim counts."),
        ("comparison_per_sample.csv", f"{len(cmp)} rows: per-document UFR, CR and claim counts for every condition, plus paired deltas."),
        ("aggregate_statistics.csv", "Mean, SD, median per condition; E1−E0 and E2−E0 deltas, % improved, Wilcoxon p, bootstrap 95% CI."),
        ("pairwise_tests.csv", "Every condition pair × metric: means, medians, mean and median delta, bootstrap CI, W, p, d_z, rank-biserial, % improved/unchanged/worse."),
        ("header_filter_effect.csv, header_filter_tests.csv", "Metrics and E1-vs-E0 tests before and after removing header lines."),
        ("ablation_thresholds.csv, ablation_topk.csv, ablation_cleaned_evidence.csv, claims_cleaned_evidence.csv", "Robustness analyses (Chapter 5)."),
        ("coverage_per_sample.csv, coverage_summary.csv", "Coverage proxy per document and per condition."),
        ("e2_error_analysis.csv, error_taxonomy_counts.csv, error_taxonomy_examples.csv", "Negation analysis and error taxonomy."),
        ("calibration_overall.csv, calibration_by_label.csv, calibration_label_counts.csv, calibration_threshold_sweep.csv, calibration_summary_level.csv, calibration_by_system_means.csv, calibration_variants.csv", "Judge validation against expert annotations (aggregate statistics only; no MIMIC-derived text)."),
        ("examples_fixed_by_rag.csv, examples_persistent.csv", "Qualitative example pairs."),
        ("judge_candidates.csv", "Judge selection study: AUROC, threshold, flag rate, precision, recall, specificity, F1 and kappa for every candidate judge, evidence mode and sentence subset (Chapter 6)."),
        ("mimic_summary.csv, mimic_pairwise_tests.csv, mimic_reference.csv", "Main study aggregates: per-condition means and medians of UFR, CR, coverage of the clinician's instructions, citation accuracy, words, claims and abstentions; paired tests against E0, E1 and the doctor-written instructions; the judge's rates on the doctor-written instructions next to the expert span counts (Chapter 7). No MIMIC-derived text."),
        ("mimic_ablations.csv, mimic_qa_summary.csv", "Retrieval ablations (paired differences of six E1 variants from the base configuration) and the question-answering pilot aggregates by condition and question type."),
        ("fig_*.png", f"{len(list(res.glob('fig_*.png')))} figures at 200 dpi used in the thesis."),
    ]

    doc = Document()
    from render_docx import _set_margins
    _set_margins(doc.sections[0])
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
    p = doc.add_paragraph(); _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing="single", after=4); r = p.add_run("Source Code and Data Description"); _font(r, 16, True)
    p = doc.add_paragraph(); _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing="single", after=2); r = p.add_run("Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization"); _font(r, 12)
    p = doc.add_paragraph(); _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing="single", after=12)
    r = p.add_run(f"Srilaya Ponangi · COMP 698C · Fall 2026 · California State University, Northridge · {date.today():%B %d, %Y}"); _font(r, 10)

    heading(doc, "1. Repository at a glance")
    code_kb = sum(p.stat().st_size for p in list(ROOT.glob("*.py")) + list((ROOT / "thesis").glob("*.py")) + list((ROOT / "deliverables").glob("*.py")) + list(ROOT.glob("*.sh")))
    res_kb = sum(p.stat().st_size for p in res.glob("*") if p.is_file())
    table(doc, ["Item", "Value"], [
        ["Source", "https://github.com/slay-a/medical-hallucination-eval (public, branch main)"],
        ["Commit", f"{git('rev-parse', '--short', 'HEAD')} ({git('log', '-1', '--format=%cd', '--date=short')})"],
        ["Type", "Python 3 research codebase; Bash helper scripts; no compiled components"],
        ["Size", f"code {code_kb/1024:.0f} KB; result artifacts {res_kb/1024/1024:.1f} MB"],
        ["Language and runtime", "Python 3.9.6 in a virtual environment (.venv) for the pipeline; Python 3.13 in .venv-llm for the mlx-lm model server; macOS on Apple silicon (M5, 24 GB). The pilot runs on the CPU; the judge study, MedNLI fine-tuning and main study use the GPU through PyTorch Metal and MLX"],
        ["License / use", "Academic use. MTSamples-derived results are public. Outputs derived from PhysioNet data are never committed (results_private/ is git-ignored)."],
        ["Reproducibility", "Sampling seed 42; pinned requirements.txt; every number in the thesis is reproducible offline from results/claims_all.csv with recompute_metrics.py"],
    ], [1.6, 4.9])

    heading(doc, "2. File inventory")
    table(doc, ["Path", "Size", "Purpose"], rows, [1.9, 0.7, 3.9], font=8.5)

    heading(doc, "3. Dependencies")
    req = (ROOT / "requirements.txt").read_text().strip().splitlines()
    para(doc, "requirements.txt (verbatim): " + "; ".join(req) + ".")
    para(doc, "Installed versions used for the reported results: openai 2.31.0, pandas 2.3.3, numpy 1.26.4, spacy 3.7.4 (en_core_web_sm), sentence-transformers 5.1.2, torch 2.8.0, scikit-learn 1.6.1, scipy 1.13.1, matplotlib 3.9.4, python-docx 1.2.0, reportlab 4.4.10, python-pptx 1.0.2. LibreOffice (soffice) converts .docx to PDF.")
    para(doc, "External models (downloaded once from the Hugging Face hub, then cached): sentence-transformers/all-MiniLM-L6-v2 (bi-encoder, 384-d, retrieval by cosine similarity); cross-encoder/nli-MiniLM2-L6-H768 (NLI classifier; outputs contradiction/entailment/neutral). External API: OpenAI GPT-4o-mini, temperature 0.3, max 450 output tokens; key supplied via the OPENAI_API_KEY environment variable and never stored in the repository. Judge study and main study: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, cross-encoder/nli-deberta-v3-large, lytang/MiniCheck-DeBERTa-v3-Large and lytang/MiniCheck-RoBERTa-Large (candidate judges); mlx-community/Qwen2.5-7B-Instruct-4bit served with mlx-lm 0.31 (mlx 0.32) through an OpenAI-compatible interface on localhost; rank-bm25 for the BM25 ablation. No protected text is sent to any external service.")

    heading(doc, "4. Data sources")
    table(doc, ["Source", "Type", "Content", "Size", "Access status and location"], [
        ["MTSamples", "Public CSV", "4,999 sample medical transcriptions, 40 specialties; 624 eligible Discharge Summary and Consult - History and Physical notes; 50 sampled (seed 42): 40 consultations, 10 discharge summaries; source length 117 to 1,809 words (mean 539); summaries generated under five conditions", "17 MB", "Public. Stored one directory above the repository (../mtsamples.csv), not tracked."],
        ["MIMIC-IV-Note v2.2", "Credentialed CSV (gzip)", "331,794 de-identified discharge summaries and about 2.3 million radiology reports (PhysioNet doi 10.13026/1n74-ne17)", "1.8 GB compressed", "Credentialing approved 4/28/2026; downloaded 5/14/2026; gzip integrity verified; SHA-256 manifest present. Stored locally (~/Desktop/Thesis). Staged for future generation experiments; not used for generation in this thesis."],
        ["ann-pt-summ v1.0.1", "Credentialed JSONL/XML", "Main-study corpus (110 MIMIC-IV hospital courses paired with the discharge instructions their clinicians wrote) and expert span annotations of unsupported facts: 100 doctor-written and 100 LLM-generated patient summaries plus 10 validation summaries; 423 agreed spans in 10 label types (PhysioNet doi 10.13026/gedc-j464)", "2.3 MB used (archive incomplete: 934 MB of >3.4 GB; the 14 annotation and derived files verified against SHA-256)", "DUA accepted; downloaded 5/12/2026. Used only to validate the NLI judge, with local models. Per-sentence outputs in results_private/ (git-ignored)."],
        ["MedNLI v1.0.0", "Credentialed JSONL", "14,049 clinician-written premise and hypothesis pairs from MIMIC-III notes (11,232 train, 1,395 dev, 1,422 test; entailment, neutral, contradiction) used to fine-tune the judge (PhysioNet doi 10.13026/C2RS98)", "9 MB", "Data use agreement accepted and downloaded 9/15/2026; stored locally (~/Desktop/Thesis/mednli); never committed"],
    ], [1.0, 0.8, 2.4, 0.8, 1.5], font=8.5)
    para(doc, "Compliance. CITI Program courses 'Data or Specimens Only Research' and 'Conflicts of Interest' completed April 10, 2026 (records 76432215 and 76432216). PhysioNet credentialed data and any output containing text derived from them are never committed, never uploaded to shared cloud storage, and never transmitted to the OpenAI API. Only MTSamples-based generations were sent to the API.")

    heading(doc, "5. Result files (results/)")
    table(doc, ["File", "Content"], result_files, [2.3, 4.2], font=8.5)

    heading(doc, "6. Pipeline architecture")
    for s in [
        "Stage 1, generation: E0 prompts GPT-4o-mini with the first 4,500 characters of the note; E1 prompts it with the top-3 retrieved five-sentence chunks only (query = description + first 300 characters; all-MiniLM-L6-v2 cosine; at most 4,000 characters); E1b adds the full note to the same excerpts; E3 verifies each E1 claim against the top-3 sentences retrieved from the full note and rewrites the summary; E2 selects the top-5 centroid sentences without an LLM.",
        "Stage 2, claim segmentation: spaCy en_core_web_sm sentences of at least 10 characters; lines that are only a markdown/section header are removed (preprocessing.is_markdown_header).",
        "Stage 3, evidence retrieval: for each claim the 3 most similar source sentences by cosine similarity of all-MiniLM-L6-v2 embeddings.",
        "Stage 4, NLI classification: cross-encoder/nli-MiniLM2-L6-H768 scores each (evidence sentence, claim) pair; the claim's p_entailment and p_contradiction are the maxima over the three pairs. Supported if p_entailment ≥ 0.5 and p_entailment > p_contradiction; Contradicted if p_contradiction ≥ 0.5 and p_contradiction > p_entailment; otherwise Not-Supported.",
        "Stage 5, aggregation: per summary, UFR = (Contradicted + Not-Supported) / claims and CR = Contradicted / claims.",
        "Stage 6, statistics: two-sided Wilcoxon signed-rank tests on paired per-document metrics, percentile bootstrap 95% CIs (10,000 resamples), d_z and rank-biserial effect sizes, Holm adjustment over the six primary tests (recompute_metrics.py).",
        "Validation: the unchanged judge is applied to 210 expert-annotated summaries; sentence-level agreement with expert spans, threshold sweep, AUROC, per-label recall, summary-level Spearman correlation, and five evidence-aggregation variants (calibration_annptsumm.py, calibration_variants.py).",
        "Judge selection: six candidate judges in two evidence modes are scored on the same 1,781 sentences with thresholds chosen on the other subset; the strongest general model is fine-tuned on MedNLI and scored again; the candidate with the highest kappa becomes the main-study judge (judge_candidates.py, finetune_mednli.py).",
        "Main study, generation: Qwen2.5-7B-Instruct (4-bit, mlx-lm, localhost) writes four-part patient summaries of 110 MIMIC-IV hospital courses under E0, E1 (five-sentence chunks, top-3 dense retrieval, citations required), E1b, E2 and E3, plus six E1 variants (3- or 8-sentence chunks, top-2 or top-5, BM25, no citations); the retrieval query is a fixed section list plus the first 300 characters (mimic_generate.py).",
        "Main study, evaluation: the selected judge scores every claim against the whole course in windows (UFR, CR), cited excerpts (citation accuracy) and, in the reverse direction, every sentence of the clinician's discharge instructions against the summary (coverage); the doctor-written instructions are scored as a human reference; paired Wilcoxon tests against E0, E1 and the reference, and against the E1 base configuration for the ablations (mimic_evaluate.py).",
        "Question-answering pilot: three questions per course (medications, follow-up, warning signs) answered from the full course or from three retrieved chunks with an abstention instruction; abstention rate, UFR/CR of answered questions, and support of the answers by the clinician's instructions (mimic_qa.py).",
    ]:
        p = doc.add_paragraph(style="List Bullet"); _pf(p, spacing="single", after=3); r = p.add_run(s); _font(r, 10.5)

    heading(doc, "7. How to reproduce")
    for s in ["git clone https://github.com/slay-a/medical-hallucination-eval.git && cd medical-hallucination-eval",
              "bash setup.sh                       # creates .venv, installs pinned dependencies, downloads en_core_web_sm",
              "# place mtsamples.csv one directory above the repository",
              "export OPENAI_API_KEY='sk-...'      # needed only for generation (E0, E1; optionally E1b, E3)",
              "bash run.sh                         # generation + evaluation + recomputation + ablations + figures",
              "bash run.sh --offline               # reproduce every pilot number from the stored results without any API call",
              "bash run.sh --main-study            # judge study, MedNLI fine-tuning, MIMIC-IV main study and QA pilot (credentialed data, local models)",
              "python thesis/build_thesis.py       # rebuild the thesis .docx and .pdf"]:
        p = doc.add_paragraph(); _pf(p, spacing="single", after=2, left_indent=0.3); r = p.add_run(s); _font(r, 9.5); r.font.name = "Courier New"
    para(doc, "Changes since the May 2026 description: header lines are excluded from the claim set (162 of 1,373 stored E0/E1/E2 claim rows), which changes every metric; the E1 description was corrected (excerpts only, non-overlapping five-sentence chunks); two new conditions, E1b and E3, were generated and evaluated; four dependencies that the analysis scripts import (matplotlib, scipy, python-docx, reportlab) were added to requirements.txt; bootstrap confidence intervals, effect sizes and Holm adjustment are now actually computed; the ann-pt-summ DUA is signed and the annotation files are in use. Since the September 4, 2026 description: a judge selection study replaced the pilot judge with a DeBERTa-v3-large NLI model validated against the experts, a MedNLI fine-tuning script was added, and the main study on 110 MIMIC-IV hospital courses with a locally served Qwen2.5-7B-Instruct model (five conditions with citations, coverage against the clinician's instructions, citation accuracy, six retrieval ablations, question-answering pilot) was implemented and run.", size=10)

    docx_path = OUT.with_suffix(".docx"); doc.save(docx_path)
    pdf = soffice_convert(docx_path, HERE / "_build"); (HERE / "_build" / pdf.name).replace(OUT.with_suffix(".pdf"))
    print("wrote", docx_path.name, "and", OUT.with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
