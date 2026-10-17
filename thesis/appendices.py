"""appendices.py — Appendices A to E."""
import sys
from pathlib import Path

import pandas as pd

from results_loader import Results, f3, f2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def blocks(R: Results) -> list:
    import hallucination_eval as he
    import e1b_fullnote_rag_eval as e1b
    import e3_cove_eval as e3
    import mimic_generate as mg
    import mimic_qa as mq
    import mimic_extra as mx
    import atomic_claims as ac
    b = []
    # ── Appendix A: prompts
    b += [("h1", "Appendix A: Prompts"),
          ("pni", "The prompts below are reproduced verbatim from the source code. Placeholders in braces are filled at run time. "
                  "Pilot-study calls (A.1 to A.4) use GPT-4o-mini with a maximum of 450 output tokens; temperature is 0.3 for generation and 0 for the E3 verifier. "
                  "Main-study calls (A.5 and A.6) use Qwen2.5-7B-Instruct served locally with the same limits; the E3 verifier prompts of A.4 are reused unchanged."),
          ("h2", "A.1 E0: Full-Note Summarization Without Retrieval"), ("pni", "System prompt:"), ("code", he.BASELINE_SYSTEM), ("pni", "User prompt:"), ("code", he.BASELINE_USER),
          ("h2", "A.2 E1: Retrieval-Augmented Generation (Excerpts Only)"), ("pni", "System prompt:"), ("code", he.RAG_SYSTEM), ("pni", "User prompt:"), ("code", he.RAG_USER),
          ("h2", "A.3 E1b: Retrieval-Augmented Generation with the Full Note"), ("pni", "System prompt:"), ("code", e1b.E1B_SYSTEM), ("pni", "User prompt:"), ("code", e1b.E1B_USER),
          ("h2", "A.4 E3: Retrieval-Augmented Generation with Chain-of-Verification"), ("pni", "Verifier system prompt:"), ("code", e3.VERIFY_SYSTEM), ("pni", "Verifier user prompt (one call per claim):"), ("code", e3.VERIFY_USER),
          ("pni", "Revision system prompt:"), ("code", e3.REVISE_SYSTEM), ("pni", "Revision user prompt:"), ("code", e3.REVISE_USER),
          ("h2", "A.5 Main Study: Prompts for the Local Model"),
          ("pni", "E0 system prompt:"), ("code", mg.SYSTEM_FULL), ("pni", "E0 user prompt:"), ("code", mg.USER_FULL),
          ("pni", "E1 system prompt (the citation instruction below is appended in every variant except 'citations not required'):"), ("code", mg.SYSTEM_EXCERPTS + "\n\n[Citation instruction]" + mg.CITE_INSTR),
          ("pni", "E1 user prompt:"), ("code", mg.USER_EXCERPTS),
          ("pni", "E1b system prompt:"), ("code", mg.SYSTEM_BOTH), ("pni", "E1b user prompt:"), ("code", mg.USER_BOTH),
          ("pni", "E3 revision user prompt for the local model (the verifier prompts are those of A.4):"), ("code", mg.REVISE_USER_LOCAL),
          ("pni", f"Retrieval query for E1, E1b and their variants: the fixed section list \"{mg.SECTION_QUERY}\" followed by the first 300 characters of the hospital course."),
          ("h2", "A.6 Question-Answering Pilot"),
          ("pni", "System prompt:"), ("code", mq.SYSTEM), ("pni", "User prompt, full course (QA-E0):"), ("code", mq.USER_FULL),
          ("pni", "User prompt, retrieved excerpts (QA-E1):"), ("code", mq.USER_EXCERPTS),
          ("pni", "The three questions:"), ("code", "\n".join(f"{k}: {v}" for k, v in mq.QUESTIONS.items())),
          ("h2", "A.7 Variants Added in Response to the Review"),
          ("pni", "Length-matched E0, user prompt (the system prompt is that of E0 in A.5):"), ("code", mx.USER_FULL_SHORT),
          ("pni", "E1b without citations: the E1b prompts of A.5 with the citation instruction left out. Repeated generations use the base prompts with fixed seeds."),
          ("h2", "A.8 Atomic-Fact Decomposition"),
          ("pni", "System prompt (temperature 0):"), ("code", ac.SYSTEM), ("pni", "User prompt; the sentences of one summary are numbered:"), ("code", ac.USER)]
    # ── Appendix B: per-document results
    rows = []
    for _, r in R.cmp.sort_values("doc_id").iterrows():
        row = [str(int(r.doc_id)), "C" if r.specialty.startswith("Consult") else "D", str(int(r.source_word_count))]
        for c in R.conds:
            row += [f3(r[f"{c}_UFR"]), f3(r[f"{c}_CR"])]
        row += [str(int(r[f"{c}_n_claims"])) for c in R.conds]
        rows.append(row)
    ncond = len(R.conds)
    b += [("h1", "Appendix B: Per-Document Results"),
          ("pni", "UFR and CR per document and condition (header lines excluded), with the number of claims (n). Type C is a consultation note and D a discharge summary; document identifiers are the row indices of the sampled MTSamples subset."),
          ("table", dict(caption="Per-document Unsupported Fact Rate, Contradiction Rate and claim counts under each condition.",
                         columns=["Doc", "Type", "Words"] + [f"{c} {m}" for c in R.conds for m in ("UFR", "CR")] + [f"{c} n" for c in R.conds],
                         widths=[0.32, 0.3, 0.42] + [0.4] * (2 * ncond) + [0.3] * ncond, font=7.5, rows=rows))]
    # ── Appendix C: worked examples
    b += [("h1", "Appendix C: Worked Examples"),
          ("pni", "Two documents are shown with their summaries under every condition and the judge's label and probabilities for each claim "
                  "(header lines omitted). MTSamples is a public corpus of sample transcriptions; no real patient is described.")]
    for doc_id in (16, 0):
        row = R.cmp[R.cmp.doc_id == doc_id].iloc[0]
        b += [("h2", f"C.{1 if doc_id == 16 else 2} Document {doc_id}: {row.description[:90]}")]
        b += [("pni", f"Note type: {row.specialty}. Source length: {int(row.source_word_count)} words.")]
        for c in R.conds:
            text = R.example_summary(doc_id, c).replace("**", "").strip()
            cl = R.example_claims(doc_id, c)
            b += [("h3", f"{c} summary ({int(row[f'{c}_n_claims'])} claims; UFR {f3(row[f'{c}_UFR'])}, CR {f3(row[f'{c}_CR'])})"),
                  ("code", text),
                  ("table", dict(caption=f"Judge labels for the {c} claims of document {doc_id}.", columns=["Claim", "Label", "p(e)", "p(c)"],
                                 widths=[4.3, 0.9, 0.6, 0.6], font=8.5, align=["left", "center", "center", "center"],
                                 rows=[[str(r.claim).replace("**", "").replace("\n", " ").strip(), r.label, f2(r.p_entailment), f2(r.p_contradiction)] for _, r in cl.iterrows()]))]
    # ── Appendix D: repository
    b += [("h1", "Appendix D: Code Repository and Reproduction"),
          ("pni", "The complete code is available at https://github.com/slay-a/medical-hallucination-eval (branch main). The repository "
                  "contains the scripts listed below, the result files from which every number in this thesis is computed, and the "
                  "scripts that generate this document. Credentialed PhysioNet data and any output containing text derived from them "
                  "are excluded from the repository."),
          ("table", dict(caption="Scripts in the repository and their roles.", columns=["Script", "Role"], widths=[2.2, 4.3], font=9, align=["left", "left"],
                         rows=[["hallucination_eval.py", "E0 and E1 generation on the 50 sampled notes and NLI evaluation (requires OPENAI_API_KEY)"],
                               ["e2_extractive_eval.py", "E2 extractive baseline and evaluation (no API)"],
                               ["e1b_fullnote_rag_eval.py, e3_cove_eval.py", "E1b (RAG with the full note) and E3 (RAG with Chain-of-Verification) generation and evaluation (require OPENAI_API_KEY)"],
                               ["preprocessing.py", "Header filter and MTSamples line-break repair"],
                               ["recompute_metrics.py", "Per-document metrics, paired Wilcoxon tests, bootstrap CIs, effect sizes, header-filter tables"],
                               ["ablations.py", "Threshold, top-k and cleaned-evidence ablations; coverage proxy; negation analysis; error taxonomy"],
                               ["calibration_annptsumm.py, calibration_variants.py", "Judge validation against ann-pt-summ expert annotations (local models only)"],
                               ["judge_candidates.py", "Judge selection study: six candidate judges in two evidence modes scored against the expert annotations (Chapter 6)"],
                               ["finetune_mednli.py", "MedNLI fine-tuning of the strongest general NLI model; the model is saved outside the repository"],
                               ["mimic_generate.py", "Main-study generation with the local Qwen2.5-7B-Instruct server: E0, E1 and its six retrieval variants, E1b, E2, E3 on the 110 hospital courses (resumable)"],
                               ["mimic_evaluate.py", "Main-study evaluation with the selected judge: UFR, CR, coverage of the clinician's instructions, citation accuracy, reference scoring, paired tests and ablation tests"],
                               ["mimic_qa.py", "Question-answering pilot: answer generation with abstention (QA-E0, QA-E1) and its evaluation"],
                               ["minicheck_server.py", "Local scoring server for the Bespoke-MiniCheck-7B judge candidate (MLX, 8-bit)"],
                               ["mimic_second_judge.py", "Second-judge analysis: scores every main-study claim with Bespoke-MiniCheck-7B, agreement with the selected judge, paired tests under the second judge"],
                               ["citations.py", "Claim splitting for the main study: citation markers removed before sentence splitting and attributed to their sentences; merged header lines stripped"],
                               ["mimic_extra.py", "Length-matched E0, E1b without citations, and repeated generations of E0, E1, E1b and E3 for 40 courses"],
                               ["atomic_claims.py", "Atomic-fact decomposition with the local model and scoring of every fact by the selected judge"],
                               ["data_overlap.py", "Maps every ann-pt-summ hospital course to its MIMIC-IV-Note discharge note and reports document, admission and patient overlap"],
                               ["judge_selection_bias.py", "Nested cross-validation of the judge selection by course, patient-disjoint transfer, course-clustered intervals"],
                               ["mimic_revision_analysis.py", "Length-matched and citation-free comparisons, run-to-run variation, abstention, coverage equivalence, question-answering utility, overlapping-window check"],
                               ["audit_sample.py, audit_tool.py, audit_summary.py", "Manual audit: course-clustered stratified sample, local blind labeling tool, re-weighted agreement statistics"],
                               ["check_no_protected_text.py", "Scans every tracked repository file for word sequences from the credentialed documents and generations"],
                               ["analyze.py", "All figures and the qualitative example tables"],
                               ["thesis/build_thesis.py", "Generates this document from the result files (content modules ch1_4, ch5_pilot, ch_judge, ch_mimic, ch8_9, appendices)"],
                               ["run.sh, setup.sh, requirements.txt", "Environment and end-to-end reproduction (bash run.sh --offline reproduces every pilot number without API calls; bash run.sh --main-study runs the judge study and the main study on a machine that holds the credentialed data)"]])),
          ("pni", "What can be reproduced depends on data access. [[tab:repro]] lists, for every part of the thesis, the data it needs "
                  "and what the public repository contains. No result requires sending protected text to any external service; an "
                  "automated scan of every tracked file, described in Section 3.7, found no word sequence from the credentialed sources."),
          ("table", dict(label="repro", caption="Reproducibility of the results by data access.",
                         columns=["Result", "Data needed", "In the public repository", "Reproduction"],
                         widths=[1.45, 1.55, 1.9, 1.6], font=8.5, align=["left"] * 4,
                         rows=[["Pilot study (Chapter 5)", "MTSamples (public)", "All summaries, claims, labels and probabilities", "bash run.sh --offline reproduces every number without API or model calls"],
                               ["Judge validation (Section 5.4) and selection (Chapter 6)", "ann-pt-summ (PhysioNet, credentialed)", "Aggregate agreement statistics only", "Requires PhysioNet access; local models only"],
                               ["MedNLI adaptation (Section 6.3)", "MedNLI (PhysioNet, credentialed)", "Training script and summary statistics; the fine-tuned model is not distributed", "Requires PhysioNet access; about two hours on the GPU"],
                               ["Main study and robustness checks (Chapter 7)", "ann-pt-summ hospital courses (derived from MIMIC-IV-Note)", "Aggregates in results/mimic_*.csv; generations and per-claim labels stay private", "Requires PhysioNet access and the local model server"],
                               ["Data overlap check (Section 6.5)", "MIMIC-IV-Note discharge notes (credentialed)", "Counts only", "Requires PhysioNet access"],
                               ["Manual audit (Section 7.9)", "As the main study", "Aggregates only; the labels stay private", "Requires PhysioNet access"]])),
          ("pni", "Models, checkpoints and seeds. Models are identified by their Hugging Face names and the revision used: "
                  "cross-encoder/nli-MiniLM2-L6-H768 (b95119c), sentence-transformers/all-MiniLM-L6-v2 (1110a24), "
                  "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (b3546ea), cross-encoder/nli-deberta-v3-large (bab4bc7), "
                  "lytang/MiniCheck-DeBERTa-v3-Large (2f2d01a), lytang/MiniCheck-RoBERTa-Large (74c8919), "
                  "bespokelabs/Bespoke-MiniCheck-7B (1ed7786, converted to 8-bit MLX), mlx-community/Qwen2.5-7B-Instruct-4bit "
                  "(c26a38f), and the OpenAI model gpt-4o-mini through the API. Seeds: 42 for the MTSamples sample, the MedNLI "
                  "fine-tuning, the repeated-generation subset and the selection-bias analyses; 1000r + k for the k-th course of "
                  "repeated generation r; 7 for the manual-audit sample; 0 for the pilot bootstrap. requirements.txt and "
                  "requirements-llm.txt pin the versions of Table 4.3, and the lock files list both environments completely."),
          ("pni", "Reproduction from a clean machine:"),
          ("code", "git clone https://github.com/slay-a/medical-hallucination-eval.git\ncd medical-hallucination-eval\nbash setup.sh\n# place mtsamples.csv one directory above the repository\nexport OPENAI_API_KEY='sk-...'   # only for generation\nbash run.sh            # full chain\nbash run.sh --offline  # recompute every pilot result from the stored results\nbash run.sh --main-study  # judge study, MedNLI fine-tuning, MIMIC-IV main study, QA pilot (credentialed data, local models)\npython thesis/build_thesis.py")]
    # ── Appendix E: calibration threshold sweep
    if R.cal_sweep is not None:
        sw = R.cal_sweep[R.cal_sweep.group == "all"]
        b += [("h1", "Appendix E: Judge Threshold Sweep Against Expert Annotations"),
              ("pni", "Sentence-level agreement of the any detector with the expert annotations on all 1,781 sentences as the entailment threshold τ varies."),
              ("table", dict(caption="Agreement statistics of the judge with the expert annotations as a function of the entailment threshold τ.",
                             columns=["τ", "Judge flag rate", "Precision", "Recall", "Specificity", "F1", "Balanced accuracy", "Kappa"],
                             widths=[0.5, 0.9, 0.8, 0.7, 0.85, 0.6, 1.0, 0.7], font=9,
                             rows=[[f"{r.tau:.2f}", f3(r.judge_flag_rate), f3(r.precision), f3(r.recall), f3(r.specificity), f3(r.f1), f3(r.balanced_accuracy), f3(r.kappa)] for _, r in sw.iterrows()]))]
    b += [("h1", "Appendix F: Manual Audit Protocol"),
          ("pni", "The annotator labels each sampled statement against the hospital course shown beside it, and only against that course. "
                  "The tool hides the condition that produced the statement and the judge's label."),
          ("table", dict(caption="Labels of the manual audit.", columns=["Label", "When it is used"], widths=[1.4, 5.1], font=9.5, align=["left", "left"],
                         rows=[["S, supported", "Every fact in the statement is in the course or follows directly from it; plain-language paraphrase is acceptable."],
                               ["P, partly supported", "The statement has several facts; at least one is supported and at least one is not."],
                               ["U, not supported", "The main content is not in the course, or it contradicts the course."],
                               ["?, unsure", "The annotator cannot decide from the course; a note is recorded. Excluded from the analysis."]])),
          ("pni", "For P and U the annotator records what is wrong: not stated in the course; generic advice or follow-up not in the "
                  "course; contradicts the course; wrong or invented medication, dose, number, date, side or unit; or other, with a note "
                  "(for example, certainty overstated when the course says possible or likely). Rules for consistency: only the course "
                  "counts, so a statement that is medically sensible but absent from the course is not supported; numbers, doses, units "
                  "and laterality must match exactly; a statement that fills in a name or date hidden in the course as ___ is not "
                  "supported; earlier labels are not revised to match later ones unless a statement was clearly misread. For a citation "
                  "check, the annotator decides from the cited excerpt alone whether it supports the statement (yes, partly, no, unsure)."),
          ("pni", "Example, invented and not from the data. Course: admitted with community-acquired pneumonia, treated with intravenous "
                  "ceftriaxone and azithromycin, switched to oral amoxicillin, discharged home on day four. \"You were treated for a lung "
                  "infection with antibiotics\" is S. \"You got antibiotics through a vein, and you will take amoxicillin for five more "
                  "days\" is P, not stated. \"You will take 500 mg of amoxicillin twice a day\" is U, invented dose. \"Please see your "
                  "primary care doctor in one week\" is U, generic advice or follow-up.")]
    return b
