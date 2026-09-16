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
    b = []
    # ── Appendix A: prompts
    b += [("h1", "Appendix A: Prompts"),
          ("pni", "The prompts below are reproduced verbatim from the source code. Placeholders in braces are filled at run time. "
                  "Pilot-study calls (A.1 to A.4) use GPT-4o-mini with a maximum of 450 output tokens; temperature is 0.3 for generation and 0 for the E3 verifier. "
                  "Main-study calls (A.5 and A.6) use Qwen2.5-7B-Instruct served locally with the same limits; the E3 verifier prompts of A.4 are reused unchanged."),
          ("h2", "A.1 E0: Zero-Context Summarization"), ("pni", "System prompt:"), ("code", he.BASELINE_SYSTEM), ("pni", "User prompt:"), ("code", he.BASELINE_USER),
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
          ("pni", "The three questions:"), ("code", "\n".join(f"{k}: {v}" for k, v in mq.QUESTIONS.items()))]
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
                               ["judge_candidates.py", "Judge selection study: five candidate judges in two evidence modes scored against the expert annotations (Chapter 6)"],
                               ["finetune_mednli.py", "MedNLI fine-tuning of the strongest general NLI model; the model is saved outside the repository"],
                               ["mimic_generate.py", "Main-study generation with the local Qwen2.5-7B-Instruct server: E0, E1 and its six retrieval variants, E1b, E2, E3 on the 110 hospital courses (resumable)"],
                               ["mimic_evaluate.py", "Main-study evaluation with the selected judge: UFR, CR, coverage of the clinician's instructions, citation accuracy, reference scoring, paired tests and ablation tests"],
                               ["mimic_qa.py", "Question-answering pilot: answer generation with abstention (QA-E0, QA-E1) and its evaluation"],
                               ["analyze.py", "All figures and the qualitative example tables"],
                               ["thesis/build_thesis.py", "Generates this document from the result files (content modules ch1_4, ch5_pilot, ch_judge, ch_mimic, ch8_9, appendices)"],
                               ["run.sh, setup.sh, requirements.txt", "Environment and end-to-end reproduction (bash run.sh --offline reproduces every pilot number without API calls; bash run.sh --main-study runs the judge study and the main study on a machine that holds the credentialed data)"]])),
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
    return b
