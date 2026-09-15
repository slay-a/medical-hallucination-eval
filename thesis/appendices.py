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
    b = []
    # ── Appendix A: prompts
    b += [("h1", "Appendix A: Prompts"),
          ("pni", "The prompts below are reproduced verbatim from the source code. Placeholders in braces are filled at run time. "
                  "All calls use GPT-4o-mini with a maximum of 450 output tokens; temperature is 0.3 for generation and 0 for the E3 verifier."),
          ("h2", "A.1 E0: Zero-Context Summarization"), ("pni", "System prompt:"), ("code", he.BASELINE_SYSTEM), ("pni", "User prompt:"), ("code", he.BASELINE_USER),
          ("h2", "A.2 E1: Retrieval-Augmented Generation (Excerpts Only)"), ("pni", "System prompt:"), ("code", he.RAG_SYSTEM), ("pni", "User prompt:"), ("code", he.RAG_USER),
          ("h2", "A.3 E1b: Retrieval-Augmented Generation with the Full Note (Implemented, Not Run)"), ("pni", "System prompt:"), ("code", e1b.E1B_SYSTEM), ("pni", "User prompt:"), ("code", e1b.E1B_USER),
          ("h2", "A.4 E3: Chain-of-Verification (Implemented, Not Run)"), ("pni", "Verifier system prompt:"), ("code", e3.VERIFY_SYSTEM), ("pni", "Verifier user prompt (one call per claim):"), ("code", e3.VERIFY_USER),
          ("pni", "Revision system prompt:"), ("code", e3.REVISE_SYSTEM), ("pni", "Revision user prompt:"), ("code", e3.REVISE_USER)]
    # ── Appendix B: per-document results
    rows = []
    for _, r in R.cmp.sort_values("doc_id").iterrows():
        rows.append([str(int(r.doc_id)), "Consult" if r.specialty.startswith("Consult") else "Discharge", str(int(r.source_word_count)),
                     f3(r.E0_UFR), f3(r.E0_CR), f3(r.E1_UFR), f3(r.E1_CR), f3(r.E2_UFR), f3(r.E2_CR), str(int(r.E0_n_claims)), str(int(r.E1_n_claims)), str(int(r.E2_n_claims))])
    b += [("h1", "Appendix B: Per-Document Results"),
          ("pni", "UFR and CR per document and condition (header lines excluded), with the number of claims. Document identifiers are the row indices of the sampled MTSamples subset."),
          ("table", dict(caption="Per-document Unsupported Fact Rate, Contradiction Rate and claim counts under each condition.",
                         columns=["Doc", "Type", "Words", "E0 UFR", "E0 CR", "E1 UFR", "E1 CR", "E2 UFR", "E2 CR", "E0 n", "E1 n", "E2 n"],
                         widths=[0.4, 0.75, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.45], font=8, rows=rows))]
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
                               ["e1b_fullnote_rag_eval.py, e3_cove_eval.py", "Implemented extensions E1b and E3 (require OPENAI_API_KEY; not run in this thesis)"],
                               ["preprocessing.py", "Header filter and MTSamples line-break repair"],
                               ["recompute_metrics.py", "Per-document metrics, paired Wilcoxon tests, bootstrap CIs, effect sizes, header-filter tables"],
                               ["ablations.py", "Threshold, top-k and cleaned-evidence ablations; coverage proxy; negation analysis; error taxonomy"],
                               ["calibration_annptsumm.py, calibration_variants.py", "Judge validation against ann-pt-summ expert annotations (local models only)"],
                               ["analyze.py", "All figures and the qualitative example tables"],
                               ["thesis/build_thesis.py", "Generates this document from the result files"],
                               ["run.sh, setup.sh, requirements.txt", "Environment and end-to-end reproduction (bash run.sh --offline reproduces every reported number without API calls)"]])),
          ("pni", "Reproduction from a clean machine:"),
          ("code", "git clone https://github.com/slay-a/medical-hallucination-eval.git\ncd medical-hallucination-eval\nbash setup.sh\n# place mtsamples.csv one directory above the repository\nexport OPENAI_API_KEY='sk-...'   # only for generation\nbash run.sh            # full chain\nbash run.sh --offline  # recompute everything from the stored results\npython thesis/build_thesis.py")]
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
