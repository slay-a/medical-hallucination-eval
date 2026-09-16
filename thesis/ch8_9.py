"""ch8_9.py — Chapters 8 (Discussion) and 9 (Conclusion and Future Work)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from results_loader import Results, f3, f2, pct, pct0, fp, fpn, signed, COND_NAME
from mimic_results import Mimic, _csv

FIG = "results"
NAME = {"E0": "E0 zero-context LLM", "E1": "E1 RAG (excerpts only)", "E1b": "E1b RAG (note + excerpts)", "E3": "E3 RAG + CoVe", "E2": "E2 extractive"}
VNAME = {"max3": "Maximum over top-3 sentences (pipeline)", "concat3": "Top-3 sentences concatenated", "concat5": "Top-5 sentences concatenated",
         "maxall": "Maximum over all source sentences", "concat_ctx": "Whole hospital course as premise"}


def P(t):
    return ("p", t)


def H1(t):
    return ("h1", t)


def H2(t):
    return ("h2", t)


def H3(t):
    return ("h3", t)


def rel(t):
    return f"{abs(t.rel_change_mean):.0f}%"


def blocks(R: Results) -> list:
    b = []
    C = R.conds
    M = Mimic(); jr = M.judge_row()
    _agr = _csv("mimic_judge_agreement.csv")
    agr_all = _agr.set_index("condition").loc["all"] if _agr is not None and "all" in set(_agr.condition) else None
    t10u, t10c = R.test("UFR", "E0", "E1"), R.test("CR", "E0", "E1")
    t20u, t20c = R.test("UFR", "E0", "E2"), R.test("CR", "E0", "E2")
    t21u, t21c = R.test("UFR", "E1", "E2"), R.test("CR", "E1", "E2")
    lc = R.label_counts
    tot = {c: int(lc.loc[c].sum()) for c in C}
    share = {(c, l): lc.loc[c, l] / tot[c] for c in C for l in lc.columns}
    cal_all, cal_gen, cal_doc = R.cal_row("all"), R.cal_row("generated"), R.cal_row("doctor_written")
    var_all = R.cal_var[R.cal_var.group == "all"].set_index("variant") if R.cal_var is not None else None
    cov = R.cov.set_index("condition")
    tau = R.abl_tau.set_index("tau"); topk = R.abl_topk.set_index("k"); clean = R.abl_clean.set_index("condition")
    e2c_total = int(R.e2err_value("Contradicted_total")); e2c_neg = int(R.e2err_value("Contradicted_with_negation"))
    e2c_both = int(R.e2err_value("Contradicted_both_above_0.9"))
    e2_n = tot["E2"]; e2_ns = int(lc.loc["E2", "Not-Supported"])
    trans = pd.crosstab(R.claims_clean[R.claims_clean.condition != "E2"].original_label, R.claims_clean[R.claims_clean.condition != "E2"].label)
    e2clean_contra = int(clean.loc["E2", "n_contradicted"]); e2clean_n = int(clean.loc["E2", "n_claims"])
    # per-specialty means
    spec = R.cmp.groupby("specialty")[[f"{c}_{m}" for c in C for m in ("UFR", "CR")]].mean()
    spec_n = R.cmp.groupby("specialty").size()
    rho_u, p_u = spearmanr(R.cmp.source_word_count, R.cmp.delta_UFR); rho_c, p_c = spearmanr(R.cmp.source_word_count, R.cmp.delta_CR)
    rho_e0u, p_e0u = spearmanr(R.cmp.source_word_count, R.cmp.E0_UFR); rho_e0c, p_e0c = spearmanr(R.cmp.source_word_count, R.cmp.E0_CR)
    # taxonomy
    cats = ["Follow-up or scheduling", "Findings, exam, labs or imaging", "Generic advice or patient education", "History, symptoms or timeline",
            "Procedure or treatment", "Medication or dosage", "Diagnosis or assessment", "Other or unclassified"]
    GEN = [c for c in C if c != "E2"]
    tax_rows = []
    for cat in cats:
        row = [cat]
        for c in GEN:
            tc = R.tax_counts[(R.tax_counts.condition == c) & (R.tax_counts.category == cat)]
            allc = int(tc[tc.label == "All unsupported"]["count"].sum()); con = int(tc[tc.label == "Contradicted"]["count"].sum())
            row += [str(allc), str(con)]
        tax_rows.append(row)
    uns_total = {c: int(R.tax_counts[(R.tax_counts.condition == c) & (R.tax_counts.label == "All unsupported")]["count"].sum()) for c in GEN}
    OTHERS = [c for c in C if c != "E0"]
    n_fixed = len(R.ex_fixed) if R.ex_fixed is not None else 0
    has_ext = {"E1b", "E3"} <= set(C)
    t1b0u, t1b0c = R.test("UFR", "E0", "E1b"), R.test("CR", "E0", "E1b")
    t30u, t30c = R.test("UFR", "E0", "E3"), R.test("CR", "E0", "E3")
    t1b1u, t1b1c = R.test("UFR", "E1", "E1b"), R.test("CR", "E1", "E1b")
    t31u, t31c = R.test("UFR", "E1", "E3"), R.test("CR", "E1", "E3")
    t31bu, t31bc = R.test("UFR", "E1b", "E3"), R.test("CR", "E1b", "E3")
    n_abst = int(R.claims.is_abstention.sum()) if "is_abstention" in R.claims.columns else 0
    abst = R.claims[R.claims.is_abstention] if n_abst else R.claims.iloc[0:0]
    n_abst_docs = int(abst.doc_id.nunique()) if n_abst else 0
    n_abst_contra = int((abst.label == "Contradicted").sum()) if n_abst else 0
    try:
        verif = pd.read_csv("results/e3_verification.csv")
    except Exception:  # noqa: BLE001
        verif = None
    n_verif = len(verif) if verif is not None else 0
    n_verif_unsup = int((verif.verdict != "SUPPORTED").sum()) if verif is not None else 0
    fewer_e3 = int((R.cmp.E3_n_claims < R.cmp.E1_n_claims).sum()) if has_ext else 0
    def covv(c):
        return cov.loc[c, "coverage_at_0_6"] if c in cov.index else float("nan")
    def covp(a, b):
        col = f"p_cov06_{b}_vs_{a}"
        return cov.loc["E0", col] if col in cov.columns else float("nan")

    # ══════════════════════════════════ CHAPTER 8 ══════════════════════════════════
    b += [H1("Chapter 8 Discussion"), H2("8.1 Answers to the Research Questions")]
    b += [P(f"**RQ1.** Measured by the claim-level NLI judge, {pct0(R.mean('E0','UFR'))} of the sentences in zero-context "
            f"GPT-4o-mini summaries could not be verified against the note and {pct0(R.mean('E0','CR'))} were labeled as "
            f"contradicting it. The validation study shows that these numbers overstate the problem several-fold: on summaries "
            f"where medical experts flagged 5 to 9 percent of sentences, the same judge flagged 86 to 87 percent. The honest answer "
            f"to RQ1 is therefore twofold: the summaries do contain unsupported statements, in every note and in identifiable "
            f"categories, but this instrument cannot say how many, and its absolute rates should not be reported as hallucination "
            f"rates."
            + (f" The main study answers the question with a validated instrument on real hospital courses: under the selected judge, "
               f"{pct0(M.mean('E0','UFR'))} of the claims in zero-context summaries written by the local model were not supported by the course "
               f"and {pct0(M.mean('E0','CR'))} contradicted it, against a judge error floor of {pct0(M.mean('E2','UFR'))} measured on verbatim "
               f"extracts; the clinicians' own discharge instructions scored {pct0(M.mean('REF','UFR'))} under the same judge, because they "
               f"contain what the course does not say. The honest rate of unsupported content in zero-context summaries is therefore of the "
               f"order of {pct0(M.mean('E0','UFR') - M.mean('E2','UFR'))} above the floor, an estimate made with an instrument of kappa "
               f"{f2(jr.kappa) if jr is not None else '—'}." if M.ok else "")),
          P("**RQ2.** The statements that the judge cannot support fall predominantly into follow-up instructions, generic advice, "
            "restatements of history and paraphrased findings. Two of these categories, follow-up and advice, are largely "
            "extrinsic: the note says nothing about them, the model supplies them from its knowledge of what patient instructions "
            "contain, and retrieval cannot remove them because they are not generated from the retrieved text. The judge itself "
            "errs in three characteriztic ways: it cannot match a patient-facing paraphrase that aggregates several sentences of "
            "the note to any single sentence and therefore calls it Not-Supported; it treats negated symptom lists compared with "
            "loosely related sentences as contradictions; and it inherits sentence fragments from the corpus's line-break "
            "artifact. The first error dominates, and no aggregation of evidence corrects it."),
          P(f"**RQ3.** Under the judge, excerpt-only RAG reduced contradictions by about a quarter and unsupported facts not at "
            f"all, while covering {pct(covv('E1'))} instead of {pct(covv('E0'))} of the note's sentences and producing summaries "
            f"about {R.words['E0'].mean() - R.words['E1'].mean():.0f} words shorter. Retrieval with the full note (E1b) kept the "
            f"baseline's length and coverage and lowered the unsupported fact rate by about {abs(t1b0u.rel_change_mean):.0f} percent "
            f"relative to E0. Verification on top of retrieval (E3) lowered the unsupported fact rate by about "
            f"{abs(t30u.rel_change_mean):.0f} percent relative to E0 and {abs(t31u.rel_change_mean):.0f} percent relative to E1, left "
            f"contradictions unchanged, and covered {pct(covv('E3'))} of the note. The extractive summarizer reached the lowest rates "
            f"on both metrics and the highest coverage per word, at the cost of readability. Given the judge's weak validity, the "
            f"safest statement is that retrieval and verification changed the model's output in the directions predicted, most "
            f"visibly for medication, procedure and finding statements and for extrinsic follow-up advice respectively, but that "
            f"the size of the true improvement is unknown, and that the coverage cost is real for excerpt-only retrieval but not for "
            f"retrieval with the full note or for verification at the primary threshold."
            + (f" The main study, with the validated judge and a reference written by the clinician, gives the answer the pilot could not. "
               f"Excerpt-only retrieval {M.verb(M.test('UFR','E1'))} the unsupported fact rate by {M.rel(M.test('UFR','E1'))} relative to E0 "
               f"({fp(M.test('UFR','E1').p)}), retrieval with the whole course by {M.rel(M.test('UFR','E1b'))} ({fp(M.test('UFR','E1b').p)}) and "
               f"verification by {M.rel(M.test('UFR','E3'))} ({fp(M.test('UFR','E3').p)}), and the three do not differ from one another "
               f"({fp(M.test('UFR','E1b','E1').p)} and {fp(M.test('UFR','E3','E1').p)} against E1). None of them changed the contradiction rate. "
               f"The coverage cost separates them: excerpt-only retrieval covered {pct0(M.mean('E1','coverage_ref'))} of the clinician's sentences "
               f"against {pct0(M.mean('E0','coverage_ref'))} for E0 ({fp(M.test('coverage_ref','E1').p)}), verification {pct0(M.mean('E3','coverage_ref'))} "
               f"({fp(M.test('coverage_ref','E3').p)}), whereas retrieval with the whole course kept it at {pct0(M.mean('E1b','coverage_ref'))} "
               f"({fp(M.test('coverage_ref','E1b').p)}). The extractive control reached {f3(M.mean('E2','UFR'))}, so the residual rates of the three "
               f"grounded LLM conditions are within judge error of a system that cannot fabricate." if M.ok and M.test('UFR', 'E1') is not None else ""))]
    b += [H2("8.2 Why the Judge Disagrees with the Experts")]
    b += [P("Four causes, in decreasing order of importance, explain the disagreement. The first is a *granularity mismatch*. A "
            "patient-facing sentence such as \"You came in because of stomach pain after meals and foul-smelling urine, and you "
            "were found to have a urinary infection and reflux\" compresses the chief complaint, the history and the assessment "
            "sections of a note into one sentence. No single source sentence entails it, so the maximum over three retrieved "
            "sentences is low, and concatenating three or five sentences helps only marginally because the model was trained on "
            "single-sentence premises [[cite:bowman2015,williams2018]]. Laban et al. showed that sentence-level NLI works when "
            "summary sentences are roughly co-extensive with source sentences [[cite:laban2022]]; patient-facing paraphrase "
            "violates that assumption systematically. The experts, by contrast, marked a sentence only when it introduced a fact "
            "that the hospital course did not contain, which is a question about content, not about sentence alignment."),
          P("The second cause is *model capacity and domain*. The cross-encoder used here has six layers and was fine-tuned on "
            "crowd-sourced general-domain pairs. The TRUE meta-evaluation found that reliable factual-consistency judgments "
            "required NLI models of billions of parameters fine-tuned on diverse consistency data [[cite:honovich2022]], and "
            "Romanov and Shivade documented the loss of accuracy when general NLI models meet clinical language, abbreviations "
            "and negation [[cite:romanov2018]]. The negation results of Section 5.5.4 are a direct manifestation. The third "
            "cause is the *decision rule*: taking the maximum contradiction probability over several evidence sentences turns "
            "any spurious contradiction into the claim's label, which is why CR grows with the number of evidence sentences even "
            "for verbatim text. The fourth cause is the *definition of the target*. The UFR counts every claim that is not "
            "entailed, including appropriate extrinsic advice, whereas the experts marked unsupported facts. Some of the "
            "judge's Not-Supported labels are therefore correct descriptions of extrinsic content that a clinician would accept. "
            "This does not rescue the metric, because the same labels are assigned to faithful paraphrases, but it means that "
            "the two instruments were never measuring exactly the same construct."),
          P("These findings agree with the broader literature on evaluating the evaluators. Falke et al. warned in 2019 that "
            "off-the-shelf NLI models could not rank summaries by correctness [[cite:falke2019]]; the present study shows that "
            "the warning still applies to small models in the clinical domain six years later, even after the retrieval and "
            "aggregation improvements that made SummaC and AlignScore successful on news data [[cite:laban2022,zha2023]].")]
    jc = _csv("judge_candidates.csv")
    if jc is not None:
        A = jc[jc.group == "all"].set_index(["judge", "mode"])
        def jrow(j, m):
            return A.loc[(j, m)] if (j, m) in A.index else None
        mini, deb, med = jrow("minilm_nli", "top3"), jrow("deberta_large_nli", "doc"), jrow("mednli_deberta_large", "doc")
        bmc = jrow("bespoke_minicheck_7b", "doc")
        ftj = M.finetune_summary()
        b += [P(f"The judge selection study of Chapter 6 sharpens this diagnosis. Replacing the six-layer cross-encoder with a 24-layer "
                f"DeBERTa-v3-large model trained on five NLI and fact-verification datasets raised kappa from {f2(mini.kappa)} to {f2(deb.kappa)} and "
                f"AUROC from {f2(mini.auroc)} to {f2(deb.auroc)} on all sentences, and letting the model see the whole hospital course in windows "
                f"was better than three retrieved sentences; the second cause, capacity, is therefore real and large."
                + (f" Adapting the same model to clinical language with MedNLI raised its MedNLI accuracy from {pct(ftj['zero_shot_dev_accuracy'])} to "
                   f"{pct(ftj['dev_accuracy'])} but {'lowered' if med.kappa < deb.kappa else 'raised'} its agreement with the experts (kappa {f2(med.kappa)} versus "
                   f"{f2(deb.kappa)}), which argues that clinical vocabulary is not what limits the judge on this task; the granularity mismatch is."
                   if med is not None and ftj is not None and "zero_shot_dev_accuracy" in ftj else "")
                + ((f" A seven-billion-parameter language model fine-tuned for grounding checks (Bespoke-MiniCheck-7B), which reads the whole "
                    f"course in one pass and leads the general fact-checking leaderboards, reached kappa {f2(bmc.kappa)} and AUROC {f2(bmc.auroc)} on the "
                    f"same sentences, {'above' if bmc.kappa > deb.kappa else 'below'} the encoder NLI model; ") if bmc is not None else " ")
                + f"{'the selected judge' if bmc is None else ('it' if bmc.kappa > deb.kappa else 'even the best candidate')} flags "
                f"{pct0(jr.flag_rate) if jr is not None else pct0(deb.flag_rate)} of sentences where the experts flag {pct0(deb.expert_rate)}, agrees with them at a "
                f"kappa of {f2(jr.kappa) if jr is not None else f2(deb.kappa)} and misses {100 - round(100 * (jr.recall if jr is not None else deb.recall))} of every 100 "
                f"expert-flagged sentences, so the main study's rates are estimates from a moderately valid instrument, and Chapter 7 reads them as such.")]
    b += [H2("8.3 Interpreting the Effects of Retrieval and Verification")]
    b += [P(f"The RAG effect must be read in the light of Sections 5.4 and 5.5. It is modest (a {rel(t10c)} relative reduction of "
            f"CR, d(z) = {f2(t10c.d_z)}), it disappears when the judge is made more conservative, and part of the change it "
            f"measures is a change in *how* the model writes rather than in *what* it asserts: RAG summaries are shorter, contain "
            f"fewer claims and fewer generic sentences, and lower CR is partly a consequence of having fewer sentences that a "
            f"noisy judge can call contradictions. The categories in which RAG helped most, medications, procedures and findings, "
            f"are precisely those for which the excerpts contain specific, near-verbatim facts, which is the mechanism by which "
            f"grounding is expected to work [[cite:lewis2020]]. The categories in which it did not help, follow-up instructions "
            f"and advice, are generated from the model's prior over patient instructions rather than from the input, and neither "
            f"retrieval nor a stricter prompt is likely to remove them; a verification step that checks each sentence against the "
            f"note and removes what it cannot support (E3) or an abstention rule (\"Not stated in the note\") is the natural remedy. "
            f"The coverage result is the clearest cost of excerpt-only RAG: what the retriever does not return, the summary "
            f"cannot contain, and the query used here (the document description and the opening lines) is a weak proxy for the "
            f"four sections the summary must cover. E1b, which shows the model the excerpts and the note, tests whether focus can "
            f"be gained without losing coverage, and the answer is a qualified yes: E1b matched the baseline's coverage and "
            f"length and produced a modest reduction in unsupported claims, but no reduction in contradictions. Focusing "
            f"attention helps; withholding content is what reduced contradictions in E1, and it did so partly by producing fewer "
            f"and shorter sentences."),
          P(f"Verification behaves differently from retrieval because it acts after generation on individual sentences. The E3 "
            f"verifier judged about a quarter of the draft claims unsupported by the retrieved evidence, and the revision removed or "
            f"replaced them and abstained where a section was left empty. The result is the largest reduction in unsupported "
            f"claims among the LLM conditions and the only reduction in the extrinsic categories, follow-up instructions and generic "
            f"advice, that retrieval could not touch. Two costs follow. Verified summaries are shorter, cover less of the note at the "
            f"looser similarity threshold, and their abstentions, while honest, are themselves a form of omission; at the primary "
            f"threshold, however, their coverage matched the baseline because the verifier's corrections import content from the "
            f"full note. Verification also did not lower the judge's "
            f"contradiction rate, which is consistent with two readings that the present data cannot separate: the verifier, which "
            f"is the same LLM, shares the judge's difficulty with negated findings and near-verbatim clinical statements, or the "
            f"judge's contradiction label is too noisy to register a change of the size that verification could plausibly produce. "
            f"The validation study favors the second reading. Finally, verification depends on the same retrieval as the judge, so "
            f"a claim whose support lies in an unretrieved sentence is deleted rather than confirmed; the {fewer_e3} of {R.n_docs} documents in "
            f"which E3 has fewer claims than its draft include such losses.")]
    if M.ok:
        e1b_cov, e1_cov = M.test("coverage_ref", "E1b"), M.test("coverage_ref", "E1")
        b += [P(f"The main study confirms the mechanism the pilot could only suggest, and resolves the trade-off in favour of one design. "
                f"Every grounded condition, whether it withholds the course (E1), adds excerpts to it (E1b) or checks each sentence after "
                f"generation (E3), cut the unsupported fact rate by about {M.rel(M.test('UFR','E1b'))} relative to the zero-context baseline, and the "
                f"three are statistically indistinguishable. What separates them is omission. Withholding the course cost "
                f"{abs(e1_cov.delta_mean):.2f} of coverage of the clinician's instructions ({fp(e1_cov.p)}) and verification cost "
                f"{abs(M.test('coverage_ref','E3').delta_mean):.2f} ({fp(M.test('coverage_ref','E3').p)}), partly through its "
                f"{M.mean('E3','abstentions'):.1f} abstentions per summary, whereas showing the model the excerpts together with the whole course "
                f"{'kept coverage at the baseline level' if e1b_cov.p >= 0.05 else 'changed coverage'} ({fp(e1b_cov.p)}). Focusing the model's attention "
                f"on retrieved passages therefore reduces unsupported content on its own; withholding the rest of the document adds nothing "
                f"to faithfulness and takes away completeness. The price E1b pays is in its citations: because it may draw on the whole "
                f"course while citing only the excerpts, only {pct0(M.mean('E1b','citation_accuracy'))} of its citations are supported by the passage "
                f"they name, against {pct0(M.mean('E1','citation_accuracy'))} for E1, so a reviewer who follows E1b's citations will often not find "
                f"the evidence there."),
              P(f"Two further results guard against over-reading these gains. The verbatim extractive control reached a UFR of "
                f"{f3(M.mean('E2','UFR'))}, which is the judge's error floor on this corpus, and the grounded LLM conditions sit within a few "
                f"hundredths of it; the judge cannot tell how much of their residual unsupported content is real"
                + ((lambda g: f" (a second judge of a different family flags only {pct(g.loc['E2','flag_rate_second'])} of the same verbatim claims, confirms "
                              f"{pct0(g.loc['all','first_flags_confirmed'])} of the selected judge's flags overall, and bounds the unsupported share of the grounded "
                              f"conditions between {f3(min(g.loc[c,'UFR_both'] for c in ('E1','E1b','E3')))} and {f3(max(g.loc[c,'UFR_either'] for c in ('E1','E1b','E3')))}, "
                              f"Section 7.8)")(_agr.set_index("condition")) if _agr is not None and all(c in set(_agr.condition) for c in ("E2", "all", "E1", "E1b", "E3")) else "")
                + f". And the clinicians' own "
                f"instructions received a UFR of {f3(M.mean('REF','UFR'))} under the same judge, higher than every LLM condition, because they "
                f"contain medication changes, appointments and advice that the hospital course never states. A metric of support by the "
                f"source is a metric of source-faithfulness, not of clinical correctness, and a summarizer that only paraphrases the course "
                f"will always beat a clinician on it. The retrieval ablations sharpen the picture from the pilot. "
                f"{M.faithfulness_phrase()[0].upper() + M.faithfulness_phrase()[1:]}, whereas coverage rose and fell with the amount of retrieved "
                f"text and citation accuracy fell when five chunks were retrieved. "
                + ("Requiring the model to cite an excerpt after every sentence is therefore itself a grounding device: it costs nothing and "
                   "kept the model closer to the retrieved text. " if any(v == "nocite" and d > 0 for v, d, _ in M.sig_ablations("UFR")) else "")
                + f"How much of the course the model sees governs omission; how it is retrieved matters little."),
              P(f"The question-answering pilot points the same way from a different angle. Questions whose answers are usually in the course "
                f"(medications) were answered faithfully; questions about warning signs, which a hospital course rarely states, drew the most "
                f"abstentions and, when answered, the highest unsupported rates, because the model supplied the standard advice from its "
                f"training data. Retrieval {'raised' if M.qa_tot is not None and M.qa_tot.loc['QA-E1','abstention_rate'] > M.qa_tot.loc['QA-E0','abstention_rate'] else 'changed'} "
                f"abstention and lowered the unsupported rate of the answered questions, again partly by making the model say less. The "
                f"extrinsic categories identified in the pilot are thus the same categories that limit document-grounded question "
                f"answering, and abstention, not retrieval, is the appropriate response to them.")]
    b += [H2("8.4 Comparison of the Three Approaches")]
    b += [P("[[tab:compare]] summarizes the strengths, weaknesses and appropriate uses of the five conditions in the light of "
            "all results. No approach dominates. The zero-context LLM is the most readable and complete but adds the most "
            "extrinsic content. Excerpt-only RAG reduces contradictions modestly, shortens summaries and loses coverage; RAG with "
            "the full note keeps coverage and gains a little in supported content. Verification removes the most unsupported "
            "content, including the extrinsic advice that retrieval cannot reach, but does so by deleting and abstaining. The "
            "extractive summarizer is faithful by construction and covers the note broadly but produces text that no patient "
            "should be handed unedited: fragments, abbreviations and clinician-facing phrasing. The ordering of the conditions by "
            "the judge's metrics is consistent with their designs, which is some reassurance that the judge responds to real "
            "differences between systems when those differences are large; what it cannot do is quantify them."),
          ("table", dict(label="compare", caption="Comparison of the five conditions across both studies. Pilot values are per-document means under the MiniLM judge on MTSamples; main-study values are per-document means under the selected judge on MIMIC-IV hospital courses; coverage is the share of the clinician's discharge-instruction sentences supported by the summary.",
                         columns=["Approach", "Pilot UFR / CR", "Main study UFR / CR", "Coverage of clinician's sentences", "Strengths", "Weaknesses", "Appropriate use"],
                         widths=[0.85, 0.7, 0.75, 0.75, 1.15, 1.15, 1.15], font=8, align=["left", "center", "center", "center", "left", "left", "left"],
                         rows=[[NAME[c], f"{f3(R.mean(c,'UFR'))} / {f3(R.mean(c,'CR'))}",
                                (f"{f3(M.mean(c,'UFR'))} / {f3(M.mean(c,'CR'))}" if M.ok and c in M.conds else "—"),
                                (pct0(M.mean(c, "coverage_ref")) if M.ok and c in M.conds else "—"), st, wk, use]
                               for c, st, wk, use in [
                                   ("E0", "Fluent, patient-friendly, complete structure; simplest to deploy", "Most extrinsic content; generic advice and follow-up invented from priors",
                                    "Drafts reviewed and edited by a clinician; low-stakes communication"),
                                   ("E1", "Fewer generic statements; shorter; every sentence cited", "Lowest coverage of the source; effect small and threshold-dependent in the pilot; retrieval query crude",
                                    "Focused summaries of long notes when omission is acceptable and a reviewer checks completeness"),
                                   ("E1b", "Keeps the baseline's coverage; cited; modestly fewer unsupported claims", "Still adds extrinsic advice; longest prompts",
                                    "Drop-in replacement for zero-context prompting when the note fits the context window"),
                                   ("E3", "Largest reduction in unsupported claims in the pilot; removes extrinsic advice; explicit abstentions", "Shortest LLM summaries; abstentions are omissions; about ten model calls per note",
                                    "High-stakes patient communication where omission is safer than invention and a clinician fills gaps"),
                                   ("E2", "Cannot fabricate; no model cost; deterministic", "Unreadable for patients: fragments, abbreviations, no explanation; still mislabelled by the judge",
                                    "Clinician-facing digests; a grounding step before an LLM rewrites the selected sentences")]]))]
    b += [H2("8.5 Threats to Validity and Limitations")]
    jk = (f"kappa {f2(jr.kappa)}, precision {f2(jr.precision)}, recall {f2(jr.recall)}" if jr is not None else "moderate agreement")
    b += [("bullets", [
        "**Construct validity of the judge.** The pilot judge agrees with medical experts at near-chance levels, so the pilot's UFR and CR "
        "are indices of how a small NLI model reacts to a summary rather than measurements of hallucination. The main-study judge was "
        f"selected for its agreement with the same experts ({jk}), which is moderate rather than high: a sizeable share of the sentences "
        "it flags were not flagged by the experts, and it misses a sizeable share of what they flagged. Every rate in Chapter 7 is "
        "therefore an estimate produced by an instrument with a known, but non-negligible, error profile. The between-condition "
        "comparisons remain informative because every condition is scored on the same documents by the same instrument, but the "
        "magnitude of any true effect carries this measurement error."
        + ((lambda g: f" Two checks bound it: a second judge of a different model family confirms {pct0(g.first_flags_confirmed)} of the selected "
                      f"judge's flags on the main-study claims (kappa {f2(g.kappa)} between the judges, Section 7.8), and the ordering of the "
                      f"conditions is unchanged under the second judge and under the conjunction of the two.")(agr_all) if agr_all is not None else ""),
        "**Transfer of the validation.** For the pilot, the judge was validated on MIMIC-IV summaries and applied to MTSamples notes. The "
        "main study removes that gap, since the judge is applied to the very hospital courses the expert annotations concern, but the "
        "annotations cover doctor-written instructions and summaries written by GPT-4 and Llama models, not the output of the local "
        "model used here, so the judge's threshold and error profile are transferred across generators rather than measured on them.",
        f"**Sample size and power.** Fifty pilot documents give adequate power for large paired effects and marginal power for small ones "
        f"(the E1 effect on CR, d(z) = {f2(t10c.d_z)}, crosses the significance boundary under small changes to the judge). The 110 "
        f"main-study courses give adequate power for moderate paired effects; the question-answering pilot, with three questions per course, "
        f"supports descriptive comparison only.",
        "**Single generation.** Each summary and each answer was generated once at temperature 0.3, so per-document differences confound "
        "the effect of a condition with sampling noise in both studies.",
        "**External validity of the corpora.** MTSamples documents are sample transcriptions, not patient records; they are shorter and "
        "more formulaic than hospital notes, their line breaks are corrupted, and the corpus is public and may be present in the "
        "generator's training data. The main-study courses come from a single institution and a single section of the discharge "
        "summary, and the doctor-written instructions used as the coverage reference reflect one clinician's choices of what to tell "
        "the patient; they also contain content that the hospital course does not (the experts marked such spans), so complete coverage "
        "of them is neither attainable nor desirable.",
        "**Two generators of different capability.** The pilot used GPT-4o-mini through an API and the main study a four-bit, "
        "seven-billion-parameter open-weight model on a laptop, the largest model that the data use agreement and the hardware allowed. "
        "Results are compared across the studies qualitatively, never numerically, and each study used one prompt per condition, one "
        "retriever and one verification design.",
        "**Heuristic components.** The header filter, the abstention rule and the citation parser are regular expressions, the pilot's "
        "coverage proxy is an embedding-similarity threshold without human validation, and the error taxonomy is keyword-based with "
        "author review. Each is reported transparently so that it can be replaced.",
        "**Reference-based coverage and citation accuracy use the same judge.** Both main-study measures run the selected NLI model in a "
        "new direction (summary as premise, instruction sentence as hypothesis; cited excerpt as premise, claim as hypothesis) for which "
        "its agreement with experts was not separately validated.",
        "**Verification design.** E3 uses one verifier (the generator itself, at temperature 0), one prompt and the judge's own retriever; "
        "a claim whose support lies in an unretrieved sentence is deleted rather than confirmed, and abstentions, while honest, are "
        "omissions. The abstention rule was written after the mislabeling of abstentions was observed in the pilot; pre-exclusion numbers "
        "are reported in Section 5.5.4.",
        "**Question-answering pilot.** Three fixed questions without clinician-written gold answers; agreement with the clinician's "
        "discharge instructions is a proxy for correctness, and a correct answer that the clinician did not write is counted as unsupported."])]
    b += [H2("8.6 Implications for the Design and Evaluation of Clinical Summarization Systems")]
    b += [("numbers", [
        "**Validate the evaluator before the system.** An automatic hallucination metric should be reported together with its agreement "
        "with clinician annotations on the target domain; without that, a lower score cannot be interpreted. In this thesis the same "
        "check, on the same 1,781 sentences, rejected one judge and selected another, and the ann-pt-summ annotations make it "
        "inexpensive for any patient-summary metric.",
        "**Choose the judge by capacity and evidence mode, not by aggregation tweaks.** Changing how a small model's evidence was "
        "aggregated changed little; a larger consistency-trained NLI model with the whole document as evidence roughly doubled kappa. "
        "Even so, the selected judge's precision and recall leave room for error, so its rates should be published with those figures.",
        "**Prefer paired, within-document comparisons and report effect sizes.** Absolute rates from a noisy judge are not meaningful, "
        "but the same judge applied to two systems on the same notes still ranks large differences correctly; confidence intervals and "
        "robustness to the judge's thresholds should accompany every claim of improvement.",
        "**Always include a verbatim extractive control and a human reference.** The extractive control measures the metric's error "
        "floor directly, and scoring the clinician's own text under the same judge shows what rate a faithful human writer receives; "
        "any reported rate should be read against both.",
        "**Measure omission against what a clinician actually told the patient.** Retrieval that shrinks the model's view trades "
        "unsupported content for omissions; coverage of the clinician-written instructions makes that trade-off visible in a way that a "
        "source-side proxy cannot.",
        "**Require citations.** Sentence-level citations cost nothing at generation time, give reviewers a direct path to the evidence, "
        "and turn an unverifiable statement into a checkable one; citation accuracy is a second, cheaper check on grounding.",
        "**Target the failure categories directly.** Extrinsic follow-up and advice sentences are not removed by retrieval; abstention "
        "rules, per-claim verification against the note, or templated advice sections approved by clinicians are more appropriate remedies.",
        "**Keep protected text local.** An open-weight model on a laptop made a study on real hospital courses possible without any data "
        "leaving the machine; the price is a smaller generator, which the evaluation must take into account."])]

    # ══════════════════════════════════ CHAPTER 7 ══════════════════════════════════
    b += [H1("Chapter 9 Conclusion and Future Work"), H2("9.1 Summary")]
    b += [P(f"This thesis set out to measure and reduce hallucinations in LLM-generated patient-facing summaries of clinical "
            f"notes. It built a reproducible claim-level evaluation pipeline from open components, compared five summarization "
            f"conditions on {R.n_docs} MTSamples notes with paired statistics: zero-context GPT-4o-mini summarization, retrieval-"
            f"augmented generation with excerpts only and with the full note, retrieval-augmented generation followed by "
            f"Chain-of-Verification, and a centroid-based extractive summarizer. It corrected two evaluation artifacts that had "
            f"inflated an earlier report of the same experiment and identified a third, the mislabeling of abstentions, and, for "
            f"the first time for this class of judge on patient summaries, it validated the pipeline's labels against "
            f"medical-expert annotations of unsupported facts."),
          P(f"Under the judge, excerpt-only retrieval reduced the contradiction rate by about a quarter, left the unsupported-fact "
            f"rate unchanged and reduced coverage of the note; retrieval with the full note lowered the unsupported-fact rate "
            f"modestly without losing coverage; verification lowered it by about {abs(t30u.rel_change_mean):.0f} percent, the "
            f"largest change of any LLM condition, by deleting unsupported claims and abstaining; and the extractive summarizer "
            f"bounded both rates from below and exposed a non-zero error floor. Against the experts, the judge flagged {pct0(cal_all.judge_UFR)} of sentences where the experts "
            f"flagged {pct0(cal_all.expert_flag_rate)}, agreed with them at chance level, and ranked systems in nearly the "
            f"opposite order; no evidence-aggregation rule changed that conclusion. The principal contribution of the thesis is "
            f"therefore methodological: it shows, with quantitative evidence, that a small general-domain NLI judge applied at "
            f"sentence granularity is not a valid hallucination detector for patient-facing clinical summaries, and it provides "
            f"the tools, the controls and the validation protocol needed to evaluate better judges and better summarizers.")
          ] + ([P(f"The thesis then acted on that finding. A judge selection study scored six off-the-shelf judges and a MedNLI-adapted "
                  f"model against the same expert annotations and selected {M.judge_name()} with whole-course evidence, which "
                  f"agrees with the experts at kappa {f2(jr.kappa) if jr is not None else '—'} and AUROC {f2(jr.auroc) if jr is not None else '—'}; the "
                  f"clinical adaptation raised MedNLI accuracy by nine points but not agreement with the experts. With that judge, the main "
                  f"study repeated the five conditions on {M.n_docs} MIMIC-IV hospital courses with an open-weight model running on the "
                  f"author's laptop, measured omission against the discharge instructions the clinicians actually wrote, and added citation "
                  f"accuracy, six retrieval ablations and a question-answering pilot. Retrieval, retrieval with the whole course and "
                  f"verification each lowered the unsupported fact rate by about {M.rel(M.test('UFR','E1b'))}; only retrieval with the whole "
                  f"course did so without losing coverage; {M.faithfulness_short()}; and the clinicians' own instructions "
                  f"scored worse than every model under a metric of support by the source.")] if M.ok else [])
    b += [H2("9.2 Key Findings")]
    b += [("numbers", [
        f"Excerpt-only RAG reduced the judge's contradiction rate from {f3(R.mean('E0','CR'))} to {f3(R.mean('E1','CR'))} "
        f"({fp(t10c.p)}; Holm-adjusted {fp(t10c.p_holm)}) but not the unsupported-fact rate ({fp(t10u.p)}), and it reduced coverage "
        f"of the note ({fp(covp('E0','E1'))}). RAG with the full note (E1b) lowered the unsupported-fact rate to "
        f"{f3(R.mean('E1b','UFR'))} ({fp(t1b0u.p)}) with the baseline's coverage.",
        f"Chain-of-Verification on top of RAG (E3) lowered the unsupported-fact rate to {f3(R.mean('E3','UFR'))} ({fp(t30u.p)} versus "
        f"E0; {fp(t31u.p)} versus E1; d(z) = {f2(t30u.d_z)}), removed extrinsic follow-up and advice statements, and left the "
        f"contradiction rate unchanged; it did so by deleting claims and abstaining, producing the shortest LLM summaries.",
        f"The extractive summarizer reached UFR {f3(R.mean('E2','UFR'))} and CR {f3(R.mean('E2','CR'))} on verbatim text; its "
        f"non-zero rates are judge errors, two thirds of them on negated sentences.",
        f"Against 1,781 expert-annotated sentences, the judge's precision was {pct0(cal_all.any_precision)}, recall "
        f"{pct0(cal_all.any_recall)}, kappa {f2(cal_all.any_kappa)} and AUROC {f2(cal_all.auroc_1_minus_p_entail)}; the best "
        f"aggregation variant reached kappa {f2(var_all.best_kappa.max()) if var_all is not None else '—'}.",
        f"Counting markdown header lines as claims had inflated the RAG effect on CR from {rel(t10c)} to 32 percent relative; the "
        f"correction changes every previously reported number.",
        "Unsupported statements concentrate in follow-up instructions, generic advice and paraphrased history; retrieval reduces "
        "the categories that contain retrievable facts and not the extrinsic ones."]
        + ([f"Judge selection: a 24-layer DeBERTa-v3-large NLI model with whole-course evidence raised kappa against the experts from "
            f"{f2(cal_all.any_kappa)} to {f2(jr.kappa)} and AUROC from {f2(cal_all.auroc_1_minus_p_entail)} to {f2(jr.auroc)}; MedNLI fine-tuning raised "
            f"MedNLI accuracy from 80.6% to 87.8% but left kappa at 0.31, so granularity rather than clinical vocabulary limits agreement.",
            f"Main study ({M.n_docs} MIMIC-IV hospital courses, local Qwen2.5-7B-Instruct, validated judge): zero-context UFR {f3(M.mean('E0','UFR'))}; "
            f"E1 {f3(M.mean('E1','UFR'))}, E1b {f3(M.mean('E1b','UFR'))} and E3 {f3(M.mean('E3','UFR'))} (all {fp(max(M.test('UFR', c).p for c in ('E1','E1b','E3')))} "
            f"versus E0, indistinguishable from one another); CR unchanged at about {f3(M.mean('E0','CR'))}; verbatim extraction {f3(M.mean('E2','UFR'))} "
            f"(the judge's floor); the clinicians' own instructions {f3(M.mean('REF','UFR'))}.",
            f"Coverage of the clinician's instructions: E0 {pct0(M.mean('E0','coverage_ref'))}, E1b {pct0(M.mean('E1b','coverage_ref'))} "
            f"({fp(M.test('coverage_ref','E1b').p)} versus E0), E1 {pct0(M.mean('E1','coverage_ref'))} and E3 {pct0(M.mean('E3','coverage_ref'))} "
            f"(both {fp(max(M.test('coverage_ref', c).p for c in ('E1','E3')))}); citation accuracy {pct0(M.mean('E1','citation_accuracy'))} for E1 and "
            f"{pct0(M.mean('E1b','citation_accuracy'))} for E1b; {M.faithfulness_phrase()}, while coverage followed the amount of retrieved text.",
            f"Question answering with abstention: the model abstained on {pct0(M.qa_tot.loc['QA-E0','abstention_rate'])} of questions from the full course "
            f"and {pct0(M.qa_tot.loc['QA-E1','abstention_rate'])} from excerpts, most often for warning signs, which the course rarely states; UFR of "
            f"answered questions {f3(M.qa_tot.loc['QA-E0','UFR_mean'])} and {f3(M.qa_tot.loc['QA-E1','UFR_mean'])}." if M.qa_tot is not None else "",
            (lambda g: f"Second judge: Bespoke-MiniCheck-7B agrees with the selected judge on {pct0(g.loc['all','agreement'])} of the {int(g.loc['all','n_claims']):,} "
                       f"main-study claims (kappa {f2(g.loc['all','kappa'])}), confirms {pct0(g.loc['all','first_flags_confirmed'])} of its flags, flags "
                       f"{pct(g.loc['E2','flag_rate_second'])} of verbatim extracts (the selected judge's floor is its own error), and preserves the ordering of the "
                       f"conditions; the unsupported share of the grounded conditions lies between {f3(min(g.loc[c,'UFR_both'] for c in ('E1','E1b','E3')))} "
                       f"(flagged by both judges) and {f3(max(g.loc[c,'UFR_either'] for c in ('E1','E1b','E3')))} (flagged by either).")(_agr.set_index("condition"))
            if _agr is not None and all(c in set(_agr.condition) for c in ("E2", "all", "E1", "E1b", "E3")) else ""]
           if M.ok and jr is not None else []))]
    b += [H2("9.3 Future Work")]
    b += [("numbers", [
        "**Validate the judge on the generator's own output.** A small clinician annotation of local-model summaries would let the "
        "selected judge's threshold and error profile be measured, rather than transferred, for the system under test, and would show "
        "whether citation accuracy and reference coverage agree with clinical judgment.",
        "**Evaluate the verifier as a judge.** The E3 verifier is itself an LLM-based claim judge; scoring it on the ann-pt-summ "
        "annotations exactly as the NLI candidates were scored would show whether an LLM verifier agrees with medical experts better "
        "than the NLI model, and a verifier that sees the whole note should be compared with the retrieval-based design.",
        "**Stronger judges.** Atomic-claim decomposition [[cite:min2023]], larger alignment models [[cite:zha2023]], LLM-based judges "
        "calibrated against expert labels, and longer clinical NLI fine-tuning are natural next candidates for the same selection "
        "protocol; the scripts released here make each a one-day experiment.",
        "**Larger generators and larger samples on protected data.** A workstation-class GPU would allow open-weight models of 30 to 70 "
        "billion parameters, several generations per course to measure sampling variance, and more institutions and note types; a "
        "zero-data-retention agreement would allow the pilot's generator to be run on the same courses for a direct comparison.",
        "**Clinician study of omission and severity.** A small study in which clinicians rate the omissions and the clinical severity "
        "of unsupported statements would convert the coverage measures and the keyword taxonomy into validated measures and would "
        "settle whether extrinsic advice sentences are acceptable.",
        "**A full question-answering benchmark.** Clinician-written gold answers for a broader question set, calibrated abstention, and "
        "the same judge and controls would turn the pilot of Section 7.7 into the second task the proposal described."])]
    return b


