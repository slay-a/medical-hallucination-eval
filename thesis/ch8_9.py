"""ch8_9.py — Chapters 8 (Discussion) and 9 (Conclusion and Future Work)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from results_loader import Results, f3, f2, pct, pct0, fp, fpn, signed, COND_NAME

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
            f"rates."),
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
            f"retrieval with the full note or for verification at the primary threshold.")]
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
          ("table", dict(label="compare", caption="Comparison of the five conditions. Metric values are per-document means under the NLI judge; coverage is the share of source sentences matched at cosine 0.6.",
                         columns=["Approach", "Judge UFR / CR", "Coverage", "Words", "Strengths", "Weaknesses", "Appropriate use"],
                         widths=[0.9, 0.75, 0.6, 0.5, 1.3, 1.3, 1.15], font=8.5, align=["left", "center", "center", "center", "left", "left", "left"],
                         rows=[["E0 zero-context LLM", f"{f3(R.mean('E0','UFR'))} / {f3(R.mean('E0','CR'))}", pct0(covv('E0')), f"{R.words['E0'].mean():.0f}",
                                "Fluent, patient-friendly, complete structure; simplest to deploy", "Most extrinsic content; generic advice and follow-up invented from priors; highest CR",
                                "Drafts reviewed and edited by a clinician; low-stakes communication"],
                               ["E1 RAG, excerpts only", f"{f3(R.mean('E1','UFR'))} / {f3(R.mean('E1','CR'))}", pct0(covv('E1')), f"{R.words['E1'].mean():.0f}",
                                f"Fewer contradictions ({rel(t10c)} relative); fewer generic statements; shorter", "Low coverage; effect small and threshold-dependent; retrieval query crude",
                                "Focused summaries of long notes when omission is acceptable and a reviewer checks completeness"],
                               ["E1b RAG, note + excerpts", f"{f3(R.mean('E1b','UFR'))} / {f3(R.mean('E1b','CR'))}", pct0(covv('E1b')), f"{R.words['E1b'].mean():.0f}",
                                "Baseline length and coverage; modestly fewer unsupported claims", "No reduction in contradictions; still adds extrinsic advice",
                                "Drop-in replacement for zero-context prompting when the note fits the context window"],
                               ["E3 RAG + verification", f"{f3(R.mean('E3','UFR'))} / {f3(R.mean('E3','CR'))}", pct0(covv('E3')), f"{R.words['E3'].mean():.0f}",
                                "Largest reduction in unsupported claims; removes extrinsic advice; explicit abstentions; baseline coverage kept", "Shortest LLM summaries; abstentions are omissions; about ten API calls per note; no gain in contradictions",
                                "High-stakes patient communication where omission is safer than invention and a clinician fills gaps"],
                               ["E2 extractive", f"{f3(R.mean('E2','UFR'))} / {f3(R.mean('E2','CR'))}", pct0(covv('E2')), f"{R.words['E2'].mean():.0f}",
                                "Cannot fabricate; highest coverage per word; no API cost; deterministic", "Unreadable for patients: fragments, abbreviations, no explanation; still mislabelled by the judge",
                                "Clinician-facing digests; a grounding step before an LLM rewrites the selected sentences"]]))]
    b += [H2("8.5 Threats to Validity and Limitations")]
    b += [("bullets", [
        "**Construct validity.** The central limitation is the judge itself. Its labels agree with medical experts at near-chance "
        "levels, so UFR and CR are indices of how an NLI model reacts to a summary, not measurements of hallucination. All "
        "between-condition comparisons in this thesis are comparisons of that index. They remain informative because every "
        "condition is scored on the same documents by the same instrument, but the magnitude of any true effect is unknown.",
        "**Transfer of the validation.** The expert-annotated summaries come from MIMIC-IV hospital courses and discharge "
        "instructions, whereas the comparison uses MTSamples consultations and discharge summaries. The judge's behavior was "
        "similar on both (it flagged about four sentences in five in both corpora), but the calibration is formally an assumption "
        "when transferred.",
        f"**Sample size and power.** Fifty documents give adequate power for large paired effects (the E2 comparisons) and marginal "
        f"power for the small E1 effect on CR (d(z) = {f2(t10c.d_z)}), which is why that effect crosses the significance boundary "
        f"under small changes to the judge. The ten discharge summaries permit no subgroup inference.",
        "**Single generation.** Each summary was generated once at temperature 0.3. Variation between samples of the same model was "
        "not measured, so the per-document differences confound the effect of retrieval with sampling noise.",
        "**External validity of the corpus.** MTSamples documents are sample transcriptions, not patient records; they are shorter "
        "and more formulaic than MIMIC notes, their line breaks are corrupted, and the corpus is public and may be present in the "
        "generator's training data, which could make E0 look more faithful than it would be on unseen notes.",
        "**Single generator and configuration.** Only GPT-4o-mini, one prompt per condition, one retriever, one chunk size and one "
        "retrieval query were tested. Different choices could change the size or even the direction of the RAG effect.",
        "**Heuristic components.** The header filter is a regular expression, the coverage proxy is an embedding-similarity "
        "threshold without human validation, and the error taxonomy is keyword-based with author review rather than an "
        "annotated typology. Each is reported transparently so that it can be replaced.",
        "**Single verification design.** E3 uses one verifier (the generator itself, at temperature 0), one prompt and the judge's "
        "own retriever; a verifier that sees the whole note, or a different model, could keep claims that this design deleted. "
        "Abstentions were excluded from the claim set by a rule written after their mislabeling was observed; the pre-exclusion "
        "numbers are reported in Section 5.5.4 so that the reader can judge the effect of that decision."])]
    b += [H2("8.6 Implications for the Design and Evaluation of Clinical Summarization Systems")]
    b += [("numbers", [
        "**Validate the evaluator before the system.** An automatic hallucination metric should be reported together with its "
        "agreement with clinician annotations on the target domain; without that, a lower score cannot be interpreted. The "
        "ann-pt-summ annotations make such a check inexpensive for any patient-summary metric.",
        "**Prefer paired, within-document comparisons and report effect sizes.** Absolute rates from a noisy judge are not "
        "meaningful, but the same judge applied to two systems on the same notes still ranks large differences correctly; "
        "confidence intervals and robustness to the judge's thresholds should accompany every claim of improvement.",
        "**Always include a verbatim extractive control.** It costs nothing, and it measures the metric's error floor directly; "
        "any reported rate below that floor is noise.",
        "**Measure coverage alongside faithfulness.** Retrieval that shrinks the model's view trades unsupported content for "
        "omissions; a system that reports only hallucination rates hides half of the trade-off.",
        "**Target the failure categories directly.** Extrinsic follow-up and advice sentences are not removed by retrieval; "
        "abstention rules, per-claim verification against the note, or templated advice sections approved by clinicians are more "
        "appropriate remedies.",
        "**Use stronger or domain-adapted judges.** Larger consistency-tuned NLI models, alignment models, atomic-claim "
        "decomposition, clinical NLI fine-tuning, or LLM-based judges calibrated against expert labels are all candidates; the "
        "pipeline released with this thesis allows any of them to be substituted for the cross-encoder without other changes."])]

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
            f"the tools, the controls and the validation protocol needed to evaluate better judges and better summarizers.")]
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
        "the categories that contain retrievable facts and not the extrinsic ones."])]
    b += [H2("9.3 Future Work")]
    b += [("numbers", [
        "**Evaluate the verifier as a judge.** The E3 verifier is itself an LLM-based claim judge; running it on the ann-pt-summ "
        "annotations exactly as the NLI judge was run would show whether an LLM verifier agrees with medical experts better than "
        "the cross-encoder, and a verifier that sees the whole note rather than three retrieved sentences should be compared "
        "with the present design so that deletions of supported claims can be counted.",
        "**Replace the judge.** Evaluate larger consistency-tuned NLI models and alignment models [[cite:honovich2022,zha2023]], "
        "atomic-claim decomposition [[cite:min2023]], a MedNLI-adapted classifier [[cite:romanov2018]], and an LLM-based judge, "
        "all on the ann-pt-summ annotations first, and adopt only a judge whose kappa and AUROC justify it. The validation scripts "
        "released here make this a one-day experiment per candidate.",
        "**Migrate generation to MIMIC-IV-Note.** With a locally hosted open-weight generator or a zero-data-retention agreement, "
        "the same three conditions can be run on real discharge summaries, removing the corpus-contamination and line-break "
        "concerns of MTSamples and allowing the judge's calibration to be applied in-domain.",
        "**Measure sampling variance.** Generate several summaries per note and condition to separate the effect of retrieval "
        "from decoding noise, and increase the document sample once generation is cheap and local.",
        "**Validate coverage and taxonomy with clinicians.** A small clinician study rating omissions and the clinical severity of "
        "unsupported statements would convert the coverage proxy and the keyword taxonomy into validated measures, and would "
        "answer whether the extrinsic advice sentences are acceptable.",
        "**Implement the deferred question-answering task** from the proposal with the same judge and controls, since "
        "document-grounded QA over discharge instructions is the other use case in which patients meet these models."])]
    return b


