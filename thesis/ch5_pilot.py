"""ch5_pilot.py: Chapter 5 (pilot study results on MTSamples) and the abstract."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from results_loader import Results, f3, f2, pct, pct0, fp, fpn, signed, COND_NAME

FIG = "results"
NAME = {"E0": "E0 full-note LLM", "E1": "E1 RAG (excerpts only)", "E1b": "E1b RAG (note + excerpts)", "E3": "E3 RAG + CoVe", "E2": "E2 extractive"}
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

    # ══════════════════════════════════ CHAPTER 5 ══════════════════════════════════
    neg = R.e2err[R.e2err.has_negation.isin([True, False, "True", "False"])].copy()
    neg["has_negation"] = neg.has_negation.astype(str)
    def negcount(c, h, l):
        v = neg[(neg.condition == c) & (neg.has_negation == h) & (neg.label == l)]["count"]
        return int(v.iloc[0]) if len(v) else 0
    def negtot(c, h):
        return sum(negcount(c, h, l) for l in ("Contradicted", "Supported", "Not-Supported"))
    cons, dis = "Consult - History and Phy.", "Discharge Summary"
    cs = R.cal_sum[(R.cal_sum.group == "all") & (R.cal_sum.judge_metric == "judge_UFR")].iloc[0]
    contra_recall = R.cal_lab[R.cal_lab.expert_label == "contradicted_fact"].judge_recall_contradicted.iloc[0]
    b += [H1("Chapter 5 Pilot Study: Results on MTSamples"),
          P(f"The pilot compared the five conditions on {R.n_docs} public notes with GPT-4o-mini and the pipeline's original judge, a "
            f"six-layer MiniLM cross-encoder. Its lasting results are the validation of that judge against medical experts (Section "
            f"5.3), which found it invalid and motivated Chapter 6, and the evaluation artifacts and judge errors it exposed (Section "
            f"5.4). The comparative results (Sections 5.2, 5.5 and 5.6) describe how that judge reacted to the conditions and are "
            f"not interpreted as hallucination rates."),
          H2("5.1 Descriptive Statistics of the Generated Summaries"),
          P(f"[[tab:desc]] summarizes the summaries. Full-note summaries were the longest (mean {R.words['E0'].mean():.0f} words), "
            f"excerpt-only RAG summaries shorter ({R.words['E1'].mean():.0f}) although both prompts asked for 150 to 250 words, E1b "
            f"restored the baseline's length ({R.words['E1b'].mean():.0f}), and verified summaries were the shortest "
            f"({R.words['E3'].mean():.0f}) because the revision deleted claims the verifier could not support and abstained in "
            f"{n_abst_docs} of {R.n_docs} summaries. The judge labeled only about one claim in five as Supported under the baseline and "
            f"the retrieval conditions ({pct0(share[('E0','Supported')])} for E0) and {pct0(share[('E3','Supported')])} under "
            f"verification; of the verbatim E2 sentences it labeled {e2_ns} Not-Supported and {e2c_total} Contradicted, judge errors by "
            f"construction (Section 5.4)."),
          ("table", dict(label="desc", caption=f"Length, number of claims, share of claims labeled Supported, and per-document Unsupported Fact Rate and Contradiction Rate per condition (n = {R.n_docs} documents; header lines excluded).",
                         columns=["Condition", "Words per summary, mean (SD)", "Claims per summary", "Total claims", "Supported share", "UFR, mean (SD)", "CR, mean (SD)"],
                         widths=[1.35, 1.2, 0.8, 0.7, 0.8, 0.9, 0.9], font=9.5,
                         rows=[[NAME[c], f"{R.words[c].mean():.0f} ({R.words[c].std():.0f})", f"{R.claims_per[c]:.1f}", f"{tot[c]:,}",
                                pct0(share[(c, 'Supported')]), f"{f3(R.mean(c, 'UFR'))} ({f3(R.sd(c, 'UFR'))})", f"{f3(R.mean(c, 'CR'))} ({f3(R.sd(c, 'CR'))})"] for c in C]))]
    b += [H2("5.2 Comparison of the Five Conditions"),
          P(f"[[tab:tests]] reports the paired comparisons. UFR did not differ between the full-note and the excerpt-only RAG condition "
            f"({f3(t10u.mean_a)} against {f3(t10u.mean_b)}). RAG lowered CR by {rel(t10c)} ({fp(t10c.p)}), but the difference does not "
            f"survive the Holm adjustment over the twenty pilot tests (adjusted {fp(t10c.p_holm)}) and is suggestive only, as is E1b's "
            f"lower UFR (adjusted {fp(t1b0u.p_holm)}). Verification produced the largest change: E3 lowered UFR from "
            f"{f3(t30u.mean_a)} to {f3(t30u.mean_b)} (adjusted {fp(t30u.p_holm)}, d(z) = {f2(t30u.d_z)}) with CR unchanged, mainly by "
            f"saying less: the verifier judged {pct0(n_verif_unsup / max(1, n_verif))} of the {n_verif} draft claims unsupported by its "
            f"retrieved sentences, and the revision deleted or replaced them (Section 5.5). The extractive condition had the lowest "
            f"rates, as a reference for the judge's errors on verbatim text should. The pattern was the same in consultation notes and "
            f"discharge summaries, and note length did not moderate the effects (Spearman's rho with the paired E1 minus E0 difference "
            f"{rho_u:+.2f} for UFR and {rho_c:+.2f} for CR)."),
          ("table", dict(label="tests", caption="Paired comparisons between conditions. Delta is the mean of per-document differences (second condition minus first); negative values favor the second condition. CI is the 95 percent percentile bootstrap interval (10,000 resamples). p is the two-sided Wilcoxon signed-rank p-value and p(Holm) its Holm adjustment over the twenty tests of the table. d(z) is the standardized mean difference and r the matched-pairs rank-biserial correlation (positive = improvement).",
                         columns=["Comparison", "Metric", "Delta", "95% CI", "Median delta", "p", "p (Holm)", "d(z)", "r", "Improved / same / worse"],
                         widths=[0.85, 0.55, 0.65, 1.0, 0.6, 0.6, 0.6, 0.5, 0.5, 0.95], font=8.5,
                         rows=[[t.comparison, t.metric, signed(t.delta_mean), f"[{t.ci95_low:+.3f}, {t.ci95_high:+.3f}]", signed(t.delta_median), fpn(t.p), fpn(t.p_holm),
                                f2(t.d_z), f2(t.rank_biserial), f"{t.pct_improved:.0f}% / {t.pct_unchanged:.0f}% / {t.pct_worse:.0f}%"]
                               for _, t in R.tests.iterrows()]))]
    b += [H2("5.3 Validation of the Judge Against Medical Experts"),
          P(f"The judge was applied unchanged to the {int(cal_all.n_summaries)} expert-annotated summaries, {int(cal_all.n_sentences):,} "
            f"sentences of which {pct(cal_all.expert_flag_rate)} overlap an expert-annotated unsupported fact; the judge flagged "
            f"{pct(cal_all.judge_UFR)} ([[tab:calib]], [[fig:calib]]). Agreement was poor: because the judge flags four sentences in five, "
            f"its recall is high ({pct(cal_all.any_recall)}) but its precision is {pct(cal_all.any_precision)} and Cohen's kappa "
            f"{f2(cal_all.any_kappa)}, essentially chance, and the continuous scores separate the two groups only weakly (AUROC "
            f"{f2(cal_all.auroc_1_minus_p_entail)}). For the three GPT-4 configurations, where experts found unsupported content in 5 to 9 "
            f"percent of sentences, the judge still flagged 86 to 87 percent, and ranked by group it orders the systems almost in reverse "
            f"of the experts, who found the most unsupported content in doctor-written instructions and the least in GPT-4 summaries "
            f"[[cite:hegselmann2024]]. At the level of individual summaries the judge's UFR is only weakly correlated with the experts' "
            f"flagged share (Spearman rho = {f2(cs.spearman_rho)}), and its contradiction label recovered {pct0(contra_recall)} of the "
            f"sentences experts marked as contradicting the source while firing on 28 to 32 percent of all sentences."),
          ("table", dict(label="calib", caption="Agreement between the judge and the medical-expert annotations at the sentence level. The any detector flags Not-Supported or Contradicted sentences; AUROC uses one minus the entailment probability as the score. Kappa is Cohen's kappa.",
                         columns=["Group", "Summaries", "Sentences", "Expert-flagged", "Judge-flagged (UFR)", "Precision", "Recall", "Specificity", "F1", "Kappa", "AUROC"],
                         widths=[1.2, 0.6, 0.6, 0.65, 0.65, 0.55, 0.5, 0.6, 0.45, 0.45, 0.5], font=8.5,
                         rows=[[r.group.replace("_", " "), str(int(r.n_summaries)), f"{int(r.n_sentences):,}", pct(r.expert_flag_rate), pct(r.judge_UFR), f2(r.any_precision), f2(r.any_recall),
                                f2(r.any_specificity), f2(r.any_f1), f2(r.any_kappa), f2(r.auroc_1_minus_p_entail)] for _, r in R.cal.iterrows()])),
          ("figure", dict(label="calib", path=f"{FIG}/fig_calibration.png", width=6.5, caption="Left: share of sentences flagged by the experts and by the judge, with the precision and recall of the judge's flags, per group. Right: distribution of the judge's entailment probability for sentences that the experts did and did not flag."))]
    if var_all is not None:
        best_auc = var_all.auroc_1_minus_pe.idxmax()
        b += [P(f"[[tab:variants]] tests whether the disagreement stems from how evidence is aggregated. It does not: concatenating the "
                f"three retrieved sentences into one premise lowered the flag rate to {pct(var_all.loc['concat3','judge_flag_rate'])} and "
                f"raised the AUROC only to {f2(var_all.loc['concat3','auroc_1_minus_pe'])}, scoring against every sentence of the course "
                f"(the SummaC zero-shot design) made matters worse (kappa {f2(var_all.loc['maxall','kappa'])}), and the whole course as "
                f"one premise flagged {pct(var_all.loc['concat_ctx','judge_flag_rate'])} of sentences, because a six-layer model truncated "
                f"to 512 tokens cannot find a sentence's support in a long premise. The best attainable kappa over every threshold and "
                f"variant was {f2(var_all.best_kappa.max())}; the limiting factor is the NLI model's judgment of individual "
                f"patient-facing sentences, not the retrieval or aggregation rule."),
              ("table", dict(label="variants", caption="Evidence-aggregation variants of the judge evaluated on all 1,781 expert-annotated sentences with the same cross-encoder. Flag rate, precision, recall and kappa are at τ = 0.5; best kappa is the maximum over thresholds from 0.05 to 0.95; rho is Spearman's correlation between judge and expert rates per summary.",
                             columns=["Variant", "Judge flag rate", "AUROC", "Precision", "Recall", "Kappa", "Best kappa (τ)", "Summary rho"],
                             widths=[2.2, 0.75, 0.6, 0.7, 0.6, 0.6, 0.85, 0.7], font=9.5,
                             rows=[[VNAME[k], pct(r.judge_flag_rate), f2(r.auroc_1_minus_pe), f2(r.precision), f2(r.recall), f2(r.kappa), f"{f2(r.best_kappa)} ({r.best_kappa_tau:.2f})", f2(r.summary_spearman)]
                                   for k, r in var_all.loc[[k for k in VNAME if k in var_all.index]].iterrows()]))]
    b += [H2("5.4 Robustness of the Comparison and the Judge's Errors on Verbatim Text"),
          P(f"Four checks varied the judge's design. Raising the decision threshold τ makes the judge more conservative; the E1 versus E0 "
            f"difference in UFR is not significant at any threshold, its CR difference holds only at τ = 0.5 and disappears from τ = 0.7, "
            f"whereas the UFR advantage of E3 held at {int((tau['p_E3_vs_E0_UFR'] < 0.05).sum())} of the {len(tau)} thresholds. Re-running "
            f"retrieval with k = 3 reproduced every stored label, so the pipeline is deterministic; with five evidence sentences "
            f"contradictions rise for every condition, including the verbatim E2 sentences, because each extra sentence gives the "
            f"maximum-over-evidence rule another chance to find a sentence the model calls a contradiction. Repairing the comma-encoded "
            f"line breaks on the evidence side left E2 with {e2clean_contra} Contradicted claims out of {e2clean_n} instead of {e2c_total} "
            f"out of {e2_n}, so roughly a third of the judge's false contradictions on verbatim text came from malformed fragments and "
            f"the rest are model errors."),
          P(f"Those errors have a pattern. For the {e2c_total} verbatim sentences labeled Contradicted, {e2c_both} had both an entailment "
            f"and a contradiction probability above 0.9 against different evidence sentences, and the symmetric rule broke the tie; "
            f"{e2c_neg} of them contain a negation cue ({negcount('E2','True','Contradicted')} of {negtot('E2','True')} E2 sentences with a cue "
            f"were called Contradicted against {negcount('E2','False','Contradicted')} of {negtot('E2','False')} without), because a negated "
            f"list such as \"Denies nausea, vomiting, or diarrhea\", compared with a neighboring sentence that mentions one of those "
            f"symptoms, reads to the model like a contradiction. A related artifact appeared in the verified condition: the judge labeled "
            f"{n_abst_contra} of the {n_abst} \"Not stated in the note.\" lines E3 wrote as Contradicted before they were excluded from the "
            f"claim set; counted as claims, they would have raised E3's contradiction rate from {f3(R.mean('E3','CR'))} to about 0.27 and "
            f"reversed the conclusion about verification, which is why Section 3.5 treats abstentions as a separate category.")]
    b += [H2("5.5 The Coverage Cost of Retrieval"),
          P(f"[[tab:cov]] reports the coverage proxy, the share of source sentences that some claim resembles closely. Full-note "
            f"summaries touched {pct(covv('E0'))} of source sentences at the 0.6 threshold, excerpt-only RAG summaries {pct(covv('E1'))} "
            f"({fp(covp('E0','E1'))}), RAG with the full note {pct(covv('E1b'))}, verified summaries {pct(covv('E3'))} and extractive "
            f"summaries {pct(covv('E2'))}. Showing the model only three retrieved chunks reduced the breadth of the summary, as the "
            f"proposal anticipated, and restoring the note (E1b) restored it; E3's coverage did not fall below the baseline despite its "
            f"brevity, because the verifier's corrections draw on the full note. Among the LLM conditions, only excerpt-only retrieval "
            f"paid for its faithfulness gain with a clear loss of coverage."),
          ("table", dict(label="cov", caption="Coverage proxy per condition: mean share of source sentences whose best cosine similarity to any claim reaches the threshold, and mean best similarity.",
                         columns=["Condition", "Coverage at 0.5", "Coverage at 0.6", "Coverage at 0.7", "Mean best similarity", "Claims per summary"],
                         widths=[1.7, 0.95, 0.95, 0.95, 1.05, 0.9], font=10,
                         rows=[[NAME[c], pct(r.coverage_at_0_5), pct(r.coverage_at_0_6), pct(r.coverage_at_0_7), f3(r.mean_best_similarity), f"{r.n_claims:.1f}"] for c, r in cov.iterrows()],
                         note="Paired tests on coverage at 0.6: " + "; ".join(f"{col[8:].replace('_vs_', ' vs ')} {fp(cov.loc['E0', col])}" for col in cov.columns if col.startswith("p_cov06_")) + "."))]
    b += [H2("5.6 What Remains Unsupported"),
          P(f"[[tab:tax]] categorizes the claims the judge did not label Supported. Follow-up and scheduling statements are the largest "
            f"category under every LLM condition and the only one that grows under excerpt-only RAG "
            f"({R.tax_total('E0','Follow-up or scheduling')} to {R.tax_total('E1','Follow-up or scheduling')}); they are almost never "
            f"contradicted, because the note says nothing about them. RAG reduced unsupported claims about generic advice, procedures, "
            f"findings and medications, the categories with specific, retrievable facts: excerpts constrain what the model says about "
            f"the note but not what it adds from its knowledge of how patient instructions usually end (Section 2.6). Only verification "
            f"cut the follow-up category substantially ({R.tax_total('E0','Follow-up or scheduling')} to "
            f"{R.tax_total('E3','Follow-up or scheduling')}), because a verifier removes the sentences no passage supports. Paired claims "
            f"show the fragility of the metric: in document 16 the note states \"COMPLICATIONS: None\", the E0 phrasing \"there were no "
            f"complications\" was called Contradicted and the E1 phrasing \"there were no complications from the procedures\" Supported, "
            f"and claims that appear almost verbatim under both conditions and are unsupported under both are faithful paraphrases that "
            f"the NLI model cannot match to any single sentence, the failure mode Section 5.3 quantified."),
          ("table", dict(label="tax", caption="Keyword-assisted categories of claims that the judge did not label Supported, for the LLM conditions. Contradicted counts are the subset labeled Contradicted.",
                         columns=["Category"] + [f"{c} {k}" for c in GEN for k in ("unsupp.", "contra.")], widths=[1.9] + [0.575] * (2 * len(GEN)), font=9, rows=tax_rows))]
    b += [H2("5.7 Summary of the Pilot"),
          ("bullets", [
              f"**The judge is not valid for this task.** Against medical-expert annotations it reaches kappa {f2(cal_all.any_kappa)} "
              f"(AUROC {f2(cal_all.auroc_1_minus_p_entail)}), flags four sentences in five, and ranks systems in nearly the reverse order "
              f"of the experts; no aggregation rule repairs this, so its pilot rates are not hallucination rates.",
              f"**Under that judge,** excerpt-only RAG lowered CR by {rel(t10c)} (suggestive only), left UFR unchanged and reduced coverage; "
              f"E1b lowered UFR modestly without the coverage loss; verification lowered UFR the most (adjusted {fp(t30u.p_holm)}) by "
              f"deleting claims and abstaining. The main study re-tests these effects with a validated judge.",
              f"**Judge errors on verbatim text** come from negated sentences and malformed fragments, and the judge labeled "
              f"{n_abst_contra} of E3's {n_abst} abstentions as contradictions.",
              f"**Two evaluation artifacts** had inflated earlier results, header lines counted as claims and abstentions counted as "
              f"claims; both are corrected in every analysis of this thesis."])]

    return b


def abstract(R: Results) -> list:
    import numpy as np
    from mimic_results import Mimic, _csv
    from ch_mimic import Data
    M = Mimic(); jr = M.judge_row(); D = Data()
    cal = R.cal_row("all")
    main = M.ok and D.ok and jr is not None
    txt = (f"Large language models (LLMs) can turn clinical notes into fluent patient-facing summaries, but they also produce "
           f"statements that the note does not support. This thesis builds a reproducible, reference-free pipeline that splits a "
           f"summary into claims, checks each claim against the source with a natural language inference (NLI) judge, and reports a "
           f"judge-estimated Unsupported Fact Rate (UFR) and Contradiction Rate (CR). Five conditions are compared: full-note "
           f"summarization without retrieval, retrieval-augmented generation (RAG) with excerpts only or with the whole document, RAG "
           f"with Chain-of-Verification, and an extractive summarizer. In a pilot on {R.n_docs} public MTSamples notes with GPT-4o-mini, "
           f"a small NLI judge was validated against {int(cal.n_summaries)} patient summaries annotated by medical experts: it flagged "
           f"{pct0(cal.judge_UFR)} of sentences where the experts flagged {pct0(cal.expert_flag_rate)} (Cohen's kappa {f2(cal.any_kappa)}), "
           f"so it is not a valid detector for this task. ")
    if main:
        sb = _csv("judge_selection_bias.csv")
        kcv = sb[(sb.analysis == "nested_cv_selection_procedure") & (sb.metric == "kappa")] if sb is not None else None
        tu = [D.T("UFR", f"{c} vs E0") for c in ("E1", "E1b", "E3")]
        ph = max((D.holm_p(t) if not np.isnan(D.holm_p(t)) else t.p) for t in tu if t is not None)
        lo = min(M.mean(c, "UFR") for c in ("E1", "E1b", "E3")); hi = max(M.mean(c, "UFR") for c in ("E1", "E1b", "E3"))
        txt += (f"A selection study of seven candidate judges chose {M.judge_name()} with whole-course evidence (kappa {f2(jr.kappa)}"
                + (f"; {f2(float(kcv.estimate.iloc[0]))} under nested cross-validation" if kcv is not None and len(kcv) else "")
                + f"). With that judge, the comparison was repeated on {M.n_docs} MIMIC-IV hospital courses with an open-weight model "
                f"running locally, and omission was measured against the clinicians' discharge instructions. Every grounded condition "
                f"lowered the judge-estimated UFR from {f3(M.mean('E0', 'UFR'))} to between {f3(lo)} and {f3(hi)} "
                f"({'Holm-adjusted p < 0.001' if ph < 0.001 else f'Holm-adjusted p ≤ {ph:.3f}'}), partly by producing shorter summaries, "
                f"without changing CR. RAG with the whole document achieved coverage similar to the full-note baseline, with no "
                f"significant difference detected, while excerpt-only RAG and verification omitted more of the clinicians' content. The "
                f"reductions held under a second judge, but whether verification adds to retrieval depended on the judge. ")
        repl_g, norepl_g = [], []
        for tag, nm in (("llama31", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B")):
            gp = _csv(f"mimic_pairwise_tests_{tag}.csv")
            if gp is None or not len(gp):
                continue
            g = gp[(gp.metric == "UFR") & gp.comparison.isin([f"{c} vs E0" for c in ("E1", "E1b", "E3")])]
            ok = bool(len(g)) and bool(((g.p_holm_primary.fillna(g.p) < 0.05) & (g.delta_mean < 0)).any())
            (repl_g if ok else norepl_g).append(nm)
        if repl_g and norepl_g:
            txt += f"They replicated under {' and '.join(repl_g)} but not under {' and '.join(norepl_g)}. "
        elif repl_g:
            txt += f"They replicated under {' and '.join(repl_g)}. "
        elif norepl_g:
            txt += f"They did not replicate under {' and '.join(norepl_g)}. "
        aud = _csv("audit_summary.csv")
        if aud is not None and "diff" in aud.columns:
            ad = aud[aud["diff"].notna()].set_index("condition")
            names_a = {"E1 minus E0": "excerpt-only RAG", "E1b minus E0": "whole-document RAG", "E3 minus E0": "verification"}
            conf = [k for k in names_a if k in ad.index and ad.loc[k, "diff_hi"] < 0]
            if conf:
                txt += (f"A two-pass blind manual audit confirmed the reduction for {' and '.join(names_a[k] for k in conf)} "
                        f"(interval below zero) and showed that the judge understates absolute unsupported shares. ")
            elif len(ad):
                txt += ("A two-pass blind manual audit could not confirm the condition differences at its sample size and showed "
                        "that the judge understates absolute unsupported shares. ")
        txt += (f"The clinicians' own instructions scored {f3(M.mean('REF', 'UFR'))} under the same judge because they state what the "
                f"course does not, so the metric measures faithfulness to the source, not summary quality. ")
    txt += ("The thesis concludes that hallucination judges must be validated against expert labels and checked on the outputs they "
            "score, and that grounding with the whole document in view reduced unsupported content without the omission of "
            "excerpt-only retrieval.")
    return [txt]
