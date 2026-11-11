"""ch5_pilot.py — Chapter 5 (pilot study results on MTSamples) and the abstract."""
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
    cs_cr = R.cal_sum[(R.cal_sum.group == "all") & (R.cal_sum.judge_metric == "judge_CR")].iloc[0]
    contra_recall = R.cal_lab[R.cal_lab.expert_label == "contradicted_fact"].judge_recall_contradicted.iloc[0]
    b += [H1("Chapter 5 Pilot Study: Results on MTSamples"),
          P(f"The pilot compared the five conditions on {R.n_docs} public notes with GPT-4o-mini and the pipeline's original judge, a "
            f"six-layer MiniLM cross-encoder. Its lasting results are the validation of that judge against medical experts "
            f"(Section 5.3), which found it invalid for the task and motivated the selection study of Chapter 6, and the evaluation "
            f"artifacts and judge errors it exposed (Section 5.4), which shaped the main study. The comparative results (Sections "
            f"5.2, 5.5 and 5.6) are reported for completeness: they describe how that judge reacted to the conditions and are not "
            f"interpreted as hallucination rates."),
          H2("5.1 Descriptive Statistics of the Generated Summaries"),
          P(f"[[tab:desc]] summarizes the summaries. Full-note summaries were the longest (mean {R.words['E0'].mean():.0f} words); "
            f"excerpt-only RAG summaries were shorter (mean {R.words['E1'].mean():.0f} words) although both prompts requested 150 to "
            f"250 words, and giving the model the full note as well (E1b) restored the length of the baseline (mean "
            f"{R.words['E1b'].mean():.0f} words). Verified summaries (E3) were shorter still (mean {R.words['E3'].mean():.0f} words) "
            f"because the revision step deleted claims the verifier could not support and wrote an abstention where a section had "
            f"no supported content: {n_abst} abstention lines in {n_abst_docs} of the {R.n_docs} summaries, and fewer claims than "
            f"the E1 draft in {fewer_e3} of {R.n_docs} documents. In total {R.n_claims_total:,} claims were labeled. The judge "
            f"labeled only about one claim in five as Supported under the baseline and the retrieval conditions "
            f"({pct0(share[('E0','Supported')])} for E0, {pct0(share[('E1','Supported')])} for E1, {pct0(share[('E1b','Supported')])} "
            f"for E1b) and {pct0(share[('E3','Supported')])} under verification; of the verbatim E2 sentences it labeled "
            f"{e2_ns} Not-Supported and {e2c_total} Contradicted, which are judge errors by construction (Section 5.4)."),
          ("table", dict(label="desc", caption=f"Length, number of claims, share of claims labeled Supported, and per-document Unsupported Fact Rate and Contradiction Rate per condition (n = {R.n_docs} documents; header lines excluded).",
                         columns=["Condition", "Words per summary, mean (SD)", "Claims per summary", "Total claims", "Supported share", "UFR, mean (SD)", "CR, mean (SD)"],
                         widths=[1.35, 1.2, 0.8, 0.7, 0.8, 0.9, 0.9], font=9.5,
                         rows=[[NAME[c], f"{R.words[c].mean():.0f} ({R.words[c].std():.0f})", f"{R.claims_per[c]:.1f}", f"{tot[c]:,}",
                                pct0(share[(c, 'Supported')]), f"{f3(R.mean(c, 'UFR'))} ({f3(R.sd(c, 'UFR'))})", f"{f3(R.mean(c, 'CR'))} ({f3(R.sd(c, 'CR'))})"] for c in C]))]
    b += [H2("5.2 Comparison of the Five Conditions"),
          P(f"[[tab:tests]] reports the paired comparisons. The Unsupported Fact Rate did not differ between the full-note and the "
            f"excerpt-only RAG condition ({f3(t10u.mean_a)} against {f3(t10u.mean_b)}, {fp(t10u.p)}). RAG lowered the Contradiction "
            f"Rate ({f3(t10c.mean_a)} against {f3(t10c.mean_b)}, {rel(t10c)} relative, {fp(t10c.p)}, d(z) = {f2(t10c.d_z)}), but the "
            f"difference does not survive the Holm adjustment over the twenty pilot tests (adjusted {fp(t10c.p_holm)}) and is "
            f"suggestive only. Showing the model the full note together with the excerpts (E1b) lowered UFR relative to E0 "
            f"({signed(t1b0u.delta_mean)}, {fp(t1b0u.p)}, adjusted {fp(t1b0u.p_holm)}), again suggestive only, without a change in CR. "
            f"Verification produced the largest change of any LLM condition: E3 lowered UFR from {f3(t30u.mean_a)} to "
            f"{f3(t30u.mean_b)} ({fp(t30u.p)}, adjusted {fp(t30u.p_holm)}, d(z) = {f2(t30u.d_z)}) and improved on its own E1 drafts "
            f"(adjusted {fp(t31u.p_holm)}), with CR unchanged. The verification log shows the mechanism: the verifier examined "
            f"{n_verif} draft claims, judged {n_verif_unsup} ({pct0(n_verif_unsup / max(1, n_verif))}) unsupported by the retrieved "
            f"sentences, and the revision deleted or replaced them and abstained for {n_abst} sections, so E3 reduced unsupported "
            f"content mainly by saying less (Section 5.5). The extractive condition had the lowest rates (UFR {f3(t20u.mean_b)}, "
            f"CR {f3(t20c.mean_b)}), as a reference for the judge's errors on verbatim text should. The pattern was the same in "
            f"the {int(spec_n[cons])} consultation notes and the {int(spec_n[dis])} discharge summaries, and source length did not "
            f"moderate the effects (Spearman's rho between note length and the paired E1 minus E0 difference {rho_u:+.2f} for UFR, "
            f"{fp(p_u)}, and {rho_c:+.2f} for CR, {fp(p_c)})."),
          ("table", dict(label="tests", caption="Paired comparisons between conditions. Delta is the mean of per-document differences (second condition minus first); negative values favor the second condition. CI is the 95 percent percentile bootstrap interval (10,000 resamples). p is the two-sided Wilcoxon signed-rank p-value and p(Holm) its Holm adjustment over the twenty tests of the table. d(z) is the standardized mean difference and r the matched-pairs rank-biserial correlation (positive = improvement).",
                         columns=["Comparison", "Metric", "Delta", "95% CI", "Median delta", "p", "p (Holm)", "d(z)", "r", "Improved / same / worse"],
                         widths=[0.85, 0.55, 0.65, 1.0, 0.6, 0.6, 0.6, 0.5, 0.5, 0.95], font=8.5,
                         rows=[[t.comparison, t.metric, signed(t.delta_mean), f"[{t.ci95_low:+.3f}, {t.ci95_high:+.3f}]", signed(t.delta_median), fpn(t.p), fpn(t.p_holm),
                                f2(t.d_z), f2(t.rank_biserial), f"{t.pct_improved:.0f}% / {t.pct_unchanged:.0f}% / {t.pct_worse:.0f}%"]
                               for _, t in R.tests.iterrows()]))]
    b += [H2("5.3 Validation of the Judge Against Medical Experts"),
          P(f"The judge was applied unchanged to the {int(cal_all.n_summaries)} expert-annotated summaries, which contain "
            f"{int(cal_all.n_sentences):,} sentences after segmentation, {pct(cal_all.expert_flag_rate)} of which overlap at least one "
            f"expert-annotated unsupported fact ({pct(cal_gen.expert_flag_rate)} of sentences in LLM-generated summaries and "
            f"{pct(cal_doc.expert_flag_rate)} in doctor-written ones). The judge flagged {pct(cal_all.judge_UFR)} of the same "
            f"sentences as Not-Supported or Contradicted ([[tab:calib]], [[fig:calib]]). Agreement with the experts was poor. "
            f"Because the judge flags four sentences in five, its recall for expert-flagged sentences is high "
            f"({pct(cal_all.any_recall)}) but its precision is {pct(cal_all.any_precision)}, its specificity "
            f"{pct(cal_all.any_specificity)}, and Cohen's kappa is {f2(cal_all.any_kappa)}, essentially chance agreement. The "
            f"continuous scores separate the two groups of sentences only weakly: the area under the ROC curve is "
            f"{f2(cal_all.auroc_1_minus_p_entail)} for one minus the entailment probability and {f2(cal_all.auroc_p_contra)} for "
            f"the contradiction probability, where 0.5 is chance, and the contradiction detector's precision is "
            f"{pct(cal_all.contra_precision)} (kappa {f2(cal_all.contra_kappa)}). For the three GPT-4 configurations, in which "
            f"experts found unsupported content in only 5 to 9 percent of sentences, the judge still flagged 86 to 87 percent."),
          ("table", dict(label="calib", caption="Agreement between the judge and the medical-expert annotations at the sentence level. The any detector flags Not-Supported or Contradicted sentences; AUROC uses one minus the entailment probability as the score. Kappa is Cohen's kappa.",
                         columns=["Group", "Summaries", "Sentences", "Expert-flagged", "Judge-flagged (UFR)", "Precision", "Recall", "Specificity", "F1", "Kappa", "AUROC"],
                         widths=[1.2, 0.6, 0.6, 0.65, 0.65, 0.55, 0.5, 0.6, 0.45, 0.45, 0.5], font=8.5,
                         rows=[[r.group.replace("_", " "), str(int(r.n_summaries)), f"{int(r.n_sentences):,}", pct(r.expert_flag_rate), pct(r.judge_UFR), f2(r.any_precision), f2(r.any_recall),
                                f2(r.any_specificity), f2(r.any_f1), f2(r.any_kappa), f2(r.auroc_1_minus_p_entail)] for _, r in R.cal.iterrows()])),
          ("figure", dict(label="calib", path=f"{FIG}/fig_calibration.png", width=6.5, caption="Left: share of sentences flagged by the experts and by the judge, with the precision and recall of the judge's flags, per group. Right: distribution of the judge's entailment probability for sentences that the experts did and did not flag.")),
          P(f"Ranked by group, the experts found the most unsupported content in doctor-written instructions and in the original "
            f"Llama-2 summaries and the least in the GPT-4 summaries, reproducing the ordering reported by Hegselmann et al. "
            f"[[cite:hegselmann2024]]; the judge's UFR orders the groups almost in reverse, rating the three GPT-4 configurations "
            f"as the worst. At the level of individual summaries the judge's UFR is weakly but significantly correlated with the "
            f"share of expert-flagged sentences (Spearman rho = {f2(cs.spearman_rho)}, {fp(cs.spearman_p)}), whereas its CR is not "
            f"(rho = {f2(cs_cr.spearman_rho)}). Recall was uniformly high across expert label types simply because almost everything "
            f"is flagged; the contradiction label recovered {pct0(contra_recall)} of the sentences that experts marked as "
            f"contradicting the source, but also fired on 28 to 32 percent of all sentences in every group.")]
    if var_all is not None:
        best_auc = var_all.auroc_1_minus_pe.idxmax()
        b += [P(f"[[tab:variants]] tests whether the disagreement stems from how evidence is aggregated. It does not. Concatenating "
                f"the three retrieved sentences into a single premise reduced the flag rate from {pct(var_all.loc['max3','judge_flag_rate'])} "
                f"to {pct(var_all.loc['concat3','judge_flag_rate'])} and raised the area under the curve from "
                f"{f2(var_all.loc['max3','auroc_1_minus_pe'])} to {f2(var_all.loc['concat3','auroc_1_minus_pe'])}; scoring the claim "
                f"against every sentence of the hospital course, the SummaC zero-shot design, made matters worse (flag rate "
                f"{pct(var_all.loc['maxall','judge_flag_rate'])}, kappa {f2(var_all.loc['maxall','kappa'])}); and presenting the whole "
                f"hospital course as one premise flagged {pct(var_all.loc['concat_ctx','judge_flag_rate'])} of sentences, because a "
                f"six-layer model truncated to 512 tokens cannot find a sentence's support in a long premise. The best attainable "
                f"kappa over every threshold and variant was {f2(var_all.best_kappa.max())} and the best area under the curve "
                f"{f2(var_all.auroc_1_minus_pe.max())} (variant {VNAME[best_auc].lower()}). The limiting factor is the NLI model's "
                f"judgment of individual patient-facing sentences, not the retrieval or aggregation rule."),
              ("table", dict(label="variants", caption="Evidence-aggregation variants of the judge evaluated on all 1,781 expert-annotated sentences with the same cross-encoder. Flag rate, precision, recall and kappa are at τ = 0.5; best kappa is the maximum over thresholds from 0.05 to 0.95; rho is Spearman's correlation between judge and expert rates per summary.",
                             columns=["Variant", "Judge flag rate", "AUROC", "Precision", "Recall", "Kappa", "Best kappa (τ)", "Summary rho"],
                             widths=[2.2, 0.75, 0.6, 0.7, 0.6, 0.6, 0.85, 0.7], font=9.5,
                             rows=[[VNAME[k], pct(r.judge_flag_rate), f2(r.auroc_1_minus_pe), f2(r.precision), f2(r.recall), f2(r.kappa), f"{f2(r.best_kappa)} ({r.best_kappa_tau:.2f})", f2(r.summary_spearman)]
                                   for k, r in var_all.loc[[k for k in VNAME if k in var_all.index]].iterrows()]))]
    b += [H2("5.4 Robustness of the Comparison and the Judge's Errors on Verbatim Text"),
          P(f"Four checks varied the judge's design. Raising the decision threshold τ makes the judge more conservative in both "
            f"directions; the E1 versus E0 difference in UFR is not significant at any threshold, its CR difference reaches p < 0.05 "
            f"only at τ = 0.5 ({fp(tau.loc[0.5,'p_E1_vs_E0_CR'])}, unadjusted) and disappears from τ = 0.7 "
            f"({fp(tau.loc[0.7,'p_E1_vs_E0_CR'])}), exactly the probability region in which the validation found the model least "
            f"trustworthy, whereas the UFR advantage of E3 over E0 held at {int((tau['p_E3_vs_E0_UFR'] < 0.05).sum())} of the "
            f"{len(tau)} thresholds, the most robust effect in the pilot. Re-running retrieval and classification with k = 3 "
            f"reproduced every stored label ({pct0(topk.loc[3,'label_agreement_with_stored'])} agreement), so the pipeline is "
            f"deterministic; with one evidence sentence contradictions almost vanish and with five they rise for every condition, "
            f"including the verbatim E2 sentences (CR {f3(topk.loc[5,'E2_CR_mean'])}), because each additional sentence gives the "
            f"maximum-over-evidence rule another chance to find a sentence the model calls a contradiction. Repairing the "
            f"comma-encoded line breaks of MTSamples on the evidence side changed few labels for the LLM conditions, but E2 "
            f"regenerated from repaired sentences produced no Not-Supported claims and {e2clean_contra} Contradicted claims out of "
            f"{e2clean_n} instead of {e2c_total} out of {e2_n}, so roughly a third of the judge's false contradictions on verbatim "
            f"text were caused by malformed sentence fragments; the remainder are model errors."),
          P(f"Those errors have a pattern. For the {e2c_total} verbatim sentences labeled Contradicted, the mean contradiction "
            f"probability was {f2(R.e2err_value('Contradicted_mean_p_contra'))} but the mean entailment probability was also "
            f"{f2(R.e2err_value('Contradicted_mean_p_entail'))}, and {e2c_both} of the {e2c_total} had both probabilities above 0.9: "
            f"the model assigns near-certainty to both labels against different evidence sentences, and the symmetric decision "
            f"rule breaks the tie by whichever maximum is larger. {e2c_neg} of the {e2c_total} contain a negation cue "
            f"({negcount('E2','True','Contradicted')} of {negtot('E2','True')} E2 sentences with a cue were called Contradicted, "
            f"against {negcount('E2','False','Contradicted')} of {negtot('E2','False')} without): a negated list of symptoms such as "
            f"\"Denies nausea, vomiting, or diarrhea\", compared with a neighboring sentence that mentions one of those symptoms in "
            f"a different context, reads to the model like a contradiction, and negation cues raise the contradiction rate of the "
            f"LLM conditions as well ({negcount('E0','True','Contradicted')} of {negtot('E0','True')} E0 claims with a cue against "
            f"{negcount('E0','False','Contradicted')} of {negtot('E0','False')} without). A related artifact appeared in the "
            f"verified condition: of the {n_abst} abstention lines that E3 wrote (\"Not stated in the note.\"), the judge labeled "
            f"{n_abst_contra} as Contradicted before they were excluded from the claim set. An abstention is a negated statement "
            f"about the note itself, and paired with any retrieved sentence that does state something the cross-encoder reads it "
            f"as a contradiction; had abstentions been counted as claims, the E3 contradiction rate would have risen from "
            f"{f3(R.mean('E3','CR'))} to about 0.27 and reversed the conclusion about verification, which is why Section 3.5 treats "
            f"abstentions as a separate category.")]
    b += [H2("5.5 The Coverage Cost of Retrieval"),
          P(f"[[tab:cov]] reports the coverage proxy, the share of source sentences whose best cosine similarity to any claim reaches "
            f"a threshold. Full-note summaries touched {pct(covv('E0'))} of source sentences at the 0.6 threshold, excerpt-only RAG "
            f"summaries {pct(covv('E1'))} ({fp(covp('E0','E1'))} for the paired difference), RAG summaries with the full note "
            f"{pct(covv('E1b'))} ({fp(covp('E0','E1b'))} against E0), verified summaries {pct(covv('E3'))} ({fp(covp('E0','E3'))} "
            f"against E0) and five-sentence extractive summaries {pct(covv('E2'))}. Showing the model only three retrieved chunks "
            f"reduced the breadth of the summary, as the proposal anticipated, and restoring the note (E1b) restored it. "
            f"Verification is the notable case: although E3 summaries are the shortest LLM summaries, their coverage at the primary "
            f"threshold did not fall below the baseline, because the verifier's corrections draw on sentences retrieved from the "
            f"full note rather than from the three excerpts; at the looser 0.5 threshold E3 does cover less than E0 "
            f"({pct(cov.loc['E3','coverage_at_0_5'])} against {pct(cov.loc['E0','coverage_at_0_5'])}). Among the LLM conditions, "
            f"only excerpt-only retrieval paid for its faithfulness gain with a clear loss of coverage."),
          ("table", dict(label="cov", caption="Coverage proxy per condition: mean share of source sentences whose best cosine similarity to any claim reaches the threshold, and mean best similarity.",
                         columns=["Condition", "Coverage at 0.5", "Coverage at 0.6", "Coverage at 0.7", "Mean best similarity", "Claims per summary"],
                         widths=[1.7, 0.95, 0.95, 0.95, 1.05, 0.9], font=10,
                         rows=[[NAME[c], pct(r.coverage_at_0_5), pct(r.coverage_at_0_6), pct(r.coverage_at_0_7), f3(r.mean_best_similarity), f"{r.n_claims:.1f}"] for c, r in cov.iterrows()],
                         note="Paired tests on coverage at 0.6: " + "; ".join(f"{col[8:].replace('_vs_', ' vs ')} {fp(cov.loc['E0', col])}" for col in cov.columns if col.startswith("p_cov06_")) + "."))]
    b += [H2("5.6 What Remains Unsupported"),
          P(f"[[tab:tax]] categorizes the claims that the judge did not label Supported: {uns_total['E0']} for E0, "
            f"{uns_total['E1']} for E1, {uns_total.get('E1b', 0)} for E1b and {uns_total.get('E3', 0)} for E3. Follow-up and "
            f"scheduling statements are the largest category under every LLM condition and the only one that grows under "
            f"excerpt-only RAG ({R.tax_total('E0','Follow-up or scheduling')} to {R.tax_total('E1','Follow-up or scheduling')}); "
            f"they are almost never contradicted, because the note usually says nothing about them at all. RAG reduced the number "
            f"of unsupported claims about generic advice, procedures, findings and medications, the categories in which specific, "
            f"retrievable facts occur: excerpts constrain what the model says about the note's content, but not what it adds from "
            f"its parametric knowledge of how patient instructions usually end (Section 2.6). Verification is the only condition "
            f"that cut the follow-up category substantially ({R.tax_total('E0','Follow-up or scheduling')} for E0 to "
            f"{R.tax_total('E3','Follow-up or scheduling')} for E3), because a verifier that checks each sentence against the note "
            f"removes exactly the sentences that no passage supports, whereas retrieval only changes what the generator sees."),
          ("table", dict(label="tax", caption="Keyword-assisted categories of claims that the judge did not label Supported, for the LLM conditions. Contradicted counts are the subset labeled Contradicted.",
                         columns=["Category"] + [f"{c} {k}" for c in GEN for k in ("unsupp.", "contra.")], widths=[1.9] + [0.575] * (2 * len(GEN)), font=9, rows=tax_rows)),
          P(f"Paired claims show both the promise and the fragility of the metric. Of the E0 claims labeled Contradicted whose most "
            f"similar E1 claim in the same document was labeled Supported ({n_fixed} pairs), some are genuine corrections, but in "
            f"document 16 the note states \"COMPLICATIONS: None\", the E0 phrasing \"The procedure went smoothly, and there were no "
            f"complications\" was called Contradicted and the E1 phrasing \"there were no complications from the procedures\" "
            f"Supported; the two claims say the same thing, and the difference lies in the judge. Claims that appear almost "
            f"verbatim under both conditions and are unsupported under both are opening sentences that restate the reason for the "
            f"visit in patient-facing language, standard closing advice, and summaries of the patient's history; several are "
            f"faithful paraphrases of the note that the NLI model cannot match to any single sentence, the failure mode that "
            f"Section 5.3 quantified.")]
    b += [H2("5.7 Summary of the Pilot"),
          ("bullets", [
              f"**The judge is not valid for this task.** Against medical-expert annotations it reaches kappa {f2(cal_all.any_kappa)} "
              f"and AUROC {f2(cal_all.auroc_1_minus_p_entail)}, flags four sentences in five, and ranks systems in nearly the reverse "
              f"order of the experts; no evidence-aggregation rule repairs this. Its pilot rates (E0 UFR {f3(R.mean('E0','UFR'))}, "
              f"CR {f3(R.mean('E0','CR'))}) cannot be read as hallucination rates.",
              f"**Under that judge,** excerpt-only RAG lowered CR by {rel(t10c)} (adjusted {fp(t10c.p_holm)}, suggestive only), left "
              f"UFR unchanged and reduced coverage; E1b lowered UFR modestly without the coverage loss; verification lowered UFR the "
              f"most (adjusted {fp(t30u.p_holm)}) by deleting claims and abstaining, with coverage at the primary threshold matching "
              f"the baseline. These are the effects the main study re-tests with a validated judge.",
              f"**Judge errors on verbatim text** come from negated sentences ({e2c_neg} of {e2c_total} contradicted E2 sentences) and "
              f"from malformed sentence fragments in the public corpus (about a third); the judge also labeled {n_abst_contra} of "
              f"E3's {n_abst} abstentions as contradictions.",
              f"**Two evaluation artifacts** had inflated earlier results: header lines counted as claims raised the apparent RAG "
              f"effect on CR from {rel(t10c)} to 32 percent, and abstentions counted as claims would have reversed the conclusion "
              f"about verification. Both are corrected in every analysis of this thesis."])]

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
