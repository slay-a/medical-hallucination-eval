"""ch5_pilot.py — Chapter 5 (pilot study results on MTSamples) and the abstract."""
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

    # ══════════════════════════════════ CHAPTER 5 ══════════════════════════════════
    b += [H1("Chapter 5 Pilot Study: Results on MTSamples"), H2("5.1 Descriptive Statistics of the Generated Summaries")]
    b += [P(f"[[tab:desc]] summarizes the summaries produced under the five conditions for the {R.n_docs} documents, and "
            f"[[fig:lengths]] shows their distributions. Zero-context summaries were the longest (mean "
            f"{R.words['E0'].mean():.0f} words); excerpt-only RAG summaries were shorter (mean {R.words['E1'].mean():.0f} words) "
            f"although both prompts requested 150 to 250 words, and giving the model the full note as well (E1b) restored the "
            f"length of the baseline (mean {R.words['E1b'].mean():.0f} words). Verified summaries (E3) were shorter still (mean "
            f"{R.words['E3'].mean():.0f} words) because the revision step deleted claims that the verifier could not support and "
            f"wrote an abstention where a section had no supported content; extractive summaries of five sentences averaged "
            f"{R.words['E2'].mean():.0f} words with a wide spread that follows the sentence length of each note. After removal "
            f"of header lines and abstentions, E0 summaries yielded {R.claims_per['E0']:.1f} claims per document, E1 "
            f"{R.claims_per['E1']:.1f}, E1b {R.claims_per['E1b']:.1f}, E3 {R.claims_per['E3']:.1f} and E2 {R.claims_per['E2']:.1f}. "
            f"E3 produced {n_abst} abstention lines in {n_abst_docs} of the {R.n_docs} summaries, and its summaries contained fewer "
            f"claims than the E1 drafts they were revised from in {fewer_e3} of {R.n_docs} documents. In total {R.n_claims_total:,} "
            f"claims were labeled by the judge."),
          ("table", dict(label="desc", caption="Length, number of claims and distribution of judge labels per condition (header lines excluded). Shares are of all claims in the condition.",
                         columns=["Condition", "Words per summary, mean (SD)", "Claims per summary", "Total claims", "Supported", "Not-Supported", "Contradicted"],
                         widths=[1.3, 1.2, 0.85, 0.75, 0.9, 0.9, 0.9], font=9.5,
                         rows=[[NAME[c], f"{R.words[c].mean():.0f} ({R.words[c].std():.0f})", f"{R.claims_per[c]:.1f}", f"{tot[c]:,}",
                                f"{int(lc.loc[c,'Supported'])} ({pct0(share[(c,'Supported')])})", f"{int(lc.loc[c,'Not-Supported'])} ({pct0(share[(c,'Not-Supported')])})",
                                f"{int(lc.loc[c,'Contradicted'])} ({pct0(share[(c,'Contradicted')])})"] for c in C])),
          ("figure", dict(label="lengths", path=f"{FIG}/fig_length_claims.png", width=6.3, caption="Words per summary and claims per summary (header lines excluded) under the three conditions.")),
          P(f"The label distributions in [[fig:labels]] already show the shape of the main result. Under the two retrieval "
            f"conditions and the baseline the judge labeled only about one claim in five as Supported ({pct0(share[('E0','Supported')])} for E0, "
            f"{pct0(share[('E1','Supported')])} for E1 and {pct0(share[('E1b','Supported')])} for E1b), most claims as Not-Supported, and "
            f"{pct0(share[('E0','Contradicted')])}, {pct0(share[('E1','Contradicted')])} and {pct0(share[('E1b','Contradicted')])} as "
            f"Contradicted. The verified condition had the highest Supported share of any LLM condition ({pct0(share[('E3','Supported')])}) "
            f"and a Contradicted share of {pct0(share[('E3','Contradicted')])}. For the "
            f"extractive condition, whose claims are verbatim source sentences, {pct0(share[('E2','Supported')])} were labeled "
            f"Supported, {e2_ns} claims Not-Supported and {e2c_total} Contradicted; those {e2c_total + e2_ns} labels are judge "
            f"errors by construction and are analyzed in Section 5.5.4."),
          ("figure", dict(label="labels", path=f"{FIG}/fig_label_distribution.png", width=6.3, caption="Share of claims labeled Supported, Not-Supported and Contradicted by the NLI judge under each condition."))]
    b += [H2("5.2 Primary Comparison of the Three Approaches")]
    b += [P(f"[[tab:agg]] reports the per-document means, standard deviations and medians of UFR and CR, and [[tab:tests]] the "
            f"paired comparisons. The Unsupported Fact Rate did not differ between the zero-context and RAG conditions: mean "
            f"{f3(t10u.mean_a)} versus {f3(t10u.mean_b)}, a difference of {signed(t10u.delta_mean)} with a 95 percent bootstrap "
            f"interval from {signed(t10u.ci95_low)} to {signed(t10u.ci95_high)} ({fp(t10u.p)}); RAG lowered UFR in only "
            f"{t10u.pct_improved:.0f} percent of documents and raised it in {t10u.pct_worse:.0f} percent. The Contradiction Rate "
            f"did differ: mean {f3(t10c.mean_a)} for E0 versus {f3(t10c.mean_b)} for E1, a reduction of {signed(t10c.delta_mean)} "
            f"({rel(t10c)} relative; 95 percent interval {signed(t10c.ci95_low)} to {signed(t10c.ci95_high)}; W = {t10c.W:.0f}, "
            f"{fp(t10c.p)}, Holm-adjusted {fp(t10c.p_holm)}), with a small standardized effect (d(z) = {f2(t10c.d_z)}, "
            f"rank-biserial r = {f2(t10c.rank_biserial)}) and improvement in {t10c.pct_improved:.0f} percent of documents "
            f"([[fig:scatter]], [[fig:pct]])."),
          P(f"The extractive condition behaved as a faithfulness bound should. Its UFR ({f3(t20u.mean_b)}) was lower than that of "
            f"E0 by {signed(t20u.delta_mean)} ({fp(t20u.p)}) and lower than that of E1 by {signed(t21u.delta_mean)} ({fp(t21u.p)}), "
            f"with every one of the {R.n_docs} documents improving. Its CR ({f3(t20c.mean_b)}) was lower than E0 "
            f"({signed(t20c.delta_mean)}, {fp(t20c.p)}) but not significantly lower than E1 ({signed(t21c.delta_mean)}, "
            f"{fp(t21c.p)}, Holm-adjusted {fp(t21c.p_holm)}). The medians in [[fig:ufrbox]] and [[fig:crbox]] tell the same story: "
            f"the UFR distributions of E0 and E1 overlap almost completely, while E2 sits near zero; the CR distribution of E1 is "
            f"shifted below E0, but the E2 distribution has a heavy right tail produced by the judge's errors on verbatim "
            f"negations."),
          P(f"Showing the model the full note together with the excerpts (E1b) moved the metrics in the direction of fewer "
            f"unsupported claims without a clear effect on contradictions. E1b lowered UFR relative to E0 (mean {f3(t1b0u.mean_a)} "
            f"versus {f3(t1b0u.mean_b)}, {signed(t1b0u.delta_mean)}, 95 percent interval {signed(t1b0u.ci95_low)} to "
            f"{signed(t1b0u.ci95_high)}, {fp(t1b0u.p)}, d(z) = {f2(t1b0u.d_z)}), but the difference from E1 did not reach "
            f"significance ({signed(t1b1u.delta_mean)}, {fp(t1b1u.p)}). Its CR ({f3(t1b0c.mean_b)}) was statistically "
            f"indistinguishable from both E0 ({fp(t1b0c.p)}) and E1 ({fp(t1b1c.p)}). E1b therefore recovered the length and "
            f"breadth of the baseline while keeping a small advantage in supported content, which suggests that the excerpts "
            f"focus the model's attention even when the whole note is available."),
          P(f"Verification produced the largest change of any LLM condition on the unsupported fact rate. E3 lowered UFR from "
            f"{f3(t30u.mean_a)} (E0) to {f3(t30u.mean_b)} ({signed(t30u.delta_mean)}, 95 percent interval {signed(t30u.ci95_low)} to "
            f"{signed(t30u.ci95_high)}, {fp(t30u.p)}, Holm-adjusted {fp(t30u.p_holm)}, d(z) = {f2(t30u.d_z)}, improvement in "
            f"{t30u.pct_improved:.0f} percent of documents), and relative to its own E1 drafts from {f3(t31u.mean_a)} to "
            f"{f3(t31u.mean_b)} ({fp(t31u.p)}, d(z) = {f2(t31u.d_z)}); it also improved on E1b ({fp(t31bu.p)}). Its contradiction "
            f"rate ({f3(t30c.mean_b)}) did not differ from E0 ({fp(t30c.p)}) and was not significantly higher than E1 "
            f"({signed(t31c.delta_mean)}, {fp(t31c.p)}). The mechanism is visible in the verification log: the verifier examined "
            f"{n_verif} draft claims, judged {n_verif_unsup} of them ({pct0(n_verif_unsup / max(1, n_verif))}) unsupported by the "
            f"retrieved sentences, and the revision deleted or replaced those claims and wrote \"Not stated in the note\" for "
            f"{n_abst} sections. E3 thus reduced unsupported content mainly by saying less, and Section 5.6 shows what that cost in "
            f"coverage. Because the same judge scores all conditions, the E3 versus E1 comparison isolates the effect of the "
            f"verification and revision step alone."),
          ("table", dict(label="agg", caption=f"Per-document Unsupported Fact Rate and Contradiction Rate under each condition (n = {R.n_docs} documents; header lines excluded).",
                         columns=["Metric", "Condition", "Mean", "SD", "Median"], widths=[1.0, 2.4, 1.0, 1.0, 1.1], font=10,
                         rows=[[m if i == 0 else "", NAME[c], f3(R.mean(c, m)), f3(R.sd(c, m)), f3(R.median(c, m))] for m in ("UFR", "CR") for i, c in enumerate(C)])),
          ("table", dict(label="tests", caption="Paired comparisons between conditions. Delta is the mean of per-document differences (second condition minus first); negative values favor the second condition. CI is the 95 percent percentile bootstrap interval (10,000 resamples). p is the two-sided Wilcoxon signed-rank p-value and p(Holm) its adjustment over the six tests. d(z) is the standardized mean difference and r the matched-pairs rank-biserial correlation (positive = improvement).",
                         columns=["Comparison", "Metric", "Delta", "95% CI", "Median delta", "p", "p (Holm)", "d(z)", "r", "Improved / same / worse"],
                         widths=[0.85, 0.55, 0.65, 1.0, 0.6, 0.6, 0.6, 0.5, 0.5, 0.95], font=8.5,
                         rows=[[t.comparison, t.metric, signed(t.delta_mean), f"[{t.ci95_low:+.3f}, {t.ci95_high:+.3f}]", signed(t.delta_median), fpn(t.p), fpn(t.p_holm),
                                f2(t.d_z), f2(t.rank_biserial), f"{t.pct_improved:.0f}% / {t.pct_unchanged:.0f}% / {t.pct_worse:.0f}%"]
                               for _, t in R.tests.iterrows()])),
          ("figure", dict(label="ufrbox", path=f"{FIG}/fig_ufr_boxplot.png", width=5.6, caption="Distribution of the per-document Unsupported Fact Rate under the three conditions. Boxes show quartiles, the bar the median, and points individual documents.")),
          ("figure", dict(label="crbox", path=f"{FIG}/fig_cr_boxplot.png", width=5.6, caption="Distribution of the per-document Contradiction Rate under the three conditions.")),
          ("figure", dict(label="scatter", path=f"{FIG}/fig_cr_scatter.png", width=6.3, caption="Per-document Contradiction Rate of E1 (left) and E2 (right) against E0. Points below the diagonal are documents in which the condition produced a lower rate than the zero-context baseline.")),
          ("figure", dict(label="pct", path=f"{FIG}/fig_pct_improved.png", width=6.3, caption="Share of documents whose UFR (left) and CR (right) improved, were unchanged or worsened in each paired comparison."))]
    b += [H2("5.3 Effects of Note Type and Length")]
    cons, dis = "Consult - History and Phy.", "Discharge Summary"
    b += [P(f"[[fig:spec]] separates the {int(spec_n[cons])} consultation notes from the {int(spec_n[dis])} discharge summaries. "
            f"The pattern is the same in both note types: UFR is high and indistinguishable between E0 and E1 "
            f"({f3(spec.loc[cons,'E0_UFR'])} versus {f3(spec.loc[cons,'E1_UFR'])} for consultations, {f3(spec.loc[dis,'E0_UFR'])} "
            f"versus {f3(spec.loc[dis,'E1_UFR'])} for discharge summaries), E1 has a lower CR than E0 in consultations "
            f"({f3(spec.loc[cons,'E0_CR'])} versus {f3(spec.loc[cons,'E1_CR'])}) with a smaller gap in the few discharge summaries "
            f"({f3(spec.loc[dis,'E0_CR'])} versus {f3(spec.loc[dis,'E1_CR'])}), and E2 is lowest on both metrics, particularly for "
            f"discharge summaries, whose sentences are more often complete statements than the terse fragments of a physical "
            f"examination. With only ten discharge summaries these subgroup estimates are imprecise and no subgroup tests were "
            f"performed."),
          P(f"Source length did not moderate the effects. Spearman's correlation between the number of words in the note and the "
            f"paired E1 minus E0 difference was {rho_u:+.2f} for UFR ({fp(p_u)}) and {rho_c:+.2f} for CR ({fp(p_c)}). Longer notes "
            f"did not produce higher baseline rates either (UFR: rho = {rho_e0u:+.2f}, {fp(p_e0u)}; CR: rho = {rho_e0c:+.2f}, "
            f"{fp(p_e0c)}), which is consistent with the E0 prompt truncating every note to 4,500 characters and with the judge "
            f"retrieving evidence from the complete note regardless of its length."),
          ("figure", dict(label="spec", path=f"{FIG}/fig_specialty.png", width=6.3, caption="Mean UFR (left) and CR (right) by note type and condition; error bars are standard errors of the mean."))]
    b += [H2("5.4 Validation of the Judge Against Medical Experts")]
    b += [P(f"The judge was applied unchanged to the {int(cal_all.n_summaries)} expert-annotated summaries, which contain "
            f"{int(cal_all.n_sentences):,} sentences after segmentation, {pct(cal_all.expert_flag_rate)} of which overlap at least one "
            f"expert-annotated unsupported fact ({pct(cal_gen.expert_flag_rate)} of sentences in LLM-generated summaries and "
            f"{pct(cal_doc.expert_flag_rate)} in doctor-written ones). The judge flagged {pct(cal_all.judge_UFR)} of the same "
            f"sentences as Not-Supported or Contradicted. [[tab:calib]] gives the agreement statistics and [[fig:calib]] "
            f"illustrates them."),
          P(f"Agreement with the experts was poor. Because the judge flags four sentences in five, its recall for expert-flagged "
            f"sentences is high ({pct(cal_all.any_recall)}) but its precision is {pct(cal_all.any_precision)}, its specificity "
            f"{pct(cal_all.any_specificity)}, and Cohen's kappa is {f2(cal_all.any_kappa)}, that is, essentially chance agreement. "
            f"The continuous scores separate the two groups of sentences only weakly: the area under the ROC curve is "
            f"{f2(cal_all.auroc_1_minus_p_entail)} for one minus the entailment probability and {f2(cal_all.auroc_p_contra)} for "
            f"the contradiction probability, where 0.5 is chance. The contradiction detector is no better: its precision is "
            f"{pct(cal_all.contra_precision)} and its kappa {f2(cal_all.contra_kappa)}. For the LLM-generated subset, where "
            f"experts flagged {pct(cal_gen.expert_flag_rate)} of sentences, precision falls to {pct(cal_gen.any_precision)}; for the "
            f"three GPT-4 configurations, in which experts found unsupported content in only 5 to 9 percent of sentences, the judge "
            f"still flagged 86 to 87 percent."),
          ("table", dict(label="calib", caption="Agreement between the judge and the medical-expert annotations at the sentence level. The any detector flags Not-Supported or Contradicted sentences; AUROC uses one minus the entailment probability as the score. Kappa is Cohen's kappa.",
                         columns=["Group", "Summaries", "Sentences", "Expert-flagged", "Judge-flagged (UFR)", "Precision", "Recall", "Specificity", "F1", "Kappa", "AUROC"],
                         widths=[1.2, 0.6, 0.6, 0.65, 0.65, 0.55, 0.5, 0.6, 0.45, 0.45, 0.5], font=8.5,
                         rows=[[r.group.replace("_", " "), str(int(r.n_summaries)), f"{int(r.n_sentences):,}", pct(r.expert_flag_rate), pct(r.judge_UFR), f2(r.any_precision), f2(r.any_recall),
                                f2(r.any_specificity), f2(r.any_f1), f2(r.any_kappa), f2(r.auroc_1_minus_p_entail)] for _, r in R.cal.iterrows()])),
          ("figure", dict(label="calib", path=f"{FIG}/fig_calibration.png", width=6.5, caption="Left: share of sentences flagged by the experts and by the judge, with the precision and recall of the judge's flags, per group. Right: distribution of the judge's entailment probability for sentences that the experts did and did not flag.")),
          P(f"[[tab:calsys]] compares the six groups of summaries as ranked by the experts and by the judge. The experts found the "
            f"most unsupported content in doctor-written instructions and in the original Llama-2 summaries and the least in the "
            f"GPT-4 summaries, reproducing the ordering reported by Hegselmann et al. [[cite:hegselmann2024]]. The judge's UFR "
            f"orders the groups almost in reverse, rating the three GPT-4 configurations as the worst, and its CR varies by only a "
            f"few points across groups. At the level of individual summaries the judge's UFR is weakly but significantly "
            f"correlated with the share of expert-flagged sentences (Spearman rho = "
            f"{f2(R.cal_sum[(R.cal_sum.group=='all')&(R.cal_sum.judge_metric=='judge_UFR')].spearman_rho.iloc[0])}, "
            f"{fp(R.cal_sum[(R.cal_sum.group=='all')&(R.cal_sum.judge_metric=='judge_UFR')].spearman_p.iloc[0])}), whereas its CR is "
            f"not (rho = {f2(R.cal_sum[(R.cal_sum.group=='all')&(R.cal_sum.judge_metric=='judge_CR')].spearman_rho.iloc[0])}). "
            f"Recall was uniformly high across expert label types ([[tab:callabel]]) simply because almost everything is flagged; "
            f"the contradiction label recovered {pct0(R.cal_lab[R.cal_lab.expert_label=='contradicted_fact'].judge_recall_contradicted.iloc[0])} "
            f"of the sentences that experts marked as contradicting the source, but also fired on 28 to 32 percent of all "
            f"sentences in every group."),
          ("table", dict(label="calsys", caption="Mean share of expert-flagged sentences per summary and mean judge UFR and CR per summary, by group of summaries.",
                         columns=["Group", "Expert-flagged share", "Judge UFR", "Judge CR"], widths=[2.4, 1.4, 1.3, 1.4], font=10,
                         rows=[[r.group.replace("_", " "), f3(r.expert_frac), f3(r.judge_UFR), f3(r.judge_CR)] for _, r in R.cal_sys.sort_values("expert_frac", ascending=False).iterrows()])),
          ("table", dict(label="callabel", caption="Recall of the judge for expert-flagged sentences by expert label type. A sentence may carry more than one label.",
                         columns=["Expert label", "Sentences", "Recall, any flag", "Recall, Contradicted", "Mean p(e)"], widths=[2.0, 0.9, 1.2, 1.3, 1.1], font=10,
                         rows=[[r.expert_label.replace("_", " "), str(int(r.n_sentences)), f2(r.judge_recall_any), f2(r.judge_recall_contradicted), f2(r.mean_p_entail)]
                               for _, r in R.cal_lab[R.cal_lab.expert_label != "__span_counts__"].sort_values("n_sentences", ascending=False).iterrows()]))]
    if var_all is not None:
        best_auc = var_all.auroc_1_minus_pe.idxmax(); best_kappa = var_all.best_kappa.idxmax()
        b += [P(f"[[tab:variants]] and [[fig:variants]] test whether the disagreement stems from how evidence is aggregated. It does "
                f"not. Concatenating the three retrieved sentences into a single premise reduced the flag rate from "
                f"{pct(var_all.loc['max3','judge_flag_rate'])} to {pct(var_all.loc['concat3','judge_flag_rate'])} and raised the "
                f"area under the curve from {f2(var_all.loc['max3','auroc_1_minus_pe'])} to {f2(var_all.loc['concat3','auroc_1_minus_pe'])}; "
                f"concatenating five sentences gave the highest summary-level correlation (rho = {f2(var_all.loc['concat5','summary_spearman'])}). "
                f"Scoring the claim against every sentence of the hospital course, the SummaC zero-shot design, made matters worse "
                f"(flag rate {pct(var_all.loc['maxall','judge_flag_rate'])}, kappa {f2(var_all.loc['maxall','kappa'])}), and "
                f"presenting the whole hospital course as one premise flagged {pct(var_all.loc['concat_ctx','judge_flag_rate'])} of "
                f"sentences, because a 6-layer model truncated to 512 tokens cannot find a sentence's support in a long premise. "
                f"The best attainable kappa over every threshold and variant was {f2(var_all.best_kappa.max())} and the best area "
                f"under the curve {f2(var_all.auroc_1_minus_pe.max())} (variant {VNAME[best_auc].lower()}). The limiting factor is "
                f"therefore the NLI model's judgment of individual patient-facing sentences, not the retrieval or aggregation rule."),
              ("table", dict(label="variants", caption="Evidence-aggregation variants of the judge evaluated on all 1,781 expert-annotated sentences with the same cross-encoder. Flag rate, precision, recall and kappa are at τ = 0.5; best kappa is the maximum over thresholds from 0.05 to 0.95; rho is Spearman's correlation between judge and expert rates per summary.",
                             columns=["Variant", "Judge flag rate", "AUROC", "Precision", "Recall", "Kappa", "Best kappa (τ)", "Summary rho"],
                             widths=[2.2, 0.75, 0.6, 0.7, 0.6, 0.6, 0.85, 0.7], font=9.5,
                             rows=[[VNAME[k], pct(r.judge_flag_rate), f2(r.auroc_1_minus_pe), f2(r.precision), f2(r.recall), f2(r.kappa), f"{f2(r.best_kappa)} ({r.best_kappa_tau:.2f})", f2(r.summary_spearman)]
                                   for k, r in var_all.loc[[k for k in VNAME if k in var_all.index]].iterrows()])),
              ("figure", dict(label="variants", path=f"{FIG}/fig_variants.png", width=6.5, caption="Left: area under the ROC curve, best attainable kappa and flag rate of each aggregation variant against the expert annotations, with the experts' own flag rate as a dashed line. Right: summary-level Spearman correlation between judge UFR and the expert-flagged share, by subset."))]
    b += [H2("5.5 Robustness of the Comparison to the Judge's Design Choices"), H3("5.5.1 Decision Threshold")]
    b += [P(f"Raising the decision threshold τ makes the judge more conservative in both directions: fewer claims reach the "
            f"entailment threshold, so UFR rises, and fewer reach the contradiction threshold, so CR falls ([[tab:tau]], "
            f"[[fig:tau]]). The E1 versus E0 difference in UFR is not significant at any threshold. The difference in CR is "
            f"significant only at τ = 0.5 ({fp(tau.loc[0.5,'p_E1_vs_E0_CR'])}); at τ = 0.6 it is marginal "
            f"({fp(tau.loc[0.6,'p_E1_vs_E0_CR'])}) and at τ = 0.7 and above it disappears ({fp(tau.loc[0.7,'p_E1_vs_E0_CR'])}, "
            f"{fp(tau.loc[0.8,'p_E1_vs_E0_CR'])}, {fp(tau.loc[0.9,'p_E1_vs_E0_CR'])}). The E2 advantage in UFR is significant at every "
            f"threshold, while its CR advantage over E0 also weakens above τ = 0.6. The RAG effect on contradictions therefore "
            f"depends on counting claims for which the model's contradiction probability lies between 0.5 and 0.7, exactly the "
            f"region in which the validation study found the model least trustworthy. The new conditions behave differently: "
            f"E1b and E3 never differed from E0 in CR at any threshold, whereas the UFR advantage of E3 over E0 was significant at "
            f"{int((tau['p_E3_vs_E0_UFR'] < 0.05).sum())} of the {len(tau)} thresholds and that of E1b at "
            f"{int((tau['p_E1b_vs_E0_UFR'] < 0.05).sum())} of {len(tau)} ([[tab:tauufr]]). The verification effect is therefore the "
            f"most robust effect in the study to the judge's decision threshold."),
          ("table", dict(label="tau", caption="Mean CR per condition and paired p-values against E0 as a function of the decision threshold τ applied to the stored probabilities.",
                         columns=["τ"] + [f"{c} CR" for c in C] + [f"p {c} vs E0" for c in OTHERS],
                         widths=[0.4] + [0.55] * len(C) + [0.72] * len(OTHERS), font=9,
                         rows=[[f"{t:.1f}"] + [f3(r[f"{c}_CR_mean"]) for c in C] + [fpn(r[f"p_{c}_vs_E0_CR"]) for c in OTHERS] for t, r in tau.iterrows()])),
          ("table", dict(label="tauufr", caption="Mean UFR per condition and paired p-values against E0 as a function of the decision threshold τ.",
                         columns=["τ"] + [f"{c} UFR" for c in C] + [f"p {c} vs E0" for c in OTHERS],
                         widths=[0.4] + [0.55] * len(C) + [0.72] * len(OTHERS), font=9,
                         rows=[[f"{t:.1f}"] + [f3(r[f"{c}_UFR_mean"]) for c in C] + [fpn(r[f"p_{c}_vs_E0_UFR"]) for c in OTHERS] for t, r in tau.iterrows()])),
          ("figure", dict(label="tau", path=f"{FIG}/fig_threshold_ablation.png", width=6.3, caption="Mean UFR (left) and CR (right) per condition as the decision threshold varies (solid lines, left axes) and the paired E1 versus E0 p-value (dotted line, right axes, log scale; the red line marks 0.05)."))]
    b += [H3("5.5.2 Number of Evidence Sentences")]
    b += [P(f"Re-running retrieval and classification with k = 3 reproduced every stored label exactly (agreement "
            f"{pct0(topk.loc[3,'label_agreement_with_stored'])}), confirming that the pipeline is deterministic. With a single "
            f"evidence sentence (k = 1) contradictions almost vanish (E0 CR {f3(topk.loc[1,'E0_CR_mean'])}, E1 "
            f"{f3(topk.loc[1,'E1_CR_mean'])}) and the E1 advantage becomes marginal ({fp(topk.loc[1,'p_E1_vs_E0_CR'])}); with five "
            f"sentences contradictions become more frequent for every condition, including the verbatim E2 sentences "
            f"(CR {f3(topk.loc[5,'E2_CR_mean'])}), while the E1 advantage persists ({fp(topk.loc[5,'p_E1_vs_E0_CR'])}) "
            f"([[tab:topk]], [[fig:topk]]). UFR moves little with k. The rise of CR with k for E2 is diagnostic: each additional "
            f"retrieved sentence gives the maximum-over-evidence rule another chance to find a sentence that the model calls a "
            f"contradiction, so the contradiction score of the pipeline measures, in part, how many loosely related sentences a "
            f"claim is compared with."),
          ("table", dict(label="topk", caption="Mean UFR and CR per condition when k = 1, 3 or 5 source sentences are retrieved per claim, with paired CR p-values against E0. k = 3 is the pipeline setting.",
                         columns=["k"] + [f"{c} UFR" for c in C] + [f"{c} CR" for c in C] + [f"p CR {c} vs E0" for c in OTHERS],
                         widths=[0.3] + [0.5] * (2 * len(C)) + [0.62] * len(OTHERS), font=8,
                         rows=[[str(int(k))] + [f3(r[f"{c}_UFR_mean"]) for c in C] + [f3(r[f"{c}_CR_mean"]) for c in C] + [fpn(r[f"p_{c}_vs_E0_CR"]) for c in OTHERS] for k, r in topk.iterrows()])),
          ("figure", dict(label="topk", path=f"{FIG}/fig_topk_ablation.png", width=6.3, caption="Mean UFR (left) and CR (right) per condition as a function of the number of evidence sentences retrieved per claim."))]
    b += [H3("5.5.3 Repaired Evidence Text")]
    b += [P(f"Repairing the comma-encoded line breaks of MTSamples on the evidence side ([[tab:clean]]) changed few labels: of the "
            f"{int(trans.values.sum())} E0 and E1 claims, {int(np.trace(trans.loc[trans.columns, trans.columns].values)) if set(trans.index)==set(trans.columns) else int(sum(trans.loc[l,l] for l in trans.index if l in trans.columns))} kept their label. "
            f"Mean CR fell slightly for both LLM conditions (E0 {f3(clean.loc['E0','CR_mean'])}, E1 {f3(clean.loc['E1','CR_mean'])}) "
            f"and the paired E1 advantage remained at the boundary of significance ({fp(clean.loc['E0','p_E1_vs_E0_CR'])}). The "
            f"clearest effect was on the extractive control: regenerated from repaired sentences, E2 produced no Not-Supported "
            f"claims and {e2clean_contra} Contradicted claims out of {e2clean_n} (CR {f3(clean.loc['E2','CR_mean'])}) instead of "
            f"{e2c_total} out of {e2_n}. Roughly a third of the judge's false contradictions on verbatim text were therefore "
            f"caused by malformed sentence fragments; the remainder are genuine model errors."),
          ("table", dict(label="clean", caption="UFR and CR per condition after repairing the MTSamples line breaks in the evidence text (E0 and E1 claims re-scored; E2 regenerated from the repaired sentences), with paired tests.",
                         columns=["Condition", "Claims", "Contradicted", "Not-Supported", "Mean UFR", "Median UFR", "Mean CR", "Median CR"],
                         widths=[1.6, 0.6, 0.85, 0.9, 0.7, 0.75, 0.7, 0.75], font=9.5,
                         rows=[[NAME[c], str(int(r.n_claims)), str(int(r.n_contradicted)), str(int(r.n_not_supported)), f3(r.UFR_mean), f3(r.UFR_median), f3(r.CR_mean), f3(r.CR_median)] for c, r in clean.iterrows()],
                         note="Paired tests with repaired evidence, CR versus E0: " + "; ".join(f"{c} {fp(clean.loc['E0', f'p_{c}_vs_E0_CR'])}" for c in OTHERS) + ". UFR E1 vs E0 " + fp(clean.loc['E0','p_E1_vs_E0_UFR']) + "."))]
    b += [H3("5.5.4 Judge Errors on Verbatim Sentences and the Role of Negation")]
    neg = R.e2err[R.e2err.has_negation.isin([True, False, "True", "False"])].copy()
    neg["has_negation"] = neg.has_negation.astype(str)
    def negcount(c, h, l):
        v = neg[(neg.condition == c) & (neg.has_negation == h) & (neg.label == l)]["count"]
        return int(v.iloc[0]) if len(v) else 0
    e2_neg_total = negcount("E2", "True", "Contradicted") + negcount("E2", "True", "Supported") + negcount("E2", "True", "Not-Supported")
    e2_pos_total = negcount("E2", "False", "Contradicted") + negcount("E2", "False", "Supported") + negcount("E2", "False", "Not-Supported")
    b += [P(f"[[fig:e2probs]] plots the judge's two scores for every E2 claim. For the {e2c_total} verbatim sentences labeled "
            f"Contradicted, the mean contradiction probability was {f2(R.e2err_value('Contradicted_mean_p_contra'))} but the mean "
            f"entailment probability was also {f2(R.e2err_value('Contradicted_mean_p_entail'))}, and {e2c_both} of the "
            f"{e2c_total} had both probabilities above 0.9. The model is not uncertain about these sentences; it assigns near-"
            f"certainty to both entailment and contradiction across different evidence sentences, and the symmetric decision "
            f"rule breaks the tie by whichever maximum is larger. {e2c_neg} of the {e2c_total} contradicted E2 sentences contain "
            f"a negation cue ([[tab:neg]]): among E2 sentences with such a cue, {negcount('E2','True','Contradicted')} of "
            f"{e2_neg_total} were called Contradicted, against {negcount('E2','False','Contradicted')} of {e2_pos_total} without. "
            f"Sentences such as \"Denies nausea, vomiting, or diarrhea\" or \"No swelling, pain, or numbness\" are the typical "
            f"victims: a negated list of symptoms, compared with a neighbouring sentence that mentions one of those symptoms in a "
            f"different context, reads to the model like a contradiction. Negation cues raise the contradiction rate of the LLM "
            f"conditions as well ({negcount('E0','True','Contradicted')} of {negcount('E0','True','Contradicted')+negcount('E0','True','Supported')+negcount('E0','True','Not-Supported')} "
            f"E0 claims with a cue versus {negcount('E0','False','Contradicted')} of {negcount('E0','False','Contradicted')+negcount('E0','False','Supported')+negcount('E0','False','Not-Supported')} without)."),
          P(f"A related artifact appeared in the verified condition. Of the {n_abst} abstention lines that E3 wrote "
            f"(\"Not stated in the note.\"), the judge labeled {n_abst_contra} as Contradicted before they were excluded from the "
            f"claim set. An abstention is a negated existential statement about the note itself; paired with any retrieved "
            f"sentence that does state something, the cross-encoder reads it as a contradiction. Had abstentions been counted as "
            f"claims, the E3 contradiction rate would have risen from {f3(R.mean('E3','CR'))} to about 0.27 and reversed the "
            f"conclusion about verification, which is why Section 3.5 treats abstentions as a separate category. Among the "
            f"remaining E3 claims labeled Contradicted, {negcount('E3','True','Contradicted')} of "
            f"{negcount('E3','True','Contradicted') + negcount('E3','False','Contradicted')} contain a negation cue, the same pattern as "
            f"in the other conditions."),
          ("table", dict(label="neg", caption="Judge labels by the presence of a negation cue in the claim (no, not, denies, without, negative, unremarkable, absent and similar), per condition.",
                         columns=["Condition", "Negation cue", "Supported", "Not-Supported", "Contradicted", "Share Contradicted"],
                         widths=[1.6, 1.0, 0.9, 1.0, 1.0, 1.0], font=10,
                         rows=[[NAME[c], "yes" if h == "True" else "no", str(negcount(c, h, "Supported")), str(negcount(c, h, "Not-Supported")), str(negcount(c, h, "Contradicted")),
                                pct(negcount(c, h, "Contradicted") / max(1, negcount(c, h, "Supported") + negcount(c, h, "Not-Supported") + negcount(c, h, "Contradicted")))]
                               for c in C for h in ("True", "False")])),
          ("figure", dict(label="e2probs", path=f"{FIG}/fig_e2_probs.png", width=4.8, caption="Maximum entailment and maximum contradiction probability of every E2 claim. Every E2 claim is a verbatim source sentence, so every point not labeled Supported is a judge error."))]
    b += [H2("5.6 The Coverage Cost of Retrieval")]
    b += [P(f"[[tab:cov]] and [[fig:cov]] report the coverage proxy. Zero-context summaries touched {pct(covv('E0'))} "
            f"of source sentences at the 0.6 similarity threshold, excerpt-only RAG summaries {pct(covv('E1'))} "
            f"({fp(covp('E0','E1'))} for the paired difference), RAG summaries with the full note {pct(covv('E1b'))} "
            f"({fp(covp('E0','E1b'))} versus E0; {fp(covp('E1','E1b'))} versus E1), verified summaries {pct(covv('E3'))} "
            f"({fp(covp('E0','E3'))} versus E0; {fp(covp('E1','E3'))} versus E1), and five-sentence extractive summaries "
            f"{pct(covv('E2'))} ({fp(covp('E0','E2'))} versus E0; {fp(covp('E1','E2'))} versus E1). The ordering is the same at "
            f"thresholds of 0.5 and 0.7 for E1 and E1b. Showing the model only three retrieved chunks therefore reduced the "
            f"breadth of the summary, as anticipated in the thesis proposal, and restoring the note (E1b) more than restored it. "
            f"Verification is the surprising case: although E3 summaries are the shortest LLM summaries, their coverage at the "
            f"primary threshold did not fall below the baseline and exceeded that of the E1 drafts they were revised from, because "
            f"the verifier's corrections draw on sentences retrieved from the full note rather than from the three excerpts. At "
            f"the looser 0.5 threshold E3 does cover less than E0 ({pct(cov.loc['E3','coverage_at_0_5'])} versus "
            f"{pct(cov.loc['E0','coverage_at_0_5'])}), so some breadth is lost, but the loss is smaller than the one produced by "
            f"excerpt-only retrieval. The extractive summaries, despite containing far fewer words, covered the note at least as "
            f"broadly as the zero-context LLM because their sentences are the note's most central ones. Among the LLM conditions, "
            f"only excerpt-only retrieval paid for its faithfulness gain with a clear loss of coverage."),
          ("table", dict(label="cov", caption="Coverage proxy per condition: mean share of source sentences whose best cosine similarity to any claim reaches the threshold, and mean best similarity.",
                         columns=["Condition", "Coverage at 0.5", "Coverage at 0.6", "Coverage at 0.7", "Mean best similarity", "Claims per summary"],
                         widths=[1.7, 0.95, 0.95, 0.95, 1.05, 0.9], font=10,
                         rows=[[NAME[c], pct(r.coverage_at_0_5), pct(r.coverage_at_0_6), pct(r.coverage_at_0_7), f3(r.mean_best_similarity), f"{r.n_claims:.1f}"] for c, r in cov.iterrows()],
                         note="Paired tests on coverage at 0.6: " + "; ".join(f"{col[8:].replace('_vs_', ' vs ')} {fp(cov.loc['E0', col])}" for col in cov.columns if col.startswith("p_cov06_")) + ".")),
          ("figure", dict(label="cov", path=f"{FIG}/fig_coverage.png", width=6.3, caption="Coverage proxy at the 0.6 threshold (left) and mean best similarity of source sentences to any claim (right), per condition."))]
    b += [H2("5.7 What Remains Unsupported: Taxonomy and Examples")]
    fixed_rows = []
    if R.ex_fixed is not None:
        for _, r in R.ex_fixed.head(5).iterrows():
            fixed_rows.append([str(int(r.doc_id)), str(r.e0_claim).replace("**", "").replace("\n", " ").strip(), str(r.e1_claim).replace("**", "").replace("\n", " ").strip(), str(r.e0_evidence)[:160].replace("\n", " ") + ("…" if len(str(r.e0_evidence)) > 160 else "")])
    pers_rows = []
    if R.ex_persist is not None:
        seen = set()
        for _, r in R.ex_persist.iterrows():
            key = str(r.e0_claim)[:60]
            if key in seen or r.jaccard < 0.85:
                continue
            seen.add(key); pers_rows.append([str(int(r.doc_id)), str(r.e0_claim).replace("**", "").replace("\n", " ").strip(), r.e0_label, r.e1_label])
            if len(pers_rows) >= 8:
                break
    b += [P(f"[[tab:tax]] and [[fig:tax]] categorize the claims that the judge did not label Supported: {uns_total['E0']} for E0, "
            f"{uns_total['E1']} for E1, {uns_total.get('E1b', 0)} for E1b and {uns_total.get('E3', 0)} for E3. Follow-up and "
            f"scheduling statements are the largest category under every LLM condition and the only one that grows under "
            f"excerpt-only RAG ({R.tax_total('E0','Follow-up or scheduling')} to {R.tax_total('E1','Follow-up or scheduling')}); "
            f"they are also almost never contradicted, because the note usually says nothing about them at all. RAG reduced the "
            f"number of unsupported claims about generic advice ({R.tax_total('E0','Generic advice or patient education')} to "
            f"{R.tax_total('E1','Generic advice or patient education')}), procedures ({R.tax_total('E0','Procedure or treatment')} to "
            f"{R.tax_total('E1','Procedure or treatment')}), findings ({R.tax_total('E0','Findings, exam, labs or imaging')} to "
            f"{R.tax_total('E1','Findings, exam, labs or imaging')}) and medications ({R.tax_total('E0','Medication or dosage')} to "
            f"{R.tax_total('E1','Medication or dosage')}), which are the categories in which specific, retrievable facts occur. This "
            f"is consistent with the mechanism proposed in Section 2.6: excerpts constrain what the model says about the note's "
            f"content, but not what it adds from its parametric knowledge of how patient instructions usually end. Verification "
            f"is the only condition that cut the follow-up category substantially ({R.tax_total('E0','Follow-up or scheduling')} for E0 to "
            f"{R.tax_total('E3','Follow-up or scheduling')} for E3) and the generic-advice category ({R.tax_total('E0','Generic advice or patient education')} "
            f"to {R.tax_total('E3','Generic advice or patient education')}), because a verifier that checks each sentence against the "
            f"note removes exactly the sentences that no passage supports, whereas retrieval only changes what the generator sees."),
          ("table", dict(label="tax", caption="Keyword-assisted categories of claims that the judge did not label Supported, for the LLM conditions. Contradicted counts are the subset labeled Contradicted.",
                         columns=["Category"] + [f"{c} {k}" for c in GEN for k in ("unsupp.", "contra.")], widths=[1.9] + [0.575] * (2 * len(GEN)), font=9, rows=tax_rows)),
          ("figure", dict(label="tax", path=f"{FIG}/fig_taxonomy.png", width=6.0, caption="Number of claims not labeled Supported, by keyword-assisted category, for the zero-context and RAG conditions.")),
          P(f"Qualitative inspection of paired claims illustrates both the promise and the fragility of the metric. "
            f"[[tab:exfixed]] lists E0 claims labeled Contradicted for which the most similar E1 claim in the same document was "
            f"labeled Supported (there were {n_fixed} such pairs). Some are genuine corrections: in document 16 the note states "
            f"\"COMPLICATIONS: None\" and \"Postoperative course was uncomplicated\", the E0 phrasing \"The procedure went smoothly, "
            f"and there were no complications\" was called Contradicted, and the E1 phrasing \"there were no complications from the "
            f"procedures\" Supported; the two claims say the same thing, and the difference lies in the judge, not in the summaries. "
            f"[[tab:expersist]] lists claims that appear almost verbatim under both conditions and are unsupported under both: "
            f"opening sentences that restate the reason for the visit in patient-facing language, standard closing advice, and "
            f"summaries of the patient's history. Several of these are faithful paraphrases of the note that the NLI model "
            f"cannot match to any single sentence, which is the failure mode that the validation study quantified."),
          ("table", dict(label="exfixed", caption="Examples of E0 claims labeled Contradicted whose closest E1 claim in the same document was labeled Supported, with the evidence retrieved for the E0 claim.",
                         columns=["Doc", "E0 claim (Contradicted)", "E1 claim (Supported)", "Evidence retrieved for the E0 claim"], widths=[0.4, 1.9, 1.9, 2.3], font=8.5, align=["center", "left", "left", "left"], rows=fixed_rows)),
          ("table", dict(label="expersist", caption="Examples of claims that appear in nearly identical form under E0 and E1 and are not labeled Supported under either condition.",
                         columns=["Doc", "Claim", "E0 label", "E1 label"], widths=[0.4, 4.3, 0.9, 0.9], font=9, align=["center", "left", "center", "center"], rows=pers_rows))]
    b += [H2("5.8 Summary of Findings")]
    b += [("bullets", [
        f"**H1 (measurability and validity)** is supported in its first half and rejected in its second. The judge produces stable, "
        f"reproducible rates, and zero-context GPT-4o-mini summaries receive a mean UFR of {f3(R.mean('E0','UFR'))} and CR of "
        f"{f3(R.mean('E0','CR'))}. But against medical-expert annotations the same judge reaches a kappa of {f2(cal_all.any_kappa)} "
        f"and an AUROC of {f2(cal_all.auroc_1_minus_p_entail)}, flags four sentences in five, and ranks systems in nearly the "
        f"reverse order of the experts. Its absolute rates cannot be read as hallucination rates.",
        f"**H2 (retrieval grounding)** is supported for contradictions and largely rejected for unsupported facts. Excerpt-only RAG "
        f"reduced the judge's CR by {rel(t10c)} (Holm-adjusted {fp(t10c.p_holm)}) with a small effect size, left UFR unchanged, and "
        f"the effect held only at the default threshold and with three or more evidence sentences; it also reduced coverage of the "
        f"note ({fp(covp('E0','E1'))}). Adding the full note to the excerpts (E1b) lowered UFR modestly ({fp(t1b0u.p)}) without "
        f"changing CR and without the coverage loss.",
        f"**H3 (verification)** is supported for unsupported facts and not for contradictions. Chain-of-Verification on top of RAG "
        f"lowered UFR from {f3(t30u.mean_a)} to {f3(t30u.mean_b)} ({fp(t30u.p)}, d(z) = {f2(t30u.d_z)}), the largest effect among "
        f"the LLM conditions, and improved on its own E1 drafts ({fp(t31u.p)}); CR was unchanged. The gain came from deleting "
        f"unsupported claims and abstaining: E3 summaries are the shortest LLM summaries, yet their coverage of the note at the "
        f"primary threshold matched the baseline ({fp(covp('E0','E3'))}) because corrected claims draw on the full note.",
        f"**The extractive bound** shows that the judge's floor is not zero: verbatim sentences received {pct(share[('E2','Contradicted')])} "
        f"Contradicted labels, two thirds of them on negated sentences and one third attributable to malformed sentence fragments in the "
        f"public corpus.",
        f"**Header lines** in LLM output had inflated the previously reported RAG effect from a {rel(t10c)} to a 32 percent relative "
        f"reduction in CR; correcting the claim set is a prerequisite for any comparison of this kind.",
        f"**The judge mislabels abstentions.** {n_abst_contra} of the {n_abst} \"Not stated in the note\" lines written by E3 were "
        f"labeled Contradicted; had they been counted as claims, verification would have appeared to increase contradictions."])]

    return b


def abstract(R: Results) -> list:
    from mimic_results import Mimic
    M = Mimic(); jr = M.judge_row()
    t10c = R.test("CR", "E0", "E1"); t30u = R.test("UFR", "E0", "E3"); t1b0u = R.test("UFR", "E0", "E1b")
    cal = R.cal_row("all")
    main = M.ok and jr is not None
    txt = (f"Large language models (LLMs) can turn clinical notes into fluent patient-facing summaries, but they also produce statements "
           f"that the note does not support. This thesis builds a reproducible, reference-free pipeline that segments a summary into "
           f"claims, retrieves evidence from the source, labels each claim with a natural language "
           f"inference (NLI) model, and reports an Unsupported Fact Rate (UFR) and a Contradiction Rate (CR) per summary. Five conditions are "
           f"compared: zero-context LLM summarization, retrieval-augmented generation (RAG) with excerpts only or with the full document, RAG "
           f"with Chain-of-Verification, and an extractive summarizer. In a pilot study on {R.n_docs} public MTSamples notes with GPT-4o-mini and "
           f"a small MiniLM judge, excerpt-only retrieval lowered CR but not UFR and reduced coverage, and verification lowered UFR from "
           f"{f3(R.mean('E0','UFR'))} to {f3(R.mean('E3','UFR'))} ({fp(t30u.p)}). Validated against {int(cal.n_summaries)} patient summaries annotated by "
           f"medical experts, that judge flagged "
           f"{pct0(cal.judge_UFR)} of sentences where the experts flagged {pct0(cal.expert_flag_rate)}, with Cohen's kappa {f2(cal.any_kappa)}: it is not a "
           f"valid hallucination detector for this task. "
           + (f"A judge selection study of six off-the-shelf judges and a MedNLI-adapted model then selected a DeBERTa-v3-large NLI model "
              f"(kappa {f2(jr.kappa)}, AUROC {f2(jr.auroc)}); clinical fine-tuning raised MedNLI accuracy by nine points but not "
              f"agreement with the experts. With that judge, a main study repeated the five conditions on {M.n_docs} MIMIC-IV hospital courses with "
              f"an open-weight model running locally, measuring omission against the clinicians' own discharge instructions. Retrieval, retrieval "
              f"with the whole course and verification each lowered UFR from {f3(M.mean('E0','UFR'))} to about "
              f"{f3((M.mean('E1','UFR') + M.mean('E1b','UFR') + M.mean('E3','UFR')) / 3)}, "
              f"close to the {f3(M.mean('E2','UFR'))} error floor of verbatim extraction, without changing CR; only retrieval with the whole course kept "
              f"the baseline's coverage of the clinician's instructions ({pct0(M.mean('E1b','coverage_ref'))} versus {pct0(M.mean('E1','coverage_ref'))} for "
              f"excerpts only); {M.faithfulness_abstract()}; and the clinicians' own text scored {f3(M.mean('REF','UFR'))} under the same "
              f"judge, because it states what the course does not. " if main else "")
           + f"The thesis concludes that hallucination judges must be validated against expert labels before use, that grounding reduces "
           f"unsupported content to the judge's error floor but only whole-document retrieval avoids omission, and that extractive controls "
           f"and a human reference should accompany any source-faithfulness metric.")
    return [txt]
