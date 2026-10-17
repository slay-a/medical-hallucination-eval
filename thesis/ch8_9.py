"""ch8_9.py — Chapters 8 (Discussion) and 9 (Conclusion and Future Work)."""
import numpy as np
import pandas as pd

from results_loader import Results, f3, f2, pct, pct0, fp, fpn, signed
from mimic_results import Mimic, _csv, NAME as MNAME
from ch_mimic import Data, Flip, pts, ci_pts, ci3, padj, GROUNDED

NAME = {"E0": "E0 full note, no retrieval", "E1": "E1 RAG, excerpts only", "E1b": "E1b RAG, course and excerpts", "E3": "E3 RAG with verification",
        "E2": "E2 extractive"}


def P(t):
    return ("p", t)


def H1(t):
    return ("h1", t)


def H2(t):
    return ("h2", t)


def rel(t):
    return f"{abs(t.rel_change_mean):.0f}%"


def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _flag_share(D, c):
    """Share of a condition's claims that the selected judge flags (pooled over courses)."""
    n_all, sup = D.m(c, "claims_total"), D.m(c, "supported_mean") * D.m(c, "n_docs")
    if not _nan(n_all) and not _nan(sup) and n_all > 0:
        return 1 - sup / n_all
    agr = _csv("mimic_judge_agreement.csv")
    if agr is not None and c in set(agr.condition):
        return float(agr.set_index("condition").loc[c, "flag_rate_first"])
    return np.nan


def blocks(R: Results) -> list:
    b = []
    C = R.conds
    M = Mimic(); jr = M.judge_row(); D = Data()
    main = M.ok and D.ok and jr is not None
    _agr = _csv("mimic_judge_agreement.csv")
    G = _agr.set_index("condition") if _agr is not None and "all" in set(_agr.condition) else None
    _st2 = _csv("mimic_second_judge_tests.csv")
    ST2 = _st2[_st2.measure == "UFR_second"].set_index("comparison") if _st2 is not None else None
    SB2 = _st2[_st2.measure == "UFR_both"].set_index("comparison") if _st2 is not None else None
    aud = _csv("audit_summary.csv")
    A = aud[aud.manual_unsupported.notna()].set_index("condition") if aud is not None and "manual_unsupported" in aud.columns else None
    AD = aud[aud["diff"].notna()].set_index("condition") if aud is not None and "diff" in aud.columns else None
    sbias = _csv("judge_selection_bias.csv")
    def sb(analysis, metric, col="estimate"):
        if sbias is None:
            return np.nan
        r = sbias[(sbias.analysis == analysis) & (sbias.metric == metric)]
        return float(r[col].iloc[0]) if len(r) else np.nan
    k_cv = sb("nested_cv_selection_procedure", "kappa")
    vt = _csv("mimic_variant_tests.csv")
    def vrow(comp, metric, flip=False):
        if vt is None:
            return None
        r = vt[(vt.comparison == comp) & (vt.metric == metric)]
        return (Flip(r.iloc[0]) if flip else r.iloc[0]) if len(r) else None
    rep = _csv("mimic_replicates.csv")
    at = _csv("atomic_results.csv")
    t10u, t10c = R.test("UFR", "E0", "E1"), R.test("CR", "E0", "E1")
    t30u = R.test("UFR", "E0", "E3"); t31u = R.test("UFR", "E1", "E3"); t1b0u = R.test("UFR", "E0", "E1b")
    lc = R.label_counts
    tot = {c: int(lc.loc[c].sum()) for c in C}
    share = {(c, l): lc.loc[c, l] / tot[c] for c in C for l in lc.columns}
    cal_all = R.cal_row("all")
    var_all = R.cal_var[R.cal_var.group == "all"].set_index("variant") if R.cal_var is not None else None
    cov = R.cov.set_index("condition")
    has_ext = {"E1b", "E3"} <= set(C)
    fewer_e3 = int((R.cmp.E3_n_claims < R.cmp.E1_n_claims).sum()) if has_ext else 0
    def covv(c):
        return cov.loc[c, "coverage_at_0_6"] if c in cov.index else float("nan")
    def covp(a, b_):
        col = f"p_cov06_{b_}_vs_{a}"
        return cov.loc["E0", col] if col in cov.columns else float("nan")
    tu = {c: D.T("UFR", f"{c} vs E0") for c in GROUNDED} if D.ok else {}
    tcov = {c: D.T("coverage_ref", f"{c} vs E0") for c in GROUNDED} if D.ok else {}
    lo_u = min(abs(t.delta_mean) for t in tu.values()) if tu else np.nan
    hi_u = max(abs(t.delta_mean) for t in tu.values()) if tu else np.nan

    # ══════════════════════════════════ CHAPTER 8 ══════════════════════════════════
    b += [H1("Chapter 8 Discussion"), H2("8.1 Answers to the Research Questions")]
    rq1 = (f"**RQ1.** Measured by the claim-level NLI judge of the pilot, {pct0(R.mean('E0','UFR'))} of the sentences in full-note "
           f"GPT-4o-mini summaries could not be verified against the note and {pct0(R.mean('E0','CR'))} were labeled as contradicting "
           f"it. The validation study shows that these numbers overstate the problem several-fold: on summaries where medical experts "
           f"flagged 5 to 9 percent of sentences, the same judge flagged 86 to 87 percent. The pilot therefore answers RQ1 only "
           f"qualitatively: the summaries contain unsupported statements, in identifiable categories, but that instrument cannot say "
           f"how many.")
    if main:
        rq1 += (f" The main study answers it with the judge selected in Chapter 6. Under that judge, {pct(M.mean('E0','UFR'))} of the claims "
                f"in full-note summaries written by the local model were not supported by the hospital course and {pct(M.mean('E0','CR'))} "
                f"were contradicted (per-summary means). These are judge-estimated rates from an instrument whose agreement with the "
                f"experts is moderate (kappa {f2(jr.kappa)}"
                + (f", about {f2(k_cv)} when the optimism of the selection is removed, Section 6.5" if not _nan(k_cv) else "")
                + ")")
        if A is not None and "E0" in A.index:
            rq1 += (f"; the manual audit puts the share of E0 claims that are actually unsupported by the course at "
                    f"{pct(A.loc['E0', 'manual_unsupported'])} (95 percent interval {pct(A.loc['E0', 'manual_unsupported_lo'])} to "
                    f"{pct(A.loc['E0', 'manual_unsupported_hi'])}), against the {pct(A.loc['E0', 'judge_flag_share_pipeline'])} that the judge flags")
        rq1 += (f". The clinicians' own instructions received a judge-estimated UFR of {f3(M.mean('REF','UFR'))} because they state what "
                f"the course does not, and the verbatim extracts of E2 received {f3(M.mean('E2','UFR'))}, which is judge error on text "
                f"that is supported by construction. Neither value is a bound on the rates of the generated summaries.")
    rq2 = ("**RQ2.** The statements that the judges cannot support fall predominantly into follow-up instructions, generic advice, "
           "restatements of history and paraphrased findings. Two of these categories, follow-up and advice, are largely extrinsic: "
           "the note says nothing about them, the model supplies them from its knowledge of what patient instructions contain, and "
           "retrieval does not remove them because they are not generated from the retrieved text. The pilot judge also erred in "
           "three characteristic ways: it could not match a patient-facing paraphrase that aggregates several sentences of the note "
           "to any single sentence, it treated negated symptom lists as contradictions, and it inherited sentence fragments from the "
           "corpus's line-break artifact.")
    if main:
        ty = _csv("audit_types.csv")
        if ty is not None and len(ty):
            tt = ty.groupby("type", dropna=False).n.sum().sort_values(ascending=False)
            from ch_mimic import AUDIT_TYPE
            rq2 += (f" In the main study the manual audit found the same pattern on the local model's summaries: the most frequent "
                    f"problems among unsupported claims were {_join([f'{AUDIT_TYPE.get(k, str(k))} ({pct0(v / tt.sum())})' for k, v in tt.head(2).items()])}.")
        rq2 += (" Questions about warning signs, which a hospital course rarely states, drew the most abstentions and the highest "
                "judge-estimated rates in the question-answering pilot (Section 7.11)"
                + (f", and the second judge confirmed {'none' if G.loc['E2', 'first_flags_confirmed'] == 0 else pct0(G.loc['E2', 'first_flags_confirmed'])} "
                   f"of the selected judge's flags on verbatim sentences, which places that error in the judge rather than in the text "
                   f"(Section 7.10)." if G is not None and "E2" in G.index else "."))
    rq3 = (f"**RQ3.** In the pilot, under a judge later shown to be invalid for this task, excerpt-only RAG reduced contradictions "
           f"by about a quarter and unsupported facts not at all, while covering {pct(covv('E1'))} instead of {pct(covv('E0'))} of the "
           f"note's sentences; retrieval with the full note lowered the unsupported fact rate by about {abs(t1b0u.rel_change_mean):.0f} "
           f"percent and verification by about {abs(t30u.rel_change_mean):.0f} percent relative to E0. These pilot effects describe how "
           f"that judge reacted and are not interpreted further.")
    if main and tu:
        rq3 += (f" In the main study, excerpt-only retrieval, retrieval with the whole course and verification lowered the "
                f"judge-estimated UFR by between {lo_u:.3f} and {hi_u:.3f} relative to E0 ("
                + "; ".join(f"{c} {signed(t.delta_mean)}, 95 percent interval {ci3(t.ci95_low, t.ci95_high)}" for c, t in tu.items())
                + f"; all {padj(max(D.holm_p(t) for t in tu.values()))}), without a significant change in the contradiction rate, and "
                  f"the three grounded conditions do not differ significantly from one another. The reductions hold at a second "
                  f"threshold and under a second judge (Section 7.10)")
        if rep is not None and len(rep) and "kind" in rep.columns:
            rq3 += " and in each of three generations on 40 courses (Section 7.8)"
        rq3 += (". Part of the reduction comes from saying less: the grounded summaries contain fewer claims, and Section 7.5 "
                "separates the contributions of length and of the citation instruction. The cost is omission for the conditions that "
                "withhold the course: E1 and E3 covered fewer of the clinician's sentences than E0")
        if "E1b" in tcov and tcov["E1b"] is not None:
            rq3 += (f", whereas E1b achieved coverage similar to E0, with no statistically significant difference detected "
                    f"({pts(tcov['E1b'].delta_mean)}, 95 percent interval {ci_pts(tcov['E1b'].ci95_low, tcov['E1b'].ci95_high)})")
        rq3 += "."
    b += [P(rq1), P(rq2), P(rq3)]
    if main and tu:
        t31m = D.T("UFR", "E3 vs E1"); tcr1 = D.T("CR", "E1 vs E0")
        s31 = ST2.loc["E3 vs E1"] if ST2 is not None and "E3 vs E1" in ST2.index else None
        nocite = vrow("E1/nocite vs E0/base", "UFR")
        b += [P(f"**Hypotheses.** H1, that full-note summaries contain a measurable share of unsupported statements and that a judge "
                f"can detect them with agreement substantially better than chance, holds in its first half: the judge-estimated share in "
                f"the main study is {pct(M.mean('E0','UFR'))} of claims"
                + (f", and the manual audit confirms unsupported content in E0 summaries" if A is not None and "E0" in A.index else "")
                + f". Its second half is rejected for the pilot judge (kappa {f2(cal_all.any_kappa)}) and holds only moderately for the "
                f"selected judge (kappa {f2(jr.kappa)}, an optimistic figure). H2, that retrieval reduces unsupported content, is "
                f"supported for the judge-estimated UFR in the main study ({padj(D.holm_p(tu['E1']))} for E1)"
                + (f", including when the citation instruction is removed ({signed(nocite.delta_mean)} relative to E0, {fp(nocite.p)})"
                   if nocite is not None and nocite.p < 0.05 and nocite.delta_mean < 0 else "")
                + f", but not for contradictions ({padj(D.holm_p(tcr1))}), which were rare under every condition; in the pilot neither "
                f"rate changed significantly after the adjustment. H3, that verification adds to retrieval, is not supported under the selected judge in the main study "
                f"(E3 versus E1, {fp(t31m.p)})"
                + (f"; under the second judge E3 has the lowest UFR of the LLM conditions ({f3(s31.mean_b)} against {f3(s31.mean_a)} for E1, "
                   f"{fp(s31.p)}), so the evidence on H3 depends on the judge and remains open." if s31 is not None else "."))]

    b += [H2("8.2 Why the Judge Disagrees with the Experts")]
    b += [P("Four causes are consistent with the pilot judge's disagreement with the experts; the evidence for each differs. The first "
            "is a *granularity mismatch*. A patient-facing sentence such as \"You came in because of stomach pain after meals and "
            "foul-smelling urine, and you were found to have a urinary infection and reflux\" compresses the chief complaint, the "
            "history and the assessment of a note into one sentence. No single source sentence entails it, so the maximum over three "
            "retrieved sentences is low, and concatenating three or five sentences helps only marginally because the model was trained "
            "on single-sentence premises [[cite:bowman2015,williams2018]]. Laban et al. showed that sentence-level NLI works when summary "
            "sentences are roughly co-extensive with source sentences [[cite:laban2022]]; patient-facing paraphrase violates that "
            "assumption systematically. The second is *model capacity and domain*: the pilot judge has six layers and was fine-tuned on "
            "general-domain pairs, while the TRUE meta-evaluation found that reliable consistency judgments required much larger models "
            "trained on diverse consistency data [[cite:honovich2022]], and Romanov and Shivade documented the loss of accuracy of "
            "general NLI models on clinical language, abbreviations and negation [[cite:romanov2018]]. The third is the *decision rule*: "
            "taking the maximum contradiction probability over several evidence sentences turns any spurious contradiction into the "
            "claim's label. The fourth is the *definition of the target*: the UFR counts every claim that is not entailed, including "
            "appropriate extrinsic advice, whereas the experts marked unsupported facts, so the two instruments never measured exactly "
            "the same construct.")]
    jc = _csv("judge_candidates.csv")
    if jc is not None:
        AA = jc[jc.group == "all"].set_index(["judge", "mode"])
        def jrow(j, m):
            return AA.loc[(j, m)] if (j, m) in AA.index else None
        mini, deb, med = jrow("minilm_nli", "top3"), jrow("deberta_large_nli", "doc"), jrow("mednli_deberta_large", "doc")
        ftj = M.finetune_summary()
        txt = (f"The judge selection study tests the second cause directly and the first indirectly. Replacing the six-layer "
               f"cross-encoder with a 24-layer DeBERTa-v3-large model trained on five NLI and fact-verification datasets raised kappa "
               f"from {f2(mini.kappa)} to {f2(deb.kappa)} and AUROC from {f2(mini.auroc)} to {f2(deb.auroc)}, and whole-course evidence "
               f"was better than three retrieved sentences, so capacity matters. ")
        if med is not None and ftj is not None and "zero_shot_dev_accuracy" in ftj:
            txt += (f"Adapting the same model to clinical language with MedNLI raised its MedNLI accuracy from "
                    f"{pct(ftj['zero_shot_dev_accuracy'])} to {pct(ftj['dev_accuracy'])} but not its agreement with the experts (kappa "
                    f"{f2(med.kappa)} against {f2(deb.kappa)}). This is consistent with granularity rather than vocabulary limiting "
                    f"agreement, but it does not establish it")
            if at is not None and len(at):
                a_ = at[(at.analysis == "expert_agreement_atomic_min") & (at.group == "all")]
                s_ = at[(at.analysis == "expert_agreement_sentence") & (at.group == "all")]
                if len(a_) and len(s_):
                    ka, ks = float(a_.kappa.iloc[0]), float(s_.kappa.iloc[0])
                    txt += (f". Scoring atomic facts instead of sentences, which should help if granularity were the limit, "
                            + (f"raised kappa from {f2(ks)} to {f2(ka)} (Section 6.6), which supports the explanation"
                               if ka > ks + 0.02 else
                               f"gave a kappa of {f2(ka)} against {f2(ks)} (Section 6.6), so splitting sentences with the same judge "
                               f"does not remove the limitation, and the granularity explanation remains a plausible hypothesis rather "
                               f"than a finding"))
            txt += ". "
        txt += (f"Even the selected judge flags {pct0(jr.flag_rate) if jr is not None else pct0(deb.flag_rate)} of sentences where the "
                f"experts flag {pct0(deb.expert_rate)} and misses about {100 - round(100 * (jr.recall if jr is not None else deb.recall))} "
                f"of every 100 expert-flagged sentences, so every main-study rate is an estimate from a moderately valid instrument.")
        b += [P(txt)]

    # ── 8.3 judge errors and comparisons
    if main:
        b += [H2("8.3 What the Judge's Errors Mean for the Comparisons")]
        b += [P("Every main-study comparison is a comparison of judge-estimated rates, so it is worth stating exactly how the judge's "
                "errors enter it. Let f be the share of a condition's claims that the judge flags, p its precision on that condition "
                "(the share of flagged claims that are truly unsupported) and q its false omission rate (the share of accepted claims "
                "that are truly unsupported). The true unsupported share is then π = q + f (p − q). If p and q were the same in two "
                "conditions, the true difference would be Δπ = (p − q) Δf. The judge would preserve the direction of a difference, "
                "but because p − q is less than one, the true difference would be smaller than the judge-estimated one, in absolute "
                "and in relative terms. If p or q differ between the conditions, for example because the judge falsely flags generic "
                "advice more often in long summaries, a judge-estimated difference can be larger or smaller than the true one, and a "
                "small difference can even reverse. Differences between conditions in the judge's error rates are therefore the main "
                "threat to the comparisons, and they are what the manual audit measures.")]
        Se, Sp = float(jr.recall), float(jr.specificity) if "specificity" in jr.index else np.nan
        fs = {c: _flag_share(D, c) for c in ("E0", "E1", "E1b", "E3")}
        rg = {c: (f + Sp - 1) / (Se + Sp - 1) for c, f in fs.items() if not _nan(f)}
        if rg and not _nan(Sp):
            neg = [c for c, v in rg.items() if v < 0]
            b += [P(f"The error rates measured on the expert set do not transfer to this generator. On the expert set the selected judge "
                    f"has a sensitivity of {f2(Se)} and a specificity of {f2(Sp)}. Applying them to the main study with the Rogan-Gladen "
                    f"estimator, π = (f + specificity − 1) / (sensitivity + specificity − 1) [[cite:rogan1978]], gives "
                    + "; ".join(f"{c} {f3(v)} from a flag share of {pct(fs[c])}" for c, v in rg.items())
                    + (f". {'All' if len(neg) == len(rg) else _join(neg) + (' give' if len(neg) > 1 else ' gives')} "
                       f"{'of these estimates are' if len(neg) == len(rg) else ''} negative, which is impossible: the judge flags fewer "
                       f"of the local model's claims than it would flag falsely at its expert-set specificity, so its specificity on "
                       f"these summaries must be higher than on the expert set. The validation therefore shows that the judge ranks "
                       f"sentences moderately well, but it does not supply the error rates needed to convert the main study's flag "
                       f"shares into true rates; only an audit of the main study's own outputs can." if neg else
                       ". The estimates assume that the expert-set error rates hold for this generator, which the audit tests."))]
        if A is not None and AD is not None and len(AD):
            conf = [k.split(" minus ")[0] for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in AD.index and AD.loc[k, "diff_hi"] < 0]
            pq = "; ".join(f"{c} p = {f2(A.loc[c, 'precision'])}, q = {f2(A.loc[c, 'false_omission'])}" for c in ("E0", "E1", "E1b", "E3") if c in A.index)
            b += [P(f"The audit gives these quantities per condition ({pq}; Section 7.9). "
                    + (f"For {_join(conf)} the manually confirmed reduction relative to E0 has an interval below zero, so the direction "
                       f"of the judge's comparison is confirmed by manual reading; " if conf else
                       "No manually confirmed reduction relative to E0 has an interval below zero, so the audit, at its sample size, "
                       "does not confirm the direction of the judge's comparisons; ")
                    + "the size of the confirmed differences, rather than the judge's, is the better estimate of the true effect.")]
        among = [D.T("UFR", "E1b vs E1"), D.T("UFR", "E3 vs E1")]
        among = [t for t in among if t is not None]
        if among:
            b += [P(f"The same reasoning explains why Chapter 7 does not rank the grounded conditions against one another. Their "
                    f"judge-estimated UFRs differ by at most {max(abs(t.delta_mean) for t in among):.3f}, a difference that a change of a "
                    f"few points in the judge's precision or false omission rate between conditions would produce on its own.")]

    # ── 8.4 interpreting retrieval and verification
    b += [H2("8.4 Interpreting the Effects of Retrieval and Verification")]
    b += [P(f"The pilot's RAG effect must be read in the light of Sections 5.4 and 5.5. It is modest (a {rel(t10c)} relative reduction of "
            f"CR, d(z) = {f2(t10c.d_z)}), it disappears when the judge is made more conservative, and part of what it measures is a "
            f"change in how the model writes rather than in what it asserts: RAG summaries are shorter and contain fewer claims and "
            f"fewer generic sentences. The categories in which RAG helped most, medications, procedures and findings, are those for "
            f"which the excerpts contain specific, near-verbatim facts, which is the mechanism by which grounding is expected to work "
            f"[[cite:lewis2020]]; the categories in which it did not help, follow-up instructions and advice, come from the model's "
            f"prior over patient instructions rather than from the input. Verification in the pilot judged about a quarter of the draft "
            f"claims unsupported, removed or replaced them and abstained where a section was left empty, and it was the only condition "
            f"that removed extrinsic follow-up and advice sentences; it depends on the same retrieval as the judge, so a claim whose "
            f"support lies in an unretrieved sentence is deleted rather than confirmed ({fewer_e3} of {R.n_docs} E3 summaries have fewer "
            f"claims than their drafts).")]
    if main and tu:
        short = vrow("E0/short vs E0/base", "UFR")
        lm = {c: vrow(k, "UFR", fl) for c, k, fl in (("E1", "E0/short vs E1/base", True), ("E1b", "E1b/base vs E0/short", False), ("E3", "E3/base vs E0/short", False))}
        nocite = vrow("E1/nocite vs E0/base", "UFR")
        txt = (f"In the main study the three grounded conditions lowered the judge-estimated UFR by similar amounts, and the variants of "
               f"Section 7.5 show how much of that is due to the mechanism under study. ")
        if short is not None:
            beyond = [c for c in GROUNDED if lm[c] is not None and lm[c].p < 0.05 and lm[c].delta_mean < 0]
            txt += (f"Shortening E0 to the length of E1 changed its UFR by {signed(short.delta_mean)} ({fp(short.p)})"
                    + (f", and {_join(beyond)} remained below the length-matched baseline, so their reductions are not a by-product of "
                       f"brevity alone. " if beyond else ", and no grounded condition remained significantly below the length-matched "
                       "baseline, so brevity may account for much of the reduction. "))
        if nocite is not None:
            txt += (f"Without the citation instruction E1 changed UFR by {signed(nocite.delta_mean)} relative to E0 ({fp(nocite.p)}), so "
                    + ("retrieval is associated with fewer unsupported claims without any instruction to cite. " if nocite.p < 0.05 and nocite.delta_mean < 0 else
                       "the reduction cannot be separated from the citation instruction. "))
        txt += ("Because the grounded summaries also contain fewer claims, the most defensible reading is that grounding lowers the share "
                "of claims the judge cannot verify, partly by changing what the model writes and partly by how much it writes, and that "
                "the true size of the reduction is smaller than the judge-estimated one (Section 8.3).")
        b += [P(txt)]
        if "E1b" in tcov and tcov["E1b"] is not None:
            e1c, e3c = tcov.get("E1"), tcov.get("E3")
            b += [P(f"What separates the grounded conditions is omission. E1 and E3 covered {abs(100 * e1c.delta_mean):.1f} and "
                    f"{abs(100 * e3c.delta_mean):.1f} percentage points fewer of the clinician's sentences than E0, while E1b achieved "
                    f"coverage similar to E0. E1b differs from E1 only in also seeing the whole course, so the contrast associates the "
                    f"coverage loss with withholding the course rather than with retrieval. A plausible mechanism is that the retrieval "
                    f"query, built from fixed section topics and the opening of the course, misses passages that the clinician chose to "
                    f"mention; the data show the association, not the mechanism, and they do not show whether the omitted sentences "
                    f"matter clinically. Verification added nothing to coverage: E3 inherits its draft's excerpts, and its abstentions were "
                    f"not associated with lower coverage within E3 (Section 7.4).")]
        s31 = ST2.loc["E3 vs E1"] if ST2 is not None and "E3 vs E1" in ST2.index else None
        b31 = SB2.loc["E3 vs E1"] if SB2 is not None and "E3 vs E1" in SB2.index else None
        if s31 is not None:
            b += [P(f"Whether verification adds to retrieval is the clearest open question. Under the selected judge E3 and E1 do not differ, "
                    f"under the second judge E3 has a lower UFR ({f3(s31.mean_b)} against {f3(s31.mean_a)}, {fp(s31.p)})"
                    + (f", and under the conservative rule that requires both judges the difference is {signed(b31.delta_mean)} ({fp(b31.p)})" if b31 is not None else "")
                    + ". The E3 verifier is the generator itself, and it removes the claims it cannot match to retrieved evidence; the "
                    "two judges may simply disagree about the claims it removes. The question needs a judge whose error rates on these "
                    "outputs are known, which the audit provides for the selected judge only.")]
        cc = {c: (D.m(c, "claims_total"), D.m(c, "cited_claims"), D.m(c, "cite_correct")) for c in ("E1", "E1b")}
        if all(not _nan(v[0]) and v[0] > 0 for v in cc.values()):
            b += [P(f"The citation results qualify the value of citations as a grounding device. E1 cited {pct0(cc['E1'][1] / cc['E1'][0])} of "
                    f"its claims and E1b only {pct0(cc['E1b'][1] / cc['E1b'][0])}; a plausible explanation, consistent with E1b's design, is "
                    f"that E1b writes many statements from parts of the course outside the three excerpts, which have no excerpt to cite. "
                    f"A reviewer can use a citation only when it is present and correct, so citation reliability, {pct0(cc['E1'][2] / cc['E1'][0])} "
                    f"of E1's claims and {pct0(cc['E1b'][2] / cc['E1b'][0])} of E1b's, is the share of a summary that a reviewer could check "
                    f"by following its citations.")]
        b += [P(f"Two references guard against over-reading the rates. The clinicians' own instructions received a judge-estimated UFR of "
                f"{f3(M.mean('REF','UFR'))}, higher than every LLM condition, because they contain medication changes, appointments and "
                f"advice that the hospital course never states: support by the source measures source faithfulness, not clinical quality. "
                f"And the verbatim extracts of E2 received {f3(M.mean('E2','UFR'))}, which shows that the judge errs even on text copied from "
                f"the course; it is a reference for that error, not a floor that the generated summaries cannot go below.")]
        qa = M.qa_tot
        if qa is not None and {"QA-E0", "QA-E1"} <= set(qa.index):
            b += [P(f"The question-answering pilot points the same way from a different angle. Retrieval "
                    f"{'raised' if qa.loc['QA-E1', 'abstention_rate'] > qa.loc['QA-E0', 'abstention_rate'] else 'changed'} abstention and "
                    f"lowered the judge-estimated UFR of the answers given, partly by making the model say less, and questions about "
                    f"warning signs, which the course rarely states, were answered from the model's general knowledge. Abstention, not "
                    f"retrieval, is the appropriate response to such questions, but the pilot has no gold answers and does not show "
                    f"whether the answers given were correct.")]

    # ── 8.5 comparison of the approaches
    b += [H2("8.5 Comparison of the Approaches")]
    if main and tu:
        cut = {c: abs(tu[c].rel_change_mean) for c in GROUNDED}
        cc = {c: (D.m(c, "claims_total"), D.m(c, "cited_claims"), D.m(c, "cite_correct")) for c in ("E1", "E1b")}
        cmp_text = (f"[[tab:compare]] summarizes the strengths, weaknesses and appropriate uses of the five conditions in the light of both "
                    f"studies. No approach dominates on every criterion. The full-note LLM without retrieval is the most fluent and, with "
                    f"E1b, the most complete, and in both studies it has the highest judge-estimated rate of unsupported claims. On the "
                    f"hospital courses every form of grounding lowered that rate by {min(cut.values()):.0f} to {max(cut.values()):.0f} "
                    f"percent under the judge. Excerpt-only RAG and verification paid for the reduction with coverage. RAG with the whole "
                    f"course in view reduced the judge-estimated rate as much as the other grounded conditions while achieving coverage "
                    f"similar to E0, so it is the design these results favor. The recommendation is conditional: it rests on judge-estimated "
                    f"rates for one generator, one corpus and one retrieval design, on a coverage measure scored by the same judge, and on "
                    f"citations that cover only {pct0(cc['E1b'][1] / cc['E1b'][0]) if not _nan(cc['E1b'][0]) else '—'} of E1b's claims. "
                    f"The extractive summarizer cannot fabricate but produces text that no patient should be handed unedited. That the "
                    f"ordering of E0 and the grounded conditions is the same under both judges, at both thresholds and in repeated "
                    f"generations is some assurance that the judges respond to real differences when those differences are large; it does "
                    f"not settle the smaller differences among the grounded conditions.")
    else:
        cmp_text = "[[tab:compare]] summarizes the strengths, weaknesses and appropriate uses of the five conditions."
    def cit_txt(c):
        if not main:
            return "—"
        n_all, n_cit, n_ok = D.m(c, "claims_total"), D.m(c, "cited_claims"), D.m(c, "cite_correct")
        return f"{pct0(n_cit / n_all)} of claims cited, {pct0(n_ok / n_cit) if n_cit else '—'} of citations supported" if not _nan(n_all) and n_all else "—"
    b += [P(cmp_text),
          ("table", dict(label="compare", caption="Comparison of the five conditions across both studies. Pilot values are per-document means under the MiniLM judge on MTSamples; main-study values are judge-estimated per-document means under the selected judge on MIMIC-IV hospital courses; coverage is the judge-estimated share of the clinician's discharge-instruction sentences supported by the summary.",
                         columns=["Approach", "Pilot UFR / CR", "Main study UFR / CR", "Coverage of clinician's sentences", "Strengths", "Weaknesses", "Appropriate use"],
                         widths=[0.85, 0.7, 0.75, 0.75, 1.15, 1.15, 1.15], font=8, align=["left", "center", "center", "center", "left", "left", "left"],
                         rows=[[NAME[c], f"{f3(R.mean(c,'UFR'))} / {f3(R.mean(c,'CR'))}",
                                (f"{f3(M.mean(c,'UFR'))} / {f3(M.mean(c,'CR'))}" if M.ok and c in M.conds else "—"),
                                (pct0(M.mean(c, "coverage_ref")) if M.ok and c in M.conds else "—"), st, wk, use]
                               for c, st, wk, use in [
                                   ("E0", "Fluent, patient-friendly, complete structure; simplest to deploy", "Highest judge-estimated rate of unsupported claims among the LLM conditions; generic advice and follow-up from priors",
                                    "Drafts reviewed and edited by a clinician; low-stakes communication"),
                                   ("E1", f"Lower judge-estimated rate of unsupported claims; {cit_txt('E1')}",
                                    "Loses about a third of the clinician's content; small, threshold-dependent effect in the pilot; crude retrieval query",
                                    "Focused summaries of long notes when omission is acceptable and a reviewer checks completeness"),
                                   ("E1b", "Lower judge-estimated rate of unsupported claims with coverage similar to E0",
                                    f"Few claims carry citations ({cit_txt('E1b')}); still adds extrinsic advice; longest prompts",
                                    "The design these results favor when the note fits the context window, with clinician review"),
                                   ("E3", "Lowest rate under the second judge; removes extrinsic advice in the pilot; explicit abstentions",
                                    "Coverage as low as E1's; abstentions are omissions; about ten model calls per note; benefit over E1 judge-dependent",
                                    "High-stakes communication where omission is safer than invention and a clinician fills gaps"),
                                   ("E2", "Cannot fabricate; no model cost; deterministic", "Unreadable for patients: fragments, abbreviations, no explanation; flagged by the judge even though verbatim",
                                    "Clinician-facing digests; a grounding step before an LLM rewrites the selected sentences")]]))]

    # ── 8.6 findings, explanations and open questions
    if main and tu:
        b += [H2("8.6 Findings, Explanations and Open Questions")]
        b += [P("The discussion has mixed three kinds of statement, which [[tab:findings]] separates: findings, which the data measure "
                "directly; explanations, which are offered for a finding and are tested only in part or not at all; and open questions, "
                "which this thesis cannot answer.")]
        ka = ks = np.nan
        if at is not None and len(at):
            a_ = at[(at.analysis == "expert_agreement_atomic_min") & (at.group == "all")]; s_ = at[(at.analysis == "expert_agreement_sentence") & (at.group == "all")]
            if len(a_) and len(s_):
                ka, ks = float(a_.kappa.iloc[0]), float(s_.kappa.iloc[0])
        t_b = tcov.get("E1b")
        rows = [
            ["Pilot judge", f"Kappa {f2(cal_all.any_kappa)} against the experts; flags {pct0(cal_all.judge_UFR)} of sentences where they flag {pct0(cal_all.expert_flag_rate)}",
             "Granularity mismatch, small capacity, decision rule (capacity tested in Chapter 6)", "—"],
            ["Selected judge", f"Kappa {f2(jr.kappa)}" + (f"; about {f2(k_cv)} under nested cross-validation" if not _nan(k_cv) else ""),
             "Larger model and whole-course evidence help (tested)", "Agreement on other generators and institutions"],
            ["Clinical adaptation", "MedNLI accuracy up nine points; agreement not improved",
             "Granularity rather than vocabulary limits agreement (plausible" + (f"; atomic scoring kappa {f2(ka)} against {f2(ks)}" if not _nan(ka) else "") + ")",
             "What would improve agreement on patient-facing paraphrase"],
            ["Grounding", f"Judge-estimated UFR lower by {lo_u:.3f} to {hi_u:.3f} for E1, E1b and E3 (Holm-adjusted); robust to threshold, second judge" + (" and repeated generation" if rep is not None and len(rep) and "kind" in rep.columns else ""),
             "Focus on retrieved passages; fewer claims; citation instruction (separated in part by Section 7.5)", "True size of the reduction (Section 8.3)"],
            ["Omission", f"E1 and E3 lose coverage; E1b similar to E0 ({pts(t_b.delta_mean) if t_b is not None else '—'})",
             "Withholding the course causes the loss (associated, via E1b against E1)", "Clinical importance of the omitted sentences"],
            ["Verification", "No difference from E1 under the selected judge; lower UFR under the second judge",
             "Verifier removes claims the judges disagree about (untested)", "Whether verification adds to retrieval"],
            ["Citations", f"E1: {cit_txt('E1')}; E1b: {cit_txt('E1b')}",
             "E1b draws on uncited parts of the course (plausible)", "Whether citations speed clinician review"],
            ["Clinician's instructions", f"Highest judge-estimated UFR ({f3(M.mean('REF','UFR'))})",
             "They state content the course does not (supported by the experts' spans)", "—"],
            ["Question answering", "Retrieval raised abstention and lowered the UFR of answers",
             "Warning-sign answers come from general knowledge (plausible)", "Correctness against clinician-written answers"]]
        b += [("table", dict(label="findings", caption="Findings of the thesis, the explanations offered for them and the status of each explanation, and the questions that remain open.",
                             columns=["Topic", "Finding", "Explanation offered (status)", "Open question"],
                             widths=[1.05, 2.05, 1.85, 1.55], font=8.5, align=["left"] * 4, rows=rows))]

    # ── 8.7 threats
    b += [H2("8.7 Threats to Validity and Limitations")]
    jk = (f"kappa {f2(jr.kappa)}, precision {f2(jr.precision)}, recall {f2(jr.recall)}" if jr is not None else "moderate agreement")
    bullets = [
        "**Construct validity of the judge.** The pilot judge agrees with medical experts at near-chance levels, so the pilot's UFR and CR "
        "are indices of how a small NLI model reacts to a summary rather than measurements of hallucination. The main-study judge was "
        f"selected for its agreement with the same experts ({jk}), which is moderate, and every rate in Chapter 7 is judge-estimated. "
        "The second judge and the two combined flagging rules are alternative estimates of the same quantity, not bounds on it, "
        "because the judges can share errors.",
        "**Optimism of the selection and overlap with the main study.** The judge was chosen on the sentences on which its agreement is "
        "reported" + (f"; nested cross-validation puts the procedure's kappa at about {f2(k_cv)} against an apparent {f2(jr.kappa)}" if not _nan(k_cv) and jr is not None else "")
        + ". The doctor-written half of the expert set belongs to the main-study courses, so the judge was selected partly on text "
        "written for the courses it later scores (Section 6.5).",
        "**Transfer of the validation.** The expert annotations cover doctor-written instructions and summaries by GPT-4 and Llama "
        "models, not the local model's output, and the expert-set error rates do not transfer to it (Section 8.3). The manual audit "
        "measures the judge on the local model's output, but with one annotator, the author, who is not a clinician, and with a "
        "sample of 60 courses.",
        "**Sample size and generation variability.** The 110 courses give adequate power for moderate paired effects. Each summary "
        "was generated once at temperature 0.3; repeated generation on 40 courses measures the run-to-run variation (Section 7.8), but "
        "most results rest on one generation.",
        "**External validity of the corpora.** MTSamples documents are sample transcriptions, shorter and more formulaic than hospital "
        "notes, with corrupted line breaks, and the corpus is public and may be in the generator's training data. The main-study courses "
        "come from one institution and one section of the discharge summary, and the clinician's instructions used as the coverage "
        "reference contain content that the course does not, so complete coverage of them is neither attainable nor desirable.",
        "**Two generators of different capability.** The pilot used GPT-4o-mini through an API and the main study a four-bit, "
        "seven-billion-parameter open-weight model, the largest that the data use agreement and the hardware allowed. Results are "
        "compared across the studies qualitatively, and each study used one prompt per condition, one retriever and one verifier.",
        "**Heuristic components.** The header filter, the abstention rule and the citation parser are rule-based. An earlier version of "
        "the main-study parser mishandled citation markers and merged headers (Section 3.5); the corrected parser is used throughout, "
        "but rule-based components remain a source of error.",
        "**Measures scored by the same judge.** Coverage and citation accuracy run the selected judge in new directions (summary as "
        "premise, instruction sentence as hypothesis; cited excerpt as premise, claim as hypothesis) whose agreement with experts was not "
        "validated; the audit verified a sample of citations manually.",
        "**Verification design.** E3 uses one verifier (the generator itself, at temperature 0), one prompt and the judge's own retriever, "
        "and the abstention rule was written after the mislabeling of abstentions was observed in the pilot.",
        "**Question-answering pilot.** Three fixed questions without clinician-written gold answers; agreement with the clinician's "
        "instructions is a proxy for correctness."]
    b += [("bullets", bullets)]

    # ── 8.8 implications
    b += [H2("8.8 Implications for the Design and Evaluation of Clinical Summarization Systems")]
    b += [("numbers", [
        "**Validate the evaluator before the system, and again on the system's own outputs.** An automatic hallucination metric should "
        "be reported with its agreement with clinician annotations on the target domain, corrected for the optimism of choosing it, "
        "and with an audit of the outputs it scores, because error rates measured on other generators do not transfer.",
        "**Choose the judge by capacity and evidence mode, not by aggregation tweaks.** Changing how a small model's evidence was "
        "aggregated changed little; a larger consistency-trained NLI model with the whole document as evidence raised kappa markedly.",
        "**Report paired differences with intervals and read small differences with caution.** A judge preserves the direction of a "
        "large difference when its error rates are similar across systems, but a small difference can come from a difference in error "
        "rates alone.",
        "**Report counts, length and coverage with every rate.** A lower share of unsupported claims can come from writing fewer "
        "claims; supported and unsupported counts, length and coverage of a clinician-written reference make that visible.",
        "**Separate the variables that differ between conditions.** Length-matched and instruction-matched variants are cheap and show "
        "how much of an effect belongs to the mechanism under study.",
        "**Use an extractive and a human reference as references, not as bounds.** Verbatim extracts show how the judge scores text "
        "that is supported by construction, and the clinician's own text shows what the metric measures; neither is a floor or a "
        "ceiling for a generated summary.",
        "**Parse citations before splitting sentences, and report coverage as well as accuracy.** A citation is useful only when it is "
        "present and correct, and naive sentence splitting misattributes many markers.",
        "**Keep protected text local and check the repository.** An open-weight model on a laptop made a study on real hospital "
        "courses possible without any data leaving the machine, and an automated scan confirms that no protected text was published."])]

    # ══════════════════════════════════ CHAPTER 9 ══════════════════════════════════
    b += [H1("Chapter 9 Conclusion and Future Work"), H2("9.1 Summary")]
    s1 = (f"This thesis set out to measure and reduce hallucinations in LLM-generated patient-facing summaries of clinical notes. It "
          f"built a reproducible claim-level evaluation pipeline from open components, compared five summarization conditions on "
          f"{R.n_docs} public MTSamples notes, validated the pipeline's judge against medical-expert annotations, found it invalid for this "
          f"task, selected a better judge by the same validation, and repeated the comparison on {M.n_docs if M.ok else 110} MIMIC-IV hospital "
          f"courses with an open-weight model running on the author's computer, measuring omission against the discharge instructions "
          f"the clinicians wrote.")
    s2 = ""
    if main and tu:
        s2 = (f"Its first contribution is methodological. A small general-domain NLI judge applied at sentence granularity agreed "
              f"with the experts at chance level (kappa {f2(cal_all.any_kappa)}); a larger NLI model with whole-course evidence agreed "
              f"moderately (kappa {f2(jr.kappa)}" + (f", about {f2(k_cv)} after correcting for the selection" if not _nan(k_cv) else "")
              + "), and its error rates on the expert set did not transfer to the local model's summaries. Its second contribution "
              f"is empirical and conditional on that judge. Every form of grounding lowered the judge-estimated rate of unsupported "
              f"claims by {lo_u:.3f} to {hi_u:.3f}, partly by making the summaries shorter; only retrieval with the whole course in view "
              f"did so with coverage similar to the full-note baseline; whether verification adds to retrieval depends on the judge; "
              f"and the clinicians' own instructions scored worse than every model under a metric of support by the source, which "
              f"measures faithfulness to the course rather than the quality of a summary.")
    b += [P(s1)] + ([P(s2)] if s2 else [])
    b += [H2("9.2 Key Findings")]
    kf = [f"**Pilot judge.** Against 1,781 expert-annotated sentences, the pilot judge's precision was {pct0(cal_all.any_precision)}, recall "
          f"{pct0(cal_all.any_recall)}, kappa {f2(cal_all.any_kappa)} and AUROC {f2(cal_all.auroc_1_minus_p_entail)}; no evidence-aggregation "
          f"rule raised kappa above {f2(var_all.best_kappa.max()) if var_all is not None else '—'}, so its pilot rates are not hallucination rates.",
          "**Evaluation artifacts.** Counting markdown header lines as claims and labeling abstentions as contradictions distort "
          "claim-level rates; both, and the misattribution of citation markers in the main study, were corrected."]
    if main and tu:
        kf += [f"**Judge selection.** DeBERTa-v3-large NLI with whole-course evidence reached kappa {f2(jr.kappa)} and AUROC {f2(jr.auroc)}"
               + (f" (about {f2(k_cv)} under nested cross-validation)" if not _nan(k_cv) else "")
               + "; MedNLI fine-tuning raised MedNLI accuracy by nine points but not agreement with the experts.",
               f"**Grounding.** On the hospital courses, E1, E1b and E3 lowered the judge-estimated UFR from {f3(M.mean('E0','UFR'))} to "
               f"{f3(M.mean('E1','UFR'))}, {f3(M.mean('E1b','UFR'))} and {f3(M.mean('E3','UFR'))} (all {padj(max(D.holm_p(t) for t in tu.values()))}), "
               f"without changing CR; the three do not differ significantly from one another, and part of the reduction comes from shorter summaries.",
               (f"**Coverage.** E1 and E3 covered fewer of the clinician's sentences than E0; E1b achieved coverage similar to E0, with no "
                f"statistically significant difference detected ({pts(tcov['E1b'].delta_mean)}, 95 percent interval "
                f"{ci_pts(tcov['E1b'].ci95_low, tcov['E1b'].ci95_high)})." if tcov.get("E1b") is not None else ""),
               (f"**Citations.** E1 cited {pct0(D.m('E1', 'cited_claims') / D.m('E1', 'claims_total'))} of its claims and E1b "
                f"{pct0(D.m('E1b', 'cited_claims') / D.m('E1b', 'claims_total'))}; {pct0(D.m('E1', 'cite_correct') / D.m('E1', 'cited_claims'))} of "
                f"E1's citations were supported by the cited excerpts." if not _nan(D.m('E1', 'claims_total')) else ""),
               "**Robustness.** The reductions relative to E0 hold at a second threshold and under a second judge"
               + (" and in repeated generations" if rep is not None and len(rep) and "kind" in rep.columns else "")
               + "; verification is better than retrieval alone only under the second judge."]
        if A is not None and AD is not None and "E1 minus E0" in AD.index:
            r = AD.loc["E1 minus E0"]
            kf.append(f"**Manual audit.** The manually confirmed difference between E1 and E0 was {pts(r['diff'])} (95 percent interval "
                      f"{ci_pts(r.diff_lo, r.diff_hi)}), against {pts(r.judge_diff)} by the judge.")
        if M.qa_tot is not None and {"QA-E0", "QA-E1"} <= set(M.qa_tot.index):
            kf.append(f"**Question answering (exploratory).** Retrieval raised abstention from {pct0(M.qa_tot.loc['QA-E0','abstention_rate'])} to "
                      f"{pct0(M.qa_tot.loc['QA-E1','abstention_rate'])} and lowered the judge-estimated UFR of the answers given; warning-sign "
                      f"questions are rarely answerable from the course.")
    b += [("numbers", [k for k in kf if k])]
    b += [H2("9.3 Future Work")]
    b += [("numbers", [
        "**Clinician annotation of the generator's own output.** The audit of this thesis was performed by one non-clinician annotator; "
        "two clinicians labeling the same sample would give agreement between annotators and clinically validated error rates for the "
        "selected judge on these outputs.",
        "**External validation of the judge.** Clinician-labeled statements about other patients, such as the VeriFact-BHC annotations "
        "of MIMIC-III hospital courses [[cite:chung2026verifact]], would test the selected judge outside the courses it was chosen on.",
        "**Better judges for patient-facing paraphrase.** LLM-based judges calibrated against expert labels, alignment models trained "
        "on clinical consistency data, and decomposition designs that keep the context of each fact are natural candidates for the same "
        "selection protocol.",
        "**Larger generators and more generations.** A workstation-class GPU would allow open-weight models of 30 to 70 billion "
        "parameters, several generations for every course, and more institutions and note types.",
        "**A clinician study of omission and of the question-answering task.** Clinicians rating the importance of omitted sentences, "
        "and clinician-written gold answers for a broader question set, would turn the coverage measure and the question-answering "
        "pilot into validated measures."])]
    return b
