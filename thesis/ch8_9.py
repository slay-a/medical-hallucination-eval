"""ch8_9.py: Chapters 8 (Discussion) and 9 (Conclusion and Future Work)."""
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
    rq1 = (f"**RQ1.** The pilot's judge answers RQ1 only qualitatively: it labeled {pct0(R.mean('E0','UFR'))} of the sentences in "
           f"full-note GPT-4o-mini summaries unsupported but overstated the problem several-fold against the experts (Section 5.3), "
           f"so the summaries contain unsupported statements in identifiable categories and that instrument cannot say how many.")
    if main:
        rq1 += (f" Under the selected judge, {pct(M.mean('E0','UFR'))} of the claims in the local model's full-note summaries were "
                f"not supported by the hospital course and {pct(M.mean('E0','CR'))} contradicted it, judge-estimated rates from an "
                f"instrument of moderate validity (kappa {f2(jr.kappa)}"
                + (f", about {f2(k_cv)} once the optimism of the selection is removed" if not _nan(k_cv) else "") + ")")
        if A is not None and "E0" in A.index:
            rq1 += (f"; by manual reading the share of E0 claims actually unsupported is {pct(A.loc['E0', 'manual_unsupported'])} "
                    f"({pct(A.loc['E0', 'manual_unsupported_lo'])} to {pct(A.loc['E0', 'manual_unsupported_hi'])})")
        rq1 += (f". The clinicians' own instructions score {f3(M.mean('REF','UFR'))} because they state what the course does not, "
                f"and E2's verbatim extracts {f3(M.mean('E2','UFR'))}, which is judge error on supported text; neither bounds the "
                f"rates of the generated summaries.")
    rq2 = ("**RQ2.** The statements the judges cannot support are mostly follow-up instructions, generic advice, restatements of "
           "history and paraphrased findings. Follow-up and advice are largely extrinsic: the note says nothing about them, the "
           "model supplies them from its knowledge of what patient instructions contain, and retrieval does not remove them. The "
           "pilot judge erred in three characteristic ways: it could not match a paraphrase that aggregates several sentences of "
           "the note to any single sentence, it treated negated symptom lists as contradictions, and it inherited sentence fragments "
           "from the corpus.")
    if main:
        ty = _csv("audit_types.csv")
        typed = ty[ty.type.notna() & ~ty.type.astype(str).str.strip().isin(["", "nan"])] if ty is not None and len(ty) else None
        if typed is not None and len(typed) and typed.n.sum() >= 0.6 * ty.n.sum():
            tt = typed.groupby("type").n.sum().sort_values(ascending=False)
            from ch_mimic import AUDIT_TYPE
            rq2 += (f" The manual audit found the same pattern on the local model's summaries: the most frequent problems were "
                    f"{_join([f'{AUDIT_TYPE.get(k, str(k))} ({pct0(v / tt.sum())})' for k, v in tt.head(2).items()])}.")
        elif A is not None and "pooled" in A.index and not _nan(A.loc["pooled", "partly_share_of_unsupported"]):
            rq2 += (f" In the manual audit, {pct0(A.loc['pooled', 'partly_share_of_unsupported'])} of the claims labeled unsupported were "
                    f"partly supported sentences rather than whole fabrications (Section 7.9).")
    rq3 = ("**RQ3.** The pilot's effects (Section 5.7) describe how a judge later shown to be invalid reacted and are not "
           "interpreted further.")
    if main and tu:
        rq3 += (f" In the main study, retrieval with excerpts, retrieval with the whole course and verification lowered the "
                f"judge-estimated UFR by {lo_u:.3f} to {hi_u:.3f} relative to E0 (all {padj(max(D.holm_p(t) for t in tu.values()))}), "
                f"without changing the contradiction rate; the three grounded conditions do not differ significantly from one "
                f"another, and the reductions hold at a second threshold, under a second judge (Section 7.10)"
                + (" and in each of three generations (Section 7.8)" if rep is not None and len(rep) and "kind" in rep.columns else "")
                + ". Part of the reduction comes from saying less, and Section 7.5 separates length and the citation instruction "
                "from retrieval. The cost is omission for the conditions that withhold the course: E1 and E3 covered fewer of "
                "the clinician's sentences than E0")
        if "E1b" in tcov and tcov["E1b"] is not None:
            rq3 += ", whereas E1b achieved coverage similar to E0, with no statistically significant difference detected (Section 7.4)"
        rq3 += "."
    b += [P(rq1), P(rq2), P(rq3)]
    if main and tu:
        t31m = D.T("UFR", "E3 vs E1"); tcr1 = D.T("CR", "E1 vs E0")
        s31 = ST2.loc["E3 vs E1"] if ST2 is not None and "E3 vs E1" in ST2.index else None
        nocite = vrow("E1/nocite vs E0/base", "UFR")
        b += [P("**Hypotheses.** H1 holds in its first half (full-note summaries contain a measurable share of unsupported "
                "statements" + (", confirmed by the manual audit" if A is not None and "E0" in A.index else "") + ") and only "
                "moderately in its second, since the pilot judge agrees with experts at chance and the selected judge moderately. "
                f"H2 is supported for the judge-estimated UFR ({padj(D.holm_p(tu['E1']))} for E1"
                + (", also without the citation instruction" if nocite is not None and nocite.p < 0.05 and nocite.delta_mean < 0 else "")
                + f") but not for contradictions ({padj(D.holm_p(tcr1))}), which were rare everywhere. H3, that verification adds "
                f"to retrieval, is not supported under the selected judge (E3 versus E1, {fp(t31m.p)})"
                + (f" and is under the second judge ({fp(s31.p)}), so it depends on the judge and remains open." if s31 is not None else "."))]

    b += [H2("8.2 Why the Judge Disagrees with the Experts")]
    txt = ("Four causes are consistent with the pilot judge's disagreement with the experts. A *granularity mismatch*: a "
           "patient-facing sentence compresses the chief complaint, the history and the assessment of a note into one sentence that "
           "no single source sentence entails, and sentence-level NLI works only when summary sentences are roughly co-extensive "
           "with source sentences [[cite:laban2022]], which patient-facing paraphrase violates systematically. *Model capacity and "
           "domain*: a six-layer model fine-tuned on general-domain pairs, whereas reliable consistency judgments required much "
           "larger models trained on diverse consistency data [[cite:honovich2022]], and general NLI models lose accuracy on "
           "clinical language and negation [[cite:romanov2018]]. The *decision rule*: a maximum over evidence sentences turns any "
           "spurious contradiction into the claim's label. And the *definition of the target*: UFR counts every claim that is not "
           "entailed, including appropriate extrinsic advice, whereas the experts marked unsupported facts. ")
    jc = _csv("judge_candidates.csv")
    if jc is not None:
        AA = jc[jc.group == "all"].set_index(["judge", "mode"])
        def jrow(j, m):
            return AA.loc[(j, m)] if (j, m) in AA.index else None
        mini, deb, med = jrow("minilm_nli", "top3"), jrow("deberta_large_nli", "doc"), jrow("mednli_deberta_large", "doc")
        ftj = M.finetune_summary()
        txt += (f"The selection study tests capacity directly: a 24-layer DeBERTa-v3-large model with whole-course evidence raised "
                f"kappa from {f2(mini.kappa)} to {f2(deb.kappa)}. ")
        if med is not None and ftj is not None and "zero_shot_dev_accuracy" in ftj:
            txt += (f"Adapting it to clinical language raised MedNLI accuracy but not agreement (kappa {f2(med.kappa)}), which is "
                    f"consistent with granularity rather than vocabulary limiting agreement")
            if at is not None and len(at):
                a_ = at[(at.analysis == "expert_agreement_atomic_min") & (at.group == "all")]
                s_ = at[(at.analysis == "expert_agreement_sentence") & (at.group == "all")]
                if len(a_) and len(s_):
                    ka, ks = float(a_.kappa.iloc[0]), float(s_.kappa.iloc[0])
                    txt += (f"; yet scoring atomic facts instead of sentences " + (f"raised kappa to {f2(ka)} (Section 6.6), which supports that explanation"
                            if ka > ks + 0.02 else f"gave kappa {f2(ka)} against {f2(ks)} (Section 6.6), so the explanation remains plausible rather than established"))
            txt += ". "
        txt += (f"Even the selected judge flags {pct0(jr.flag_rate) if jr is not None else pct0(deb.flag_rate)} of sentences where the "
                f"experts flag {pct0(deb.expert_rate)} and misses about {100 - round(100 * (jr.recall if jr is not None else deb.recall))} "
                f"of every 100 expert-flagged sentences, so every main-study rate is an estimate from a moderately valid instrument.")
    b += [P(txt)]

    # ── 8.3 judge errors and comparisons
    if main:
        b += [H2("8.3 What the Judge's Errors Mean for the Comparisons")]
        txt = ("Let f be the share of a condition's claims the judge flags, p its precision on that condition and q its false omission "
               "rate (the share of accepted claims that are truly unsupported). The true unsupported share is π = q + f (p − q), so if "
               "p and q were the same in two conditions the true difference would be (p − q) times the judge-estimated one: same "
               "direction, smaller size. If p or q differ between conditions, a judge-estimated difference can be larger or smaller "
               "than the true one, and a small one can reverse; such differences in the judge's error rates are the main threat to "
               "the comparisons, and they are what the audit measures. ")
        Se, Sp = float(jr.recall), float(jr.specificity) if "specificity" in jr.index else np.nan
        fs = {c: _flag_share(D, c) for c in ("E0", "E1", "E1b", "E3")}
        rg = {c: (f + Sp - 1) / (Se + Sp - 1) for c, f in fs.items() if not _nan(f)}
        if rg and not _nan(Sp):
            neg = [c for c, v in rg.items() if v < 0]
            txt += (f"The expert-set error rates do not transfer to this generator: applying the selected judge's expert-set "
                    f"sensitivity ({f2(Se)}) and specificity ({f2(Sp)}) to the main study's flag shares with the Rogan-Gladen "
                    f"estimator [[cite:rogan1978]] gives "
                    + (f"negative true shares for {'every condition' if len(neg) == len(rg) else _join(neg)}, which is impossible; the judge "
                       f"flags fewer of the local model's claims than it would flag falsely at its expert-set specificity, so only an "
                       f"audit of the main study's own outputs can supply the error rates. " if neg else
                       "estimates that assume the expert-set error rates hold for this generator, which the audit tests. "))
        if A is not None and AD is not None and len(AD):
            keys = [k for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in AD.index]
            conf = [k.split(" minus ")[0] for k in keys if AD.loc[k, "diff_hi"] < 0]
            jconf = [k.split(" minus ")[0] for k in keys if AD.loc[k, "judge_diff_hi"] < 0]
            pq = "; ".join(f"{c} p = {f2(A.loc[c, 'precision'])}, q = {f2(A.loc[c, 'false_omission'])}" for c in ("E0", "E1", "E1b", "E3") if c in A.index)
            txt += (f"The audit gives these quantities per condition ({pq}; Section 7.9)"
                    + (f", and for {_join(conf)} the manually confirmed reduction relative to E0 has an interval below zero, so the "
                       f"direction of the judge's comparison is confirmed and the manual size is the better estimate of the true effect. "
                       if conf else
                       "; none of the manual intervals excludes zero at the audit's sample size"
                       + (f", although the judge's own paired differences on the audited courses do for {_join(jconf)}" if jconf else "") + ". "))
        among = [t for t in (D.T("UFR", "E1b vs E1"), D.T("UFR", "E3 vs E1")) if t is not None]
        if among:
            txt += (f"The same reasoning is why the grounded conditions are not ranked: their judge-estimated rates differ by at most "
                    f"{max(abs(t.delta_mean) for t in among):.3f}, which a change of a few points in the judge's precision or false "
                    f"omission rate between conditions would produce on its own.")
        b += [P(txt)]

    # ── 8.4 interpreting retrieval and verification
    b += [H2("8.4 Interpreting the Effects of Retrieval and Verification")]
    txt = (f"The pilot's RAG effect (Sections 5.3 and 5.4) was modest, disappeared when the judge was made more conservative, and "
           f"partly reflected shorter summaries with fewer generic sentences; retrieval helped most where the excerpts contain "
           f"specific facts (medications, procedures, findings) and not at all for follow-up and advice, which come from the model's "
           f"prior over patient instructions. ")
    if main and tu:
        short = vrow("E0/short vs E0/base", "UFR")
        lm = {c: vrow(k, "UFR", fl) for c, k, fl in (("E1", "E0/short vs E1/base", True), ("E1b", "E1b/base vs E0/short", False), ("E3", "E3/base vs E0/short", False))}
        nocite = vrow("E1/nocite vs E0/base", "UFR")
        beyond = [c for c in GROUNDED if lm[c] is not None and lm[c].p < 0.05 and lm[c].delta_mean < 0]
        txt += ("In the main study the grounded conditions lowered the judge-estimated UFR by similar amounts, and the variants of "
                "Section 7.5 show how much belongs to the mechanism: "
                + (f"shortening E0 to E1's length changed its UFR by only {signed(short.delta_mean)} ({fp(short.p)})"
                   + (f" and {_join(beyond)} stayed below the length-matched baseline, so brevity alone does not explain the reductions" if beyond else "") + "; "
                   if short is not None else "")
                + (f"without the citation instruction E1 still changed UFR by {signed(nocite.delta_mean)} ({fp(nocite.p)}), so retrieval works "
                   f"without an instruction to cite. " if nocite is not None and nocite.p < 0.05 and nocite.delta_mean < 0 else "")
                + "The most defensible reading is that grounding lowers the share of claims the judge cannot verify, partly by changing "
                "what the model writes and partly by how much, and that the true reduction is smaller than the judge-estimated one "
                "(Section 8.3). ")
        if "E1b" in tcov and tcov["E1b"] is not None:
            e1c, e3c = tcov.get("E1"), tcov.get("E3")
            txt += (f"What separates the grounded conditions is omission: E1 and E3 covered {abs(100 * e1c.delta_mean):.1f} and "
                    f"{abs(100 * e3c.delta_mean):.1f} points fewer of the clinician's sentences than E0 while E1b did not, and since E1b "
                    f"differs from E1 only in also seeing the course, the loss is associated with withholding the course rather than "
                    f"with retrieval; a plausible mechanism is that the fixed retrieval query misses passages the clinician chose to "
                    f"mention, and the data do not show whether the omitted sentences matter clinically. ")
        s31 = ST2.loc["E3 vs E1"] if ST2 is not None and "E3 vs E1" in ST2.index else None
        if s31 is not None:
            txt += (f"Whether verification adds to retrieval is the clearest open question: E3 and E1 do not differ under the "
                    f"selected judge but do under the second ({f3(s31.mean_b)} against {f3(s31.mean_a)}, {fp(s31.p)}), and the verifier, "
                    f"which is the generator itself, removes exactly the claims the judges disagree about. ")
        cc = {c: (D.m(c, "claims_total"), D.m(c, "cited_claims"), D.m(c, "cite_correct")) for c in ("E1", "E1b")}
        if all(not _nan(v[0]) and v[0] > 0 for v in cc.values()):
            txt += (f"Citations qualify the value of citing as a grounding device: a reviewer can use a citation only when it is present "
                    f"and correct, and that holds for {pct0(cc['E1'][2] / cc['E1'][0])} of E1's claims and {pct0(cc['E1b'][2] / cc['E1b'][0])} "
                    f"of E1b's. ")
        qa = M.qa_tot
        if qa is not None and {"QA-E0", "QA-E1"} <= set(qa.index):
            txt += ("The question-answering pilot points the same way: retrieval raised abstention and lowered the rate of the answers "
                    "given partly by making the model say less, and warning-sign questions, which the course rarely answers, were "
                    "answered from general knowledge; abstention, not retrieval, is the right response to them.")
    b += [P(txt)]

    # generator-robustness outcome, used by 8.5-8.7 and 9.2 (files written by the Section 7.10 reruns)
    repl_g, norepl_g = [], []
    for tag, gname in (("llama31", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B")):
        gpt = _csv(f"mimic_pairwise_tests_{tag}.csv")
        if gpt is None or not len(gpt):
            continue
        sig = False
        for c in ("E1", "E1b", "E3"):
            r = gpt[(gpt.metric == "UFR") & (gpt.comparison == f"{c} vs E0")]
            if len(r):
                ph = r.p_holm_primary.iloc[0] if "p_holm_primary" in r.columns and not _nan(r.p_holm_primary.iloc[0]) else r.p.iloc[0]
                if float(ph) < 0.05 and float(r.delta_mean.iloc[0]) < 0:
                    sig = True
        (repl_g if sig else norepl_g).append(gname)
    if repl_g and norepl_g:
        gen_note = (f" The reductions reproduced under {_join(repl_g)} but not under {_join(norepl_g)} (Section 7.10), so the "
                    f"recommendation is also generator-dependent.")
        gen_lim = (f" The grounding reductions replicated under {_join(repl_g)} but not under {_join(norepl_g)}, so the condition "
                   f"effects are statements about the generators tested (Section 7.10).")
    elif repl_g:
        gen_note = f" The reductions reproduced under {_join(repl_g)} (Section 7.10)."
        gen_lim = f" The grounding reductions replicated under {_join(repl_g)} (Section 7.10)."
    elif norepl_g:
        gen_note = f" The reductions did not reproduce under {_join(norepl_g)} (Section 7.10)."
        gen_lim = f" The grounding reductions did not replicate under {_join(norepl_g)} (Section 7.10)."
    else:
        gen_note = gen_lim = ""

    # ── 8.5 comparison of the approaches
    b += [H2("8.5 Comparison of the Approaches")]
    def cit_txt(c):
        if not main:
            return "n/a"
        n_all, n_cit, n_ok = D.m(c, "claims_total"), D.m(c, "cited_claims"), D.m(c, "cite_correct")
        return f"{pct0(n_cit / n_all)} of claims cited, {pct0(n_ok / n_cit) if n_cit else 'n/a'} of citations supported" if not _nan(n_all) and n_all else "n/a"
    if main and tu:
        cmp_text = (f"[[tab:compare]] summarizes the strengths, weaknesses and appropriate uses of the five conditions. No approach "
                    f"dominates on every criterion. RAG with the whole course in view reduced the judge-estimated rate of unsupported "
                    f"claims as much as the other grounded conditions while achieving coverage similar to E0, so it is the design these "
                    f"results favor, conditionally: the recommendation rests on judge-estimated rates for one corpus and one retrieval "
                    f"design, on a coverage measure scored by the same judge, and on citations that cover only "
                    f"{pct0(D.m('E1b', 'cited_claims') / D.m('E1b', 'claims_total')) if not _nan(D.m('E1b', 'claims_total')) else 'n/a'} of E1b's claims."
                    f"{gen_note} The extractive summarizer cannot fabricate but produces text no patient should be handed unedited.")
    else:
        cmp_text = "[[tab:compare]] summarizes the strengths, weaknesses and appropriate uses of the five conditions."
    b += [P(cmp_text),
          ("table", dict(label="compare", caption="Comparison of the five conditions across both studies. Pilot values are per-document means under the MiniLM judge on MTSamples; main-study values are judge-estimated per-document means under the selected judge on MIMIC-IV hospital courses; coverage is the judge-estimated share of the clinician's discharge-instruction sentences supported by the summary.",
                         columns=["Approach", "Pilot UFR / CR", "Main study UFR / CR", "Coverage of clinician's sentences", "Strengths", "Weaknesses", "Appropriate use"],
                         widths=[0.85, 0.7, 0.75, 0.75, 1.15, 1.15, 1.15], font=8, align=["left", "center", "center", "center", "left", "left", "left"],
                         rows=[[NAME[c], f"{f3(R.mean(c,'UFR'))} / {f3(R.mean(c,'CR'))}",
                                (f"{f3(M.mean(c,'UFR'))} / {f3(M.mean(c,'CR'))}" if M.ok and c in M.conds else "n/a"),
                                (pct0(M.mean(c, "coverage_ref")) if M.ok and c in M.conds else "n/a"), st, wk, use]
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
        b += [P("[[tab:findings]] separates what the data measure directly from the explanations offered for it, with the status of "
                "each explanation, and from the questions this thesis cannot answer.")]
        ka = ks = np.nan
        if at is not None and len(at):
            a_ = at[(at.analysis == "expert_agreement_atomic_min") & (at.group == "all")]; s_ = at[(at.analysis == "expert_agreement_sentence") & (at.group == "all")]
            if len(a_) and len(s_):
                ka, ks = float(a_.kappa.iloc[0]), float(s_.kappa.iloc[0])
        t_b = tcov.get("E1b")
        rows = [
            ["Pilot judge", f"Kappa {f2(cal_all.any_kappa)} against the experts; flags {pct0(cal_all.judge_UFR)} of sentences where they flag {pct0(cal_all.expert_flag_rate)}",
             "Granularity mismatch, small capacity, decision rule (capacity tested in Chapter 6)", "None"],
            ["Selected judge", f"Kappa {f2(jr.kappa)}" + (f"; about {f2(k_cv)} under nested cross-validation" if not _nan(k_cv) else ""),
             "Larger model and whole-course evidence help (tested)", "Agreement on other generators and institutions"],
            ["Clinical adaptation", "MedNLI accuracy up nine points; agreement not improved",
             "Granularity rather than vocabulary limits agreement (plausible" + (f"; atomic scoring kappa {f2(ka)} against {f2(ks)}" if not _nan(ka) else "") + ")",
             "What would improve agreement on patient-facing paraphrase"],
            ["Grounding", f"Judge-estimated UFR lower by {lo_u:.3f} to {hi_u:.3f} for E1, E1b and E3 (Holm-adjusted); robust to threshold, second judge" + (" and repeated generation" if rep is not None and len(rep) and "kind" in rep.columns else "")
             + (f"; replicates under {_join(repl_g)}" + (f" but not {_join(norepl_g)}" if norepl_g else "") if repl_g else (f"; does not replicate under {_join(norepl_g)}" if norepl_g else "")),
             "Focus on retrieved passages; fewer claims; citation instruction (separated in part by Section 7.5)",
             "True size of the reduction (Section 8.3)" + (f"; why the effect is absent for {_join(norepl_g)}" if norepl_g else "")],
            ["Omission", f"E1 and E3 lose coverage; E1b similar to E0 ({pts(t_b.delta_mean) if t_b is not None else 'n/a'})",
             "Withholding the course causes the loss (associated, via E1b against E1)", "Clinical importance of the omitted sentences"],
            ["Verification", "No difference from E1 under the selected judge; lower UFR under the second judge",
             "Verifier removes claims the judges disagree about (untested)", "Whether verification adds to retrieval"],
            ["Citations", f"E1: {cit_txt('E1')}; E1b: {cit_txt('E1b')}",
             "E1b draws on uncited parts of the course (plausible)", "Whether citations speed clinician review"],
            ["Clinician's instructions", f"Highest judge-estimated UFR ({f3(M.mean('REF','UFR'))})",
             "They state content the course does not (supported by the experts' spans)", "None"],
            ["Question answering", "Retrieval raised abstention and lowered the UFR of answers",
             "Warning-sign answers come from general knowledge (plausible)", "Correctness against clinician-written answers"]]
        b += [("table", dict(label="findings", caption="Findings of the thesis, the explanations offered for them and the status of each explanation, and the questions that remain open.",
                             columns=["Topic", "Finding", "Explanation offered (status)", "Open question"],
                             widths=[1.05, 2.05, 1.85, 1.55], font=8.5, align=["left"] * 4, rows=rows))]

    # ── 8.7 threats
    b += [H2("8.7 Threats to Validity and Limitations")]
    jk = (f"kappa {f2(jr.kappa)}, precision {f2(jr.precision)}, recall {f2(jr.recall)}" if jr is not None else "moderate agreement")
    bullets = [
        f"**Construct validity of the judge.** The pilot judge agrees with experts at chance, so the pilot's rates index how a small NLI "
        f"model reacts to a summary rather than hallucination. The main-study judge is moderate ({jk}), every rate in Chapter 7 is "
        "judge-estimated, and the second judge and the combined flagging rules are alternative estimates, not bounds, because judges share errors.",
        "**Optimism of the selection and overlap with the main study.** The judge was chosen on the sentences on which its agreement is "
        "reported" + (f" (nested cross-validation: about {f2(k_cv)} against an apparent {f2(jr.kappa)})" if not _nan(k_cv) and jr is not None else "")
        + ", and the doctor-written half of the expert set belongs to the main-study courses (Section 6.5).",
        "**Transfer of the validation.** The expert annotations cover other generators' output, and their error rates do not transfer to "
        "the local model (Section 8.3); the manual audit measures the judge on its output, but with one non-clinician annotator and 60 courses.",
        "**Sample size and generation variability.** 110 courses give adequate power for moderate paired effects; each summary was generated "
        "once, and the repeated generations on 40 courses (Section 7.8) show that per-course values are noisy.",
        "**External validity of the corpora.** MTSamples documents are public sample transcriptions that may be in the generator's training "
        "data; the main-study courses come from one institution and one section of the discharge summary, and the clinician's "
        "instructions used as the coverage reference contain content the course does not, so complete coverage is neither attainable nor desirable.",
        "**Generators and design choices.** One API model in the pilot and one four-bit seven-billion-parameter model in the main study, "
        "compared qualitatively, each with one prompt per condition, one retriever and one verifier (the generator itself)." + gen_lim,
        "**Heuristic components.** The header filter, the abstention rule and the citation parser are rule-based; an earlier parser "
        "mishandled citation markers and merged headers (Section 3.5), and the corrected one remains a source of error.",
        "**Measures scored by the same judge.** Coverage and citation accuracy run the judge in directions whose agreement with experts was "
        "not validated; the audit verified a sample of citations manually.",
        "**Question-answering pilot.** Three fixed questions without clinician-written gold answers; agreement with the clinician's "
        "instructions is a proxy for correctness."]
    b += [("bullets", bullets)]

    # ── 8.8 implications
    b += [H2("8.8 Implications for the Design and Evaluation of Clinical Summarization Systems")]
    b += [("numbers", [
        "**Validate the evaluator before the system, and again on the system's own outputs**, with agreement against clinician labels "
        "corrected for the optimism of choosing it, because error rates measured on other generators do not transfer.",
        "**Choose the judge by capacity and evidence mode, not by aggregation tweaks**: re-aggregating a small model's evidence changed "
        "little, while a larger consistency-trained model with the whole document as evidence raised kappa markedly.",
        "**Report counts, length and coverage with every rate, and paired differences with intervals**: a lower share of unsupported "
        "claims can come from writing fewer claims, and a small difference can come from the judge's error rates alone.",
        "**Separate the variables that differ between conditions** with length-matched and instruction-matched variants, and use "
        "extractive and human references as references, not as bounds.",
        "**Parse citations before splitting sentences, report citation coverage as well as accuracy, and keep protected text local**, "
        "with an automated check that none of it reaches the repository."])]

    # ══════════════════════════════════ CHAPTER 9 ══════════════════════════════════
    b += [H1("Chapter 9 Conclusion and Future Work"), H2("9.1 Summary")]
    b += [P(f"This thesis asked whether LLM-generated patient-facing summaries of clinical notes contain unsupported statements, whether "
            f"an automatic judge can measure them, and whether retrieval and verification reduce them. It built a reproducible "
            f"claim-level evaluation pipeline, found the pilot's judge invalid against medical-expert annotations, selected a better "
            f"judge by the same validation and audited it on the main study's own outputs, and compared five summarization conditions "
            f"on {M.n_docs if M.ok else 110} MIMIC-IV hospital courses with an open-weight model running locally, measuring omission "
            f"against the discharge instructions the clinicians wrote. The findings are conditional on the measured validity of that judge.")]
    b += [H2("9.2 Key Findings")]
    kf = [f"**Pilot judge.** Against 1,781 expert-annotated sentences the pilot judge agreed with the experts at chance level (kappa "
          f"{f2(cal_all.any_kappa)}), and no evidence-aggregation rule repaired it, so its pilot rates are not hallucination rates.",
          "**Evaluation artifacts.** Header lines counted as claims, abstentions labeled as contradictions and misattributed citation "
          "markers distort claim-level rates; all three were corrected."]
    if main and tu:
        kf += [f"**Judge selection.** DeBERTa-v3-large NLI with whole-course evidence reached kappa {f2(jr.kappa)} and AUROC {f2(jr.auroc)}"
               + (f" (about {f2(k_cv)} under nested cross-validation)" if not _nan(k_cv) else "")
               + "; MedNLI fine-tuning raised MedNLI accuracy but not agreement with the experts.",
               f"**Grounding.** On the hospital courses, E1, E1b and E3 lowered the judge-estimated UFR from {f3(M.mean('E0','UFR'))} to "
               f"{f3(M.mean('E1','UFR'))}, {f3(M.mean('E1b','UFR'))} and {f3(M.mean('E3','UFR'))} (all {padj(max(D.holm_p(t) for t in tu.values()))}) "
               f"without changing CR; the three do not differ from one another, and part of the reduction comes from shorter summaries.",
               (f"**Coverage.** E1 and E3 covered fewer of the clinician's sentences than E0; E1b achieved coverage similar to E0, with no "
                f"statistically significant difference detected ({pts(tcov['E1b'].delta_mean)}, 95 percent interval "
                f"{ci_pts(tcov['E1b'].ci95_low, tcov['E1b'].ci95_high)})." if tcov.get("E1b") is not None else ""),
               (f"**Citations.** E1 cited {pct0(D.m('E1', 'cited_claims') / D.m('E1', 'claims_total'))} of its claims and E1b "
                f"{pct0(D.m('E1b', 'cited_claims') / D.m('E1b', 'claims_total'))}; {pct0(D.m('E1', 'cite_correct') / D.m('E1', 'cited_claims'))} of "
                f"E1's citations were supported by the cited excerpts." if not _nan(D.m('E1', 'claims_total')) else ""),
               "**Robustness.** The reductions relative to E0 hold at a second threshold, under a second judge"
               + (" and in repeated generations" if rep is not None and len(rep) and "kind" in rep.columns else "")
               + "; verification beats retrieval alone only under the second judge."
               + (f" They replicate for {_join(repl_g)}" + (f" but not for {_join(norepl_g)}" if norepl_g else "") + " (Section 7.10)."
                  if repl_g else (f" They do not replicate under {_join(norepl_g)} (Section 7.10)." if norepl_g else ""))]
        if A is not None and AD is not None and len(AD):
            keys = [k for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in AD.index]
            conf = [k for k in keys if AD.loc[k, "diff_hi"] < 0]
            ag_ = _csv("audit_agreement.csv")
            agk = (f" after two blind passes that agree at kappa {f2(ag_[ag_.kind == 'claim'].iloc[0].kappa)}"
                   if ag_ is not None and len(ag_) and "claim" in set(ag_.kind) else "")
            if conf:
                parts = "; ".join(f"{k.split(' minus ')[0]} {pts(AD.loc[k, 'diff'])} ({ci_pts(AD.loc[k, 'diff_lo'], AD.loc[k, 'diff_hi'])})" for k in conf)
                kf.append(f"**Manual audit.** The judge understates absolute unsupported shares; the manually confirmed reduction relative "
                          f"to E0 excludes zero for {parts}{agk}, and the other condition differences are unresolved at the audit's sample size.")
            else:
                kf.append(f"**Manual audit.** The judge understates absolute unsupported shares, and none of the manually confirmed "
                          f"condition differences excludes zero at the audit's sample size{agk}.")
        ev = _csv("verifact_judge_eval.csv")
        if ev is not None and "subset" in ev.columns and "all" in set(ev.subset):
            al = ev.set_index("subset").loc["all"]
            kf.append(f"**External check.** On {int(al.n):,} clinician-labeled propositions from another institution the judge reaches "
                      f"AUROC {f2(al.auroc)} and kappa {f2(al.kappa)} (sensitivity {f2(al.sensitivity)}, specificity {f2(al.specificity)}): "
                      f"it still separates supported from unsupported statements, with the same low sensitivity the audit found.")
        if M.qa_tot is not None and {"QA-E0", "QA-E1"} <= set(M.qa_tot.index):
            kf.append(f"**Question answering (exploratory).** Retrieval raised abstention from {pct0(M.qa_tot.loc['QA-E0','abstention_rate'])} to "
                      f"{pct0(M.qa_tot.loc['QA-E1','abstention_rate'])} and lowered the judge-estimated UFR of the answers given; warning-sign "
                      f"questions are rarely answerable from the course.")
    b += [("numbers", [k for k in kf if k])]
    b += [H2("9.3 Future Work")]
    b += [("numbers", [
        "**Clinician annotation of the generator's own output.** Two clinicians labeling the audit sample would give agreement between "
        "annotators and clinically validated error rates for the selected judge on these outputs.",
        "**Better judges for patient-facing paraphrase.** LLM-based judges calibrated against expert labels, alignment models trained on "
        "clinical consistency data, and decomposition designs that keep each fact's context are candidates for the same selection protocol.",
        "**Larger generators and more generations.** A workstation-class GPU would allow open-weight models of 30 to 70 billion parameters, "
        "several generations per course, and more institutions and note types.",
        "**A clinician study of omission and of question answering.** Clinicians rating the importance of omitted sentences, and "
        "clinician-written gold answers, would turn the coverage measure and the question-answering pilot into validated measures."])]
    return b
