"""ch_mimic.py — Chapter 7: Main Study: Results on MIMIC-IV Hospital Courses (local model, selected judge).

Every number is read from the aggregate files in results/ written by mimic_evaluate.py, mimic_revision_analysis.py,
mimic_qa.py, mimic_second_judge.py, atomic_claims.py and audit_summary.py; a section whose file is missing is left out.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from results_loader import Results, f2, f3, pct, pct0, fp, fpn, signed, holm
from mimic_results import JUDGE_NAMES, NAME, VAR, VPHRASE

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ORDER = ["E0", "E1", "E1b", "E3", "E2", "REF"]
GROUNDED = ["E1", "E1b", "E3"]
MODE_NAMES = {"top3": "top-3 sentence evidence", "doc": "whole-course evidence"}
GEN_NAMES = {"base": "Main generation", "rep1": "Repeat 1", "rep2": "Repeat 2", "mean": "Mean of three"}
QUESTION = {"medications": "Medications", "follow_up": "Follow-up", "warning_signs": "Warning signs", "all": "All questions"}
AUDIT_TYPE = {"not_stated": "not stated in the course", "generic": "generic advice or follow-up not in the course",
              "contradicts": "contradicts the course", "med_number": "wrong or invented medication, dose or number", "other": "other"}


def P(t):
    return ("p", t)


def _csv(name):
    f = RES / name
    try:
        return pd.read_csv(f) if f.exists() else None
    except pd.errors.EmptyDataError:
        return None


def _nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def pts(x):
    """Coverage difference in percentage points: 0.012 -> '+1.2 points'."""
    return "—" if _nan(x) else f"{100 * x:+.1f} points"


def pts_bare(x):
    return "—" if _nan(x) else f"{100 * x:+.1f}"


def ci3(lo, hi):
    return f"{f3(lo)} to {f3(hi)}"


def ci_pts(lo, hi):
    return f"{pts_bare(lo)} to {pts_bare(hi)}"


def padj(p):
    """'adjusted p = 0.004' / 'adjusted p < 0.001'."""
    return fp(p).replace("p ", "adjusted p ", 1)


class Flip:
    """A paired-test row with the direction reversed (b - a becomes a - b)."""
    def __init__(self, t):
        self.delta_mean, self.ci95_low, self.ci95_high, self.p = -t.delta_mean, -t.ci95_high, -t.ci95_low, t.p
        self.mean_a, self.mean_b = t.mean_b, t.mean_a


class Data:
    def __init__(self):
        self.summ, self.tests = _csv("mimic_summary.csv"), _csv("mimic_pairwise_tests.csv")
        self.ok = self.summ is not None and self.tests is not None
        if not self.ok:
            return
        self.base = self.summ[self.summ.variant == "base"].set_index("condition")
        self.conds = [c for c in ORDER if c in self.base.index]
        self.cis = _csv("mimic_condition_ci.csv")
        self.judge, self.mode, self.tau = self.summ.judge.iloc[0], self.summ["mode"].iloc[0], float(self.summ.tau.iloc[0])
        self.n = int(self.base.loc["E0", "n_docs"])

    def m(self, c, col):
        if col == "claims_total" and "claims_total" not in self.base.columns and c in self.base.index:
            return float(self.base.loc[c, "claims_mean"] * self.base.loc[c, "n_docs"])
        return float(self.base.loc[c, col]) if c in self.base.index and col in self.base.columns else np.nan

    def ci(self, c, metric, variant="base"):
        """(mean, low, high) of a per-summary measure, from results/mimic_condition_ci.csv."""
        if self.cis is None:
            return (np.nan, np.nan, np.nan)
        r = self.cis[(self.cis.condition == c) & (self.cis.variant == variant) & (self.cis.metric == metric)]
        return (float(r["mean"].iloc[0]), float(r.ci95_low.iloc[0]), float(r.ci95_high.iloc[0])) if len(r) else (np.nan, np.nan, np.nan)

    def T(self, metric, comp):
        t = self.tests[(self.tests.metric == metric) & (self.tests.comparison == comp)]
        return t.iloc[0] if len(t) else None

    @staticmethod
    def holm_p(t):
        return float(t.p_holm_primary) if t is not None and "p_holm_primary" in t.index and not _nan(t.p_holm_primary) else np.nan

    def sig(self, t):
        """Significant after the Holm adjustment of the primary family (falls back to the raw p for other tests)."""
        if t is None:
            return False
        ph = self.holm_p(t)
        return (ph if not _nan(ph) else t.p) < 0.05


def mean_ci(D, c, metric, fmt=f3):
    mu, lo, hi = D.ci(c, metric)
    if _nan(mu):
        mu = D.m(c, f"{metric}_mean")
    return f"{fmt(mu)} ({fmt(lo)}–{fmt(hi)})" if not _nan(lo) else fmt(mu)


def delta_ci(t, unit="rate"):
    if t is None:
        return "—"
    if unit == "pts":
        return f"{pts_bare(t.delta_mean)} ({ci_pts(t.ci95_low, t.ci95_high)})"
    return f"{signed(t.delta_mean)} ({ci3(t.ci95_low, t.ci95_high)})"


def blocks(R: Results) -> list:
    D = Data()
    b = [("h1", "Chapter 7 Main Study: Results on MIMIC-IV Hospital Courses")]
    if not D.ok:
        return b + [P("The main study has not been evaluated yet.")]
    base, conds, T = D.base, D.conds, D.T
    llm = [c for c in conds if c not in ("E2", "REF")]

    # ── 7.1 Setting
    b += [("h2", "7.1 Setting")]
    b += [P(f"The main study applies the five conditions to the {D.n} hospital courses of ann-pt-summ (Section 3.4), each paired with "
            f"the discharge instructions its clinician wrote. Qwen2.5-7B-Instruct generated every summary on the author's computer, "
            f"the retrieval conditions had to cite the excerpts behind each sentence, and the judge selected in Chapter 6 "
            f"({JUDGE_NAMES.get(D.judge, D.judge)}, {MODE_NAMES.get(D.mode, D.mode)}, support threshold {D.tau:.2f}) scored every "
            f"summary and the clinician's instructions, with the claim splitting of Section 3.5. Three conventions hold throughout. "
            f"Every rate is judge-estimated; Section 7.9 relates the judge's figures to manually confirmed ones. The main results come "
            f"from one generation per course at temperature 0.3; Section 7.8 measures the variation between generations. Every "
            f"comparison is paired over courses and reported as a mean difference with a 95 percent bootstrap interval and a Wilcoxon "
            f"p-value; the twelve primary comparisons (E1, E1b, E2 and E3 against E0 on UFR, CR and coverage) are Holm-adjusted and "
            f"called significant only after the adjustment, and an unadjusted p-value below 0.05 elsewhere is called suggestive.")]

    # ── 7.2 Descriptive statistics
    b += [("h2", "7.2 Descriptive Statistics")]
    ts, tu = {c: T("n_supported", f"{c} vs E0") for c in GROUNDED}, {c: T("n_unsupported", f"{c} vs E0") for c in GROUNDED}
    fewer_sup = [c for c in GROUNDED if ts[c] is not None and ts[c].p < 0.05 and ts[c].delta_mean < 0]
    fewer_un = [c for c in GROUNDED if tu[c] is not None and tu[c].p < 0.05 and tu[c].delta_mean < 0]
    txt = (f"[[tab:mdesc]] describes the outputs. E0 writes the longest summaries ({D.m('E0', 'words_mean'):.0f} words, "
           f"{D.m('E0', 'claims_mean'):.1f} claims), the grounded conditions between {min(D.m(c, 'words_mean') for c in GROUNDED):.0f} "
           f"and {max(D.m(c, 'words_mean') for c in GROUNDED):.0f} words, and the clinician's instructions {D.m('REF', 'words_mean'):.0f}. "
           f"Because UFR is a share of claims, the counts behind it matter: "
           + (f"{_join(fewer_un)} state{'s' if len(fewer_un) == 1 else ''} fewer unsupported claims than E0" if fewer_un else "the grounded conditions state about as many unsupported claims as E0")
           + (f", and {_join(fewer_sup)} also state{'s' if len(fewer_sup) == 1 else ''} fewer supported ones, so part of {'its' if len(fewer_sup) == 1 else 'their'} lower rate comes from saying less"
              if fewer_sup else ", while stating as many supported claims as E0 within the precision of the test")
           + ". A lower rate is therefore not by itself a better summary; Section 7.4 measures what the summaries leave out.")
    b += [P(txt),
          ("table", dict(label="mdesc", caption="Length, claims and abstentions per summary in the main study (means over the 110 courses). Supported and unsupported claims are counted by the selected judge.",
                         columns=["Condition", "Words", "Claims", "Supported claims", "Unsupported claims", "Abstentions"],
                         widths=[2.3, 0.7, 0.7, 0.95, 1.0, 0.85], font=9.5,
                         rows=[[NAME.get(c, c), f"{D.m(c, 'words_mean'):.0f}", f"{D.m(c, 'claims_mean'):.1f}", f"{D.m(c, 'supported_mean'):.1f}",
                                f"{D.m(c, 'unsupported_mean'):.1f}", f"{D.m(c, 'abstentions_mean'):.2f}"] for c in conds]))]

    # ── 7.3 Unsupported fact rate and contradiction rate
    b += [("h2", "7.3 Judge-Estimated Unsupported Fact Rate and Contradiction Rate")]
    rows = []
    for metric in ("UFR", "CR"):
        for i, c in enumerate(conds):
            t0, t1 = T(metric, f"{c} vs E0"), T(metric, f"{c} vs E1")
            rows.append([metric if i == 0 else "", NAME.get(c, c), mean_ci(D, c, metric), f3(D.m(c, f"{metric}_median")),
                         delta_ci(t0), fpn(D.holm_p(t0)) if t0 is not None else "—",
                         f"{signed(t1.delta_mean)} ({fpn(t1.p)})" if t1 is not None else "—"])
    lower = [c for c in GROUNDED if D.sig(T("UFR", f"{c} vs E0")) and T("UFR", f"{c} vs E0").delta_mean < 0]
    cr_sig = [c for c in GROUNDED if D.sig(T("CR", f"{c} vs E0"))]
    among = [(c, T("UFR", f"{c} vs E1")) for c in ("E1b", "E3")]
    among = [(c, t) for c, t in among if t is not None]
    zero_in = [c for c, t in among if t.ci95_low <= 0 <= t.ci95_high]
    lo_g, hi_g = min(D.m(c, "UFR_mean") for c in GROUNDED), max(D.m(c, "UFR_mean") for c in GROUNDED)
    txt = ("[[tab:mrates]] gives the rates with their intervals and paired comparisons, and [[fig:mbox]] their distributions. "
           + (f"All three grounded conditions had a lower judge-estimated UFR than E0 ({f3(D.m('E0', 'UFR_mean'))} against "
              f"{f3(lo_g)} to {f3(hi_g)}), and every difference remains significant after the Holm adjustment"
              if len(lower) == len(GROUNDED) else
              f"{_join(lower) if lower else 'No grounded condition'} had a significantly lower judge-estimated UFR than E0 "
              f"({f3(D.m('E0', 'UFR_mean'))}) after the Holm adjustment")
           + ("; none changed the contradiction rate, which is low under every condition. "
              if not cr_sig else f"; {_join(cr_sig)} changed the contradiction rate significantly. ")
           + ("The differences among the grounded conditions are small and their intervals include zero, and a difference of that "
              "size could come from a small difference in the judge's error rates between conditions (Section 7.9), so these data do "
              "not rank the three grounded conditions against one another. "
              if len(zero_in) == len(among) else
              f"Among the grounded conditions only {_join([c for c, _ in among if c not in zero_in])} differs from E1 with an interval "
              f"that excludes zero, a secondary comparison that Section 7.10 examines under a second judge. "))
    b += [P(txt),
          ("table", dict(label="mrates", caption="Judge-estimated UFR and CR per summary in the main study: mean with 95 percent bootstrap interval, median, paired mean difference from E0 with 95 percent interval and Holm-adjusted p-value (primary family of twelve tests), and paired difference from E1 with unadjusted Wilcoxon p-value. Negative differences favor the row condition.",
                         columns=["Metric", "Condition", "Mean (95% CI)", "Median", "Δ vs E0 (95% CI)", "Holm p", "Δ vs E1 (p)"],
                         widths=[0.45, 1.5, 1.1, 0.55, 1.4, 0.55, 0.95], font=8, rows=rows)),
          ("figure", dict(label="mbox", path="results/fig_mimic_box.png", width=6.5, caption="Per-document judge-estimated UFR (left) and CR (right) in the main study under each condition and for the clinician's instructions."))]
    ref = _csv("mimic_reference.csv")
    sj = _csv("mimic_judge_agreement.csv")
    sj_e2 = sj.set_index("condition").loc["E2"] if sj is not None and "E2" in set(sj.condition) else None
    refp = ""
    if ref is not None and "REF" in conds:
        refp = (f"The clinician's own instructions receive the highest rate of all, {f3(D.m('REF', 'UFR_mean'))}, not because clinicians "
                f"hallucinate but because discharge instructions state medication changes, appointments and advice that the hospital "
                f"course does not (the experts marked {ref.expert_spans.mean():.1f} unsupported spans per document in these texts): "
                f"support by the course measures faithfulness to the course, not the quality of a summary. ")
    if "E2" in conds:
        refp += (f"At the other end, E2 copies sentences of the course verbatim, so its rate of {f3(D.m('E2', 'UFR_mean'))} is judge "
                 f"error on text that is supported by construction, mostly sentences taken out of context or containing a negated "
                 f"list. It is a reference for that error, not a floor for the other conditions, whose sentences differ in length, "
                 f"negation and dependence on context"
                 + (f"; the second judge flags only {pct(sj_e2.flag_rate_second)} of E2's claims (Section 7.10)." if sj_e2 is not None else "."))
    if refp:
        b += [P(refp)]

    # ── 7.4 Coverage
    b += [("h2", "7.4 Coverage of the Clinician's Instructions")]
    cov_conds = [c for c in conds if c != "REF"]
    rows = []
    for c in cov_conds:
        t0, t1 = T("coverage_ref", f"{c} vs E0"), T("coverage_ref", f"{c} vs E1")
        rows.append([NAME.get(c, c), mean_ci(D, c, "coverage_ref", fmt=lambda x: f"{100 * x:.1f}"),
                     f"{D.m(c, 'omitted_ref_mean'):.1f} of {D.m(c, 'ref_sents_mean'):.1f}" if not _nan(D.m(c, "omitted_ref_mean")) else "—",
                     delta_ci(t0, "pts"), fpn(D.holm_p(t0)) if t0 is not None else "—",
                     f"{pts_bare(t1.delta_mean)} ({fpn(t1.p)})" if t1 is not None else "—"])
    eq = _csv("mimic_equivalence.csv")
    E = eq.set_index("comparison") if eq is not None else None
    t_b = T("coverage_ref", "E1b vs E0")
    lost = [c for c in ("E1", "E3", "E2") if c in cov_conds and D.sig(T("coverage_ref", f"{c} vs E0")) and T("coverage_ref", f"{c} vs E0").delta_mean < 0]
    txt = (f"Coverage is the share of the clinician's sentences that the generated summary supports, judged by the same model in the "
           f"reverse direction (Section 4.10.3), so it is judge-estimated like UFR. [[tab:mcov]] and [[fig:mcov]] report it with the "
           f"number of the clinician's sentences each summary leaves uncovered. E0 covered {pct(D.m('E0', 'coverage_ref_mean'))} of "
           f"them"
           + (f"; {_join(lost)} covered significantly less (" + ", ".join(f"{c} {pct(D.m(c, 'coverage_ref_mean'))}" for c in lost) + ")" if lost else "")
           + ". ")
    if t_b is not None:
        if not D.sig(t_b):
            txt += (f"E1b achieved coverage similar to E0 ({pct(D.m('E1b', 'coverage_ref_mean'))}), with no statistically significant "
                    f"difference detected ({pts(t_b.delta_mean)}, 95 percent interval {ci_pts(t_b.ci95_low, t_b.ci95_high)}). A "
                    f"non-significant difference is not evidence of equivalence")
            if E is not None and "E1b vs E0" in E.index:
                e = E.loc["E1b vs E0"]
                txt += (f"; as a post hoc check with a margin of {100 * e.margin:.0f} points, the 90 percent interval of the difference "
                        f"({ci_pts(e.ci90_low, e.ci90_high)} points) "
                        + ("lies within the margin, so a coverage loss of more than five points is unlikely on this sample. "
                           if e.equivalent_at_margin else "extends beyond the margin, so a loss of more than five points cannot be excluded. "))
            else:
                txt += ". "
        else:
            txt += f"E1b covered {pct(D.m('E1b', 'coverage_ref_mean'))}, a significant difference from E0 ({pts(t_b.delta_mean)}). "
    t_bb, t_bu = T("coverage_ref", "E1b vs E1"), T("UFR", "E1b vs E1")
    if t_bb is not None and t_bu is not None:
        txt += (f"E1b differs from E1 only in also seeing the whole course; it {'raised' if t_bb.delta_mean > 0 else 'lowered'} coverage "
                f"by {abs(100 * t_bb.delta_mean):.1f} points relative to E1 ({fp(t_bb.p)}) while its UFR "
                f"{'did not differ significantly from' if t_bu.p >= 0.05 else ('was lower than' if t_bu.delta_mean < 0 else 'was higher than')} "
                f"E1's, so the coverage loss of E1 is associated with withholding the course rather than with retrieval as such. ")
    ab = _csv("mimic_abstention.csv")
    if ab is not None and "E3" in set(ab.condition):
        a3 = ab.set_index("condition").loc["E3"]
        if not _nan(a3.get("rho_abst_coverage", np.nan)):
            assoc = a3.p_abst_coverage < 0.05 or a3.p_abst_ufr < 0.05
            txt += (f"E3 abstained in {pct0(a3.share_with_abstention)} of its summaries; within E3 the number of abstentions is "
                    + ("not associated with coverage or UFR" if not assoc else "associated with the measures")
                    + f" (Spearman {a3.rho_abst_coverage:.2f} and {a3.rho_abst_ufr:.2f}), so "
                    + ("its coverage loss is inherited from its E1 draft rather than caused by abstaining." if not assoc else
                       "part of its coverage loss accompanies its abstentions."))
    b += [P(txt),
          ("table", dict(label="mcov", caption="Judge-estimated coverage of the clinician's discharge instructions: mean percentage of the clinician's sentences supported by the summary (95 percent interval), mean number of the clinician's sentences not covered, and paired differences in percentage points from E0 (95 percent interval, Holm-adjusted p) and from E1 (unadjusted p).",
                         columns=["Condition", "Coverage, % (95% CI)", "Sentences not covered", "Δ vs E0, points (95% CI)", "Holm p", "Δ vs E1 (p)"],
                         widths=[1.6, 1.15, 0.95, 1.35, 0.55, 0.9], font=8.5, rows=rows)),
          ("figure", dict(label="mcov", path="results/fig_mimic_coverage.png", width=5.6, caption="Judge-estimated coverage of the clinician's discharge instructions per document under each condition."))]

    # ── retrieval-ablation families (Holm within each measure), used in Sections 7.5 and 7.7
    abl = _csv("mimic_ablations.csv")
    variants = [v for v in VAR if v != "base"]
    cite_measures = ("citation_coverage", "citation_accuracy")
    def fam(metric):
        vs = [v for v in variants if not (metric in cite_measures and v == "nocite")]
        got = []
        for v in vs:
            r = abl[(abl.metric == metric) & (abl.variant == v)]
            if len(r) and not _nan(r.p.iloc[0]):
                got.append((v, float(r.delta_mean.iloc[0]), float(r.p.iloc[0])))
        adj = holm([g[2] for g in got]) if got else []
        return {v: (d, pv, float(a)) for (v, d, pv), a in zip(got, adj)}
    F = {m: fam(m) for m in ("UFR", "CR", "coverage_ref") + cite_measures} if abl is not None and len(abl) else {}

    # ── 7.5 Length, citation instruction and retrieval
    vt = _csv("mimic_variant_tests.csv")
    if vt is not None and len(vt):
        b += [("h2", "7.5 Separating Length, the Citation Instruction and Retrieval")]
        spec = [("Length-matched E0 minus E0", "E0/short vs E0/base", False),
                ("E1 minus length-matched E0", "E0/short vs E1/base", True),
                ("E1b minus length-matched E0", "E1b/base vs E0/short", False),
                ("E3 minus length-matched E0", "E3/base vs E0/short", False),
                ("E1 without citations minus E0", "E1/nocite vs E0/base", False),
                ("E1b without citations minus E1b", "E1b/nocite vs E1b/base", False)]
        def vrow(comp, metric, flip):
            r = vt[(vt.comparison == comp) & (vt.metric == metric)]
            return (Flip(r.iloc[0]) if flip else r.iloc[0]) if len(r) else None
        adjp = {}
        for metric in ("UFR", "coverage_ref"):
            got = [(lab, vrow(comp, metric, fl)) for lab, comp, fl in spec]
            got = [(lab, t) for lab, t in got if t is not None]
            for (lab, t), a in zip(got, holm([t.p for _, t in got]) if got else []):
                adjp[(lab, metric)] = float(a)
        rows = []
        for lab, comp, fl in spec:
            u, cv, w = vrow(comp, "UFR", fl), vrow(comp, "coverage_ref", fl), vrow(comp, "words", fl)
            if u is None:
                continue
            rows.append([lab, f"{w.mean_a:.0f} → {w.mean_b:.0f}" if w is not None else "—", delta_ci(u), fpn(adjp.get((lab, "UFR"))),
                         delta_ci(cv, "pts") if cv is not None else "—", fpn(adjp.get((lab, "coverage_ref")))])
        ab_n = abl[(abl.metric == "UFR") & (abl.variant == "nocite")] if abl is not None else None
        ab_cov = abl[(abl.metric == "coverage_ref") & (abl.variant == "nocite")] if abl is not None else None
        if ab_n is not None and len(ab_n):
            fu = Flip(ab_n.iloc[0]); fc = Flip(ab_cov.iloc[0]) if ab_cov is not None and len(ab_cov) else None
            rows.append(["E1 minus E1 without citations", "—", delta_ci(fu), "see Section 7.7", delta_ci(fc, "pts") if fc is not None else "—", "see Section 7.7"])
        short_u = vrow("E0/short vs E0/base", "UFR", False)
        nocite_u = vrow("E1/nocite vs E0/base", "UFR", False)
        nb = vrow("E1b/nocite vs E1b/base", "UFR", False)
        lm = {c: vrow(k, "UFR", fl) for c, k, fl in (("E1", "E0/short vs E1/base", True), ("E1b", "E1b/base vs E0/short", False), ("E3", "E3/base vs E0/short", False))}
        txt = ("Section 4.10.2 listed what differs between the conditions besides the mechanism under study; the variants in "
               "[[tab:mvar]] separate the two largest confounds, the length of the summary and the citation instruction, with "
               "Holm-adjusted p-values within this family of six comparisons. ")
        if short_u is not None:
            lab = "Length-matched E0 minus E0"; sig_s = adjp.get((lab, "UFR"), 1) < 0.05
            still = [c for c in GROUNDED if lm[c] is not None and lm[c].delta_mean < 0 and adjp.get((f"{c} minus length-matched E0", "UFR"), 1) < 0.05]
            txt += (f"An E0 asked for about 150 words {'lowered' if sig_s and short_u.delta_mean < 0 else 'did not change'} its UFR "
                    f"({signed(short_u.delta_mean)}, {padj(adjp.get((lab, 'UFR')))}), and "
                    + (f"{_join(still)} stayed below this length-matched baseline, so the reductions of Section 7.3 are not a by-product of brevity"
                       + (", although brevity accounts for part of them" if sig_s and short_u.delta_mean < 0 else "") + ". "
                       if still else "no grounded condition stayed significantly below this length-matched baseline, so the reductions cannot be separated from length. "))
        if nocite_u is not None:
            lab = "E1 without citations minus E0"; sig_n = adjp.get((lab, "UFR"), 1) < 0.05
            txt += (f"Without the citation instruction E1 {'still had a lower UFR than E0' if sig_n and nocite_u.delta_mean < 0 else 'did not differ significantly from E0'} "
                    f"({signed(nocite_u.delta_mean)}, {padj(adjp.get((lab, 'UFR')))}), so retrieval "
                    + ("lowers the judge-estimated rate without any instruction to cite" if sig_n and nocite_u.delta_mean < 0 else "cannot be separated from the instruction")
                    + ". ")
        if nb is not None:
            lab = "E1b without citations minus E1b"
            txt += (f"For E1b, dropping the instruction {'raised' if nb.delta_mean > 0 else 'lowered'} UFR by {abs(nb.delta_mean):.3f} "
                    f"({padj(adjp.get((lab, 'UFR')))}), so part of E1b's effect "
                    + ("does come from citing." if adjp.get((lab, "UFR"), 1) < 0.05 and nb.delta_mean > 0 else "may come from citing, but the separate effect is not established."))
        b += [P(txt),
              ("table", dict(label="mvar", caption="Variants that separate length and the citation instruction from retrieval: mean words of the two summaries compared, paired differences in judge-estimated UFR and in coverage (percentage points) with 95 percent intervals, and Holm-adjusted p-values within the family of six comparisons. The last row is the citation ablation of Section 7.7, reversed.",
                             columns=["Comparison", "Words", "Δ UFR (95% CI)", "Holm p", "Δ coverage, points (95% CI)", "Holm p"],
                             widths=[1.9, 0.8, 1.35, 0.55, 1.35, 0.55], font=8.5, align=["left"] + ["center"] * 5, rows=rows))]

    # ── 7.6 Citations
    cited = [c for c in ("E1", "E1b") if c in base.index and D.m(c, "cited_claims") > 0]
    if cited:
        b += [("h2", "7.6 Citation Coverage, Accuracy and Reliability")]
        rows = []
        for c in cited:
            n_all, n_cit, n_ok = int(D.m(c, "claims_total")), int(D.m(c, "cited_claims")), int(D.m(c, "cite_correct"))
            rows.append([NAME.get(c, c), f"{n_all:,}", f"{n_cit:,}", pct(n_cit / n_all), f"{n_ok:,}", pct(n_ok / n_cit) if n_cit else "—", pct(n_ok / n_all)])
        cc = {c: (int(D.m(c, "claims_total")), int(D.m(c, "cited_claims")), int(D.m(c, "cite_correct"))) for c in cited}
        txt = ("[[tab:mcite]] reports, for the retrieval conditions, citation coverage (the share of claims with a valid citation), "
               "citation accuracy (the share of cited claims whose cited excerpts support the claim under the judge) and citation "
               "reliability (the share of all claims with a valid and supporting citation), from the pooled counts and with the "
               "corrected attribution of citation markers (Section 3.5). ")
        if "E1" in cc:
            n_all, n_cit, n_ok = cc["E1"]
            txt += (f"E1 cited {pct0(n_cit / n_all)} of its claims, {pct0(n_ok / n_cit)} of those citations are supported by the excerpt "
                    f"they cite, so {pct0(n_ok / n_all)} of all E1 claims carry a citation that supports them. ")
        if "E1b" in cc:
            n_all, n_cit, n_ok = cc["E1b"]
            txt += (f"E1b cited only {pct0(n_cit / n_all)} of its claims, which its design predicts: it also sees the whole course, and "
                    f"statements drawn from outside the three excerpts have no excerpt to cite; its accuracy is "
                    f"{pct0(n_ok / n_cit) if n_cit else '—'} and its reliability {pct0(n_ok / n_all)}. ")
        ac = _csv("audit_citations.csv")
        if ac is not None and len(ac):
            A = ac.set_index("condition")
            txt += ("Reading the cited excerpts of a random sample confirms the picture (Section 4.10.5): "
                    + "; ".join(f"of {int(A.loc[c, 'n'])} {c} citations, {pct0(A.loc[c, 'manual_yes'])} fully and "
                                f"{pct0(A.loc[c, 'manual_yes_or_partly'])} at least partly support the claim"
                                + (f" (judge agreement {pct0(A.loc[c, 'agreement'])})" if "agreement" in A.columns and not _nan(A.loc[c, "agreement"]) else "")
                                for c in A.index) + ".")
        b += [P(txt),
              ("table", dict(label="mcite", caption="Citation coverage, accuracy and reliability of the retrieval conditions, from the pooled counts over the 110 courses. Accuracy is the share of cited claims supported by their cited excerpts under the judge; reliability is the share of all claims that carry a valid and supporting citation.",
                             columns=["Condition", "Claims", "Cited claims", "Citation coverage", "Cited and supported", "Citation accuracy", "Citation reliability"],
                             widths=[1.7, 0.65, 0.7, 0.8, 0.85, 0.8, 0.85], font=8.5, rows=rows))]

    # ── 7.7 Retrieval ablations
    if abl is not None and len(abl):
        b += [("h2", "7.7 Retrieval Ablations")]
        def cell(m, v):
            if v not in F[m]:
                return "—"
            d, pv, _ = F[m][v]
            return f"{pts_bare(d) if m in ('coverage_ref',) + cite_measures else signed(d)} ({fpn(pv)})"
        rows = [[VAR[v]] + [cell(m, v) for m in ("UFR", "CR", "coverage_ref") + cite_measures] for v in variants]
        def changed(m):
            unit = (lambda d: f"{abs(100 * d):.1f} points") if m in ("coverage_ref",) + cite_measures else (lambda d: f"{abs(d):.3f}")
            return [f"{VPHRASE[v]} {'raised' if d > 0 else 'lowered'} it by {unit(d)} ({padj(pa)})" for v, (d, pv, pa) in F[m].items() if pa < 0.05]
        faith = changed("UFR") + changed("CR"); covc = changed("coverage_ref"); citc = changed("citation_coverage") + changed("citation_accuracy")
        txt = ("Six variants of E1 change one retrieval setting at a time (Section 4.10.4); [[tab:mabl]] gives each variant's paired "
               "difference from the base configuration, and the text uses Holm-adjusted p-values within each measure. "
               + ("No variant changed UFR or CR after the adjustment. " if not faith else "For faithfulness, " + "; ".join(faith) + ". ")
               + ("Coverage followed the amount of the course shown to the model: " + "; ".join(covc) + ". " if covc else "No variant changed coverage after the adjustment. ")
               + ("For the citation measures, " + "; ".join(citc) + ". " if citc else "")
               + ("Lexical BM25 and dense retrieval produced indistinguishable rates and coverage. "
                  if all("bm25" not in F[m] or F[m]["bm25"][1] >= 0.05 for m in ("UFR", "CR", "coverage_ref")) else ""))
        b += [P(txt),
              ("table", dict(label="mabl", caption="Retrieval ablations: paired mean difference of each variant from the E1 base configuration (five-sentence chunks, top three, dense retrieval, citations required), with unadjusted Wilcoxon p-values. UFR and CR differences are in rate units, the others in percentage points. Negative UFR and CR differences and positive coverage and citation differences favor the variant.",
                             columns=["Variant", "Δ UFR (p)", "Δ CR (p)", "Δ coverage (p)", "Δ citation coverage (p)", "Δ citation accuracy (p)"],
                             widths=[1.75, 0.95, 0.95, 0.95, 0.95, 0.95], font=8.5, align=["left"] + ["center"] * 5, rows=rows))]

    # ── 7.8 Run-to-run variation
    rep = _csv("mimic_replicates.csv")
    if rep is not None and len(rep) and "kind" in rep.columns:
        var = rep[rep.kind == "variation"]; cmp_ = rep[rep.kind == "comparison"]
        vu = var[var.metric == "UFR"].set_index("condition"); vc = var[var.metric == "coverage_ref"].set_index("condition")
        b += [("h2", "7.8 Run-to-Run Variation")]
        n_rep = int(vu.n_docs.max()) if len(vu) else 0
        rows = [[NAME.get(c, c), f3(vu.loc[c, "mean"]), f3(vu.loc[c, "within_doc_sd"]), f3(vu.loc[c, "between_doc_sd"]), f2(vu.loc[c, "icc"]),
                 f"{100 * vc.loc[c, 'within_doc_sd']:.1f}" if c in vc.index else "—", f2(vc.loc[c, "icc"]) if c in vc.index else "—"]
                for c in ("E0", "E1", "E1b", "E3") if c in vu.index]
        comps = ["E1 vs E0", "E1b vs E0", "E3 vs E0", "E3 vs E1", "E1b vs E1"]
        crow = []
        stable, unstable = [], []
        for comp in comps:
            cells, sig_all, dirs = [], True, set()
            for g in ("base", "rep1", "rep2", "mean"):
                r = cmp_[(cmp_.metric == "UFR") & (cmp_.condition == comp) & (cmp_.generation == g)]
                if len(r):
                    r = r.iloc[0]; cells.append(f"{signed(r.delta_mean)} ({fpn(r.p)})")
                    if g != "mean":
                        sig_all &= r.p < 0.05; dirs.add(np.sign(r.delta_mean))
                else:
                    cells.append("—")
            crow.append([comp] + cells)
            (stable if sig_all and len(dirs) == 1 else unstable).append(comp)
        icc_lo, icc_hi = vu.icc.min(), vu.icc.max()
        txt = (f"Each summary was generated once, so E0, E1, E1b and E3 were generated twice more for {n_rep} randomly chosen courses "
               f"(Section 4.10.4). [[tab:mrepvar]] shows that for UFR the intraclass correlation is {f2(icc_lo)} to {f2(icc_hi)}: "
               + ("most of the variation lies between courses, so one generation characterizes a course reasonably well. "
                  if icc_lo >= 0.5 else
                  "a substantial part of the variation lies between generations of the same course, so a single per-course value is "
                  "noisy and only the paired condition comparisons should be read. ")
               + (f"[[tab:mrepcmp]] repeats the principal UFR comparisons in each generation: {_join(stable)} "
                  f"{'is' if len(stable) == 1 else 'are'} significant in the same direction in all three" if stable else "")
               + (f", whereas {_join(unstable)} change{'s' if len(unstable) == 1 else ''} sign or significance between generations, "
                  f"which is why the grounded conditions are not ranked against one another." if unstable else "."))
        b += [P(txt),
              ("table", dict(label="mrepvar", caption=f"Run-to-run variation over three generations on {n_rep} courses: mean UFR, standard deviation of UFR between generations of the same course and between courses, intraclass correlation (ICC), and the same for coverage (standard deviation in percentage points).",
                             columns=["Condition", "Mean UFR", "UFR SD within course", "UFR SD between courses", "UFR ICC", "Coverage SD within course", "Coverage ICC"],
                             widths=[1.75, 0.65, 0.85, 0.9, 0.6, 0.95, 0.75], font=8.5, rows=rows)),
              ("table", dict(label="mrepcmp", caption=f"Paired UFR differences (Wilcoxon p) on the {n_rep} courses in each generation and for the mean of the three generations.",
                             columns=["Comparison"] + [GEN_NAMES[g] for g in ("base", "rep1", "rep2", "mean")],
                             widths=[1.4, 1.3, 1.3, 1.3, 1.2], font=9, rows=crow))]

    # ── 7.9 Manual audit
    aud = _csv("audit_summary.csv")
    if aud is not None and len(aud) and "manual_unsupported" in aud.columns:
        A = aud[aud.manual_unsupported.notna()].set_index("condition")
        Dd = aud[aud["diff"].notna()].set_index("condition") if "diff" in aud.columns else None
        conds_a = [c for c in ("E0", "E1", "E1b", "E3") if c in A.index]
        b += [("h2", "7.9 Manual Audit of the Judge on the Main-Study Outputs")]
        n_lab = int(A.loc[conds_a, "n_labeled"].sum())
        rows = [[NAME.get(c, c), str(int(A.loc[c, "n_labeled"])), f"{pct(A.loc[c, 'manual_unsupported'])} ({pct(A.loc[c, 'manual_unsupported_lo'])}–{pct(A.loc[c, 'manual_unsupported_hi'])})",
                 pct(A.loc[c, "judge_flag_share_pipeline"]), f2(A.loc[c, "precision"]), f2(A.loc[c, "false_omission"]), f2(A.loc[c, "sensitivity"]),
                 f2(A.loc[c, "specificity"]), f2(A.loc[c, "kappa"])] for c in conds_a]
        over = [c for c in conds_a if A.loc[c, "judge_flag_share_pipeline"] > A.loc[c, "manual_unsupported_hi"]]
        under = [c for c in conds_a if A.loc[c, "judge_flag_share_pipeline"] < A.loc[c, "manual_unsupported_lo"]]
        txt = (f"The manual audit (Section 4.10.5) measures the judge on the summaries this generator wrote: the author labeled {n_lab} "
               f"sampled claims as supported, partly supported or not supported by the course, blind to the judge's label and the "
               f"condition. [[tab:maudit]] sets the manually confirmed share of unsupported claims, weighted to all of each condition's "
               f"claims, against the share the judge flags, with the judge's precision, false omission rate, sensitivity, specificity "
               f"and kappa against the manual labels. "
               + (f"The judge's share lies above the manual interval for {_join(over)}" if over else "")
               + (f"{' and ' if over else 'The judge'} {'below it' if over else 'flags fewer claims than the manual share'} for {_join(under)}, so the judge understates unsupported content there" if under else "")
               + ("; for every condition the judge's share lies within the manual interval" if not over and not under else "")
               + ". ")
        if "pooled" in A.index and not _nan(A.loc["pooled", "partly_share_of_unsupported"]):
            txt += (f"Most of what it misses is partial: {pct0(A.loc['pooled', 'partly_share_of_unsupported'])} of the manually "
                    f"unsupported claims were sentences in which some facts hold and one does not. ")
        ty = _csv("audit_types.csv")
        if ty is not None and len(ty):
            typed = ty[ty.type.notna() & ~ty.type.astype(str).str.strip().isin(["", "nan"])]
            tt = typed.groupby("type").n.sum().sort_values(ascending=False)
            if len(tt) and tt.sum() >= 0.6 * ty.n.sum():
                txt += ("The problems recorded were " + _join([f"{AUDIT_TYPE.get(k, str(k))} ({pct0(v / tt.sum())})" for k, v in tt.items()]) + ". ")
        if Dd is not None and len(Dd):
            keys_d = [k for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in Dd.index]
            confirmed = [k for k in keys_d if Dd.loc[k, "diff_hi"] < 0]
            parts = "; ".join(f"{k.split(' minus ')[0]} {pts(Dd.loc[k, 'diff'])} ({ci_pts(Dd.loc[k, 'diff_lo'], Dd.loc[k, 'diff_hi'])}) against {pts(Dd.loc[k, 'judge_diff'])} by the judge" for k in keys_d)
            txt += (f"For the differences between conditions, which carry the conclusions, the manual estimates are {parts}. "
                    + (f"The manual interval excludes zero for {_join([k.split(' minus ')[0] for k in confirmed])}, which confirms the "
                       f"direction of the judge's difference there; the other differences are unresolved at this sample size. "
                       if confirmed else "None of the manual intervals excludes zero, so the audit cannot confirm the judge's differences at its sample size. "))
        ag = _csv("audit_agreement.csv")
        if ag is not None and len(ag) and "claim" in set(ag.kind):
            ac = ag[ag.kind == "claim"].iloc[0]
            cg = ag[ag.kind == "citation"]
            txt += (f"The audit was labeled in two blind passes, the second after the rubric's paraphrase rule was clarified (Section "
                    f"4.10.5); the tables report the second pass, and on the {int(ac.n)} claims labeled twice the passes agree at kappa "
                    f"{f2(ac.kappa)} ({pct0(ac.agreement)} agreement"
                    + (f"; citation checks kappa {f2(cg.iloc[0].kappa)}" if len(cg) else "")
                    + "), with the most common change from partly supported to supported, in the direction of the clarified rule.")
        b += [P(txt),
              ("table", dict(label="maudit", caption="Manual audit of the selected judge on the main-study claims: number of claims labeled, manually confirmed share of unsupported claims weighted to the condition's claims (95 percent course-bootstrap interval), share of the condition's claims the judge flags, and the judge's precision, false omission rate, sensitivity, specificity and kappa against the manual labels.",
                             columns=["Condition", "Labeled", "Manual unsupported (95% CI)", "Judge flags", "Precision", "False omission", "Sensitivity", "Specificity", "Kappa"],
                             widths=[1.45, 0.55, 1.25, 0.55, 0.6, 0.6, 0.65, 0.65, 0.5], font=8, rows=rows))]

    # ── 7.10 Robustness to the judge and the generator
    b += [("h2", "7.10 Robustness to the Judge and the Generator")]
    alt_s, alt_t = _csv("mimic_tau05/mimic_summary.csv"), _csv("mimic_tau05/mimic_pairwise_tests.csv")
    if alt_s is not None and alt_t is not None and abs(D.tau - 0.5) > 1e-6:
        alt = alt_s[alt_s.variant == "base"].set_index("condition")
        def agree(metric):
            same = tot = 0
            for c in [x for x in conds if x not in ("E0", "REF")]:
                a = T(metric, f"{c} vs E0"); r = alt_t[(alt_t.metric == metric) & (alt_t.comparison == f"{c} vs E0")]
                if a is None or not len(r):
                    continue
                tot += 1
                sa, sb = a.p < 0.05, r.p.iloc[0] < 0.05
                same += int((not sa and not sb) or (sa and sb and (a.delta_mean < 0) == (r.delta_mean.iloc[0] < 0)))
            return same, tot
        su_, sc_, scv_ = agree("UFR"), agree("CR"), agree("coverage_ref")
        b += [P(f"*Threshold.* Scoring every summary at the pilot's threshold of 0.5 instead of the selected {D.tau:.2f} "
                f"{'raises' if alt.loc['E0', 'UFR_mean'] > D.m('E0', 'UFR_mean') else 'lowers'} the absolute rates ([[tab:mtau]]) but leaves "
                f"the paired comparisons against E0 unchanged in direction and significance for {su_[0]} of {su_[1]} UFR, {sc_[0]} of "
                f"{sc_[1]} CR and {scv_[0]} of {scv_[1]} coverage comparisons."),
              ("table", dict(label="mtau", caption=f"Main-study means at the selected threshold ({D.tau:.2f}) and at the pilot pipeline's threshold (0.5).",
                             columns=["Condition", f"UFR (τ={D.tau:.2f})", "UFR (τ=0.5)", f"CR (τ={D.tau:.2f})", "CR (τ=0.5)", f"Coverage (τ={D.tau:.2f})", "Coverage (τ=0.5)"],
                             widths=[1.9, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8], font=9,
                             rows=[[NAME.get(c, c), f3(D.m(c, 'UFR_mean')), f3(alt.loc[c, 'UFR_mean']), f3(D.m(c, 'CR_mean')), f3(alt.loc[c, 'CR_mean']),
                                    (pct(D.m(c, 'coverage_ref_mean')) if c != 'REF' else '—'), (pct(alt.loc[c, 'coverage_ref_mean']) if c != 'REF' else '—')]
                                   for c in conds if c in alt.index]))]
    wc = _csv("mimic_window_check.csv")
    vc = _csv("mimic_verifier_check.csv")
    txt = ""
    if wc is not None and len(wc):
        W = wc.set_index("condition")
        tot_flag = int(W.flagged.sum()); resc = float((W.flagged * W.rescued_by_overlap).sum() / tot_flag)
        txt += (f"*Evidence windows.* The judge reads the course in windows of at most 400 tokens, so evidence split across a window "
                f"boundary can cost a claim its support. Re-scoring the {tot_flag:,} flagged claims with half-overlapping windows turned "
                f"{pct(resc)} of them into supported claims"
                + (", similarly across conditions" if (W.rescued_by_overlap.max() - W.rescued_by_overlap.min()) < 0.1 else "")
                + ", so " + ("window boundaries explain only a small part of the flags. " if resc < 0.1 else
                             "the rates of Section 7.3 include this artifact, which affects the conditions alike. "))
    if vc is not None and len(vc):
        V = vc.set_index("group")
        if {"rejected by the verifier", "accepted by the verifier"} <= set(V.index):
            rj, ac = V.loc["rejected by the verifier"], V.loc["accepted by the verifier"]
            txt += (f"*Retrieval failures of the verifier.* Of the {int(rj.n)} E1 draft claims the E3 verifier rejected, the selected judge "
                    f"supports {pct0(rj.judge_supports_whole)} from the whole course but only {pct0(rj.judge_supports_top3)} from the "
                    f"verifier's three retrieved sentences, so {pct0(rj.whole_not_top3)} of the rejections are candidate retrieval "
                    f"failures: the course held support that retrieval did not return"
                    + (", which is consistent with verification not raising coverage. " if rj.whole_not_top3 >= 0.25 else ". "))
    if txt:
        b += [P(txt)]
    at = _csv("atomic_results.csv")
    if at is not None and len(at) and "condition" in at.columns:
        mc = at[at.analysis == "main_condition"].set_index("condition")
        mt = at[at.analysis == "main_test"]
        if len(mc):
            def at_t(metric, comp):
                r = mt[(mt.metric == metric) & (mt.comparison == comp)]
                return r.iloc[0] if len(r) else None
            low_f = [c for c in GROUNDED if at_t("fact_ufr", f"{c} vs E0") is not None and at_t("fact_ufr", f"{c} vs E0").p < 0.05 and at_t("fact_ufr", f"{c} vs E0").delta_mean < 0]
            rows = [[NAME.get(c, c), f"{mc.loc[c, 'facts_per_claim']:.2f}", pct0(mc.loc[c, "share_claims_mixed"]), f3(mc.loc[c, "fact_ufr"]),
                     f3(mc.loc[c, "sentence_any_fact_ufr"]), f3(D.m(c, "UFR_mean"))] for c in ORDER if c in mc.index]
            b += [P(f"*Atomic facts.* Splitting every claim into atomic facts and scoring each fact ([[tab:matomic]], Section 4.10.4) "
                    f"shows that claims carry {mc.facts_per_claim.min():.1f} to {mc.facts_per_claim.max():.1f} facts and that "
                    f"{pct0(mc.share_claims_mixed.min())} to {pct0(mc.share_claims_mixed.max())} of them mix supported and unsupported "
                    f"facts, which a sentence-level label cannot express. "
                    + (f"At the level of facts, {_join(low_f)} again had a lower rate than E0, so the main comparison does not depend on the unit of the claim."
                       if len(low_f) == len(GROUNDED) else
                       (f"At the level of facts only {_join(low_f)} had a significantly lower rate than E0." if low_f else
                        "At the level of facts no grounded condition had a significantly lower rate than E0."))),
                  ("table", dict(label="matomic", caption="Atomic-fact analysis of the main-study claims: mean facts per claim, share of claims mixing supported and unsupported facts, share of facts flagged by the judge (per-summary mean), share of claims with at least one flagged fact, and the sentence-level UFR of Section 7.3.",
                                 columns=["Condition", "Facts per claim", "Mixed claims", "Fact-level UFR", "Claims with a flagged fact", "Sentence-level UFR"],
                                 widths=[1.9, 0.8, 0.75, 0.85, 1.1, 0.95], font=9, rows=rows))]
    agr = _csv("mimic_judge_agreement.csv")
    if agr is not None and len(agr) and "all" in set(agr.condition):
        G = agr.set_index("condition"); g_all = G.loc["all"]
        _st = _csv("mimic_second_judge_tests.csv")
        st = _st[_st.measure == "UFR_second"].set_index("comparison") if _st is not None else None
        sb = _st[_st.measure == "UFR_both"].set_index("comparison") if _st is not None else None
        sjn = JUDGE_NAMES.get(g_all.second_judge, g_all.second_judge)
        same_order = all(G.loc[c, "UFR_second"] < G.loc["E0", "UFR_second"] and G.loc[c, "UFR_both"] < G.loc["E0", "UFR_both"]
                         for c in GROUNDED if c in G.index)
        txt = (f"*A second judge.* {sjn}, the strongest candidate of a different model family in Chapter 6, also scored every claim "
               f"([[tab:magree]]). The two judges agree on {pct0(g_all.agreement)} of claims (kappa {f2(g_all.kappa)}), and the table "
               f"adds two combined rules, conservative (flagged by both) and inclusive (flagged by either), which are alternative "
               f"estimates rather than bounds on the true rate, because both judges can share an error. "
               + ("Under the second judge and under both rules every grounded condition keeps a lower rate than E0 and the "
                  "clinician's instructions the highest. " if same_order else ""))
        if st is not None and "E3 vs E1" in st.index:
            s31 = st.loc["E3 vs E1"]
            txt += (f"The judges differ on verification: under the second judge E3 is below E1 ({f3(G.loc['E3', 'UFR_second'])} against "
                    f"{f3(G.loc['E1', 'UFR_second'])}, {fp(s31.p)}), whereas under the selected judge the two do not differ, so whether "
                    f"verification adds to retrieval depends on the judge and this thesis does not claim that it does.")
        b += [P(txt),
              ("table", dict(label="magree", caption="Agreement between the selected judge (S) and the second judge (2) on the base-condition claims: share of claims flagged by each, agreement and kappa, share of the selected judge's flags confirmed by the second judge, and per-document UFR under each judge, under the conservative rule (flagged by both) and under the inclusive rule (flagged by either).",
                             columns=["Condition", "Claims", "Flagged S", "Flagged 2", "Agree", "Kappa", "S flags confirmed", "UFR S", "UFR 2", "UFR, both", "UFR, either"],
                             widths=[0.85, 0.55, 0.6, 0.6, 0.5, 0.5, 0.75, 0.5, 0.5, 0.55, 0.6], font=8,
                             rows=[[{"REF": "Clinician", "all": "All"}.get(c, c), f"{int(r.n_claims):,}", pct0(r.flag_rate_first), pct0(r.flag_rate_second), pct0(r.agreement), f2(r.kappa),
                                    pct0(r.first_flags_confirmed), f3(r.UFR_first), f3(r.UFR_second), f3(r.UFR_both), f3(r.UFR_either)]
                                   for c, r in G.iterrows() if c in list(ORDER) + ["all"]]))]

    gens = []
    for tag, gname in (("llama31", "Llama-3.1-8B-Instruct"), ("mistral7b", "Mistral-7B-Instruct-v0.3")):
        gs, gpt = _csv(f"mimic_summary_{tag}.csv"), _csv(f"mimic_pairwise_tests_{tag}.csv")
        if gs is not None and gpt is not None and len(gs):
            gens.append((gname, gs[gs.variant == "base"].set_index("condition"), gpt))
    if gens:
        def g_ufr_t(gpt, comp):
            r = gpt[(gpt.metric == "UFR") & (gpt.comparison == comp)]
            return r.iloc[0] if len(r) else None
        gconds = [c for c in ("E0", "E1", "E1b", "E3", "REF") if all(c in gb.index for _, gb, _ in gens)]
        grows = [[NAME.get(c, c), f3(D.m(c, "UFR_mean"))] + [f3(gb.loc[c, "UFR_mean"]) for _, gb, _ in gens] for c in gconds]
        sents, repl_names, norepl_names = [], [], []
        for gname, gb, gpt in gens:
            res = []
            for c in GROUNDED:
                t = g_ufr_t(gpt, f"{c} vs E0")
                if t is None:
                    continue
                ph = float(t.p_holm_primary) if "p_holm_primary" in t.index and not _nan(t.p_holm_primary) else float(t.p)
                res.append((c, float(t.delta_mean), ph))
            sig = [r for r in res if r[2] < 0.05 and r[1] < 0]
            (repl_names if sig else norepl_names).append(gname)
            if res and len(sig) == len(res):
                sents.append(f"under {gname} every grounded condition is again significantly below E0 after the adjustment")
            elif sig:
                sents.append(f"under {gname} only {_join([c for c, _, _ in sig])} {'is' if len(sig) == 1 else 'are'} significantly below E0")
            else:
                worse = [c for c, d, _ in res if d > 0]
                sents.append(f"under {gname} no grounded condition differs significantly from E0 after the adjustment"
                             + (f", and {_join(worse)}'s point estimate is higher than E0's" if worse else ""))
        if not norepl_names:
            concl = f"The reductions therefore replicate on {_join(repl_names)}."
        elif repl_names:
            concl = (f"The effect of grounding is therefore generator-dependent: it replicates under {_join(repl_names)} but not under "
                     f"{_join(norepl_names)}, and the reductions of Section 7.3 are statements about the generators tested, not about "
                     f"grounding in general.")
        else:
            concl = f"The reductions of Section 7.3 did not replicate under {_join(norepl_names)}."
        b += [P(f"*Other generators.* The four text conditions were regenerated for all {D.n} courses with {_join([g for g, _, _ in gens])} "
                f"under the same prompts, decoding and retrieval, and scored by the same judge ([[tab:mgen]]; the clinician-written "
                f"reference scores identically in every run, as it must). " + (lambda x: x[0].upper() + x[1:])("; ".join(sents)) + ". " + concl),
              ("table", dict(label="mgen", caption="Judge-estimated UFR under three generators: per-document means on the same 110 courses, scored by the selected judge at the selected threshold. The reference row is identical by construction.",
                             columns=["Condition", "Qwen2.5-7B (main)"] + [g for g, _, _ in gens],
                             widths=[1.9] + [1.35] * (1 + len(gens)), font=9, rows=grows))]

    # ── 7.11 Question answering
    qa, qu = _csv("mimic_qa_summary.csv"), _csv("mimic_qa_utility.csv")
    if qa is not None and len(qa):
        b += [("h2", "7.11 Question-Answering Pilot")]
        Q = qa.set_index(["condition", "question"])
        U = qu.set_index(["condition", "question"]) if qu is not None else None
        tot = qa[qa.question == "all"].set_index("condition")
        rows = []
        for (c, q), r in Q.sort_index().iterrows():
            u = U.loc[(c, q)] if U is not None and (c, q) in U.index else None
            rows.append([c, QUESTION.get(q, q), pct0(r.abstention_rate), pct0(u.fully_supported_of_all) if u is not None else "—",
                         pct0(u.any_unsupported_of_all) if u is not None else "—", f3(r.UFR_mean), pct0(r.ref_support_mean)])
        txt = ("This exploratory pilot of the proposal's second task has no clinician-written gold answers, so it shows how abstention "
               "and grounding interact, not whether the answers are correct. Three questions per course (medications, follow-up, "
               "warning signs) were answered from the full course (QA-E0) or from three retrieved excerpts (QA-E1), with permission "
               "to answer \"Not stated in the note.\" (Section 4.11). [[tab:mqa]] separates the reliability of the answers given "
               "(UFR of answered questions) from what a user receives (the shares of all questions abstained on, answered fully "
               "supported, or answered with an unsupported sentence). ")
        if {"QA-E0", "QA-E1"} <= set(tot.index) and U is not None and ("QA-E0", "all") in U.index and ("QA-E1", "all") in U.index:
            u0, u1 = U.loc[("QA-E0", "all")], U.loc[("QA-E1", "all")]
            txt += (f"Retrieval {'raised' if tot.loc['QA-E1', 'abstention_rate'] > tot.loc['QA-E0', 'abstention_rate'] else 'lowered'} "
                    f"abstention from {pct0(tot.loc['QA-E0', 'abstention_rate'])} to {pct0(tot.loc['QA-E1', 'abstention_rate'])} of "
                    f"questions and {'lowered' if tot.loc['QA-E1', 'UFR_mean'] < tot.loc['QA-E0', 'UFR_mean'] else 'raised'} the "
                    f"judge-estimated UFR of the answers given ({f3(tot.loc['QA-E0', 'UFR_mean'])} to {f3(tot.loc['QA-E1', 'UFR_mean'])}), "
                    f"{'and also raised' if u1.fully_supported_of_all > u0.fully_supported_of_all else 'but did not raise'} the share of all "
                    f"questions that received a fully supported answer: part of its benefit to reliability is that the model says less. ")
        if ("QA-E0", "warning_signs") in Q.index and ("QA-E1", "warning_signs") in Q.index:
            w0, w1 = Q.loc[("QA-E0", "warning_signs")], Q.loc[("QA-E1", "warning_signs")]
            txt += (f"The question type dominates: warning signs are rarely stated in a course, so the model abstained most on that "
                    f"question ({pct0(w0.abstention_rate)} and {pct0(w1.abstention_rate)}) and its answers, drawn from general "
                    f"knowledge of discharge advice, had the highest rates. Agreement with the clinician's own instructions is low for "
                    f"every condition and is a proxy for correctness, not a measurement of it.")
        b += [P(txt),
              ("table", dict(label="mqa", caption="Question-answering pilot by condition and question type: share of questions abstained on, answered with every sentence supported by the course, and answered with at least one unsupported sentence (the three shares sum to one); judge-estimated UFR of the answered questions; and share of answer sentences supported by the clinician's discharge instructions.",
                             columns=["Condition", "Question", "Abstained", "Answered, all supported", "Answered, some unsupported", "UFR of answers", "Supported by clinician"],
                             widths=[0.75, 1.15, 0.75, 1.0, 1.05, 0.75, 0.95], font=8.5, rows=rows))]

    return b
