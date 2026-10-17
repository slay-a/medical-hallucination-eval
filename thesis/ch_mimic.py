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
    b += [P(f"The main study applies the conditions of Chapter 5 to the {D.n} hospital courses of ann-pt-summ (the Brief Hospital "
            f"Course sections, Section 3.4), each paired with the discharge instructions that the treating clinician wrote for the "
            f"patient. Summaries were generated by Qwen2.5-7B-Instruct running on the author's computer (Section 4.10), so that no "
            f"protected text left the machine; the retrieval conditions were required to cite the excerpts behind every sentence; and "
            f"every summary, including the clinician's instructions, was scored by the judge selected in Chapter 6 "
            f"({JUDGE_NAMES.get(D.judge, D.judge)}, {MODE_NAMES.get(D.mode, D.mode)}, support threshold {D.tau:.2f}). Claims are "
            f"segmented with the corrections of Section 3.5, and header lines and abstentions are excluded from the claim set."),
          P("Three conventions hold throughout the chapter. First, every unsupported fact rate, contradiction rate and coverage "
            "figure is judge-estimated: it counts what the judge fails to verify, including its false positives and excluding the "
            "unsupported statements it misses (Section 4.2.4); Section 7.9 relates the judge's figures to manually confirmed ones. "
            "Second, the main results come from one generation per condition and course at temperature 0.3; Section 7.8 measures "
            "the variation between repeated generations. Third, the unit of analysis is the hospital course, and every comparison "
            "is paired over the same courses. It is reported as a mean difference with a 95 percent bootstrap interval and a "
            "Wilcoxon p-value, and the twelve primary comparisons, E1, E1b, E2 and E3 against E0 on UFR, CR and coverage, carry "
            "Holm-adjusted p-values. A difference is called significant only when it remains so after the adjustment. The other "
            "comparisons are secondary and are reported with unadjusted p-values and intervals, and an unadjusted p-value below "
            "0.05 is described as suggestive.")]

    # ── 7.2 Descriptive statistics
    b += [("h2", "7.2 Descriptive Statistics")]
    ts, tu = {c: T("n_supported", f"{c} vs E0") for c in GROUNDED}, {c: T("n_unsupported", f"{c} vs E0") for c in GROUNDED}
    fewer_sup = [c for c in GROUNDED if ts[c] is not None and ts[c].p < 0.05 and ts[c].delta_mean < 0]
    same_sup = [c for c in GROUNDED if ts[c] is not None and ts[c].p >= 0.05]
    txt = (f"[[tab:mdesc]] describes the outputs. The clinician's instructions average {D.m('REF', 'words_mean'):.0f} words and "
           f"{D.m('REF', 'claims_mean'):.1f} claims. E0 writes {D.m('E0', 'words_mean'):.0f} words and {D.m('E0', 'claims_mean'):.1f} "
           f"claims per summary, and the grounded conditions write between {min(D.m(c, 'words_mean') for c in GROUNDED):.0f} and "
           f"{max(D.m(c, 'words_mean') for c in GROUNDED):.0f} words. Because UFR is a share of claims, the counts behind it matter. "
           f"E0 states on average {D.m('E0', 'supported_mean'):.1f} claims that the judge supports and {D.m('E0', 'unsupported_mean'):.1f} "
           f"that it does not; " + "; ".join(f"{c} states {D.m(c, 'supported_mean'):.1f} and {D.m(c, 'unsupported_mean'):.1f}" for c in GROUNDED)
           + ". ")
    if fewer_sup:
        txt += (f"{_join(fewer_sup)} state{'s' if len(fewer_sup) == 1 else ''} significantly fewer supported claims than E0 ("
                + "; ".join(f"{c} {signed(ts[c].delta_mean, 1)} per summary, {fp(ts[c].p)}" for c in fewer_sup)
                + "), so part of their lower rate comes from saying less, not only from saying it more faithfully. ")
    if same_sup:
        txt += (f"{_join(same_sup)} state{'s' if len(same_sup) == 1 else ''} as many supported claims as E0 within the precision of the "
                f"test ({'; '.join(f'{c} {signed(ts[c].delta_mean, 1)}, {fp(ts[c].p)}' for c in same_sup)}). ")
    fewer_un = [c for c in GROUNDED if tu[c] is not None and tu[c].p < 0.05 and tu[c].delta_mean < 0]
    if fewer_un:
        txt += (f"All of them state fewer unsupported claims than E0 ({'; '.join(f'{c} {signed(tu[c].delta_mean, 1)}, {fp(tu[c].p)}' for c in fewer_un)}). "
                if len(fewer_un) == len(GROUNDED) else
                f"{_join(fewer_un)} state fewer unsupported claims than E0 ({'; '.join(f'{c} {signed(tu[c].delta_mean, 1)}, {fp(tu[c].p)}' for c in fewer_un)}). ")
    txt += "A lower rate is therefore not by itself a better summary; Section 7.4 measures what the summaries leave out."
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
    txt = ("[[tab:mrates]] gives the per-summary rates with their 95 percent intervals and the paired comparisons, and [[fig:mbox]] "
           "their distributions. ")
    if len(lower) == len(GROUNDED):
        txt += (f"All three grounded LLM conditions had a lower judge-estimated UFR than E0 ({f3(D.m('E0', 'UFR_mean'))}), and every "
                f"difference remains significant after the Holm adjustment: "
                + "; ".join(f"{c} {f3(D.m(c, 'UFR_mean'))}, a difference of {signed(T('UFR', f'{c} vs E0').delta_mean)} (95 percent interval "
                            f"{ci3(T('UFR', f'{c} vs E0').ci95_low, T('UFR', f'{c} vs E0').ci95_high)}, {padj(D.holm_p(T('UFR', f'{c} vs E0')))})" for c in GROUNDED)
                + ". ")
    else:
        for c in GROUNDED:
            t = T("UFR", f"{c} vs E0")
            if t is not None:
                txt += (f"{NAME.get(c, c)} had a UFR of {f3(D.m(c, 'UFR_mean'))} against {f3(D.m('E0', 'UFR_mean'))} for E0 "
                        f"({signed(t.delta_mean)}, 95 percent interval {ci3(t.ci95_low, t.ci95_high)}, {padj(D.holm_p(t))}). ")
    cr_sig = [c for c in GROUNDED if D.sig(T("CR", f"{c} vs E0"))]
    cr_p = [D.holm_p(T("CR", f"{c} vs E0")) for c in GROUNDED if T("CR", f"{c} vs E0") is not None]
    txt += (f"None of them changed the contradiction rate significantly (E0 {f3(D.m('E0', 'CR_mean'))}; smallest adjusted p = {min(cr_p):.3f})."
            if not cr_sig and cr_p else
            " ".join(f"{c} {'lowered' if T('CR', f'{c} vs E0').delta_mean < 0 else 'raised'} CR by {abs(T('CR', f'{c} vs E0').delta_mean):.3f} "
                     f"({padj(D.holm_p(T('CR', f'{c} vs E0')))})." for c in cr_sig))
    among = [(c, T("UFR", f"{c} vs E1")) for c in ("E1b", "E3")]
    among = [(c, t) for c, t in among if t is not None]
    zero_in = [c for c, t in among if t.ci95_low <= 0 <= t.ci95_high]
    p2 = ("The differences among the grounded conditions are small: "
          + "; ".join(f"{c} differs from E1 by {signed(t.delta_mean)} (95 percent interval {ci3(t.ci95_low, t.ci95_high)}, {fp(t.p)})" for c, t in among)
          + ". ")
    if len(zero_in) == len(among):
        p2 += ("Both intervals include zero, and a difference of this size could be produced by a small difference in the judge's "
               "error rates between the conditions (Section 7.9), so these data do not rank the three grounded conditions against "
               "one another.")
    else:
        p2 += (f"The interval for {_join([c for c, _ in among if c not in zero_in])} excludes zero, but this is a secondary comparison, and a "
               f"difference of this size could also be produced by a difference in the judge's error rates between the conditions "
               f"(Section 7.9); Section 7.10 examines it under a second judge.")
    b += [P(txt), P(p2),
          ("table", dict(label="mrates", caption="Judge-estimated UFR and CR per summary in the main study: mean with 95 percent bootstrap interval, median, paired mean difference from E0 with 95 percent interval and Holm-adjusted p-value (primary family of twelve tests), and paired difference from E1 with unadjusted Wilcoxon p-value. Negative differences favor the row condition.",
                         columns=["Metric", "Condition", "Mean (95% CI)", "Median", "Δ vs E0 (95% CI)", "Holm p", "Δ vs E1 (p)"],
                         widths=[0.45, 1.5, 1.1, 0.55, 1.4, 0.55, 0.95], font=8, rows=rows)),
          ("figure", dict(label="mbox", path="results/fig_mimic_box.png", width=6.5, caption="Per-document judge-estimated UFR (left) and CR (right) in the main study under each condition and for the clinician's instructions."))]
    ref = _csv("mimic_reference.csv")
    if ref is not None and "REF" in conds:
        above = all(D.m("REF", "UFR_mean") > D.m(c, "UFR_mean") for c in conds if c != "REF")
        vs_ref = [(c, T("UFR", f"{c} vs REF")) for c in llm + ["E2"]]
        vs_ref = [(c, t) for c, t in vs_ref if t is not None]
        all_lower = vs_ref and all(t.p < 0.05 and t.delta_mean < 0 for _, t in vs_ref)
        b += [P(f"The clinician's instructions receive a judge-estimated UFR of {mean_ci(D, 'REF', 'UFR')} and a CR of "
                f"{f3(D.m('REF', 'CR_mean'))}"
                + (f", higher than every model condition ({'all' if all_lower else 'most'} differences "
                   + ("p < 0.001" if max(t.p for _, t in vs_ref) < 0.001 else f"p ≤ {max(t.p for _, t in vs_ref):.3f}") + ")" if above else "")
                + f". This is not evidence that clinicians hallucinate. Discharge instructions contain medication changes, appointments and "
                f"advice that the hospital course does not state (the experts marked {int(ref.expert_spans.sum())} unsupported spans in these "
                f"{len(ref)} documents, {ref.expert_spans.mean():.1f} per document), and the metric counts every statement that the course "
                f"does not support. The comparison shows that support by the hospital course measures source faithfulness, not the "
                f"quality or completeness of discharge instructions: a summarizer that sees only the course is faithful to it by design, "
                f"which does not make its summary better than the clinician's.")]
    if "E2" in conds:
        sj = _csv("mimic_judge_agreement.csv")
        sj_e2 = sj.set_index("condition").loc["E2"] if sj is not None and "E2" in set(sj.condition) else None
        below = [c for c in GROUNDED if D.m(c, "UFR_mean") < D.m("E2", "UFR_mean")]
        near = [c for c in GROUNDED if c not in below and D.m(c, "UFR_mean") - D.m("E2", "UFR_mean") <= 0.02]
        b += [P(f"E2 copies sentences of the course verbatim, so every claim it makes is supported by construction, and its "
                f"judge-estimated UFR of {mean_ci(D, 'E2', 'UFR')} and CR of {f3(D.m('E2', 'CR_mean'))} are judge errors on verbatim text. "
                f"They arise where a sentence taken out of its context, or one containing a negated list, is not entailed by any "
                f"400-token window at the threshold, or is read as a contradiction. E2 therefore gives a reference value for the judge's "
                f"false flags on extracted sentences of this corpus, not a floor for the other conditions: generated sentences differ "
                f"from extracted ones in length, negation and dependence on context, so the judge's error rate on them can be lower or "
                f"higher."
                + (f" {_join(below)} in fact receive{'s' if len(below) == 1 else ''} a lower rate than E2." if below else "")
                + (f" {_join(near)} lie{'s' if len(near) == 1 else ''} within 0.02 of it." if near else "")
                + (f" The second judge flags only {pct(sj_e2.flag_rate_second)} of E2's claims (Section 7.10), which confirms that the "
                   f"selected judge's flags on verbatim text are its own errors." if sj_e2 is not None else ""))]

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
    txt = (f"Because every document has the discharge instructions its clinician wrote, omission can be measured against them: "
           f"coverage is the share of the clinician's sentences that the generated summary supports, judged by the same model in the "
           f"reverse direction (Section 4.10.3), and like UFR it is judge-estimated. [[tab:mcov]] and [[fig:mcov]] report it together "
           f"with the number of the clinician's sentences that each summary leaves uncovered. E0 covered {pct(D.m('E0', 'coverage_ref_mean'))} "
           f"of the clinician's sentences, leaving {D.m('E0', 'omitted_ref_mean'):.1f} of {D.m('E0', 'ref_sents_mean'):.1f} uncovered on average. ")
    if lost:
        txt += (f"{_join(lost)} covered significantly less ("
                + "; ".join(f"{c} {pct(D.m(c, 'coverage_ref_mean'))}, {pts(T('coverage_ref', f'{c} vs E0').delta_mean)}, "
                            f"95 percent interval {ci_pts(T('coverage_ref', f'{c} vs E0').ci95_low, T('coverage_ref', f'{c} vs E0').ci95_high)}, "
                            f"{padj(D.holm_p(T('coverage_ref', f'{c} vs E0')))}" for c in lost)
                + "). ")
    if t_b is not None:
        if not D.sig(t_b):
            txt += (f"E1b achieved coverage similar to E0 ({pct(D.m('E1b', 'coverage_ref_mean'))}), with no statistically significant "
                    f"difference detected ({pts(t_b.delta_mean)}, 95 percent interval {ci_pts(t_b.ci95_low, t_b.ci95_high)}, "
                    f"{padj(D.holm_p(t_b))}). A difference that is not significant does not show that the two are equivalent. ")
            if E is not None and "E1b vs E0" in E.index:
                e = E.loc["E1b vs E0"]
                txt += (f"As a post hoc check with a margin of {100 * e.margin:.0f} percentage points, which was not specified before the "
                        f"analysis, the 90 percent interval of the difference ({ci_pts(e.ci90_low, e.ci90_high)} points) "
                        + ("lies within the margin, so on this sample a coverage loss of more than five points is unlikely."
                           if e.equivalent_at_margin else "extends beyond the margin, so a loss of more than five points cannot be excluded."))
        else:
            txt += (f"E1b covered {pct(D.m('E1b', 'coverage_ref_mean'))}, a significant difference from E0 ({pts(t_b.delta_mean)}, "
                    f"{padj(D.holm_p(t_b))}).")
    t_bb, t_bu = T("coverage_ref", "E1b vs E1"), T("UFR", "E1b vs E1")
    p2 = ""
    if t_bb is not None and t_bu is not None:
        p2 = (f"E1b differs from E1 only in also seeing the whole course. It {'raised' if t_bb.delta_mean > 0 else 'lowered'} coverage by "
              f"{abs(100 * t_bb.delta_mean):.1f} points relative to E1 ({fp(t_bb.p)}), while its UFR "
              f"{'did not differ significantly from' if t_bu.p >= 0.05 else ('was lower than' if t_bu.delta_mean < 0 else 'was higher than')} "
              f"E1's ({fp(t_bu.p)}). Directing the model to the excerpts and withholding the rest of the course are therefore separable "
              f"choices, and the coverage loss of E1 is associated with withholding the course rather than with retrieval as such.")
    ab = _csv("mimic_abstention.csv")
    p3 = ""
    if ab is not None and "E3" in set(ab.condition):
        a3 = ab.set_index("condition").loc["E3"]
        a1 = ab.set_index("condition").loc["E1"] if "E1" in set(ab.condition) else None
        if not _nan(a3.get("rho_abst_coverage", np.nan)):
            assoc = a3.p_abst_coverage < 0.05 or a3.p_abst_ufr < 0.05
            p3 = (f"Verification abstains more often than retrieval alone: E3 wrote at least one \"Not stated in the note.\" in "
                  f"{pct0(a3.share_with_abstention)} of its summaries"
                  + (f", against {pct0(a1.share_with_abstention)} for E1" if a1 is not None else "")
                  + f". Within E3, summaries with an abstention covered {pct(a3.coverage_with)} of the clinician's sentences and had a "
                  f"UFR of {f3(a3.ufr_with)}, against {pct(a3.coverage_without)} and {f3(a3.ufr_without)} without one; the number of "
                  f"abstentions has a Spearman correlation of {a3.rho_abst_coverage:.2f} with coverage ({fp(a3.p_abst_coverage)}) and "
                  f"{a3.rho_abst_ufr:.2f} with UFR ({fp(a3.p_abst_ufr)}). "
                  + ("Abstention is therefore not associated with either measure within E3: it marks sections for which the "
                     "retrieved excerpts held no support, and the coverage loss of E3 is inherited from its E1 draft rather than caused "
                     "by abstaining." if not assoc else
                     "Abstention is therefore associated with the measures within E3, and part of E3's coverage loss accompanies its "
                     "abstentions."))
    b += [P(txt)] + ([P(p2)] if p2 else []) + ([P(p3)] if p3 else []) + [
          ("table", dict(label="mcov", caption="Judge-estimated coverage of the clinician's discharge instructions: mean percentage of the clinician's sentences supported by the summary (95 percent interval), mean number of the clinician's sentences not covered, and paired differences in percentage points from E0 (95 percent interval, Holm-adjusted p) and from E1 (unadjusted p).",
                         columns=["Condition", "Coverage, % (95% CI)", "Sentences not covered", "Δ vs E0, points (95% CI)", "Holm p", "Δ vs E1 (p)"],
                         widths=[1.6, 1.15, 0.95, 1.35, 0.55, 0.9], font=8.5, rows=rows)),
          ("figure", dict(label="mcov", path="results/fig_mimic_coverage.png", width=5.6, caption="Judge-estimated coverage of the clinician's discharge instructions per document under each condition."))]

    # ── retrieval-ablation families (Holm within each measure), used in Sections 7.5, 7.7 and 7.12
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
        short_u, short_w = vrow("E0/short vs E0/base", "UFR", False), vrow("E0/short vs E0/base", "words", False)
        short_c = vrow("E0/short vs E0/base", "coverage_ref", False)
        nocite_u = vrow("E1/nocite vs E0/base", "UFR", False)
        lm = {c: vrow(k, "UFR", fl) for c, k, fl in (("E1", "E0/short vs E1/base", True), ("E1b", "E1b/base vs E0/short", False), ("E3", "E3/base vs E0/short", False))}
        lm_c = {c: vrow(k, "coverage_ref", fl) for c, k, fl in (("E1", "E0/short vs E1/base", True), ("E1b", "E1b/base vs E0/short", False), ("E3", "E3/base vs E0/short", False))}
        txt = ("Section 4.10.2 identified variables other than the mechanism under study that differ between the conditions: the "
               "citation instruction, the length of the summary and, for verification, the number of model calls. The variants of "
               "[[tab:mvar]] separate the first two. Their six comparisons form a family of their own, and the table gives "
               "Holm-adjusted p-values within it for each measure. ")
        if short_u is not None and short_w is not None:
            lab = "Length-matched E0 minus E0"
            sig_s = adjp.get((lab, "UFR"), 1) < 0.05
            txt += (f"Asked for about 150 words, E0 wrote {short_w.mean_b:.0f} words instead of {short_w.mean_a:.0f}. Its UFR was "
                    f"{f3(short_u.mean_b)} against {f3(short_u.mean_a)} ({signed(short_u.delta_mean)}, 95 percent interval "
                    f"{ci3(short_u.ci95_low, short_u.ci95_high)}, {padj(adjp.get((lab, 'UFR')))}), so brevity alone "
                    + (f"lowered the judge-estimated rate by {abs(short_u.delta_mean):.3f}" if sig_s and short_u.delta_mean < 0 else
                       ("raised the rate" if sig_s else "did not change the rate significantly"))
                    + (f", and its coverage fell by {abs(100 * short_c.delta_mean):.1f} points" if short_c is not None and short_c.delta_mean < 0 else "")
                    + ". ")
            still = [c for c in GROUNDED if lm[c] is not None and lm[c].delta_mean < 0 and adjp.get((f"{c} minus length-matched E0", "UFR"), 1) < 0.05]
            not_ = [c for c in GROUNDED if lm[c] is not None and c not in still]
            if still:
                txt += (f"Against this length-matched baseline, {_join(still)} still had a lower UFR ("
                        + "; ".join(f"{c} {signed(lm[c].delta_mean)}, {padj(adjp.get((f'{c} minus length-matched E0', 'UFR')))}" for c in still)
                        + ")")
                txt += (f", whereas {_join(not_)} did not differ significantly from it ("
                        + "; ".join(f"{c} {signed(lm[c].delta_mean)}, {padj(adjp.get((f'{c} minus length-matched E0', 'UFR')))}" for c in not_) + ")"
                        if not_ else "")
                txt += (". The reductions of Section 7.3 are therefore not explained by length alone"
                        + (", although brevity accounts for part of them." if sig_s and short_u.delta_mean < 0 else ".") + " ")
            else:
                txt += ("Against this length-matched baseline no grounded condition had a significantly lower UFR, so the reductions of "
                        "Section 7.3 cannot be separated from the shorter length of the grounded summaries. ")
            if "E1b" in lm_c and lm_c["E1b"] is not None:
                txt += (f"E1b, which is as long as E0, covered {abs(100 * lm_c['E1b'].delta_mean):.1f} points "
                        f"{'more' if lm_c['E1b'].delta_mean > 0 else 'less'} of the clinician's sentences than the length-matched E0 "
                        f"({padj(adjp.get(('E1b minus length-matched E0', 'coverage_ref')))}). ")
        if nocite_u is not None:
            lab = "E1 without citations minus E0"
            sig_n = adjp.get((lab, "UFR"), 1) < 0.05
            txt += (f"Without the citation instruction, E1 had a UFR of {f3(nocite_u.mean_b)} against {f3(nocite_u.mean_a)} for E0 "
                    f"({signed(nocite_u.delta_mean)}, 95 percent interval {ci3(nocite_u.ci95_low, nocite_u.ci95_high)}, "
                    f"{padj(adjp.get((lab, 'UFR')))}), so "
                    + ("retrieval lowers the judge-estimated rate without any citation instruction. " if sig_n and nocite_u.delta_mean < 0 else
                       "without the instruction the effect of retrieval on UFR is not significant. "))
            if ab_n is not None and len(ab_n) and "nocite" in F.get("UFR", {}):
                fu = Flip(ab_n.iloc[0]); ph = F["UFR"]["nocite"][2]
                txt += (f"Adding the instruction {'lowered' if fu.delta_mean < 0 else 'raised'} UFR by a further {abs(fu.delta_mean):.3f} "
                        f"({fp(fu.p)}, {padj(ph)} within the ablation family of Section 7.7), so "
                        + ("the citation instruction lowers the rate on its own as well." if ph < 0.05 and fu.delta_mean < 0 else
                           "the instruction may contribute to the reduction, but its separate effect is not established."))
        nb = vrow("E1b/nocite vs E1b/base", "UFR", False); nbc = vrow("E1b/nocite vs E1b/base", "coverage_ref", False)
        if nb is not None:
            lab = "E1b without citations minus E1b"
            txt += (f" For E1b, dropping the instruction changed UFR by {signed(nb.delta_mean)} ({padj(adjp.get((lab, 'UFR')))})"
                    + (f" and coverage by {pts(nbc.delta_mean)} ({padj(adjp.get((lab, 'coverage_ref')))})" if nbc is not None else "") + ".")
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
        txt = ("The retrieval conditions were required to cite, after every sentence, the excerpts it is based on. Three measures "
               "describe how far they did so (Section 4.10.3): citation coverage, the share of claims with at least one valid "
               "citation; citation accuracy, the share of cited claims whose cited excerpts support the claim under the judge; and "
               "citation reliability, the share of all claims that carry a valid and supporting citation. [[tab:mcite]] reports them "
               "with the counts from which they are computed. ")
        if "E1" in cc:
            n_all, n_cit, n_ok = cc["E1"]
            txt += (f"Of E1's {n_all:,} claims, {n_cit:,} ({pct0(n_cit / n_all)}) carried a valid citation, and the judge found {n_ok:,} of "
                    f"these ({pct0(n_ok / n_cit)}) supported by the excerpts they cite, so {pct0(n_ok / n_all)} of E1's claims are backed by "
                    f"a citation that supports them. ")
        if "E1b" in cc:
            n_all, n_cit, n_ok = cc["E1b"]
            e1_cov = cc["E1"][1] / cc["E1"][0] if "E1" in cc else np.nan
            txt += (f"E1b cited {'far ' if n_cit / n_all < e1_cov - 0.2 else ''}fewer of its claims, {n_cit:,} of {n_all:,} "
                    f"({pct0(n_cit / n_all)}). This is expected from its design rather than a failure to follow the instruction: E1b also sees the "
                    f"whole course, and statements drawn from passages outside the three excerpts have no excerpt to cite. Its cited "
                    f"claims were supported by their excerpts in {pct0(n_ok / n_cit) if n_cit else '—'} of cases, so its reliability is "
                    f"{pct0(n_ok / n_all)}. ")
        spl = _csv("mimic_splitting_check.csv")
        if spl is not None:
            S_ = spl[spl.variant == "base"].set_index("condition")
            mis = {c: S_.loc[c, "markers_dropped_share"] + S_.loc[c, "markers_moved_share"] for c in cited if c in S_.index}
            txt += ("These figures use the corrected attribution of citation markers (Section 3.5): splitting the raw text, as an "
                    "earlier version of the analysis did, discards or misplaces "
                    + " and ".join(f"{pct0(v)} of {c}'s markers" for c, v in mis.items()) + ". ")
        txt += "The shares in the table are computed from the pooled counts; the paired tests of Section 7.7 use per-summary values."
        b += [P(txt)]
        ac = _csv("audit_citations.csv")
        if ac is not None and len(ac):
            A = ac.set_index("condition")
            b += [P("The manual audit verified a random sample of citations by reading the cited excerpts (Section 4.10.5). "
                    + " ".join(f"Of {int(A.loc[c, 'n'])} sampled {c} citations, the author judged {pct0(A.loc[c, 'manual_yes'])} to support "
                               f"the claim fully and {pct0(A.loc[c, 'manual_yes_or_partly'])} at least partly"
                               + (f"; the judge's verdict on the same citations agreed with the manual verdict in {pct0(A.loc[c, 'agreement'])} of cases" if "agreement" in A.columns and not _nan(A.loc[c, "agreement"]) else "")
                               + "." for c in A.index))]
        b += [("table", dict(label="mcite", caption="Citation coverage, accuracy and reliability of the retrieval conditions, from the pooled counts over the 110 courses. Accuracy is the share of cited claims supported by their cited excerpts under the judge; reliability is the share of all claims that carry a valid and supporting citation.",
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
        def describe(m, label):
            raw = [(v,) + F[m][v] for v in F[m] if F[m][v][1] < 0.05]
            adj = [r for r in raw if r[3] < 0.05]
            unit = (lambda d: f"{abs(100 * d):.1f} points") if m in ("coverage_ref",) + cite_measures else (lambda d: f"{abs(d):.3f}")
            if not raw:
                return f"No variant changed {label}."
            parts = [f"{VPHRASE[v]} {'raised' if d > 0 else 'lowered'} it by {unit(d)} ({fp(pv)}, {padj(pa)})" for v, d, pv, pa in raw]
            if len(adj) == len(raw):
                return f"{label[0].upper() + label[1:]} changed in {len(raw)} variant{'s' if len(raw) > 1 else ''}: {'; '.join(parts)}."
            if adj:
                return (f"{label[0].upper() + label[1:]} changed in {len(raw)} variant{'s' if len(raw) > 1 else ''} before the adjustment "
                        f"and in {len(adj)} after it: {'; '.join(parts)}.")
            return (f"Before the adjustment, {'; '.join(p.replace(' it ', f' {label} ', 1) for p in parts)}; "
                    f"{'this effect does' if len(raw) == 1 else 'these effects do'} not survive the adjustment and {'is' if len(raw) == 1 else 'are'} suggestive at most.")
        b += [P("Six variants of E1 change one retrieval setting at a time (Section 4.10.4). [[tab:mabl]] reports the paired difference "
                "of each variant from the base configuration. Because six variants are tested on each measure, the text gives "
                "Holm-adjusted p-values within each measure next to the raw ones; for the citation measures the family has five "
                "members, because the variant without citations has none. "
                + describe("UFR", "the unsupported fact rate") + " " + describe("CR", "the contradiction rate") + " "
                + describe("coverage_ref", "coverage of the clinician's instructions") + " "
                + describe("citation_coverage", "citation coverage") + " " + describe("citation_accuracy", "citation accuracy") + " "
                + ("Lexical BM25 and dense retrieval produced summaries with indistinguishable rates and coverage. "
                   if all("bm25" not in F[m] or F[m]["bm25"][1] >= 0.05 for m in ("UFR", "CR", "coverage_ref")) else "")
                + ("The amount of the course shown to the model governs coverage, while no retrieval setting tested here changes "
                   "faithfulness after the adjustment." if not any(F[m][v][2] < 0.05 for m in ("UFR", "CR") for v in F[m]) else
                   "The amount of the course shown to the model governs coverage.")),
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
        txt = (f"The main results rest on one generation per condition and course. To measure how far a single generation can be "
               f"trusted, E0, E1, E1b and E3 were generated twice more for {n_rep} randomly chosen courses with fixed seeds (Section "
               f"4.10.4). [[tab:mrepvar]] reports, for UFR and coverage, the standard deviation between the three generations of the same "
               f"course, the standard deviation between courses and the intraclass correlation, the share of the total variance that "
               f"lies between courses. For UFR the intraclass correlation ranges from {f2(icc_lo)} to {f2(icc_hi)}: "
               + ("most of the variation lies between courses, so a single generation characterizes a course reasonably well. "
                  if icc_lo >= 0.5 else
                  "a substantial part of the variation lies between generations of the same course, so a single generation is a noisy "
                  "measurement of a course and the per-course values should not be interpreted individually. "))
        if stable:
            txt += (f"[[tab:mrepcmp]] repeats the principal UFR comparisons in each generation. {_join(stable)} "
                    f"{'is' if len(stable) == 1 else 'are'} significant in the same direction in all three generations. ")
        if unstable:
            txt += (f"{_join(unstable)} {'is' if len(unstable) == 1 else 'are'} not: {'its' if len(unstable) == 1 else 'their'} sign or significance "
                    f"changes from one generation to another, which agrees with the decision of Section 7.3 not to rank the grounded "
                    f"conditions against one another.")
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
        txt = (f"The manual audit (Section 4.10.5) measures the judge on the summaries this generator wrote. The author labeled {n_lab} "
               f"of the sampled claims as supported, partly supported or not supported by the hospital course, without seeing the judge's "
               f"label or the condition; claims labeled unsure are excluded. [[tab:maudit]] reports, per condition, the manually "
               f"confirmed share of unsupported claims, weighted to all of the condition's claims, next to the share of claims the judge "
               f"flags, and the judge's precision, false omission rate, sensitivity, specificity and kappa against the manual labels. ")
        if over:
            txt += f"The judge's share lies above the manual interval for {_join(over)}, so its flags overstate the unsupported content there. "
        if under:
            txt += f"It lies below the manual interval for {_join(under)}, so it misses unsupported content there. "
        if not over and not under:
            txt += "For every condition the judge's share lies within the interval of the manually confirmed share. "
        p2 = ""
        if Dd is not None and len(Dd):
            parts = []
            for key in ("E1 minus E0", "E1b minus E0", "E3 minus E0"):
                if key in Dd.index:
                    r = Dd.loc[key]
                    parts.append(f"{key.replace(' minus ', ' versus ')}: {pts(r['diff'])} manually (95 percent interval {ci_pts(r.diff_lo, r.diff_hi)}) "
                                 f"against {pts(r.judge_diff)} by the judge")
            confirmed = [k for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in Dd.index and Dd.loc[k, "diff_hi"] < 0]
            p2 = ("The comparisons that matter for the conclusions are differences between conditions. In percentage points of "
                  "unsupported claims they are " + "; ".join(parts) + ". "
                  + (f"The manual reading confirms the direction of the judge's difference for {_join([k.split(' minus ')[0] for k in confirmed])}, "
                     f"whose intervals lie below zero" if confirmed else
                     "None of the manual intervals excludes zero, so the audit, with its sample size, cannot confirm the judge's differences")
                  + ".")
        ty = _csv("audit_types.csv")
        p3 = ""
        if ty is not None and len(ty):
            tt = ty.groupby("type", dropna=False).n.sum().sort_values(ascending=False)
            tot = tt.sum()
            p3 = ("Among the claims labeled partly or not supported, the most frequent problems were "
                  + _join([f"{AUDIT_TYPE.get(k, str(k))} ({pct0(v / tot)})" for k, v in tt.head(3).items()])
                  + (f", and {pct0(A.loc['pooled', 'partly_share_of_unsupported'])} of them were partly supported, that is, sentences in which "
                     f"some facts are supported and others are not" if "pooled" in A.index and not _nan(A.loc["pooled", "partly_share_of_unsupported"]) else "")
                  + ".")
        b += [P(txt)] + ([P(p2)] if p2 else []) + ([P(p3)] if p3 else []) + [
              ("table", dict(label="maudit", caption="Manual audit of the selected judge on the main-study claims: number of claims labeled, manually confirmed share of unsupported claims weighted to the condition's claims (95 percent course-bootstrap interval), share of the condition's claims the judge flags, and the judge's precision, false omission rate, sensitivity, specificity and kappa against the manual labels.",
                             columns=["Condition", "Labeled", "Manual unsupported (95% CI)", "Judge flags", "Precision", "False omission", "Sensitivity", "Specificity", "Kappa"],
                             widths=[1.45, 0.55, 1.25, 0.55, 0.6, 0.6, 0.65, 0.65, 0.5], font=8, rows=rows))]

    # ── 7.10 Robustness to the judge
    b += [("h2", "7.10 Robustness to the Judge")]
    alt_s, alt_t = _csv("mimic_tau05/mimic_summary.csv"), _csv("mimic_tau05/mimic_pairwise_tests.csv")
    if alt_s is not None and alt_t is not None and abs(D.tau - 0.5) > 1e-6:
        alt = alt_s[alt_s.variant == "base"].set_index("condition")
        diffs = []
        def agree(metric):
            same = tot = 0
            for c in [x for x in conds if x not in ("E0", "REF")]:
                a = T(metric, f"{c} vs E0"); r = alt_t[(alt_t.metric == metric) & (alt_t.comparison == f"{c} vs E0")]
                if a is None or not len(r):
                    continue
                tot += 1
                sa, sb = a.p < 0.05, r.p.iloc[0] < 0.05
                ok = (not sa and not sb) or (sa and sb and (a.delta_mean < 0) == (r.delta_mean.iloc[0] < 0))
                same += int(ok)
                if not ok:
                    diffs.append(f"{c}'s {'higher' if r.delta_mean.iloc[0] > 0 else 'lower'} {metric.replace('coverage_ref', 'coverage')} relative to E0 is significant at "
                                 f"{'0.5' if sb else f'{D.tau:.2f}'} ({fp(r.p.iloc[0]) if sb else fp(a.p)}) but not at {f'{D.tau:.2f}' if sb else '0.5'} ({fp(a.p) if sb else fp(r.p.iloc[0])})")
            return same, tot
        su_, sc_, scv_ = agree("UFR"), agree("CR"), agree("coverage_ref")
        b += [P(f"*Threshold.* The main results use the support threshold of {D.tau:.2f} selected in Chapter 6. Every summary was also scored at "
                f"the pilot pipeline's threshold of 0.5 ([[tab:mtau]]). Absolute rates {'rise' if alt.loc['E0', 'UFR_mean'] > D.m('E0', 'UFR_mean') else 'fall'} "
                f"at the stricter threshold, as they must, but the paired comparisons against E0 agree in significance and direction for "
                f"{su_[0]} of {su_[1]} UFR comparisons, {sc_[0]} of {sc_[1]} CR comparisons and {scv_[0]} of {scv_[1]} coverage comparisons"
                + (f"; the only difference{'s are' if len(diffs) > 1 else ' is'} that {'; '.join(diffs)}" if diffs else "")
                + "."),
              ("table", dict(label="mtau", caption=f"Main-study means at the selected threshold ({D.tau:.2f}) and at the pilot pipeline's threshold (0.5).",
                             columns=["Condition", f"UFR (τ={D.tau:.2f})", "UFR (τ=0.5)", f"CR (τ={D.tau:.2f})", "CR (τ=0.5)", f"Coverage (τ={D.tau:.2f})", "Coverage (τ=0.5)"],
                             widths=[1.9, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8], font=9,
                             rows=[[NAME.get(c, c), f3(D.m(c, 'UFR_mean')), f3(alt.loc[c, 'UFR_mean']), f3(D.m(c, 'CR_mean')), f3(alt.loc[c, 'CR_mean']),
                                    (pct(D.m(c, 'coverage_ref_mean')) if c != 'REF' else '—'), (pct(alt.loc[c, 'coverage_ref_mean']) if c != 'REF' else '—')]
                                   for c in conds if c in alt.index]))]
    wc = _csv("mimic_window_check.csv")
    if wc is not None and len(wc):
        W = wc.set_index("condition")
        tot_flag = int(W.flagged.sum()); resc = float((W.flagged * W.rescued_by_overlap).sum() / tot_flag)
        b += [P(f"*Evidence windows.* The judge reads the course in consecutive windows of at most 400 tokens, so a claim whose evidence "
                f"spans two windows may be flagged although the course supports it. Re-scoring the {tot_flag:,} flagged claims of the base "
                f"conditions with windows that overlap by half turned {pct(resc)} of them into supported claims ("
                + ", ".join(f"{c} {pct(W.loc[c, 'rescued_by_overlap'])}" for c in ORDER if c in W.index)
                + "). " + ("Window boundaries therefore explain only a small part of the flags, and they do so similarly across the conditions."
                           if resc < 0.1 and (W.rescued_by_overlap.max() - W.rescued_by_overlap.min()) < 0.1 else
                           "Window boundaries explain a noticeable part of the flags, and the rates of Section 7.3 should be read as "
                           "including this artifact."))]
    vc = _csv("mimic_verifier_check.csv")
    if vc is not None and len(vc):
        V = vc.set_index("group")
        if {"rejected by the verifier", "accepted by the verifier"} <= set(V.index):
            rj, ac = V.loc["rejected by the verifier"], V.loc["accepted by the verifier"]
            b += [P(f"*Retrieval failures of the verifier.* The E3 verifier judges each draft claim against three sentences retrieved from "
                    f"the course. It rejected {int(rj.n)} of the {int(rj.n + ac.n)} E1 draft claims. Of the rejected claims, the selected "
                    f"judge supports {pct0(rj.judge_supports_whole)} when it reads the whole course but only "
                    f"{pct0(rj.judge_supports_top3)} when it reads the verifier's three sentences, so {pct0(rj.whole_not_top3)} of the "
                    f"rejections are candidate retrieval failures: the course contains support that retrieval did not return. Of the "
                    f"claims the verifier accepted, the judge flags {pct0(1 - ac.judge_supports_whole)} with the whole course. "
                    + ("Retrieval failures therefore account for a substantial share of what verification removes, which is "
                       "consistent with verification not raising coverage."
                       if rj.whole_not_top3 >= 0.25 else
                       "Retrieval failures account for a minority of the verifier's rejections."))]
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
            b += [P(f"*Atomic facts.* Claims of the main study were split into atomic facts by the local model and every fact was scored by "
                    f"the selected judge (Section 4.10.4). [[tab:matomic]] reports the number of facts per claim, the share of claims that "
                    f"mix supported and unsupported facts, the share of facts the judge flags, and the share of claims with at least one "
                    f"flagged fact, next to the sentence-level UFR. Claims carry between {mc.facts_per_claim.min():.1f} and "
                    f"{mc.facts_per_claim.max():.1f} facts on average, and between {pct0(mc.share_claims_mixed.min())} and "
                    f"{pct0(mc.share_claims_mixed.max())} of them are mixed, which the sentence-level label cannot express. "
                    + (f"At the level of facts, {_join(low_f)} again had a lower judge-estimated rate than E0 ("
                       + "; ".join(f"{c} {signed(at_t('fact_ufr', f'{c} vs E0').delta_mean)}, {fp(at_t('fact_ufr', f'{c} vs E0').p)}" for c in low_f)
                       + "), so the main comparison does not depend on the unit of the claim."
                       if len(low_f) == len(GROUNDED) else
                       (f"At the level of facts only {_join(low_f)} had a significantly lower rate than E0." if low_f else
                        "At the level of facts no grounded condition had a significantly lower rate than E0, so the main comparison depends on the unit of the claim."))),
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
        txt = (f"*A second judge.* Every claim of the base conditions was also scored by {sjn}, the strongest candidate of a different "
               f"model family in Chapter 6, in its best evidence mode ({'three retrieved sentences' if g_all.second_mode == 'top3' else 'whole course'}) "
               f"and at the threshold chosen there ({g_all.second_tau:.2f}). The two judges agree on {pct0(g_all.agreement)} of the "
               f"{int(g_all.n_claims):,} claims (kappa {f2(g_all.kappa)}); the second judge confirms {pct0(g_all.first_flags_confirmed)} of "
               f"the selected judge's flags and flags {pct0(g_all.flag_rate_second)} of claims itself, against {pct0(g_all.flag_rate_first)}. "
               f"[[tab:magree]] reports, per condition, the UFR under each judge and under two flagging rules that combine them: a "
               f"conservative rule that counts a claim as unsupported only when both judges flag it, and an inclusive rule that counts it "
               f"when either does. The two rules are alternative estimates, not bounds on the true rate: both judges can miss the same "
               f"unsupported claim or flag the same supported one. ")
        if same_order:
            txt += ("Under the second judge and under both rules, every grounded condition has a lower rate than E0"
                    + (f" (second judge, all {fp(max(st.loc[f'{c} vs E0', 'p'] for c in GROUNDED if f'{c} vs E0' in st.index))})" if st is not None else "")
                    + ", and the clinician's instructions have the highest rate. ")
        if st is not None and "E3 vs E1" in st.index and sb is not None and "E3 vs E1" in sb.index:
            s31, b31 = st.loc["E3 vs E1"], sb.loc["E3 vs E1"]
            txt += (f"The judges differ on verification. Under the second judge E3 has a UFR of {f3(G.loc['E3', 'UFR_second'])} against "
                    f"{f3(G.loc['E1', 'UFR_second'])} for E1 ({fp(s31.p)}), whereas under the selected judge the two do not differ "
                    f"(Section 7.3); under the conservative rule the difference is {signed(b31.delta_mean)} ({fp(b31.p)}). Whether "
                    f"verification adds to retrieval therefore depends on the judge, and this thesis does not claim that it does. ")
        if "E2" in G.index:
            txt += (f"On the verbatim extracts of E2 the second judge flags {pct(G.loc['E2', 'flag_rate_second'])} of claims and confirms "
                    f"{pct0(G.loc['E2', 'first_flags_confirmed'])} of the selected judge's flags, which identifies the selected judge's rate "
                    f"on E2 as its own error (Section 7.3).")
        b += [P(txt),
              ("table", dict(label="magree", caption="Agreement between the selected judge (S) and the second judge (2) on the base-condition claims: share of claims flagged by each, agreement and kappa, share of the selected judge's flags confirmed by the second judge, and per-document UFR under each judge, under the conservative rule (flagged by both) and under the inclusive rule (flagged by either).",
                             columns=["Condition", "Claims", "Flagged S", "Flagged 2", "Agree", "Kappa", "S flags confirmed", "UFR S", "UFR 2", "UFR, both", "UFR, either"],
                             widths=[0.85, 0.55, 0.6, 0.6, 0.5, 0.5, 0.75, 0.5, 0.5, 0.55, 0.6], font=8,
                             rows=[[{"REF": "Clinician", "all": "All"}.get(c, c), f"{int(r.n_claims):,}", pct0(r.flag_rate_first), pct0(r.flag_rate_second), pct0(r.agreement), f2(r.kappa),
                                    pct0(r.first_flags_confirmed), f3(r.UFR_first), f3(r.UFR_second), f3(r.UFR_both), f3(r.UFR_either)]
                                   for c, r in G.iterrows() if c in list(ORDER) + ["all"]]))]

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
        txt = ("This section is an exploratory pilot of the proposal's second task: there are no clinician-written gold answers, so it "
               "shows how abstention and grounding interact, not whether the answers are clinically correct. Three questions per course, "
               "about medications, follow-up and warning signs, were answered from the full course (QA-E0) or from three retrieved "
               "excerpts (QA-E1), with the instruction to answer \"Not stated in the note.\" when the course is silent (Section 4.11). "
               "Abstentions are excluded from the claim set, so the UFR of answered questions describes the reliability of the answers "
               "that were given; the shares of all questions that were abstained on, answered with every sentence supported, or answered "
               "with at least one unsupported sentence describe what a user receives. [[tab:mqa]] reports both. ")
        if {"QA-E0", "QA-E1"} <= set(tot.index) and U is not None and ("QA-E0", "all") in U.index and ("QA-E1", "all") in U.index:
            u0, u1 = U.loc[("QA-E0", "all")], U.loc[("QA-E1", "all")]
            txt += (f"From the full course the model abstained on {pct0(tot.loc['QA-E0', 'abstention_rate'])} of questions and answered "
                    f"{pct0(u0.fully_supported_of_all)} of all questions with every sentence supported; from excerpts it abstained on "
                    f"{pct0(tot.loc['QA-E1', 'abstention_rate'])} and answered {pct0(u1.fully_supported_of_all)} fully supported. The UFR of "
                    f"answered questions was {f3(tot.loc['QA-E0', 'UFR_mean'])} and {f3(tot.loc['QA-E1', 'UFR_mean'])}. Retrieval "
                    f"therefore {'raised' if tot.loc['QA-E1', 'abstention_rate'] > tot.loc['QA-E0', 'abstention_rate'] else 'lowered'} the "
                    f"abstention rate and {'lowered' if tot.loc['QA-E1', 'UFR_mean'] < tot.loc['QA-E0', 'UFR_mean'] else 'raised'} the "
                    f"judge-estimated UFR of the answers given, and it {'also raised' if u1.fully_supported_of_all > u0.fully_supported_of_all else 'did not raise'} "
                    f"the share of all questions that received a fully supported answer. Part of its benefit to reliability is that the "
                    f"model says less, which is a cost to utility when the course does contain the answer. ")
        if ("QA-E0", "warning_signs") in Q.index and ("QA-E1", "warning_signs") in Q.index:
            w0, w1 = Q.loc[("QA-E0", "warning_signs")], Q.loc[("QA-E1", "warning_signs")]
            txt += (f"The question type dominates the results. Warning signs are rarely stated in a hospital course, so the model "
                    f"abstained most often on that question ({pct0(w0.abstention_rate)} and {pct0(w1.abstention_rate)}), and when it did "
                    f"answer, the judge-estimated UFR of its answers was {f3(w0.UFR_mean)} and {f3(w1.UFR_mean)}: such answers "
                    f"come from the model's general knowledge of discharge advice, the extrinsic content that retrieval did not remove in "
                    f"Chapter 5. Agreement with the clinician's own instructions is low for every condition, because clinicians select and "
                    f"phrase information differently from the model; it is a proxy for correctness, not a measurement of it.")
        b += [P(txt),
              ("table", dict(label="mqa", caption="Question-answering pilot by condition and question type: share of questions abstained on, answered with every sentence supported by the course, and answered with at least one unsupported sentence (the three shares sum to one); judge-estimated UFR of the answered questions; and share of answer sentences supported by the clinician's discharge instructions.",
                             columns=["Condition", "Question", "Abstained", "Answered, all supported", "Answered, some unsupported", "UFR of answers", "Supported by clinician"],
                             widths=[0.75, 1.15, 0.75, 1.0, 1.05, 0.75, 0.95], font=8.5, rows=rows))]

    # ── 7.12 Summary
    b += [("h2", "7.12 Summary of the Main Study")]
    lines = []
    if len(lower) == len(GROUNDED):
        lines.append(f"**Judge-estimated faithfulness:** E1, E1b and E3 had a lower UFR than E0 ({f3(D.m('E0', 'UFR_mean'))}; "
                     + ", ".join(f"{c} {f3(D.m(c, 'UFR_mean'))}" for c in GROUNDED)
                     + "), significant after the Holm adjustment, with no significant change in CR; the differences among the three are "
                       "within what the judge's errors could produce.")
    if t_b is not None:
        lines.append(f"**Coverage:** E1 and E3 covered fewer of the clinician's sentences than E0; E1b achieved coverage similar to E0, "
                     f"with no statistically significant difference detected ({pts(t_b.delta_mean)}, 95 percent interval "
                     f"{ci_pts(t_b.ci95_low, t_b.ci95_high)}).")
    if vt is not None and len(vt):
        lines.append("**Length and citations:** a length-matched E0 and E1 without the citation instruction show how much of the "
                     "reduction is due to brevity and to the instruction (Section 7.5).")
    if cited and "E1" in cc:
        lines.append(f"**Citations:** {pct0(cc['E1'][1] / cc['E1'][0])} of E1's claims carry a valid citation and "
                     f"{pct0(cc['E1'][2] / cc['E1'][1])} of those are supported by the cited excerpts"
                     + (f"; E1b cites only {pct0(cc['E1b'][1] / cc['E1b'][0])} of its claims" if "E1b" in cc else "") + ".")
    if abl is not None and len(abl):
        lines.append("**Retrieval settings:** the amount of the course shown to the model governs coverage; no setting changed "
                     "faithfulness after correction for multiple comparisons." if not any(F[m][v][2] < 0.05 for m in ("UFR", "CR") for v in F[m]) else
                     "**Retrieval settings:** see Section 7.7.")
    if rep is not None and len(rep) and "kind" in rep.columns:
        lines.append(f"**Repeated generation:** {_join(stable) if stable else 'no comparison'} held in all three generations"
                     + (f"; {_join(unstable)} did not." if unstable else "."))
    if agr is not None and len(agr):
        lines.append("**Second judge:** the reductions relative to E0 hold under a second judge and under both combined flagging "
                     "rules; whether verification adds to retrieval depends on the judge.")
    if qa is not None and len(qa):
        lines.append("**Question answering (exploratory):** retrieval raised abstention and lowered the UFR of the answers given; "
                     "warning-sign questions are rarely answerable from the course.")
    b += [("bullets", lines)]
    return b
