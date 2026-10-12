"""mimic_results.py — access layer for the main-study result files (results/mimic_*.csv) used by ch_mimic, ch8_9,
the abstract and the slides. Every accessor tolerates missing files so that the thesis still builds before the run."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
ORDER = ["E0", "E1", "E1b", "E3", "E2", "REF"]
NAME = {"E0": "E0 zero-context LLM", "E1": "E1 RAG (excerpts only, cited)", "E1b": "E1b RAG (course + excerpts, cited)", "E3": "E3 RAG + CoVe",
        "E2": "E2 extractive", "REF": "Doctor-written instructions"}
SHORT = {"E0": "E0", "E1": "E1", "E1b": "E1b", "E3": "E3", "E2": "E2", "REF": "clinician"}
VAR = {"base": "E1 base (5-sentence chunks, top-3, dense, cited)", "chunk3": "3-sentence chunks", "chunk8": "8-sentence chunks", "top2": "top-2 chunks",
       "top5": "top-5 chunks", "bm25": "BM25 retrieval", "nocite": "citations not required"}
VPHRASE = {"chunk3": "three-sentence chunks", "chunk8": "eight-sentence chunks", "top2": "two retrieved chunks", "top5": "five retrieved chunks",
           "bm25": "BM25 retrieval", "nocite": "dropping the citation requirement"}
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
JUDGE_NAMES = {"minilm_nli": "MiniLM cross-encoder", "deberta_large_nli": "DeBERTa-v3-large NLI", "ce_deberta_large_nli": "DeBERTa-v3-large cross-encoder NLI",
               "minicheck_deberta": "MiniCheck DeBERTa-v3-large", "minicheck_roberta": "MiniCheck RoBERTa-large", "mednli_deberta_large": "DeBERTa-v3-large NLI fine-tuned on MedNLI",
               "bespoke_minicheck_7b": "Bespoke-MiniCheck-7B"}


def _join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _csv(name):
    f = RES / name
    return pd.read_csv(f) if f.exists() else None


class Mimic:
    def __init__(self):
        self.summ, self.tests, self.abl, self.ref, self.qa = (_csv("mimic_summary.csv"), _csv("mimic_pairwise_tests.csv"), _csv("mimic_ablations.csv"),
                                                              _csv("mimic_reference.csv"), _csv("mimic_qa_summary.csv"))
        self.ok = self.summ is not None and self.tests is not None
        if self.ok:
            self.base = self.summ[self.summ.variant == "base"].set_index("condition")
            self.conds = [c for c in ORDER if c in self.base.index]
            self.llm_conds = [c for c in self.conds if c not in ("REF", "E2")]
            self.judge, self.mode, self.tau = self.summ.judge.iloc[0], self.summ["mode"].iloc[0], float(self.summ.tau.iloc[0])
            self.n_docs = int(self.base.loc["E0", "n_docs"]) if "E0" in self.base.index else 0
        else:
            self.base, self.conds, self.llm_conds, self.judge, self.mode, self.tau, self.n_docs = None, [], [], None, None, np.nan, 0
        self.qa_tot = self.qa[self.qa.question == "all"].set_index("condition") if self.qa is not None and len(self.qa) else None

    # ---- per-condition values
    def mean(self, cond, metric):
        return float(self.base.loc[cond, f"{metric}_mean"]) if self.ok and cond in self.base.index else np.nan

    def median(self, cond, metric):
        return float(self.base.loc[cond, f"{metric}_median"]) if self.ok and cond in self.base.index else np.nan

    def val(self, cond, col):
        return self.base.loc[cond, col] if self.ok and cond in self.base.index else np.nan

    # ---- paired tests: row condition `b` against reference condition `a` (negative delta = b lower)
    def test(self, metric, b, a="E0"):
        if not self.ok:
            return None
        t = self.tests[(self.tests.metric == metric) & (self.tests.comparison == f"{b} vs {a}")]
        return t.iloc[0] if len(t) else None

    def ablation(self, metric, variant):
        if self.abl is None:
            return None
        t = self.abl[(self.abl.metric == metric) & (self.abl.variant == variant)]
        return t.iloc[0] if len(t) else None

    # ---- wording helpers
    @staticmethod
    def sig(t):
        return t is not None and isinstance(t.p, float) and not np.isnan(t.p) and t.p < 0.05

    @staticmethod
    def verb(t, lower="reduced", higher="increased", none="did not change"):
        """Past-tense verb for 'condition b <verb> metric relative to a'."""
        if t is None or (isinstance(t.p, float) and np.isnan(t.p)):
            return none
        if t.p >= 0.05:
            return none
        return lower if t.delta_mean < 0 else higher

    @staticmethod
    def rel(t):
        return f"{abs(t.rel_change_mean):.0f} percent" if t is not None and not np.isnan(t.rel_change_mean) else "—"

    # ---- retrieval ablations: which variants changed a metric significantly
    def ablation_tests(self, metric):
        """[(variant, delta, p, p_holm)] for every variant with a test; Holm adjustment within the metric's family."""
        from results_loader import holm
        if self.abl is None:
            return []
        rows = []
        for v in [x for x in VAR if x != "base"]:
            r = self.abl[(self.abl.metric == metric) & (self.abl.variant == v)]
            if len(r) and not np.isnan(r.p.iloc[0]):
                rows.append((v, float(r.delta_mean.iloc[0]), float(r.p.iloc[0])))
        if not rows:
            return []
        adj = holm([pv for _, _, pv in rows])
        return [(v, d, pv, float(a)) for (v, d, pv), a in zip(rows, adj)]

    def sig_ablations(self, metric, adjusted=False):
        """Variants whose raw (or Holm-adjusted) p-value is below 0.05: [(variant, delta, p, p_holm)]."""
        return [t for t in self.ablation_tests(metric) if (t[3] if adjusted else t[2]) < 0.05]

    def faithfulness_phrase(self):
        """Sentence about which retrieval variants changed UFR or CR, before and after the Holm correction (Chapters 8 and 9)."""
        from results_loader import fp
        su, sc = self.sig_ablations("UFR"), self.sig_ablations("CR")
        au, ac = self.sig_ablations("UFR", True), self.sig_ablations("CR", True)
        raw = [f"{VPHRASE[v]} {'raised' if d > 0 else 'lowered'} UFR by {abs(d):.3f} ({fp(pv)}, adjusted {fp(ph).replace('p ', 'p ')})" for v, d, pv, ph in su]
        raw += [f"{VPHRASE[v]} {'raised' if d > 0 else 'lowered'} CR by {abs(d):.3f} ({fp(pv)}, adjusted {fp(ph)})" for v, d, pv, ph in sc]
        if not raw:
            return "no retrieval variant changed UFR or CR"
        if not au and not ac:
            return ("after a Holm correction within each measure no retrieval variant changed UFR or CR; before correction, "
                    + "; ".join(raw) + ", effects that are suggestive at most")
        keep = [VPHRASE[v] for v, *_ in au + ac]
        return f"{_join(keep)} changed faithfulness after a Holm correction within each measure ({'; '.join(raw)})"

    def faithfulness_short(self):
        """Compact version for the conclusion and slides."""
        from results_loader import fp
        su, sc = self.sig_ablations("UFR"), self.sig_ablations("CR")
        au, ac = self.sig_ablations("UFR", True), self.sig_ablations("CR", True)
        if not su and not sc:
            return "no retrieval setting changed faithfulness"
        if not au and not ac:
            big = max(su + sc, key=lambda t: abs(t[1]))
            which = "UFR" if big in su else "CR"
            return (f"no retrieval setting changed faithfulness after correction for multiple comparisons (the largest effect, "
                    f"{VPHRASE[big[0]]}, {'raised' if big[1] > 0 else 'lowered'} {which} by {abs(big[1]):.3f}, {fp(big[2])} before correction)")
        return f"only {_join([VPHRASE[v] for v, *_ in au + ac])} changed faithfulness after correction for multiple comparisons"

    def faithfulness_abstract(self):
        """Shortest version for the abstract."""
        au, ac = self.sig_ablations("UFR", True), self.sig_ablations("CR", True)
        if not au and not ac:
            return "no retrieval setting changed faithfulness after correction for multiple comparisons"
        return f"only {_join([VPHRASE[v] for v, *_ in au + ac])} changed faithfulness"

    def second_judge_test(self, metric_cond, ref="E0"):
        """Paired test of UFR under the second judge: row of results/mimic_second_judge_tests.csv or None."""
        t = _csv("mimic_second_judge_tests.csv")
        if t is None:
            return None
        r = t[t.comparison == f"{metric_cond} vs {ref}"]
        return r.iloc[0] if len(r) else None

    @staticmethod
    def finetune_summary():
        import json
        f = Path.home() / "Desktop" / "Thesis" / "models" / "mednli-deberta-v3-large" / "mednli_finetune_summary.json"
        return json.load(open(f)) if f.exists() else None

    def judge_name(self):
        return JUDGE_NAMES.get(self.judge, self.judge or "—")

    def judge_row(self):
        jc = _csv("judge_candidates.csv")
        if jc is None or not self.ok:
            return None
        r = jc[(jc.judge == self.judge) & (jc["mode"] == self.mode) & (jc.group == "all")]
        if not len(r):   # fall back to the best row of that judge (the selection rule) if the stored mode does not match
            r = jc[(jc.judge == self.judge) & (jc.group == "all")].sort_values("kappa", ascending=False)
        return r.iloc[0] if len(r) else None

    # ---- ranking of the LLM conditions by a metric (lowest first)
    def ranked(self, metric, conds=None, ascending=True):
        conds = conds or self.llm_conds
        return sorted(conds, key=lambda c: self.mean(c, metric), reverse=not ascending)
