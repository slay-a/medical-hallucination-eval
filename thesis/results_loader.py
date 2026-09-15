"""results_loader.py — Load every result file into one object with formatting helpers used by the thesis text."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
COND_NAME = {"E0": "E0 (zero-context LLM)", "E1": "E1 (RAG, excerpts only)", "E1b": "E1b (RAG, note + excerpts)",
             "E2": "E2 (extractive)", "E3": "E3 (RAG + CoVe)"}


def _csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def f3(x):           # 0.1721 -> "0.172"
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def pct(x, d=1):     # 0.1721 -> "17.2%"
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def pct0(x):
    return pct(x, 0)


def fp(p):           # p-value text
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fpn(p):          # bare number for tables
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def signed(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{d}f}"


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx]); running = max(running, val); adj[idx] = running
    return adj


class Results:
    def __init__(self):
        self.agg = _csv("aggregate_statistics.csv")
        self.tests = _csv("pairwise_tests.csv")
        self.cmp = _csv("comparison_per_sample.csv")
        self.summ = _csv("summaries.csv")
        self.claims = _csv("claims_all.csv")
        self.hdr_eff = _csv("header_filter_effect.csv")
        self.hdr_tests = _csv("header_filter_tests.csv")
        self.abl_tau = _csv("ablation_thresholds.csv")
        self.abl_topk = _csv("ablation_topk.csv")
        self.abl_clean = _csv("ablation_cleaned_evidence.csv")
        self.claims_clean = _csv("claims_cleaned_evidence.csv")
        self.cov = _csv("coverage_summary.csv")
        self.cov_ps = _csv("coverage_per_sample.csv")
        self.e2err = _csv("e2_error_analysis.csv")
        self.tax_counts = _csv("error_taxonomy_counts.csv")
        self.tax_ex = _csv("error_taxonomy_examples.csv")
        self.cal = _csv("calibration_overall.csv")
        self.cal_lab = _csv("calibration_by_label.csv")
        self.cal_labcounts = _csv("calibration_label_counts.csv")
        self.cal_sweep = _csv("calibration_threshold_sweep.csv")
        self.cal_sum = _csv("calibration_summary_level.csv")
        self.cal_sys = _csv("calibration_by_system_means.csv")
        self.cal_var = _csv("calibration_variants.csv")
        self.ex_fixed = _csv("examples_fixed_by_rag.csv")
        self.ex_persist = _csv("examples_persistent.csv")
        self.conds = [c for c in ["E0", "E1", "E1b", "E2", "E3"] if f"{c}_UFR" in self.cmp.columns]
        self.n_docs = len(self.cmp)
        self._dataset_facts()
        self._holm()

    # ── dataset facts ─────────────────────────────────────────────────────
    def _dataset_facts(self):
        self.total_rows, self.n_eligible, self.n_consult_eligible, self.n_discharge_eligible, self.n_specialties = 4999, 624, 516, 108, 40
        mt = ROOT.parent / "mtsamples.csv"
        if mt.exists():
            df = pd.read_csv(mt); df.columns = [c.strip() for c in df.columns]
            self.total_rows = len(df); self.n_specialties = int(df["medical_specialty"].nunique())
            m = df["medical_specialty"].str.strip().isin(["Discharge Summary", "Consult - History and Phy."])
            f = df[m].dropna(subset=["transcription"]); f = f[f["transcription"].str.strip().ne("")]
            self.n_eligible = len(f); vc = f["medical_specialty"].str.strip().value_counts()
            self.n_consult_eligible, self.n_discharge_eligible = int(vc.get("Consult - History and Phy.", 0)), int(vc.get("Discharge Summary", 0))
        vc = self.cmp["specialty"].value_counts()
        self.n_consult, self.n_discharge = int(vc.get("Consult - History and Phy.", 0)), int(vc.get("Discharge Summary", 0))
        w = self.cmp["source_word_count"]
        self.src_words = dict(mean=w.mean(), sd=w.std(), min=w.min(), median=w.median(), max=w.max())
        self.words = {c: self.cmp[f"{c.lower()}_summary_words"] for c in self.conds if f"{c.lower()}_summary_words" in self.cmp.columns}
        nh = self.claims[~self.claims.is_header]
        self.label_counts = pd.crosstab(nh.condition, nh.label)
        self.claims_per = nh.groupby("condition").size() / self.n_docs
        self.n_claims_total = int(len(nh)); self.n_rows_total = int(len(self.claims)); self.n_headers = int(self.claims.is_header.sum())
        self.headers_by_cond = self.claims[self.claims.is_header].groupby("condition").size().to_dict()

    def _holm(self):
        t = self.tests.copy()
        t["p_holm"] = holm(t["p"].fillna(1.0).values)
        self.tests = t

    # ── accessors ─────────────────────────────────────────────────────────
    def test(self, metric, a, b):
        t = self.tests[(self.tests.metric == metric) & (self.tests.condition_a == a) & (self.tests.condition_b == b)]
        return t.iloc[0] if len(t) else None

    def mean(self, cond, metric):
        return float(self.cmp[f"{cond}_{metric}"].mean())

    def median(self, cond, metric):
        return float(self.cmp[f"{cond}_{metric}"].median())

    def sd(self, cond, metric):
        return float(self.cmp[f"{cond}_{metric}"].std())

    def hdr(self, cond, metric, col):
        h = self.hdr_eff[(self.hdr_eff.condition == cond) & (self.hdr_eff.metric == metric)]
        return h.iloc[0][col] if len(h) else np.nan

    def hdr_test(self, stage, metric):
        h = self.hdr_tests[(self.hdr_tests.stage == stage) & (self.hdr_tests.metric == metric)]
        return h.iloc[0] if len(h) else None

    def cal_row(self, group):
        c = self.cal[self.cal.group == group]
        return c.iloc[0] if len(c) else None

    def e2err_value(self, label):
        e = self.e2err[(self.e2err.condition == "E2") & (self.e2err.label == label)]
        return e.iloc[0]["count"] if len(e) else np.nan

    def tax_total(self, cond, category):
        t = self.tax_counts[(self.tax_counts.condition == cond) & (self.tax_counts.category == category) & (self.tax_counts.label == "All unsupported")]
        return int(t.iloc[0]["count"]) if len(t) else 0

    def example_summary(self, doc_id, cond):
        row = self.summ[self.summ.doc_id == doc_id].iloc[0]
        return str(row.get(f"{cond.lower()}_summary", ""))

    def example_claims(self, doc_id, cond):
        c = self.claims[(self.claims.doc_id == doc_id) & (self.claims.condition == cond) & (~self.claims.is_header)]
        return c[["claim", "label", "p_entailment", "p_contradiction"]]


if __name__ == "__main__":
    R = Results()
    print("conditions", R.conds, "docs", R.n_docs, "eligible", R.n_eligible)
    print(R.label_counts); print(R.claims_per)
    print(R.tests[["comparison", "metric", "p", "p_holm"]])
