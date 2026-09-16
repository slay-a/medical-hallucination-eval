#!/usr/bin/env python3
"""
audit_summary.py — Judge precision, recall and kappa against the manual labels in results_private/audit_sample.csv.

Only aggregate numbers are written (results/audit_summary.csv); the labeled sample itself stays private.
A judge flag counts as correct when the human label is U; an accepted claim counts as correct when the label is S.
Claims labeled ? are excluded.  Because flagged and accepted claims were sampled at fixed sizes, the corrected
unsupported rate per condition is estimated as flag_rate * precision + (1 - flag_rate) * (1 - negative predictive value),
using each condition's flag rate from results_private/mimic_claims.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
PRIVATE, RESULTS = HERE / "results_private", HERE / "results"


def main():
    a = pd.read_csv(PRIVATE / "audit_sample.csv")
    a = a[a.human_label.astype(str).str.upper().isin(["S", "U"])].copy()
    if a.empty:
        print("no labeled rows yet (fill human_label with S or U)"); sys.exit(0)
    a["human_unsupported"] = a.human_label.str.upper() == "U"
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[cl.variant == "base"]
    rows = []
    for cond, g in list(a.groupby("condition")) + [("all", a)]:
        tp = int((g.judge_flagged & g.human_unsupported).sum()); fp = int((g.judge_flagged & ~g.human_unsupported).sum())
        fn = int((~g.judge_flagged & g.human_unsupported).sum()); tn = int((~g.judge_flagged & ~g.human_unsupported).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
        flag_rate = (cl.label != "Supported").mean() if cond == "all" else (cl[cl.condition == cond].label != "Supported").mean()
        corrected = flag_rate * prec + (1 - flag_rate) * (1 - npv) if not (np.isnan(prec) or np.isnan(npv)) else np.nan
        rows.append(dict(condition=cond, n_labeled=len(g), n_flagged=int(g.judge_flagged.sum()), precision=prec, npv=npv,
                         recall_within_sample=tp / (tp + fn) if tp + fn else np.nan,
                         kappa=cohen_kappa_score(g.human_unsupported, g.judge_flagged) if g.human_unsupported.nunique() > 1 and g.judge_flagged.nunique() > 1 else np.nan,
                         judge_flag_rate=flag_rate, corrected_unsupported_rate=corrected, human_unsupported_share_of_flagged=prec,
                         n_unsure_excluded=int(0)))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "audit_summary.csv", index=False)
    pd.set_option("display.width", 200); print(out.round(3).to_string(index=False)); print("wrote results/audit_summary.csv")


if __name__ == "__main__":
    main()
