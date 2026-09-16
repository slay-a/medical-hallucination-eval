#!/usr/bin/env python3
"""
audit_sample.py — Draw a stratified sample of main-study claims for a manual precision audit of the judge.

The sample is written to results_private/audit_sample.csv (git-ignored: it contains MIMIC-derived text).  Fill the
column `human_label` with S (supported by the hospital course), U (unsupported: not stated or contradicted) or ? (unsure),
then run `python audit_summary.py` to compute the judge's precision, recall and kappa against your labels.

Stratification: per condition (E0, E1, E1b, E3 by default) a fixed number of claims the judge flagged (Not-Supported or
Contradicted) and of claims it accepted, drawn at random with a fixed seed; the judge's label and score are kept in
hidden columns (judge_label, p_support) that you should not look at while labeling — sort by the `order` column.

Usage: python audit_sample.py [--per-cell 40] [--conditions E0,E1,E1b,E3] [--seed 7]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_annptsumm import DEFAULT_DATA, read_jsonl  # noqa: E402

PRIVATE = HERE / "results_private"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=40); ap.add_argument("--conditions", default="E0,E1,E1b,E3"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--claims", default=str(PRIVATE / "mimic_claims.csv")); ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cl = pd.read_csv(args.claims); cl = cl[cl.variant == "base"]
    docs = {}
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di.jsonl")):
        docs[f"doc_{i}"] = r["text"]
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di_validation.jsonl")):
        docs[f"val_{i}"] = r["text"]
    rows = []
    for cond in args.conditions.split(","):
        d = cl[cl.condition == cond]
        for flagged, sub in ((True, d[d.label != "Supported"]), (False, d[d.label == "Supported"])):
            take = sub.sample(n=min(args.per_cell, len(sub)), random_state=int(rng.integers(1e9)))
            for _, r in take.iterrows():
                rows.append(dict(sid=r.sid, condition=cond, claim=r.claim, hospital_course=docs[r.sid], human_label="", note="",
                                 judge_label=r.label, p_support=r.p_support, judge_flagged=flagged))
    out = pd.DataFrame(rows); out["order"] = rng.permutation(len(out)); out = out.sort_values("order")
    cols = ["order", "sid", "condition", "claim", "hospital_course", "human_label", "note", "judge_label", "p_support", "judge_flagged"]
    out[cols].to_csv(PRIVATE / "audit_sample.csv", index=False)
    print(f"wrote {len(out)} claims to results_private/audit_sample.csv "
          f"({int(out.judge_flagged.sum())} flagged by the judge, {int((~out.judge_flagged).sum())} accepted); label the human_label column S/U/?")


if __name__ == "__main__":
    main()
