#!/usr/bin/env python3
"""
audit_summary.py — Compare the selected judge with the manual labels of the audit (results_private/audit_labels.json).

Claims: a manual label of P (partly supported) or U (not supported) counts as unsupported, S as supported; "?" is excluded.
Flagged and accepted claims were sampled in fixed numbers per condition (the sampling strata, stored with each item), so
every labeled claim is weighted by W_h / n_h, where W_h is the stratum's share of the condition's claims over all 110 courses
(design.strata_population of audit_items.json) and n_h the number of labeled claims in the stratum.  From the weighted 2x2
table of manual label against the judge's current label (judge_flagged_v2 after `audit_sample.py --refresh-judge`, otherwise
the label at sampling time):
    manual-estimated unsupported share  pi = sum_h W_h * (unsupported share in stratum h)
    precision = share of judge-flagged claims that are unsupported;  false omission = share of accepted claims that are
    unsupported;  sensitivity, specificity and kappa from the same table.
When the judge's labels are those used for sampling this reduces to pi = f * precision + (1 - f) * FOR.
95% intervals: percentile bootstrap over hospital courses (2,000 resamples).
Citations: manual verdict on whether the cited excerpt supports the claim (Yes; Partly; No), compared with the judge's verdict
from results_private/mimic_citation_claims.csv when that file exists.
Writes only aggregates: results/audit_summary.csv, results/audit_types.csv, results/audit_citations.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIVATE, RESULTS = HERE / "results_private", HERE / "results"
CONDS = ["E0", "E1", "E1b", "E3"]


def kappa2(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def estimates(d, W):
    """Stratified estimates for one condition; d has columns stratum, unsup and flag, W maps stratum -> population share."""
    n = d.groupby("stratum").size()
    if set(n.index) != set(W):
        return None
    w = d.stratum.map(lambda h: W[h] / n[h]).to_numpy()
    u, f = d.unsup.to_numpy(bool), d.flag.to_numpy(bool)
    tp, fp, fn, tn = (w * (u & f)).sum(), (w * (~u & f)).sum(), (w * (u & ~f)).sum(), (w * (~u & ~f)).sum()
    pi = tp + fn
    return dict(manual_unsupported=pi, judge_flag_share=tp + fp, precision=tp / (tp + fp) if tp + fp else np.nan,
                false_omission=fn / (fn + tn) if fn + tn else np.nan, sensitivity=tp / pi if pi else np.nan,
                specificity=tn / (1 - pi) if pi < 1 else np.nan, kappa=kappa2(tp, fp, fn, tn))


def main():
    items = json.load(open(PRIVATE / "audit_items.json"))
    labels = json.load(open(PRIVATE / "audit_labels.json")) if (PRIVATE / "audit_labels.json").exists() else {}
    rows = [dict(it, label=(labels.get(it["item_id"]) or {}).get("label"), type=(labels.get(it["item_id"]) or {}).get("type"))
            for c in items["courses"] for it in c["items"]]
    df = pd.DataFrame(rows)
    claims = df[(df.kind == "claim") & df.label.isin(["S", "P", "U"])].copy()
    n_items = int((df.kind == "claim").sum()); n_lab = int(df[df.kind == "claim"].label.notna().sum())
    print(f"claims labeled: {n_lab} of {n_items} ({int((df[df.kind == 'claim'].label == 'Q').sum())} unsure)")
    if claims.empty:
        print("no claim labels yet"); return
    claims["unsup"] = claims.label.isin(["P", "U"])
    claims["flag"] = claims.judge_flagged_v2 if "judge_flagged_v2" in claims and claims.judge_flagged_v2.notna().all() else claims.judge_flagged
    claims["flag"] = claims.flag.astype(bool)
    pop = items["design"]["strata_population"]
    W = {c: {h: pop[c][h] / (pop[c]["flagged"] + pop[c]["accepted"]) for h in ("flagged", "accepted")} for c in CONDS}
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[cl.variant == "base"]
    f_now = {c: float((cl[cl.condition == c].label != "Supported").mean()) for c in CONDS}   # judge flag share, current pipeline
    rng = np.random.default_rng(7); courses = claims.sid.unique()
    by_course = {s: g for s, g in claims.groupby("sid")}
    boot_sets = [pd.concat([by_course[s] for s in rng.choice(courses, size=len(courses), replace=True)]) for _ in range(2000)]

    def pooled(d):          # average of the per-condition tables
        parts = [estimates(d[d.condition == x], W[x]) for x in CONDS]
        parts = [p_ for p_ in parts if p_ is not None]
        return {k: float(np.nanmean([p_[k] for p_ in parts])) for k in parts[0]} if len(parts) == len(CONDS) else None

    out = []
    for c in CONDS + ["pooled"]:
        d = claims if c == "pooled" else claims[claims.condition == c]
        if d.empty:
            continue
        est = pooled(d) if c == "pooled" else estimates(d, W[c])
        if est is None:
            continue
        B = pd.DataFrame([e for e in ((pooled(bd) if c == "pooled" else estimates(bd[bd.condition == c], W[c])) for bd in boot_sets) if e is not None])
        row = dict(condition=c, n_labeled=len(d), n_flagged_stratum=int((d.stratum == "flagged").sum()),
                   n_label_changed=int((d.flag != (d.stratum == "flagged")).sum()),
                   judge_flag_share_pipeline=float(np.mean(list(f_now.values()))) if c == "pooled" else f_now[c], **est)
        for k in ("manual_unsupported", "judge_flag_share", "precision", "false_omission", "sensitivity", "specificity", "kappa"):
            row[f"{k}_lo"], row[f"{k}_hi"] = (float(B[k].quantile(0.025)), float(B[k].quantile(0.975))) if len(B) else (np.nan, np.nan)
        row["partly_share_of_unsupported"] = float((d.label == "P").sum() / max(1, d.unsup.sum()))
        out.append(row)
    res = pd.DataFrame(out)
    # differences between conditions in the manual-estimated share and in the judge's share (bootstrap over courses)
    diffs = []
    for a, b in (("E0", "E1"), ("E0", "E1b"), ("E0", "E3"), ("E1", "E3"), ("E1", "E1b")):
        ea, eb = estimates(claims[claims.condition == a], W[a]), estimates(claims[claims.condition == b], W[b])
        if ea is None or eb is None:
            continue
        bs = []
        for bd in boot_sets:
            xa, xb = estimates(bd[bd.condition == a], W[a]), estimates(bd[bd.condition == b], W[b])
            if xa is not None and xb is not None:
                bs.append((xb["manual_unsupported"] - xa["manual_unsupported"], xb["judge_flag_share"] - xa["judge_flag_share"]))
        bs = np.array(bs)
        diffs.append(dict(condition=f"{b} minus {a}", diff=eb["manual_unsupported"] - ea["manual_unsupported"],
                          diff_lo=float(np.quantile(bs[:, 0], 0.025)), diff_hi=float(np.quantile(bs[:, 0], 0.975)),
                          judge_diff=eb["judge_flag_share"] - ea["judge_flag_share"],
                          judge_diff_lo=float(np.quantile(bs[:, 1], 0.025)), judge_diff_hi=float(np.quantile(bs[:, 1], 0.975))))
    res = pd.concat([res, pd.DataFrame(diffs)], ignore_index=True)
    res.to_csv(RESULTS / "audit_summary.csv", index=False)
    types = claims[claims.unsup].groupby(["condition", "type"], dropna=False).size().rename("n").reset_index()
    types.to_csv(RESULTS / "audit_types.csv", index=False)
    cit = df[(df.kind == "citation") & df.label.isin(["Yes", "Partly", "No"])].copy()
    if len(cit):
        cit["cited_key"] = cit.cited.apply(lambda x: ",".join(map(str, x)))
        jf = PRIVATE / "mimic_citation_claims.csv"
        if jf.exists():
            j = pd.read_csv(jf); j["cited_key"] = j.cited_key.astype(str)
            cit = cit.merge(j[["sid", "condition", "claim", "cited_key", "judge_supported"]], on=["sid", "condition", "claim", "cited_key"], how="left")
        crow = []
        for c in ("E1", "E1b"):
            d = cit[cit.condition == c]
            if d.empty:
                continue
            r = dict(condition=c, n=len(d), manual_yes=float((d.label == "Yes").mean()), manual_yes_or_partly=float(d.label.isin(["Yes", "Partly"]).mean()))
            if "judge_supported" in d and d.judge_supported.notna().any():
                dj = d[d.judge_supported.notna()]
                r.update(judge_supported=float(dj.judge_supported.mean()), agreement=float(((dj.label == "Yes") == dj.judge_supported.astype(bool)).mean()))
            crow.append(r)
        pd.DataFrame(crow).to_csv(RESULTS / "audit_citations.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print("wrote results/audit_summary.csv, results/audit_types.csv" + (", results/audit_citations.csv" if len(cit) else ""))


if __name__ == "__main__":
    main()
