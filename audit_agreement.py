#!/usr/bin/env python3
"""
audit_agreement.py — Intra-annotator agreement between the first audit pass (results_private/audit_labels.json)
and the second blind pass under the clarified rubric (results_private/audit_labels_v2.json), on the items labeled
in both.  Claims use the S/P/U scale (pairs with an unsure label excluded); citation checks use Yes/Partly/No.
The binary columns collapse the scales to supported vs not (S vs P+U; Yes vs Partly+No).  Writes only aggregates:
results/audit_agreement.csv, with the confusion matrices on stdout (category counts, no item content).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIVATE, RESULTS = HERE / "results_private", HERE / "results"


def kappa(tab):
    a = tab.to_numpy()
    n = a.sum()
    po = np.trace(a) / n
    pe = (a.sum(1) * a.sum(0)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    items = json.load(open(PRIVATE / "audit_items.json"))
    kind = {it["item_id"]: it["kind"] for c in items["courses"] for it in c["items"]}
    v1 = json.load(open(PRIVATE / "audit_labels.json"))
    v2 = json.load(open(PRIVATE / "audit_labels_v2.json"))
    df = pd.DataFrame([(kind[i], v1[i].get("label"), v2[i].get("label")) for i in v1 if i in v2 and i in kind],
                      columns=["kind", "v1", "v2"]).dropna()
    out = []
    for k, cats in (("claim", ["S", "P", "U"]), ("citation", ["Yes", "Partly", "No"])):
        d = df[(df.kind == k) & df.v1.isin(cats) & df.v2.isin(cats)]
        n_unsure = int((df.kind == k).sum()) - len(d)
        tab = pd.crosstab(d.v1, d.v2).reindex(index=cats, columns=cats, fill_value=0)
        print(f"\n{k}s, pass 1 (rows) vs pass 2 (columns), n={len(d)}"
              + (f" ({n_unsure} pairs with an unsure label excluded)" if n_unsure else ""))
        print(tab.to_string())
        bin1, bin2 = d.v1 == cats[0], d.v2 == cats[0]
        btab = pd.crosstab(bin1, bin2).reindex(index=[True, False], columns=[True, False], fill_value=0)
        out.append(dict(kind=k, n=len(d), excluded_unsure=n_unsure,
                        agreement=round(float((d.v1 == d.v2).mean()), 4), kappa=round(float(kappa(tab)), 4),
                        agreement_binary=round(float((bin1 == bin2).mean()), 4),
                        kappa_binary=round(float(kappa(btab)), 4),
                        v1_counts="/".join(str(int((d.v1 == c).sum())) for c in cats),
                        v2_counts="/".join(str(int((d.v2 == c).sum())) for c in cats)))
    res = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "audit_agreement.csv", index=False)
    print()
    print(res.to_string(index=False))
    print("wrote results/audit_agreement.csv")


if __name__ == "__main__":
    main()
