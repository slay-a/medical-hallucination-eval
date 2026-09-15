#!/usr/bin/env python3
"""
recompute_metrics.py — Rebuild every per-sample and aggregate metric from
results/claims_all.csv with markdown section headers excluded from the claim set.

Runs offline (no API or model calls) and is idempotent.  Run it after any
pipeline script (hallucination_eval.py, e2_extractive_eval.py, e1b_*.py, e3_*.py).

Inputs : results/claims_all.csv, results/summaries.csv
Outputs: results/claims_all.csv             (adds/refreshes boolean column is_header)
         results/summaries.csv              (metric columns refreshed for every condition present)
         results/comparison_per_sample.csv  (all conditions side by side + paired deltas)
         results/aggregate_statistics.csv   (mean/SD/median per condition, deltas, % improved)
         results/pairwise_tests.csv         (Wilcoxon signed-rank, bootstrap 95% CI, effect sizes)
         results/header_filter_effect.csv   (metrics before vs after the header filter)
         results/header_filter_tests.csv    (E1 vs E0 tests before vs after the header filter)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import is_markdown_header, is_abstention  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
CONDITION_ORDER = ["E0", "E1", "E1b", "E2", "E3"]
METRICS = ["UFR", "CR"]
N_BOOT = 10_000
SEED = 0


def order_conditions(conds):
    conds = set(conds)
    return [c for c in CONDITION_ORDER if c in conds] + sorted(c for c in conds if c not in CONDITION_ORDER)


def per_sample_metrics(claims: pd.DataFrame, doc_ids, conds) -> pd.DataFrame:
    """Per (doc_id, condition) claim counts, UFR and CR. Missing pairs get n_claims = 0."""
    counts = {k: v.value_counts() for k, v in claims.groupby(["doc_id", "condition"])["label"]}
    rows = []
    for d in doc_ids:
        for c in conds:
            vc = counts.get((d, c))
            if vc is None or int(vc.sum()) == 0:
                rows.append(dict(doc_id=d, condition=c, n_claims=0, n_supported=0,
                                 n_contradicted=0, n_not_supported=0, UFR=np.nan, CR=np.nan))
                continue
            n = int(vc.sum())
            ns, nc, nn = int(vc.get("Supported", 0)), int(vc.get("Contradicted", 0)), int(vc.get("Not-Supported", 0))
            rows.append(dict(doc_id=d, condition=c, n_claims=n, n_supported=ns, n_contradicted=nc,
                             n_not_supported=nn, UFR=(nc + nn) / n, CR=nc / n))
    return pd.DataFrame(rows)


def paired_stats(a: pd.Series, b: pd.Series, rng=None) -> dict:
    """Paired comparison of condition B against condition A (negative delta = B lower)."""
    rng = rng or np.random.default_rng(SEED)
    m = a.notna() & b.notna()
    a, b = a[m].to_numpy(float), b[m].to_numpy(float)
    d = b - a
    n = len(d)
    if n >= 5 and not np.all(d == 0):
        W, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    else:
        W, p = np.nan, np.nan
    boots = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1) if n else np.array([np.nan])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = d.std(ddof=1) if n > 1 else np.nan
    dz = d.mean() / sd if sd and sd > 0 else np.nan
    nz = d[d != 0]
    if len(nz):
        ranks = pd.Series(np.abs(nz)).rank().to_numpy()
        rb = (ranks[nz < 0].sum() - ranks[nz > 0].sum()) / ranks.sum()   # >0 means B improved on A
    else:
        rb = np.nan
    return dict(n=n, mean_a=a.mean(), mean_b=b.mean(), median_a=float(np.median(a)), median_b=float(np.median(b)),
                delta_mean=d.mean(), delta_median=float(np.median(d)), ci95_low=lo, ci95_high=hi,
                W=W, p=p, d_z=dz, rank_biserial=rb,
                pct_improved=100 * (d < 0).mean(), pct_unchanged=100 * (d == 0).mean(), pct_worse=100 * (d > 0).mean(),
                rel_change_mean=(b.mean() - a.mean()) / a.mean() * 100 if a.mean() else np.nan)


def pairwise_table(ps: pd.DataFrame, conds) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    wide = {m: ps.pivot(index="doc_id", columns="condition", values=m) for m in METRICS}
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            for m in METRICS:
                st = paired_stats(wide[m][a], wide[m][b], rng)
                rows.append(dict(metric=m, condition_a=a, condition_b=b, comparison=f"{b} vs {a}", **st))
    return pd.DataFrame(rows)


def main() -> None:
    claims = pd.read_csv(RESULTS / "claims_all.csv")
    claims["is_header"] = claims["claim"].astype(str).map(is_markdown_header)
    claims["is_abstention"] = claims["claim"].astype(str).map(is_abstention)
    claims.to_csv(RESULTS / "claims_all.csv", index=False)

    conds = order_conditions(claims["condition"].unique())
    doc_ids = sorted(claims["doc_id"].unique())
    before = per_sample_metrics(claims, doc_ids, conds)
    after = per_sample_metrics(claims[~claims["is_header"] & ~claims["is_abstention"]], doc_ids, conds)

    # ── summaries.csv: refresh metric columns ─────────────────────────────────
    summ = pd.read_csv(RESULTS / "summaries.csv")
    for c in conds:
        sub = after[after.condition == c].set_index("doc_id")
        for col in ["UFR", "CR", "n_claims", "n_supported", "n_contradicted", "n_not_supported"]:
            summ[f"{c}_{col}"] = summ["doc_id"].map(sub[col])
    summ.to_csv(RESULTS / "summaries.csv", index=False)

    # ── comparison_per_sample.csv ─────────────────────────────────────────────
    base_cols = [c for c in ["doc_id", "specialty", "description", "source_word_count"] if c in summ.columns]
    cmp = summ[base_cols].copy()
    for c in conds:
        wcol = f"{c.lower()}_summary_words"
        if wcol in summ.columns:
            cmp[wcol] = summ[wcol]
        for col in ["UFR", "CR", "n_claims", "n_supported", "n_contradicted", "n_not_supported"]:
            cmp[f"{c}_{col}"] = summ[f"{c}_{col}"]
    if {"E0", "E1"} <= set(conds):
        cmp["delta_UFR"] = cmp["E1_UFR"] - cmp["E0_UFR"]
        cmp["delta_CR"] = cmp["E1_CR"] - cmp["E0_CR"]
    if {"E0", "E2"} <= set(conds):
        cmp["E2_delta_UFR"] = cmp["E2_UFR"] - cmp["E0_UFR"]
        cmp["E2_delta_CR"] = cmp["E2_CR"] - cmp["E0_CR"]
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            for m in METRICS:
                cmp[f"d_{b}_vs_{a}_{m}"] = cmp[f"{b}_{m}"] - cmp[f"{a}_{m}"]
    cmp.to_csv(RESULTS / "comparison_per_sample.csv", index=False)

    # ── pairwise tests ────────────────────────────────────────────────────────
    tests = pairwise_table(after, conds)
    tests.to_csv(RESULTS / "pairwise_tests.csv", index=False)

    # ── aggregate_statistics.csv ──────────────────────────────────────────────
    rows = []
    for m in METRICS:
        row = {"Metric": m}
        for c in conds:
            v = after.loc[after.condition == c, m].dropna()
            row[f"{c}_Mean"], row[f"{c}_Std"], row[f"{c}_Median"] = round(v.mean(), 4), round(v.std(), 4), round(v.median(), 4)
        def _t(a, b):
            t = tests[(tests.metric == m) & (tests.condition_a == a) & (tests.condition_b == b)]
            return t.iloc[0] if len(t) else None
        t10 = _t("E0", "E1"); t20 = _t("E0", "E2"); t21 = _t("E1", "E2")
        if t10 is not None:
            row.update(Delta_Mean=round(t10.delta_mean, 4), Delta_Std=round((cmp["E1_" + m] - cmp["E0_" + m]).std(), 4),
                       Pct_Improved=round(t10.pct_improved, 1), p_E1_vs_E0=t10.p,
                       CI95_E1_vs_E0=f"[{t10.ci95_low:+.4f}, {t10.ci95_high:+.4f}]")
        if t20 is not None:
            row.update(E2_Delta_Mean=round(t20.delta_mean, 4), E2_Delta_Std=round((cmp["E2_" + m] - cmp["E0_" + m]).std(), 4),
                       E2_Pct_Improved=round(t20.pct_improved, 1), p_E2_vs_E0=t20.p,
                       CI95_E2_vs_E0=f"[{t20.ci95_low:+.4f}, {t20.ci95_high:+.4f}]")
        if t21 is not None:
            row.update(p_E2_vs_E1=t21.p)
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "aggregate_statistics.csv", index=False)

    # ── header filter effect ──────────────────────────────────────────────────
    hdr = claims[claims["is_header"]]
    eff_rows = []
    for c in conds:
        h = hdr[hdr.condition == c]
        for m in METRICS:
            eff_rows.append(dict(
                condition=c, metric=m,
                mean_before=before.loc[before.condition == c, m].mean(), mean_after=after.loc[after.condition == c, m].mean(),
                median_before=before.loc[before.condition == c, m].median(), median_after=after.loc[after.condition == c, m].median(),
                n_claims_before=int(before.loc[before.condition == c, "n_claims"].sum()),
                n_claims_after=int(after.loc[after.condition == c, "n_claims"].sum()),
                n_headers=len(h), headers_contradicted=int((h.label == "Contradicted").sum()),
                headers_not_supported=int((h.label == "Not-Supported").sum()), headers_supported=int((h.label == "Supported").sum())))
    pd.DataFrame(eff_rows).to_csv(RESULTS / "header_filter_effect.csv", index=False)

    if {"E0", "E1"} <= set(conds):
        rng = np.random.default_rng(SEED)
        trows = []
        for label, ps in (("before_filter", before), ("after_filter", after)):
            w = {m: ps.pivot(index="doc_id", columns="condition", values=m) for m in METRICS}
            for m in METRICS:
                st = paired_stats(w[m]["E0"], w[m]["E1"], rng)
                trows.append(dict(stage=label, metric=m, comparison="E1 vs E0", **st))
        pd.DataFrame(trows).to_csv(RESULTS / "header_filter_tests.csv", index=False)

    # ── console report ────────────────────────────────────────────────────────
    print("Conditions:", conds, "| documents:", len(doc_ids))
    print(f"Header lines removed: {int(claims.is_header.sum())} of {len(claims)} claim rows "
          f"({claims[claims.is_header].groupby('condition').size().to_dict()})")
    print(f"Abstention lines excluded: {int(claims.is_abstention.sum())} ({claims[claims.is_abstention].groupby('condition').size().to_dict()})")
    print("\nPer-sample means AFTER header filter:")
    print(after.groupby("condition")[["UFR", "CR", "n_claims"]].mean().round(4))
    print("\nPairwise tests (Wilcoxon signed-rank, two-sided):")
    print(tests[["comparison", "metric", "mean_a", "mean_b", "delta_mean", "ci95_low", "ci95_high", "p",
                 "d_z", "pct_improved"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
