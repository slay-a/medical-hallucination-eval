#!/usr/bin/env python3
"""
mimic_revision_analysis.py — Analyses added in response to the advisor's review (October 2026).  Aggregates only.

  * results/mimic_variant_tests.csv   length-matched E0 ("short") against E0 and E1; E1 without citations against E0;
                                      E1b without citations against E1b (paired Wilcoxon tests, bootstrap 95% intervals)
  * results/mimic_replicates.csv      run-to-run variation on 40 courses with three generations per condition
  * results/mimic_abstention.csv      relation between abstentions, judge-estimated UFR and coverage (Spearman)
  * results/mimic_equivalence.csv     paired coverage differences with 90% and 95% intervals and a post hoc two one-sided test
  * results/mimic_qa_utility.csv      question answering: answered, abstained, and fully supported answers over all questions
  * results/mimic_window_check.csv    (--windows) share of flagged claims that become supported with overlapping windows
  * results/mimic_splitting_check.csv how the first claim splitter treated section headers and citation markers, per condition
  * results/mimic_condition_ci.csv    per-condition means of every measure with bootstrap 95% intervals over courses
  * results/mimic_verifier_check.csv  (--windows) E3 verifier verdicts against the selected judge with the whole course and with
                                      the verifier's own three retrieved sentences: candidate retrieval failures of the verifier
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute_metrics import paired_stats  # noqa: E402

PRIVATE, RESULTS = HERE / "results_private", HERE / "results"
METRICS = ["UFR", "CR", "coverage_ref", "words", "n_claims", "n_supported", "n_unsupported"]


def variant_tests(df):
    pairs = [(("E0", "base"), ("E0", "short")), (("E1", "base"), ("E0", "short")), (("E0", "base"), ("E1", "nocite")),
             (("E1b", "base"), ("E1b", "nocite")), (("E0", "short"), ("E1b", "base")), (("E0", "short"), ("E3", "base"))]
    rows = []
    for (ca, va), (cb, vb) in pairs:
        a = df[(df.condition == ca) & (df.variant == va)].set_index("sid"); b = df[(df.condition == cb) & (df.variant == vb)].set_index("sid")
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        for m in METRICS:
            if m in a and m in b:
                rows.append(dict(comparison=f"{cb}/{vb} vs {ca}/{va}", metric=m, **paired_stats(a.loc[common, m], b.loc[common, m])))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_variant_tests.csv", index=False)
    return pd.DataFrame(rows)


def replicates(df):
    reps = df[df.variant.isin(["base", "rep1", "rep2"]) & df.condition.isin(["E0", "E1", "E1b", "E3"])]
    sids = reps[reps.variant == "rep1"].sid.unique()
    reps = reps[reps.sid.isin(sids)]
    rows = []
    for m in ("UFR", "coverage_ref", "words"):
        for c in ("E0", "E1", "E1b", "E3"):
            d = reps[reps.condition == c].pivot(index="sid", columns="variant", values=m).dropna()
            if d.empty:
                continue
            within = d.std(axis=1, ddof=1).mean(); between = d.mean(axis=1).std(ddof=1)
            ms_b = d.shape[1] * d.mean(axis=1).var(ddof=1); ms_w = d.var(axis=1, ddof=1).mean(); k = d.shape[1]
            icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if ms_b + (k - 1) * ms_w > 0 else np.nan
            rows.append(dict(kind="variation", metric=m, condition=c, n_docs=len(d), mean=float(d.values.mean()), within_doc_sd=float(within),
                             between_doc_sd=float(between), icc=float(icc)))
        for a, b in (("E0", "E1"), ("E0", "E1b"), ("E0", "E3"), ("E1", "E3"), ("E1", "E1b")):
            for v in ("base", "rep1", "rep2", "mean"):
                if v == "mean":
                    wa = reps[reps.condition == a].groupby("sid")[m].mean(); wb = reps[reps.condition == b].groupby("sid")[m].mean()
                else:
                    wa = reps[(reps.condition == a) & (reps.variant == v)].set_index("sid")[m]; wb = reps[(reps.condition == b) & (reps.variant == v)].set_index("sid")[m]
                common = wa.index.intersection(wb.index)
                if len(common) >= 5:
                    rows.append(dict(kind="comparison", metric=m, condition=f"{b} vs {a}", generation=v, **paired_stats(wa.loc[common], wb.loc[common])))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_replicates.csv", index=False)
    return pd.DataFrame(rows)


def abstention(df):
    rows = []
    base = df[df.variant == "base"]
    for c in ("E3", "E1", "E1b", "E0"):
        d = base[base.condition == c].dropna(subset=["UFR", "coverage_ref"])
        if d.n_abstentions.nunique() < 2:
            rows.append(dict(condition=c, n_docs=len(d), share_with_abstention=float((d.n_abstentions > 0).mean()))); continue
        r1 = spearmanr(d.n_abstentions, d.coverage_ref); r2 = spearmanr(d.n_abstentions, d.UFR)
        g = d.groupby(d.n_abstentions > 0)
        rows.append(dict(condition=c, n_docs=len(d), share_with_abstention=float((d.n_abstentions > 0).mean()),
                         rho_abst_coverage=float(r1.correlation), p_abst_coverage=float(r1.pvalue), rho_abst_ufr=float(r2.correlation), p_abst_ufr=float(r2.pvalue),
                         coverage_with=float(g.coverage_ref.mean().get(True, np.nan)), coverage_without=float(g.coverage_ref.mean().get(False, np.nan)),
                         ufr_with=float(g.UFR.mean().get(True, np.nan)), ufr_without=float(g.UFR.mean().get(False, np.nan))))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_abstention.csv", index=False)
    return pd.DataFrame(rows)


def equivalence(df, margin=0.05):
    base = df[df.variant == "base"].pivot(index="sid", columns="condition", values="coverage_ref")
    rng = np.random.default_rng(42); rows = []
    for b in ("E1b", "E1", "E3", "E2"):
        d = (base[b] - base["E0"]).dropna().to_numpy()
        boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        lo90, hi90 = np.percentile(boots, [5, 95]); lo95, hi95 = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(comparison=f"{b} vs E0", metric="coverage_ref", delta_mean=float(d.mean()), ci90_low=float(lo90), ci90_high=float(hi90),
                         ci95_low=float(lo95), ci95_high=float(hi95), margin=margin, equivalent_at_margin=bool(lo90 > -margin and hi90 < margin)))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_equivalence.csv", index=False)
    return pd.DataFrame(rows)


def qa_utility():
    f = PRIVATE / "mimic_qa_per_answer.csv"
    if not f.exists():
        return None
    a = pd.read_csv(f)
    a["answered"] = ~a.abstained.astype(bool)
    a["fully_supported"] = a.answered & (a.UFR.fillna(1) == 0)
    a["any_unsupported"] = a.answered & (a.UFR.fillna(0) > 0)
    rows = []
    for (c, q), d in list(a.groupby(["condition", "question"])) + [((c, "all"), d) for c, d in a.groupby("condition")]:
        rows.append(dict(condition=c, question=q, n=len(d), abstained=float(d.abstained.mean()), answered=float(d.answered.mean()),
                         fully_supported_of_all=float(d.fully_supported.mean()), any_unsupported_of_all=float(d.any_unsupported.mean()),
                         fully_supported_of_answered=float(d[d.answered].fully_supported.mean()) if d.answered.any() else np.nan,
                         ufr_answered=float(d[d.answered].UFR.mean()) if d.answered.any() else np.nan))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "mimic_qa_utility.csv", index=False)
    return out


def window_check():
    """Re-score flagged claims with half-overlapping windows: a flag that disappears was caused by evidence split across windows."""
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    from mimic_generate import load_docs
    from calibration_annptsumm import DEFAULT_DATA
    J = Judge("deberta_large_nli", CANDIDATES["deberta_large_nli"]); tau = 0.20
    docs = {d["sid"]: d["text"] for d in load_docs(Path(DEFAULT_DATA))}
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[(cl.variant == "base") & (cl.label != "Supported")]
    rows = []
    for sid, g in cl.groupby("sid"):
        sents = he.sentencize(docs[sid]); half = J.windows(sents, max_tokens=200)
        wins = [" ".join(half[i:i + 2]) for i in range(max(1, len(half) - 1))]   # 400-token windows starting every 200 tokens
        prem, hyp, own = [], [], []
        for k, r in enumerate(g.itertuples()):
            for w in wins:
                prem.append(w); hyp.append(str(r.claim)); own.append(k)
        ps, _ = J.support_scores(prem, hyp); own = np.array(own)
        for k, r in enumerate(g.itertuples()):
            rows.append(dict(condition=r.condition, rescued=bool(ps[own == k].max() >= tau)))
    d = pd.DataFrame(rows)
    out = d.groupby("condition").agg(flagged=("rescued", "size"), rescued_by_overlap=("rescued", "mean")).reset_index()
    out.to_csv(RESULTS / "mimic_window_check.csv", index=False)
    return out


def verifier_check():
    """For every E1 draft claim that the E3 verifier judged, compare its verdict (based on three retrieved course sentences)
    with the selected judge reading the whole course and reading the same three sentences.  A claim the verifier rejected,
    that the judge supports from the whole course but not from the three sentences, is a candidate retrieval failure: the
    support exists in the course but was not retrieved."""
    import json
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    from mimic_generate import load_docs
    from calibration_annptsumm import DEFAULT_DATA
    summ = pd.read_csv(RESULTS / "mimic_summary.csv"); jname, tau = summ.judge.iloc[0], float(summ.tau.iloc[0])
    docs = {d["sid"]: d["text"] for d in load_docs(Path(DEFAULT_DATA))}
    norm = lambda x: " ".join(str(x).split())
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[(cl.condition == "E1") & (cl.variant == "base")]
    whole = {(r.sid, norm(r.claim)): float(r.p_support) for r in cl.itertuples()}
    items = []
    for l in open(PRIVATE / "mimic_generations.jsonl"):
        r = json.loads(l)
        if r["condition"] != "E3" or r["variant"] != "base" or r["sid"] not in docs:
            continue
        sents = he.sentencize(docs[r["sid"]])
        for v in r.get("verification", []):
            ev = [e for e, _ in he.retrieve_top_k(v["claim"], sents, he.TOP_K_EVIDENCE)]
            ev = sorted(ev, key=sents.index)                     # document order, as in the judge's top-3 mode
            items.append(dict(verdict=v["verdict"], claim=v["claim"], premise=" ".join(ev), p_whole=whole.get((r["sid"], norm(v["claim"])))))
    d = pd.DataFrame(items)
    J = Judge(jname, CANDIDATES[jname])
    ps, _ = J.support_scores(d.premise.tolist(), d.claim.tolist()); d["p_top3"] = np.asarray(ps, float)
    rows = []
    for name, g in (("rejected by the verifier", d[d.verdict != "SUPPORTED"]), ("accepted by the verifier", d[d.verdict == "SUPPORTED"])):
        m = g[g.p_whole.notna()]
        w, t = m.p_whole >= tau, m.p_top3 >= tau
        rows.append(dict(group=name, n=len(g), matched=len(m), judge_supports_whole=float(w.mean()), judge_supports_top3=float(t.mean()),
                         whole_not_top3=float((w & ~t).mean()), both=float((w & t).mean()), neither=float((~w & ~t).mean()), tau=tau))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "mimic_verifier_check.csv", index=False)
    return out


def condition_ci(df, n_boot=10000):
    """Mean of each measure per condition and variant, with a percentile bootstrap 95% interval over hospital courses."""
    rng = np.random.default_rng(0); rows = []
    measures = ["UFR", "CR", "coverage_ref", "words", "n_claims", "n_supported", "n_unsupported", "n_abstentions", "omitted_ref",
                "citation_coverage", "citation_accuracy", "citation_reliability"]
    for (c, v), d in df.groupby(["condition", "variant"]):
        for m in [x for x in measures if x in d]:
            x = d[m].dropna().to_numpy(float)
            if len(x) < 5:
                continue
            boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
            rows.append(dict(condition=c, variant=v, metric=m, n=len(x), mean=float(x.mean()), ci95_low=float(np.percentile(boots, 2.5)),
                             ci95_high=float(np.percentile(boots, 97.5))))
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "mimic_condition_ci.csv", index=False)
    return out


def splitting_check():
    """Compare the first claim splitter (spaCy on the raw summary) with citations.attributed_claims.

    Counts, per condition: claims of the first splitter that had a section header merged into them, claims of the first
    splitter that are abstentions once the header is removed (counted as claims by mistake), and citation-marker groups
    that the first splitter dropped (they formed a fragment of fewer than ten characters, such as "[1].") or attached to the
    following sentence (a marker written after the period starts the next spaCy sentence)."""
    import json
    import re
    import hallucination_eval as he
    from citations import MARK, attributed_claims, strip_headers
    from preprocessing import is_abstention, is_markdown_header
    cite_re = re.compile(r"\[(\d+)\]"); norm = lambda x: " ".join(str(x).split())
    nlp = he.get_nlp(); rows = []
    for l in open(PRIVATE / "mimic_generations.jsonl"):
        r = json.loads(l)
        if r["variant"] not in ("base", "nocite") and r["condition"] != "E1":
            continue
        raw = r["summary"]; doc = nlp(raw); sents = list(doc.sents)
        kept, n_first, merged, hidden_abst = [], 0, 0, 0
        for s_ in sents:
            t = s_.text.strip(); clean = cite_re.sub("", t).strip()
            ok = len(t) >= 10 and not is_markdown_header(clean) and len(clean) >= 10
            kept.append(ok)
            if ok and not is_abstention(clean):
                n_first += 1
                if norm(strip_headers(clean)) != norm(clean):
                    merged += 1; hidden_abst += int(is_abstention(strip_headers(clean)))
        dropped = moved = groups = 0
        for m in MARK.finditer(raw):
            pos = m.start() + m.group(0).index("["); groups += 1
            k = next((i for i, s_ in enumerate(sents) if s_.start_char <= pos < s_.end_char), None)
            if k is None or not kept[k]:
                dropped += 1
            elif k > 0 and not raw[sents[k].start_char:pos].strip():
                moved += 1
        new = attributed_claims(raw)
        rows.append(dict(condition=r["condition"], variant=r["variant"], claims_first=n_first, merged_header=merged, abstentions_counted_as_claims=hidden_abst,
                         claims_new=sum(1 for c in new if not c[2]), marker_groups=groups, dropped=dropped, moved=moved,
                         cited_claims_new=sum(1 for c in new if c[1] and not c[2])))
    d = pd.DataFrame(rows)
    out = d.groupby(["condition", "variant"]).sum(numeric_only=True).reset_index()
    out["merged_header_share"] = out.merged_header / out.claims_first
    out["markers_dropped_share"] = out.dropped / out.marker_groups.where(out.marker_groups > 0)
    out["markers_moved_share"] = out.moved / out.marker_groups.where(out.marker_groups > 0)
    out.to_csv(RESULTS / "mimic_splitting_check.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(PRIVATE / "mimic_per_summary.csv")
    pd.set_option("display.width", 220)
    for name, fn in (("variant tests", lambda: variant_tests(df)), ("replicates", lambda: replicates(df)), ("abstention", lambda: abstention(df)),
                     ("equivalence", lambda: equivalence(df)), ("qa utility", qa_utility), ("condition intervals", lambda: condition_ci(df)),
                     ("splitting check", splitting_check)):
        r = fn()
        print(f"=== {name}"); print(r.round(4).to_string(index=False) if r is not None and len(r) else "(no data)")
    if a.windows:
        print("=== window check"); print(window_check().round(3).to_string(index=False))
        print("=== verifier check"); print(verifier_check().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
