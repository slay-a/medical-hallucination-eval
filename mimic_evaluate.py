#!/usr/bin/env python3
"""
mimic_evaluate.py — Score the MIMIC generations (results_private/mimic_generations.jsonl) with a chosen judge.

For every (condition, variant) and for the doctor-written reference summaries:
  UFR, CR             claim-level rates from the judge (header lines and abstentions excluded)
  abstentions         number of "Not stated in the note" lines per summary
  coverage_ref        share of reference (doctor-written) sentences that the generated summary supports
                      (judge run in the reverse direction: premise = generated summary, hypothesis = reference sentence)
  citation_accuracy   share of cited claims whose cited excerpt(s) support the claim (E1/E1b variants with citations)
  words, claims
Paired Wilcoxon tests against E0 and E1 and bootstrap CIs are computed with recompute_metrics.paired_stats.
The doctor-written summaries are scored too, and their expert-flag rate is reported next to the judge's rate.

The judge is any candidate from judge_candidates.CANDIDATES (default: the one with the best expert agreement,
read from results/judge_candidates.csv) with evidence mode 'top3' or 'doc'.

Outputs (aggregates only, safe to commit): results/mimic_summary.csv, results/mimic_pairwise_tests.csv,
results/mimic_ablations.csv, results/mimic_reference.csv; per-claim details -> results_private/mimic_claims.csv
Usage: python mimic_evaluate.py [--judge NAME] [--mode top3|doc] [--tau 0.5]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from preprocessing import is_abstention, is_markdown_header  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA, read_jsonl  # noqa: E402
from judge_candidates import CANDIDATES, Judge  # noqa: E402
from recompute_metrics import paired_stats  # noqa: E402

RESULTS, PRIVATE = HERE / "results", HERE / "results_private"
GEN = PRIVATE / "mimic_generations.jsonl"
CITE_RE = re.compile(r"\[(\d+)\]")


def best_judge():
    f = RESULTS / "judge_candidates.csv"
    if not f.exists():
        return "minilm_nli", "top3"
    df = pd.read_csv(f); df = df[df.group == "all"].sort_values("kappa", ascending=False)
    return str(df.iloc[0]["judge"]), str(df.iloc[0]["mode"])   # bracket access: .mode is a pandas method


def split_claims(summary):
    """[(claim, cited excerpt numbers, is_abstention)]: markers are removed before sentence splitting and attributed to the
    sentence they follow (citations.attributed_claims); merged section-header lines are stripped from claims."""
    from citations import attributed_claims
    return [(c, cites, ab) for c, cites, ab, _ in attributed_claims(summary)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None); ap.add_argument("--mode", default=None); ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--reuse-reverse", action="store_true", help="reuse saved reverse (coverage) scores for summaries scored before")
    ap.add_argument("--reuse-forward", action="store_true", help="reuse the per-claim support scores saved by a previous run (results_private/mimic_claims.csv) when the claim texts are unchanged")
    ap.add_argument("--contra-from", default=None, help="claims CSV of an NLI judge whose p_contra provides the contradiction channel when the selected judge is a binary grounding checker")
    args = ap.parse_args()
    jname, jmode = best_judge()
    jname = args.judge or jname; jmode = args.mode or jmode
    jc = pd.read_csv(RESULTS / "judge_candidates.csv") if (RESULTS / "judge_candidates.csv").exists() else None
    tau = args.tau
    if tau is None and jc is not None:
        row = jc[(jc.judge == jname) & (jc["mode"] == jmode) & (jc.group == "all")]
        tau = float(row.tau_from_other_subset.iloc[0]) if len(row) else 0.5
    tau = tau or 0.5
    print(f"judge={jname} mode={jmode} tau={tau}", flush=True)
    J = Judge(jname, CANDIDATES[jname]); nlp = he.get_nlp(); enc = he.get_bi_encoder()

    docs = {}
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di.jsonl")):
        docs[f"doc_{i}"] = dict(text=r["text"], reference=r["summary"], labels=r["labels"])
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di_validation.jsonl")):
        docs[f"val_{i}"] = dict(text=r["text"], reference=r["summary"], labels=r["labels"])
    gens = [json.loads(l) for l in open(GEN)]
    gens.append(None)  # placeholder to also score references
    records = [dict(sid=s, condition="REF", variant="base", summary=d["reference"], excerpts=[]) for s, d in docs.items()]
    records += [g for g in gens if g is not None]

    def support(premises, hyps):
        if not hyps:
            return np.array([]), np.array([])
        return J.support_scores(premises, hyps)

    claim_rows, summ_rows, raw_rows, cite_rows = [], [], [], []
    ctx_cache = {}
    # claim-level cache: the support of a claim depends only on the claim text and its hospital course
    fwd = {}
    if args.reuse_forward and (PRIVATE / "mimic_claims.csv").exists():
        prev = pd.read_csv(PRIVATE / "mimic_claims.csv")
        fwd = {(r.sid, str(r.claim)): (float(r.p_support), float(r.p_contra)) for r in prev.itertuples()}
        print(f"forward cache: {len(fwd)} scored claims", flush=True)
    rev = {}
    if args.reuse_reverse and (PRIVATE / "mimic_raw_scores.jsonl").exists():
        for l in open(PRIVATE / "mimic_raw_scores.jsonl"):
            r = json.loads(l); rev[(r["sid"], r["condition"], r["variant"])] = np.array(r["reverse_support"], float)
        print(f"reverse cache: {len(rev)} summaries", flush=True)
    cite_cache = {}
    if (PRIVATE / "mimic_citation_claims.csv").exists():
        pc_prev = pd.read_csv(PRIVATE / "mimic_citation_claims.csv")
        cite_cache = {(r.sid, str(r.claim), str(r.cited_key)): float(r.p_cite) for r in pc_prev.itertuples()}
    n_fwd_new = n_rev_new = n_cite_new = 0
    contra = {}
    if args.contra_from and Path(args.contra_from).exists():
        cprev = pd.read_csv(args.contra_from)
        for key, g in cprev.groupby(["sid", "condition", "variant"], sort=False):
            contra[key] = dict(zip(g.claim.astype(str), g.p_contra.astype(float)))
        print(f"contradiction channel taken from {args.contra_from} for {len(contra)} summaries", flush=True)

    def label_of(p, q):
        return "Supported" if p >= tau else ("Contradicted" if (q >= 0.5 and q > p) else "Not-Supported")

    for rec in records:
        sid, cond, var = rec["sid"], rec["condition"], rec["variant"]
        if sid not in docs:
            continue
        text, ref = docs[sid]["text"], docs[sid]["reference"]
        if sid not in ctx_cache:
            cs = he.sentencize(text); ctx_cache[sid] = (cs, enc.encode(cs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False),
                                                       [s for s in he.sentencize(ref) if not is_markdown_header(s)])
        ctx_sents, S, ref_sents = ctx_cache[sid]
        parts = split_claims(rec["summary"])
        claims = [(c, cites) for c, cites, ab in parts if not ab]; n_abst = sum(1 for _, _, ab in parts if ab)
        # ---- forward: claims vs course (cached per claim)
        ps = np.zeros(len(claims)); pc = np.zeros(len(claims))
        todo = [i for i, (c, _) in enumerate(claims) if (sid, c) not in fwd]
        for i, (c, _) in enumerate(claims):
            if (sid, c) in fwd:
                ps[i], pc[i] = fwd[(sid, c)]
        if todo:
            n_fwd_new += len(todo)
            if jmode == "top3":
                C = enc.encode([claims[i][0] for i in todo], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                order = np.argsort(-(C @ S.T), axis=1)
                prem = [" ".join(ctx_sents[j] for j in sorted(order[k][:3].tolist())) for k in range(len(todo))]
                a, b = support(prem, [claims[i][0] for i in todo])
            else:
                prem, hyps, owners = [], [], []
                wins = J.windows(ctx_sents)
                for k, i in enumerate(todo):
                    for w in wins:
                        prem.append(w); hyps.append(claims[i][0]); owners.append(k)
                pw, cw = support(prem, hyps); owners = np.array(owners)
                a = np.array([pw[owners == k].max() for k in range(len(todo))]); b = np.array([cw[owners == k].max() for k in range(len(todo))])
            for k, i in enumerate(todo):
                ps[i], pc[i] = float(a[k]), float(b[k]); fwd[(sid, claims[i][0])] = (ps[i], pc[i])
        if claims and contra and getattr(J, "kind", "") == "llm_yesno":
            cmap = contra.get((sid, cond, var), {})
            pc = np.array([cmap.get(c, 0.0) for c, _ in claims])
        labels = [label_of(p, q) for p, q in zip(ps, pc)]
        # ---- citations: claims with at least one valid excerpt number; cited excerpt(s) vs claim (cached)
        cite_ok = cite_tot = 0
        ex = rec.get("excerpts") or []
        if ex:
            for c, cites in claims:
                valid = sorted({k for k in cites if 1 <= k <= len(ex)})
                if not valid:
                    continue
                key = ",".join(map(str, valid))
                if (sid, c, key) not in cite_cache:
                    pcs, _ = support([" ".join(ex[k - 1] for k in valid)], [c]); cite_cache[(sid, c, key)] = float(pcs[0]); n_cite_new += 1
                pcite = cite_cache[(sid, c, key)]
                cite_tot += 1; cite_ok += int(pcite >= tau)
                cite_rows.append(dict(sid=sid, condition=cond, variant=var, claim=c, cited_key=key, p_cite=round(pcite, 4), judge_supported=pcite >= tau))
        # ---- reverse: reference sentences supported by the generated summary (coverage), cached per summary
        if ref_sents:
            pr = rev.get((sid, cond, var))
            if pr is None or len(pr) != len(ref_sents):
                pr, _ = support([rec["summary"]] * len(ref_sents), ref_sents); n_rev_new += 1
            coverage = float((np.asarray(pr) >= tau).mean())
        else:
            pr = np.array([]); coverage = np.nan
        raw_rows.append(dict(sid=sid, condition=cond, variant=var, reverse_support=[round(float(x), 4) for x in pr]))
        for (c, cites), lab, p, q in zip(claims, labels, ps, pc):
            claim_rows.append(dict(sid=sid, condition=cond, variant=var, claim=c, label=lab, p_support=round(float(p), 4), p_contra=round(float(q), 4), cites=cites))
        n = len(claims); nc = labels.count("Contradicted"); nn = labels.count("Not-Supported")
        summ_rows.append(dict(sid=sid, condition=cond, variant=var, n_claims=n, n_abstentions=n_abst, UFR=(nc + nn) / n if n else np.nan, CR=nc / n if n else np.nan,
                              n_supported=n - nc - nn, n_unsupported=nc + nn, n_contradicted=nc,
                              words=len(CITE_RE.sub("", rec["summary"]).split()), coverage_ref=coverage, n_ref_sents=len(ref_sents),
                              omitted_ref=len(ref_sents) * (1 - coverage) if ref_sents else np.nan,
                              cited_claims=cite_tot, cite_correct=cite_ok,
                              citation_accuracy=cite_ok / cite_tot if cite_tot else np.nan,
                              citation_coverage=(cite_tot / n if n else np.nan) if ex else np.nan,
                              citation_reliability=(cite_ok / n if n else np.nan) if ex else np.nan))
    df = pd.DataFrame(summ_rows); pd.DataFrame(claim_rows).to_csv(PRIVATE / "mimic_claims.csv", index=False)
    PRIVATE.mkdir(exist_ok=True); df.to_csv(PRIVATE / "mimic_per_summary.csv", index=False)
    pd.DataFrame(cite_rows).to_csv(PRIVATE / "mimic_citation_claims.csv", index=False)
    with open(PRIVATE / "mimic_raw_scores.jsonl", "w") as fh:   # raw reverse scores, so that thresholds can be changed offline
        for r in raw_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"newly scored: {n_fwd_new} claims, {n_rev_new} summaries for coverage, {n_cite_new} citations", flush=True)

    # ---- aggregates
    agg = df.groupby(["condition", "variant"]).agg(n_docs=("sid", "nunique"), UFR_mean=("UFR", "mean"), UFR_median=("UFR", "median"), CR_mean=("CR", "mean"),
                                                   CR_median=("CR", "median"), claims_mean=("n_claims", "mean"), abstentions_mean=("n_abstentions", "mean"),
                                                   supported_mean=("n_supported", "mean"), unsupported_mean=("n_unsupported", "mean"),
                                                   contradicted_mean=("n_contradicted", "mean"),
                                                   words_mean=("words", "mean"), coverage_ref_mean=("coverage_ref", "mean"),
                                                   ref_sents_mean=("n_ref_sents", "mean"), omitted_ref_mean=("omitted_ref", "mean"),
                                                   citation_accuracy_mean=("citation_accuracy", "mean"), citation_coverage_mean=("citation_coverage", "mean"),
                                                   citation_reliability_mean=("citation_reliability", "mean"),
                                                   cited_claims=("cited_claims", "sum"), cite_correct=("cite_correct", "sum"), claims_total=("n_claims", "sum")).reset_index()
    agg["judge"], agg["mode"], agg["tau"] = jname, jmode, tau
    agg.to_csv(RESULTS / "mimic_summary.csv", index=False)
    # expert-flag rate on the doctor-written references (for the same documents)
    exp = []
    for sid, d in docs.items():
        sents = [s for s in he.sentencize(d["reference"]) if not is_markdown_header(s)]
        exp.append(dict(sid=sid, n_ref_sents=len(sents), expert_spans=len(d["labels"])))
    pd.DataFrame(exp).merge(df[df.condition == "REF"][["sid", "UFR", "CR"]], on="sid").to_csv(RESULTS / "mimic_reference.csv", index=False)
    # paired tests (base variants) vs E0, vs E1 and vs the clinician's instructions; Holm adjustment over the primary family
    # (each of E1, E1b, E2, E3 against E0 on UFR, CR and coverage: twelve tests)
    def holm(pvals):
        pv = np.asarray(pvals, float); n = len(pv); order = np.argsort(pv); adj = np.empty(n); running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (n - rank) * pv[idx])); adj[idx] = running
        return adj
    base = df[df.variant == "base"]
    rows = []
    for m in ("UFR", "CR", "coverage_ref", "words", "n_claims", "n_supported", "n_unsupported", "omitted_ref", "citation_coverage", "citation_reliability", "citation_accuracy"):
        w = base.pivot(index="sid", columns="condition", values=m)
        for a in ("E0", "E1"):
            for b in [c for c in w.columns if c not in ("REF", a)]:
                if a in w.columns and b in w.columns and a != b and w[[a, b]].notna().all(axis=1).sum() >= 5:
                    st = paired_stats(w[a], w[b]); rows.append(dict(metric=m, comparison=f"{b} vs {a}", **st))
        if "REF" in w.columns:
            for b in [c for c in w.columns if c != "REF"]:
                if w[["REF", b]].notna().all(axis=1).sum() >= 5:
                    st = paired_stats(w["REF"], w[b]); rows.append(dict(metric=m, comparison=f"{b} vs REF", **st))
    tests = pd.DataFrame(rows)
    fam = tests.metric.isin(["UFR", "CR", "coverage_ref"]) & tests.comparison.isin([f"{c} vs E0" for c in ("E1", "E1b", "E2", "E3")])
    tests["p_holm_primary"] = np.nan
    tests.loc[fam, "p_holm_primary"] = holm(tests.loc[fam, "p"].fillna(1.0).values)
    tests.to_csv(RESULTS / "mimic_pairwise_tests.csv", index=False)
    # retrieval ablations: E1 variants vs E1 base (only the six pre-specified variants)
    from mimic_generate import VARIANTS
    ab = df[(df.condition == "E1") & df.variant.isin(list(VARIANTS))]; rows = []
    for m in ("UFR", "CR", "coverage_ref", "citation_accuracy", "citation_coverage", "citation_reliability", "words", "n_claims"):
        w = ab.pivot(index="sid", columns="variant", values=m)
        for v in [c for c in w.columns if c != "base"]:
            if w[["base", v]].notna().all(axis=1).sum() >= 5:
                st = paired_stats(w["base"], w[v]); rows.append(dict(metric=m, variant=v, **st))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_ablations.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg[["condition", "variant", "n_docs", "UFR_mean", "CR_mean", "claims_mean", "supported_mean", "unsupported_mean", "words_mean",
               "coverage_ref_mean", "citation_coverage_mean", "citation_accuracy_mean", "citation_reliability_mean"]].round(3).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
