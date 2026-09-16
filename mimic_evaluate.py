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
    return df.iloc[0].judge, df.iloc[0].mode


def split_claims(summary):
    """Return list of (claim_text_without_citations, cited_excerpt_numbers) for non-header, non-abstention sentences."""
    out = []
    for s in he.sentencize(summary):
        cites = [int(x) for x in CITE_RE.findall(s)]
        clean = CITE_RE.sub("", s).strip()
        if is_markdown_header(clean) or len(clean) < 10:
            continue
        out.append((clean, cites, is_abstention(clean)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None); ap.add_argument("--mode", default=None); ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    jname, jmode = best_judge()
    jname = args.judge or jname; jmode = args.mode or jmode
    jc = pd.read_csv(RESULTS / "judge_candidates.csv") if (RESULTS / "judge_candidates.csv").exists() else None
    tau = args.tau
    if tau is None and jc is not None:
        row = jc[(jc.judge == jname) & (jc.mode == jmode) & (jc.group == "all")]
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

    claim_rows, summ_rows = [], []
    ctx_cache = {}
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
        # ---- forward: claims vs note
        if claims:
            C = enc.encode([c for c, _ in claims], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            order = np.argsort(-(C @ S.T), axis=1)
            if jmode == "top3":
                prem = [" ".join(ctx_sents[j] for j in sorted(order[i][:3].tolist())) for i in range(len(claims))]
                ps, pc = support(prem, [c for c, _ in claims])
            else:
                prem, hyps, owners = [], [], []
                for i, (c, _) in enumerate(claims):
                    for w in J.windows(ctx_sents):
                        prem.append(w); hyps.append(c); owners.append(i)
                pw, cw = support(prem, hyps); owners = np.array(owners)
                ps = np.array([pw[owners == i].max() for i in range(len(claims))]); pc = np.array([cw[owners == i].max() if len(cw) else 0 for i in range(len(claims))])
            labels = ["Supported" if p >= tau else ("Contradicted" if (q >= 0.5 and q > p) else "Not-Supported") for p, q in zip(ps, pc)]
        else:
            ps = pc = np.array([]); labels = []
        # ---- citations: cited excerpt(s) vs claim
        cite_ok = cite_tot = 0
        if rec.get("excerpts"):
            ex = rec["excerpts"]; cited = [(c, [ex[k - 1] for k in cites if 1 <= k <= len(ex)]) for c, cites in claims if cites]
            if cited:
                pcs, _ = support([" ".join(e) for _, e in cited], [c for c, _ in cited])
                cite_ok = int((pcs >= tau).sum()); cite_tot = len(cited)
        # ---- reverse: reference sentences supported by the generated summary (coverage)
        if ref_sents:
            prem_r = [rec["summary"]] * len(ref_sents)
            pr, _ = support(prem_r, ref_sents); coverage = float((pr >= tau).mean())
        else:
            coverage = np.nan
        for (c, cites), lab, p, q in zip(claims, labels, ps, pc):
            claim_rows.append(dict(sid=sid, condition=cond, variant=var, claim=c, label=lab, p_support=round(float(p), 4), p_contra=round(float(q), 4), cites=cites))
        n = len(claims); nc = labels.count("Contradicted"); nn = labels.count("Not-Supported")
        summ_rows.append(dict(sid=sid, condition=cond, variant=var, n_claims=n, n_abstentions=n_abst, UFR=(nc + nn) / n if n else np.nan, CR=nc / n if n else np.nan,
                              words=len(CITE_RE.sub("", rec["summary"]).split()), coverage_ref=coverage, cited_claims=cite_tot, cite_correct=cite_ok,
                              citation_accuracy=cite_ok / cite_tot if cite_tot else np.nan))
    df = pd.DataFrame(summ_rows); pd.DataFrame(claim_rows).to_csv(PRIVATE / "mimic_claims.csv", index=False)
    PRIVATE.mkdir(exist_ok=True); df.to_csv(PRIVATE / "mimic_per_summary.csv", index=False)

    # ---- aggregates
    agg = df.groupby(["condition", "variant"]).agg(n_docs=("sid", "nunique"), UFR_mean=("UFR", "mean"), UFR_median=("UFR", "median"), CR_mean=("CR", "mean"),
                                                   CR_median=("CR", "median"), claims_mean=("n_claims", "mean"), abstentions_mean=("n_abstentions", "mean"),
                                                   words_mean=("words", "mean"), coverage_ref_mean=("coverage_ref", "mean"),
                                                   citation_accuracy_mean=("citation_accuracy", "mean"), cited_claims=("cited_claims", "sum"), cite_correct=("cite_correct", "sum")).reset_index()
    agg["judge"], agg["mode"], agg["tau"] = jname, jmode, tau
    agg.to_csv(RESULTS / "mimic_summary.csv", index=False)
    # expert-flag rate on the doctor-written references (for the same documents)
    exp = []
    for sid, d in docs.items():
        sents = [s for s in he.sentencize(d["reference"]) if not is_markdown_header(s)]
        exp.append(dict(sid=sid, n_ref_sents=len(sents), expert_spans=len(d["labels"])))
    pd.DataFrame(exp).merge(df[df.condition == "REF"][["sid", "UFR", "CR"]], on="sid").to_csv(RESULTS / "mimic_reference.csv", index=False)
    # paired tests (base variants) vs E0 and vs E1
    base = df[df.variant == "base"]
    rows = []
    for m in ("UFR", "CR", "coverage_ref", "words"):
        w = base.pivot(index="sid", columns="condition", values=m)
        for a in ("E0", "E1"):
            for b in [c for c in w.columns if c not in ("REF", a)]:
                if a in w.columns and b in w.columns and a != b:
                    st = paired_stats(w[a], w[b]); rows.append(dict(metric=m, comparison=f"{b} vs {a}", **st))
        if "REF" in w.columns:
            for b in [c for c in w.columns if c != "REF"]:
                st = paired_stats(w["REF"], w[b]); rows.append(dict(metric=m, comparison=f"{b} vs REF", **st))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_pairwise_tests.csv", index=False)
    # ablations: E1 variants vs E1 base
    ab = df[df.condition == "E1"]; rows = []
    for m in ("UFR", "CR", "coverage_ref", "citation_accuracy", "words"):
        w = ab.pivot(index="sid", columns="variant", values=m)
        for v in [c for c in w.columns if c != "base"]:
            st = paired_stats(w["base"], w[v]); rows.append(dict(metric=m, variant=v, **st))
    pd.DataFrame(rows).to_csv(RESULTS / "mimic_ablations.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg[["condition", "variant", "n_docs", "UFR_mean", "CR_mean", "claims_mean", "abstentions_mean", "words_mean", "coverage_ref_mean", "citation_accuracy_mean"]].round(3).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
