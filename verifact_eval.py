#!/usr/bin/env python3
"""
verifact_eval.py — External validation of the selected judge on MIMIC-III-Ext-VeriFact-BHC (PhysioNet, credentialed).

The dataset contains 13,070 propositions decomposed from human-written and LLM-written Brief Hospital Course
summaries of 100 MIMIC-III patients, each labeled Supported / Not Supported / Not Addressed against the patient's
full EHR notes by 2-5 clinicians with adjudication (column human_gt), plus the verdicts of VeriFact's own LLM
raters under many retrieval configurations.  Everything runs locally; only aggregate metrics are written to
results/.  Per-proposition scores go to results_private/ (git-ignored).  Nothing here prints note or proposition text.

Usage:
  python verifact_eval.py --plan                    count windows and judge calls before committing to the full run
  python verifact_eval.py --score                   score every valid proposition with the selected judge
                                                    (resumable; appends to results_private/verifact_scores.jsonl)
  python verifact_eval.py --score --limit-patients 2   smoke run
  python verifact_eval.py --summarize               metrics vs the clinician labels -> results/verifact_judge_eval.csv
  python verifact_eval.py --baseline                agreement of VeriFact's own AI raters with the clinicians on the
                                                    same propositions -> results/verifact_ai_raters.csv

Premise: all EHR notes of the proposition's patient in chart-time order — the same reference the clinicians judged
against — scored with the main study's exact windowing (400-token windows, max over windows) and threshold tau.
Binary task: clinician label Supported vs not (Not Supported and Not Addressed both count as unsupported), matching
the judge's flag = p_support < tau.  Invalid propositions (imperative/interrogative/incomplete/vague) are excluded,
as in the dataset's own analysis.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "results_private"
RESULTS = HERE / "results"
DEFAULT_ROOT = Path.home() / "Desktop" / "Thesis" / "verifact-bhc"
SCORES = PRIVATE / "verifact_scores.jsonl"


def load_propositions(root):
    hv = pd.read_csv(root / "propositions" / "human_verdicts.csv.gz",
                     usecols=["proposition_id", "text", "author_type", "proposition_type", "human_gt"])
    pv = pd.read_csv(root / "propositions" / "proposition_validity.csv.gz", usecols=["proposition_id", "invalid"])
    pr = pd.read_csv(root / "propositions" / "propositions.csv.gz", usecols=["proposition_id", "subject_id"])
    df = hv.merge(pv, on="proposition_id").merge(pr, on="proposition_id")
    return df[~df.invalid.astype(bool)].reset_index(drop=True)


def load_premises(root):
    ne = pd.read_csv(root / "reference_ehr" / "ehr_noteevents.csv.gz")
    t = pd.to_datetime(ne.CHARTTIME, errors="coerce").fillna(pd.to_datetime(ne.CHARTDATE, errors="coerce"))
    ne = ne.assign(_t=t).sort_values(["SUBJECT_ID", "_t", "ROW_ID"])
    return {sid: "\n\n".join(g.TEXT.astype(str)) for sid, g in ne.groupby("SUBJECT_ID")}


def selected_judge():
    summ = pd.read_csv(RESULTS / "mimic_summary.csv")
    return summ.judge.iloc[0], float(summ.tau.iloc[0])


def score(args):
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    jname = args.judge or selected_judge()[0]
    df = load_propositions(args.root)
    prem = load_premises(args.root)
    done = set()
    if SCORES.exists():
        for line in open(SCORES):
            r = json.loads(line)
            if r["judge"] == jname:
                done.add(r["pid"])
    todo = df[~df.proposition_id.isin(done)]
    print(f"{jname}: {len(df)} valid propositions, {len(done)} already scored, {len(todo)} to do", flush=True)
    if not len(todo) and not args.plan:
        return
    J = Judge(jname, CANDIDATES[jname])
    groups = sorted(todo.groupby("subject_id"), key=lambda kv: len(prem[kv[0]]) * len(kv[1]))
    if args.limit_patients:
        groups = groups[: args.limit_patients]
    if args.plan:
        total = 0
        for sid, g in df.groupby("subject_id"):
            total += len(g) * len(J.windows(he.sentencize(prem[sid])))
        print(f"full run: {df.subject_id.nunique()} patients, {len(df)} propositions, {total} judge calls")
        return
    PRIVATE.mkdir(exist_ok=True)
    wins_cache = {}
    for k, (sid, g) in enumerate(groups):
        if sid not in wins_cache:
            wins_cache[sid] = J.windows(he.sentencize(prem[sid]))
        wins = wins_cache[sid]
        hyps = g.text.astype(str).tolist()
        pids = g.proposition_id.tolist()
        best = np.full(len(hyps), -1.0)
        pairs = [(i, w) for i in range(len(hyps)) for w in wins]
        for s in range(0, len(pairs), args.chunk):
            blk = pairs[s: s + args.chunk]
            ps, _ = J.support_scores([w for _, w in blk], [hyps[i] for i, _ in blk])
            for (i, _), p in zip(blk, ps):
                if p > best[i]:
                    best[i] = float(p)
        with open(SCORES, "a") as out:
            for pid, p in zip(pids, best):
                out.write(json.dumps(dict(judge=jname, pid=pid, p=float(p), wins=len(wins))) + "\n")
        print(f"  patient {k + 1}/{len(groups)}: {len(hyps)} propositions x {len(wins)} windows", flush=True)


def binary_metrics(y, flag):
    """y: clinicians say unsupported; flag: judge flags.  Plain counts, no dependencies."""
    tp = int((y & flag).sum()); fn = int((y & ~flag).sum())
    fp = int((~y & flag).sum()); tn = int((~y & ~flag).sum())
    n = tp + fn + fp + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(n=n, prevalence=round((tp + fn) / n, 4),
                sensitivity=round(tp / (tp + fn), 4) if tp + fn else np.nan,
                specificity=round(tn / (tn + fp), 4) if tn + fp else np.nan,
                ppv=round(tp / (tp + fp), 4) if tp + fp else np.nan,
                npv=round(tn / (tn + fn), 4) if tn + fn else np.nan,
                accuracy=round(po, 4), kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else np.nan)


def auroc(y, s):
    """Mann-Whitney AUROC: probability a clinician-unsupported proposition gets a higher unsupportedness score."""
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def summarize(args):
    jname, tau_sel = args.judge or selected_judge()[0], selected_judge()[1]
    tau = args.tau if args.tau is not None else tau_sel
    rows = [json.loads(l) for l in open(SCORES) if json.loads(l)["judge"] == jname]
    sc = pd.DataFrame(rows).drop_duplicates(subset="pid", keep="last")
    df = load_propositions(args.root).merge(sc, left_on="proposition_id", right_on="pid")
    print(f"{jname} at tau {tau}: {len(df)} of {len(load_propositions(args.root))} valid propositions scored")
    y = (df.human_gt != "Supported").to_numpy()
    flag = (df.p < tau).to_numpy()
    out = []
    subsets = [("all", np.ones(len(df), bool))]
    subsets += [(f"author={a}", (df.author_type == a).to_numpy()) for a in sorted(df.author_type.unique())]
    subsets += [(f"type={t}", (df.proposition_type == t).to_numpy()) for t in sorted(df.proposition_type.unique())]
    for name, m in subsets:
        row = dict(subset=name, judge=jname, tau=tau, auroc=round(auroc(y[m], -df.p.to_numpy()[m]), 4))
        row.update(binary_metrics(y[m], flag[m]))
        out.append(row)
    for lab, g in df.groupby("human_gt"):
        out.append(dict(subset=f"label={lab}", judge=jname, tau=tau, n=len(g),
                        flag_rate=round(float((g.p < tau).mean()), 4), median_p=round(float(g.p.median()), 4)))
    res = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "verifact_judge_eval.csv", index=False)
    print(res.to_string(index=False))
    f = RESULTS / "verifact_ai_raters.csv"
    if f.exists():
        best = pd.read_csv(f).iloc[0]
        print(f"context: best VeriFact AI-rater configuration reaches kappa {best.kappa:.3f} "
              f"(sens {best.sensitivity:.3f}, spec {best.specificity:.3f}) on the same propositions")


def baseline(args):
    df = load_propositions(args.root)
    gt = dict(zip(df.proposition_id, df.human_gt))
    cols = ["proposition_id", "rater_name", "rater_alias", "verdict", "retrieval_method", "top_n",
            "reference_format", "reference_only_admission"]
    key_cols = ["rater_name", "rater_alias", "retrieval_method", "top_n", "reference_format", "reference_only_admission"]
    agg = {}
    for ch in pd.read_csv(args.root / "propositions" / "verifact_verdicts.csv.gz", usecols=cols, chunksize=250_000):
        ch = ch[ch.proposition_id.isin(gt)]
        if not len(ch):
            continue
        truth = ch.proposition_id.map(gt)
        ft, fj = truth != "Supported", ch.verdict != "Supported"
        g = (ch.assign(tp=ft & fj, tn=~ft & ~fj, fp=~ft & fj, fn=ft & ~fj, exact=ch.verdict == truth)
               .groupby(key_cols, dropna=False)[["tp", "tn", "fp", "fn", "exact"]].sum())
        for k, r in g.iterrows():
            a = agg.setdefault(k, dict(tp=0, tn=0, fp=0, fn=0, exact=0))
            for c in a:
                a[c] += int(r[c])
    rows = []
    for k, a in agg.items():
        n = a["tp"] + a["tn"] + a["fp"] + a["fn"]
        po = (a["tp"] + a["tn"]) / n
        pe = ((a["tp"] + a["fp"]) * (a["tp"] + a["fn"]) + (a["fn"] + a["tn"]) * (a["fp"] + a["tn"])) / n ** 2
        rows.append({**dict(zip(key_cols, k)),
                     **dict(n=n, sensitivity=round(a["tp"] / max(a["tp"] + a["fn"], 1), 4),
                            specificity=round(a["tn"] / max(a["tn"] + a["fp"], 1), 4),
                            kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else np.nan,
                            agree3way=round(a["exact"] / n, 4))})
    out = pd.DataFrame(rows).sort_values("kappa", ascending=False).reset_index(drop=True)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "verifact_ai_raters.csv", index=False)
    print(f"{len(out)} VeriFact AI-rater configurations -> results/verifact_ai_raters.csv")
    print(out.head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--score", action="store_true"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--baseline", action="store_true"); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--judge", default=None); ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--limit-patients", type=int, default=None); ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()
    if args.baseline:
        baseline(args)
    if args.score or args.plan:
        score(args)
    if args.summarize:
        summarize(args)
    if not (args.baseline or args.score or args.plan or args.summarize):
        print("nothing to do: pass --plan, --score, --summarize or --baseline")


if __name__ == "__main__":
    main()
