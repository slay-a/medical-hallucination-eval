#!/usr/bin/env python3
"""
verifact_eval.py — External validation of the selected judge on MIMIC-III-Ext-VeriFact-BHC (PhysioNet, credentialed).

The dataset contains 13,070 propositions decomposed from human-written and LLM-written Brief Hospital Course
summaries of 100 MIMIC-III patients, each labeled Supported / Not Supported / Not Addressed against the patient's
full EHR notes by 2-5 clinicians with adjudication (column human_gt), plus the verdicts of VeriFact's own LLM
raters under many retrieval configurations.  Everything runs locally; only aggregate metrics are written to
results/.  Per-proposition scores go to results_private/ (git-ignored).  Nothing here prints note or proposition text.

Usage:
  python verifact_eval.py --plan --premise admission     count windows and judge calls before committing to the full run
  python verifact_eval.py --score --premise admission    score every valid proposition with the selected judge
                                                         (resumable; appends to results_private/verifact_scores.jsonl)
  python verifact_eval.py --score --premise admission --limit-patients 2   smoke run
  python verifact_eval.py --summarize --premise admission   metrics vs the clinician labels -> results/verifact_judge_eval.csv
  python verifact_eval.py --baseline                     agreement of VeriFact's own AI raters with the clinicians on the
                                                         same propositions -> results/verifact_ai_raters.csv

Premise: with --premise patient, all EHR notes of the proposition's patient in chart-time order (the reference the
clinicians judged against); with --premise admission, the notes of the summarized admission only (the reference of
VeriFact's reference_only_admission raters).  Scoring uses the main study's exact windowing (400-token windows, max
over windows) and threshold tau.  An admission's notes span 15-606 windows against 1-3 for a main-study course, so
--topk K (default 5) scores each proposition only against the K windows ranked highest by --retriever: the
judge-selection bi-encoder (dense; all-MiniLM-L6-v2, a window ranks by its most similar sentence), BM25 over windows,
or both lists interleaved (hybrid, the default and the thesis setting: on the 11 shortest admissions it agreed with the
clinicians better than dense, kappa .38 vs .31) — the retrieve-then-judge design of VeriFact's own raters.  --topk 0
scores every window; the shortest admissions were scored that way too, and --summarize compares the two on them.
--summarize --tag NAME writes a side file (results/verifact_judge_eval_NAME.csv) instead of the main one.
Binary task: clinician label Supported vs not (Not Supported and Not Addressed both count as unsupported), matching
the judge's flag = p_support < tau.  Invalid propositions (imperative/interrogative/incomplete/vague) are excluded,
as in the dataset's own analysis.
Memory: judge batches are padded to a multiple of 64 tokens and the MPS cache is released after every chunk.  The Metal
graph cache keeps a compiled kernel per tensor shape, so with dynamic padding a day-long run grew to 22 GB and swapped;
padding is masked and does not change the scores.  The bi-encoder runs on the CPU for the same reason.  The Metal
working pool of DeBERTa-large inference is about 10 GB at the main study's batch of 16 on a 24 GB laptop; --batch 8
halves it (a pair's scores do not depend on the batch), so the full run was scored with --batch 8 --chunk 256.
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
    pr = pd.read_csv(root / "propositions" / "propositions.csv.gz", usecols=["proposition_id", "subject_id", "hadm_id"])
    df = hv.merge(pv, on="proposition_id").merge(pr, on="proposition_id")
    return df[~df.invalid.astype(bool)].reset_index(drop=True)


def load_premises(root, mode="patient"):
    """Reference text per key.  'patient': all of the patient's notes, keyed by subject_id.  'admission': the notes of
    one admission, keyed by hadm_id — the reference that VeriFact's reference_only_admission raters use."""
    ne = pd.read_csv(root / "reference_ehr" / "ehr_noteevents.csv.gz")
    t = pd.to_datetime(ne.CHARTTIME, errors="coerce").fillna(pd.to_datetime(ne.CHARTDATE, errors="coerce"))
    ne = ne.assign(_t=t).sort_values(["SUBJECT_ID", "_t", "ROW_ID"])
    if mode == "admission":
        ne = ne.dropna(subset=["HADM_ID"])
        return {int(h): "\n\n".join(g.TEXT.astype(str)) for h, g in ne.groupby("HADM_ID")}
    return {sid: "\n\n".join(g.TEXT.astype(str)) for sid, g in ne.groupby("SUBJECT_ID")}


def with_premise_keys(df, root, mode):
    """Adds df.pkey and returns the premise dict it indexes.  In admission mode a proposition whose admission has no
    notes falls back to the patient's notes; the count is printed."""
    pat = {("pat", k): v for k, v in load_premises(root, "patient").items()}
    if mode == "admission":
        adm = load_premises(root, "admission")
        df["pkey"] = [("adm", int(h)) if pd.notna(h) and int(h) in adm else ("pat", s) for h, s in zip(df.hadm_id, df.subject_id)]
        n_fb = sum(1 for k in df.pkey if k[0] == "pat")
        print(f"premise: notes of the summarized admission; {n_fb} of {len(df)} propositions fall back to the patient's notes", flush=True)
        return {**pat, **{("adm", k): v for k, v in adm.items()}}
    df["pkey"] = [("pat", s) for s in df.subject_id]
    return pat


def selected_judge():
    summ = pd.read_csv(RESULTS / "mimic_summary.csv")
    return summ.judge.iloc[0], float(summ.tau.iloc[0])


def rank_windows(hyps, sents, members, wins, enc, retriever, k):
    """Per proposition, the indices of the k windows to score, best first.  dense: a window ranks by its sentence most
    similar to the proposition under the judge-selection bi-encoder; bm25: BM25 over whole windows; hybrid: the two
    lists interleaved (dense first) without repeats."""
    import re
    order = {}
    if retriever in ("dense", "hybrid"):
        S = enc.encode(sents, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        Q = enc.encode(hyps, convert_to_numpy=True, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        sim = Q @ S.T
        W = np.stack([sim[:, m].max(axis=1) for m in members], axis=1)      # propositions x windows
        order["dense"] = np.argsort(-W, axis=1)
    if retriever in ("bm25", "hybrid"):
        from rank_bm25 import BM25Okapi
        tokz = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        bm = BM25Okapi([tokz(w) for w in wins])
        order["bm25"] = np.stack([np.argsort(-bm.get_scores(tokz(h))) for h in hyps])
    if retriever != "hybrid":
        return order[retriever][:, :k]
    out = []
    for d, b in zip(order["dense"], order["bm25"]):
        seen = []
        for a, c in zip(d, b):
            for w in (int(a), int(c)):
                if w not in seen:
                    seen.append(w)
        out.append(seen[:k])
    return np.array(out)


def release_cache():
    """Hand the MPS allocator's cached blocks back to the system.  Dynamically padded batches come in many shapes and
    the allocator keeps a cached block for each until its collector starts at 1.4x the GPU's recommended working set
    (25 GB on a 24 GB laptop); without this the full run grew to 22 GB and swapped."""
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def gpu_gb():
    """(Metal memory held by this process, part of it owned by torch's allocator), in GB; zeros without MPS."""
    import torch
    if not torch.backends.mps.is_available():
        return 0.0, 0.0
    return torch.mps.driver_allocated_memory() / 2**30, torch.mps.current_allocated_memory() / 2**30


def score(args):
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    jname = args.judge or selected_judge()[0]
    df = load_propositions(args.root)
    prem = with_premise_keys(df, args.root, args.premise)
    done = set()
    if SCORES.exists():
        for line in open(SCORES):
            r = json.loads(line)
            if r["judge"] == jname and r.get("premise", "patient") == args.premise and r.get("topk", 0) == args.topk \
                    and (not args.topk or r.get("retriever", "dense") == args.retriever):
                done.add(r["pid"])
    todo = df[~df.proposition_id.isin(done)]
    print(f"{jname}, premise {args.premise}, top-{args.topk or 'all'} windows ({args.retriever}): {len(df)} valid propositions, "
          f"{len(done)} already scored, {len(todo)} to do", flush=True)
    if not len(todo) and not args.plan:
        return
    J = Judge(jname, CANDIDATES[jname])
    enc = he.get_bi_encoder() if args.topk else None
    if enc is not None:
        enc.to("cpu")          # ~10k sentences per admission take seconds on the CPU; on MPS every new padded shape stays compiled
    groups = sorted(todo.groupby("pkey"), key=lambda kv: len(prem[kv[0]]) * len(kv[1]))
    if args.limit_patients:
        groups = groups[: args.limit_patients]
    if args.plan:
        total = 0
        for key, g in df.groupby("pkey"):
            n_w = len(J.windows(he.sentencize(prem[key])))
            total += len(g) * (min(n_w, args.topk) if args.topk else n_w)
        print(f"full run: {df.pkey.nunique()} reference texts, {len(df)} propositions, {total} judge calls")
        return
    PRIVATE.mkdir(exist_ok=True)
    for k, (key, g) in enumerate(groups):
        sents = he.sentencize(prem[key])
        wins, members = J.windows(sents, members=True)
        hyps = g.text.astype(str).tolist()
        pids = g.proposition_id.tolist()
        if args.topk and len(wins) > args.topk:
            cand = rank_windows(hyps, sents, members, wins, enc, args.retriever, args.topk)
        else:
            cand = np.tile(np.arange(len(wins)), (len(hyps), 1))
        release_cache()
        best = np.full(len(hyps), -1.0); best_rank = np.zeros(len(hyps), int)
        pairs = [(i, r, int(w)) for i in range(len(hyps)) for r, w in enumerate(cand[i])]
        for s in range(0, len(pairs), args.chunk):
            blk = pairs[s: s + args.chunk]
            ps, _ = J.support_scores([wins[w] for _, _, w in blk], [hyps[i] for i, _, _ in blk], pad_to_multiple_of=64,
                                    batch=args.batch)
            for (i, r, _), p in zip(blk, ps):
                if p > best[i]:
                    best[i], best_rank[i] = float(p), r
            release_cache()
            print(f"    {min(s + args.chunk, len(pairs))}/{len(pairs)} pairs, gpu %.1f GB (torch %.1f)" % gpu_gb(), flush=True)
        with open(SCORES, "a") as out:
            for pid, p, r in zip(pids, best, best_rank):
                out.write(json.dumps(dict(judge=jname, premise=args.premise, topk=args.topk, retriever=args.retriever,
                                          pid=pid, p=float(p), wins=len(wins), best_rank=int(r))) + "\n")
        print(f"  reference text {k + 1}/{len(groups)}: {len(hyps)} propositions x {cand.shape[1]} of {len(wins)} windows",
              flush=True)


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
    same_ref = [r for r in (json.loads(l) for l in open(SCORES))
                if r["judge"] == jname and r.get("premise", "patient") == args.premise]
    rows = [r for r in same_ref if r.get("topk", 0) == args.topk
            and (not args.topk or r.get("retriever", "dense") == args.retriever)]
    sc = pd.DataFrame(rows).drop_duplicates(subset="pid", keep="last")
    full = load_propositions(args.root)
    df = full.merge(sc, left_on="proposition_id", right_on="pid")
    retr = args.retriever if args.topk else "none"
    print(f"{jname} at tau {tau}, premise {args.premise}, top-{args.topk or 'all'} windows ({retr}): "
          f"{len(df)} of {len(full)} valid propositions scored")
    y = (df.human_gt != "Supported").to_numpy()
    flag = (df.p < tau).to_numpy()
    out = []
    subsets = [("all", np.ones(len(df), bool))]
    subsets += [(f"author={a}", (df.author_type == a).to_numpy()) for a in sorted(df.author_type.unique())]
    subsets += [(f"type={t}", (df.proposition_type == t).to_numpy()) for t in sorted(df.proposition_type.unique())]
    for name, m in subsets:
        row = dict(subset=name, judge=jname, premise=args.premise, topk=args.topk, retriever=retr, tau=tau,
                   auroc=round(auroc(y[m], -df.p.to_numpy()[m]), 4))
        row.update(binary_metrics(y[m], flag[m]))
        if name == "all":
            row.update(judge_calls=int((np.minimum(df.wins, args.topk) if args.topk else df.wins).sum()),
                       min_windows=int(df.wins.min()), median_windows=int(df.wins.median()), max_windows=int(df.wins.max()))
            if args.topk:   # how often the cap binds, and how often the best window was the last one retrieved
                capped = df[df.wins > args.topk]
                row.update(share_capped=round(len(capped) / len(df), 4),
                           share_best_at_last_rank=round(float((capped.best_rank == args.topk - 1).mean()), 4) if len(capped) else np.nan)
        out.append(row)
    for lab, g in df.groupby("human_gt"):
        out.append(dict(subset=f"label={lab}", judge=jname, premise=args.premise, topk=args.topk, retriever=retr, tau=tau,
                        n=len(g), flag_rate=round(float((g.p < tau).mean()), 4), median_p=round(float(g.p.median()), 4)))
    if args.topk:   # propositions also scored against every window (the smallest references): what the cap changed
        every = {r["pid"]: r["p"] for r in same_ref if r.get("topk", 0) == 0}
        both = df[df.pid.isin(every)]
        if len(both):
            yb = (both.human_gt != "Supported").to_numpy()
            pa, pe = both.p.to_numpy(), both.pid.map(every).to_numpy()
            ma, me = binary_metrics(yb, pa < tau), binary_metrics(yb, pe < tau)
            out.append(dict(subset="cap_check_vs_all_windows", judge=jname, premise=args.premise, topk=args.topk,
                            retriever=retr, tau=tau, n=len(both), n_reference_texts=int(both.hadm_id.nunique()),
                            max_windows=int(both.wins.max()),
                            flag_agreement=round(float(((pa < tau) == (pe < tau)).mean()), 4),
                            n_flag_changed=int(((pa < tau) != (pe < tau)).sum()),
                            n_newly_flagged=int(((pa < tau) & ~(pe < tau)).sum()),   # a subset of the windows can only lower the max
                            n_unflagged=int((~(pa < tau) & (pe < tau)).sum()),
                            median_abs_dp=round(float(np.median(np.abs(pa - pe))), 4),
                            kappa=ma["kappa"], sensitivity=ma["sensitivity"], specificity=ma["specificity"],
                            auroc=round(auroc(yb, -pa), 4),
                            kappa_all_windows=me["kappa"], sensitivity_all_windows=me["sensitivity"],
                            specificity_all_windows=me["specificity"], auroc_all_windows=round(auroc(yb, -pe), 4)))
    res = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    out_name = f"verifact_judge_eval{'_' + args.tag if args.tag else ''}.csv"
    res.to_csv(RESULTS / out_name, index=False)
    print(f"-> results/{out_name}")
    print(res.to_string(index=False))
    f = RESULTS / "verifact_ai_raters.csv"
    if f.exists():
        ai = pd.read_csv(f)
        if "reference_only_admission" in ai.columns:   # compare with the raters that saw the same reference
            ai = ai[ai.reference_only_admission.astype(str).str.lower() == str(args.premise == "admission").lower()]
        if len(ai):
            best = ai.iloc[0]
            print(f"context: best VeriFact AI-rater configuration with the same reference reaches kappa {best.kappa:.3f} "
                  f"(sens {best.sensitivity:.3f}, spec {best.specificity:.3f}) on the same propositions")


def retriever_check(args):
    """Agreement with the clinicians for every scoring setting present in the score file (each ranking, and every window),
    on the propositions all of them scored — the shortest admissions — written to results/verifact_retriever_check.csv.
    This is how the ranking for the full run was chosen."""
    jname, tau_sel = args.judge or selected_judge()[0], selected_judge()[1]
    tau = args.tau if args.tau is not None else tau_sel
    by = {}
    for r in (json.loads(l) for l in open(SCORES)):
        if r["judge"] == jname and r.get("premise", "patient") == args.premise:
            name = f"top{r['topk']}_{r.get('retriever', 'dense')}" if r.get("topk", 0) else "all_windows"
            by.setdefault(name, {})[r["pid"]] = r["p"]
    if not by:
        print("no scores for this judge and premise"); return
    common = set.intersection(*(set(d) for d in by.values()))
    full = load_propositions(args.root)
    df = full[full.proposition_id.isin(common)]
    y = (df.human_gt != "Supported").to_numpy()
    out = []
    for name, d in sorted(by.items()):
        p = df.proposition_id.map(d).to_numpy()
        row = dict(setting=name, judge=jname, premise=args.premise, tau=tau, n_reference_texts=int(df.hadm_id.nunique()),
                   auroc=round(auroc(y, -p), 4))
        row.update(binary_metrics(y, p < tau))
        out.append(row)
    res = pd.DataFrame(out)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "verifact_retriever_check.csv", index=False)
    print(f"{len(df)} propositions scored under every setting -> results/verifact_retriever_check.csv")
    print(res.to_string(index=False))


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
    ap.add_argument("--compare-retrievers", action="store_true",
                    help="agreement of every scoring setting in the score file on the propositions all of them scored")
    ap.add_argument("--judge", default=None); ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--premise", choices=["patient", "admission"], default="patient",
                    help="reference text: all of the patient's notes, or only the summarized admission's notes")
    ap.add_argument("--topk", type=int, default=5,
                    help="score each proposition against the K best-ranked windows only (0 = every window)")
    ap.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid",
                    help="how the K windows are ranked: the judge-selection bi-encoder, BM25, or both interleaved")
    ap.add_argument("--tag", default=None, help="--summarize writes results/verifact_judge_eval_TAG.csv instead of the main file")
    ap.add_argument("--limit-patients", type=int, default=None); ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16,
                    help="judge inference batch size; 8 halves the Metal working pool (about 10 GB at 16) without changing scores")
    args = ap.parse_args()
    if args.baseline:
        baseline(args)
    if args.score or args.plan:
        score(args)
    if args.summarize:
        summarize(args)
    if args.compare_retrievers:
        retriever_check(args)
    if not (args.baseline or args.score or args.plan or args.summarize or args.compare_retrievers):
        print("nothing to do: pass --plan, --score, --summarize, --compare-retrievers or --baseline")


if __name__ == "__main__":
    main()
