#!/usr/bin/env python3
"""
audit_sample.py — Draw the manual-audit sample of main-study claims and citations (local, credentialed data).

Design (course-clustered, so that the annotator reads each hospital course once):
  * 60 hospital courses drawn at random (seed 7) from the 110 main-study courses;
  * within them, for each LLM condition E0, E1, E1b and E3, up to 30 claims that the selected judge flagged and 30 that it
    accepted, drawn at random;
  * from the same courses, 30 cited claims of E1 and 30 of E1b, each with the excerpt(s) it cites.
The judge's labels are stored but never shown by the labeling tool (blind annotation).

Output: results_private/audit_items.json (git-ignored; contains MIMIC-derived text).  Prints counts only.
Usage:  python audit_sample.py [--courses 60] [--per-cell 30] [--cite-per-cond 30] [--seed 7]
        python audit_sample.py --refresh-judge   after re-running mimic_evaluate.py: stores the judge's current label of every
                                                 audited claim (judge_flagged_v2), taken from results_private/mimic_claims.csv
                                                 when the claim text matches and otherwise scored directly with the selected
                                                 judge against the whole course.  The sampling stratum is kept.
"""
import argparse
import ast
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mimic_generate import load_docs, DEFAULT_DATA  # noqa: E402
from citations import attributed_claims, strip_headers  # noqa: E402

PRIVATE = HERE / "results_private"


def refresh_judge():
    import os
    import numpy as np
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    f = PRIVATE / "audit_items.json"; items = json.load(open(f))
    summ = pd.read_csv(HERE / "results" / "mimic_summary.csv")
    jname, tau = summ.judge.iloc[0], float(summ.tau.iloc[0])
    norm = lambda x: " ".join(str(x).split())
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[cl.variant == "base"]
    lookup = {(r.sid, r.condition, norm(r.claim)): float(r.p_support) for r in cl.itertuples()}
    J = None; matched = rescored = changed = 0
    for c in items["courses"]:
        todo = []
        for it in c["items"]:
            if it["kind"] != "claim":
                continue
            key = (it["sid"], it["condition"], norm(it["claim"]))
            if key in lookup:
                it["p_support_v2"] = lookup[key]; it["judge_match"] = "pipeline"; matched += 1
            else:
                todo.append(it)
        if todo:
            J = J or Judge(jname, CANDIDATES[jname])
            wins = J.windows(he.sentencize(c["text"]))
            prem = [w for _ in todo for w in wins]; hyp = [it["claim"] for it in todo for _ in wins]
            ps, _ = J.support_scores(prem, hyp); ps = np.asarray(ps).reshape(len(todo), len(wins))
            for it, row in zip(todo, ps):
                it["p_support_v2"] = float(row.max()); it["judge_match"] = "rescored"; rescored += 1
        for it in c["items"]:
            if it["kind"] == "claim":
                it["judge_flagged_v2"] = bool(it["p_support_v2"] < tau)
                changed += int(it["judge_flagged_v2"] != (it["stratum"] == "flagged"))
    tmp = f.with_suffix(".json.tmp"); json.dump(items, open(tmp, "w")); os.replace(tmp, f)
    print(f"judge {jname} at tau {tau}: {matched} claims matched the pipeline, {rescored} scored directly; "
          f"{changed} changed label since sampling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--courses", type=int, default=60); ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--cite-per-cond", type=int, default=30); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--refresh-judge", action="store_true")
    args = ap.parse_args()
    if args.refresh_judge:
        return refresh_judge()
    out_f = PRIVATE / "audit_items.json"
    if out_f.exists() and (PRIVATE / "audit_labels.json").exists():
        sys.exit("labels already exist for the current sample; delete results_private/audit_labels.json first if you really want a new sample")
    rng = random.Random(args.seed)
    docs = {d["sid"]: d for d in load_docs(Path(DEFAULT_DATA))}
    courses = sorted(rng.sample(sorted(docs), args.courses))
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv")
    cl = cl[(cl.variant == "base") & cl.sid.isin(courses)].reset_index(drop=True)
    gens = {}
    for l in open(PRIVATE / "mimic_generations.jsonl"):
        r = json.loads(l)
        if r["variant"] == "base" and r["condition"] in ("E1", "E1b") and r["sid"] in courses:
            gens[(r["sid"], r["condition"])] = (r["summary"], r.get("excerpts") or [])
    items, counts = [], {}
    for cond in ("E0", "E1", "E1b", "E3"):
        d = cl[cl.condition == cond]
        for flagged, sub in ((True, d[d.label != "Supported"]), (False, d[d.label == "Supported"])):
            take = sub.sample(n=min(args.per_cell, len(sub)), random_state=rng.randrange(10 ** 9))
            counts[(cond, "flagged" if flagged else "accepted")] = len(take)
            for _, r in take.iterrows():
                items.append(dict(kind="claim", sid=r.sid, condition=cond, claim=strip_headers(str(r.claim)), judge_label=r.label,
                                  p_support=float(r.p_support), judge_flagged=bool(flagged), stratum="flagged" if flagged else "accepted"))
    for cond in ("E1", "E1b"):        # citations attributed with citations.attributed_claims (markers removed before splitting)
        rows = []
        for sid in courses:
            summary, ex = gens.get((sid, cond), ("", []))
            for claim, cites, ab, _ in attributed_claims(summary):
                cited = [k for k in cites if 1 <= k <= len(ex)]
                if cited and not ab:
                    rows.append((sid, claim, cited, [ex[k - 1] for k in cited]))
        rng.shuffle(rows)
        counts[(cond, "citations")] = min(args.cite_per_cond, len(rows))
        for sid, claim, cited, exs in rows[: args.cite_per_cond]:
            items.append(dict(kind="citation", sid=sid, condition=cond, claim=claim, cited=cited, excerpts=exs, cited_key=",".join(map(str, cited))))
    for i, it in enumerate(items):
        it["item_id"] = f"i{i:04d}"
    rng.shuffle(items)
    by_course = {}
    for it in items:
        by_course.setdefault(it["sid"], []).append(it)
    order = [c for c in courses if c in by_course]
    rng.shuffle(order)
    payload = dict(design=dict(courses=args.courses, per_cell=args.per_cell, cite_per_cond=args.cite_per_cond, seed=args.seed),
                   courses=[dict(sid=c, text=docs[c]["text"], items=sorted(by_course[c], key=lambda x: x["kind"] == "citation")) for c in order])
    PRIVATE.mkdir(exist_ok=True)
    json.dump(payload, open(out_f, "w"))
    n_claims = sum(1 for it in items if it["kind"] == "claim"); n_cit = sum(1 for it in items if it["kind"] == "citation")
    print(f"{len(order)} hospital courses, {n_claims} claims and {n_cit} citation checks written to results_private/audit_items.json")
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:4s} {k[1]:9s} {v}")


if __name__ == "__main__":
    main()
