#!/usr/bin/env python3
"""
mimic_second_judge.py — Score every main-study claim (base variants) with a second, independent judge and measure its
agreement with the selected judge.  Default second judge: Bespoke-MiniCheck-7B through minicheck_server.py, in its
best evidence mode from the judge selection study (top-3 retrieved sentences), at the threshold that study chose.

Reads results_private/mimic_claims.csv (selected judge's labels), writes results_private/mimic_claims_second.csv
(per claim; git-ignored) and results/mimic_judge_agreement.csv (per condition: flag rates, agreement, kappa, share of
the selected judge's flags confirmed by the second judge, and UFR when both or either judge flags a claim).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA, read_jsonl  # noqa: E402
from judge_candidates import CANDIDATES, Judge  # noqa: E402

PRIVATE, RESULTS = HERE / "results_private", HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="bespoke_minicheck_7b"); ap.add_argument("--mode", default=None); ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    jc = pd.read_csv(RESULTS / "judge_candidates.csv"); rows = jc[(jc.judge == args.judge) & (jc.group == "all")].sort_values("kappa", ascending=False)
    mode = args.mode or rows.iloc[0]["mode"]; tau = args.tau if args.tau is not None else float(rows[rows["mode"] == mode].tau_from_other_subset.iloc[0])
    print(f"second judge {args.judge} mode={mode} tau={tau}", flush=True)
    J = Judge(args.judge, CANDIDATES[args.judge]); he.get_nlp(); enc = he.get_bi_encoder()
    docs = {}
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di.jsonl")):
        docs[f"doc_{i}"] = r["text"]
    for i, r in enumerate(read_jsonl(Path(args.data_dir) / "hallucinations_mimic_di_validation.jsonl")):
        docs[f"val_{i}"] = r["text"]
    cl = pd.read_csv(PRIVATE / "mimic_claims.csv"); cl = cl[cl.variant == "base"].reset_index(drop=True)
    print(f"{len(cl)} base claims across {cl.condition.nunique()} conditions", flush=True)
    p2 = np.zeros(len(cl))
    for sid, g in cl.groupby("sid", sort=False):
        ctx = he.sentencize(docs[sid]); S = enc.encode(ctx, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        claims = g.claim.astype(str).tolist()
        if mode == "top3":
            C = enc.encode(claims, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            order = np.argsort(-(C @ S.T), axis=1)
            prem = [" ".join(ctx[j] for j in sorted(order[i][:3].tolist())) for i in range(len(claims))]
        else:
            prem = [" ".join(ctx)] * len(claims)
        ps, _ = J.support_scores(prem, claims); p2[g.index] = ps
    cl["p_support_second"] = p2; cl["flag_first"] = cl.label != "Supported"; cl["flag_second"] = cl.p_support_second < tau
    cl.to_csv(PRIVATE / "mimic_claims_second.csv", index=False)
    out = []
    for cond, g in list(cl.groupby("condition")) + [("all", cl)]:
        a, b = g.flag_first.to_numpy(bool), g.flag_second.to_numpy(bool)
        per = g.assign(both=a & b, either=a | b).groupby("sid").agg(u1=("flag_first", "mean"), u2=("flag_second", "mean"), ub=("both", "mean"), ue=("either", "mean"))
        out.append(dict(condition=cond, n_claims=len(g), flag_rate_first=a.mean(), flag_rate_second=b.mean(), agreement=(a == b).mean(),
                        kappa=cohen_kappa_score(a, b) if a.any() and b.any() and (~a).any() and (~b).any() else np.nan,
                        first_flags_confirmed=(a & b).sum() / a.sum() if a.sum() else np.nan, second_flags_confirmed=(a & b).sum() / b.sum() if b.sum() else np.nan,
                        UFR_first=per.u1.mean(), UFR_second=per.u2.mean(), UFR_both=per.ub.mean(), UFR_either=per.ue.mean(), second_judge=args.judge, second_mode=mode, second_tau=tau))
    res = pd.DataFrame(out); res.to_csv(RESULTS / "mimic_judge_agreement.csv", index=False)
    pd.set_option("display.width", 250); print(res.round(3).to_string(index=False)); print("done")


if __name__ == "__main__":
    main()
