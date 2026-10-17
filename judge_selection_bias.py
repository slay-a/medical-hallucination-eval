#!/usr/bin/env python3
"""
judge_selection_bias.py — How optimistic is the agreement of the selected judge, and does it hold on independent documents?

Uses the per-sentence support scores of every candidate judge and evidence mode (results_private/judge_candidates_sentences.csv)
and the expert labels of the 1,781 ann-pt-summ sentences.  The unit of independence is the hospital course: each doctor-written
summary has its own course (these 110 courses are also the main-study documents), and the 100 LLM-generated summaries come from
20 further courses (five systems each) whose patients do not appear in the main study (results/data_overlap.csv).

  1. Nested cross-validation of the whole selection procedure (candidate, evidence mode and threshold chosen by maximum kappa on
     the training folds), five folds by course, repeated 20 times: an estimate of the agreement to expect from the procedure on
     unseen courses, compared with the apparent (in-sample) agreement of the selected judge.
  2. Patient-disjoint transfer: selection on the doctor-written sentences only (the main-study courses), evaluation on the
     LLM-generated sentences of the 20 other courses, and the reverse.
  3. Course-clustered bootstrap intervals for the selected judge at its threshold, overall and per summary source.
Writes results/judge_selection_bias.csv, results/judge_selection_frequency.csv, results/judge_per_source.csv (aggregates only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PRIVATE, RESULTS = HERE / "results_private", HERE / "results"
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
SELECTED = ("deberta_large_nli", "doc", 0.20)


def metrics(e, s, tau):
    f = s < tau
    tp = int((f & e).sum()); fp = int((f & ~e).sum()); fn = int((~f & e).sum())
    k = cohen_kappa_score(e, f) if len(set(f)) > 1 and len(set(e)) > 1 else 0.0
    auc = roc_auc_score(e, 1 - s) if len(set(e)) > 1 else np.nan
    return dict(kappa=k, auroc=auc, precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / (tp + fn) if tp + fn else np.nan, flag_rate=float(f.mean()))


def select(df, cols):
    best = (-9, None, None)
    e = df.expert_flag.to_numpy(bool)
    for c in cols:
        s = df[c].to_numpy(float)
        for t in TAUS:
            f = s < t
            if len(set(f)) < 2:
                continue
            k = cohen_kappa_score(e, f)
            if k > best[0]:
                best = (k, c, t)
    return best


def main():
    from judge_candidates import load_items
    from calibration_annptsumm import DEFAULT_DATA
    s = pd.read_csv(PRIVATE / "judge_candidates_sentences.csv")
    cols = [c for c in s.columns if c.endswith("_sup")]
    s = s.dropna(subset=cols)
    items = load_items(Path(DEFAULT_DATA))
    norm = lambda t: " ".join(str(t).split()).lower()
    course_of_text, course = {}, {}
    for it in items:                       # course identifier = identical hospital-course text
        course[it["sid"]] = course_of_text.setdefault(norm(it["text"]), f"c{len(course_of_text)}")
    s["course"] = s.sid.map(course)
    e_all = s.expert_flag.to_numpy(bool)
    print(f"{len(s)} sentences, {s.course.nunique()} courses, {len(cols)} candidate score columns", flush=True)
    sel_col = f"{SELECTED[0]}__{SELECTED[1]}_sup"
    out = []
    app = metrics(e_all, s[sel_col].to_numpy(float), SELECTED[2])
    out += [dict(analysis="apparent_selected_judge", metric=k, estimate=v) for k, v in app.items()]
    k_ins, c_ins, t_ins = select(s, cols)
    out += [dict(analysis="apparent_best_in_sample", metric="kappa", estimate=k_ins, note=f"{c_ins} at {t_ins}")]
    # ---- 1. nested cross-validation by course
    rng = np.random.default_rng(42); courses = s.course.unique(); freq = {}; reps = []
    for rep in range(20):
        perm = rng.permutation(courses); folds = np.array_split(perm, 5)
        preds = np.zeros(len(s), bool); aucs = []
        for f in folds:
            test = s.course.isin(f).to_numpy(); train = ~test
            k, c, t = select(s[train], cols); freq[c] = freq.get(c, 0) + 1
            sc = s.loc[test, c].to_numpy(float); preds[test] = sc < t
            if len(set(e_all[test])) > 1:
                aucs.append(roc_auc_score(e_all[test], 1 - sc))
        tp = (preds & e_all).sum(); fp = (preds & ~e_all).sum(); fn = (~preds & e_all).sum()
        reps.append(dict(kappa=cohen_kappa_score(e_all, preds), auroc=float(np.mean(aucs)), precision=tp / (tp + fp), recall=tp / (tp + fn), flag_rate=preds.mean()))
    R = pd.DataFrame(reps)
    for m in R.columns:
        out.append(dict(analysis="nested_cv_selection_procedure", metric=m, estimate=float(R[m].mean()), lo=float(R[m].quantile(0.025)), hi=float(R[m].quantile(0.975))))
    pd.DataFrame([dict(candidate=k.replace("_sup", ""), times_selected=v, share=v / 100) for k, v in sorted(freq.items(), key=lambda x: -x[1])]).to_csv(RESULTS / "judge_selection_frequency.csv", index=False)
    # ---- 2. patient-disjoint transfer between the two subsets
    for train_src, test_src in (("doctor", "generated"), ("generated", "doctor")):
        tr, te = s[s.source == train_src], s[s.source == test_src]
        k, c, t = select(tr, cols)
        m = metrics(te.expert_flag.to_numpy(bool), te[c].to_numpy(float), t)
        out += [dict(analysis=f"select_on_{train_src}_test_on_{test_src}", metric=kk, estimate=v, note=f"{c.replace('_sup', '')} at {t}") for kk, v in m.items()]
        out.append(dict(analysis=f"select_on_{train_src}_test_on_{test_src}", metric="training_kappa", estimate=k, note=f"{c.replace('_sup', '')} at {t}"))
    # ---- 3. course-clustered bootstrap for the selected judge
    groups = {g: idx for g, idx in s.groupby("course").indices.items()}
    keys = list(groups); boots = []
    for _ in range(2000):
        pick = rng.choice(keys, size=len(keys), replace=True); idx = np.concatenate([groups[k] for k in pick])
        boots.append(metrics(e_all[idx], s[sel_col].to_numpy(float)[idx], SELECTED[2]))
    B = pd.DataFrame(boots)
    for m in B.columns:
        out.append(dict(analysis="selected_judge_course_bootstrap", metric=m, estimate=app[m], lo=float(B[m].quantile(0.025)), hi=float(B[m].quantile(0.975))))
    pd.DataFrame(out).to_csv(RESULTS / "judge_selection_bias.csv", index=False)
    # ---- per summary source (doctor-written and each LLM system) for the selected judge
    per = []
    for g, d in s.groupby("group"):
        m = metrics(d.expert_flag.to_numpy(bool), d[sel_col].to_numpy(float), SELECTED[2])
        per.append(dict(group=g, n=len(d), expert_rate=float(d.expert_flag.mean()), **m))
    pd.DataFrame(per).to_csv(RESULTS / "judge_per_source.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out).round(3).to_string(index=False)); print(pd.DataFrame(per).round(3).to_string(index=False))
    print(pd.read_csv(RESULTS / "judge_selection_frequency.csv").to_string(index=False))


if __name__ == "__main__":
    main()
