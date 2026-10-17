#!/usr/bin/env python3
"""
atomic_claims.py — Atomic-fact decomposition of claims, scored by the selected judge (local models only).

  decompose --set expert   split each of the 1,781 expert-annotated summary sentences into atomic facts (local Qwen2.5-7B)
  decompose --set main     the same for every base-condition claim of the main study (E0, E1, E1b, E2, E3, REF)
  score                    support probability of every atomic fact against its hospital course (DeBERTa-v3-large NLI, whole
                           course in 400-token windows, maximum over windows)
  analyze                  sentence-level agreement with the experts when a sentence is scored by its weakest atomic fact,
                           and judge-estimated atomic-level rates per main-study condition with paired tests
  example                  public illustration on MTSamples: the longest E0 claim of each of the first three pilot documents
                           is decomposed and every fact is scored against its note -> results/atomic_example.csv

Decompositions and per-fact scores contain MIMIC-derived text and stay in results_private/ (git-ignored); only aggregates
are written to results/atomic_*.csv.  Nothing is printed except counts and statistics.
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
PRIVATE, RESULTS = HERE / "results_private", HERE / "results"
DEC = {"expert": PRIVATE / "atomic_expert.jsonl", "main": PRIVATE / "atomic_main.jsonl"}
SCORES = PRIVATE / "atomic_fact_scores.csv"
MAIN_CONDS = ["E0", "E1", "E1b", "E2", "E3", "REF"]

SYSTEM = ("You split sentences from a patient-facing medical summary into atomic facts. An atomic fact is a short, "
          "self-contained statement that carries exactly one piece of information: one diagnosis, one symptom, one test or "
          "result, one treatment or procedure, one medication with its dose or change, one instruction, or one time or place. "
          "Copy the wording of the sentence as closely as possible. Do not add, infer, explain or correct anything. Replace "
          "pronouns with what they refer to so that each fact can be read on its own. A sentence that carries a single piece "
          "of information is returned unchanged as one fact. Answer with JSON only.")
USER = ("Split each numbered sentence into atomic facts.\n\n{sentences}\n\nReturn a JSON object whose keys are the sentence "
        "numbers as strings and whose values are lists of atomic facts, for example {{\"1\": [\"fact\", \"fact\"], \"2\": [\"fact\"]}}.")


def parse(reply, n):
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for i in range(1, n + 1):
        v = obj.get(str(i))
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            v = [str(x).strip() for x in v if str(x).strip()]
        out[str(i)] = v if v else None
    return out


def units(which):
    """[(key, source_text, [sentences])] in a fixed order."""
    from calibration_annptsumm import DEFAULT_DATA
    if which == "expert":
        s = pd.read_csv(PRIVATE / "judge_candidates_sentences.csv")
        from judge_candidates import load_items
        texts = {it["sid"]: it["text"] for it in load_items(Path(DEFAULT_DATA))}
        return [(sid, texts[sid], g.sentence.astype(str).tolist()) for sid, g in s.groupby("sid", sort=False)]
    from mimic_generate import load_docs
    from mimic_evaluate import split_claims          # the same claim splitter as the main evaluation
    docs = {d["sid"]: d for d in load_docs(Path(DEFAULT_DATA))}
    recs = [dict(sid=s, condition="REF", summary=d["reference"]) for s, d in docs.items()]
    recs += [r for r in map(json.loads, open(PRIVATE / "mimic_generations.jsonl")) if r["variant"] == "base" and r["condition"] in MAIN_CONDS]
    out = []
    for r in recs:
        claims = [c for c, _, ab in split_claims(r["summary"]) if not ab]
        if claims:
            out.append((f"{r['sid']}|{r['condition']}", docs[r["sid"]]["text"], claims))
    return out


def decompose(which, base_url, model):
    from mimic_generate import LocalLLM
    llm = LocalLLM(base_url, model, temperature=0.0, max_tokens=2000)
    done = {json.loads(l)["key"] for l in open(DEC[which])} if DEC[which].exists() else set()
    todo = [u for u in units(which) if u[0] not in done]
    n_fallback = n_sent = 0
    for k, (key, _, sents) in enumerate(todo):
        facts = None
        for attempt in range(2):
            reply = llm.chat(SYSTEM, USER.format(sentences="\n".join(f"{i + 1}. {t}" for i, t in enumerate(sents))))
            facts = parse(reply, len(sents))
            if facts is not None:
                break
        if facts is None:
            facts = {str(i + 1): None for i in range(len(sents))}
        fb = [i for i in range(len(sents)) if facts[str(i + 1)] is None]
        for i in fb:
            facts[str(i + 1)] = [sents[i]]          # fall back to the sentence itself
        n_fallback += len(fb); n_sent += len(sents)
        with open(DEC[which], "a") as fh:
            fh.write(json.dumps(dict(key=key, facts=[facts[str(i + 1)] for i in range(len(sents))], fallback=fb)) + "\n")
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(todo)} units; sentences falling back to themselves so far: {n_fallback} of {n_sent}", flush=True)
    print(f"done: {len(todo)} units decomposed; fallbacks {n_fallback} of {n_sent} sentences")


def score():
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    J = Judge("deberta_large_nli", CANDIDATES["deberta_large_nli"])
    rows = []
    for which in ("expert", "main"):
        if not DEC[which].exists():
            continue
        src = {k: (t, s) for k, t, s in units(which)}
        for l in open(DEC[which]):
            r = json.loads(l); text, sents = src[r["key"]]
            wins = J.windows(he.sentencize(text))
            prem, hyp, own = [], [], []
            for i, facts in enumerate(r["facts"]):
                for j, f in enumerate(facts):
                    for w in wins:
                        prem.append(w); hyp.append(f); own.append((i, j))
            if not prem:
                continue
            ps, pc = J.support_scores(prem, hyp)
            best = {}
            for (i, j), a, c in zip(own, ps, pc):
                b = best.get((i, j), (0.0, 0.0)); best[(i, j)] = (max(b[0], a), max(b[1], c))
            for (i, j), (a, c) in best.items():
                rows.append(dict(set=which, key=r["key"], sent_idx=i, fact_idx=j, n_facts=len(r["facts"][i]), fallback=i in r["fallback"],
                                 p_support=round(float(a), 4), p_contra=round(float(c), 4)))
        print(f"{which}: {sum(1 for x in rows if x['set'] == which)} atomic facts scored", flush=True)
    pd.DataFrame(rows).to_csv(SCORES, index=False); print("done")


def analyze():
    from judge_candidates import evaluate
    from recompute_metrics import paired_stats
    sc = pd.read_csv(SCORES)
    out = []
    # ---- expert set: sentence scored by its weakest atomic fact
    s = pd.read_csv(PRIVATE / "judge_candidates_sentences.csv")
    s["sent_idx"] = s.groupby("sid", sort=False).cumcount()
    ex = sc[sc.set == "expert"].groupby(["key", "sent_idx"]).agg(min_sup=("p_support", "min"), n_facts=("n_facts", "first"),
                                                                 any_fallback=("fallback", "max")).reset_index()
    m = s.merge(ex, left_on=["sid", "sent_idx"], right_on=["key", "sent_idx"], how="inner")
    for r in evaluate(m, "min_sup"):
        out.append(dict(analysis="expert_agreement_atomic_min", **r))
    if "deberta_large_nli__doc_sup" in m.columns:
        for r in evaluate(m, "deberta_large_nli__doc_sup"):
            out.append(dict(analysis="expert_agreement_sentence", **r))
    facts = dict(analysis="expert_facts_per_sentence", mean=float(m.n_facts.mean()), share_multi=float((m.n_facts > 1).mean()),
                 share_fallback=float(m.any_fallback.mean()), n=len(m))
    # ---- main study: judge-estimated atomic rates per condition
    tau = float(pd.DataFrame([r for r in out if r["analysis"] == "expert_agreement_atomic_min" and r["group"] == "all"]).tau_from_other_subset.iloc[0])
    mm = sc[sc.set == "main"].copy()
    mm[["sid", "condition"]] = mm.key.str.split("|", expand=True)
    mm["unsup"] = mm.p_support < tau
    per_sent = mm.groupby(["sid", "condition", "sent_idx"]).agg(any_unsup=("unsup", "max"), all_unsup=("unsup", "min"), n_facts=("n_facts", "first")).reset_index()
    sent_sum = per_sent.groupby(["sid", "condition"]).any_unsup.mean().rename("sent_any_ufr")
    per_sum = mm.groupby(["sid", "condition"]).agg(fact_ufr=("unsup", "mean"), n_facts=("unsup", "size")).join(sent_sum).reset_index()
    mixed = per_sent.assign(mixed=per_sent.any_unsup & ~per_sent.all_unsup)
    rows = []
    for c in MAIN_CONDS:
        d = per_sum[per_sum.condition == c]; ms = mixed[mixed.condition == c]
        rows.append(dict(analysis="main_condition", condition=c, n_docs=d.sid.nunique(), fact_ufr=float(d.fact_ufr.mean()),
                         sentence_any_fact_ufr=float(d.sent_any_ufr.mean()), facts_per_claim=float(ms.n_facts.mean()),
                         share_claims_mixed=float(ms.mixed.mean()), tau=tau))
    tests = []
    for metric in ("fact_ufr", "sent_any_ufr"):
        w = per_sum.pivot(index="sid", columns="condition", values=metric)
        for a, b in [("E0", c) for c in ("E1", "E1b", "E2", "E3", "REF")] + [("E1", "E3"), ("E1", "E1b")]:
            if a in w and b in w:
                tests.append(dict(analysis="main_test", metric=metric, comparison=f"{b} vs {a}", **paired_stats(w[a], w[b])))
    res = pd.concat([pd.DataFrame(out), pd.DataFrame([facts]), pd.DataFrame(rows), pd.DataFrame(tests)], ignore_index=True)
    res.to_csv(RESULTS / "atomic_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(out)[["analysis", "group", "auroc", "kappa", "precision", "recall", "flag_rate", "tau_from_other_subset"]].round(3).to_string(index=False))
    print(facts); print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(tests)[["metric", "comparison", "mean_a", "mean_b", "delta_mean", "p"]].round(4).to_string(index=False))


def example(base_url, model, n=3):
    """Public data only (pilot E0 claims written by GPT-4o-mini from MTSamples notes)."""
    import hallucination_eval as he
    from judge_candidates import CANDIDATES, Judge
    from mimic_generate import LocalLLM
    docs = he.load_samples(str(HERE.parent / "mtsamples.csv"))
    c = pd.read_csv(RESULTS / "claims_all.csv")
    c = c[(c.condition == "E0") & ~c.is_header.astype(bool) & ~c.is_abstention.fillna(False).astype(bool)].copy()
    c["words"] = c.claim.str.split().str.len()
    picks = c.sort_values(["doc_id", "words"], ascending=[True, False]).drop_duplicates("doc_id").head(n)
    llm = LocalLLM(base_url, model, temperature=0.0, max_tokens=800)
    J = Judge("deberta_large_nli", CANDIDATES["deberta_large_nli"]); tau = 0.20
    rows = []
    for r in picks.itertuples():
        facts = (parse(llm.chat(SYSTEM, USER.format(sentences=f"1. {r.claim}")), 1) or {}).get("1") or [r.claim]
        wins = J.windows(he.sentencize(docs.loc[r.doc_id, "transcription"]))
        p_sent = float(np.max(J.support_scores(wins, [r.claim] * len(wins))[0]))
        for f in facts:
            p = float(np.max(J.support_scores(wins, [f] * len(wins))[0]))
            rows.append(dict(doc_id=int(r.doc_id), sentence=r.claim, sentence_support=round(p_sent, 3), fact=f, fact_support=round(p, 3), tau=tau))
    pd.DataFrame(rows).to_csv(RESULTS / "atomic_example.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decompose", "score", "analyze", "example"]); ap.add_argument("--set", choices=["expert", "main"], default="expert")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1"); ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    a = ap.parse_args()
    if a.cmd == "decompose":
        decompose(a.set, a.base_url, a.model)
    elif a.cmd == "score":
        score()
    elif a.cmd == "example":
        example(a.base_url, a.model)
    else:
        analyze()


if __name__ == "__main__":
    main()
