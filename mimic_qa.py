#!/usr/bin/env python3
"""
mimic_qa.py — Document-grounded question answering pilot on MIMIC hospital courses (proposal Task B).

Three fixed questions per hospital course (medications, follow-up, warning signs) are answered by the LOCAL model
under two conditions, with an explicit abstention instruction:
  QA-E0  full hospital course + question
  QA-E1  top-3 retrieved chunks (query = the question) + question
Evaluation (subcommand `evaluate`, needs a judge from judge_candidates):
  abstention rate; for answered questions: claim-level UFR/CR against the hospital course; and the share of answer
  sentences supported by the doctor-written discharge instructions (agreement with what the clinician told the patient).

Usage: python mimic_qa.py generate [--max-docs N]      (server must be running, see mimic_generate.py)
       python mimic_qa.py evaluate [--judge NAME] [--mode top3|doc] [--tau T]
Outputs: results_private/mimic_qa.jsonl (answers; git-ignored), results/mimic_qa_summary.csv (aggregates)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from preprocessing import is_abstention, is_markdown_header  # noqa: E402
from mimic_generate import LocalLLM, load_docs, retrieve_chunks  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA  # noqa: E402

PRIVATE, RESULTS = HERE / "results_private", HERE / "results"
OUT = PRIVATE / "mimic_qa.jsonl"
QUESTIONS = {
    "medications": "Which medications should the patient take after leaving the hospital, and were any medications started, stopped or changed?",
    "follow_up": "What follow-up appointments, visits or tests should the patient have after leaving the hospital?",
    "warning_signs": "Which symptoms or warning signs should make the patient seek medical care after leaving the hospital?",
}
SYSTEM = ("You are a medical scribe assistant answering a patient's question using ONLY the provided hospital course text. "
          "Answer in plain language in at most three sentences. If the text does not contain the answer, reply exactly: "
          "Not stated in the note.")
USER_FULL = "=== HOSPITAL COURSE ===\n{text}\n=== END ===\n\nQuestion: {q}\nAnswer:"
USER_EXCERPTS = "=== RETRIEVED EXCERPTS ===\n{context}\n=== END ===\n\nQuestion: {q}\nAnswer:"


def generate(args):
    docs = load_docs(Path(args.data_dir))[: args.max_docs] if args.max_docs else load_docs(Path(args.data_dir))
    PRIVATE.mkdir(exist_ok=True)
    done = {(r["sid"], r["question"], r["condition"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    he.get_nlp(); he.get_bi_encoder(); llm = LocalLLM(args.base_url, args.model, max_tokens=200)
    n = 0
    for d in docs:
        for qk, q in QUESTIONS.items():
            if (d["sid"], qk, "QA-E0") not in done:
                a = llm.chat(SYSTEM, USER_FULL.format(text=d["text"][:6000], q=q)); n += 1
                open(OUT, "a").write(json.dumps(dict(sid=d["sid"], question=qk, condition="QA-E0", answer=a, excerpts=[])) + "\n")
            if (d["sid"], qk, "QA-E1") not in done:
                sents = he.sentencize(d["text"]); chunks = [" ".join(sents[i:i + 5]) for i in range(0, len(sents), 5)] or [d["text"][:3500]]
                top = [c for c, _ in he.retrieve_top_k(q, chunks, k=3)]
                ctx = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(top))[:4000]
                a = llm.chat(SYSTEM, USER_EXCERPTS.format(context=ctx, q=q)); n += 1
                open(OUT, "a").write(json.dumps(dict(sid=d["sid"], question=qk, condition="QA-E1", answer=a, excerpts=top)) + "\n")
        print(f"  {d['sid']} done ({n} calls)", flush=True)
    print(f"done: {n} calls")


def evaluate(args):
    from judge_candidates import CANDIDATES, Judge
    from mimic_evaluate import best_judge
    jname, jmode = best_judge(); jname = args.judge or jname; jmode = args.mode or jmode
    jc = pd.read_csv(RESULTS / "judge_candidates.csv")
    row = jc[(jc.judge == jname) & (jc.mode == jmode) & (jc.group == "all")]
    tau = args.tau or (float(row.tau_from_other_subset.iloc[0]) if len(row) else 0.5)
    print(f"judge={jname} mode={jmode} tau={tau}", flush=True)
    J = Judge(jname, CANDIDATES[jname]); he.get_nlp(); enc = he.get_bi_encoder()
    docs = {d["sid"]: d for d in load_docs(Path(args.data_dir))}
    rows = []
    for r in map(json.loads, open(OUT)):
        d = docs[r["sid"]]; ans = r["answer"].strip()
        abst = is_abstention(ans) or ans.lower().startswith("not stated")
        sents = [s for s in he.sentencize(ans) if not is_markdown_header(s) and not is_abstention(s)]
        rec = dict(sid=r["sid"], question=r["question"], condition=r["condition"], abstained=abst, n_sents=len(sents), words=len(ans.split()))
        if sents and not abst:
            ctx = he.sentencize(d["text"]); S = enc.encode(ctx, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            C = enc.encode(sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            order = np.argsort(-(C @ S.T), axis=1)
            prem = [" ".join(ctx[j] for j in sorted(order[i][:3].tolist())) for i in range(len(sents))]
            ps, pc = J.support_scores(prem, sents)
            rec.update(UFR=float((ps < tau).mean()), CR=float(((pc >= 0.5) & (pc > ps)).mean()))
            pr, _ = J.support_scores([d["reference"]] * len(sents), sents)
            rec.update(ref_support=float((pr >= tau).mean()))
        rows.append(rec)
    df = pd.DataFrame(rows); df.to_csv(PRIVATE / "mimic_qa_per_answer.csv", index=False)
    agg = df.groupby(["condition", "question"]).agg(n=("sid", "size"), abstention_rate=("abstained", "mean"), UFR_mean=("UFR", "mean"), CR_mean=("CR", "mean"),
                                                    ref_support_mean=("ref_support", "mean"), words_mean=("words", "mean")).reset_index()
    tot = df.groupby("condition").agg(n=("sid", "size"), abstention_rate=("abstained", "mean"), UFR_mean=("UFR", "mean"), CR_mean=("CR", "mean"),
                                      ref_support_mean=("ref_support", "mean"), words_mean=("words", "mean")).reset_index().assign(question="all")
    out = pd.concat([agg, tot], ignore_index=True); out["judge"], out["tau"] = jname, tau
    out.to_csv(RESULTS / "mimic_qa_summary.csv", index=False)
    pd.set_option("display.width", 200); print(out.round(3).to_string(index=False)); print("done")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["generate", "evaluate"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1"); ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--max-docs", type=int, default=None); ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--judge", default=None); ap.add_argument("--mode", default=None); ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()
    generate(args) if args.cmd == "generate" else evaluate(args)


if __name__ == "__main__":
    main()
