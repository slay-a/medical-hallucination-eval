#!/usr/bin/env python3
"""
mimic_extra.py — Additional main-study generations requested in the advisor's review (October 2026).

  --length-matched   E0 variant "short": full-note summarization without retrieval, asked for about 150 words, the length of
                     the E1 summaries, to test whether brevity alone lowers the judge-estimated unsupported fact rate.
  --nocite-e1b       E1b variant "nocite": RAG with the full note without the citation instruction, to separate the effect of
                     the citation instruction from the effect of the excerpts.
  --replicates N     N repeated generations (variants "rep1".."repN", fixed seeds) of E0, E1, E1b and E3 for a random subset
                     of --rep-docs hospital courses (seed 42), to measure run-to-run variation.

Same local model, prompts and decoding as mimic_generate.py; records are appended to results_private/mimic_generations.jsonl
(git-ignored) and are resumable.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
import mimic_generate as mg  # noqa: E402
from preprocessing import is_abstention, is_markdown_header  # noqa: E402

USER_FULL_SHORT = mg.USER_FULL.replace("Write a patient-facing summary (150-250 words) of the hospital course below.",
                                       "Write a patient-facing summary of about 150 words (no more than 170 words) of the hospital course below.")
assert USER_FULL_SHORT != mg.USER_FULL


def e3_from_draft(llm, text, draft, seed=None):
    src = he.sentencize(text)
    claims = [c for c in he.sentencize(mg.strip_citations(draft)) if not is_markdown_header(c) and not is_abstention(c)]
    block, verif = [], []
    for claim in claims:
        ev = [e for e, _ in he.retrieve_top_k(claim, src, he.TOP_K_EVIDENCE)]
        reply = llm.chat(mg.VERIFY_SYSTEM, mg.VERIFY_USER.format(evidence="\n".join(f"- {e}" for e in ev), claim=claim), temperature=0.0)
        verdict, corr = mg.parse_verdict(reply)
        block.append(mg.format_block_line(claim, verdict, corr)); verif.append(dict(claim=claim, verdict=verdict, correction=corr))
    revised = llm.chat(mg.REVISE_SYSTEM, mg.REVISE_USER_LOCAL.format(draft=mg.strip_citations(draft), verification_block="\n".join(block)), seed=seed)
    return revised, verif, len(claims) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1"); ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--length-matched", action="store_true"); ap.add_argument("--nocite-e1b", action="store_true")
    ap.add_argument("--replicates", type=int, default=0); ap.add_argument("--rep-docs", type=int, default=40)
    ap.add_argument("--data-dir", default=str(mg.DEFAULT_DATA))
    args = ap.parse_args()
    docs = mg.load_docs(Path(args.data_dir))
    done = set()
    if mg.OUT.exists():
        for l in open(mg.OUT):
            r = json.loads(l); done.add((r["sid"], r["condition"], r["variant"]))
    he.get_nlp(); he.get_bi_encoder()
    llm = mg.LocalLLM(args.base_url, args.model)

    def emit(rec):
        with open(mg.OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done.add((rec["sid"], rec["condition"], rec["variant"]))

    t0, n = time.time(), 0
    if args.length_matched or args.nocite_e1b:
        for d in tqdm(docs, desc="variants"):
            sid, text = d["sid"], d["text"]
            if args.length_matched and (sid, "E0", "short") not in done:
                s = llm.chat(mg.SYSTEM_FULL, USER_FULL_SHORT.format(text=text[:6000])); n += 1
                emit(dict(sid=sid, condition="E0", variant="short", summary=s, excerpts=[], model=args.model))
            if args.nocite_e1b and (sid, "E1b", "nocite") not in done:
                top, ctx = mg.retrieve_chunks(text, 5, 3, "dense")
                s = llm.chat(mg.SYSTEM_BOTH, mg.USER_BOTH.format(cite_instr="", context=ctx, text=text[:6000])); n += 1
                emit(dict(sid=sid, condition="E1b", variant="nocite", summary=s, excerpts=top, model=args.model))
    if args.replicates:
        subset = random.Random(42).sample([d["sid"] for d in docs], args.rep_docs)
        bysid = {d["sid"]: d for d in docs}
        for r in range(1, args.replicates + 1):
            v = f"rep{r}"
            for k, sid in enumerate(tqdm(subset, desc=v)):
                text = bysid[sid]["text"]; seed = 1000 * r + k
                if (sid, "E0", v) not in done:
                    s = llm.chat(mg.SYSTEM_FULL, mg.USER_FULL.format(text=text[:6000]), seed=seed); n += 1
                    emit(dict(sid=sid, condition="E0", variant=v, summary=s, excerpts=[], model=args.model, seed=seed))
                top, ctx = mg.retrieve_chunks(text, 5, 3, "dense")
                if (sid, "E1", v) not in done:
                    s = llm.chat(mg.SYSTEM_EXCERPTS, mg.USER_EXCERPTS.format(cite_instr=mg.CITE_INSTR, context=ctx), seed=seed); n += 1
                    emit(dict(sid=sid, condition="E1", variant=v, summary=s, excerpts=top, model=args.model, seed=seed))
                if (sid, "E1b", v) not in done:
                    s = llm.chat(mg.SYSTEM_BOTH, mg.USER_BOTH.format(cite_instr=mg.CITE_INSTR, context=ctx, text=text[:6000]), seed=seed); n += 1
                    emit(dict(sid=sid, condition="E1b", variant=v, summary=s, excerpts=top, model=args.model, seed=seed))
                if (sid, "E3", v) not in done:
                    draft = next(json.loads(l)["summary"] for l in open(mg.OUT)
                                 if (lambda q: q["sid"] == sid and q["condition"] == "E1" and q["variant"] == v)(json.loads(l)))
                    revised, verif, calls = e3_from_draft(llm, text, draft, seed=seed); n += calls
                    emit(dict(sid=sid, condition="E3", variant=v, summary=revised, excerpts=[], model=args.model, seed=seed, verification=verif))
    print(f"done: {n} local LLM calls in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
