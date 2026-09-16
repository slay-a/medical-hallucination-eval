#!/usr/bin/env python3
"""
mimic_generate.py — Generate patient-facing summaries from MIMIC-IV hospital courses with a LOCAL language model.

Documents: the 110 Brief Hospital Course texts of ann-pt-summ (100 + 10 validation), each paired with the
doctor-written Discharge Instructions (the reference for coverage) and with expert unsupported-fact labels.
Generator: any OpenAI-compatible server on localhost, by default mlx_lm.server running Qwen2.5-7B-Instruct
(4-bit) on the laptop's GPU.  No MIMIC text leaves this machine.

Conditions (same designs as the MTSamples study, plus citations):
  E0   full hospital course, no retrieval
  E1   top-k retrieved chunks only, citations required            (variants below)
  E1b  full hospital course + the same excerpts, citations required
  E2   centroid extractive summary, top-5 sentences (no LLM)
  E3   E1 draft -> per-claim verification against top-3 sentences of the full course -> revision
E1 ablation variants (--ablations): chunk3, chunk8, top2, top5, bm25, nocite   (base E1 = chunk5, top3, dense, cite)

Output (git-ignored, contains MIMIC-derived text): results_private/mimic_generations.jsonl, one record per
(sid, condition, variant), resumable.  Aggregate results are produced later by mimic_evaluate.py.

Start the server first, e.g.:
  .venv-llm/bin/python -m mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
Usage: python mimic_generate.py [--conditions E0,E1,E1b,E2,E3] [--ablations] [--max-docs N] [--dry-run]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402  (sentencize, retrieve_top_k, models)
from preprocessing import is_abstention, is_markdown_header  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA, read_jsonl  # noqa: E402
from e3_cove_eval import VERIFY_SYSTEM, VERIFY_USER, REVISE_SYSTEM, format_block_line, parse_verdict  # noqa: E402
from ablations import extractive_topk  # noqa: E402

PRIVATE = HERE / "results_private"
OUT = PRIVATE / "mimic_generations.jsonl"
SECTION_QUERY = "reason for hospital stay, diagnosis, key findings, treatments and procedures, medications, follow-up instructions"

SYSTEM_FULL = ("You are a medical scribe assistant. Produce patient-facing discharge summaries that are accurate and grounded "
               "only in the provided hospital course. Do not add or invent information. If information for a section is not "
               "in the hospital course, write \"Not stated in the note.\" for that section.")
USER_FULL = """\
Write a patient-facing summary (150-250 words) of the hospital course below.
Structure: (1) Why you were in the hospital, (2) Key findings, (3) Treatments or procedures, (4) Medications and follow-up.
Avoid medical jargon where possible. Do not add information that is not in the hospital course.

=== HOSPITAL COURSE ===
{text}
=== END ===

Patient-Facing Summary:"""

SYSTEM_EXCERPTS = ("You are a medical scribe assistant. Use ONLY the supplied excerpts to write the summary. Do NOT add any "
                   "information not present in the excerpts. If information for a section is not in the excerpts, write "
                   "\"Not stated in the note.\" for that section.")
USER_EXCERPTS = """\
Using ONLY the excerpts below, write a patient-facing summary (150-250 words).
Structure: (1) Why you were in the hospital, (2) Key findings, (3) Treatments or procedures, (4) Medications and follow-up.
Avoid medical jargon where possible.{cite_instr}

=== RETRIEVED EXCERPTS ===
{context}
=== END ===

Patient-Facing Summary:"""

SYSTEM_BOTH = ("You are a medical scribe assistant. Produce patient-facing summaries that are accurate and grounded only in the "
               "provided hospital course. The retrieved excerpts highlight the most relevant passages; give them priority, and do "
               "not add or invent information. If information for a section is not in the hospital course, write \"Not stated in "
               "the note.\" for that section.")
USER_BOTH = """\
Write a patient-facing summary (150-250 words) of the hospital course below. The retrieved excerpts highlight its most \
relevant passages; every statement must be supported by the hospital course.
Structure: (1) Why you were in the hospital, (2) Key findings, (3) Treatments or procedures, (4) Medications and follow-up.
Avoid medical jargon where possible.{cite_instr}

=== RETRIEVED EXCERPTS ===
{context}
=== END EXCERPTS ===

=== HOSPITAL COURSE ===
{text}
=== END ===

Patient-Facing Summary:"""

CITE_INSTR = (" After each sentence, cite the excerpt(s) it is based on in square brackets, for example [2] or [1][3]. "
              "Every sentence must carry at least one citation.")

REVISE_USER_LOCAL = """\
Draft summary:
{draft}

Verification results (one per statement):
{verification_block}

Rewrite the summary (150-250 words) applying every correction: delete statements marked REMOVE, replace statements with \
their corrections, and keep supported statements unchanged. Keep the four-part structure (why you were in the hospital, \
key findings, treatments or procedures, medications and follow-up). If a section has no supported information, write \
"Not stated in the note."

Revised Patient-Facing Summary:"""

VARIANTS = {"base": dict(chunk=5, k=3, retriever="dense", cite=True),
            "chunk3": dict(chunk=3, k=3, retriever="dense", cite=True),
            "chunk8": dict(chunk=8, k=3, retriever="dense", cite=True),
            "top2": dict(chunk=5, k=2, retriever="dense", cite=True),
            "top5": dict(chunk=5, k=5, retriever="dense", cite=True),
            "bm25": dict(chunk=5, k=3, retriever="bm25", cite=True),
            "nocite": dict(chunk=5, k=3, retriever="dense", cite=False)}


def load_docs(data_dir: Path):
    docs = []
    for i, r in enumerate(read_jsonl(data_dir / "hallucinations_mimic_di.jsonl")):
        docs.append(dict(sid=f"doc_{i}", text=r["text"], reference=r["summary"]))
    for i, r in enumerate(read_jsonl(data_dir / "hallucinations_mimic_di_validation.jsonl")):
        docs.append(dict(sid=f"val_{i}", text=r["text"], reference=r["summary"]))
    return docs


def chunks_of(sents, size):
    return [" ".join(sents[i:i + size]) for i in range(0, len(sents), size) if sents[i:i + size]]


def retrieve_chunks(text, size, k, retriever):
    sents = he.sentencize(text)
    chunks = chunks_of(sents, size) or [text[:3500]]
    query = f"{SECTION_QUERY}. {text[:300]}"
    if retriever == "bm25":
        tokz = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        bm = BM25Okapi([tokz(c) for c in chunks]); scores = bm.get_scores(tokz(query))
        idx = list(np.argsort(-scores)[:min(k, len(chunks))])
    else:
        idx = [chunks.index(c) for c, _ in he.retrieve_top_k(query, chunks, k=min(k, len(chunks)))]
    top = [chunks[i] for i in idx]
    context = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(top))
    return top, context[:4000]


class LocalLLM:
    def __init__(self, base_url, model, temperature=0.3, max_tokens=450):
        # a single request never legitimately needs more than a few minutes on the local server; a hung server
        # (e.g. after a Metal out-of-memory error) must surface as an exception, not as an empty summary
        self.client = OpenAI(base_url=base_url, api_key="local", timeout=300.0, max_retries=0); self.model = model
        self.temperature, self.max_tokens = temperature, max_tokens

    def chat(self, system, user, temperature=None):
        last = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                        temperature=self.temperature if temperature is None else temperature, max_tokens=self.max_tokens)
                text = (r.choices[0].message.content or "").strip()
                if text:
                    return text
                last = RuntimeError("empty completion")
            except Exception as exc:  # noqa: BLE001
                last = exc
            print(f"  local LLM call failed ({attempt+1}/3): {last}", flush=True); time.sleep(2 ** attempt)
        raise RuntimeError(f"local LLM unavailable after 3 attempts: {last}")


def strip_citations(text):
    return re.sub(r"\s*\[\d+\](\[\d+\])*", "", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--conditions", default="E0,E1,E1b,E2,E3")
    ap.add_argument("--ablations", action="store_true", help="also run the E1 variants chunk3, chunk8, top2, top5, bm25, nocite")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",")]
    docs = load_docs(Path(args.data_dir))
    if args.max_docs:
        docs = docs[:args.max_docs]
    PRIVATE.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l); done.add((r["sid"], r["condition"], r["variant"]))
    he.get_nlp(); he.get_bi_encoder()
    llm = LocalLLM(args.base_url, args.model)
    variants = ["base"] + (list(VARIANTS)[1:] if args.ablations else [])

    def emit(rec):
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done.add((rec["sid"], rec["condition"], rec["variant"]))

    if args.dry_run:
        d = docs[0]; top, ctx = retrieve_chunks(d["text"], 5, 3, "dense")
        print(USER_EXCERPTS.format(cite_instr=CITE_INSTR, context=ctx)[:1500]); return

    t0 = time.time(); n_calls = 0
    for d in tqdm(docs, desc="MIMIC"):
        sid, text = d["sid"], d["text"]
        e1_drafts = {}
        if "E0" in conds and (sid, "E0", "base") not in done:
            s = llm.chat(SYSTEM_FULL, USER_FULL.format(text=text[:6000])); n_calls += 1
            emit(dict(sid=sid, condition="E0", variant="base", summary=s, excerpts=[], model=args.model))
        if "E1" in conds:
            for v in variants:
                if (sid, "E1", v) in done:
                    continue
                cfg = VARIANTS[v]; top, ctx = retrieve_chunks(text, cfg["chunk"], cfg["k"], cfg["retriever"])
                s = llm.chat(SYSTEM_EXCERPTS, USER_EXCERPTS.format(cite_instr=CITE_INSTR if cfg["cite"] else "", context=ctx)); n_calls += 1
                emit(dict(sid=sid, condition="E1", variant=v, summary=s, excerpts=top, model=args.model))
                if v == "base":
                    e1_drafts[sid] = s
        if "E1b" in conds and (sid, "E1b", "base") not in done:
            top, ctx = retrieve_chunks(text, 5, 3, "dense")
            s = llm.chat(SYSTEM_BOTH, USER_BOTH.format(cite_instr=CITE_INSTR, context=ctx, text=text[:6000])); n_calls += 1
            emit(dict(sid=sid, condition="E1b", variant="base", summary=s, excerpts=top, model=args.model))
        if "E2" in conds and (sid, "E2", "base") not in done:
            sents = he.sentencize(text); s = " ".join(extractive_topk(sents, k=5))
            emit(dict(sid=sid, condition="E2", variant="base", summary=s, excerpts=[], model="extractive"))
        if "E3" in conds and (sid, "E3", "base") not in done:
            draft = e1_drafts.get(sid)
            if draft is None:  # reuse the stored E1 base draft
                for l in open(OUT):
                    r = json.loads(l)
                    if r["sid"] == sid and r["condition"] == "E1" and r["variant"] == "base":
                        draft = r["summary"]
            if draft is None:
                continue
            src_sents = he.sentencize(text)
            claims = [c for c in he.sentencize(strip_citations(draft)) if not is_markdown_header(c) and not is_abstention(c)]
            block, verif = [], []
            for claim in claims:
                ev = [e for e, _ in he.retrieve_top_k(claim, src_sents, he.TOP_K_EVIDENCE)]
                reply = llm.chat(VERIFY_SYSTEM, VERIFY_USER.format(evidence="\n".join(f"- {e}" for e in ev), claim=claim), temperature=0.0); n_calls += 1
                verdict, corr = parse_verdict(reply)
                block.append(format_block_line(claim, verdict, corr)); verif.append(dict(claim=claim, verdict=verdict, correction=corr))
            revised = llm.chat(REVISE_SYSTEM, REVISE_USER_LOCAL.format(draft=strip_citations(draft), verification_block="\n".join(block))); n_calls += 1
            emit(dict(sid=sid, condition="E3", variant="base", summary=revised, excerpts=[], model=args.model, verification=verif))
    print(f"done: {n_calls} local LLM calls in {(time.time()-t0)/60:.1f} min; records in {OUT}: {sum(1 for _ in open(OUT))}")


if __name__ == "__main__":
    main()
