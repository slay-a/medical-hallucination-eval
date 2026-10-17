#!/usr/bin/env python3
"""
data_overlap.py — Independence check between the judge-selection data and the main study.

Maps every ann-pt-summ source document (the 100 doctor-written and 10 validation hospital courses, which are also the 110
main-study documents, and the 20 hospital courses behind the 100 LLM-generated summaries) to MIMIC-IV-Note discharge notes by
exact matching of normalized text snippets, and reports document-, admission- and patient-level overlap.

Runs locally on credentialed data.  Writes only counts to results/data_overlap.csv; the per-document mapping (note, admission
and patient identifiers) goes to results_private/ (git-ignored).  No text is printed.
"""
import gzip
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_annptsumm import DEFAULT_DATA, read_jsonl  # noqa: E402

NOTE = Path.home() / "Desktop" / "Thesis" / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2" / "note" / "discharge.csv.gz"
RESULTS, PRIVATE = HERE / "results", HERE / "results_private"
_SNIPS = None


def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def snippets(t, k=3, length=60):
    t = norm(t)
    if len(t) < length + 10:
        return [t]
    step = max(1, (len(t) - length) // (k + 1))
    return [t[step * (i + 1): step * (i + 1) + length] for i in range(k)]


def _init(snips):
    global _SNIPS
    _SNIPS = snips


def _scan(chunk):
    out = []
    for note_id, subject_id, hadm_id, text in chunk:
        nt = norm(text)
        for doc, sn in _SNIPS:
            if all(s in nt for s in sn):
                out.append((doc, note_id, subject_id, hadm_id))
    return out


def main():
    data = Path(DEFAULT_DATA)
    docs = {}
    for i, r in enumerate(read_jsonl(data / "hallucinations_mimic_di.jsonl")):
        docs[f"doc_{i}"] = r["text"]
    for i, r in enumerate(read_jsonl(data / "hallucinations_mimic_di_validation.jsonl")):
        docs[f"val_{i}"] = r["text"]
    gen_ctx = {}
    for i, r in enumerate(read_jsonl(data / "hallucinations_generated_di.jsonl")):
        gen_ctx.setdefault(norm(r["text"]), f"gen_ctx_{len(gen_ctx)}")
    gen_docs = {v: k for k, v in gen_ctx.items()}
    main_norm = {norm(t): d for d, t in docs.items()}
    rows = [dict(check="main-study documents (doctor-written + validation courses)", value=len(docs)),
            dict(check="distinct hospital courses behind the 100 LLM-generated summaries", value=len(gen_docs)),
            dict(check="generated-summary courses identical to a main-study course", value=sum(1 for t in gen_ctx if t in main_norm)),
            dict(check="duplicate texts among the main-study courses", value=len(docs) - len(main_norm))]
    targets = {**{d: t for d, t in docs.items()}, **{g: t for g, t in gen_docs.items()}}
    snips = [(d, snippets(t)) for d, t in targets.items()]
    found = defaultdict(set)
    chunk, chunks = [], []
    with gzip.open(NOTE, "rt") as fh:
        for part in pd.read_csv(fh, usecols=["note_id", "subject_id", "hadm_id", "text"], chunksize=20000):
            chunks.append(list(part.itertuples(index=False, name=None)))
    with Pool(10, initializer=_init, initargs=(snips,)) as pool:
        for res in pool.imap_unordered(_scan, chunks):
            for doc, note_id, subject_id, hadm_id in res:
                found[doc].add((note_id, int(subject_id), int(hadm_id)))
    mapping = []
    for d in targets:
        hits = sorted(found.get(d, set()))
        mapping.append(dict(doc=d, set=("generated_ctx" if d.startswith("gen_ctx") else "main"), n_matches=len(hits),
                            note_ids=";".join(h[0] for h in hits), subject_ids=";".join(str(h[1]) for h in hits), hadm_ids=";".join(str(h[2]) for h in hits)))
    mp = pd.DataFrame(mapping); PRIVATE.mkdir(exist_ok=True); mp.to_csv(PRIVATE / "data_overlap_mapping.csv", index=False)
    main_m = mp[mp.set == "main"]; gen_m = mp[mp.set == "generated_ctx"]
    uniq = lambda df, col: [x for x in df[df.n_matches == 1][col]]
    main_subj = uniq(main_m, "subject_ids"); gen_subj = uniq(gen_m, "subject_ids")
    main_hadm = uniq(main_m, "hadm_ids"); gen_hadm = uniq(gen_m, "hadm_ids")
    rows += [dict(check="main-study courses matched to exactly one discharge note", value=int((main_m.n_matches == 1).sum())),
             dict(check="main-study courses matched to several notes", value=int((main_m.n_matches > 1).sum())),
             dict(check="main-study courses not matched", value=int((main_m.n_matches == 0).sum())),
             dict(check="generated-summary courses matched to exactly one discharge note", value=int((gen_m.n_matches == 1).sum())),
             dict(check="distinct patients among matched main-study courses", value=len(set(main_subj))),
             dict(check="distinct admissions among matched main-study courses", value=len(set(main_hadm))),
             dict(check="main-study patients with more than one course", value=sum(1 for s in set(main_subj) if main_subj.count(s) > 1)),
             dict(check="generated-summary courses whose patient also appears in the main study", value=sum(1 for s in gen_subj if s in set(main_subj))),
             dict(check="generated-summary courses whose admission also appears in the main study", value=sum(1 for h in gen_hadm if h in set(main_hadm)))]
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "data_overlap.csv", index=False)
    pd.set_option("display.width", 200); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
