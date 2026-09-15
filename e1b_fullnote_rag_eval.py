#!/usr/bin/env python3
"""
e1b_fullnote_rag_eval.py — E1b: retrieval-augmented generation with the FULL note
plus retrieved excerpts (the variant the April draft described, but which was
never run).  E1 proper shows the model only the excerpts.

Requires OPENAI_API_KEY.  Cost: 50 GPT-4o-mini calls.
Usage:  python e1b_fullnote_rag_eval.py [--dry-run] [--max-docs N]
Afterwards run:  python recompute_metrics.py && python analyze.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402

RESULTS = HERE / "results"
COND = "E1b"

E1B_SYSTEM = (
    "You are a medical scribe assistant. Produce patient-facing summaries that are accurate and "
    "grounded only in the provided clinical note. The retrieved excerpts highlight the most relevant "
    "passages; give them priority, and do not add or invent information."
)
E1B_USER = """\
Write a patient-facing summary (150-250 words) of the clinical note below. The retrieved excerpts \
highlight its most relevant passages; every statement must be supported by the note.
Structure: (1) Reason for visit / diagnosis, (2) Key findings, \
(3) Treatment or procedures performed, (4) Follow-up instructions.
Avoid medical jargon where possible.

=== RETRIEVED EXCERPTS ===
{context}
=== END EXCERPTS ===

=== CLINICAL NOTE ===
{text}
=== END NOTE ===

Patient-Facing Summary:"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build prompts for the first document; no API calls")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    samples = he.load_samples(str((HERE / he.INPUT_CSV).resolve()), n=he.N_SAMPLES)
    if args.max_docs:
        samples = samples.head(args.max_docs)
    he.get_nlp(); he.get_bi_encoder()

    if args.dry_run:
        row = samples.iloc[0]
        ctx = he.build_rag_context(str(row["transcription"]), str(row["description"]))
        print(E1B_SYSTEM, "\n---\n", E1B_USER.format(context=ctx[:4000], text=str(row["transcription"])[:4500]))
        return
    he.get_cross_encoder()

    summ = pd.read_csv(RESULTS / "summaries.csv")
    claims = pd.read_csv(RESULTS / "claims_all.csv")
    claims = claims[claims.condition != COND]
    new_claim_rows, new_summ = [], {}
    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc=COND):
        doc_id = int(idx)
        source = str(row["transcription"]).strip()
        ctx = he.build_rag_context(source, str(row["description"]).strip())
        summary = he.call_gpt(E1B_SYSTEM, E1B_USER.format(context=ctx[:4000], text=source[:4500]))
        recs, _ = he.evaluate_summary(summary, source)
        for c in recs:
            new_claim_rows.append(dict(doc_id=doc_id, condition=COND, specialty=str(row["medical_specialty"]).strip(),
                                       description=str(row["description"]).strip(), **c))
        new_summ[doc_id] = (summary, len(summary.split()))

    # re-read the shared result files right before writing so that a concurrently finished run is not overwritten
    summ = pd.read_csv(RESULTS / "summaries.csv")
    claims = pd.read_csv(RESULTS / "claims_all.csv"); claims = claims[claims.condition != COND]
    summ["e1b_summary"] = summ.doc_id.map(lambda d: new_summ.get(d, ("", 0))[0])
    summ["e1b_summary_words"] = summ.doc_id.map(lambda d: new_summ.get(d, ("", 0))[1])
    summ.to_csv(RESULTS / "summaries.csv", index=False)
    pd.concat([claims, pd.DataFrame(new_claim_rows)], ignore_index=True).to_csv(RESULTS / "claims_all.csv", index=False)
    print(f"{COND}: {len(new_summ)} summaries, {len(new_claim_rows)} claims written. Now run recompute_metrics.py")


if __name__ == "__main__":
    main()
