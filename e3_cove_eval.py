#!/usr/bin/env python3
"""
e3_cove_eval.py — E3: retrieval-augmented generation + Chain-of-Verification (CoVe).

Design (factored CoVe, Dhuliawala et al. 2023, adapted to document grounding):
  1. Draft        : the E1 (RAG) summary already stored in results/summaries.csv.
  2. Verification : every claim of the draft is checked independently against the
                    top-3 source sentences retrieved for THAT claim from the full note
                    (not only the excerpts the draft was written from).  The verifier
                    returns SUPPORTED / NOT SUPPORTED / CONTRADICTED plus a corrected
                    statement or REMOVE.
  3. Revision     : the model rewrites the draft applying every verdict; sections with
                    no supported content read "Not stated in the note."
The revised summary is then scored by the same NLI judge as all other conditions.

Requires OPENAI_API_KEY.  Cost: about 50 x (claims + 1) GPT-4o-mini calls (~500 calls).
Usage:  python e3_cove_eval.py [--dry-run] [--max-docs N]
Afterwards run:  python recompute_metrics.py && python analyze.py
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from preprocessing import is_markdown_header  # noqa: E402

RESULTS = HERE / "results"
COND = "E3"

VERIFY_SYSTEM = "You are a careful clinical fact-checker. Judge statements strictly against the provided note excerpts."
VERIFY_USER = """\
Note excerpts:
{evidence}

Statement to verify:
"{claim}"

Is the statement fully supported by the excerpts? Reply in exactly this format:
VERDICT: SUPPORTED | NOT SUPPORTED | CONTRADICTED
CORRECTION: <a corrected statement that uses only the excerpts, or REMOVE if the excerpts do not address it>"""

REVISE_SYSTEM = ("You are a medical scribe assistant. Revise the draft so that every statement is supported by "
                 "the clinical note. Do not add new information.")
REVISE_USER = """\
Draft summary:
{draft}

Verification results (one per statement):
{verification_block}

Rewrite the summary (150-250 words) applying every correction: delete statements marked REMOVE, replace \
statements with their corrections, and keep supported statements unchanged. Keep the four-part structure \
(reason for visit / diagnosis, key findings, treatment or procedures, follow-up). If a section has no \
supported information, write "Not stated in the note."

Revised Patient-Facing Summary:"""


def call_gpt(system: str, user: str, temperature: float) -> str:
    client = he.get_openai_client()
    for attempt in range(he.GPT_RETRIES):
        try:
            resp = client.chat.completions.create(model=he.OPENAI_MODEL,
                                                  messages=[{"role": "system", "content": system},
                                                            {"role": "user", "content": user}],
                                                  temperature=temperature, max_tokens=he.GPT_MAX_TOKENS)
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            he.log.warning(f"GPT call failed ({attempt+1}/{he.GPT_RETRIES}): {exc}")
            time.sleep(2 ** attempt)
    return ""


def parse_verdict(text: str):
    verdict, correction = "NOT SUPPORTED", "REMOVE"
    for line in text.splitlines():
        u = line.strip().upper()
        if u.startswith("VERDICT:"):
            v = u.split(":", 1)[1].strip()
            verdict = "CONTRADICTED" if "CONTRADICT" in v else ("SUPPORTED" if v.startswith("SUPPORTED") else "NOT SUPPORTED")
        elif u.startswith("CORRECTION:"):
            correction = line.split(":", 1)[1].strip()
    return verdict, correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    samples = he.load_samples(str((HERE / he.INPUT_CSV).resolve()), n=he.N_SAMPLES)
    if args.max_docs:
        samples = samples.head(args.max_docs)
    summ = pd.read_csv(RESULTS / "summaries.csv")
    if "e1_summary" not in summ.columns:
        sys.exit("results/summaries.csv has no e1_summary column; run hallucination_eval.py first")
    drafts = summ.set_index("doc_id")["e1_summary"].to_dict()
    he.get_nlp(); he.get_bi_encoder()

    if args.dry_run:
        row = samples.iloc[0]; source = str(row["transcription"]); draft = str(drafts[0])
        src_sents = he.sentencize(source)
        claims = [c for c in he.sentencize(draft) if not is_markdown_header(c)]
        ev = [e for e, _ in he.retrieve_top_k(claims[0], src_sents, he.TOP_K_EVIDENCE)]
        print(VERIFY_SYSTEM, "\n---\n", VERIFY_USER.format(evidence="\n".join(f"- {e}" for e in ev), claim=claims[0]))
        print("\n=== revision prompt (skeleton) ===\n", REVISE_USER.format(draft=draft, verification_block="<one line per claim>"))
        print(f"\n{len(claims)} claims would be verified for document 0.")
        return
    he.get_cross_encoder()

    claims_all = pd.read_csv(RESULTS / "claims_all.csv")
    claims_all = claims_all[claims_all.condition != COND]
    new_claim_rows, new_summ, verif_rows = [], {}, []
    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc=COND):
        doc_id = int(idx)
        source = str(row["transcription"]).strip()
        draft = str(drafts.get(doc_id, "")).strip()
        src_sents = he.sentencize(source)
        claims = [c for c in he.sentencize(draft) if not is_markdown_header(c)]
        block = []
        for claim in claims:
            ev = [e for e, _ in he.retrieve_top_k(claim, src_sents, he.TOP_K_EVIDENCE)]
            reply = call_gpt(VERIFY_SYSTEM, VERIFY_USER.format(evidence="\n".join(f"- {e}" for e in ev), claim=claim), 0.0)
            verdict, corr = parse_verdict(reply)
            block.append(f'- "{claim}" -> {verdict}; correction: {corr}')
            verif_rows.append(dict(doc_id=doc_id, claim=claim, verdict=verdict, correction=corr, evidence=" | ".join(ev)))
        revised = call_gpt(REVISE_SYSTEM, REVISE_USER.format(draft=draft, verification_block="\n".join(block)), he.GPT_TEMPERATURE)
        recs, _ = he.evaluate_summary(revised, source)
        for c in recs:
            new_claim_rows.append(dict(doc_id=doc_id, condition=COND, specialty=str(row["medical_specialty"]).strip(),
                                       description=str(row["description"]).strip(), **c))
        new_summ[doc_id] = (revised, len(revised.split()))

    summ["e3_summary"] = summ.doc_id.map(lambda d: new_summ.get(d, ("", 0))[0])
    summ["e3_summary_words"] = summ.doc_id.map(lambda d: new_summ.get(d, ("", 0))[1])
    summ.to_csv(RESULTS / "summaries.csv", index=False)
    pd.concat([claims_all, pd.DataFrame(new_claim_rows)], ignore_index=True).to_csv(RESULTS / "claims_all.csv", index=False)
    pd.DataFrame(verif_rows).to_csv(RESULTS / "e3_verification.csv", index=False)
    print(f"{COND}: {len(new_summ)} revised summaries, {len(new_claim_rows)} claims. Now run recompute_metrics.py")


if __name__ == "__main__":
    main()
