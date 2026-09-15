#!/usr/bin/env python3
"""
preprocessing.py — Shared text-preprocessing utilities for the evaluation pipeline.

Two data-quality problems in the original pipeline are handled here.

1. Markdown section headers in generated summaries.
   GPT-4o-mini follows the requested four-part structure by emitting bold
   headers such as "**Key Findings:**".  These lines contain no verifiable
   proposition, but the original pipeline treated them as claims.  The NLI
   cross-encoder frequently labels them "Contradicted", which inflated both the
   claim counts and the contradiction rates, more so for E0 than for E1.
   `is_markdown_header()` identifies such lines so they can be excluded.

2. Line-break artefacts in MTSamples transcriptions.
   In the public MTSamples CSV the original line breaks were replaced by
   commas, producing text such as
       "CHIEF COMPLAINT:,  Chest pain.,HISTORY OF PRESENT ILLNESS:,  The ..."
   `clean_mtsamples_text()` restores the line breaks so that sentence
   segmentation yields well-formed evidence sentences, and
   `sentencize_cleaned()` segments the repaired text while dropping lines that
   consist only of a section heading.
"""
import re
from typing import List

# "**Key Findings:**", "(2) **Key Findings**", "**Reason for Visit / Diagnosis:**"
_BOLD_HEADER_RE = re.compile(r"^\s*(?:\(?\d+\)?[.)]?\s*)?\*\*[^*\n]{1,80}\*\*\s*:?\s*$")
# "### Key Findings"
_HASH_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S.{0,80}$")
# "Follow-Up Instructions:"  (every word capitalised, ends with a colon)
_PLAIN_HEADER_RE = re.compile(r"^\s*(?:\(?\d+\)?[.)]?\s*)?(?:[A-Z][A-Za-z/&\-]*\s*){1,6}:\s*$")
# A line that is only an ALL-CAPS section heading in a clinical note: "PAST MEDICAL HISTORY:"
_SECTION_ONLY_RE = re.compile(r"^[A-Z0-9][A-Z0-9 /&\-()',.]{1,60}:$")


def is_markdown_header(text: str) -> bool:
    """Return True when `text` is only a section header and therefore not a claim."""
    t = (text or "").strip()
    if not t:
        return True
    if _BOLD_HEADER_RE.match(t) or _HASH_HEADER_RE.match(t):
        return True
    if _PLAIN_HEADER_RE.match(t) and len(t.split()) <= 6:
        return True
    return False


_ABSTENTION_RE = re.compile(r"^\s*(?:\*\*[^*]{1,60}\*\*:?\s*)?(?:not stated in the (?:clinical )?note|not stated|not mentioned in the note|no information (?:is )?(?:provided|available) in the note)\.?\s*$", re.I)


def is_abstention(text: str) -> bool:
    """True for an explicit abstention such as "Not stated in the note." (E3 writes these for sections without support).
    Abstentions are not factual claims about the patient and are excluded from UFR and CR."""
    return bool(_ABSTENTION_RE.match((text or "").strip()))


def clean_mtsamples_text(text: str) -> str:
    """Repair the comma-for-line-break artefact of the MTSamples corpus."""
    t = (text or "").replace("\r", "")
    t = re.sub(r"\.,\s*", ".\n", t)                          # sentence end + comma  -> line break
    t = re.sub(r":,\s*", ": ", t)                            # "HEADING:,  text"     -> "HEADING: text"
    t = re.sub(r",\s*([A-Z][A-Z0-9 /&\-()']{2,}:)", r"\n\1", t)  # comma before ALL-CAPS heading
    t = re.sub(r",,+", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def sentencize_cleaned(text: str, nlp, min_len: int = 10) -> List[str]:
    """Sentence-segment repaired MTSamples text, skipping heading-only lines."""
    out: List[str] = []
    for line in clean_mtsamples_text(text).split("\n"):
        line = line.strip()
        if not line or _SECTION_ONLY_RE.match(line):
            continue
        for s in nlp(line).sents:
            st = s.text.strip()
            if len(st) >= min_len:
                out.append(st)
    return out


if __name__ == "__main__":
    tests = ["**Key Findings:**", "(1) **Reason for Visit / Diagnosis:**", "Follow-Up Instructions:",
             "### Medications", "Your blood pressure was elevated.", "**Key Findings:** Your labs were normal."]
    for s in tests:
        print(f"{is_markdown_header(s)!s:5}  {s}")
    print(clean_mtsamples_text("CHIEF COMPLAINT:,  Chest pain.,HISTORY OF PRESENT ILLNESS:,  The patient is a 45-year-old man.,1.  Hypertension.,2.  Diabetes."))
