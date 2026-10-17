"""
citations.py — Attribute citation markers such as [2] or [1][3] to the sentences they belong to.

The generators were told to cite "after each sentence".  Splitting the raw text into sentences directly is unreliable: a
marker written before the period ("...pneumonia [1].") becomes a separate fragment that the sentence filter discards, and a
marker written after the period ("...pneumonia. [1] You were...") is attached to the next sentence.  This module removes the
markers first, splits the clean text with the same spaCy pipeline, and assigns each marker group to the sentence that
contains it or, when it sits at a sentence boundary, to the sentence it follows.
"""
import re

import hallucination_eval as he
from preprocessing import is_abstention, is_markdown_header

MARK = re.compile(r"(?:\s*\[\d+\])+")
NUM = re.compile(r"\[(\d+)\]")


_INLINE_HDR = re.compile(r"^\s*(?:\*\*[^*\n]{1,80}\*\*:?|#{1,6}\s[^\n]{1,80})\s*")


def strip_headers(sentence):
    """Remove section-header lines that the sentence splitter merged into a sentence, and a leading bold or # header."""
    lines = [ln for ln in str(sentence).split("\n") if ln.strip() and not is_markdown_header(ln.strip())]
    text = " ".join(ln.strip() for ln in lines).strip()
    m = _INLINE_HDR.match(text)
    if m and m.end() < len(text):
        text = text[m.end():].strip()
    return text


def attributed_claims(summary):
    """[(claim_text, [excerpt numbers], is_abstention, line_index)] for non-header sentences of at least ten characters."""
    clean, marks, pos = [], [], 0
    for m in MARK.finditer(summary):
        clean.append(summary[pos:m.start()]); cp = sum(len(x) for x in clean)
        marks.append((cp, [int(k) for k in NUM.findall(m.group(0))])); pos = m.end()
    clean.append(summary[pos:]); text = "".join(clean)
    doc = he.get_nlp()(text)
    sents = [(s.start_char, s.end_char, s.text) for s in doc.sents]
    owner = []
    for cp, nums in marks:
        idx = None
        for i, (a, b, t) in enumerate(sents):
            core_end = a + len(t.rstrip())
            if a < cp <= core_end:            # inside the sentence (before or at its final punctuation)
                idx = i; break
        if idx is None:                        # at a boundary: belongs to the last sentence that ends before it
            prev = [i for i, (a, b, t) in enumerate(sents) if a + len(t.rstrip()) <= cp]
            idx = prev[-1] if prev else 0
        owner.append((idx, nums))
    cites = {}
    for i, nums in owner:
        cites.setdefault(i, []).extend(nums)
    out = []
    for i, (a, b, t) in enumerate(sents):
        c = strip_headers(t)
        if not c or is_markdown_header(c) or len(c) < 10:
            continue
        line = text.count("\n", 0, a)
        out.append((c, sorted(set(cites.get(i, []))), is_abstention(c), line))
    return out
