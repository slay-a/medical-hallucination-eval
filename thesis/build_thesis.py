#!/usr/bin/env python3
"""
build_thesis.py — Assemble and render the thesis (CSUN format) from the results in ../results.

Usage: python thesis/build_thesis.py [--passes 2] [--only-front]
Outputs: thesis/Ponangi_Thesis_Fall2026.docx and .pdf
"""
import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from render_docx import build           # noqa: E402
from references import resolve_citations  # noqa: E402
from results_loader import Results      # noqa: E402

META = dict(
    title="Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization",
    author="Srilaya Ponangi",
    degree="Master of Science",
    program="Computer Science",
    date="December 2026",
    chair="Taehyung Wang, Ph.D.",
    committee=["Committee Member, Ph.D.", "Committee Member, Ph.D."],   # TODO: replace with the committee members' names
    acknowledgments=[
        "I thank my advisor and committee chair, Dr. Taehyung Wang, for his guidance throughout this project, and the "
        "members of my committee for their time and feedback.",
        "I am grateful to the PhysioNet team and to Hegselmann and colleagues for making MIMIC-IV-Note and the ann-pt-summ "
        "expert annotations available to credentialed researchers; the validation study in Chapter 5 would not have been "
        "possible without them.",
    ],
    abstract=[],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--chapters", default="ch1_4,ch5_7,appendices", help="comma-separated content modules to include")
    ap.add_argument("--out", default="Ponangi_Thesis_Fall2026")
    args = ap.parse_args()

    R = Results()
    blocks = []
    abstract = None
    for mod in args.chapters.split(","):
        mod = mod.strip()
        if not mod:
            continue
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError:
            print(f"  (module {mod} not found, skipped)")
            continue
        blocks += m.blocks(R)
        if hasattr(m, "abstract"):
            abstract = m.abstract(R)
    blocks, refs = resolve_citations(blocks)
    # place the reference list after the last chapter and before the appendices
    idx = next((i for i, b in enumerate(blocks) if b[0] == "h1" and b[1].startswith("Appendix")), len(blocks))
    blocks = blocks[:idx] + [("h1", "Bibliography"), ("refs", refs)] + blocks[idx:]
    meta = dict(META)
    meta["abstract"] = abstract or ["Abstract to be generated from the final results."]
    out_docx, out_pdf = HERE / f"{args.out}.docx", HERE / f"{args.out}.pdf"
    numbers, missing = build(meta, blocks, out_docx, out_pdf, HERE / "_build", passes=args.passes)
    words = sum(len(b[1].split()) for b in blocks if b[0] in ("p", "pni")) + sum(len(x.split()) for b in blocks if b[0] in ("bullets", "numbers") for x in b[1])
    print(f"body words (paragraphs and lists): {words:,}; references: {len(refs)}; missing TOC entries: {missing}")
    print(f"wrote {out_docx.name} and {out_pdf.name}")


if __name__ == "__main__":
    main()
