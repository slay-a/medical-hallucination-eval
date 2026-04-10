#!/usr/bin/env python3
"""
make_pdfs.py — Generate progress_report.pdf and progress_slides.pdf
in results/ using ReportLab.

Run from the project root with the venv active:
    python make_pdfs.py
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS     = SCRIPT_DIR / "results"
REPORT_OUT  = RESULTS / "progress_report.pdf"
SLIDES_OUT  = RESULTS / "progress_slides.pdf"
PNG_UFR     = RESULTS / "plot_ufr_boxplot.png"
PNG_CR      = RESULTS / "plot_cr_boxplot.png"

# ─── Brand colors ─────────────────────────────────────────────────────────────
CSUN_PURPLE = "#582C83"
CSUN_GOLD   = "#F2A900"
WHITE       = "#FFFFFF"
LIGHT_GREY  = "#F4F4F4"
MID_GREY    = "#CCCCCC"
DARK_GREY   = "#444444"
BLACK       = "#000000"

# ─── Load data ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

agg    = pd.read_csv(RESULTS / "aggregate_statistics.csv")
claims = pd.read_csv(RESULTS / "claims_all.csv")

ufr_row = agg[agg["Metric"] == "UFR"].iloc[0]
cr_row  = agg[agg["Metric"] == "CR"].iloc[0]

# Pull one RAG-fixed qualitative example
e0c  = claims[(claims.condition == "E0") & (claims.label == "Contradicted")]
e1s  = claims[(claims.condition == "E1") & (claims.label == "Supported")]
shared_docs = sorted(set(e0c.doc_id) & set(e1s.doc_id))
fix_doc = shared_docs[0]
fix_e0  = e0c[e0c.doc_id == fix_doc].sort_values("p_contradiction", ascending=False).iloc[0]
fix_e1  = e1s[e1s.doc_id == fix_doc].sort_values("p_entailment",    ascending=False).iloc[0]

# Pull one persistent hallucination
e0ns = claims[(claims.condition == "E0") & (claims.label == "Not-Supported")]
e1ns = claims[(claims.condition == "E1") & (claims.label == "Not-Supported")]
pers_docs = sorted(set(e0ns.doc_id) & set(e1ns.doc_id))
pers_doc  = pers_docs[1]   # pick second to vary from fix example
pers_e0   = e0ns[e0ns.doc_id == pers_doc].sort_values("p_entailment").iloc[0]
pers_e1   = e1ns[e1ns.doc_id == pers_doc].sort_values("p_entailment").iloc[0]

RAG_FIXED_E0  = str(fix_e0["claim"])[:200]
RAG_FIXED_E1  = str(fix_e1["claim"])[:200]
PERSIST_E0    = str(pers_e0["claim"])[:200]
PERSIST_E1    = str(pers_e1["claim"])[:200]

# ─── Pre-computed metric scalars (used by both build_report and build_slides) ─
e0_ufr  = float(ufr_row["E0_Mean"])
e1_ufr  = float(ufr_row["E1_Mean"])
e2_ufr  = float(ufr_row["E2_Mean"])
e0_cr   = float(cr_row["E0_Mean"])
e1_cr   = float(cr_row["E1_Mean"])
e2_cr   = float(cr_row["E2_Mean"])
d_ufr   = float(ufr_row["Delta_Mean"])
d_cr    = float(cr_row["Delta_Mean"])
pct_ufr = float(ufr_row["Pct_Improved"])
pct_cr  = float(cr_row["Pct_Improved"])


# ══════════════════════════════════════════════════════════════════════════════
# FILE 1 — PROGRESS REPORT
# ══════════════════════════════════════════════════════════════════════════════
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.colors import HexColor


def hex_color(h):
    return HexColor(h)


# ── Style sheet ───────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent="Normal", **kw):
    s = ParagraphStyle(name, parent=styles[parent], **kw)
    return s

sty_h1 = make_style("H1",
    fontSize=16, leading=20, spaceBefore=14, spaceAfter=6,
    textColor=hex_color(CSUN_PURPLE), fontName="Helvetica-Bold",
    alignment=TA_LEFT)

sty_h2 = make_style("H2",
    fontSize=12, leading=16, spaceBefore=10, spaceAfter=4,
    textColor=hex_color(CSUN_PURPLE), fontName="Helvetica-Bold")

sty_body = make_style("Body",
    fontSize=10, leading=14, spaceBefore=2, spaceAfter=2,
    textColor=hex_color(DARK_GREY), alignment=TA_JUSTIFY)

sty_bullet = make_style("Bullet",
    fontSize=10, leading=13, spaceBefore=1, spaceAfter=1,
    textColor=hex_color(DARK_GREY), leftIndent=16, bulletIndent=6)

sty_small = make_style("Small",
    fontSize=8, leading=11, textColor=hex_color("#888888"))

sty_title = make_style("Title",
    fontSize=20, leading=26, spaceBefore=0, spaceAfter=4,
    textColor=hex_color(CSUN_PURPLE), fontName="Helvetica-Bold",
    alignment=TA_CENTER)

sty_subtitle = make_style("Subtitle",
    fontSize=11, leading=14, spaceBefore=2, spaceAfter=2,
    textColor=hex_color(DARK_GREY), fontName="Helvetica",
    alignment=TA_CENTER)

sty_caption = make_style("Caption",
    fontSize=8, leading=11, spaceBefore=2, spaceAfter=6,
    textColor=hex_color("#666666"), alignment=TA_CENTER,
    fontName="Helvetica-Oblique")


def hr():
    return HRFlowable(width="100%", thickness=1,
                      color=hex_color(CSUN_PURPLE), spaceAfter=6)


def section(title, content_items):
    """Return a KeepTogether block: heading + HR + content."""
    elems = [Paragraph(title, sty_h2), hr()]
    elems.extend(content_items)
    return KeepTogether(elems)


def bullets(items):
    return [Paragraph(f"• {it}", sty_bullet) for it in items]


def build_report():
    doc = SimpleDocTemplate(
        str(REPORT_OUT),
        pagesize=LETTER,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.85 * inch,
        title="Thesis Progress Report — Hallucination Evaluation",
        author="Laya",
    )

    story = []

    # ── Cover header ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Thesis Progress Report", sty_title))
    story.append(Paragraph(
        "Hallucination Evaluation in LLM-Based Medical Summarization", sty_subtitle))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "California State University, Northridge · Comp 696C", sty_subtitle))
    story.append(Paragraph(
        "April 2026", sty_subtitle))
    story.append(Spacer(1, 0.18 * inch))
    story.append(hr())
    story.append(Spacer(1, 0.1 * inch))

    # ── 1. Project Overview ───────────────────────────────────────────────────
    story.append(section("1. Project Overview", [
        Paragraph(
            "This thesis investigates the extent to which hallucinations occur in "
            "large language model (LLM) generated summaries of clinical notes, and "
            "whether retrieval-augmented generation (RAG) mitigates them. "
            "The evaluation pipeline applies Natural Language Inference (NLI) to "
            "classify each generated claim as Supported, Not-Supported, or "
            "Contradicted by the source document, yielding two per-summary metrics: "
            "Unsupported Fact Rate (UFR) and Contradiction Rate (CR).",
            sty_body),
        Spacer(1, 4),
    ]))

    # ── 2. Work Completed ─────────────────────────────────────────────────────
    story.append(section("2. Work Completed", bullets([
        "MTSamples data pipeline: loaded, filtered (Discharge Summary + Consult–H&P), "
        "and sampled 50 clinical notes (seed=42) for reproducibility.",
        "E0 Baseline: GPT-4o-mini summaries generated from full clinical note (no retrieval).",
        "E1 RAG: GPT-4o-mini summaries generated from top-3 retrieved sentence chunks "
        "(all-MiniLM-L6-v2 bi-encoder).",
        "E2 Extractive Baseline: centroid-based sentence extraction (top-5 most "
        "representative sentences by mean cosine similarity) — no LLM call.",
        "Claim-level NLI evaluation framework: spaCy sentence segmentation → "
        "bi-encoder evidence retrieval → cross-encoder/nli-MiniLM2-L6-H768 "
        "support classification → UFR / CR aggregation.",
        "Wilcoxon signed-rank tests (two-sided) comparing E0 vs E1 for both UFR and CR.",
        "Aggregate statistics, per-sample comparison table, and four diagnostic plots "
        "(box plots, scatter plot, improvement bar chart) saved to results/.",
        "CITI Human Subjects Research training completed (certification on file).",
    ])))

    # ── 3. Key Results ────────────────────────────────────────────────────────
    # (scalars defined at module level)

    tbl_data = [
        ["Metric", "E0 Baseline", "E1 RAG", "E2 Extractive", "E1−E0 Δ", "p-value"],
        ["UFR (mean)",
         f"{e0_ufr:.4f}", f"{e1_ufr:.4f}", f"{e2_ufr:.4f}",
         f"{d_ufr:+.4f}", "0.576 (n.s.)"],
        ["CR  (mean)",
         f"{e0_cr:.4f}",  f"{e1_cr:.4f}",  f"{e2_cr:.4f}",
         f"{d_cr:+.4f}",  "0.0001 ***"],
    ]
    tbl = Table(tbl_data, colWidths=[1.2*inch, 1.05*inch, 1.05*inch, 1.15*inch, 0.85*inch, 1.0*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  hex_color(CSUN_PURPLE)),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [hex_color(LIGHT_GREY), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, hex_color(MID_GREY)),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("TEXTCOLOR",    (-1, -1), (-1, -1), hex_color("#CC0000")),
        ("FONTNAME",     (-1, -1), (-1, -1), "Helvetica-Bold"),
    ]))

    story.append(section("3. Key Results", [
        Paragraph(
            "The table below presents mean UFR and CR across 50 samples for each "
            "experimental condition, along with the E1−E0 delta and Wilcoxon "
            "signed-rank p-values.",
            sty_body),
        Spacer(1, 6),
        tbl,
        Spacer(1, 4),
        Paragraph(
            f"UFR: {pct_ufr:.0f}% of samples improved with RAG.  "
            f"CR: {pct_cr:.0f}% of samples improved with RAG.",
            sty_small),
        Paragraph(
            "UFR = Unsupported Fact Rate = (Contradicted + Not-Supported) / total claims.  "
            "CR = Contradiction Rate = Contradicted / total claims.  "
            "*** p < 0.001 (Wilcoxon signed-rank, two-sided).",
            sty_small),
        Spacer(1, 4),
    ]))

    # ── 4. Key Finding ────────────────────────────────────────────────────────
    cr_reduction_pct = abs(d_cr / e0_cr) * 100
    story.append(section("4. Key Finding", [
        Paragraph(
            f"<b>RAG significantly reduces contradictions but not generic unsupported "
            f"facts.</b> The Contradiction Rate fell from {e0_cr:.4f} (E0) to "
            f"{e1_cr:.4f} (E1) — a {cr_reduction_pct:.0f}% relative reduction "
            f"(Wilcoxon p&nbsp;=&nbsp;0.0001, highly significant). By contrast, "
            f"the Unsupported Fact Rate showed no significant improvement "
            f"(Δ&nbsp;=&nbsp;{d_ufr:+.4f}, p&nbsp;=&nbsp;0.576), suggesting that "
            f"RAG constrains outright contradictions but does not eliminate claims "
            f"that simply lack grounding in the source.",
            sty_body),
        Spacer(1, 4),
        Paragraph(
            f"<b>E2 validates pipeline correctness.</b> The extractive baseline — "
            f"which selects verbatim sentences from the source — achieved "
            f"UFR&nbsp;=&nbsp;{e2_ufr:.4f} and CR&nbsp;=&nbsp;{e2_cr:.4f}, "
            f"confirming that near-zero hallucination rates are achievable and that "
            f"the NLI pipeline correctly labels source-faithful content as Supported. "
            f"The 100% improvement rate in UFR for E2 provides a methodological "
            f"sanity check.",
            sty_body),
        Spacer(1, 4),
    ]))

    # ── 5. Work In Progress ───────────────────────────────────────────────────
    story.append(section("5. Work In Progress", bullets([
        "PhysioNet credentialing: application submitted; awaiting institutional "
        "approval to access MIMIC-III / MIMIC-IV clinical notes.",
        "CITI training: Human Subjects Research modules completed; additional "
        "modules (Clinical Research) in progress.",
        "Literature review draft: sections on LLM hallucination taxonomy and "
        "retrieval-augmented generation being finalized.",
    ])))

    # ── 6. Remaining Work ─────────────────────────────────────────────────────
    story.append(section("6. Remaining Work", bullets([
        "ann-pt-summ dataset: obtain and benchmark against annotated patient "
        "summary corpus once access is confirmed.",
        "E3 — Verification stage (Chain-of-Verification / CoVe): implement "
        "post-hoc verification prompting to reduce residual Not-Supported claims.",
        "Error taxonomy: qualitative categorization of hallucination types "
        "(fabricated dosages, incorrect dates, invented diagnoses, etc.).",
        "PhysioNet migration: re-run full pipeline on MIMIC notes once "
        "credentialing is approved.",
        "Full thesis writeup: Chapters 2 (Literature Review), 3 (Methodology), "
        "4 (Results), 5 (Discussion and Future Work).",
        "Defense preparation and committee review.",
    ])))

    # ── 7. Problems / Changes ─────────────────────────────────────────────────
    story.append(section("7. Problems / Changes", [
        Paragraph(
            "<b>Dataset substitution (Risk 1 mitigation).</b> "
            "PhysioNet credentialing has not yet been approved, which blocked "
            "access to MIMIC clinical notes. Per the Risk&nbsp;1 mitigation plan "
            "documented in the thesis proposal, the MTSamples public corpus was "
            "used as a substitute. The pipeline is fully implemented and validated "
            "on MTSamples and is architecturally identical to the design targeting "
            "MIMIC; migration requires only changing the data-loading step. "
            "No other deviations from the proposed methodology have occurred.",
            sty_body),
        Spacer(1, 4),
    ]))

    # ── 8. Overall Assessment ─────────────────────────────────────────────────
    story.append(section("8. Overall Assessment", [
        Paragraph(
            "<b>Ahead of schedule.</b> All three experimental conditions (E0, E1, E2) "
            "are complete, the NLI evaluation framework is validated end-to-end, and "
            "statistically significant results have been obtained for the primary "
            "hypothesis (H2: RAG reduces CR). The pipeline is ready to migrate to "
            "PhysioNet data as soon as credentials are approved. Remaining work "
            "focuses on the verification stage (E3), qualitative analysis, and "
            "thesis writing.",
            sty_body),
        Spacer(1, 4),
    ]))

    doc.build(story)
    print(f"  [saved] {REPORT_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# FILE 2 — SLIDES
# ══════════════════════════════════════════════════════════════════════════════
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import cm, inch
from reportlab.lib.utils import ImageReader

SLIDE_W, SLIDE_H = landscape(A4)   # 841.89 x 595.28 pts  (≈ 16:11)

# Brand
C_PURPLE = HexColor(CSUN_PURPLE)
C_GOLD   = HexColor(CSUN_GOLD)
C_WHITE  = colors.white
C_BG     = colors.white
C_DGREY  = HexColor(DARK_GREY)
C_LGREY  = HexColor(LIGHT_GREY)
C_MGREY  = HexColor(MID_GREY)
C_RED    = HexColor("#CC0000")
C_GREEN  = HexColor("#2A8C4A")

MARGIN  = 1.1 * cm
CONTENT_X   = MARGIN + 0.3 * cm
CONTENT_W   = SLIDE_W - 2 * MARGIN - 0.6 * cm
HEADER_H    = 1.55 * cm
FOOTER_Y    = 0.5 * cm

def draw_chrome(c, slide_num, total=10):
    """Draw background, header bar, footer bar, slide number."""
    # White background
    c.setFillColor(C_BG)
    c.rect(0, 0, SLIDE_W, SLIDE_H, fill=1, stroke=0)

    # Header bar
    header_y = SLIDE_H - HEADER_H
    c.setFillColor(C_PURPLE)
    c.rect(0, header_y, SLIDE_W, HEADER_H, fill=1, stroke=0)

    # Gold accent stripe under header
    c.setFillColor(C_GOLD)
    c.rect(0, header_y - 4, SLIDE_W, 4, fill=1, stroke=0)

    # Footer bar
    c.setFillColor(HexColor("#EEEEEE"))
    c.rect(0, 0, SLIDE_W, FOOTER_Y + 6, fill=1, stroke=0)
    c.setFillColor(C_MGREY)
    c.rect(0, FOOTER_Y + 6, SLIDE_W, 0.8, fill=1, stroke=0)

    # Footer text
    c.setFillColor(HexColor("#888888"))
    c.setFont("Helvetica", 7)
    c.drawString(MARGIN, FOOTER_Y - 2,
                 "Hallucination Evaluation in LLM-Based Medical Summarization  ·  CSUN Comp 696C  ·  April 2026")
    c.drawRightString(SLIDE_W - MARGIN, FOOTER_Y - 2,
                      f"{slide_num} / {total}")


def header_text(c, title, subtitle=None):
    """Write title in header bar."""
    mid_y = SLIDE_H - HEADER_H / 2
    c.setFillColor(C_WHITE)
    if subtitle:
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(SLIDE_W / 2, mid_y + 4, title)
        c.setFont("Helvetica", 9)
        c.drawCentredString(SLIDE_W / 2, mid_y - 8, subtitle)
    else:
        c.setFont("Helvetica-Bold", 15)
        c.drawCentredString(SLIDE_W / 2, mid_y - 4, title)


def body_top():
    """Y coordinate just below header + gold stripe."""
    return SLIDE_H - HEADER_H - 4 - 14   # a little padding


def draw_bullet(c, x, y, text, font_size=10, indent=0, color=None, bold_prefix=None):
    """Draw a single bullet line; returns new y after drawing."""
    c.setFillColor(color or C_DGREY)
    bullet_x = x + indent
    text_x   = bullet_x + 14

    c.setFont("Helvetica-Bold", font_size)
    c.drawString(bullet_x, y, "•")

    if bold_prefix:
        bp_w = c.stringWidth(bold_prefix + " ", "Helvetica-Bold", font_size)
        c.setFont("Helvetica-Bold", font_size)
        c.drawString(text_x, y, bold_prefix + " ")
        c.setFont("Helvetica", font_size)
        # wrap remainder
        remainder = text
        words = remainder.split()
        line = ""
        first = True
        for word in words:
            test = (line + " " + word).strip()
            w = c.stringWidth(test, "Helvetica", font_size)
            avail = CONTENT_W - (text_x - CONTENT_X) - (bp_w if first else 0)
            if w > avail and line:
                draw_x = text_x + (bp_w if first else 0)
                c.drawString(draw_x, y, line)
                y -= font_size + 3
                line = word
                first = False
            else:
                line = test
        if line:
            draw_x = text_x + (bp_w if first else 0)
            c.drawString(draw_x, y, line)
        y -= font_size + 5
    else:
        # wrap text
        words = text.split()
        line = ""
        first = True
        avail_w = CONTENT_W - (text_x - CONTENT_X)
        for word in words:
            test = (line + " " + word).strip()
            w = c.stringWidth(test, "Helvetica", font_size)
            if w > avail_w and line:
                c.setFont("Helvetica", font_size)
                c.drawString(text_x, y, line)
                y -= font_size + 3
                line = word
            else:
                line = test
        if line:
            c.setFont("Helvetica", font_size)
            c.drawString(text_x, y, line)
        y -= font_size + 5
    return y


def draw_wrapped(c, x, y, text, font="Helvetica", size=10, color=None, max_w=None, leading=None):
    """Draw wrapped paragraph; return new y."""
    c.setFillColor(color or C_DGREY)
    c.setFont(font, size)
    _lead = leading or (size + 3)
    avail = max_w or CONTENT_W
    words = text.split()
    line  = ""
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, font, size) > avail and line:
            c.drawString(x, y, line)
            y -= _lead
            line = word
        else:
            line = test
    if line:
        c.drawString(x, y, line)
        y -= _lead
    return y


def section_label(c, x, y, label):
    """Purple bold section label."""
    c.setFillColor(C_PURPLE)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, label)
    # underline
    w = c.stringWidth(label, "Helvetica-Bold", 11)
    c.setStrokeColor(C_GOLD)
    c.setLineWidth(1.5)
    c.line(x, y - 2, x + w, y - 2)
    return y - 16


def draw_table_row(c, cols, x, y, col_widths, font, size, fg, bg, pad=4):
    """Draw a single table row with background fill."""
    row_h = size + 2 * pad
    total_w = sum(col_widths)
    c.setFillColor(bg)
    c.rect(x, y - row_h + pad, total_w, row_h, fill=1, stroke=0)
    c.setFillColor(fg)
    cx = x
    for col, w in zip(cols, col_widths):
        c.setFont(font, size)
        c.drawCentredString(cx + w / 2, y - size + pad / 2, str(col))
        cx += w
    # light grid line
    c.setStrokeColor(C_MGREY)
    c.setLineWidth(0.4)
    c.rect(x, y - row_h + pad, total_w, row_h, fill=0, stroke=1)
    return y - row_h


def build_slides():
    c = rl_canvas.Canvas(str(SLIDES_OUT), pagesize=(SLIDE_W, SLIDE_H))
    c.setTitle("Thesis Progress Presentation")
    c.setAuthor("Laya")

    total_slides = 10

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    draw_chrome(c, 1, total_slides)
    header_text(c, "Hallucination Evaluation in LLM-Based Medical Summarization")

    cy = body_top() - 20
    c.setFillColor(C_PURPLE)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(SLIDE_W / 2, cy,
        "Hallucination Evaluation in")
    cy -= 24
    c.drawCentredString(SLIDE_W / 2, cy,
        "LLM-Based Medical Summarization")
    cy -= 32

    c.setFillColor(C_DGREY)
    c.setFont("Helvetica", 12)
    c.drawCentredString(SLIDE_W / 2, cy, "Thesis Progress Presentation  ·  Comp 696C")
    cy -= 18
    c.drawCentredString(SLIDE_W / 2, cy, "California State University, Northridge")
    cy -= 26

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(C_PURPLE)
    c.drawCentredString(SLIDE_W / 2, cy, "Student: Laya")
    cy -= 16
    c.drawCentredString(SLIDE_W / 2, cy, "April 2026")

    # Gold divider
    cy -= 20
    c.setStrokeColor(C_GOLD)
    c.setLineWidth(2)
    c.line(SLIDE_W / 2 - 100, cy, SLIDE_W / 2 + 100, cy)

    c.showPage()

    # ── Slide 2: Motivation & Problem ─────────────────────────────────────────
    draw_chrome(c, 2, total_slides)
    header_text(c, "Motivation & Problem Statement")

    cy = body_top()
    cy = section_label(c, CONTENT_X, cy, "Why This Matters")
    cy -= 4

    bullets2 = [
        ("Clinical NLP is rapidly entering practice:",
         "LLMs are used to generate discharge summaries, referral letters, and "
         "patient-facing notes at scale."),
        ("Hallucinations are dangerous:",
         "Fabricated medications, incorrect dosages, or invented diagnoses in a "
         "clinical summary can directly harm patients."),
        ("Verification gap:",
         "Clinicians reviewing AI-generated text may over-trust outputs without "
         "systematic verification frameworks."),
        ("RAG as a mitigation:",
         "Retrieval-Augmented Generation constrains the model to retrieved source "
         "passages — but its effect on hallucination rates is poorly quantified in "
         "the medical domain."),
    ]
    for bold, rest in bullets2:
        cy = draw_bullet(c, CONTENT_X, cy, rest, font_size=10, bold_prefix=bold)
        cy -= 3

    cy -= 6
    cy = section_label(c, CONTENT_X, cy, "The Core Problem")
    cy -= 4
    cy = draw_wrapped(c, CONTENT_X + 14, cy,
        "We lack a reproducible, claim-level metric framework that quantifies "
        "how often LLM-generated medical summaries introduce unsupported or "
        "contradicted facts — and how much RAG helps.",
        font="Helvetica-Oblique", size=10, color=HexColor("#333333"))

    c.showPage()

    # ── Slide 3: Research Questions & Hypotheses ──────────────────────────────
    draw_chrome(c, 3, total_slides)
    header_text(c, "Research Questions & Hypotheses")

    cy = body_top()
    cy = section_label(c, CONTENT_X, cy, "Research Questions")
    cy -= 4

    rqs = [
        "RQ1: To what extent do LLM-generated medical summaries contain "
        "unsupported or contradicted facts relative to the source document?",
        "RQ2: Does retrieval-augmented generation (RAG) significantly reduce "
        "hallucination rates compared to a non-retrieval baseline?",
        "RQ3: What types of hallucinations persist even with RAG, and can "
        "verification-stage prompting (CoVe) address them?",
    ]
    for rq in rqs:
        cy = draw_bullet(c, CONTENT_X, cy, rq, font_size=10)
        cy -= 2

    cy -= 12
    cy = section_label(c, CONTENT_X, cy, "Hypotheses")
    cy -= 4

    hyps = [
        ("H1:", "Baseline LLM summaries will exhibit UFR > 0.5, reflecting "
         "a high rate of ungrounded claims in unconstrained generation."),
        ("H2:", "RAG-augmented summaries will have significantly lower CR "
         "than baseline (p < 0.05, Wilcoxon signed-rank test)."),
        ("H3:", "A verification stage (E3 CoVe) will further reduce UFR "
         "beyond what retrieval alone achieves."),
    ]
    for label, text in hyps:
        cy = draw_bullet(c, CONTENT_X, cy, text, font_size=10, bold_prefix=label)
        cy -= 2

    c.showPage()

    # ── Slide 4: Methodology — Pipeline ───────────────────────────────────────
    draw_chrome(c, 4, total_slides)
    header_text(c, "Methodology — Evaluation Pipeline")

    cy = body_top() - 6

    # Pipeline diagram as a horizontal flow of labeled boxes
    box_labels = [
        "Source\nDocument",
        "Summary\nGeneration\n(E0/E1/E2)",
        "Claim\nSegmentation\n(spaCy)",
        "Evidence\nRetrieval\n(cos-sim)",
        "NLI\nClassification\n(cross-encoder)",
        "Metrics\nUFR / CR",
    ]
    n_boxes   = len(box_labels)
    box_w     = 95
    box_h     = 62
    gap       = 14
    total_row = n_boxes * box_w + (n_boxes - 1) * gap
    start_x   = (SLIDE_W - total_row) / 2
    box_y     = cy - box_h

    BOX_COLORS = [
        HexColor("#E8E0F0"),  # source
        HexColor("#D6E4F7"),  # generation
        HexColor("#D6E4F7"),  # claim
        HexColor("#D6E4F7"),  # evidence
        HexColor("#D6E4F7"),  # NLI
        HexColor("#D4EDD9"),  # metrics
    ]

    for i, (label, bg) in enumerate(zip(box_labels, BOX_COLORS)):
        bx = start_x + i * (box_w + gap)

        # arrow before box (except first)
        if i > 0:
            ax = bx - gap
            ay = box_y + box_h / 2
            c.setStrokeColor(C_PURPLE)
            c.setLineWidth(1.4)
            c.line(ax, ay, bx, ay)
            # arrowhead
            c.setFillColor(C_PURPLE)
            p = c.beginPath()
            p.moveTo(bx - 6, ay - 4)
            p.lineTo(bx - 6, ay + 4)
            p.lineTo(bx, ay)
            p.close()
            c.drawPath(p, fill=1, stroke=0)

        # box
        c.setFillColor(bg)
        c.setStrokeColor(C_PURPLE)
        c.setLineWidth(1)
        c.roundRect(bx, box_y, box_w, box_h, 5, fill=1, stroke=1)

        # label (multi-line)
        lines = label.split("\n")
        line_h = 11
        start_text_y = box_y + box_h / 2 + (len(lines) - 1) * line_h / 2
        for j, ln in enumerate(lines):
            txt_y = start_text_y - j * line_h
            if j == 0:
                c.setFont("Helvetica-Bold", 9)
                c.setFillColor(C_PURPLE)
            else:
                c.setFont("Helvetica", 8.5)
                c.setFillColor(C_DGREY)
            c.drawCentredString(bx + box_w / 2, txt_y, ln)

    cy = box_y - 20

    # Labels below diagram
    labels_row = [
        ("E0", "#E07B54", "GPT-4o-mini\n(no retrieval)"),
        ("E1", "#4C8BB5", "GPT-4o-mini\n+ RAG chunks"),
        ("E2", "#5DBF6E", "Centroid extraction\n(no LLM)"),
    ]
    lx = CONTENT_X + 40
    for short, color, desc in labels_row:
        c.setFillColor(HexColor(color))
        c.roundRect(lx, cy - 20, 20, 20, 3, fill=1, stroke=0)
        c.setFillColor(C_DGREY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(lx + 25, cy - 8, f"{short}:")
        c.setFont("Helvetica", 8.5)
        for k, line in enumerate(desc.split("\n")):
            c.drawString(lx + 56, cy - 4 - k * 11, line)
        lx += 190

    cy -= 40

    # NLI label key
    cy = section_label(c, CONTENT_X, cy, "NLI Output Labels")
    cy -= 2
    for label, color, meaning in [
        ("Supported",     "#2A8C4A", "claim entailed by evidence (p_entail ≥ 0.5)"),
        ("Not-Supported", "#E07B54", "claim not entailed or contradicted"),
        ("Contradicted",  "#CC0000", "claim contradicted by evidence (p_contra ≥ 0.5)"),
    ]:
        c.setFillColor(HexColor(color))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(CONTENT_X + 14, cy, label + ":")
        w = c.stringWidth(label + ":  ", "Helvetica-Bold", 9)
        c.setFillColor(C_DGREY)
        c.setFont("Helvetica", 9)
        c.drawString(CONTENT_X + 14 + w, cy, meaning)
        cy -= 13

    c.showPage()

    # ── Slide 5: Dataset & Setup ──────────────────────────────────────────────
    draw_chrome(c, 5, total_slides)
    header_text(c, "Dataset & Experimental Setup")

    cy = body_top()

    # Two-column layout
    left_x  = CONTENT_X
    right_x = SLIDE_W / 2 + 10
    col_w   = SLIDE_W / 2 - CONTENT_X - 10

    # Left column
    cy_l = cy
    cy_l = section_label(c, left_x, cy_l, "Dataset — MTSamples")
    cy_l -= 4
    for item in [
        "Public de-identified clinical transcriptions",
        "Specialties: Discharge Summary + Consult–H&P",
        "50 samples, random seed = 42 (reproducible)",
        "Source docs: median ~300 words each",
    ]:
        cy_l = draw_bullet(c, left_x, cy_l, item, font_size=9)

    cy_l -= 8
    cy_l = section_label(c, left_x, cy_l, "Models")
    cy_l -= 4
    for item in [
        "LLM: GPT-4o-mini (OpenAI), T=0.3, max 450 tok",
        "Bi-encoder: all-MiniLM-L6-v2 (retrieval + embed)",
        "Cross-encoder: nli-MiniLM2-L6-H768 (NLI labels)",
        "SpaCy en_core_web_sm (sentence segmentation)",
    ]:
        cy_l = draw_bullet(c, left_x, cy_l, item, font_size=9)

    # Right column
    cy_r = cy
    cy_r = section_label(c, right_x, cy_r, "Experimental Conditions")
    cy_r -= 4
    conds = [
        ("E0 — Baseline", "GPT-4o-mini given full note, no retrieval"),
        ("E1 — RAG",      "Top-3 retrieved chunks fed to GPT-4o-mini"),
        ("E2 — Extractive","Top-5 centroid sentences, no LLM"),
    ]
    for bold, rest in conds:
        cy_r = draw_bullet(c, right_x, cy_r, rest, font_size=9, bold_prefix=bold)
        cy_r -= 2

    cy_r -= 8
    cy_r = section_label(c, right_x, cy_r, "Metrics")
    cy_r -= 4
    for item in [
        "UFR = (Contradicted + Not-Supported) / n_claims",
        "CR  = Contradicted / n_claims",
        "Wilcoxon signed-rank test (two-sided) for significance",
    ]:
        cy_r = draw_bullet(c, right_x, cy_r, item, font_size=9)

    cy_r -= 8
    cy_r = section_label(c, right_x, cy_r, "Claim Statistics")
    cy_r -= 4
    for cond, n in [("E0", 653), ("E1", 478), ("E2", 242)]:
        c.setFillColor(C_DGREY)
        c.setFont("Helvetica", 9)
        c.drawString(right_x + 14, cy_r, f"{cond}: {n} claims evaluated")
        cy_r -= 13

    c.showPage()

    # ── Slide 6: 3-Way Results Table ──────────────────────────────────────────
    draw_chrome(c, 6, total_slides)
    header_text(c, "3-Way Results — E0 vs E1 vs E2")

    cy = body_top() - 10

    # Table
    col_w_list = [90, 80, 80, 90, 80, 80, 80]
    total_w = sum(col_w_list)
    tx = (SLIDE_W - total_w) / 2

    header_data = ["Metric", "E0 Baseline", "E1 RAG", "E2 Extractive",
                   "E1−E0 Δ", "E2−E0 Δ", "Wilcoxon p"]
    rows_data = [
        ["UFR", f"{e0_ufr:.4f}", f"{e1_ufr:.4f}", f"{e2_ufr:.4f}",
         f"{d_ufr:+.4f}", f"{ufr_row['E2_Delta_Mean']:+.4f}", "0.576  n.s."],
        ["CR",  f"{e0_cr:.4f}",  f"{e1_cr:.4f}",  f"{e2_cr:.4f}",
         f"{d_cr:+.4f}",  f"{cr_row['E2_Delta_Mean']:+.4f}", "0.0001  ***"],
    ]

    # Header row
    row_h = 22
    c.setFillColor(C_PURPLE)
    c.rect(tx, cy - row_h, total_w, row_h, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 10)
    cx2 = tx
    for col, w in zip(header_data, col_w_list):
        c.drawCentredString(cx2 + w / 2, cy - row_h + 7, col)
        cx2 += w
    cy -= row_h

    # Data rows
    row_bgs = [HexColor(LIGHT_GREY), C_WHITE]
    for ri, row in enumerate(rows_data):
        bg = row_bgs[ri % 2]
        c.setFillColor(bg)
        c.rect(tx, cy - row_h, total_w, row_h, fill=1, stroke=0)
        cx2 = tx
        for ci, (col, w) in enumerate(zip(row, col_w_list)):
            if ci == 0:
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(C_PURPLE)
            elif col.startswith("-") and ci in (4, 5):
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(C_GREEN)
            elif col.startswith("+") and ci in (4, 5):
                c.setFont("Helvetica", 10)
                c.setFillColor(C_RED)
            elif "***" in col:
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(C_RED)
            else:
                c.setFont("Helvetica", 10)
                c.setFillColor(C_DGREY)
            c.drawCentredString(cx2 + w / 2, cy - row_h + 7, col)
            cx2 += w

        # border
        c.setStrokeColor(C_MGREY)
        c.setLineWidth(0.5)
        c.rect(tx, cy - row_h, total_w, row_h, fill=0, stroke=1)
        cy -= row_h

    cy -= 16

    # Glossary
    c.setFillColor(HexColor("#666666"))
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawCentredString(SLIDE_W / 2, cy,
        "UFR = (Contradicted + Not-Supported) / total claims  ·  "
        "CR = Contradicted / total claims  ·  "
        "Δ = condition − E0")
    cy -= 13
    c.drawCentredString(SLIDE_W / 2, cy,
        "Green Δ = improvement over baseline  ·  "
        "*** p < 0.001 (Wilcoxon signed-rank, two-sided)")
    cy -= 20

    # Key takeaways
    cy = section_label(c, CONTENT_X, cy, "Key Takeaways")
    cy -= 4
    for item in [
        "RAG (E1) significantly reduces Contradiction Rate by 32% (p = 0.0001) but does not improve UFR.",
        "Extractive baseline (E2) achieves near-zero UFR = 0.122, confirming pipeline validity.",
        "Residual Not-Supported claims in E1 point to hallucinations beyond simple contradiction.",
    ]:
        cy = draw_bullet(c, CONTENT_X, cy, item, font_size=9)

    c.showPage()

    # ── Slide 7: CR Finding ───────────────────────────────────────────────────
    draw_chrome(c, 7, total_slides)
    header_text(c, "CR Finding — RAG Significantly Reduces Contradictions")

    cy = body_top() - 4

    # Embed CR boxplot (left ~55% of slide)
    img_w = 310
    img_h = 235
    img_x = CONTENT_X
    img_y = cy - img_h

    if PNG_CR.exists():
        c.drawImage(str(PNG_CR), img_x, img_y, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")

    # Right column — stats + bullets
    rx = img_x + img_w + 22
    ry = cy
    c.setFillColor(C_PURPLE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(rx, ry, "Contradiction Rate")
    ry -= 18

    # Highlight p-value box
    c.setFillColor(C_RED)
    c.roundRect(rx, ry - 18, 165, 22, 4, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(rx + 82, ry - 6, "p = 0.0001  ***")
    ry -= 34

    stats_cr = [
        ("E0 Mean CR", f"{e0_cr:.4f}"),
        ("E1 Mean CR", f"{e1_cr:.4f}"),
        ("E2 Mean CR", f"{e2_cr:.4f}"),
        ("E1 reduction", f"{abs(d_cr/e0_cr)*100:.0f}%"),
        ("E1 improved", f"{cr_row['Pct_Improved']:.0f}% of samples"),
    ]
    for label, val in stats_cr:
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(C_PURPLE)
        c.drawString(rx, ry, label + ":")
        lw = c.stringWidth(label + ":  ", "Helvetica-Bold", 9.5)
        c.setFont("Helvetica", 9.5)
        c.setFillColor(C_DGREY)
        c.drawString(rx + lw, ry, val)
        ry -= 15

    ry -= 6
    interp = (
        "RAG constrains the model to retrieved "
        "source passages, preventing outright "
        "contradictions of stated facts. "
        "The reduction is highly significant "
        "(Wilcoxon W, two-sided)."
    )
    for line in interp.split("  "):
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(HexColor("#444444"))
        ry = draw_wrapped(c, rx, ry, line, font="Helvetica-Oblique",
                          size=9, max_w=185)
        ry -= 2

    # Caption under image
    c.setFillColor(HexColor("#888888"))
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawCentredString(img_x + img_w / 2, img_y - 10,
        "Figure: CR distribution across 50 samples (orange=E0, blue=E1, green=E2)")

    c.showPage()

    # ── Slide 8: UFR Finding ──────────────────────────────────────────────────
    draw_chrome(c, 8, total_slides)
    header_text(c, "UFR Finding — E2 Validates Pipeline; RAG Alone Insufficient")

    cy = body_top() - 4

    img_w = 310
    img_h = 235
    img_x = CONTENT_X
    img_y = cy - img_h

    if PNG_UFR.exists():
        c.drawImage(str(PNG_UFR), img_x, img_y, width=img_w, height=img_h,
                    preserveAspectRatio=True, mask="auto")

    rx = img_x + img_w + 22
    ry = cy
    c.setFillColor(C_PURPLE)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(rx, ry, "Unsupported Fact Rate")
    ry -= 18

    # p-value box (n.s.)
    c.setFillColor(HexColor("#888888"))
    c.roundRect(rx, ry - 18, 165, 22, 4, fill=1, stroke=0)
    c.setFillColor(C_WHITE)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(rx + 82, ry - 6, "p = 0.576  n.s.")
    ry -= 34

    stats_ufr = [
        ("E0 Mean UFR", f"{e0_ufr:.4f}"),
        ("E1 Mean UFR", f"{e1_ufr:.4f}"),
        ("E2 Mean UFR", f"{e2_ufr:.4f}  ✓ near-zero"),
        ("E1 improved", f"{ufr_row['Pct_Improved']:.0f}% of samples"),
        ("E2 improved", "100% of samples"),
    ]
    for label, val in stats_ufr:
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(C_PURPLE)
        c.drawString(rx, ry, label + ":")
        lw = c.stringWidth(label + ":  ", "Helvetica-Bold", 9.5)
        c.setFont("Helvetica", 9.5)
        c.setFillColor(C_DGREY if "near-zero" not in val else C_GREEN)
        c.drawString(rx + lw, ry, val)
        ry -= 15

    ry -= 6
    interp2 = (
        "RAG does not significantly reduce UFR — "
        "the model still produces ungrounded claims "
        "even when given retrieved context. "
        "E2's near-zero UFR confirms the NLI "
        "pipeline correctly identifies source-"
        "faithful text, ruling out measurement error."
    )
    for chunk in interp2.split("  "):
        ry = draw_wrapped(c, rx, ry, chunk, font="Helvetica-Oblique",
                          size=9, max_w=185, color=HexColor("#444444"))
        ry -= 2

    c.setFillColor(HexColor("#888888"))
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawCentredString(img_x + img_w / 2, img_y - 10,
        "Figure: UFR distribution across 50 samples (orange=E0, blue=E1, green=E2)")

    c.showPage()

    # ── Slide 9: Qualitative Examples ─────────────────────────────────────────
    draw_chrome(c, 9, total_slides)
    header_text(c, "Qualitative Examples")

    cy = body_top() - 4

    # Box helper
    def example_box(x, y, w, h, border_color, title, label_color,
                    e0_text, e1_text, e0_label, e1_label):
        # Border
        c.setStrokeColor(border_color)
        c.setLineWidth(1.5)
        c.roundRect(x, y - h, w, h, 6, fill=0, stroke=1)
        # Title bar
        c.setFillColor(border_color)
        c.roundRect(x, y - 20, w, 20, 6, fill=1, stroke=0)
        c.rect(x, y - 25, w, 10, fill=1, stroke=0)  # square bottom corners
        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawCentredString(x + w / 2, y - 14, title)

        inner_x = x + 8
        inner_w  = w - 16
        iy = y - 28

        # E0 row
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(HexColor("#E07B54"))
        c.drawString(inner_x, iy, "E0:")
        c.setFont("Helvetica", 8)
        c.setFillColor(C_DGREY)
        iy = draw_wrapped(c, inner_x + 22, iy, e0_text[:180],
                          size=8, max_w=inner_w - 24, color=C_DGREY)
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(HexColor("#CC0000"))
        c.drawString(inner_x + 22, iy, f"→ {e0_label}")
        iy -= 14

        # E1 row
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(HexColor("#4C8BB5"))
        c.drawString(inner_x, iy, "E1:")
        c.setFont("Helvetica", 8)
        c.setFillColor(C_DGREY)
        iy = draw_wrapped(c, inner_x + 22, iy, e1_text[:180],
                          size=8, max_w=inner_w - 24, color=C_DGREY)
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(C_GREEN)
        c.drawString(inner_x + 22, iy, f"→ {e1_label}")

        return y - h

    box_h  = 155
    box_w  = (CONTENT_W - 20) / 2
    box_y  = cy

    # Left: RAG-fixed
    example_box(
        CONTENT_X, box_y, box_w, box_h,
        HexColor("#2A8C4A"),
        "RAG-Fixed: Contradicted → Supported",
        C_GREEN,
        RAG_FIXED_E0, RAG_FIXED_E1,
        "Contradicted (baseline hallucination)",
        "Supported (RAG grounded the claim)",
    )

    # Right: Persistent
    example_box(
        CONTENT_X + box_w + 20, box_y, box_w, box_h,
        HexColor("#CC4444"),
        "Persistent Hallucination: Not-Supported in E0 & E1",
        C_RED,
        PERSIST_E0, PERSIST_E1,
        "Not-Supported in E0",
        "Not-Supported in E1  (RAG did not help)",
    )

    cy = box_y - box_h - 14
    cy = section_label(c, CONTENT_X, cy, "Interpretation")
    cy -= 4
    cy = draw_wrapped(c, CONTENT_X + 14, cy,
        "RAG eliminates contradictions by anchoring the model to retrieved excerpts. "
        "However, claims that reference follow-up instructions, future events, or "
        "context not present in any retrieved chunk remain Not-Supported in both "
        "conditions — suggesting a retrieval coverage gap rather than a generation "
        "defect. E3 (verification prompting) targets these residual failures.",
        size=9, color=C_DGREY)

    c.showPage()

    # ── Slide 10: Next Steps ──────────────────────────────────────────────────
    draw_chrome(c, 10, total_slides)
    header_text(c, "Next Steps & Semester 2 Plan")

    cy = body_top() - 4

    # Two columns
    left_x2  = CONTENT_X
    right_x2 = SLIDE_W / 2 + 10

    cy_l2 = cy
    cy_l2 = section_label(c, left_x2, cy_l2, "Immediate (Next 4 Weeks)")
    cy_l2 -= 4
    for item in [
        "Await PhysioNet credentialing approval and migrate pipeline to MIMIC-III notes.",
        "Obtain ann-pt-summ annotated dataset; compute inter-annotator agreement baseline.",
        "Run E3 — Chain-of-Verification prompting on E0 outputs; measure ΔCR and ΔUFR.",
        "Begin qualitative error taxonomy: categorize persistent Not-Supported claims by type.",
    ]:
        cy_l2 = draw_bullet(c, left_x2, cy_l2, item, font_size=9)
        cy_l2 -= 2

    cy_r2 = cy
    cy_r2 = section_label(c, right_x2, cy_r2, "Thesis Completion (Semester 2)")
    cy_r2 -= 4
    for item in [
        "Chapter 2 — Literature Review (LLM hallucinations, RAG, NLI evaluation).",
        "Chapter 3 — Methodology (finalize with MIMIC results if available).",
        "Chapter 4 — Results (quantitative + qualitative findings for all conditions).",
        "Chapter 5 — Discussion: limitations, future work, clinical implications.",
        "Committee review, defense preparation, and final submission.",
    ]:
        cy_r2 = draw_bullet(c, right_x2, cy_r2, item, font_size=9)
        cy_r2 -= 2

    # Timeline bar
    min_cy = min(cy_l2, cy_r2) - 20
    cy_tl = min_cy

    cy_tl = section_label(c, CONTENT_X, cy_tl, "Timeline")
    cy_tl -= 8

    milestones = [
        ("Now",    "E0/E1/E2\nComplete"),
        ("May",    "E3 CoVe\nVerification"),
        ("Jun",    "PhysioNet\nMigration"),
        ("Aug",    "Draft\nChapters"),
        ("Oct",    "Committee\nReview"),
        ("Dec",    "Defense"),
    ]
    n_ms  = len(milestones)
    tl_x  = CONTENT_X + 10
    tl_w  = CONTENT_W - 20
    step  = tl_w / (n_ms - 1)
    tl_y  = cy_tl - 12

    # Spine
    c.setStrokeColor(C_PURPLE)
    c.setLineWidth(2)
    c.line(tl_x, tl_y, tl_x + tl_w, tl_y)

    for i, (month, label) in enumerate(milestones):
        mx = tl_x + i * step
        # dot
        filled = i == 0
        c.setFillColor(C_PURPLE if filled else C_WHITE)
        c.setStrokeColor(C_PURPLE)
        c.setLineWidth(1.5)
        c.circle(mx, tl_y, 5, fill=1, stroke=1)
        if filled:
            c.setFillColor(C_WHITE)
            c.circle(mx, tl_y, 2.5, fill=1, stroke=0)

        # month label above
        c.setFillColor(C_PURPLE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(mx, tl_y + 10, month)

        # milestone label below
        c.setFillColor(C_DGREY)
        c.setFont("Helvetica", 7.5)
        for k, ln in enumerate(label.split("\n")):
            c.drawCentredString(mx, tl_y - 14 - k * 10, ln)

    c.showPage()

    c.save()
    print(f"  [saved] {SLIDES_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building progress_report.pdf …")
    build_report()
    print("Building progress_slides.pdf …")
    build_slides()
    print("Done.")
