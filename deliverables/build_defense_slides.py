#!/usr/bin/env python3
"""
build_defense_slides.py: thesis defense slides (PowerPoint via python-pptx, PDF converted from it by LibreOffice).

Every number is read from ../results at build time and the charts are drawn from the same files, so the deck cannot
drift from the thesis. The talk has six sections: Introduction and Problem Statement, Background and Related Work,
Proposed Approach and Methodology, Experimental Setup and Results, Results Analysis and Discussion, and Conclusions and
Future Work. An overview slide lists them and a divider slide opens each one. The slides after the closing slide are
backup slides for questions and reuse the thesis tables.
The deck in deliverables/ carries no speaker notes. When ../../defense_notes/defense_notes.py exists, a copy with the
notes as short bullet points and a PDF with each slide and its notes on one page are written to that folder, outside
the repository.

Usage: python deliverables/build_defense_slides.py
Outputs: deliverables/Defense_Slides_Fall2026.pptx and .pdf; with notes: ../defense_notes/Defense_Slides_Fall2026_with_notes.pptx and .pdf
"""
import importlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
BUILD = HERE / "_build" / "defense"
sys.path.insert(0, str(ROOT / "thesis"))
from results_loader import Results, f2, f3, fp, holm, pct, pct0  # noqa: E402
from mimic_results import Mimic, JUDGE_NAMES  # noqa: E402
from render_docx import set_properties, soffice_convert  # noqa: E402

OUT = "Defense_Slides_Fall2026"
DOC_TITLE = "Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization: Thesis Defense"
DOC_AUTHOR = "Srilaya Ponangi"
DEFENSE_DATE = "November 23, 2026"
SW, SH = 13.333, 7.5            # slide size in inches
MX = 0.6                        # side margin
CW = SW - 2 * MX                # content width
TOP, BOTTOM = 1.95, 6.85        # content area

# A quiet palette: a light gray slide, white cards, charcoal text, gray shapes, and one soft red accent that is kept
# for what the eye should find first: the condition the results favor, the selected judge, unsupported claims.
BG, CARD = "EEF0F2", "FFFFFF"
ACC, ACC_SOFT = "C4515A", "F6E6E8"
INK, TEXT, MUTED, FAINT = "2F3237", "3A3D42", "6B7078", "9499A1"
STRONG, SLATE, MID, LIGHT = "43474E", "5F646C", "969BA4", "D3D6DB"
HAIR, TINT, TINT2, WHITE = "DCDFE3", "EEF0F2", "E2E5E9", "FFFFFF"
ROW = ACC_SOFT                                          # the highlighted table row
# one color per summarization condition, the same on every slide: charcoal for the baseline, the accent for the condition
# the results favor, two grays for the excerpt-only conditions. The two references are drawn hollow.
CCOL = {"E0": STRONG, "E1": MID, "E1b": ACC, "E3": "C9CDD3", "E2": "FFFFFF", "REF": "FFFFFF"}
HOLLOW, EDGE = ("E2", "REF"), MID
HUMAN, JUDGE = STRONG, "BEC2C9"       # manual or expert labels, automatic judge
CNAME = {"E0": "E0 full note", "E1": "E1 RAG, excerpts", "E1b": "E1b RAG + course", "E3": "E3 RAG + verification",
         "E2": "E2 extractive", "REF": "Clinician's instructions"}
SECTIONS = ["Introduction and Problem Statement", "Background and Related Work", "Proposed Approach and Methodology",
            "Experimental Setup and Results", "Results Analysis and Discussion", "Conclusions and Future Work"]
# what each section goes through, on the overview slide and on the section's divider slide
SECTION_ITEMS = [["Why unsupported statements matter", "The two problems", "Research questions and hypotheses"],
                 ["Measuring faithfulness", "Clinical summarization", "Reducing hallucination", "The gap"],
                 ["Three studies", "The claim-level judge", "How the judge is chosen", "RAG and Chain-of-Verification"],
                 ["Datasets and setup", "Metrics and statistical tests", "Threats and checks", "Eight results"],
                 ["What the judge's errors mean", "Answers to the research questions", "Existing approaches", "Limitations"],
                 ["Main contributions", "Conclusions", "Future work"]]
# the order of the talk; every slide key of the talk appears once
MAIN_ORDER = ["title", "agenda",
              "sec1", "useful", "motivation", "p_measure", "p_mitigate", "p_thesis", "rq", "hyp",
              "sec2", "rel_measure", "rel_clinical", "rel_reduce", "gap",
              "sec3", "overview", "pipeline", "labelrule", "j_cand", "j_rule", "conditions", "rag", "cove",
              "sec4", "data", "laptop", "rates", "measures", "stats", "controls", "controls2",
              "r_pilot", "r_pilot2", "r_judge", "r_judge2", "r_ufr", "r_ufr_ref", "r_cov", "r_length", "r_citeinst", "r_cites",
              "r_robust", "r_robust2", "r_gen", "r_audit_how", "r_audit", "r_audit2",
              "sec5", "d_errors", "d_formula", "d_pq", "a_rq1", "a_rq2", "a_rq3", "d_tradeoff", "d_favor", "d_existing", "d_existing2",
              "lim_judge", "lim_audit", "lim_data", "scope",
              "sec6", "contrib", "conclusions", "implications", "future", "closing"]
FONT = "Arial"
_FONT_FILES = {False: "/System/Library/Fonts/Supplemental/Arial.ttf", True: "/System/Library/Fonts/Supplemental/Arial Bold.ttf"}
_FONTS = {}
WARNINGS = []


def csv(name):
    f = RES / name
    try:
        return pd.read_csv(f) if f.exists() else None
    except pd.errors.EmptyDataError:
        return None


def rgb(hexstr):
    return RGBColor.from_string(hexstr)


# ───────────────────────────── text measuring ─────────────────────────────
def text_w(s, size, bold=False):
    """Width of a string in inches at a font size in points."""
    if bold not in _FONTS:
        _FONTS[bold] = ImageFont.truetype(_FONT_FILES[bold], 200)
    return _FONTS[bold].getlength(s) / 200 * size / 72


def parse_runs(s):
    """'plain **bold** plain' -> [(text, bold)]."""
    out = []
    for i, part in enumerate(re.split(r"\*\*", s)):
        if part:
            out.append((part, i % 2 == 1))
    return out or [("", False)]


def n_lines(s, size, width, bold=False):
    """Number of lines a paragraph needs when wrapped by words in a box of the given width (inches)."""
    lines, cur = 1, 0.0
    space = text_w(" ", size, bold)
    for txt, b in parse_runs(s):
        for word in [x for x in re.split(r"[ \t\n]+", txt) if x]:
            w = text_w(word, size, bold or b)
            if cur and cur + space + w > width:
                lines += 1; cur = w
            else:
                cur = cur + (space if cur else 0) + w
    return lines


def text_h(paras, size, width, bold=False, line=1.1, after=6):
    """Estimated height in inches of a list of paragraphs."""
    h = 0.0
    for p in paras:
        h += n_lines(p, size, width, bold) * size * line * 1.14 / 72 + after / 72
    return h - after / 72


def table_heights(columns, rows, cw, size, head_size, pad, bold_col0=False):
    """Row heights in inches of a table whose column widths in inches are cw."""
    def rh(vals, s, bold_first=False, bold_all=False):
        m = max(n_lines(str(v), s, cw[j] - 2 * pad - 0.02, bold_all or (bold_first and j == 0)) for j, v in enumerate(vals))
        return m * s * 1.2 / 72 + 2 * pad
    return [rh(columns, head_size, bold_all=True)] + [rh(r, size, bold_col0) for r in rows]


def _flat(shape):
    """Remove the theme style reference of a shape, so that no application draws a theme shadow on it."""
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def P(p):
    """A p-value as text that never breaks across lines."""
    return fp(p).replace(" ", "\u00a0")


# ───────────────────────────── slide primitives ─────────────────────────────
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        set_properties(self.prs, DOC_TITLE, DOC_AUTHOR)
        self.blank = self.prs.slide_layouts[6]
        self.meta = []          # (slide, key, kind) with kind in {"plain", "main", "backup"}
        self.late = []          # (text box, slide key): boxes that show the number of another slide, filled in by finish()
        self.key = None

    def new(self, key, title=None, section=None, kind="main", kicker=None):
        sl = self.prs.slides.add_slide(self.blank)
        self.key = key
        self.meta.append((sl, key, kind))
        sl.background.fill.solid(); sl.background.fill.fore_color.rgb = rgb(BG)
        if kind == "plain":
            return sl
        if kicker is None and section:
            kicker = f"{section} of 6   ·   {SECTIONS[section - 1]}"
        if kicker:
            self.tx(sl, MX, 0.36, CW, 0.3, kicker.upper(), size=11, color=MUTED, bold=True, spacing=60)
        if title:
            size = 28
            if n_lines(title, size, CW, True) > 2:
                size = 26
            if n_lines(title, size, CW, True) > 2:
                WARNINGS.append(f"{key}: title needs more than two lines")
            self.tx(sl, MX, 0.68, CW, 1.1, title, size=size, color=INK, bold=True, line=1.02)
        return sl

    # text ---------------------------------------------------------------
    def tx(self, sl, x, y, w, h, paras, size=16, color=TEXT, bold=False, align="l", anchor="t", line=1.1, after=6,
           margin=0.0, italic=False, spacing=None, check=True):
        """A text box. paras is a string or a list of strings; **double stars** mark bold runs."""
        if isinstance(paras, str):
            paras = [paras]
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
        tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
            p.line_spacing = line
            p.space_after = Pt(after if i < len(paras) - 1 else 0)
            for txt, b in parse_runs(para):
                r = p.add_run(); r.text = txt
                f = r.font; f.name = FONT; f.size = Pt(size); f.bold = bool(bold or b); f.italic = italic; f.color.rgb = rgb(color)
                if spacing:
                    r._r.get_or_add_rPr().set("spc", str(spacing))
        if check:
            need = text_h(paras, size, w - 2 * margin, bold, line, after) + 2 * margin
            if need > h * 1.03 + 0.02:
                WARNINGS.append(f"{self.key}: text needs {need:.2f} in, box has {h:.2f} in: {paras[0][:50]!r}")
            wide = max((text_w(word, size, bold or b) for para in paras for txt, b in parse_runs(para) for word in txt.split()), default=0)
            if wide > w - 2 * margin + 0.02:
                WARNINGS.append(f"{self.key}: a word is {wide:.2f} in wide, box has {w - 2 * margin:.2f} in: {paras[0][:50]!r}")
        return tb

    def bullets(self, sl, x, y, w, h, items, size=16, color=TEXT, after=9, line=1.1, dot=MID, indent=0.24):
        """A bulleted list with a hanging indent."""
        tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT; p.line_spacing = line
            p.space_after = Pt(after if i < len(items) - 1 else 0)
            for txt, b in parse_runs(item):
                r = p.add_run(); r.text = txt
                f = r.font; f.name = FONT; f.size = Pt(size); f.bold = b; f.color.rgb = rgb(color)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(indent)))); pPr.set("indent", str(-int(Inches(indent))))
            clr = pPr.makeelement(qn("a:buClr"), {}); srgb = clr.makeelement(qn("a:srgbClr"), {"val": dot}); clr.append(srgb)
            pPr.append(clr)
            pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": FONT}))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        need = text_h(items, size, w - indent, False, line, after)
        if need > h * 1.03 + 0.02:
            WARNINGS.append(f"{self.key}: bullets need {need:.2f} in, box has {h:.2f} in: {items[0][:50]!r}")
        return tb

    # shapes -------------------------------------------------------------
    def box(self, sl, x, y, w, h, fill=CARD, radius=0.1, line=None, line_w=0.75):
        sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        if radius:
            sh.adjustments[0] = min(0.5, radius / min(w, h))
        _flat(sh)
        if fill:
            sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill)
        else:
            sh.fill.background()
        if line:
            sh.line.color.rgb = rgb(line); sh.line.width = Pt(line_w)
        else:
            sh.line.fill.background()
        return sh

    def dot(self, sl, x, y, d, color, line=None):
        sh = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
        _flat(sh)
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(color)
        if line:
            sh.line.color.rgb = rgb(line); sh.line.width = Pt(1.25)
        else:
            sh.line.fill.background()
        return sh

    def line(self, sl, x1, y1, x2, y2, color=HAIR, width=1.0, arrow=False, dash=False, cap=False):
        ln = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        _flat(ln)
        ln.line.color.rgb = rgb(color); ln.line.width = Pt(width)
        el = ln.line._get_or_add_ln()
        if cap:
            el.set("cap", "rnd")
        if dash:
            el.append(el.makeelement(qn("a:prstDash"), {"val": "dash"}))
        if arrow:
            el.append(el.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
        return ln

    # pictograms ---------------------------------------------------------
    def col(self, sl, x, y, w, h, fill):
        """A column of a small bar chart drawn with shapes: rounded top, square base."""
        sh = sl.shapes.add_shape(MSO_SHAPE.ROUND_2_SAME_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.adjustments[0] = min(0.5, 0.06 / min(w, h)); sh.adjustments[1] = 0
        _flat(sh)
        sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.fill.background()
        return sh

    def doc(self, sl, x, y, w, h, rows=4, mark=(), color=MUTED, fill=WHITE):
        """A page: an outlined sheet with text lines. The lines listed in mark are drawn in the accent color."""
        self.box(sl, x, y, w, h, fill=fill, radius=0.05, line=color, line_w=1.25)
        for i in range(rows):
            yy = y + h * (i + 1) / (rows + 1)
            short = w * 0.22 if i == rows - 1 else 0
            self.line(sl, x + w * 0.2, yy, x + w * 0.8 - short, yy, color=ACC if i in mark else color, width=2.5 if i in mark else 1.5, cap=True)

    def bars(self, sl, x, y, w, n=3, h=0.1, gap=0.08, color=STRONG, widths=None):
        """Stacked short bars: retrieved excerpts, or the claims of a summary."""
        for i in range(n):
            self.box(sl, x, y + i * (h + gap), w * (widths[i] if widths else 1), h, fill=color, radius=h / 2)

    def node(self, sl, x, y, w, h, label, fill=STRONG, color=WHITE, size=12):
        """A model in a diagram: a filled box with a short label."""
        self.box(sl, x, y, w, h, fill=fill, radius=0.08)
        self.tx(sl, x, y, w, h, label, size=size, color=color, bold=True, align="c", anchor="m", check=False)

    def check(self, sl, x, y, s, color=STRONG, width=2.25):
        """A check mark that fits a square of side s."""
        self.line(sl, x + 0.16 * s, y + 0.55 * s, x + 0.4 * s, y + 0.8 * s, color=color, width=width, cap=True)
        self.line(sl, x + 0.4 * s, y + 0.8 * s, x + 0.86 * s, y + 0.22 * s, color=color, width=width, cap=True)

    def cross(self, sl, x, y, s, color=ACC, width=2.25):
        """A cross that fits a square of side s."""
        self.line(sl, x, y, x + s, y + s, color=color, width=width, cap=True)
        self.line(sl, x + s, y, x, y + s, color=color, width=width, cap=True)

    def squares(self, sl, x, y, kinds, per_row=10, s=0.2, gap=0.05, line_w=1.25):
        """An icon array: one small square per item. kinds gives a (fill, outline) pair for each item."""
        for i, (fill, edge) in enumerate(kinds):
            self.box(sl, x + (i % per_row) * (s + gap), y + (i // per_row) * (s + gap), s, s, fill=fill, radius=min(0.07, s / 5), line=edge, line_w=line_w)

    def meter(self, sl, x, y, w, share, h=0.14, fill=STRONG, track=TINT2):
        """A horizontal meter: a track with a filled share."""
        self.box(sl, x, y, w, h, fill=track, radius=h / 2)
        if share > 0:
            self.box(sl, x, y, max(h, w * share), h, fill=fill, radius=h / 2)

    def pill(self, sl, x, y, label, size=13, h=0.38, fill=CARD, color=INK, line="C3C7CD", bold=False, w=None):
        """A capsule with a short label. Returns its width."""
        w = w or text_w(label, size, bold) + 0.36 * max(1, size / 13)
        self.box(sl, x, y, w, h, fill=fill, radius=h / 2, line=line)
        self.tx(sl, x, y, w, h, label, size=size, color=color, bold=bold, align="c", anchor="m", check=False)
        return w

    def num(self, sl, x, y, n, d=0.44, fill=SLATE, color=WHITE, size=15):
        """A numbered circle."""
        self.dot(sl, x, y, d, fill)
        self.tx(sl, x, y, d, d, str(n), size=size, color=color, bold=True, align="c", anchor="m", check=False)

    def chip(self, sl, x, y, cond, label=None, w=None, size=14, h=0.36):
        """Condition tag: a white capsule with a dot in the condition's color (a ring for a reference) and its name in ink."""
        label = label or cond
        w = w or (0.44 + text_w(label, size, True) + 0.16)
        self.box(sl, x, y, w, h, fill=CARD, radius=h / 2, line="C3C7CD")
        self.dot(sl, x + 0.12, y + (h - 0.16) / 2, 0.16, CCOL[cond], line=EDGE if cond in HOLLOW else None)
        self.tx(sl, x + 0.36, y, w - 0.4, h, label, size=size, color=INK, bold=True, anchor="m", check=False)
        return w

    @staticmethod
    def tag_w(label, size=12):
        return text_w(label, size, True) + 0.3 * max(1, size / 12)

    def tag(self, sl, x, y, label, color, size=12, h=0.3, text=WHITE, line=None):
        """Status tag: a capsule with a text label (the label carries the meaning, fill and outline support it)."""
        w = self.tag_w(label, size)
        self.box(sl, x, y, w, h, fill=color, radius=h / 2, line=line, line_w=1.25)
        self.tx(sl, x, y, w, h, label, size=size, color=text, bold=True, align="c", anchor="m", check=False)
        return w

    def stat(self, sl, x, y, w, value, label, size=40, label_size=14, h=1.25, color=INK):
        """A headline number with a short label under it."""
        self.tx(sl, x, y, w, size * 1.25 / 72, value, size=size, color=color, bold=True, check=False)
        self.tx(sl, x, y + size * 1.25 / 72 + 0.02, w, h - size * 1.25 / 72, label, size=label_size, color=MUTED, line=1.08)

    def chart(self, sl, path, x, y, w, h, pad=0.18):
        """A chart on a white card, as large as the card allows."""
        self.box(sl, x, y, w, h, fill=CARD)
        return self.pic(sl, path, x + pad, y + pad, w - 2 * pad, h - 2 * pad)

    def pic(self, sl, path, x, y, w=None, h=None):
        if w is not None and h is not None:
            from PIL import Image
            with Image.open(path) as im:
                iw, ih = im.size
            sc = min(w / iw, h / ih)
            pw, ph = iw * sc, ih * sc
            return sl.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
        return sl.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)

    # tables -------------------------------------------------------------
    def table(self, sl, x, y, w, columns, rows, widths, size=14, head_size=None, align=None, pad=0.07, bold_col0=False,
              highlight=(), max_h=None, zebra=False):
        """A native table with hairlines between rows. Returns the estimated bottom edge in inches."""
        head_size = head_size or size
        ncol = len(columns)
        total = float(sum(widths))
        cw = [w * v / total for v in widths]
        align = align or ["l"] + ["l"] * (ncol - 1)

        heights = table_heights(columns, rows, cw, size, head_size, pad, bold_col0)
        gf = sl.shapes.add_table(len(rows) + 1, ncol, Inches(x), Inches(y), Inches(w), Inches(sum(heights)))
        tbl = gf.table
        tblPr = tbl._tbl.tblPr
        tblPr.set("bandRow", "0"); tblPr.set("firstRow", "1")
        sid = tblPr.find(qn("a:tableStyleId"))
        if sid is not None:
            sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"      # no style, no grid
        for j in range(ncol):
            tbl.columns[j].width = Inches(cw[j])

        def put(cell, text, s, bold, fill, al, color=TEXT):
            cell.margin_left = cell.margin_right = Inches(pad); cell.margin_top = cell.margin_bottom = Inches(pad * 0.75)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[al]
            for txt, b in parse_runs(str(text) if str(text).strip() else " "):
                r = p.add_run(); r.text = txt
                f = r.font; f.name = FONT; f.size = Pt(s); f.bold = bool(bold or b); f.color.rgb = rgb(color)
            cell.fill.solid(); cell.fill.fore_color.rgb = rgb(fill)
            tcPr = cell._tc.get_or_add_tcPr()
            for k, side in enumerate(("a:lnL", "a:lnR", "a:lnT")):
                e = tcPr.makeelement(qn(side), {"w": "0"}); e.append(e.makeelement(qn("a:noFill"), {})); tcPr.insert(k, e)
            e = tcPr.makeelement(qn("a:lnB"), {"w": "6350"})
            sf = e.makeelement(qn("a:solidFill"), {}); sf.append(sf.makeelement(qn("a:srgbClr"), {"val": HAIR})); e.append(sf)
            tcPr.insert(3, e)
        for j, c in enumerate(columns):
            put(tbl.cell(0, j), c, head_size, True, TINT2, align[j], INK)
        for i, r in enumerate(rows, 1):
            fill = ROW if (i - 1) in highlight else (TINT if zebra and i % 2 == 0 else WHITE)
            for j, v in enumerate(r):
                put(tbl.cell(i, j), v, size, bold_col0 and j == 0, fill, align[j])
        for i, hgt in enumerate(heights):
            tbl.rows[i].height = Inches(hgt)
        bottom = y + sum(heights)
        if bottom > (max_h or (BOTTOM + 0.1)):
            WARNINGS.append(f"{self.key}: table ends at {bottom:.2f} in")
        return bottom

    # footers and notes ---------------------------------------------------
    def add_notes(self, notes):
        """Speaker notes as short bullet points, one paragraph per point. The bullet is a plain character, so that
        PowerPoint, Keynote and Google Slides all show it the same way."""
        for sl, key, _ in self.meta:
            if notes.get(key):
                sl.notes_slide.notes_text_frame.text = "\n".join("•  " + b.strip() for b in notes[key])

    def reorder(self, keys):
        """Put the slides of the talk in the order of keys. Every slide of the talk must be named exactly once."""
        have = [k for _, k, kind in self.meta if kind != "backup"]
        if sorted(have) != sorted(keys):
            raise SystemExit(f"slide order does not match the slides: missing {sorted(set(have) - set(keys))}, unknown {sorted(set(keys) - set(have))}")
        pos = {k: i for i, (_, k, _) in enumerate(self.meta)}
        order = [pos[k] for k in keys] + [i for i, m in enumerate(self.meta) if m[2] == "backup"]
        lst = self.prs.slides._sldIdLst
        ids = list(lst)
        for el in ids:
            lst.remove(el)
        for i in order:
            lst.append(ids[i])
        self.meta = [self.meta[i] for i in order]

    def finish(self):
        """Footers with the real slide number, so that typing it during the questions jumps to the slide."""
        mains = [m for m in self.meta if m[2] in ("main", "plain")]
        pos = {key: i for i, (_, key, _) in enumerate(self.meta, 1)}
        for tb, key in self.late:
            tb.text_frame.paragraphs[0].runs[0].text = f"Backup slide {pos[key]}"
        nb = 0
        for i, (sl, key, kind) in enumerate(self.meta, 1):
            if kind == "backup":
                nb += 1
            label = f"Backup slide {i}" if kind == "backup" else f"{i} / {len(mains)}"
            if kind != "plain":
                self.key = key
                self.tx(sl, MX, 7.04, 9.5, 0.26, "Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization   ·   Srilaya Ponangi",
                        size=10, color=FAINT, check=False)
                self.tx(sl, SW - MX - 2.0, 7.04, 2.0, 0.26, label, size=10, color=FAINT, align="r", check=False)
        return len(mains), nb


# ───────────────────────────── charts ─────────────────────────────
C_INK, C_SEC, C_MUTED, C_GRID, C_AXIS = "#2f3237", "#6b7078", "#9499a1", "#e6e8eb", "#c3c7cd"
MODE = {"top3": "top 3 sentences", "doc": "whole course"}
JSHORT = {"minilm_nli": "MiniLM cross-encoder (pilot judge)", "ce_deberta_large_nli": "DeBERTa-v3-large cross-encoder",
          "mednli_deberta_large": "DeBERTa-v3-large NLI + MedNLI"}


def h(c):
    return "#" + c


def new_fig(w, hgt, rect):
    """A figure of exactly w by hgt inches with one axes at rect = [left, bottom, width, height] in figure fractions."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": C_AXIS, "axes.linewidth": 0.8, "xtick.color": C_SEC, "ytick.color": C_SEC,
                         "axes.labelcolor": C_SEC, "text.color": C_INK, "legend.frameon": False, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(w, hgt))
    ax = fig.add_axes(rect)
    return fig, ax


def quiet(ax, value_axis="y"):
    """Recessive frame: hairline grid on the value axis, one baseline, no tick marks."""
    ax.set_axisbelow(True)
    if value_axis == "y":
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.8); ax.spines["left"].set_visible(False)
    else:
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8); ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0, pad=6)


def units(ax):
    """Data units per point on the x and y axes."""
    fig = ax.figure
    fw, fh = fig.get_size_inches()
    pos = ax.get_position()
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / (pos.width * fw * 72), abs(y1 - y0) / (pos.height * fh * 72)


def vbar(ax, x, v, w, color, r=4, edge=None):
    """A column with a 4 pt rounded top and a square base; with an edge color it is drawn as an outline. Call after the axis limits are set."""
    ux, uy = units(ax)
    rx, ry, k = min(r * ux, w / 2), min(r * uy, v / 2), 0.5523
    x0, x1 = x - w / 2, x + w / 2
    verts = [(x0, 0), (x0, v - ry), (x0, v - ry + k * ry), (x0 + rx - k * rx, v), (x0 + rx, v), (x1 - rx, v),
             (x1 - rx + k * rx, v), (x1, v - ry + k * ry), (x1, v - ry), (x1, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor=edge or "none", linewidth=1.3 if edge else 0, joinstyle="round"))


def whisker(ax, x, m, lo, hi, fill):
    """A 95 percent interval: dark above the column, and white inside a dark column so that the lower half stays visible."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    inside = "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.4 else C_INK
    cap = 3.5 * units(ax)[0]
    ax.plot([x, x], [lo, m], color=inside, linewidth=1.1, solid_capstyle="butt")
    ax.plot([x - cap, x + cap], [lo, lo], color=inside, linewidth=1.1, solid_capstyle="butt")
    ax.plot([x, x], [m, hi], color=C_INK, linewidth=1.1, solid_capstyle="butt")
    ax.plot([x - cap, x + cap], [hi, hi], color=C_INK, linewidth=1.1, solid_capstyle="butt")


def hbar(ax, y, v, t, color, r=4):
    """A horizontal bar with a 4 pt rounded end and a square base."""
    ux, uy = units(ax)
    if v <= 2.2 * r * ux:
        ax.add_patch(Rectangle((0, y - t / 2), v, t, facecolor=color, edgecolor="none")); return
    ax.add_patch(FancyBboxPatch((0, y - t / 2), v, t, boxstyle=f"round,pad=0,rounding_size={r * ux}", mutation_aspect=uy / ux,
                                facecolor=color, edgecolor="none"))
    ax.add_patch(Rectangle((0, y - t / 2), v / 2, t, facecolor=color, edgecolor="none"))


def save(fig, name):
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / name
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def chart_pilot(R):
    """Share of sentences flagged by the experts and by the pilot judge, by group of summaries."""
    groups = [("all", "All sentences"), ("doctor_written", "Doctor-written"), ("generated", "LLM-generated"), ("gpt4_zero_shot", "GPT-4 zero-shot")]
    cal = R.cal.set_index("group")
    fig, ax = new_fig(7.4, 4.4, [0.115, 0.17, 0.865, 0.70])
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(0, 100)
    quiet(ax)
    w = 0.26
    for i, (g, lab) in enumerate(groups):
        e, j = 100 * cal.loc[g, "expert_flag_rate"], 100 * cal.loc[g, "judge_UFR"]
        vbar(ax, i - w / 2 - 0.02, e, w, h(HUMAN)); vbar(ax, i + w / 2 + 0.02, j, w, h(JUDGE))
        ax.text(i - w / 2 - 0.02, e + 2, f"{e:.1f}%", ha="center", va="bottom", fontsize=12, color=C_INK)
        ax.text(i + w / 2 + 0.02, j + 2, f"{j:.1f}%", ha="center", va="bottom", fontsize=12, color=C_INK, fontweight="bold")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\n({int(cal.loc[g, 'n_sentences']):,} sentences)" for g, lab in groups], fontsize=12, color=C_INK)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=11.5)
    ax.set_ylabel("Share of sentences flagged as unsupported", fontsize=12)
    hs = [Rectangle((0, 0), 1, 1, color=h(HUMAN)), Rectangle((0, 0), 1, 1, color=h(JUDGE))]
    fig.legend(hs, ["Medical experts", "Pilot judge (MiniLM NLI)"], loc="upper left", bbox_to_anchor=(0.105, 1.0), ncol=2, fontsize=12,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    return save(fig, "pilot_judge.png")


def chart_judges():
    """Cohen's kappa of every candidate judge and evidence mode against the expert annotations."""
    jc = csv("judge_candidates.csv")
    a = jc[jc.group == "all"].sort_values("kappa", ascending=True).reset_index(drop=True)
    best = a.kappa.idxmax()
    fig, ax = new_fig(7.6, 4.75, [0.515, 0.115, 0.46, 0.87])
    ax.set_xlim(0, 0.40); ax.set_ylim(-0.6, len(a) - 0.4)
    quiet(ax, "x")
    labels = []
    for i, r in a.iterrows():
        name = JSHORT.get(r.judge, JUDGE_NAMES.get(r.judge, r.judge))
        labels.append(f"{name}  ·  {MODE.get(r['mode'], r['mode'])}")
        sel = i == best
        hbar(ax, i, r.kappa, 0.62, h(ACC) if sel else h(JUDGE))
        ax.text(r.kappa + 0.006, i, f"{r.kappa:.2f}", va="center", ha="left", fontsize=11.5 if sel else 11, color=C_INK if sel else C_SEC,
                fontweight="bold" if sel else "normal")
    ax.set_yticks(range(len(a))); ax.set_yticklabels(labels, fontsize=10.8, color=C_INK)
    ax.get_yticklabels()[best].set_fontweight("bold")
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_xticklabels(["0", "0.10", "0.20", "0.30", "0.40"], fontsize=11)
    ax.set_xlabel("Cohen's kappa against the medical experts (1,781 sentences)", fontsize=11.5, labelpad=6)
    ax.xaxis.set_label_coords(0.10, -0.085)
    return save(fig, "judge_selection.png")


XT = {"E0": "E0\nfull note", "E1": "E1\nRAG,\nexcerpts", "E1b": "E1b\nRAG +\ncourse", "E3": "E3\nRAG +\nverification", "E2": "E2\nextractive",
      "REF": "Clinician's\ninstructions"}


def _cond_bars(ax, D, conds, metric, scale, fmt, w=0.5, pad=0.0):
    """Bars with 95 percent intervals for the listed conditions; a gap separates the references from the LLM conditions."""
    xs, x = {}, 0.0
    for c in conds:
        if c in ("E2", "REF") and "gap" not in xs:
            x += 0.45; xs["gap"] = x - 0.725
        xs[c] = x; x += 1.0
    for c in conds:
        r = D[(D.condition == c) & (D.variant == "base") & (D.metric == metric)].iloc[0]
        m, lo, hi = scale * r["mean"], scale * r.ci95_low, scale * r.ci95_high
        vbar(ax, xs[c], m, w, h(CCOL[c]), edge=h(EDGE) if c in HOLLOW else None)
        whisker(ax, xs[c], m, lo, hi, h(CCOL[c]))
        ax.text(xs[c], hi + pad, fmt(m), ha="center", va="bottom", fontsize=12.5, color=C_INK, fontweight="bold")
    return xs


def chart_ufr(conds=("E0", "E1", "E1b", "E3", "E2", "REF"), name="main_ufr.png"):
    """The judge-estimated unsupported fact rate by condition. Without the two references the scale is finer."""
    D = csv("mimic_condition_ci.csv")
    refs = any(c in HOLLOW for c in conds)
    top = 0.40 if refs else 0.20
    fig, ax = new_fig(7.4, 4.5, [0.105, 0.20, 0.875, 0.76])
    ax.set_xlim(-0.6, len(conds) - 1 + (0.45 if refs else 0) + 0.6); ax.set_ylim(0, top)
    quiet(ax)
    xs = _cond_bars(ax, D, conds, "UFR", 1.0, lambda v: f"{v:.3f}", pad=top * 0.015)
    ax.set_xticks([xs[c] for c in conds]); ax.set_xticklabels([XT[c] for c in conds], fontsize=11.5, color=C_INK, linespacing=1.15)
    ticks = [0, 0.1, 0.2, 0.3] if refs else [0, 0.05, 0.10, 0.15, 0.20]
    ax.set_yticks(ticks); ax.set_yticklabels(["0"] + [f"{t:.2f}" for t in ticks[1:]], fontsize=11.5)
    ax.set_ylabel("Unsupported fact rate (judge-estimated)", fontsize=12)
    if refs:
        ax.axvline(xs["gap"], color=C_GRID, linewidth=1.0)
        ax.text(xs["gap"] + 0.12, 0.395, "References, not bounds", fontsize=11.5, color=C_SEC, va="top", ha="left")
        ax.text(-0.5, 0.395, "LLM conditions", fontsize=11.5, color=C_SEC, va="top", ha="left")
    return save(fig, name)


def chart_coverage(conds=("E0", "E1", "E1b", "E3", "E2")):
    D = csv("mimic_condition_ci.csv")
    fig, ax = new_fig(7.4, 4.5, [0.105, 0.20, 0.875, 0.76])
    ax.set_xlim(-0.6, len(conds) - 1 + 0.45 + 0.6); ax.set_ylim(0, 70)
    quiet(ax)
    xs = _cond_bars(ax, D, conds, "coverage_ref", 100.0, lambda v: f"{v:.1f}%", pad=1.2)
    ax.set_xticks([xs[c] for c in conds]); ax.set_xticklabels([XT[c] for c in conds], fontsize=11.5, color=C_INK, linespacing=1.15)
    ax.set_yticks([0, 20, 40, 60]); ax.set_yticklabels(["0", "20%", "40%", "60%"], fontsize=11.5)
    ax.set_ylabel("Clinician's sentences covered (judge-estimated)", fontsize=12)
    ax.axvline(xs["gap"], color=C_GRID, linewidth=1.0)
    ax.text(xs["gap"] + 0.12, 68.5, "Reference", fontsize=11.5, color=C_SEC, va="top", ha="left")
    return save(fig, "main_coverage.png")


def variant_rows():
    """The six comparisons that separate length and the citation instruction from retrieval, as (label, delta, lo, hi, Holm p)."""
    vt = csv("mimic_variant_tests.csv")
    u = vt[vt.metric == "UFR"].copy()
    u["ph"] = holm(u.p.values)
    u = u.set_index("comparison")
    spec = [("E0 limited to about 150 words vs E0", "E0/short vs E0/base", 1), ("E1 vs length-matched E0", "E0/short vs E1/base", -1),
            ("E1b vs length-matched E0", "E1b/base vs E0/short", 1), ("E3 vs length-matched E0", "E3/base vs E0/short", 1),
            ("E1 without the citation instruction vs E0", "E1/nocite vs E0/base", 1), ("E1b without the citation instruction vs E1b", "E1b/nocite vs E1b/base", 1)]
    rows = []
    for lab, key, sign in spec:
        r = u.loc[key]
        d, lo, hi = sign * r.delta_mean, sign * r.ci95_low, sign * r.ci95_high
        rows.append((lab, d, min(lo, hi), max(lo, hi), float(r.ph)))
    return rows


def chart_forest(part):
    """Paired differences in UFR with 95 percent intervals: 'length' for the four comparisons around the length-matched
    baseline, 'cite' for the two comparisons without the citation instruction."""
    rows = variant_rows()
    rows = rows[:4] if part == "length" else rows[4:]
    k, fw, left, plot = len(rows), 10.9, 4.2, 4.2
    fh = 0.55 + 0.62 * k + 0.85
    fig, ax = new_fig(fw, fh, [left / fw, 0.85 / fh, plot / fw, 0.62 * k / fh])
    ax.set_xlim(-0.10, 0.07); ax.set_ylim(-0.5, k - 0.5)
    quiet(ax, "x")
    ax.axvline(0, color=C_AXIS, linewidth=1.2)
    per_inch = 0.17 / plot
    xd, xp = 0.07 + 0.35 * per_inch, 0.07 + 1.6 * per_inch
    for i, (lab, d_, lo, hi, ph) in enumerate(rows):
        y = k - 1 - i
        ax.plot([lo, hi], [y, y], color=C_INK, linewidth=1.8, solid_capstyle="round")
        ax.plot([d_], [y], marker="o", markersize=10, markerfacecolor=C_INK, markeredgecolor="white", markeredgewidth=1.8)
        ax.text(xd, y, f"{d_:+.3f}", va="center", ha="left", fontsize=13, color=C_INK, clip_on=False, fontweight="bold")
        ax.text(xp, y, fp(ph), va="center", ha="left", fontsize=13, color=C_SEC, clip_on=False)
    ax.set_yticks(range(k)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=13, color=C_INK)
    ax.set_xticks([-0.10, -0.05, 0, 0.05]); ax.set_xticklabels(["-0.10", "-0.05", "0", "+0.05"], fontsize=11.5)
    ax.set_xlabel("Paired difference in judge-estimated UFR (95% CI)", fontsize=12, labelpad=6)
    ax.text(xd, k - 0.42, "Difference", fontsize=11.5, color=C_SEC, clip_on=False, va="bottom")
    ax.text(xp, k - 0.42, "Holm p", fontsize=11.5, color=C_SEC, clip_on=False, va="bottom")
    return save(fig, f"variants_{part}.png")


GEN_FILES = [("Qwen2.5-7B (main study)", ""), ("Llama-3.1-8B", "_llama31"), ("Mistral-7B", "_mistral7b")]


def generator_runs():
    """[(name, {condition: UFR mean}, [conditions below E0 after Holm adjustment])] for the three generators."""
    out = []
    for name, tag in GEN_FILES:
        s, t = csv(f"mimic_summary{tag}.csv"), csv(f"mimic_pairwise_tests{tag}.csv")
        if s is None or t is None:
            continue
        b = s[s.variant == "base"].set_index("condition")
        sig = []
        for c in ("E1", "E1b", "E3"):
            r = t[(t.metric == "UFR") & (t.comparison == f"{c} vs E0")].iloc[0]
            if float(r.p_holm_primary) < 0.05 and r.delta_mean < 0:
                sig.append(c)
        out.append((name, {c: float(b.loc[c, "UFR_mean"]) for c in ("E0", "E1", "E1b", "E3")}, sig))
    return out


def chart_generators():
    runs = generator_runs()
    plt.rcParams.update({"font.family": "Arial"})
    fig = plt.figure(figsize=(8.3, 4.4))
    for k, (name, means, sig) in enumerate(runs):
        ax = fig.add_axes([0.085 + k * 0.31, 0.12, 0.275, 0.66])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C_AXIS); ax.spines["bottom"].set_linewidth(0.8)
        ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 0.20)
        quiet(ax)
        for i, c in enumerate(("E0", "E1", "E1b", "E3")):
            vbar(ax, i, means[c], 0.56, h(CCOL[c]))
            ax.text(i, means[c] + 0.004, f"{means[c]:.3f}", ha="center", va="bottom", fontsize=11.5, color=C_INK, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(["E0", "E1", "E1b", "E3"], fontsize=12, color=C_INK)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_yticklabels(["0", "0.05", "0.10", "0.15", "0.20"] if k == 0 else [""] * 5, fontsize=11, color=C_SEC)
        if k == 0:
            ax.set_ylabel("Unsupported fact rate (judge-estimated)", fontsize=11.5, color=C_SEC)
        ax.text(-0.6, 0.238, name, fontsize=13, fontweight="bold", color=C_INK, va="bottom", clip_on=False)
        verdict = ("E1, E1b, E3 below E0" if len(sig) == 3 else (", ".join(sig) + " below E0" if sig else "No difference from E0"))
        ax.text(-0.6, 0.221, verdict, fontsize=11, color=C_SEC, va="bottom", clip_on=False)
    return save(fig, "generators.png")


def chart_audit():
    aud = csv("audit_summary.csv").set_index("condition")
    conds = ["E0", "E1", "E1b", "E3"]
    fig, ax = new_fig(7.4, 4.5, [0.09, 0.19, 0.89, 0.69])
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 40)
    quiet(ax)
    w = 0.3
    for i, c in enumerate(conds):
        m, lo, hi = (100 * aud.loc[c, k] for k in ("manual_unsupported", "manual_unsupported_lo", "manual_unsupported_hi"))
        j = 100 * aud.loc[c, "judge_flag_share_pipeline"]
        vbar(ax, i - w / 2 - 0.02, m, w, h(HUMAN)); vbar(ax, i + w / 2 + 0.02, j, w, h(JUDGE))
        whisker(ax, i - w / 2 - 0.02, m, lo, hi, h(HUMAN))
        ax.text(i - w / 2 - 0.02, hi + 0.8, f"{m:.1f}%", ha="center", va="bottom", fontsize=12, color=C_INK, fontweight="bold")
        ax.text(i + w / 2 + 0.02, j + 0.8, f"{j:.1f}%", ha="center", va="bottom", fontsize=12, color=C_INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([XT[c].replace("RAG +\nverification", "RAG + verification") for c in conds], fontsize=11.5, color=C_INK, linespacing=1.15)
    ax.set_yticks([0, 10, 20, 30, 40]); ax.set_yticklabels(["0", "10%", "20%", "30%", "40%"], fontsize=11.5)
    ax.set_ylabel("Share of claims unsupported", fontsize=12)
    hs = [Rectangle((0, 0), 1, 1, color=h(HUMAN)), Rectangle((0, 0), 1, 1, color=h(JUDGE))]
    fig.legend(hs, ["Manual reading, weighted (95% CI)", "Flagged by the judge"], loc="upper left", bbox_to_anchor=(0.085, 1.0), ncol=2, fontsize=12,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    return save(fig, "audit.png")


def chart_tradeoff():
    D = csv("mimic_condition_ci.csv")
    D = D[D.variant == "base"]
    def g(c, metric):
        r = D[(D.condition == c) & (D.metric == metric)].iloc[0]
        return float(r["mean"]), float(r.ci95_low), float(r.ci95_high)
    fig, ax = new_fig(6.5, 4.6, [0.125, 0.145, 0.85, 0.83])
    ax.set_xlim(0.03, 0.17); ax.set_ylim(15, 65)
    ax.set_axisbelow(True); ax.grid(True, color=C_GRID, linewidth=0.8); ax.tick_params(length=0, pad=6)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_AXIS)
    place = {"E0": (0.004, 2.6, "left"), "E1": (0.004, 2.6, "left"), "E1b": (0.0, 6.2, "center"), "E3": (0.0045, -3.4, "left"), "E2": (0.004, 2.6, "left")}
    for c in ("E0", "E1", "E3", "E2", "E1b"):
        (x, xl, xh), (y, yl, yh) = g(c, "UFR"), g(c, "coverage_ref")
        y, yl, yh = 100 * y, 100 * yl, 100 * yh
        ax.plot([xl, xh], [y, y], color=C_AXIS, linewidth=1.0); ax.plot([x, x], [yl, yh], color=C_AXIS, linewidth=1.0)
        hi = c == "E1b"
        hollow = c in HOLLOW
        ax.plot([x], [y], marker="o", markersize=13 if hi else 10, markerfacecolor=h(CCOL[c]), markeredgecolor=h(EDGE) if hollow or c == "E3" else "white",
                markeredgewidth=1.4 if hollow or c == "E3" else 2)
        dx, dy, ha = place[c]
        ax.text(x + dx, y + dy, CNAME[c], fontsize=12.5 if hi else 11.5, fontweight="bold" if hi else "normal", color=C_INK, ha=ha, va="center")
    ax.annotate("", xy=(0.126, 41.5), xytext=(0.150, 31.0), arrowprops=dict(arrowstyle="-|>", color=C_SEC, linewidth=1.2))
    ax.text(0.120, 29.0, "Better: fewer unsupported\nclaims and less omitted", fontsize=11, color=C_SEC, ha="left", va="top", linespacing=1.25)
    ax.set_xticks([0.04, 0.08, 0.12, 0.16]); ax.set_xticklabels(["0.04", "0.08", "0.12", "0.16"], fontsize=11)
    ax.set_yticks([20, 30, 40, 50, 60]); ax.set_yticklabels(["20%", "30%", "40%", "50%", "60%"], fontsize=11)
    ax.set_xlabel("Unsupported fact rate (judge-estimated)", fontsize=12, labelpad=6)
    ax.set_ylabel("Clinician's sentences covered", fontsize=12)
    return save(fig, "tradeoff.png")


def chart_robust(rows):
    """The unsupported fact rate of the four LLM conditions under each analysis: one row per analysis, one marker per condition."""
    k = len(rows)
    fig, ax = new_fig(7.7, 4.6, [0.445, 0.125, 0.535, 0.755])
    ax.set_xlim(0, 0.30); ax.set_ylim(-0.6, k - 0.4)
    quiet(ax, "x")
    ax.axhspan(k - 1.46, k - 0.54, color="#eef0f2", zorder=0)                 # the main analysis
    for i, (lab, v) in enumerate(rows):
        y = k - 1 - i
        lo = min(v[c] for c in ("E1", "E1b", "E3"))
        ax.plot([lo, v["E0"]], [y, y], color=C_AXIS, linewidth=1.4, zorder=1)
        for c in ("E3", "E1", "E1b", "E0"):
            ax.plot([v[c]], [y], marker="o", markersize=11.5, markerfacecolor=h(CCOL[c]), markeredgecolor=h(EDGE) if c == "E3" else "white",
                    markeredgewidth=1.1 if c == "E3" else 1.6, zorder=3 if c in ("E0", "E1b") else 2)
        ax.text(v["E0"] + 0.011, y, f"{v['E0']:.3f}", va="center", ha="left", fontsize=10.5, color=C_INK)
        if lo > 0.04:
            ax.text(lo - 0.011, y, f"{lo:.3f}", va="center", ha="right", fontsize=10.5, color=C_SEC)
    ax.set_yticks(range(k)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=11.5, color=C_INK)
    ax.get_yticklabels()[k - 1].set_fontweight("bold")
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]); ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"], fontsize=11)
    ax.set_xlabel("Unsupported fact rate (judge-estimated)", fontsize=11.5, labelpad=6)
    hs = [plt.Line2D([], [], marker="o", linestyle="", markersize=10.5, markerfacecolor=h(CCOL[c]), markeredgecolor=h(EDGE) if c == "E3" else "white",
                     markeredgewidth=1.1 if c == "E3" else 1.4) for c in ("E0", "E1", "E1b", "E3")]
    fig.legend(hs, ["E0 full note", "E1 excerpts", "E1b excerpts + course", "E3 verification"], loc="upper left", bbox_to_anchor=(0.19, 1.0), ncol=4, fontsize=11,
               handletextpad=0.1, columnspacing=1.3)
    return save(fig, "robustness.png")


def waffle_counts(r):
    """Per 100 claims of a condition: flagged and unsupported, flagged but supported, accepted but unsupported, accepted and supported."""
    f_, p_, pi_ = float(r.judge_flag_share_pipeline), float(r.precision), float(r.manual_unsupported)
    flagged, hit, true = round(100 * f_), round(100 * f_ * p_), round(100 * pi_)
    return hit, flagged - hit, true - hit, 100 - flagged - (true - hit)


def chart_waffle(aud, conds=("E0", "E1b"), name="audit_waffle.png"):
    """One hundred claims of each condition as squares: what the judge flags, and what manual reading finds."""
    kinds = [(h(ACC), h(ACC)), ("#ffffff", h(ACC)), (h(STRONG), h(STRONG)), ("#e4e6ea", "#e4e6ea")]
    k = len(conds)
    fig = plt.figure(figsize=(3.3 * k, 3.5))
    for j, c in enumerate(conds):
        ax = fig.add_axes([(j + 0.012) / k, 0.01, 0.976 / k, 0.86])
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.1, 10.1); ax.set_ylim(-0.1, 10.1)
        cells = [kinds[i] for i, m in enumerate(waffle_counts(aud.loc[c])) for _ in range(m)]
        for i, (fc, ec) in enumerate(cells):                 # fill column by column from the left, each from the top
            ax.add_patch(FancyBboxPatch((i // 10 + 0.12, 9 - i % 10 + 0.12), 0.76, 0.76, boxstyle="round,pad=0,rounding_size=0.13", facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(0.1, 10.45, CNAME[c], fontsize=13, fontweight="bold", color=C_INK, va="bottom", clip_on=False, fontfamily="Arial")
    return save(fig, name)


def chart_existing(groups, name="existing.png"):
    """Cohen's kappa against clinicians' labels: existing approaches in gray, the selected judge in the accent color.
    groups is [(heading, [(label, kappa, value text, mine)])]."""
    k = sum(len(g[1]) for g in groups) + len(groups)
    fw, fh = 7.7, 0.5 * k + 0.85
    fig, ax = new_fig(fw, fh, [0.47, 0.72 / fh, 0.50, 0.5 * k / fh])
    ax.set_xlim(0, 0.60); ax.set_ylim(-0.6, k - 0.4)
    quiet(ax, "x")
    y, ticks, labels = k - 1, [], []
    for head, rows in groups:
        fig.text(0.012, ax.transData.transform((0, y))[1] / (fh * fig.dpi), head, fontsize=11.5, fontweight="bold", color=C_INK, va="center", ha="left", linespacing=1.15)
        y -= 1
        for lab, v, txt, mine in rows:
            hbar(ax, y, v, 0.6, h(ACC) if mine else h(JUDGE))
            ax.text(v + 0.01, y, txt, va="center", ha="left", fontsize=11.5, color=C_INK, fontweight="bold" if mine else "normal")
            ticks.append(y); labels.append(lab); y -= 1
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=10.8, color=C_INK, linespacing=1.12)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_xticklabels(["0", "0.10", "0.20", "0.30", "0.40", "0.50"], fontsize=11)
    ax.set_xlabel("Cohen's kappa against clinicians' labels", fontsize=11.5, labelpad=6)
    return save(fig, name)


def chart_audit_diff(aud):
    """Manual reading: the unsupported share of each grounded condition minus that of E0, with its 95 percent interval."""
    rows = [("E1", "E1 minus E0"), ("E1b", "E1b minus E0"), ("E3", "E3 minus E0")]
    k, fw, fh = len(rows), 8.2, 3.3
    fig, ax = new_fig(fw, fh, [0.215, 0.26, 0.50, 0.60])
    ax.set_xlim(-30, 16); ax.set_ylim(-0.5, k - 0.5)
    quiet(ax, "x")
    ax.axvline(0, color=C_AXIS, linewidth=1.2)
    per_inch = 46 / (0.50 * fw)
    xd, xc = 16 + 0.3 * per_inch, 16 + 1.05 * per_inch
    for i, (c, key) in enumerate(rows):
        y = k - 1 - i
        r = aud.loc[key]
        d_, lo, hi = 100 * r["diff"], 100 * r["diff_lo"], 100 * r["diff_hi"]
        col = h(ACC) if c == "E1b" else C_INK
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot([d_], [y], marker="o", markersize=11, markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.8)
        ax.text(xd, y, f"{d_:+.1f}", va="center", ha="left", fontsize=13, color=C_INK, fontweight="bold", clip_on=False)
        ax.text(xc, y, f"{lo:+.1f} to {hi:+.1f}", va="center", ha="left", fontsize=12, color=C_SEC, clip_on=False)
    ax.set_yticks(range(k)); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=13, color=C_INK)
    ax.get_yticklabels()[1].set_fontweight("bold")
    ax.set_xticks([-30, -20, -10, 0, 10]); ax.set_xticklabels(["-30", "-20", "-10", "0", "+10"], fontsize=11.5)
    ax.set_xlabel("Difference in the unsupported share by manual reading (points, 95% CI)", fontsize=12, labelpad=6)
    ax.xaxis.set_label_coords(0.42, -0.2)
    ax.text(xd, k - 0.42, "Points", fontsize=11.5, color=C_SEC, clip_on=False, va="bottom")
    ax.text(xc, k - 0.42, "95% CI", fontsize=11.5, color=C_SEC, clip_on=False, va="bottom")
    return save(fig, "audit_diff.png")


# ───────────────────────────── content ─────────────────────────────
def thesis_tables(R):
    """Every table of the thesis by its label, with its number, so the backup slides show exactly the thesis tables."""
    T, chap, app, k = {}, 0, None, 0
    for mod in ("ch1_4", "ch5_pilot", "ch_judge", "ch_mimic", "ch8_9", "appendices"):
        for b in importlib.import_module(mod).blocks(R):
            if b[0] == "h1":
                if b[1].startswith("Appendix"):
                    app = b[1].split()[1].rstrip(":")
                else:
                    chap += 1
                k = 0
            elif b[0] == "table":
                k += 1
                d = dict(b[1]); d["number"] = f"{app}.{k}" if app else f"{chap}.{k}"
                d["rows"] = [[re.sub(r"\s*\[\[cite:[^\]]+\]\]", "", str(v)) for v in r] for r in d.get("rows", [])]
                if d.get("label"):
                    T[d["label"]] = d
    return T


NOTES_DIR = ROOT.parent / "defense_notes"


def load_notes():
    f = NOTES_DIR / "defense_notes.py"
    if not f.exists():
        return {}
    spec = importlib.util.spec_from_file_location("defense_notes", f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return dict(mod.NOTES)


def context(R, M):
    """The numbers and result tables that more than one slide uses."""
    jc = csv("judge_candidates.csv")
    jsb = csv("judge_selection_bias.csv")
    ft = Mimic.finetune_summary()
    return dict(cal=R.cal_row("all"), jc=jc, jsb=jsb,
                best=jc[jc.group == "all"].sort_values("kappa", ascending=False).iloc[0],
                cv=jsb[(jsb.analysis == "nested_cv_selection_procedure") & (jsb.metric == "kappa")].iloc[0],
                aud=csv("audit_summary.csv").set_index("condition"), agr=csv("audit_agreement.csv").set_index("kind"),
                eqv=csv("mimic_equivalence.csv").set_index("comparison"), st2=csv("mimic_second_judge_tests.csv"),
                ev=csv("verifact_judge_eval.csv").set_index("subset"),
                mednli=(pct(ft["zero_shot_dev_accuracy"]), pct(ft["dev_accuracy"])) if ft else ("80.6%", "87.8%"),
                jname=M.judge_name(), n=M.n_docs, u={c: M.mean(c, "UFR") for c in M.conds},
                rel=sorted(abs(float(M.test("UFR", c).rel_change_mean)) for c in ("E1", "E1b", "E3")),
                dlt=sorted(abs(float(M.test("UFR", c).delta_mean)) for c in ("E1", "E1b", "E3")),
                pmax=max(float(M.test("UFR", c).p_holm_primary) for c in ("E1", "E1b", "E3")),
                crs=[M.mean(c, "CR") for c in ("E0", "E1", "E1b", "E3")])


def harrow(d, sl, x1, x2, y, color=MID, width=1.75):
    d.line(sl, x1, y, x2, y, color=color, width=width, arrow=True)


def caps(d, sl, x, y, w, text, color=MUTED, size=11.5):
    """A small label in capitals above a figure or a group."""
    d.tx(sl, x, y, w, 0.3, text.upper(), size=size, color=color, bold=True, spacing=60, check=False)


# the three labels of the judge: fill, text color, outline
STATUS = {"Supported": (CARD, MUTED, "B9BDC4"), "Not supported": (CARD, ACC, ACC), "Contradicted": (ACC, WHITE, None)}


def status(d, sl, x, y, lab, size=12, h=0.3):
    fill, tcol, edge = STATUS[lab]
    return d.tag(sl, x, y, lab, fill, size=size, h=h, text=tcol, line=edge)


def takeaway(d, sl, y, hgt, text, size=19, x=MX, w=CW):
    """The one sentence to remember from a slide, on a slightly darker band."""
    d.box(sl, x, y, w, hgt, fill=TINT2)
    d.tx(sl, x + 0.35, y, w - 0.7, hgt, text, size=size, color=INK, bold=True, anchor="m", line=1.12)


def slides_opening(d, X):
    # ── title ──────────────────────────────────────────────────────────
    sl = d.new("title", kind="plain")
    d.tx(sl, 0.9, 1.3, 11.5, 0.35, "MASTER'S THESIS DEFENSE", size=13, color=MUTED, bold=True, spacing=120, check=False)
    d.tx(sl, 0.9, 1.8, 11.6, 2.3, "Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization", size=38, color=INK, bold=True, line=1.06)
    motif = [(LIGHT, None)] * 26                   # a summary as a row of claims: most supported, two not supported, one contradicted
    motif[8] = motif[19] = (BG, ACC)
    motif[14] = (ACC, None)
    d.squares(sl, 0.93, 4.02, motif, per_row=26, s=0.2, gap=0.09, line_w=1.5)
    d.tx(sl, 0.9, 4.72, 11.5, 0.5, "Srilaya Ponangi", size=24, color=INK, bold=True, check=False)
    d.tx(sl, 0.9, 5.32, 11.5, 1.4, ["M.S. Computer Science, California State University, Northridge",
                                    "Committee: Dr. Taehyung (George) Wang (chair), Dr. Robert McIlhenny, Dr. Mahdi Ebrahimi", DEFENSE_DATE],
         size=16, color=MUTED, after=5)

    # ── overview: the six sections ─────────────────────────────────────
    sl = d.new("agenda", "Overview of the talk", kicker="Thesis defense")
    top, hgt = 1.95, 4.8
    rh = hgt / len(SECTIONS)
    d.box(sl, MX, top, CW, hgt)
    namex, namew = MX + 1.2, 5.15
    descx = namex + namew + 0.3
    descw = MX + CW - 0.4 - descx
    for i, name in enumerate(SECTIONS):
        y = top + i * rh
        if i:
            d.line(sl, MX + 0.35, y, MX + CW - 0.35, y, color=HAIR, width=0.75)
        d.num(sl, MX + 0.4, y + (rh - 0.46) / 2, i + 1, d=0.46, size=16)
        d.tx(sl, namex, y, namew, rh, name, size=19, color=INK, bold=True, anchor="m", line=1.04)
        items = SECTION_ITEMS[i]
        desc = ", ".join([items[0]] + [t if t[1:2].isupper() else t[0].lower() + t[1:] for t in items[1:]])
        d.tx(sl, descx, y, descw, rh, desc, size=15.5, color=MUTED, anchor="m", line=1.12)


def divider(d, i):
    """The slide that opens section i: its number, its name, what it goes through, and where it sits among the six."""
    sl = d.new(f"sec{i}")
    x = MX + 3.3
    w = MX + CW - x
    name, items = SECTIONS[i - 1], SECTION_ITEMS[i - 1]
    name_h = n_lines(name, 40, w, True) * 40 * 1.04 * 1.14 / 72 + 0.06
    list_h = text_h(items, 20, w - 0.3, False, 1.1, 8)
    top = 3.7 - (0.42 + name_h + 0.45 + list_h + 0.5 + 0.4) / 2       # the block is centered on the slide
    d.tx(sl, MX + 0.1, top - 0.45, 3.0, 3.2, str(i), size=210, color=LIGHT, bold=True, check=False)
    caps(d, sl, x, top, w, f"Section {i} of {len(SECTIONS)}", size=13)
    d.tx(sl, x, top + 0.42, w, name_h, name, size=40, color=INK, bold=True, line=1.04)
    yb = top + 0.42 + name_h + 0.45
    d.bullets(sl, x, yb, w, list_h + 0.1, items, size=20, color=TEXT, after=8, dot=ACC, indent=0.3)
    yd = yb + list_h + 0.5
    for k in range(len(SECTIONS)):                      # the six sections as a row of numbers, this one marked
        cx = x + k * 0.56
        on = k == i - 1
        d.dot(sl, cx, yd, 0.4, ACC if on else CARD, line=None if on else LIGHT)
        d.tx(sl, cx, yd, 0.4, 0.4, str(k + 1), size=13, color=WHITE if on else MUTED, bold=True, align="c", anchor="m", check=False)


def slides_intro(d, X):
    # ── why it is useful ───────────────────────────────────────────────
    sl = d.new("useful", "LLMs can write the summary that a patient reads", section=1)
    lw = 7.1
    d.box(sl, MX, 1.95, lw, 4.5)
    y0 = 3.1
    d.doc(sl, MX + 0.75, y0, 1.2, 1.55, rows=6)
    d.tx(sl, MX + 0.35, y0 + 1.72, 2.0, 0.3, "Clinical note", size=15, color=MUTED, align="c", check=False)
    harrow(d, sl, MX + 2.2, MX + 2.9, y0 + 0.78)
    d.node(sl, MX + 3.05, y0 + 0.38, 1.3, 0.8, "LLM", size=18)
    harrow(d, sl, MX + 4.5, MX + 5.2, y0 + 0.78)
    d.doc(sl, MX + 5.45, y0 + 0.18, 0.95, 1.2, rows=4)
    d.tx(sl, MX + 4.75, y0 + 1.72, 2.35, 0.3, "Summary for the patient", size=15, color=MUTED, align="c", check=False)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    d.box(sl, rx, 1.95, rw, 4.5)
    caps(d, sl, rx + 0.36, 2.3, rw - 0.7, "Reader study")
    d.tx(sl, rx + 0.36, 2.75, rw - 0.72, 1.3, "Equal to or better", size=34, color=INK, bold=True, line=1.04)
    d.tx(sl, rx + 0.36, 3.55, rw - 0.72, 1.6, "is how physicians rated summaries from adapted LLMs against summaries written by medical experts.", size=17, line=1.18)
    d.tx(sl, rx + 0.36, 5.8, rw - 0.72, 0.3, "Van Veen et al., 2024", size=14, color=MUTED)

    # ── the example ────────────────────────────────────────────────────
    sl = d.new("motivation", "But some statements have no support in the note", section=1)
    nw, ngap = 4.4, 1.5
    sx, sw = MX + nw + ngap, CW - nw - ngap
    top, hgt = 1.95, 4.15
    d.box(sl, MX, top, nw, hgt)
    d.box(sl, sx, top, sw, hgt)
    caps(d, sl, MX + 0.3, top + 0.28, nw - 0.5, "Clinical note")
    caps(d, sl, sx + 0.3, top + 0.28, sw - 0.5, "LLM summary for the patient")
    ry = [top + 0.9 + i * 0.76 for i in range(4)]
    for y, sent in zip(ry, ["Admitted with chest pain.", "Discharged on aspirin 81 mg daily.", "Troponin negative.", "No history of diabetes."]):
        d.tx(sl, MX + 0.3, y, nw - 0.5, 0.5, sent, size=16, anchor="m")
    # a summary sentence, its label, and the line of the note it rests on (None: nothing in the note)
    rows = [("You were admitted with chest pain.", "Supported", 0), ("Take aspirin 81 mg every day.", "Supported", 1),
            ("See your cardiologist in two weeks.", "Not supported", None), ("Keep managing your diabetes.", "Contradicted", 3)]
    x1, x2 = MX + nw + 0.12, sx - 0.12
    for i, (sent, lab, src) in enumerate(rows):
        y = ry[i]
        tgw = d.tag_w(lab, 13)
        d.tx(sl, sx + 0.3, y, sw - 0.6 - tgw - 0.12, 0.5, sent, size=16, anchor="m")
        status(d, sl, sx + sw - 0.3 - tgw, y + 0.08, lab, size=13, h=0.34)
        if src is None:
            d.line(sl, x2 - 0.5, y + 0.25, x2, y + 0.25, color=ACC, width=1.75, dash=True)
            d.cross(sl, x2 - 0.8, y + 0.15, 0.2)
            d.tx(sl, x1, y + 0.36, x2 - x1, 0.22, "no source", size=12, color=ACC, bold=True, align="r", check=False)
        else:
            bad = lab == "Contradicted"
            d.line(sl, x1, ry[src] + 0.25, x2, y + 0.25, color=ACC if bad else MID, width=2.5 if bad else 1.75)
            if bad:
                d.tx(sl, x1, y + 0.3, x2 - x1, 0.22, "conflict", size=12, color=ACC, bold=True, align="c", check=False)
    d.tx(sl, MX, top + hgt + 0.16, CW, 0.28, "Illustration written for this slide, not patient data.", size=13, color=MUTED)

    # ── problem 1: measurement ─────────────────────────────────────────
    sl = d.new("p_measure", "Problem 1, measurement: is each claim supported by the note?", section=1)
    d.box(sl, MX, 1.95, CW, 2.75)
    cx, yy = MX + 1.6, 2.72
    for dx, lab, kw in ((0.0, "Note", dict(rows=5)), (1.5, "Summary", dict(rows=4, mark=(2,)))):
        d.tx(sl, cx + dx - 0.3, yy - 0.42, 1.6, 0.3, lab, size=14, color=MUTED, align="c", check=False)
        d.doc(sl, cx + dx, yy, 1.0, 1.28, **kw)
    harrow(d, sl, cx + 2.85, cx + 3.65, yy + 0.64)
    d.node(sl, cx + 3.85, yy + 0.22, 1.9, 0.84, "Judge", size=19)
    d.dot(sl, cx + 5.5, yy - 0.03, 0.5, CARD, line=ACC)
    d.tx(sl, cx + 5.5, yy - 0.03, 0.5, 0.5, "?", size=19, color=ACC, bold=True, align="c", anchor="m", check=False)
    harrow(d, sl, cx + 6.1, cx + 6.9, yy + 0.64)
    for j, lab in enumerate(("Supported", "Not supported", "Contradicted")):
        status(d, sl, cx + 7.12, yy + 0.02 + j * 0.45, lab, size=15, h=0.38)
    d.tx(sl, MX, 5.0, CW, 0.9, "No reference summary exists, so ROUGE and BERTScore do not apply. Every claim is checked against the note itself.", size=22, color=INK, bold=True, line=1.1)
    d.tx(sl, MX, 5.98, CW, 0.5, "**The risk:** an unreliable judge misleads every conclusion drawn with it.", size=18, color=MUTED)

    # ── problem 2: mitigation ──────────────────────────────────────────
    sl = d.new("p_mitigate", "Problem 2, mitigation: can grounding reduce unsupported claims?", section=1)
    d.box(sl, MX, 1.95, CW, 2.75)
    cx, yy = MX + 1.85, 2.62
    for dx, lab in ((0.0, "Note"), (2.25, "Excerpts"), (7.35, "Summary")):
        d.tx(sl, cx + dx - 0.3, yy - 0.42, 1.6, 0.3, lab, size=14, color=MUTED, align="c", check=False)
    d.doc(sl, cx, yy, 1.0, 1.28, rows=5, mark=(1, 3))
    harrow(d, sl, cx + 1.25, cx + 2.0, yy + 0.64)
    d.bars(sl, cx + 2.25, yy + 0.27, 1.0, n=3, h=0.16, gap=0.13, color=ACC)
    harrow(d, sl, cx + 3.5, cx + 4.25, yy + 0.64)
    d.node(sl, cx + 4.5, yy + 0.22, 1.6, 0.84, "LLM", size=19)
    harrow(d, sl, cx + 6.35, cx + 7.1, yy + 0.64)
    d.doc(sl, cx + 7.35, yy, 1.0, 1.28, rows=4)
    lx, sxm, ly = cx + 5.3, cx + 7.85, yy + 1.7                 # the verification loop: from the summary back into the model
    d.line(sl, sxm, yy + 1.28, sxm, ly, color=MID, width=1.75)
    d.line(sl, sxm, ly, lx, ly, color=MID, width=1.75)
    d.line(sl, lx, ly, lx, yy + 1.1, color=MID, width=1.75, arrow=True)
    d.tx(sl, lx + 0.1, ly - 0.34, sxm - lx - 0.2, 0.26, "verify each claim, revise", size=13, color=MUTED, align="c", check=False)
    d.tx(sl, MX, 5.0, CW, 0.9, "Show the model the relevant passages (RAG) and make it check its own draft (Chain-of-Verification).", size=24, color=INK, bold=True, line=1.1)
    d.tx(sl, MX, 5.98, CW, 0.5, "**The cost:** what is not retrieved cannot be summarized.", size=18, color=MUTED)

    # ── what the thesis does ───────────────────────────────────────────
    sl = d.new("p_thesis", "What this thesis does", section=1)
    steps = ["Build a claim-level judge", "Validate it against medical experts", "Replace it with the candidate that agrees best",
             "Compare five summarization conditions on real hospital courses"]
    sw4, sg = 2.62, 0.551
    for i, txt in enumerate(steps):
        x = MX + i * (sw4 + sg)
        d.box(sl, x, 2.15, sw4, 4.25)
        d.num(sl, x + 0.3, 2.45, i + 1, d=0.52, size=18)
        px, py = x + 0.32, 3.35
        if i == 0:
            d.node(sl, px, py, 1.3, 0.62, "Judge", size=15)
        elif i == 1:
            d.doc(sl, px, py - 0.1, 0.62, 0.8, rows=4)
            d.check(sl, px + 0.85, py + 0.03, 0.52, color=STRONG, width=3.5)
        elif i == 2:
            for k in range(7):
                d.dot(sl, px + k * 0.28, py + 0.2, 0.21, ACC if k == 4 else LIGHT)
        else:
            for k, (hh, colr) in enumerate(((0.78, CCOL["E0"]), (0.45, CCOL["E1"]), (0.43, CCOL["E1b"]), (0.4, CCOL["E3"]), (0.34, LIGHT))):
                d.col(sl, px + k * 0.3, py + 0.7 - hh, 0.22, hh, colr)
        d.tx(sl, x + 0.3, 4.28, sw4 - 0.6, 1.75, txt, size=19, color=INK, bold=True, line=1.12)
        if i < 3:
            harrow(d, sl, x + sw4 + 0.1, x + sw4 + sg - 0.1, 4.15)

    # ── research questions ─────────────────────────────────────────────
    sl = d.new("rq", "Three research questions", section=1)
    cw3 = (CW - 2 * 0.25) / 3
    rqs = [("RQ1", "How often?", "How often do statements in LLM-generated patient summaries lack support in, or contradict, the source note?"),
           ("RQ2", "Which ones, and how does the judge err?", "Which statements are unsupported, and how does the automatic judge itself err when compared with medical experts?"),
           ("RQ3", "Does grounding help, and at what cost?", "Does retrieval-augmented generation reduce them, how does it compare with an extractive summarizer, and what does it cost in coverage?")]
    for i, (lab, key, q) in enumerate(rqs):
        x = MX + i * (cw3 + 0.25)
        d.box(sl, x, 1.95, cw3, 4.5)
        d.pill(sl, x + 0.35, 2.28, lab, size=16, h=0.5, fill=SLATE, color=WHITE, line=None, bold=True)
        d.tx(sl, x + 0.35, 3.05, cw3 - 0.7, 1.4, key, size=25, color=INK, bold=True, line=1.06)
        d.tx(sl, x + 0.35, 4.6, cw3 - 0.7, 1.75, q, size=16.5, color=TEXT, line=1.2)

    # ── hypotheses ─────────────────────────────────────────────────────
    sl = d.new("hyp", "Three hypotheses, fixed before the experiments", section=1)
    hyps = [("H1", "Unsupported content is measurable, and the judge agrees with medical experts better than chance.", "for RQ1"),
            ("H2", "RAG lowers the unsupported and the contradicted rate.", "for RQ3"),
            ("H3", "Verification on top of RAG lowers them further.", "for RQ3")]
    for i, (lab, txt, which) in enumerate(hyps):
        y = 1.95 + i * 1.38
        d.box(sl, MX, y, CW, 1.18)
        d.pill(sl, MX + 0.35, y + 0.34, lab, size=17, h=0.5, fill=SLATE, color=WHITE, line=None, bold=True, w=0.95)
        d.tx(sl, MX + 1.65, y, CW - 3.4, 1.18, txt, size=21, color=INK, anchor="m", line=1.12)
        d.tx(sl, MX + CW - 1.55, y, 1.2, 1.18, which, size=14, color=MUTED, align="r", anchor="m", check=False)
    d.tx(sl, MX, 6.2, CW, 0.3, "RQ2 is descriptive, so it has no hypothesis.", size=15, color=MUTED)


def slides_background(d, X):
    cal, n = X["cal"], X["n"]

    def lane(key, title, works):
        """One line of related work as tiles in order of year: (year, authors, name, what it showed, used or tested in this thesis)."""
        sl = d.new(key, title, section=2)
        k, gap = len(works), 0.22
        tw = (CW - (k - 1) * gap) / k
        y, hgt = 2.1, 3.85
        for i, (year, who, name, gist, used) in enumerate(works):
            x = MX + i * (tw + gap)
            d.box(sl, x, y, tw, hgt)
            d.tx(sl, x + 0.3, y + 0.3, 1.2, 0.34, year, size=18, color=SLATE, bold=True, check=False)
            if used:
                d.dot(sl, x + tw - 0.52, y + 0.36, 0.22, ACC)
            inner = tw - 0.6
            nsize = next((z for z in (22, 20, 18, 16.5) if n_lines(name, z, inner, True) <= 2 and max(text_w(p_, z, True) for p_ in name.split()) <= inner), 16.5)
            d.tx(sl, x + 0.3, y + 0.9, inner, 0.95, name, size=nsize, color=INK, bold=True, line=1.05, anchor="b")
            d.tx(sl, x + 0.3, y + 1.95, inner, 0.3, who, size=13.5, color=MUTED, check=False)
            d.tx(sl, x + 0.3, y + 2.45, inner, 1.25, gist, size=16, color=TEXT, line=1.15)
        d.dot(sl, MX + 0.05, 6.31, 0.2, ACC)
        d.tx(sl, MX + 0.38, 6.24, CW - 0.4, 0.34, "Its idea, model or data is used or tested in this thesis", size=14, color=MUTED, anchor="m", check=False)

    lane("rel_measure", "Related work 1 of 3: measuring faithfulness",
         [("2021", "Fabbri et al.", "SummEval", "metrics agree weakly with humans", False), ("2022", "Laban et al.", "SummaC", "sentence-level NLI", True),
          ("2023", "Min et al.", "FActScore", "splits text into atomic facts", True), ("2024", "Tang et al.", "MiniCheck", "grounding checkers", True),
          ("2025", "Godbole and Jia", "Metrics disagree", "validate in the target domain", False)])
    lane("rel_clinical", "Related work 2 of 3: clinical summarization",
         [("2018", "Romanov and Shivade", "MedNLI", "general NLI transfers poorly to clinical text", True),
          ("2024", "Van Veen et al.", "Reader study", "adapted LLMs match or beat medical experts", False),
          ("2024", "Hegselmann et al.", "Expert annotations", "unsupported facts marked in MIMIC-IV patient summaries", True),
          ("2026", "Chung et al.", "VeriFact", "LLM judge with a 70B model on four data-center GPUs", True)])
    lane("rel_reduce", "Related work 3 of 3: reducing hallucination",
         [("2020", "Lewis et al.", "RAG", "generation conditioned on retrieved evidence", True), ("2023", "Gao et al.", "RARR", "looks up evidence for each claim, then revises", False),
          ("2024", "Dhuliawala et al.", "Chain-of-Verification", "draft, verify each claim separately, revise", True)])

    # ── the gap ────────────────────────────────────────────────────────
    sl = d.new("gap", "The gap: judges are used to compare systems before anyone checks them against clinicians", section=2)
    gw, r = 6.9, 1.5
    d.box(sl, MX, 1.95, gw, 4.8)
    vx, vy = MX + gw / 2, 4.2
    rings = [(vx - 0.95, vy - 0.6, "Validated judge", "against clinicians", -0.55, -0.42),
             (vx + 0.95, vy - 0.6, "Compared conditions", "under one judge", 0.55, -0.42),
             (vx, vy + 0.95, "Measured omission", "on hospital courses", 0.0, 0.66)]
    for px, py, _, _, _, _ in rings:
        sh = d.dot(sl, px - r, py - r, 2 * r, CARD, line=MID)
        sh.fill.background(); sh.line.width = Pt(2)
    for px, py, lead, sub, ox, oy in rings:
        d.tx(sl, px + ox - 0.8, py + oy - 0.4, 1.6, 0.5, lead, size=14.5, color=INK, bold=True, align="c", anchor="b", line=1.02, check=False)
        d.tx(sl, px + ox - 0.8, py + oy + 0.13, 1.6, 0.24, sub, size=12.5, color=MUTED, align="c", check=False)
    d.dot(sl, vx - 0.42, vy - 0.06 - 0.42, 0.84, ACC)
    d.tx(sl, vx - 0.42, vy - 0.48, 0.84, 0.84, "This thesis", size=12, color=WHITE, bold=True, align="c", anchor="m", line=1.0, check=False)
    rx, rw = MX + gw + 0.5, CW - gw - 0.5
    caps(d, sl, rx, 2.75, rw, "What was missing")
    d.tx(sl, rx, 3.15, rw, 1.9, "One study that does all three", size=34, color=INK, bold=True, line=1.08)
    d.tx(sl, rx, 5.15, rw, 1.3, "Validate the judge, compare the conditions under it, and measure what retrieval leaves out.", size=17, color=MUTED, line=1.2)


def slides_method(d, R, M, X):
    cal, n, ev, jc = X["cal"], X["n"], X["ev"], X["jc"]
    cw3 = (CW - 2 * 0.25) / 3
    hw = (CW - 0.3) / 2

    # ── three studies ──────────────────────────────────────────────────
    sl = d.new("overview", "Three studies: build the pipeline, validate the judge, then run the comparison", section=3)
    studies = [("Pilot study", "50", "public notes from MTSamples", "Builds and debugs the pipeline"),
               ("Judge validation and selection", f"{int(cal.n_sentences):,}", "expert-annotated sentences in 210 patient summaries", "Gives a judge whose labels can be interpreted"),
               ("Main study", f"{n}", "MIMIC-IV hospital courses", "Tests whether grounding helps, and at what cost")]
    sw3, sgap = 3.62, 0.635
    for i, (head, big, unit_, purpose) in enumerate(studies):
        x = MX + i * (sw3 + sgap)
        d.box(sl, x, 1.95, sw3, 4.5)
        d.num(sl, x + 0.32, 2.25, i + 1, d=0.5, size=17)
        d.tx(sl, x + 1.0, 2.19, sw3 - 1.25, 0.62, head, size=18, color=INK, bold=True, anchor="m", line=1.04)
        d.tx(sl, x + 0.32, 3.05, sw3 - 0.64, 1.0, big, size=62, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.32, 4.22, sw3 - 0.64, 0.75, unit_, size=17, color=TEXT, line=1.12)
        d.line(sl, x + 0.32, 5.15, x + sw3 - 0.32, 5.15, color=HAIR, width=1.0)
        d.tx(sl, x + 0.32, 5.32, sw3 - 0.64, 0.9, purpose, size=16.5, color=SLATE, bold=True, line=1.12)
        if i < 2:
            harrow(d, sl, x + sw3 + 0.12, x + sw3 + sgap - 0.12, 4.2)

    # ── evaluation pipeline ────────────────────────────────────────────
    sl = d.new("pipeline", "The judge scores every summary claim by claim against the source note", section=3)
    steps = [("Split", "One sentence is one claim"), ("Find evidence", "Pilot judge: the 3 nearest sentences. Selected judge: the whole course"),
             ("Classify", "An NLI model: the evidence is the premise, the claim the hypothesis"), ("Label", "One of three labels per claim")]
    pw4, pgap = 2.74, 0.391
    for i, (head, txt) in enumerate(steps):
        x = MX + i * (pw4 + pgap)
        d.box(sl, x, 1.95, pw4, 4.4)
        d.num(sl, x + 0.28, 2.22, i + 1, d=0.5, size=17)
        d.tx(sl, x + 0.95, 2.14, pw4 - 1.1, 0.66, head, size=19, color=INK, bold=True, anchor="m", line=1.02)
        d.tx(sl, x + 0.28, 4.7, pw4 - 0.56, 1.5, txt, size=16, line=1.16)
        if i < 3:
            harrow(d, sl, x + pw4 + 0.07, x + pw4 + pgap - 0.07, 3.72)
    x = MX                                              # 1: a summary becomes a list of claims
    d.doc(sl, x + 0.3, 3.3, 0.66, 0.84, rows=4)
    harrow(d, sl, x + 1.08, x + 1.42, 3.72)
    d.bars(sl, x + 1.54, 3.42, 0.9, n=3, h=0.13, gap=0.1, color=STRONG, widths=(1, 0.78, 0.9))
    x = MX + (pw4 + pgap)                               # 2: one claim, and the part of the note that bears on it
    d.bars(sl, x + 0.3, 3.655, 0.85, n=1, h=0.13, color=STRONG)
    harrow(d, sl, x + 1.27, x + 1.61, 3.72)
    d.doc(sl, x + 1.74, 3.3, 0.66, 0.84, rows=5, mark=(1, 2))
    x = MX + 2 * (pw4 + pgap)                           # 3: evidence and claim go into the NLI model
    d.bars(sl, x + 0.3, 3.42, 0.8, n=2, h=0.12, gap=0.08, color=ACC, widths=(1, 0.8))
    d.bars(sl, x + 0.3, 3.9, 0.8, n=1, h=0.13, color=STRONG)
    harrow(d, sl, x + 1.2, x + 1.52, 3.72)
    d.node(sl, x + 1.62, 3.42, 0.86, 0.6, "NLI", size=15)
    x = MX + 3 * (pw4 + pgap)                           # 4: the three labels
    d.squares(sl, x + 0.36, 3.46, [(LIGHT, None), (CARD, ACC), (ACC, None)], s=0.52, gap=0.22, line_w=2)

    # ── how a claim gets its label ─────────────────────────────────────
    sl = d.new("labelrule", "How a claim gets its label", section=3)
    d.box(sl, MX, 1.95, 8.3, 4.6)
    qx, qw, qh = MX + 0.6, 4.3, 0.8
    ys = (2.35, 3.8)
    for (q, lab), y in zip((("entailment at least τ ?", "Supported"), ("contradiction at least 0.5 ?", "Contradicted")), ys):
        d.pill(sl, qx, y, q, size=20, h=qh, fill=TINT, color=INK, line=None, bold=True, w=qw)
        harrow(d, sl, qx + qw + 0.15, qx + qw + 1.35, y + qh / 2)
        d.tx(sl, qx + qw + 0.15, y + qh / 2 - 0.36, 1.2, 0.3, "yes", size=14, color=MUTED, align="c", check=False)
        status(d, sl, qx + qw + 1.5, y + qh / 2 - 0.25, lab, size=18, h=0.5)
        d.line(sl, qx + qw / 2, y + qh + 0.06, qx + qw / 2, y + qh + 0.58, color=MID, width=1.75, arrow=True)
        d.tx(sl, qx + qw / 2 + 0.14, y + qh + 0.17, 0.6, 0.3, "no", size=14, color=MUTED, check=False)
    wns = d.tag_w("Not supported", 18)
    status(d, sl, qx + qw / 2 - wns / 2, ys[1] + qh + 0.68, "Not supported", size=18, h=0.5)
    rx, rw = MX + 8.75, CW - 8.75
    caps(d, sl, rx, 2.55, rw, "The threshold")
    for i, (big, lab) in enumerate((("0.5", "τ for the pilot judge"), (f"{M.tau:.2f}", "τ for the selected judge"))):
        d.stat(sl, rx, 2.95 + i * 1.5, rw, big, lab, size=46, label_size=16, h=1.35)

    # ── the two rates ──────────────────────────────────────────────────
    sl = d.new("rates", "Two rates per summary, both judge-estimated", section=4)
    d.box(sl, MX, 1.95, CW, 1.7)
    s, g = 0.66, 0.2
    wsq = 10 * s + 9 * g
    d.squares(sl, MX + 0.5, 2.47, [(LIGHT, None)] * 8 + [(CARD, ACC), (ACC, None)], s=s, gap=g, line_w=2.25)
    d.tx(sl, MX + 0.5 + wsq + 0.4, 2.47, CW - wsq - 1.2, s, "One summary with 10 claims", size=17, color=MUTED, anchor="m", line=1.1)
    for i, (name, frac, what) in enumerate((("Unsupported Fact Rate", "2 / 10", "not supported and contradicted, over all claims"),
                                            ("Contradiction Rate", "1 / 10", "contradicted, over all claims"))):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 3.95, hw, 2.05)
        d.tx(sl, x + 0.4, 4.16, hw - 0.8, 0.45, name, size=20, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.4, 4.72, 2.4, 1.1, frac, size=46, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, x + 2.95, 4.72, hw - 3.35, 1.1, what, size=16, color=MUTED, anchor="m", line=1.15)
    d.tx(sl, MX, 6.25, CW, 0.4, "**Judge-estimated:** false alarms count, misses do not.", size=18, color=TEXT)

    # ── five conditions, shown one more at a time ──────────────────────
    grid = [("E0", "Full note, no retrieval", (1, 0, 0, 0)), ("E1", "RAG, excerpts only", (0, 1, 1, 0)), ("E1b", "RAG, note and excerpts", (1, 1, 1, 0)),
            ("E3", "RAG with Chain-of-Verification", (0, 1, 0, 1)), ("E2", "Extractive, no LLM", None)]
    cx0, dw = MX + 5.7, 1.55
    for shown in (len(grid),):                          # the whole matrix on one slide
        sl = d.new("conditions", "Five conditions isolate what retrieval and verification each add", section=3)
        d.box(sl, MX, 1.95, CW, 4.45)
        for j, head in enumerate(["Whole note", "3 retrieved excerpts", "Cites its excerpts", "Verifies and revises"]):
            xc = cx0 + (j + 0.5) * dw
            if j == 0:
                d.doc(sl, xc - 0.18, 2.12, 0.36, 0.45, rows=4)
            elif j == 1:
                d.bars(sl, xc - 0.24, 2.17, 0.48, n=3, h=0.09, gap=0.075, color=ACC)
            elif j == 2:
                d.tx(sl, xc - 0.4, 2.12, 0.8, 0.45, "[n]", size=16, color=INK, bold=True, align="c", anchor="m", check=False)
            else:
                d.check(sl, xc - 0.21, 2.14, 0.42, color=STRONG, width=3)
            d.tx(sl, xc - dw / 2 + 0.05, 2.65, dw - 0.1, 0.44, head, size=13, color=MUTED, bold=True, align="c", line=1.02)
        for i, (c, name, dots) in enumerate(grid[:shown]):
            y = 3.2 + i * 0.62
            if i == shown - 1 and shown < len(grid):        # the condition that was just added
                d.box(sl, MX + 0.15, y + 0.03, CW - 0.3, 0.56, fill=TINT, radius=0.08)
            else:
                d.line(sl, MX + 0.3, y, MX + CW - 0.3, y, color=HAIR, width=1.0)
            d.chip(sl, MX + 0.35, y + 0.12, c, size=15, h=0.38)
            d.tx(sl, MX + 1.6, y, 4.1, 0.62, name, size=16, color=INK, bold=True, anchor="m")
            if dots is None:
                d.tx(sl, cx0, y, 4 * dw, 0.62, "No LLM: the 5 most central sentences of the note, copied verbatim", size=14.5, color=MUTED, align="c", anchor="m", line=1.05)
            else:
                for j, on in enumerate(dots):
                    d.dot(sl, cx0 + (j + 0.5) * dw - 0.15, y + 0.16, 0.3, STRONG if on else CARD, line=None if on else LIGHT)
        if shown == len(grid):
            d.tx(sl, MX, 6.53, CW, 0.3, "Same generator and decoding settings in every LLM condition: GPT-4o-mini in the pilot, Qwen2.5-7B-Instruct in the main study.", size=13, color=MUTED)

    # ── RAG, then Chain-of-Verification ────────────────────────────────
    def flow4(sl, y, boxes, pictos, hgt=3.45, size=16.5):
        bw, bg = 2.7, 0.444
        for i, txt in enumerate(boxes):
            x = MX + i * (bw + bg)
            last = i == len(boxes) - 1
            d.box(sl, x, y, bw, hgt)
            d.tx(sl, x + 0.3, y + hgt - 1.8, bw - 0.6, 1.6, txt, size=size, line=1.16, bold=last, color=INK if last else TEXT)
            if not last:
                harrow(d, sl, x + bw + 0.07, x + bw + bg - 0.07, y + hgt / 2)
            pictos[i](x + 0.3, y + 0.38)

    def head(sl, c, name):
        w = d.chip(sl, MX, 2.0, c, size=16, h=0.42)
        d.tx(sl, MX + w + 0.2, 2.0, CW - w - 0.2, 0.42, name, size=19, color=INK, bold=True, anchor="m", check=False)

    sl = d.new("rag", "RAG retrieves from the note itself, to focus the model on the relevant passages", section=3)
    head(sl, "E1", "Retrieval-augmented generation")

    def r0(px, py):
        d.doc(sl, px, py, 0.62, 0.8, rows=5)
        for k in range(3):
            d.box(sl, px + 0.95 + k * 0.42, py + 0.15, 0.32, 0.5, fill=CARD, radius=0.05, line=MUTED, line_w=1.25)

    def r1(px, py):
        d.bars(sl, px, py + 0.12, 1.4, n=3, h=0.15, gap=0.11, color=ACC)

    def r2(px, py):
        d.bars(sl, px, py + 0.14, 0.7, n=3, h=0.13, gap=0.1, color=ACC)
        harrow(d, sl, px + 0.82, px + 1.18, py + 0.4)
        d.node(sl, px + 1.28, py + 0.12, 0.82, 0.56, "LLM", size=14)

    def r3(px, py):
        d.doc(sl, px, py, 0.62, 0.8, rows=4)
        d.tx(sl, px + 0.8, py, 1.2, 0.8, "[1] [3]", size=17, color=ACC, bold=True, anchor="m", check=False)

    flow4(sl, 2.75, ["Split the note into 5-sentence chunks and embed them", "Keep the 3 chunks most similar to the query",
                     "Prompt: use only the excerpts, cite [n] after every sentence", "Summary with citations"], [r0, r1, r2, r3])

    sl = d.new("cove", "Chain-of-Verification checks every claim of the draft", section=3)
    head(sl, "E3", "Chain-of-Verification on the E1 draft")

    def v0(px, py):
        d.doc(sl, px, py, 0.62, 0.8, rows=4)
        harrow(d, sl, px + 0.74, px + 1.08, py + 0.4)
        d.bars(sl, px + 1.2, py + 0.14, 0.9, n=3, h=0.13, gap=0.1, color=STRONG, widths=(1, 0.78, 0.9))

    def v1(px, py):
        d.bars(sl, px, py + 0.335, 0.8, n=1, h=0.13, color=STRONG)
        d.bars(sl, px + 1.0, py + 0.14, 1.0, n=3, h=0.13, gap=0.1, color=ACC, widths=(1, 0.8, 0.9))

    def v2(px, py):
        d.check(sl, px, py + 0.1, 0.56, color=STRONG, width=3.5)
        d.cross(sl, px + 0.95, py + 0.2, 0.38, color=ACC, width=3.5)

    def v3(px, py):
        d.doc(sl, px, py, 0.62, 0.8, rows=4, mark=(2,))

    flow4(sl, 2.75, ["Split the E1 draft into claims", "Retrieve 3 sentences of the full note for each claim", "Verify each claim at temperature 0",
                     "Revise: keep, correct or remove each claim"], [v0, v1, v2, v3])
    d.tx(sl, MX, 6.4, CW, 0.3, "About 10 model calls per note.", size=15, color=MUTED)

    # ── judge validation: the candidates ───────────────────────────────
    sl = d.new("j_cand", "Seven candidate judges that run on a laptop, each in two evidence modes", section=3)
    mw = 8.4
    d.box(sl, MX, 1.95, mw, 4.75)
    order = ["minilm_nli", "deberta_large_nli", "ce_deberta_large_nli", "minicheck_deberta", "minicheck_roberta", "bespoke_minicheck_7b", "mednli_deberta_large"]
    judges = [j for j in order if j in set(jc.judge)] + [j for j in jc.judge.unique() if j not in order]
    dx1, dx2 = MX + mw - 2.75, MX + mw - 1.05
    for xc, lab in ((dx1, "top 3 sentences"), (dx2, "whole course")):
        d.tx(sl, xc - 0.8, 2.14, 1.6, 0.3, lab, size=13, color=MUTED, bold=True, align="c", check=False)
    for i, j in enumerate(judges):
        y = 2.57 + i * 0.58
        d.line(sl, MX + 0.35, y, MX + mw - 0.35, y, color=HAIR, width=1.0)
        d.tx(sl, MX + 0.4, y, mw - 4.1, 0.58, JSHORT.get(j, JUDGE_NAMES.get(j, j)), size=16.5, color=INK, anchor="m")
        for xc in (dx1, dx2):
            d.dot(sl, xc - 0.14, y + 0.15, 0.28, MID)
    rx, rw = MX + mw + 0.5, CW - mw - 0.5
    d.stat(sl, rx, 2.7, rw, f"{2 * len(judges)}", f"configurations: {len(judges)} judges in 2 evidence modes", size=72, label_size=17, h=2.6)

    # ── judge validation: the rule ─────────────────────────────────────
    sl = d.new("j_rule", "The judge is chosen by a rule fixed in advance", section=3)
    lw = 5.4
    d.box(sl, MX, 1.95, lw, 4.5)
    caps(d, sl, MX + 0.4, 2.3, lw - 0.8, "Decision rule")
    d.tx(sl, MX + 0.4, 2.85, lw - 0.8, 1.9, "The highest Cohen's kappa with the medical experts", size=30, color=INK, bold=True, line=1.1)
    d.tx(sl, MX + 0.4, 4.85, lw - 0.8, 1.4, [f"On {int(cal.n_sentences):,} sentences of 210 patient summaries that two medical experts checked (ann-pt-summ)",
                                            "Also reported: precision, recall and AUROC"], size=15, color=MUTED, line=1.15)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    d.box(sl, rx, 1.95, rw, 4.5)
    caps(d, sl, rx + 0.4, 2.3, rw - 0.8, "Safeguards")
    for i, txt in enumerate(("Threshold chosen on the other subset, never on the sentences it scores", "Nested cross-validation over courses", "Patient-disjoint transfer test")):
        y = 2.85 + i * 1.15
        d.check(sl, rx + 0.42, y + 0.28, 0.38, color=ACC, width=3.25)
        d.tx(sl, rx + 1.1, y, rw - 1.5, 0.95, txt, size=19, color=TEXT, anchor="m", line=1.12)

    # ── what is measured ───────────────────────────────────────────────
    sl = d.new("measures", "Two more measures: what a summary leaves out, and whether its citations hold", section=4)
    meas = [("Coverage of the clinician's instructions", "Clinician's instructions", "Summary", "Share of the clinician's sentences that the summary supports. It measures omission."),
            ("Citations", "Claim with [n]", "Cited excerpt", "Share of claims that are cited, and share of citations that hold.")]
    for i, (lead, left, right, txt) in enumerate(meas):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 4.5)
        d.num(sl, x + 0.35, 2.22, i + 1, d=0.5, size=17)
        d.tx(sl, x + 1.05, 2.16, hw - 1.4, 0.62, lead, size=19, color=INK, bold=True, anchor="m", line=1.04)
        la, lb, yy = x + 1.2, x + hw - 1.2 - 0.8, 3.15
        if i == 0:
            d.doc(sl, la, yy, 0.8, 1.02, rows=4)
            d.doc(sl, lb, yy, 0.8, 1.02, rows=4)
        else:
            d.bars(sl, la, yy + 0.445, 0.6, n=1, h=0.13, color=STRONG)
            d.tx(sl, la + 0.62, yy + 0.26, 0.5, 0.5, "[2]", size=15, color=ACC, bold=True, anchor="m", check=False)
            d.bars(sl, lb, yy + 0.2, 0.8, n=3, h=0.13, gap=0.12, color=LIGHT)
            d.box(sl, lb, yy + 0.45, 0.8, 0.13, fill=ACC, radius=0.065)
        harrow(d, sl, la + 1.02, lb - 0.14, yy + 0.58)
        d.tx(sl, la + 0.8, yy + 0.22, lb - la - 0.8, 0.26, "checked against", size=12, color=MUTED, align="c", check=False)
        d.tx(sl, la - 0.3, yy + 1.1, 1.4, 0.5, left, size=13, color=MUTED, align="c", line=1.02, check=False)
        d.tx(sl, lb - 0.3, yy + 1.1, 1.4, 0.5, right, size=13, color=MUTED, align="c", line=1.02, check=False)
        d.tx(sl, x + 0.45, 4.95, hw - 0.9, 1.35, txt, size=18, line=1.18)

    # ── statistics ─────────────────────────────────────────────────────
    sl = d.new("stats", "How the conditions are compared", section=4)
    tiles = [("Paired", f"every condition on the same {n} courses"), ("Wilcoxon", "signed-rank test, two-sided"), ("Bootstrap", "95% interval, 10,000 resamples"),
             ("Holm", "adjustment over 12 primary tests"), ("Equivalence", "two one-sided tests, 5-point margin")]
    tw5 = (CW - 4 * 0.2) / 5
    for i, (key, txt) in enumerate(tiles):
        x = MX + i * (tw5 + 0.2)
        d.box(sl, x, 2.35, tw5, 3.1)
        d.num(sl, x + 0.28, 2.65, i + 1, d=0.46, size=16)
        d.tx(sl, x + 0.28, 3.45, tw5 - 0.5, 0.5, key, size=18, color=INK, bold=True)
        d.tx(sl, x + 0.28, 4.05, tw5 - 0.56, 1.2, txt, size=16, color=TEXT, line=1.18)

    # ── threats and checks ─────────────────────────────────────────────
    threats = [("The summaries are just shorter", "E0 limited to **150 words**, the length of E1", "Result 5"),
               ("The instruction to cite does the work", "E1 and E1b **without the citation instruction**", "Result 5"),
               ("One generation is a lucky draw", "**Two more generations** on 40 courses", "Result 6"),
               ("It depends on the judge, threshold or unit", "**Second judge**, second threshold, atomic facts", "Result 6"),
               ("The judge errs differently on this text", "**Blind manual audit**: 340 claims, 60 citations", "Result 8"),
               ("The judge only works on its own data", f"**{int(ev.loc['all', 'n']):,}** clinician-labeled propositions", None),
               ("One model, one retrieval setting", "**Two more generators**, six retrieval ablations", "Result 7")]

    def threat_rows(key, title, rows):
        sl = d.new(key, title, section=4)
        lw_ = 5.3
        cxk = MX + lw_ + 0.8
        d.tx(sl, MX + 0.3, 1.95, lw_, 0.3, "What else could explain the result", size=13, color=MUTED, bold=True)
        d.tx(sl, cxk + 0.3, 1.95, 4.5, 0.3, "Check built into the study", size=13, color=MUTED, bold=True)
        d.tx(sl, MX + CW - 2.1, 1.95, 1.8, 0.3, "Shown in", size=13, color=MUTED, bold=True, align="r")
        for i, (threat, chk, where) in enumerate(rows):
            y = 2.4 + i * 1.08
            d.box(sl, MX, y, lw_, 0.9, fill=TINT2)
            d.tx(sl, MX + 0.3, y, lw_ - 0.6, 0.9, threat, size=18, anchor="m", line=1.08)
            harrow(d, sl, MX + lw_ + 0.14, cxk - 0.14, y + 0.45)
            d.box(sl, cxk, y, MX + CW - cxk, 0.9)
            d.tx(sl, cxk + 0.3, y, MX + CW - cxk - 2.4, 0.9, chk, size=18, anchor="m", line=1.08)
            tb = d.tx(sl, MX + CW - 2.1, y, 1.8, 0.9, where or "Backup slide", size=13.5, color=MUTED, align="r", anchor="m", check=False)
            if where is None:                           # the external check is a backup slide; its number is known once the deck is complete
                d.late.append((tb, "bk_verifact"))

    threat_rows("controls", "Each threat to the comparison gets its own check", threats[:4])
    threat_rows("controls2", "Three more threats, three more checks", threats[4:])

def slides_results(d, R, M, T, X):
    cal, best, cv, aud, agr, eqv, st2, ev = (X[k] for k in ("cal", "best", "cv", "aud", "agr", "eqv", "st2", "ev"))
    n, u, rel = X["n"], X["u"], X["rel"]
    cw3 = (CW - 2 * 0.25) / 3
    hw = (CW - 0.3) / 2
    tw4 = (CW - 3 * 0.25) / 4

    def cite(c):
        r = M.base.loc[c]
        return r.cited_claims / r.claims_total, r.cite_correct / r.cited_claims

    def tv(label, cond, col):
        """A cell of a thesis table: the row whose first cell starts with the condition code."""
        t = T[label]
        j = t["columns"].index(col)
        for r in t["rows"]:
            if r[0] == cond or r[0].startswith(cond + " "):
                return r[j]
        raise KeyError((label, cond, col))

    def chart_left(sl, img, cwid=7.85, chgt=4.75):
        """A chart in a white card on the left. Returns the left edge and the width of the column beside it."""
        d.chart(sl, img, MX, 1.95, cwid, chgt)
        return MX + cwid + 0.4, CW - cwid - 0.4

    # ── data ───────────────────────────────────────────────────────────
    sl = d.new("data", "Data: public notes for the pilot, credentialed hospital data for everything else", section=4)
    sets = [("Pilot study", "50", "notes sampled from 4,999 transcriptions", "MTSamples", "public"),
            ("Judge validation", f"{int(cal.n_sentences):,}", "sentences in 210 expert-annotated patient summaries", "ann-pt-summ", "PhysioNet"),
            ("Main study", f"{n}", "hospital courses, each with the clinician's discharge instructions", "MIMIC-IV-Note", "PhysioNet"),
            ("Fine-tuning a judge", "14,049", "clinical premise and hypothesis pairs", "MedNLI", "PhysioNet"),
            ("External check", f"{int(ev.loc['all', 'n']):,}", "clinician-labeled propositions from 100 patients", "VeriFact-BHC", "PhysioNet")]
    tw5 = (CW - 4 * 0.18) / 5
    for i, (role, big, unit_, name, access) in enumerate(sets):
        x = MX + i * (tw5 + 0.18)
        d.box(sl, x, 1.95, tw5, 4.3)
        d.tx(sl, x + 0.24, 2.2, tw5 - 0.4, 0.26, role.upper(), size=10.5, color=MUTED, bold=True, spacing=50, check=False)
        d.tx(sl, x + 0.24, 2.6, tw5 - 0.4, 0.8, big, size=38, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.24, 3.45, tw5 - 0.48, 1.65, unit_, size=15.5, color=TEXT, line=1.15)
        d.tx(sl, x + 0.24, 5.18, tw5 - 0.4, 0.34, name, size=16, color=INK, bold=True, check=False)
        pub = access == "public"
        d.pill(sl, x + 0.24, 5.64, access, size=12.5, h=0.34, fill=CARD if pub else SLATE, color=INK if pub else WHITE, line="C3C7CD" if pub else None)

    # ── one laptop ─────────────────────────────────────────────────────
    sl = d.new("laptop", "Everything runs on one laptop, so no protected text leaves it", section=4)
    lap = [("24 GB", "Apple MacBook Air M5"), ("7B, 4-bit", "Qwen2.5-7B-Instruct, served locally"), ("1,210", "main-study summaries"), ("about 7 h", "to generate them")]
    for i, (big, lab) in enumerate(lap):
        x = MX + i * (tw4 + 0.25)
        d.box(sl, x, 2.15, tw4, 2.35)
        d.tx(sl, x + 0.3, 2.5, tw4 - 0.6, 0.8, big, size=32, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.3, 3.4, tw4 - 0.6, 0.9, lab, size=16, color=MUTED, line=1.15)
    takeaway(d, sl, 4.85, 1.3, "No protected text leaves the laptop, as the data use agreement requires.", size=22)

    # ── result 1: the pilot judge ──────────────────────────────────────
    sl = d.new("r_pilot", f"Result 1: the pilot judge fails. It flags {pct0(cal.judge_UFR)} of sentences where experts flag {pct0(cal.expert_flag_rate)}", section=4)
    rx, rw = chart_left(sl, chart_pilot(R))
    d.stat(sl, rx, 2.1, rw, f2(cal.any_kappa), "Cohen's kappa with the medical experts: chance level", size=60, label_size=16, h=1.95)
    d.stat(sl, rx, 4.4, rw / 2, f2(cal.any_precision), "precision", size=34, label_size=16, h=1.0)
    d.stat(sl, rx + rw / 2, 4.4, rw / 2, f2(cal.any_recall), "recall", size=34, label_size=16, h=1.0)

    sl = d.new("r_pilot2", f"No evidence rule repaired it: the best kappa at any threshold was {f2(R.cal_var.best_kappa.max())}", section=4)
    var = R.cal_var[R.cal_var.group == "all"].set_index("variant")
    vname = importlib.import_module("ch5_pilot").VNAME
    lw = 8.6
    d.box(sl, MX, 1.95, lw, 3.55)
    caps(d, sl, MX + 0.4, 2.2, lw - 0.8, "Best kappa of the pilot judge under five evidence rules")
    mx0 = MX + 5.05
    mw_ = MX + lw - 1.15 - mx0
    d.tx(sl, mx0 - 0.1, 2.56, 0.3, 0.22, "0", size=11, color=MUTED, check=False)
    d.tx(sl, mx0 + mw_ - 0.2, 2.56, 0.3, 0.22, "1", size=11, color=MUTED, align="r", check=False)
    for i, v in enumerate(("max3", "concat3", "concat5", "maxall", "concat_ctx")):
        y = 2.85 + i * 0.5
        kv = float(var.loc[v, "best_kappa"])
        d.tx(sl, MX + 0.4, y, 4.5, 0.4, vname[v], size=15, color=TEXT, anchor="m")
        d.meter(sl, mx0, y + 0.125, mw_, kv, h=0.15, fill=STRONG)
        d.tx(sl, mx0 + mw_ + 0.15, y, 0.8, 0.4, f2(kv), size=15, color=INK, bold=True, anchor="m", check=False)
    rx, rw = MX + lw + 0.4, CW - lw - 0.4
    d.stat(sl, rx, 2.1, rw, f2(R.cal_var.best_kappa.max()), "the best kappa over five rules and every threshold", size=60, label_size=16, h=2.2)
    takeaway(d, sl, 5.75, 0.95, "The pilot's rates are not hallucination rates. The judge had to be replaced.", size=20)

    # ── result 2: judge selection ──────────────────────────────────────
    sl = d.new("r_judge", "Result 2: a larger NLI model that reads the whole course agrees best with the experts", section=4)
    rx, rw = chart_left(sl, chart_judges(), 7.75, 4.9)
    caps(d, sl, rx, 2.1, rw, "Selected judge")
    d.tx(sl, rx, 2.45, rw, 0.8, [X["jname"], f"whole course, threshold {M.tau:.2f}"], size=17, color=INK, bold=True, line=1.12, after=0)
    d.stat(sl, rx, 3.55, rw / 2, f2(best.kappa), "Cohen's kappa", size=46, label_size=15, h=1.3)
    d.stat(sl, rx + rw / 2, 3.55, rw / 2, f2(best.auroc), "AUROC", size=46, label_size=15, h=1.3)
    d.tx(sl, rx, 5.15, rw, 0.4, f"Precision {f2(best.precision)}, recall {f2(best.recall)}", size=16, color=MUTED)

    sl = d.new("r_judge2", "What did not help, and how optimistic the number is", section=4)
    jall = X["jc"][X["jc"].group == "all"]
    k_med = float(jall[jall.judge == "mednli_deberta_large"].kappa.max())
    k_llm = float(jall[jall.judge == "bespoke_minicheck_7b"].kappa.max())
    tiles = [(k_med, "MedNLI fine-tuning", f"MedNLI accuracy rose from {X['mednli'][0]} to {X['mednli'][1]}, but agreement with the experts stayed here"),
             (k_llm, "A 7B LLM checker", "It leads a fact-checking leaderboard, and this is its best agreement here"),
             (float(cv.estimate), "Nested cross-validation", "The selected judge, once the optimism of choosing it is removed")]
    for i, (val, head, txt) in enumerate(tiles):
        x = MX + i * (cw3 + 0.25)
        d.box(sl, x, 1.95, cw3, 4.5)
        d.tx(sl, x + 0.35, 2.22, cw3 - 0.7, 0.4, head, size=19, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.35, 2.8, cw3 - 0.7, 1.0, f2(val), size=60, color=INK, bold=True, check=False)
        mw_ = cw3 - 0.7                                 # a meter from 0 to 0.5, with the selected judge marked
        d.meter(sl, x + 0.35, 4.13, mw_, val / 0.5, h=0.16, fill=MID)
        xm = x + 0.35 + mw_ * float(best.kappa) / 0.5
        d.line(sl, xm, 3.98, xm, 4.44, color=ACC, width=2.5, cap=True)
        d.tx(sl, xm - 1.0, 4.48, 2.0, 0.26, f"selected judge {f2(best.kappa)}", size=12, color=MUTED, bold=True, align="c", check=False)
        d.tx(sl, x + 0.35, 5.0, cw3 - 0.7, 1.3, txt, size=16, color=TEXT, line=1.18)

    # ── result 3: unsupported fact rate ────────────────────────────────
    sl = d.new("r_ufr", "Result 3: grounding roughly halves the judge-estimated unsupported fact rate", section=4)
    rx, rw = chart_left(sl, chart_ufr(("E0", "E1", "E1b", "E3"), "main_ufr_llm.png"))
    d.stat(sl, rx, 2.1, rw, f"{rel[0]:.0f} to {rel[-1]:.0f}%", f"lower than the full-note baseline for E1, E1b and E3 (all Holm-adjusted {P(X['pmax'])})", size=50, label_size=16, h=2.2)
    d.stat(sl, rx, 4.5, rw, f2(sum(X["crs"]) / len(X["crs"])), "contradiction rate in every LLM condition: no change", size=40, label_size=16, h=1.5)

    sl = d.new("r_ufr_ref", "Two references, not bounds: copied sentences and the clinician's own text", section=4)
    rx, rw = chart_left(sl, chart_ufr())
    d.stat(sl, rx, 2.05, rw, f3(u["E2"]), "E2 copies sentences verbatim, so this is pure judge error", size=44, label_size=16, h=1.6)
    d.stat(sl, rx, 3.8, rw, f3(u["REF"]), "the clinician's own instructions: they state what the course does not", size=44, label_size=16, h=1.6)
    takeaway(d, sl, 5.6, 1.1, "The metric measures faithfulness to the source, not quality.", size=16, x=rx, w=rw)

    # ── result 4: coverage ─────────────────────────────────────────────
    cov_png = chart_coverage()
    t1, t1b = M.test("coverage_ref", "E1"), M.test("coverage_ref", "E1b")
    sl = d.new("r_cov", "Result 4: excerpts alone cover a third less of the clinician's content, and adding the course restores it", section=4)
    rx, rw = chart_left(sl, cov_png)
    d.stat(sl, rx, 2.05, rw, f"{abs(100 * t1.delta_mean):.1f} points", f"less covered by E1 than by E0 (adjusted {P(t1.p_holm_primary)})", size=44, label_size=16, h=1.55)
    inside = str(eqv.loc["E1b vs E0", "equivalent_at_margin"]) == "True"
    d.stat(sl, rx, 3.75, rw, f"{abs(100 * t1b.delta_mean):.1f} points", "between E1b and E0"
           + (f", inside a margin of {100 * eqv.loc['E1b vs E0', 'margin']:.0f} points" if inside else ""), size=44, label_size=16, h=1.55)
    takeaway(d, sl, 5.55, 1.15, "The loss is associated with withholding the course, not with retrieval as such.", size=15.5, x=rx, w=rw)

    # ── result 5: confounds and citations ──────────────────────────────
    sl = d.new("r_length", "Result 5: shorter summaries do not explain the reduction", section=4)
    d.chart(sl, chart_forest("length"), MX, 1.95, CW, 4.15)
    d.tx(sl, MX, 6.3, CW, 0.4, "A length-matched E0 keeps its rate, and every grounded condition stays below it.", size=18, color=TEXT)

    sl = d.new("r_citeinst", "The instruction to cite does not explain it either", section=4)
    d.chart(sl, chart_forest("cite"), MX, 1.95, CW, 2.95)
    for i, txt in enumerate(("Retrieval lowers the rate even without the instruction to cite.",
                             "For E1b, dropping the instruction raises the rate, so part of its effect does come from citing.")):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 5.15, hw, 1.4, fill=TINT2)
        d.tx(sl, x + 0.35, 5.15, hw - 0.7, 1.4, txt, size=17, color=INK, anchor="m", line=1.15)

    sl = d.new("r_cites", "Do the citations hold?", section=4)
    for i, c in enumerate(("E1", "E1b")):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 4.3)
        d.chip(sl, x + 0.4, 2.3, c, CNAME[c], size=17, h=0.46)
        for j, (lab, v) in enumerate(zip(("of its claims carry a citation", "of those citations are supported by the excerpt"), cite(c))):
            yy = 3.2 + j * 1.45
            d.tx(sl, x + 0.4, yy, 1.9, 0.9, pct0(v), size=44, color=INK, bold=True, anchor="m", check=False)
            d.tx(sl, x + 2.35, yy - 0.05, hw - 2.75, 0.55, lab, size=16.5, color=TEXT, anchor="b", line=1.08)
            d.meter(sl, x + 2.35, yy + 0.62, hw - 2.75, float(v), h=0.16, fill=STRONG, track=TINT2)

    # ── result 6: robustness to the judge ──────────────────────────────
    sl = d.new("r_robust", "Result 6: the reduction holds at a second threshold, under a second judge and in repeated runs", section=4)
    conds = ["E0", "E1", "E1b", "E3"]
    rows = [("Main analysis: selected judge, τ = 0.20", {c: float(u[c]) for c in conds}),
            ("Threshold 0.5 instead of 0.20", {c: float(tv("mtau", c, "UFR (τ=0.5)")) for c in conds}),
            ("Second judge (Bespoke-MiniCheck-7B)", {c: float(tv("magree", c, "UFR 2")) for c in conds}),
            ("Flagged by both judges", {c: float(tv("magree", c, "UFR, both")) for c in conds}),
            ("Flagged by either judge", {c: float(tv("magree", c, "UFR, either")) for c in conds}),
            ("Atomic facts instead of sentences", {c: float(tv("matomic", c, "Fact-level UFR")) for c in conds}),
            ("Mean of three generations (40 courses)", {c: float(tv("mrepvar", c, "Mean UFR")) for c in conds})]
    rx, rw = chart_left(sl, chart_robust(rows), 8.15, 4.9)
    hold = sum(all(v[c] < v["E0"] for c in ("E1", "E1b", "E3")) for _, v in rows)
    d.stat(sl, rx, 2.4, rw, f"{hold} of {len(rows)}", "analyses put every grounded condition below the full-note baseline", size=60, label_size=17, h=2.8)

    sl = d.new("r_robust2", "One conclusion depends on the judge: whether verification adds to retrieval", section=4)
    s31 = st2[(st2.measure == "UFR_second") & (st2.comparison == "E3 vs E1")].iloc[0]
    panels = [("Second judge (Bespoke-MiniCheck-7B)", float(s31.mean_a), float(s31.mean_b), P(s31.p), "E3 is lower than E1"),
              ("Selected judge", float(u["E1"]), float(u["E3"]), P(M.test("UFR", "E3", "E1").p), "E3 and E1 do not differ")]
    base, unit = 5.0, 12.0                              # the columns stand on this line; inches per unit of rate
    for i, (head, e1, e3, pv, verdict) in enumerate(panels):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 3.75)
        d.tx(sl, x + 0.4, 2.2, hw - 0.8, 0.4, head, size=18, color=INK, bold=True, check=False)
        for k, (c, v) in enumerate((("E1", e1), ("E3", e3))):
            bx = x + 0.75 + k * 1.3
            d.col(sl, bx, base - v * unit, 0.8, v * unit, CCOL[c])
            d.tx(sl, bx - 0.25, base - v * unit - 0.36, 1.3, 0.3, f3(v), size=16, color=INK, bold=True, align="c", check=False)
            d.tx(sl, bx - 0.25, base + 0.08, 1.3, 0.3, c, size=15, color=MUTED, bold=True, align="c", check=False)
        d.line(sl, x + 0.5, base, x + 3.1, base, color=LIGHT, width=1.25)
        d.tx(sl, x + 3.5, 3.3, hw - 3.8, 0.6, pv, size=28, color=INK, bold=True, check=False)
        d.tx(sl, x + 3.5, 3.95, hw - 3.8, 0.9, verdict, size=17, color=MUTED, line=1.12)
    takeaway(d, sl, 5.95, 0.8, "The thesis does not claim that verification adds to retrieval.", size=19)

    # ── result 7: other generators ─────────────────────────────────────
    sl = d.new("r_gen", "Result 7: the effect replicates with Llama-3.1-8B but not with Mistral-7B", section=4)
    d.chart(sl, chart_generators(), MX, 1.95, 8.9, 4.75)
    gx_, gw_ = MX + 9.3, CW - 9.3
    d.tx(sl, gx_, 2.4, gw_, 2.3, "The effect of grounding depends on the generator.", size=24, color=INK, bold=True, line=1.12)
    d.tx(sl, gx_, 4.75, gw_, 1.8, "The conclusions are about the models tested, not about grounding in general.", size=16.5, color=MUTED, line=1.2)

    # ── result 8: manual audit ─────────────────────────────────────────
    sl = d.new("r_audit_how", "Result 8: a blind manual audit of the main generator's own summaries", section=4)
    tiles = [(f"{int(aud.loc['pooled', 'n_labeled'])}", "claims labeled by reading the hospital course"), ("60", f"of the {n} courses, drawn at random"),
             ("Blind", "to the condition and to the judge's label"), (f2(agr.loc["claim", "kappa"]), "kappa between two passes over the sample")]
    for i, (big, lab) in enumerate(tiles):
        x = MX + i * (tw4 + 0.25)
        d.box(sl, x, 2.3, tw4, 3.3)
        d.tx(sl, x + 0.3, 2.75, tw4 - 0.6, 1.0, big, size=50, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.3, 4.0, tw4 - 0.6, 1.3, lab, size=17, color=MUTED, line=1.18)

    sl = d.new("r_audit", "The judge undercounts: manual reading finds more unsupported claims", section=4)
    rx, rw = chart_left(sl, chart_audit())
    a0 = aud.loc["E0"]
    d.stat(sl, rx, 2.1, rw, f"{pct0(a0.manual_unsupported)} vs {pct0(u['E0'])}", "E0 by manual reading and by the judge", size=44, label_size=16, h=1.6)
    d.stat(sl, rx, 4.0, rw, pct0(aud.loc["pooled", "partly_share_of_unsupported"]), "of the manually unsupported claims are partial: one fact of several fails",
           size=44, label_size=16, h=2.0)

    sl = d.new("r_audit2", "By manual reading, the direction is confirmed for E1b", section=4)
    dd = aud.loc["E1b minus E0"]
    rx, rw = chart_left(sl, chart_audit_diff(aud), 8.6, 3.75)
    d.stat(sl, rx, 2.2, rw, f"{100 * dd['diff']:+.1f} points", f"E1b against E0 by manual reading (95% CI {100 * dd.diff_lo:+.1f} to {100 * dd.diff_hi:+.1f})",
           size=36, label_size=16, h=2.2)
    takeaway(d, sl, 5.95, 0.8, "For E1 and E3 the interval includes zero.", size=19)


def slides_discussion(d, R, M, X):
    cal, best, cv, aud, ev = (X[k] for k in ("cal", "best", "cv", "aud", "ev"))
    n, u, dlt = X["n"], X["u"], X["dlt"]
    cw3 = (CW - 2 * 0.25) / 3
    hw = (CW - 0.3) / 2
    tw4 = (CW - 3 * 0.25) / 4
    a0 = aud.loc["E0"]

    def cited(c):
        r = M.base.loc[c]
        return r.cited_claims / r.claims_total

    # ── what the judge's errors mean ───────────────────────────────────
    kinds = [("Flagged by the judge, unsupported", ACC, None), ("Flagged by the judge, supported", CARD, ACC),
             ("Accepted by the judge, unsupported", STRONG, None), ("Accepted by the judge, supported", "E4E6EA", None)]
    cnt = {c: waffle_counts(aud.loc[c]) for c in ("E0", "E1b")}
    flagged, manual = (lambda t: t[0] + t[1]), (lambda t: t[0] + t[2])

    sl = d.new("d_errors", "Why a moderate judge can still carry a comparison", section=5)
    d.chart(sl, chart_waffle(aud, ("E0",), "audit_waffle_e0.png"), MX, 1.95, 4.8, 4.75)
    wx, ww = MX + 5.1, CW - 5.1
    d.box(sl, wx, 1.95, ww, 4.75)
    caps(d, sl, wx + 0.4, 2.2, ww - 0.8, "100 claims of the baseline, E0")
    for i, (lab, fill, edge) in enumerate(kinds):
        y = 2.65 + i * 0.58
        d.squares(sl, wx + 0.4, y + 0.1, [(fill, edge)], s=0.32, line_w=1.75)
        d.tx(sl, wx + 0.95, y, ww - 2.3, 0.52, lab, size=17, anchor="m", check=False)
        d.tx(sl, wx + ww - 1.3, y, 0.9, 0.52, str(cnt["E0"][i]), size=20, color=INK, bold=True, align="r", anchor="m", check=False)
    d.line(sl, wx + 0.4, 5.1, wx + ww - 0.4, 5.1, color=LIGHT, width=1.25)
    for i, (lab, pick) in enumerate((("The judge flags", flagged), ("Manual reading finds unsupported", manual))):
        y = 5.25 + i * 0.62
        d.tx(sl, wx + 0.4, y, ww - 2.2, 0.56, lab, size=19, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, wx + ww - 1.5, y, 1.1, 0.56, str(pick(cnt["E0"])), size=30, color=INK, bold=True, align="r", anchor="m", check=False)

    sl = d.new("d_formula", "If the judge errs the same way in two conditions, the direction survives", section=5)
    d.box(sl, MX, 1.95, CW, 2.1)
    d.tx(sl, MX, 1.95, CW, 2.1, "π = q + f (p − q)", size=54, color=INK, bold=True, align="c", anchor="m", check=False)
    syms = [("π", "true share of unsupported claims"), ("f", "share of claims the judge flags"), ("p", "precision of the judge's flags"),
            ("q", "unsupported share of the claims it accepts")]
    for i, (sym, txt) in enumerate(syms):
        x = MX + i * (tw4 + 0.25)
        d.box(sl, x, 4.3, tw4, 1.45)
        d.tx(sl, x + 0.3, 4.3, 0.6, 1.45, sym, size=30, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, x + 0.95, 4.3, tw4 - 1.2, 1.45, txt, size=15.5, color=TEXT, anchor="m", line=1.15)
    d.tx(sl, MX, 6.05, CW, 0.5, "With the same p and q in two conditions, the true difference keeps its direction and shrinks.", size=18, color=TEXT)

    sl = d.new("d_pq", "Here p and q differ between conditions, so the sizes are uncertain", section=5)
    for i, (sym, lab, col) in enumerate((("p", "precision of the judge's flags", "precision"), ("q", "unsupported share of accepted claims", "false_omission"))):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 3.7)
        d.tx(sl, x + 0.4, 2.15, 0.6, 0.6, sym, size=30, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, x + 1.0, 2.15, hw - 1.4, 0.6, lab, size=17, color=MUTED, anchor="m", check=False)
        for j, c in enumerate(("E0", "E1", "E1b", "E3")):
            y = 3.05 + j * 0.6
            d.chip(sl, x + 0.4, y + 0.06, c, size=14, h=0.36)
            d.meter(sl, x + 1.75, y + 0.16, hw - 3.1, float(aud.loc[c, col]), h=0.16, fill=STRONG, track=TINT2)
            d.tx(sl, x + hw - 1.15, y, 0.75, 0.48, f2(aud.loc[c, col]), size=17, color=INK, bold=True, align="r", anchor="m", check=False)
    spread = max(u[c] for c in ("E1", "E1b", "E3")) - min(u[c] for c in ("E1", "E1b", "E3"))
    takeaway(d, sl, 5.9, 0.85, f"E1, E1b and E3 are at most {spread:.3f} apart, so they are not ranked.", size=19)

    # ── answers to the research questions ──────────────────────────────
    def verdict(sl, y, items, hgt=None):
        hgt = hgt or (0.5 + 0.62 * len(items))
        d.box(sl, MX, y, CW, hgt, fill=TINT2)
        yy = y + (hgt - 0.62 * len(items)) / 2
        for h_, txt in items:
            if h_:
                d.pill(sl, MX + 0.35, yy + 0.09, h_, size=14, h=0.42, fill=SLATE, color=WHITE, line=None, bold=True, w=0.8)
            d.tx(sl, MX + (1.4 if h_ else 0.35), yy, CW - (1.8 if h_ else 0.7), 0.62, txt, size=17, color=INK if h_ else MUTED, anchor="m", line=1.1)
            yy += 0.62

    sl = d.new("a_rq1", "Answer to RQ1: how often are statements unsupported?", section=5)
    for i, (big, lab) in enumerate(((pct(u["E0"]), "judge-estimated, in full-note summaries"),
                                    (pct(a0.manual_unsupported), f"by manual reading (95% CI {pct(a0.manual_unsupported_lo)} to {pct(a0.manual_unsupported_hi)})"))):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 2.6)
        d.tx(sl, x + 0.4, 2.2, hw - 0.8, 1.1, big, size=60, color=INK, bold=True, check=False)
        d.tx(sl, x + 0.4, 3.45, hw - 0.8, 0.9, lab, size=17, color=MUTED, line=1.15)
    d.tx(sl, MX, 4.72, CW, 0.34, f"Contradicted: {pct(M.mean('E0', 'CR'))}, judge-estimated", size=16, color=MUTED)
    verdict(sl, 5.25, [("H1", f"Supported in part. Measurable, and confirmed by the audit, but the judge agrees with experts only moderately (kappa {f2(best.kappa)}).")], hgt=1.3)

    sl = d.new("a_rq2", "Answer to RQ2: which statements, and how does the judge err?", section=5)
    for i, (lab, items) in enumerate((("Unsupported statements are mostly", ("follow-up instructions", "generic advice", "sentences where one fact of several fails")),
                                      ("The judge", ("misses partial errors", "over-flags negated lists and paraphrase")))):
        x = MX + i * (hw + 0.3)
        d.box(sl, x, 1.95, hw, 3.5)
        caps(d, sl, x + 0.4, 2.25, hw - 0.8, lab)
        for k, item in enumerate(items):
            d.pill(sl, x + 0.4, 2.8 + k * 0.78, item, size=17, h=0.56, fill=TINT, color=INK, line=None)
    verdict(sl, 5.7, [(None, "Descriptive question, no hypothesis.")], hgt=0.85)

    sl = d.new("a_rq3", "Answer to RQ3: does grounding help, and at what cost?", section=5)
    lw = 5.3
    d.box(sl, MX, 1.95, lw, 3.0)
    d.tx(sl, MX + 0.4, 2.25, lw - 0.8, 0.9, f"{dlt[0]:.3f} to {dlt[-1]:.3f}", size=40, color=INK, bold=True, check=False)
    d.tx(sl, MX + 0.4, 3.25, lw - 0.8, 1.4, "lower unsupported fact rate for E1, E1b and E3 with the main generator", size=17, color=MUTED, line=1.18)
    rx, rw = MX + lw + 0.3, CW - lw - 0.3
    d.box(sl, rx, 1.95, rw, 3.0)
    for k, (mark, txt) in enumerate((("yes", "Replicates with Llama-3.1-8B"), ("no", "Does not replicate with Mistral-7B"), ("", "Contradictions: no change"),
                                     ("", "Coverage: kept only with the course"))):
        yy = 2.2 + k * 0.64
        if mark == "yes":
            d.check(sl, rx + 0.42, yy + 0.13, 0.28, color=STRONG, width=3)
        elif mark == "no":
            d.cross(sl, rx + 0.46, yy + 0.17, 0.2, color=ACC, width=3)
        else:
            d.dot(sl, rx + 0.49, yy + 0.2, 0.14, MID)
        d.tx(sl, rx + 1.0, yy, rw - 1.3, 0.54, txt, size=18, color=TEXT, anchor="m", check=False)
    verdict(sl, 5.15, [("H2", "Supported for unsupported facts, not for contradictions."), ("H3", "Open: it depends on the judge.")], hgt=1.5)

    # ── trade-off ──────────────────────────────────────────────────────
    sl = d.new("d_tradeoff", "The trade-off: RAG that also sees the whole course lowers unsupported claims without omitting more", section=5)
    d.chart(sl, chart_tradeoff(), MX, 1.95, 7.0, 4.75)
    tx_, tw_ = MX + 7.4, CW - 7.4
    for i, (cs, txt) in enumerate(((("E0",), "Complete, but with the highest rate"), (("E1", "E3"), "Fewer unsupported claims, but more omitted"),
                                   (("E1b",), "Fewer unsupported claims, and coverage kept"))):
        y = 2.25 + i * 1.5
        xx = tx_
        for c in cs:
            xx += d.chip(sl, xx, y, c, size=15, h=0.4) + 0.12
        d.tx(sl, tx_, y + 0.52, tw_, 0.8, txt, size=18, color=INK if cs == ("E1b",) else TEXT, bold=cs == ("E1b",), line=1.12)

    sl = d.new("d_favor", "What the results favor, and under which conditions", section=5)
    favor = [("E1b", "The design these results favor when the note fits the context window, with clinician review"),
             ("E1", "Long notes, when omission is acceptable and a reviewer checks completeness"),
             ("E3", "When omission is safer than invention and a clinician fills the gaps"),
             ("E0", "Drafts that a clinician reviews and edits"),
             ("E2", "Clinician-facing digests; not readable for patients")]
    d.box(sl, MX, 1.95, CW, 3.75)
    for i, (c, txt) in enumerate(favor):
        y = 2.1 + i * 0.7
        if i:
            d.line(sl, MX + 0.35, y - 0.03, MX + CW - 0.35, y - 0.03, color=HAIR, width=0.75)
        d.chip(sl, MX + 0.4, y + 0.14, c, size=15, h=0.38)
        d.tx(sl, MX + 1.7, y, CW - 2.1, 0.64, txt, size=17, color=INK if i == 0 else TEXT, bold=i == 0, anchor="m", line=1.08)
    takeaway(d, sl, 5.9, 0.85, f"Conditional on judge-estimated rates, one corpus and the generators tested. Only {pct0(cited('E1b'))} of E1b's claims carry a citation.", size=15.5)

    # ── comparison with existing approaches ────────────────────────────
    jall = X["jc"][X["jc"].group == "all"]
    mc = jall[jall.judge.isin(["minicheck_deberta", "minicheck_roberta"])].kappa
    var = R.cal_var[R.cal_var.group == "all"].set_index("variant")
    fbest = float(jall.sort_values("kappa", ascending=False).iloc[0].f1)
    k7b, kmed = (float(jall[jall.judge == j].kappa.max()) for j in ("bespoke_minicheck_7b", "mednli_deberta_large"))
    ksum = float(var.loc["maxall", "kappa"])
    g1 = (f"On the expert-annotated summaries\n{int(cal.n_sentences):,} sentences",
          [("Small NLI model in the SummaC design\n(Laban et al., 2022)", ksum, f2(ksum), False),
           ("MiniCheck encoders\n(Tang et al., 2024)", float(mc.max()), f"{f2(mc.min())} to {f2(mc.max())}", False),
           ("Bespoke-MiniCheck-7B,\nan LLM checker", k7b, f2(k7b), False),
           ("DeBERTa-v3-large NLI\nfine-tuned on MedNLI", kmed, f2(kmed), False),
           ("Selected judge: DeBERTa-v3-large NLI,\nwhole course", float(best.kappa), f2(best.kappa), True)])
    g2 = (f"On VeriFact-BHC\n{int(ev.loc['all', 'n']):,} propositions",
          [("Best LLM rater of the dataset, 70B model\n(Chung et al., 2026)", 0.48, "0.48", False),
           ("Selected judge, unchanged", float(ev.loc["all", "kappa"]), f2(ev.loc["all", "kappa"]), True)])

    sl = d.new("d_existing", "Compared with existing approaches: benchmark strength did not transfer to patient summaries", section=5)
    d.chart(sl, chart_existing([g1], "existing_expert.png"), MX, 1.95, 8.3, 4.2)
    ex_, ew_ = MX + 8.7, CW - 8.7
    caps(d, sl, ex_, 2.1, ew_, "For context")
    d.stat(sl, ex_, 2.45, ew_, "0.16 to 0.20", "span F1 of GPT-4 as a detector (Hegselmann et al., 2024)", size=30, label_size=15, h=1.5)
    d.stat(sl, ex_, 4.05, ew_, f2(fbest), "sentence F1 of the selected judge", size=30, label_size=15, h=1.1)
    d.tx(sl, ex_, 5.3, ew_, 0.8, "The units differ, so this is context, not a ranking.", size=14, color=MUTED, line=1.15)
    d.tx(sl, MX, 6.35, CW, 0.4, "No checker passes a general NLI model, and MedNLI fine-tuning did not move agreement.", size=15.5, color=TEXT)

    sl = d.new("d_existing2", "On other clinician labels, a much larger model agrees better than the laptop judge", section=5)
    d.chart(sl, chart_existing([g2], "existing_verifact.png"), MX, 1.95, 8.3, 2.75)
    ex_, ew_ = MX + 8.7, CW - 8.7
    d.stat(sl, ex_, 2.1, ew_, f2(ev.loc["all", "auroc"]), "AUROC of the selected judge on VeriFact-BHC", size=44, label_size=16, h=1.8)
    takeaway(d, sl, 5.0, 1.2, "Chain-of-Verification adds to RAG under only one of the two judges, at about 10 model calls per note.", size=18)

    # ── limitations ────────────────────────────────────────────────────
    gens = generator_runs()
    n_rep = sum(len(sig) == 3 for _, _, sig in gens)

    def limits(key, title, tiles):
        sl = d.new(key, title, section=5)
        for i, (head_, txt, pic) in enumerate(tiles):
            x = MX + i * (hw + 0.3)
            d.box(sl, x, 1.95, hw, 4.4)
            pic(sl, x + 0.45, 2.4)
            d.tx(sl, x + 0.45, 3.5, hw - 0.9, 1.0, head_, size=25, color=INK, bold=True, anchor="b", line=1.05)
            d.tx(sl, x + 0.45, 4.7, hw - 0.9, 1.5, txt, size=19, line=1.18)

    def p_meter(sl, px, py):
        w = 4.4
        d.meter(sl, px, py + 0.42, w, float(best.kappa), h=0.2, fill=STRONG, track=TINT2)
        d.tx(sl, px, py + 0.08, 0.4, 0.26, "0", size=13, color=MUTED, check=False)
        d.tx(sl, px + w - 0.4, py + 0.08, 0.4, 0.26, "1", size=13, color=MUTED, align="r", check=False)
        d.tx(sl, px + w * float(best.kappa) - 0.5, py + 0.08, 1.0, 0.26, f2(best.kappa), size=13, color=INK, bold=True, align="c", check=False)

    def p_reverse(sl, px, py):
        d.doc(sl, px, py, 0.62, 0.8, rows=4)
        d.doc(sl, px + 1.6, py, 0.62, 0.8, rows=4)
        d.line(sl, px + 0.76, py + 0.27, px + 1.46, py + 0.27, color=MID, width=1.75, arrow=True)
        d.line(sl, px + 1.46, py + 0.55, px + 0.76, py + 0.55, color=ACC, width=1.75, arrow=True)

    def p_person(sl, px, py):
        d.dot(sl, px + 0.17, py, 0.36, STRONG)
        d.col(sl, px, py + 0.42, 0.7, 0.42, STRONG)

    def p_sample(sl, px, py):
        d.squares(sl, px, py + 0.02, [(STRONG, None)] * 60 + [(LIGHT, None)] * (n - 60), per_row=22, s=0.13, gap=0.045)

    def p_section(sl, px, py):
        d.doc(sl, px, py, 0.66, 0.84, rows=5, mark=(2,))

    def p_gens(sl, px, py):
        for k in range(len(gens)):
            d.dot(sl, px + k * 0.72, py + 0.12, 0.56, STRONG if k < n_rep else CARD, line=None if k < n_rep else MID)

    limits("lim_judge", "Limitations 1 of 3: the judge",
           [("Moderate judge", f"Kappa {f2(best.kappa)}, about {f2(cv.estimate)} after correcting for the selection.", p_meter),
            ("Coverage and citations", "The judge runs in directions not validated against experts.", p_reverse)])
    limits("lim_audit", "Limitations 2 of 3: the audit",
           [("One annotator, not a clinician", "The audit checks textual support, not clinical correctness.", p_person),
            (f"60 of {n} courses audited", "Most differences between conditions stay unresolved.", p_sample)])
    limits("lim_data", "Limitations 3 of 3: the data and the models",
           [("One hospital, one note section", "7B models in 4-bit, one prompt, one retriever, one verifier.", p_section),
            (f"{n_rep} of {len(gens)} generators", "No effect with Mistral-7B, and the reason is not known.", p_gens)])

    sl = d.new("scope", "Scope of the claims", section=5)
    d.box(sl, MX, 2.15, CW, 3.7)
    d.tx(sl, MX + 0.7, 2.35, CW - 1.4, 2.2, "No claim about the clinical safety or usefulness of any summary.", size=34, color=INK, bold=True, anchor="m", line=1.1)
    d.tx(sl, MX + 0.7, 4.6, CW - 1.4, 0.9, "A lower rate can come from saying less.", size=22, color=MUTED, anchor="m")


def slides_conclusions(d, M, X):
    cal, best, aud, u, rel = X["cal"], X["best"], X["aud"], X["u"], X["rel"]
    a0 = aud.loc["E0"]
    t1, t1b = M.test("coverage_ref", "E1"), M.test("coverage_ref", "E1b")
    concl = [(f2(cal.any_kappa), "pilot judge, kappa", "Validate the evaluator first", "The off-the-shelf judge agreed with experts at chance level."),
             (f2(best.kappa), "selected judge, kappa", "A laptop-scale judge reaches moderate agreement", "MedNLI fine-tuning and a 7B LLM checker did not do better."),
             (f"{rel[0]:.0f} to {rel[-1]:.0f}%", "lower unsupported fact rate", "Grounding lowers the judge-estimated unsupported fact rate",
              "With Llama-3.1-8B too, not with Mistral-7B."),
             (f"{abs(100 * t1.delta_mean):.0f} vs {abs(100 * t1b.delta_mean):.0f} points", "coverage lost by E1 and by E1b", "The cost is omission, and it is avoidable",
              "RAG that also sees the course keeps coverage."),
             (f"{pct0(a0.manual_unsupported)} vs {pct0(u['E0'])}", "E0 by manual reading and by the judge", "The audit keeps the claims honest",
              "Direction confirmed for E1b. The judge undercounts.")]
    # ── contributions ──────────────────────────────────────────────────
    n = X["n"]
    sl = d.new("contrib", "Main contributions", section=6)
    contribs = [("An open claim-level evaluation pipeline", "built from open-weight parts; the evaluation artifacts it exposed were found and corrected"),
                ("A validation of NLI judges", f"against {int(cal.n_sentences):,} expert-annotated sentences, with the optimism of the selection measured"),
                ("A controlled comparison of five conditions", f"on {n} MIMIC-IV hospital courses, run entirely on a laptop"),
                ("Checks on the judge itself", "a blind manual audit, an external check on clinician-labeled data, two more generators")]
    hw, hh = (CW - 0.3) / 2, 2.25
    for i, (lead, txt) in enumerate(contribs):
        x, y = MX + (i % 2) * (hw + 0.3), 1.95 + (i // 2) * (hh + 0.3)
        d.box(sl, x, y, hw, hh)
        d.num(sl, x + 0.35, y + 0.35, i + 1, d=0.52, size=18)
        d.tx(sl, x + 1.1, y + 0.26, hw - 1.45, 0.8, lead, size=20, color=INK, bold=True, line=1.08, anchor="m")
        d.tx(sl, x + 1.1, y + 1.14, hw - 1.45, 0.98, txt, size=15.5, color=MUTED, line=1.18)

    # ── conclusions ────────────────────────────────────────────────────
    sl = d.new("conclusions", "Conclusions", section=6)
    d.box(sl, MX, 1.95, CW, 4.75)
    for i, (big, cap_, lead, txt) in enumerate(concl):
        y = 2.1 + i * 0.9
        if i:
            d.line(sl, MX + 0.3, y - 0.02, MX + CW - 0.3, y - 0.02, color=HAIR, width=0.75)
        d.num(sl, MX + 0.3, y + 0.19, i + 1, d=0.46, size=15)
        d.tx(sl, MX + 1.0, y, 3.4, 0.84, big, size=24, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, MX + 4.5, y, CW - 4.9, 0.84, lead + ".", size=19, color=INK, bold=True, anchor="m", line=1.05)

    # ── implications and future work ───────────────────────────────────
    sl = d.new("implications", "Implications for building and evaluating clinical summarizers", section=6)
    advice = ["Validate the judge against clinician labels first, and again on the system's own outputs", "Report counts, length and coverage with every rate, and paired differences with intervals",
              "Separate confounds with length-matched and instruction-matched variants", "Treat extractive and human-written text as references, not as bounds"]
    d.box(sl, MX, 1.95, CW, 4.5)
    for i, txt in enumerate(advice):
        y = 2.2 + i * 1.05
        ix = MX + 0.45
        d.dot(sl, ix, y, 0.74, TINT)
        if i == 0:
            d.check(sl, ix + 0.19, y + 0.19, 0.36, color=ACC, width=3)
        elif i == 1:
            for k, hh in enumerate((0.2, 0.34, 0.27)):
                d.box(sl, ix + 0.2 + k * 0.13, y + 0.54 - hh, 0.09, hh, fill=STRONG, radius=0.02)
        elif i == 2:
            d.bars(sl, ix + 0.19, y + 0.26, 0.36, n=2, h=0.09, gap=0.08, color=STRONG, widths=(1, 0.6))
        else:
            d.dot(sl, ix + 0.23, y + 0.23, 0.28, CARD, line=STRONG)
        d.tx(sl, MX + 1.55, y - 0.05, CW - 2.1, 0.84, txt, size=20, anchor="m", line=1.12)

    sl = d.new("future", "Future work", section=6)
    nxt = ["Two clinicians annotate the generator's own outputs, for clinically validated error rates", "Better judges for patient-facing paraphrase: LLM judges calibrated on expert labels",
           "Larger open-weight generators, several generations per course, more hospitals and note types",
           "A clinician study of how much the omitted sentences matter, and gold answers for question answering"]
    d.box(sl, MX, 1.95, CW, 4.5)
    d.line(sl, MX + 0.71, 2.5, MX + 0.71, 2.5 + 3 * 1.05, color=LIGHT, width=2)
    for i, txt in enumerate(nxt):
        y = 2.2 + i * 1.05
        d.num(sl, MX + 0.45, y + 0.11, i + 1, d=0.52, size=18)
        d.tx(sl, MX + 1.55, y - 0.05, CW - 2.1, 0.84, txt, size=20, anchor="m", line=1.12)

    # ── closing ────────────────────────────────────────────────────────
    sl = d.new("closing", kind="plain")
    d.tx(sl, 0.9, 1.45, 11.5, 1.0, "Thank you. Questions?", size=44, color=INK, bold=True, check=False)
    motif = [(LIGHT, None)] * 26
    motif[8] = motif[19] = (BG, ACC)
    motif[14] = (ACC, None)
    d.squares(sl, 0.93, 2.72, motif, per_row=26, s=0.2, gap=0.09, line_w=1.5)
    d.tx(sl, 0.9, 3.3, 11.2, 1.6, "Validate the judge first. Once it can be trusted, grounding helps, and letting the model see the whole note avoids the omission cost.",
         size=22, color=TEXT, line=1.22)
    d.tx(sl, 0.9, 5.6, 11.5, 0.9, ["Code and aggregate results: github.com/slay-a/medical-hallucination-eval", "Srilaya Ponangi, California State University, Northridge"],
         size=15, color=MUTED, after=5)


def build():
    R, M = Results(), Mimic()
    T = thesis_tables(R)
    d = Deck()
    X = context(R, M)
    slides_opening(d, X)
    slides_intro(d, X)
    slides_background(d, X)
    slides_method(d, R, M, X)
    slides_results(d, R, M, T, X)
    slides_discussion(d, R, M, X)
    slides_conclusions(d, M, X)
    for i in range(1, len(SECTIONS) + 1):
        divider(d, i)
    d.reorder(MAIN_ORDER)
    return d, R, M, T, X


# ───────────────────────────── backup slides ─────────────────────────────
BACKUP = [("bk_judges", "All candidate judges against the expert annotations"),
          ("bk_optimism", "How optimistic is the selected judge's agreement?"),
          ("bk_verifact", "External check of the judge on VeriFact-BHC"),
          ("bk_pilot", "Pilot study: the five conditions under the pilot judge"),
          ("bk_artifacts", "Evaluation artifacts found and corrected"),
          ("bk_desc", "Main study: length, claims and abstentions"),
          ("bk_rates", "Main study: rates with intervals and paired tests"),
          ("bk_cov", "Main study: coverage of the clinician's instructions"),
          ("bk_variants", "Variants: length and the citation instruction"),
          ("bk_cite", "Citation coverage, accuracy and reliability"),
          ("bk_abl", "Retrieval ablations"),
          ("bk_rep", "Run-to-run variation over three generations"),
          ("bk_audit", "Manual audit: design and the judge's error rates"),
          ("bk_second", "Second judge: agreement with the selected judge"),
          ("bk_atomic", "Atomic facts, evidence windows, verifier retrieval"),
          ("bk_gen", "Other generators: rates and summary length"),
          ("bk_qa", "Question-answering pilot"),
          ("bk_stats", "Statistical analysis in detail"),
          ("bk_ethics", "Ethics, data governance and reproducibility")]


def build_backup(d, R, M, T, X):
    aud, agr, ev, jsb, n = X["aud"], X["agr"], X["ev"], X["jsb"], X["n"]

    def new(key, sub=None):
        title = dict(BACKUP)[key]
        return d.new(key, title, kind="backup", kicker="Backup" + (f"   ·   {sub}" if sub else ""))

    def fit(columns, rows, widths, top, limit, sizes=(14, 13.5, 13, 12.5, 12, 11.5, 11, 10.5, 10), pad=0.07):
        total = float(sum(widths)); cw = [CW * v / total for v in widths]
        for s in sizes:
            if top + sum(table_heights(columns, rows, cw, s, s, pad)) <= limit:
                return s
        return sizes[-1]

    def ttable(sl, label, top=1.95, limit=6.75, widths=None, cols=None, pad=0.07):
        """Place a thesis table, optionally only some of its columns, at the largest font size that fits."""
        t = T[label]
        idx = [t["columns"].index(c) for c in cols] if cols else list(range(len(t["columns"])))
        columns = [t["columns"][j] for j in idx]
        rows = [[r[j] for j in idx] for r in t["rows"]]
        w = widths or [t["widths"][j] for j in idx]
        s = fit(columns, rows, w, top, limit, pad=pad)
        return d.table(sl, MX, top, CW, columns, rows, w, size=s, pad=pad, max_h=limit + 0.05)

    def note(sl, y, items, size=13.5):
        d.bullets(sl, MX, y, CW, 6.8 - y, items, size=size, after=6)

    # index
    first = len(d.meta) + 2                         # the slide number of the first backup slide after this index
    sl = d.new("bk_index", "Backup slides: type the number and press Enter to jump", kind="backup", kicker="Backup")
    half = (len(BACKUP) + 1) // 2
    d.box(sl, MX, 1.85, CW, half * 0.47 + 0.2)
    for i, (key, title) in enumerate(BACKUP):
        col, row = divmod(i, half)
        x, y = MX + 0.3 + col * (CW / 2), 1.95 + row * 0.47
        d.tx(sl, x, y, 0.8, 0.36, f"{first + i}", size=15, color=INK, bold=True, anchor="m", check=False)
        d.tx(sl, x + 0.8, y, CW / 2 - 1.2, 0.36, title, size=13.5, anchor="m", check=False)

    sl = new("bk_judges", f"thesis Table {T['judges']['number']}")
    ttable(sl, "judges")

    sl = new("bk_optimism", f"thesis Section 6.5 and Table {T['persource']['number']}")
    g = lambda a, m: jsb[(jsb.analysis == a) & (jsb.metric == m)].iloc[0]
    boot, cvk, cva = g("selected_judge_course_bootstrap", "kappa"), g("nested_cv_selection_procedure", "kappa"), g("nested_cv_selection_procedure", "auroc")
    freq = csv("judge_selection_frequency.csv")
    share = float(freq.share.max()) if freq is not None else None
    note(sl, 1.95, [f"Apparent kappa on all sentences: {f2(boot.estimate)} (95% interval {f2(boot.lo)} to {f2(boot.hi)} when courses are resampled)",
                    f"Nested cross-validation of the whole selection, 5 folds of courses repeated 20 times: kappa {f2(cvk.estimate)} ({f2(cvk.lo)} to {f2(cvk.hi)}), AUROC {f2(cva.estimate)}"
                    + (f"; the same configuration was chosen in {pct0(share)} of the folds" if share else ""),
                    f"Transfer across patients: chosen on doctor-written sentences and tested on LLM-generated summaries of other patients, kappa {f2(g('select_on_doctor_test_on_generated', 'kappa').estimate)}; "
                    f"the reverse, kappa {f2(g('select_on_generated_test_on_doctor', 'kappa').estimate)}",
                    "The LLM-generated half of the expert set comes from 20 other courses; the doctor-written half belongs to the main-study courses"], size=14)
    ttable(sl, "persource", top=4.25)

    sl = new("bk_verifact", "thesis Section 6.7")
    sub = [("all", "All propositions"), ("author=human", "Human-written summaries"), ("author=llm", "LLM-written summaries"), ("type=claim", "Claims"), ("type=sentence", "Sentences")]
    rows = [[lab, f"{int(ev.loc[k, 'n']):,}", f2(ev.loc[k, "auroc"]), f2(ev.loc[k, "sensitivity"]), f2(ev.loc[k, "specificity"]), f2(ev.loc[k, "kappa"])] for k, lab in sub if k in ev.index]
    bottom = d.table(sl, MX, 1.95, CW, ["Propositions", "n", "AUROC", "Sensitivity", "Specificity", "Kappa"], rows, widths=[3.4, 1.4, 1.5, 1.6, 1.6, 1.4], size=14, pad=0.08,
                     align=["l", "c", "c", "c", "c", "c"])
    al = ev.loc["all"]
    ai = csv("verifact_ai_raters.csv")
    items = [f"The selected judge, unchanged, on {int(al.n):,} clinician-labeled propositions from brief hospital courses of 100 MIMIC-III patients, written by clinicians or by LLMs",
             "MIMIC-III comes from the same hospital as MIMIC-IV, in earlier years. The summaries, the models and the labeling clinicians are different",
             f"Each proposition is scored against its {int(al.topk)} best-matching 400-token windows of the admission's notes, at the main study's threshold"]
    if ai is not None and "reference_only_admission" in ai.columns:
        same = ai[ai.reference_only_admission.astype(str).str.lower() == "true"]
        if len(same):
            items.append(f"For context: the best of the dataset's own {len(ai)} LLM rater configurations with the same evidence reaches kappa {f2(same.iloc[0].kappa)}")
    items.append("The judge still separates supported from unsupported statements on another corpus, with low sensitivity. The check does not re-validate the threshold")
    note(sl, bottom + 0.25, items, size=13)

    sl = new("bk_pilot", f"thesis Table {T['desc']['number']}")
    bottom = ttable(sl, "desc", limit=4.6)
    t31, t10 = R.test("UFR", "E0", "E3"), R.test("CR", "E0", "E1")
    note(sl, bottom + 0.3, ["These rates come from the pilot judge, which agrees with the experts at chance level, so they are not hallucination rates",
                            f"Under that judge, verification lowered UFR from {f3(R.mean('E0', 'UFR'))} to {f3(R.mean('E3', 'UFR'))} (adjusted {P(t31.p_holm)}), mainly by deleting claims",
                            f"RAG lowered CR by {abs(t10.rel_change_mean):.0f}% ({P(t10.p)}), which does not survive the Holm adjustment over the twenty pilot tests ({P(t10.p_holm)})",
                            "The extractive condition copies sentences, so every flag on it is a judge error"])

    sl = new("bk_artifacts", f"thesis Sections 3.5 and 3.6, Table {T['headerfilter']['number']}")
    bottom = ttable(sl, "headerfilter", limit=4.4)
    sc = csv("mimic_splitting_check.csv")
    items = ["Header lines such as \"Key Findings:\" were counted as claims and often labeled Contradicted; removing them shrank the first reported effect of RAG on CR",
             "Abstentions (\"Not stated in the note.\") were labeled Contradicted; they are now counted separately"]
    if sc is not None:
        b = sc[sc.variant == "base"].set_index("condition")
        items.append(f"Citation markers in the main study: splitting the raw text dropped {pct0(b.loc['E1', 'markers_dropped_share'])} of E1's {int(b.loc['E1', 'marker_groups'])} marker groups and "
                     f"moved {pct0(b.loc['E1', 'markers_moved_share'])} to the next sentence ({pct0(b.loc['E1b', 'markers_dropped_share'])} and {pct0(b.loc['E1b', 'markers_moved_share'])} of E1b's "
                     f"{int(b.loc['E1b', 'marker_groups'])}). Markers are now removed before splitting")
        items.append(f"Headers merged into the next sentence: {pct(b.loc['E0', 'merged_header_share'])} of the E0 claims and {pct(b.loc['E3', 'merged_header_share'])} of the E3 claims; now stripped from every claim")
    note(sl, bottom + 0.3, items)

    sl = new("bk_desc", f"thesis Table {T['mdesc']['number']}")
    bottom = ttable(sl, "mdesc")
    note(sl, bottom + 0.3, ["E1, E1b and E3 state fewer unsupported claims than E0, and also fewer supported ones: part of their lower rate comes from saying less",
                            "That is why every rate is reported with its counts and with coverage"])

    sl = new("bk_rates", f"thesis Table {T['mrates']['number']}")
    ttable(sl, "mrates")

    sl = new("bk_cov", f"thesis Table {T['mcov']['number']}")
    bottom = ttable(sl, "mcov")
    e = X["eqv"].loc["E1b vs E0"]
    note(sl, bottom + 0.3, ["Coverage is the share of the clinician's sentences that the summary supports, with the judge run in the reverse direction",
                            f"E1b against E0: the 90% interval of the difference ({100 * e.ci90_low:+.1f} to {100 * e.ci90_high:+.1f} points) lies inside the margin of {100 * e.margin:.0f} points (post hoc check)",
                            "The clinician's instructions contain content the course does not, so complete coverage is neither attainable nor desirable"])

    sl = new("bk_variants", f"thesis Table {T['mvar']['number']}")
    ttable(sl, "mvar")

    sl = new("bk_cite", f"thesis Table {T['mcite']['number']}")
    bottom = ttable(sl, "mcite")
    ac = csv("audit_citations.csv")
    items = ["E1b cites few claims by design: it also sees the whole course, and statements drawn from outside the three excerpts have no excerpt to cite"]
    if ac is not None:
        a = ac.set_index("condition")
        items.append("Manual check of " + f"{int(a.loc['E1', 'n'])} citations per condition: " + "; ".join(
            f"{c}: {pct0(a.loc[c, 'manual_yes'])} fully and {pct0(a.loc[c, 'manual_yes_or_partly'])} at least partly support the claim (the judge agrees on {pct0(a.loc[c, 'agreement'])})" for c in ("E1", "E1b")))
    note(sl, bottom + 0.3, items)

    sl = new("bk_abl", f"thesis Table {T['mabl']['number']}")
    bottom = ttable(sl, "mabl")
    note(sl, bottom + 0.3, ["Each variant changes one setting of E1; differences are from the base configuration (5-sentence chunks, top 3, dense retrieval, citations required)",
                            "No variant changed UFR or CR after adjustment. Coverage follows how much of the course the model is shown",
                            "BM25 and dense retrieval give indistinguishable rates and coverage"])

    sl = new("bk_rep", f"thesis Tables {T['mrepvar']['number']} and {T['mrepcmp']['number']}")
    bottom = ttable(sl, "mrepvar", limit=4.2)
    bottom = ttable(sl, "mrepcmp", top=bottom + 0.3, limit=6.2)
    d.tx(sl, MX, bottom + 0.15, CW, 0.5, "Comparisons against E0 keep their direction in all three generations; E3 against E1 and E1b against E1 change sign, which is why the grounded conditions are not ranked.",
         size=13, color=MUTED, line=1.12)

    sl = new("bk_audit", f"thesis Section 4.10.5 and Table {T['maudit']['number']}")
    bottom = ttable(sl, "maudit", limit=4.2)
    note(sl, bottom + 0.3, ["Sample: 60 of the 110 courses; per condition up to 30 claims the judge flagged and 30 it accepted; 340 claims and 60 citations in all",
                            "Labels: supported, partly supported, not supported, unsure. Partly and not supported count as unsupported",
                            "Blind: the tool hides the condition and the judge's label and shows the claims in random order",
                            "Each label is weighted by its stratum's share of the condition's claims; intervals from 2,000 bootstrap resamples of courses",
                            f"Two passes: {int(agr.loc['claim', 'n'])} claims labeled twice agree at kappa {f2(agr.loc['claim', 'kappa'])} ({pct0(agr.loc['claim', 'agreement'])}); citation checks at kappa {f2(agr.loc['citation', 'kappa'])}"],
         size=13)

    sl = new("bk_second", f"thesis Table {T['magree']['number']}")
    bottom = ttable(sl, "magree")
    note(sl, bottom + 0.3, ["S is the selected judge (DeBERTa-v3-large NLI), 2 is Bespoke-MiniCheck-7B, the strongest candidate of a different model family",
                            "Flagged by both and flagged by either are alternative estimates, not bounds, because two judges can share an error"])

    sl = new("bk_atomic", f"thesis Table {T['matomic']['number']} and Section 7.10")
    bottom = ttable(sl, "matomic", limit=4.7)
    items = ["At the level of atomic facts E1, E1b and E3 are again below E0, so the comparison does not depend on the unit of the claim"]
    wc, vc = csv("mimic_window_check.csv"), csv("mimic_verifier_check.csv")
    if wc is not None:
        items.append(f"Evidence windows: re-scoring the {int(wc.flagged.sum())} flagged claims with half-overlapping windows turned {pct((wc.flagged * wc.rescued_by_overlap).sum() / wc.flagged.sum())} of them "
                     "into supported claims; the artifact affects the conditions alike")
    if vc is not None:
        r = vc[vc.group.str.startswith("rejected")].iloc[0]
        items.append(f"Verifier: of the {int(r.n)} E1 draft claims it rejected, the judge supports {pct0(r.judge_supports_whole)} from the whole course but {pct0(r.judge_supports_top3)} from the verifier's "
                     f"three sentences, so {pct0(r.whole_not_top3)} of the rejections are candidate retrieval failures")
    note(sl, bottom + 0.3, items)

    sl = new("bk_gen", f"thesis Table {T['mgen']['number']} and the result files")
    bottom = ttable(sl, "mgen", limit=4.1)
    rows = []
    for name, tag in GEN_FILES:
        s = csv(f"mimic_summary{tag}.csv")
        b = s[s.variant == "base"].set_index("condition")
        rows.append([name] + [f"{b.loc[c, 'words_mean']:.0f}" for c in ("E0", "E1", "E1b", "E3")] + [f"{b.loc['E3', 'abstentions_mean']:.2f}"])
    bottom = d.table(sl, MX, bottom + 0.3, CW, ["Words per summary", "E0", "E1", "E1b", "E3", "Abstentions per E3 summary"], rows, widths=[3.6, 1.3, 1.3, 1.3, 1.3, 3.3], size=13.5, pad=0.07,
                     align=["l", "c", "c", "c", "c", "c"])
    d.tx(sl, MX, bottom + 0.15, CW, 0.75, "Description only, not a tested explanation: with Mistral-7B the verified summaries stay almost as long as the baseline and the model never abstains, "
         "whereas with the other two generators verification shortens the summary.", size=13, color=MUTED, line=1.12)

    sl = new("bk_qa", f"thesis Table {T['mqa']['number']}")
    bottom = ttable(sl, "mqa", limit=5.6)
    note(sl, bottom + 0.25, ["Exploratory: three fixed questions per course, no clinician-written gold answers",
                             "Retrieval raised abstention and lowered the rate of the answers given, partly because the model says less"], size=13)

    sl = new("bk_stats", "thesis Section 4.2.5")
    note(sl, 1.95, ["Unit of analysis: the hospital course. Every course belongs to a different patient and admission",
                    "Rates are computed per summary, then compared as paired differences on the same courses",
                    "Two-sided Wilcoxon signed-rank test; 95% percentile bootstrap interval of the mean difference from 10,000 resamples",
                    "Effect sizes: standardized mean difference d(z) and matched-pairs rank-biserial correlation",
                    "Holm step-down adjustment: 20 tests in the pilot; 12 primary tests in the main study (E1, E1b, E2 and E3 against E0 on UFR, CR and coverage); within each measure for the ablations",
                    "A result is called significant only after the adjustment; an unadjusted p below 0.05 is called suggestive",
                    "Equivalence: two one-sided tests at the 5% level with a margin of 5 percentage points, as a post hoc check",
                    "Audit: stratum-weighted shares with intervals from 2,000 bootstrap resamples of courses"], size=15)

    sl = new("bk_ethics", "thesis Section 3.7")
    note(sl, 1.95, ["No interaction with human subjects and no identifiable patient data",
                    "MIMIC-IV-Note, ann-pt-summ, MedNLI and VeriFact-BHC are de-identified and were obtained through PhysioNet credentialed access; each data use agreement was accepted before download",
                    "No MIMIC-derived text was sent to any API. Every model that reads it runs on the laptop",
                    "Outputs that contain MIMIC-derived text stay in a folder excluded from version control; only aggregate statistics are published",
                    "An automated scan searches every tracked file for eight-word sequences from the protected texts; the only matches are generic phrases in pilot result files",
                    "Reproducibility: fixed seeds, pinned package versions, every summary and label stored; one script recomputes every pilot number without any model call"], size=15)


def notes_handout(slides_pdf, meta, notes, out):
    """Every slide with its speaker notes under it as bullet points, one slide per page."""
    from xml.sax.saxutils import escape
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    img = BUILD / "handout"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    subprocess.run(["pdftoppm", "-jpeg", "-r", "110", str(slides_pdf), str(img / "s")], check=True, capture_output=True)
    pdfmetrics.registerFont(TTFont("Arial", _FONT_FILES[False])); pdfmetrics.registerFont(TTFont("Arial-Bold", _FONT_FILES[True]))
    body = ParagraphStyle("body", fontName="Arial", fontSize=11.5, leading=15, spaceAfter=5, textColor=HexColor("#" + TEXT),
                          leftIndent=14, bulletIndent=2, bulletFontName="Arial", bulletFontSize=11.5)
    head = ParagraphStyle("head", fontName="Arial-Bold", fontSize=10, leading=13, spaceAfter=7, textColor=HexColor("#" + MUTED))
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            title=DOC_TITLE + ", with speaker notes", author=DOC_AUTHOR)
    story, n_main, nb = [], sum(1 for m in meta if m[2] != "backup"), 0
    for i, ((_, key, kind), f) in enumerate(zip(meta, sorted(img.glob("s-*.jpg"))), 1):
        if kind == "backup":
            nb += 1
        story.append(RLImage(str(f), width=7.0 * inch, height=7.0 * inch * SH / SW))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Backup slide {i}" if kind == "backup" else f"Slide {i} of {n_main}", head))
        for point in notes.get(key, []):
            story.append(Paragraph(escape(" ".join(point.split())), body, bulletText="•"))
        story.append(PageBreak())
    doc.build(story)


def main():
    d, R, M, T, X = build()
    build_backup(d, R, M, T, X)
    n_main, n_backup = d.finish()
    notes = load_notes()
    unknown = sorted(set(notes) - {k for _, k, _ in d.meta})
    if unknown:
        WARNINGS.append(f"notes for slides that do not exist: {unknown}")
    pptx = HERE / f"{OUT}.pptx"
    d.prs.save(str(pptx))                                  # the shared deck has no speaker notes
    pdf = soffice_convert(pptx, BUILD)
    shutil.copyfile(pdf, HERE / f"{OUT}.pdf")
    print(f"wrote {OUT}.pptx and .pdf: {n_main} slides and {n_backup} backup slides, no speaker notes")
    if notes:                                              # the copy with notes stays outside the repository
        d.add_notes(notes)
        have = sum(1 for _, k, _ in d.meta if notes.get(k))
        d.prs.save(str(NOTES_DIR / f"{OUT}_with_notes.pptx"))
        handout = NOTES_DIR / f"{OUT}_with_notes.pdf"
        notes_handout(HERE / f"{OUT}.pdf", d.meta, notes, handout)
        print(f"wrote {OUT}_with_notes.pptx and {handout.name} in {NOTES_DIR}: speaker notes on {have} of {len(d.meta)}")
    for w in WARNINGS:
        print("  check:", w)


if __name__ == "__main__":
    main()
