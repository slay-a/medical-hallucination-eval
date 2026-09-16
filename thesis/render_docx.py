#!/usr/bin/env python3
"""
render_docx.py — Render structured content blocks into a CSUN-formatted thesis (.docx),
convert it to PDF with LibreOffice, and fill the Table of Contents, List of Tables and
List of Figures with real page numbers using a two-pass build.

CSUN Dissertation and Thesis Format Guide rules implemented here:
  * 1-inch margins on all sides; 12-pt Times New Roman throughout, including page numbers
  * double-spaced body; captions, table text, reference entries single-spaced
  * preliminary pages numbered with lower-case roman numerals (title page counted, not printed)
  * body numbered with arabic numerals starting at 1, centred below the bottom margin
  * chapter headings centred at the top margin; each chapter, the references and each
    appendix start on a new page

Block types (tuples):
  ("h1", text)               chapter / major heading, new page
  ("h2", text), ("h3", text) subheadings (bold, left)
  ("p", text)                body paragraph; **bold** and *italic* inline markup allowed
  ("pni", text)              body paragraph without first-line indent
  ("bullets", [items]) / ("numbers", [items])
  ("eq", text)               centred single-spaced equation line
  ("table", dict(caption, columns, rows, widths=None, font=10, label=None, note=None, align=None))
  ("figure", dict(path, caption, width=6.0, label=None))
  ("refs", [entries])        reference list: single-spaced, blank line between entries
  ("code", text)             verbatim block, Courier New 9 pt, single-spaced
  ("pagebreak",)
Cross-references: write [[tab:label]] or [[fig:label]] in any text; they are replaced by the
assigned number, e.g. "Table 3".
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TNR = "Times New Roman"
TEXT_WIDTH_IN = 6.5
ROMAN = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi",
         "xvii", "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]
_TOKEN = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*)")
_XREF = re.compile(r"\[\[(tab|fig):([A-Za-z0-9_\-]+)\]\]")


# ───────────────────────────── low-level helpers ─────────────────────────────
def _font(run, size=12, bold=None, italic=None):
    run.font.name = TNR
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), TNR)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def _pf(par, *, align=None, spacing="double", before=0, after=0, first_indent=None, left_indent=None,
        keep_next=False, keep_together=False, hanging=None):
    pf = par.paragraph_format
    if align is not None:
        par.alignment = align
    if spacing == "double":
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    elif spacing == "single":
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    elif spacing == "1.5":
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if first_indent is not None:
        pf.first_line_indent = Inches(first_indent)
    if left_indent is not None:
        pf.left_indent = Inches(left_indent)
    if hanging is not None:
        pf.left_indent = Inches(hanging); pf.first_line_indent = Inches(-hanging)
    pf.keep_with_next = keep_next
    pf.keep_together = keep_together
    pf.widow_control = True
    return par


def _marked(par, text, size=12, bold=False, italic=False):
    for tok in _TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2]); _font(r, size, True, italic)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); _font(r, size, bold, True)
        else:
            r = par.add_run(tok); _font(r, size, bold, italic)


def _insert_sectpr_child(sectPr, el):
    order = ["w:footnotePr", "w:endnotePr", "w:type", "w:pgSz", "w:pgMar", "w:paperSrc", "w:pgBorders", "w:lnNumType",
             "w:pgNumType", "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection", "w:bidi",
             "w:rtlGutter", "w:docGrid"]
    tag = el.tag.split("}")[1]
    idx = order.index("w:" + tag)
    for later in order[idx + 1:]:
        found = sectPr.find(qn(later))
        if found is not None:
            found.addprevious(el); return
    sectPr.append(el)


def _page_number_format(section, fmt: str, start: int):
    sectPr = section._sectPr
    for el in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(el)
    el = OxmlElement("w:pgNumType"); el.set(qn("w:fmt"), fmt); el.set(qn("w:start"), str(start))
    _insert_sectpr_child(sectPr, el)


def _add_field(par, instr: str, placeholder: str = "1", size=12):
    def fld(t):
        r = par.add_run(); _font(r, size); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
    fld("begin")
    r = par.add_run(); _font(r, size); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "; r._r.append(it)
    fld("separate")
    r = par.add_run(placeholder); _font(r, size)
    fld("end")


def _footer_page_number(section):
    section.footer.is_linked_to_previous = False
    section.footer_distance = Inches(0.5)
    p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing="single")
    _add_field(p, "PAGE")


def _set_margins(section):
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin = section.right_margin = Inches(1)
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)


def _shade(cell, hex_fill="E7E6E6"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)


def _no_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:cantSplit"); el.set(qn("w:val"), "true"); trPr.append(el)


def _new_page(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ───────────────────────────── numbering / cross-refs ────────────────────────
def _chapter_tag(h1_text: str, current: str) -> str:
    m = re.match(r"^\s*Chapter\s+(\d+)", h1_text)
    if m:
        return m.group(1)
    m = re.match(r"^\s*Appendix\s+([A-Z])", h1_text)
    if m:
        return m.group(1)
    return current


def collect_entries(blocks):
    """Assign per-chapter table/figure numbers (e.g. Table 3.1) and gather TOC entries.
    Returns (labels, headings, tables, figures)."""
    labels, headings, tables, figures = {}, [], [], []
    tag, t, f = "0", 0, 0
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            new_tag = _chapter_tag(b[1], tag)
            if new_tag != tag:
                tag, t, f = new_tag, 0, 0
            headings.append((1, b[1]))
        elif kind in ("h2", "h3"):
            headings.append((int(kind[1]), b[1]))
        elif kind == "table":
            t += 1; d = b[1]; d["_num"] = f"{tag}.{t}"
            if d.get("label"):
                labels[f"tab:{d['label']}"] = f"Table {d['_num']}"
            tables.append((d["_num"], d["caption"]))
        elif kind == "figure":
            f += 1; d = b[1]; d["_num"] = f"{tag}.{f}"
            if d.get("label"):
                labels[f"fig:{d['label']}"] = f"Figure {d['_num']}"
            figures.append((d["_num"], d["caption"]))
    return labels, headings, tables, figures


def resolve_xrefs(text: str, labels: dict) -> str:
    def rep(m):
        key = f"{m.group(1)}:{m.group(2)}"
        return labels.get(key, f"[{key}?]")
    return _XREF.sub(rep, text)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


# ───────────────────────────── renderer ──────────────────────────────────────
class Renderer:
    def __init__(self, meta: dict, blocks: list, page_numbers: dict | None = None):
        self.meta, self.blocks = meta, blocks
        self.labels, self.headings, self.tables, self.figures = collect_entries(blocks)
        self.pages = page_numbers or {}
        self.doc = Document()
        self._break_next = False
        self._setup_styles()

    def _p(self, style=None):
        """Create a paragraph; apply a pending page break as page_break_before (never an empty break paragraph)."""
        p = self.doc.add_paragraph(style=style) if style else self.doc.add_paragraph()
        if self._break_next:
            p.paragraph_format.page_break_before = True
            self._break_next = False
        return p

    # -- setup
    def _setup_styles(self):
        st = self.doc.styles["Normal"]
        st.font.name = TNR; st.font.size = Pt(12)
        rpr = st.element.get_or_add_rPr(); rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts"); rpr.append(rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(attr), TNR)
        st.paragraph_format.space_after = Pt(0)
        st.paragraph_format.widow_control = True
        for name in ("Heading 1", "Heading 2", "Heading 3"):
            h = self.doc.styles[name]
            h.font.name = TNR; h.font.size = Pt(12); h.font.bold = True; h.font.italic = False
            h.font.color.rgb = RGBColor(0, 0, 0)
            rpr = h.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(attr), TNR)
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(attr)) is not None:
                    del rf.attrib[qn(attr)]
        sec = self.doc.sections[0]
        _set_margins(sec)

    # -- text helpers
    def _t(self, text):
        return resolve_xrefs(text, self.labels)

    def para(self, text, *, indent=True, align=WD_ALIGN_PARAGRAPH.LEFT, spacing="double", size=12, bold=False,
             italic=False, before=0, after=0, keep_next=False):
        p = self._p()
        _pf(p, align=align, spacing=spacing, before=before, after=after, first_indent=0.5 if indent else 0, keep_next=keep_next)
        _marked(p, self._t(text), size=size, bold=bold, italic=italic)
        return p

    def centered(self, text, *, size=12, bold=False, spacing="double", before=0, after=0, keep_next=False):
        p = self._p()
        _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=spacing, before=before, after=after, keep_next=keep_next)
        _marked(p, self._t(text), size=size, bold=bold)
        return p

    def heading(self, level, text):
        p = self._p(style=f"Heading {level}")
        _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT, spacing="double",
            before=0 if level == 1 else 12, after=0, keep_next=True)
        r = p.add_run(self._t(text)); _font(r, 12, True, False)
        return p

    # -- front matter
    def title_page(self):
        m = self.meta
        self.centered("")
        self.centered("CALIFORNIA STATE UNIVERSITY, NORTHRIDGE")
        for _ in range(4):
            self.centered("")
        self.centered(m["title"])
        for _ in range(3):
            self.centered("")
        self.centered("A thesis submitted in partial fulfillment of the requirements")
        self.centered(f"For the degree of {m['degree']} in {m['program']}")
        self.centered("")
        self.centered("By")
        self.centered(m["author"])
        for _ in range(3):
            self.centered("")
        self.centered(m["date"])
        self._break_next = True

    def signature_page(self):
        m = self.meta
        self.centered("")
        self.para(f"The thesis of {m['author']} is approved:", indent=False)
        for _ in range(2):
            self.centered("")
        for name in m["committee"] + [f"{m['chair']}, Chair"]:
            self.centered("")
            p = self._p(); _pf(p, spacing="single")
            r = p.add_run("_________________________________________\t______________"); _font(r, 12)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(5.0))
            p = self._p(); _pf(p, spacing="single", after=6)
            r = p.add_run(f"{name}\tDate"); _font(r, 12)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(5.0))
            self.centered("")
        for _ in range(4):
            self.centered("")
        self.centered("California State University, Northridge")
        self._break_next = True

    def simple_front_page(self, title, paragraphs):
        self.centered(title, before=0)
        self.centered("")
        for para in paragraphs:
            self.para(para, indent=True)
        self._break_next = True

    def _list_page(self, title, entries, key_prefix, indent_levels=False):
        """entries: list of (level, text, key). Right-aligned dot-leader page numbers."""
        self.centered(title)
        self.centered("")
        for level, text, key in entries:
            p = self._p()
            _pf(p, spacing="single", after=6, left_indent=0.3 * (level - 1) if indent_levels else 0.0)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            hang = 0.3 * (level - 1) if indent_levels else 0.0
            p.paragraph_format.left_indent = Inches(hang + 0.4); p.paragraph_format.first_line_indent = Inches(-0.4)
            num = self.pages.get(key, "0")
            r = p.add_run(f"{self._t(text)}\t{num}"); _font(r, 12)
        self._break_next = True

    def toc_page(self):
        entries = [(1, "Signature Page", "front:Signature Page")]
        if self.meta.get("acknowledgments"):
            entries.append((1, "Acknowledgments", "front:Acknowledgments"))
        if self.tables:
            entries.append((1, "List of Tables", "front:List of Tables"))
        if self.figures:
            entries.append((1, "List of Figures", "front:List of Figures"))
        entries.append((1, "Abstract", "front:Abstract"))
        for level, text in self.headings:
            entries.append((level, text, f"h{level}:{text}"))
        self._list_page("Table of Contents", entries, "toc", indent_levels=True)

    def list_of(self, title, items, prefix):
        entries = [(1, f"{prefix} {n}  {cap}", f"{prefix.lower()}:{n}") for n, cap in items]
        self._list_page(title, entries, prefix)

    def abstract_page(self):
        m = self.meta
        self.centered("Abstract")
        self.centered("")
        self.centered(m["title"])
        self.centered("")
        self.centered("By")
        self.centered(m["author"])
        self.centered(f"{m['degree']} in {m['program']}")
        self.centered("")
        for para in m["abstract"]:
            self.para(para, indent=True)

    # -- body blocks
    def table(self, d):
        num = d["_num"]
        cap = self._p()
        _pf(cap, spacing="single", before=12, after=6, keep_next=True)
        r = cap.add_run(f"Table {num}  "); _font(r, 12, True)
        _marked(cap, self._t(d["caption"]), size=12)
        cols, rows = d["columns"], d["rows"]
        fs = d.get("font", 10)
        tbl = self.doc.add_table(rows=1 + len(rows), cols=len(cols))
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        widths = list(d.get("widths") or [TEXT_WIDTH_IN / len(cols)] * len(cols))
        total = sum(widths)
        if total > TEXT_WIDTH_IN:          # never exceed the 1-inch margins
            widths = [w * TEXT_WIDTH_IN / total for w in widths]
        aligns = d.get("align") or ["left"] + ["center"] * (len(cols) - 1)
        amap = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER, "right": WD_ALIGN_PARAGRAPH.RIGHT}
        for j, w in enumerate(widths):        # grid column widths: LibreOffice lays the table out from these, not from the cell widths
            tbl.columns[j].width = Inches(w)
        for j, c in enumerate(cols):
            cell = tbl.rows[0].cells[j]; cell.width = Inches(widths[j]); _shade(cell)
            p = cell.paragraphs[0]; _pf(p, align=amap[aligns[j]], spacing="single")
            _marked(p, self._t(str(c)), size=fs, bold=True)
        _repeat_header(tbl.rows[0]); _no_split(tbl.rows[0])
        for i, row in enumerate(rows, start=1):
            _no_split(tbl.rows[i])
            for j, val in enumerate(row):
                cell = tbl.rows[i].cells[j]; cell.width = Inches(widths[j])
                p = cell.paragraphs[0]; _pf(p, align=amap[aligns[j]], spacing="single")
                _marked(p, self._t(str(val)), size=fs)
        if d.get("note"):
            p = self._p(); _pf(p, spacing="single", before=4, after=12)
            _marked(p, self._t(d["note"]), size=10)
        else:
            p = self._p(); _pf(p, spacing="single", after=12)

    def figure(self, d):
        num = d["_num"]
        p = self._p(); _pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing="single", before=12, after=6, keep_next=True)
        path = Path(d["path"])
        if path.exists():
            p.add_run().add_picture(str(path), width=Inches(min(d.get("width", 6.0), TEXT_WIDTH_IN)))
        else:
            r = p.add_run(f"[missing figure: {path.name}]"); _font(r, 12, True)
        cap = self._p(); _pf(cap, spacing="single", after=12)
        r = cap.add_run(f"Figure {num}  "); _font(r, 12, True)
        _marked(cap, self._t(d["caption"]), size=12)

    def references(self, entries):
        for e in entries:
            p = self._p(); _pf(p, spacing="single", after=12, hanging=0.5)
            _marked(p, self._t(e), size=12)

    def bullets(self, items, numbered=False):
        for i, it in enumerate(items, start=1):
            p = self._p(); _pf(p, spacing="double", hanging=0.35, left_indent=0.85)
            p.paragraph_format.left_indent = Inches(0.85); p.paragraph_format.first_line_indent = Inches(-0.35)
            marker = f"{i}." if numbered else "•"
            r = p.add_run(f"{marker}\t"); _font(r, 12)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(0.85))
            _marked(p, self._t(it), size=12)

    def render(self) -> Document:
        doc = self.doc
        # ── preliminary pages (section 1: lower-case roman, title page unnumbered)
        sec = doc.sections[0]
        _page_number_format(sec, "lowerRoman", 1)
        sec.different_first_page_header_footer = True
        _footer_page_number(sec)
        self.title_page()
        self.signature_page()
        if self.meta.get("acknowledgments"):
            self.simple_front_page("Acknowledgments", self.meta["acknowledgments"])
        self.toc_page()
        if self.tables:
            self.list_of("List of Tables", self.tables, "Table")
        if self.figures:
            self.list_of("List of Figures", self.figures, "Figure")
        self.abstract_page()
        # ── body (section 2: arabic from 1)
        self._break_next = False
        body = doc.add_section(WD_SECTION.NEW_PAGE)
        _set_margins(body)
        body.different_first_page_header_footer = False
        _page_number_format(body, "decimal", 1)
        _footer_page_number(body)
        first_h1 = True
        for b in self.blocks:
            kind = b[0]
            if kind == "h1":
                if not first_h1:
                    self._break_next = True
                first_h1 = False
                self.heading(1, b[1])
            elif kind == "h2":
                self.heading(2, b[1])
            elif kind == "h3":
                self.heading(3, b[1])
            elif kind == "p":
                self.para(b[1])
            elif kind == "pni":
                self.para(b[1], indent=False)
            elif kind == "bullets":
                self.bullets(b[1])
            elif kind == "numbers":
                self.bullets(b[1], numbered=True)
            elif kind == "eq":
                p = self.centered(b[1], spacing="single", before=6, after=12)
            elif kind == "table":
                self.table(b[1])
            elif kind == "figure":
                self.figure(b[1])
            elif kind == "refs":
                self.references(b[1])
            elif kind == "code":
                lines = b[1].split("\n")
                for i, line in enumerate(lines):
                    p = self._p(); _pf(p, spacing="single", after=12 if i == len(lines) - 1 else 0, left_indent=0.3)
                    r = p.add_run(line if line.strip() else " "); _font(r, 9); r.font.name = "Courier New"
                    rpr = r._element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
                    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                        rf.set(qn(attr), "Courier New")
            elif kind == "pagebreak":
                self._break_next = True
            else:
                raise ValueError(f"unknown block {kind}")
        return doc


# ───────────────────────────── PDF conversion & page lookup ─────────────────
def soffice_convert(docx_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = ["soffice", "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(outdir), str(docx_path)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    pdf = outdir / (docx_path.stem + ".pdf")
    if not pdf.exists():
        raise RuntimeError("LibreOffice did not produce a PDF")
    return pdf


def pdf_pages_text(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
    return out.split("\f")


def locate_pages(pdf: Path, renderer: Renderer) -> tuple[dict, list]:
    """Map every TOC / list entry to the printed page number found in the PDF."""
    pages = pdf_pages_text(pdf)
    info = []  # (printed_number or None, first_line_norm, page_norm)
    for txt in pages:
        lines = [l for l in txt.splitlines() if l.strip()]
        if not lines:
            info.append((None, "", "")); continue
        last = lines[-1].strip()
        printed = last if (last.isdigit() or last.lower() in ROMAN[1:]) else None
        info.append((printed, _norm(lines[0]), _norm(txt)))
    found, missing = {}, []

    def find(key, needle, roman: bool, first_line: bool):
        n = _norm(needle)
        for printed, fl, pn in info:
            if printed is None:
                continue
            is_roman = not printed.isdigit()
            if is_roman != roman:
                continue
            if (fl.startswith(n[:60]) if first_line else (n in pn)):
                found[key] = printed; return
        missing.append(key)

    for title in ("Signature Page", "Acknowledgments", "List of Tables", "List of Figures", "Abstract"):
        find(f"front:{title}", title if title != "Signature Page" else "The thesis of", True, True)
    for level, text in renderer.headings:
        find(f"h{level}:{text}", resolve_xrefs(text, renderer.labels), False, level == 1)
    for n, cap in renderer.tables:
        find(f"table:{n}", f"Table {n} {resolve_xrefs(cap, renderer.labels)[:40]}", False, False)
    for n, cap in renderer.figures:
        find(f"figure:{n}", f"Figure {n} {resolve_xrefs(cap, renderer.labels)[:40]}", False, False)
    return found, missing


def build(meta: dict, blocks: list, out_docx: Path, out_pdf: Path, workdir: Path, passes: int = 2, verbose=True):
    workdir.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    for i in range(1, passes + 1):
        r = Renderer(meta, blocks, numbers)
        doc = r.render()
        tmp_docx = workdir / f"pass{i}.docx"
        doc.save(tmp_docx)
        pdf = soffice_convert(tmp_docx, workdir)
        new_numbers, missing = locate_pages(pdf, r)
        if verbose:
            print(f"  pass {i}: {len(pdf_pages_text(pdf)) - 1} pages, {len(new_numbers)} entries located, {len(missing)} missing")
            for m in missing[:12]:
                print("     missing:", m)
        stable = (new_numbers == numbers)
        numbers = new_numbers
        if stable and i >= 2:
            break
    shutil.copy(tmp_docx, out_docx)
    shutil.copy(pdf, out_pdf)
    return numbers, missing
