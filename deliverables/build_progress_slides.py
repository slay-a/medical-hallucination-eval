#!/usr/bin/env python3
"""
build_progress_slides.py — Fall 2026 progress report slides (PDF via reportlab, PPTX via python-pptx).
Numbers are read from ../results at build time.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from pptx import Presentation
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "thesis"))
from results_loader import Results, f3, f2, pct, pct0, fp  # noqa: E402

W, H = 13.333 * 72, 7.5 * 72
NAVY, ORANGE, BLUE, GREEN, GREY = HexColor("#1F2A44"), HexColor("#E07B54"), HexColor("#4C8BB5"), HexColor("#5DBF6E"), HexColor("#666666")
for name, fn in (("TNR", "Times New Roman.ttf"), ("TNRB", "Times New Roman Bold.ttf")):
    pdfmetrics.registerFont(TTFont(name, f"/System/Library/Fonts/Supplemental/{fn}"))


def slides(R: Results):
    t10u, t10c, t20u, t20c = R.test("UFR", "E0", "E1"), R.test("CR", "E0", "E1"), R.test("UFR", "E0", "E2"), R.test("CR", "E0", "E2")
    cal = R.cal_row("all"); gen = R.cal_row("generated"); cov = R.cov.set_index("condition")
    var = R.cal_var[R.cal_var.group == "all"].set_index("variant") if R.cal_var is not None else None
    thesis_pdf = ROOT / "thesis" / "Ponangi_Thesis_Fall2026.pdf"
    pages = "—"
    try:
        out = subprocess.run(["pdfinfo", str(thesis_pdf)], capture_output=True, text=True).stdout
        pages = next(l.split(":")[1].strip() for l in out.splitlines() if l.startswith("Pages"))
    except Exception:  # noqa: BLE001
        pass
    hdr0, hdr1 = R.headers_by_cond.get("E0", 0), R.headers_by_cond.get("E1", 0)
    tb, ta = R.hdr_test("before_filter", "CR"), R.hdr_test("after_filter", "CR")
    S = []
    S.append(dict(kind="title", title="Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization",
                  sub=["Progress Report, Fall 2026 (COMP 698C)", "Srilaya Ponangi, M.S. Computer Science", "Advisor: Dr. Taehyung (George) Wang",
                       f"California State University, Northridge, {date.today():%B %d, %Y}"]))
    S.append(dict(kind="bullets", title="Project and research questions", bullets=[
        "Goal: measure hallucinations in LLM patient-facing summaries at the claim level and test whether retrieval grounding reduces them",
        "RQ1: how often are GPT-4o-mini summary statements unsupported by or contradicting the note (claim-level NLI judge)?",
        "RQ2: which statements are unsupported, and how does the judge itself err against medical experts?",
        "RQ3: does document-grounded RAG reduce unsupported and contradicted statements, versus an extractive bound, and at what coverage cost?",
        "Five conditions compared on 50 MTSamples notes: E0 zero-context LLM, E1 RAG (excerpts only), E1b RAG (note + excerpts), E3 RAG + Chain-of-Verification, E2 centroid extractive",
        "Judge: spaCy claims, all-MiniLM-L6-v2 retrieval (top-3), cross-encoder/nli-MiniLM2-L6-H768; metrics UFR and CR per summary"]))
    S.append(dict(kind="image", title="Pipeline (identical judge for every condition)", image=RES / "fig_pipeline.png", caption="Every summary is scored by the same claim-level NLI judge; evidence is retrieved from the same note."))
    S.append(dict(kind="bullets", title="Data status", bullets=[
        f"MTSamples (public): {R.total_rows:,} transcriptions, {R.n_eligible} eligible consult and discharge notes, {R.n_docs} sampled (seed 42); {R.n_claims_total:,} claims labeled",
        "MIMIC-IV-Note v2.2 (PhysioNet, credentialed): downloaded May 14, 2026, integrity verified; staged for a future local-model re-run, not sent to any API",
        "ann-pt-summ v1.0.1 (PhysioNet, DUA signed): 210 expert-annotated patient summaries, 423 unsupported-fact spans; used to validate the judge with local models",
        "The PhysioNet archive download was incomplete (934 MB of >3.4 GB); the 14 annotation files were extracted and verified against the SHA-256 manifest",
        "Ethics: CITI 'Data or Specimens Only Research' and 'Conflicts of Interest' completed April 10, 2026; no credentialed text leaves the laptop"]))
    S.append(dict(kind="table", title="Corrections since the May 2026 report", columns=["Issue found", "Correction", "Effect"],
                  rows=[["Markdown headers ('**Key Findings:**') counted as claims", f"Header filter; {hdr0} E0 and {hdr1} E1 header lines removed", f"E1 vs E0 CR effect: {tb.delta_mean:+.3f} ({fp(tb.p)}) → {ta.delta_mean:+.3f} ({fp(ta.p)})"],
                        ["E1 described as 'note + excerpts', overlapping chunks, structured query", "Description corrected: excerpts only; non-overlapping 5-sentence chunks; query = description + first 300 chars", "Adds E1b (note + excerpts) as an implemented extension"],
                        ["NLI rule described as argmax over a concatenated premise", "Corrected: max over 3 separate pairs, 0.5 thresholds", "Aggregation variants now tested against experts"],
                        ["'48 percentage-point' median CR drop; Xie et al. mis-citation; APA/IEEE mix", "Arithmetic fixed; citation replaced (Asgari 2025); IEEE with DOIs", "64-entry reference list"],
                        ["requirements.txt missing 4 packages; 'bootstrap CIs' claimed but absent", "Packages added; bootstrap CIs, effect sizes, Holm adjustment implemented", "bash run.sh --offline reproduces every number"]],
                  widths=[3.6, 4.6, 3.9]))
    t1b0u, t30u, t30c, t31u = R.test("UFR", "E0", "E1b"), R.test("UFR", "E0", "E3"), R.test("CR", "E0", "E3"), R.test("UFR", "E1", "E3")
    conds = [c for c in ["E0", "E1", "E1b", "E3", "E2"] if c in R.conds]
    S.append(dict(kind="table", title=f"Main results (n = {R.n_docs} documents; header lines and abstentions excluded)",
                  columns=["Metric"] + [{"E0": "E0 zero-context", "E1": "E1 RAG excerpts", "E1b": "E1b RAG + note", "E3": "E3 RAG + CoVe", "E2": "E2 extractive"}[c] for c in conds],
                  rows=[["UFR mean"] + [f3(R.mean(c, "UFR")) for c in conds],
                        ["CR mean"] + [f3(R.mean(c, "CR")) for c in conds],
                        ["Supported share of claims"] + [pct0((R.label_counts.loc[c, "Supported"] / R.label_counts.loc[c].sum())) for c in conds],
                        ["Claims per summary"] + [f"{R.claims_per[c]:.1f}" for c in conds],
                        ["Words per summary"] + [f"{R.words[c].mean():.0f}" for c in conds],
                        ["Coverage of source sentences (cos ≥ 0.6)"] + [pct(cov.loc[c, "coverage_at_0_6"]) for c in conds]],
                  widths=[3.0] + [1.85] * len(conds),
                  note=(f"Paired Wilcoxon versus E0: E1 CR {fp(t10c.p)} (−{abs(t10c.rel_change_mean):.0f}%), E1 UFR {fp(t10u.p)}; "
                        f"E1b UFR {fp(t1b0u.p)}; E3 UFR {fp(t30u.p)} (−{abs(t30u.rel_change_mean):.0f}%; versus E1 {fp(t31u.p)}), E3 CR {fp(t30c.p)}; "
                        f"E2 UFR {fp(t20u.p)}. Verification lowers unsupported facts the most, by deleting claims and abstaining; retrieval alone lowers contradictions modestly; "
                        f"the extractive bound has a non-zero judge error floor.")))
    S.append(dict(kind="image2", title="Contradiction rate: distributions and per-document pairs", images=[RES / "fig_cr_boxplot.png", RES / "fig_cr_scatter.png"]))
    S.append(dict(kind="table", title="Headline finding: the judge disagrees with medical experts",
                  columns=["Group", "Sentences", "Expert-flagged", "Judge-flagged (UFR)", "Precision", "Recall", "Kappa", "AUROC"],
                  rows=[[r.group.replace("_", " "), f"{int(r.n_sentences):,}", pct(r.expert_flag_rate), pct(r.judge_UFR), f2(r.any_precision), f2(r.any_recall), f2(r.any_kappa), f2(r.auroc_1_minus_p_entail)]
                        for _, r in R.cal.iterrows() if r.group in ("all", "generated", "doctor_written", "gpt4_zero_shot", "llama_70b_original")],
                  widths=[2.2, 1.2, 1.5, 1.9, 1.2, 1.1, 1.0, 1.0],
                  note=f"On 1,781 expert-annotated sentences the judge flags {pct0(cal.judge_UFR)} where experts flag {pct0(cal.expert_flag_rate)}; kappa {f2(cal.any_kappa)} is chance level, and the judge ranks systems in nearly the reverse order of the experts. Absolute UFR/CR are not hallucination rates."))
    S.append(dict(kind="image", title="No aggregation rule rescues the judge", image=RES / "fig_variants.png",
                  caption=(f"Best kappa over all variants and thresholds: {f2(var.best_kappa.max())}; best AUROC {f2(var.auroc_1_minus_pe.max())}. The limiting factor is the small general-domain NLI model on paraphrased patient-facing sentences." if var is not None else "")))
    S.append(dict(kind="image2", title="Robustness: the RAG effect holds only at the default threshold", images=[RES / "fig_threshold_ablation.png", RES / "fig_topk_ablation.png"]))
    S.append(dict(kind="image2", title="What remains unsupported, and the judge's own errors", images=[RES / "fig_taxonomy.png", RES / "fig_e2_probs.png"]))
    S.append(dict(kind="bullets", title="Thesis writing status", bullets=[
        f"Complete draft generated from the result files: {pages} pages in CSUN format (1-inch margins, 12-pt Times New Roman, double-spaced, roman-numbered preliminary pages, arabic body)",
        "Chapters: 1 Introduction, 2 Literature Review, 3 Data and Preprocessing, 4 Methodology, 5 Results, 6 Discussion, 7 Conclusion and Future Work; 64 IEEE references with DOIs; Appendices A to E (prompts, per-document results, worked examples, repository, threshold sweep)",
        "New content since May: two new conditions (E1b, E3), header-filter correction, judge validation against expert annotations, five aggregation variants, threshold/top-k/cleaned-evidence ablations, coverage proxy, negation analysis, error taxonomy, ethics and governance section",
        "Open items: committee member names on the signature page; advisor review; title alignment with the ETD planning form (question answering removed from scope)",
        "Every table and figure is regenerated by one command; the document rebuilds in about two minutes"]))
    S.append(dict(kind="table", title="Plan to the defense (aligned with CSUN ETD deadlines)", columns=["By", "Milestone"],
                  rows=[["Sep 19, 2026", "Advisor review of this complete draft; committee names confirmed"],
                        ["Sep 30, 2026", "Sampling-variance runs (several generations per note) if API budget allows; optional MIMIC re-run with a local open-weight generator"],
                        ["Oct 9, 2026", "ETD Planning Form filed (title without question answering)"],
                        ["Oct 16, 2026", "Revised draft to advisor for approval; Canvas upload and Turnitin check after approval"],
                        ["Oct 30, 2026", "Draft to committee (two-week review); Doodle poll with 1-hour slots over 3 to 4 weeks"],
                        ["Nov 6, 2026", "ETD preliminary format review"],
                        ["Nov 16 to 24, 2026", "Defense (40 min presentation, 10 min Q&A); slides to advisor one week before"],
                        ["Dec 4, 2026", "Final ETD upload after revisions"]], widths=[2.4, 9.7],
                  note="The December 12 defense date in the May report is after the December 4 ETD deadline and has been moved."))
    S.append(dict(kind="bullets", title="Risks and requests", bullets=[
        "Judge validity: the central instrument agrees with experts at chance level; the thesis reports this as its main methodological finding and proposes stronger judges as future work",
        "MIMIC generation: requires a zero-data-retention agreement or a local open-weight model; kept optional",
        "Schedule: two weeks of committee review plus advisor approval must precede the Nov 6 format review",
        "Request to advisor: confirm committee members, review the complete draft, and advise on the title change"]))
    return S


# ───────────────────────────── PDF renderer ─────────────────────────────
def draw_chrome(c, i, n, title):
    c.setFillColor(white); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, H - 62, W, 62, fill=1, stroke=0)
    c.setFillColor(white); c.setFont("TNRB", 22); c.drawString(36, H - 42, title)
    c.setFillColor(GREY); c.setFont("TNR", 9)
    c.drawString(36, 16, "Hallucination Evaluation in LLM-Based Medical Summarization · COMP 698C · Fall 2026")
    c.drawRightString(W - 36, 16, f"{i} / {n}")


def wrap(c, text, font, size, width):
    words = text.split(); lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(t, font, size) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_table(c, x, y, columns, rows, widths, font=9.5):
    scale = 72
    ws = [w * scale for w in widths]
    yy = y
    def row_height(vals, f):
        return max(len(wrap(c, str(v), "TNR", f, ws[j] - 8)) for j, v in enumerate(vals)) * (f + 3) + 6
    h = row_height(columns, font)
    c.setFillColor(HexColor("#DDE3EA")); c.rect(x, yy - h, sum(ws), h, fill=1, stroke=0)
    xx = x
    for j, v in enumerate(columns):
        c.setFillColor(black); c.setFont("TNRB", font)
        for k, line in enumerate(wrap(c, str(v), "TNRB", font, ws[j] - 8)):
            c.drawString(xx + 4, yy - 4 - (k + 1) * (font + 3) + 3, line)
        xx += ws[j]
    yy -= h
    for r in rows:
        h = row_height(r, font)
        c.setStrokeColor(HexColor("#BBBBBB")); c.line(x, yy, x + sum(ws), yy)
        xx = x
        for j, v in enumerate(r):
            c.setFillColor(black); c.setFont("TNR", font)
            for k, line in enumerate(wrap(c, str(v), "TNR", font, ws[j] - 8)):
                c.drawString(xx + 4, yy - 4 - (k + 1) * (font + 3) + 3, line)
            xx += ws[j]
        yy -= h
    c.line(x, yy, x + sum(ws), yy)
    return yy


def render_pdf(S, out: Path):
    c = canvas.Canvas(str(out), pagesize=(W, H))
    n = len(S)
    for i, s in enumerate(S, 1):
        if s["kind"] == "title":
            c.setFillColor(NAVY); c.rect(0, 0, W, H, fill=1, stroke=0); c.setFillColor(white)
            c.setFont("TNRB", 30); y = H - 200
            for line in wrap(c, s["title"], "TNRB", 30, W - 160):
                c.drawCentredString(W / 2, y, line); y -= 40
            c.setFont("TNR", 18); y -= 30
            for line in s["sub"]:
                c.drawCentredString(W / 2, y, line); y -= 28
            c.showPage(); continue
        draw_chrome(c, i, n, s["title"])
        if s["kind"] == "bullets":
            y = H - 105; c.setFillColor(black)
            for bl in s["bullets"]:
                c.setFont("TNR", 15); lines = wrap(c, bl, "TNR", 15, W - 130)
                c.setFillColor(ORANGE); c.circle(52, y + 5, 3.5, fill=1, stroke=0); c.setFillColor(black)
                for line in lines:
                    c.drawString(66, y, line); y -= 21
                y -= 10
        elif s["kind"] == "table":
            yy = draw_table(c, 36, H - 85, s["columns"], s["rows"], s["widths"], font=10.5)
            if s.get("note"):
                c.setFont("TNR", 11); c.setFillColor(NAVY); y = yy - 22
                for line in wrap(c, s["note"], "TNR", 11, W - 80):
                    c.drawString(36, y, line); y -= 15
        elif s["kind"] in ("image", "image2"):
            imgs = [s["image"]] if s["kind"] == "image" else s["images"]
            avail_w = (W - 72 - 20 * (len(imgs) - 1)) / len(imgs); avail_h = H - 62 - 80
            x = 36
            for im in imgs:
                if Path(im).exists():
                    ir = ImageReader(str(im)); iw, ih = ir.getSize(); sc = min(avail_w / iw, avail_h / ih)
                    c.drawImage(ir, x + (avail_w - iw * sc) / 2, H - 72 - ih * sc, iw * sc, ih * sc)
                x += avail_w + 20
            if s.get("caption"):
                c.setFont("TNR", 11); c.setFillColor(NAVY); y = 46
                for line in wrap(c, s["caption"], "TNR", 11, W - 80)[:3]:
                    c.drawString(36, y, line); y -= 14
        c.showPage()
    c.save()


def render_pptx(S, out: Path):
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    for s in S:
        sl = prs.slides.add_slide(blank)
        tb = sl.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.9)); tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = s["title"]; p.runs[0].font.size = Pt(28 if s["kind"] != "title" else 30); p.runs[0].font.bold = True
        if s["kind"] == "title":
            for line in s["sub"]:
                q = tf.add_paragraph(); q.text = line; q.runs[0].font.size = Pt(18)
            continue
        if s["kind"] == "bullets":
            box = sl.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8)); tf = box.text_frame; tf.word_wrap = True
            for i, bl in enumerate(s["bullets"]):
                q = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); q.text = "• " + bl; q.runs[0].font.size = Pt(16); q.space_after = Pt(8)
        elif s["kind"] == "table":
            rows, cols = len(s["rows"]) + 1, len(s["columns"])
            tbl = sl.shapes.add_table(rows, cols, Inches(0.4), Inches(1.2), Inches(12.5), Inches(0.4) * rows).table
            def set_cell(cell, text, size, bold=False):
                cell.text = str(text) if str(text).strip() else " "
                for run in cell.text_frame.paragraphs[0].runs:
                    run.font.size = Pt(size); run.font.bold = bold
            for j, cname in enumerate(s["columns"]):
                set_cell(tbl.cell(0, j), cname, 11, True)
            for i, r in enumerate(s["rows"], 1):
                for j, v in enumerate(r):
                    set_cell(tbl.cell(i, j), v, 10)
            total = sum(s["widths"])
            for j, w in enumerate(s["widths"]):
                tbl.columns[j].width = Inches(12.5 * w / total)
            if s.get("note"):
                nb = sl.shapes.add_textbox(Inches(0.4), Inches(6.5), Inches(12.5), Inches(0.8)); nb.text_frame.word_wrap = True
                nb.text_frame.paragraphs[0].text = s["note"]; nb.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        else:
            imgs = [s["image"]] if s["kind"] == "image" else s["images"]
            wpic = (12.5 - 0.3 * (len(imgs) - 1)) / len(imgs); x = 0.4
            for im in imgs:
                if Path(im).exists():
                    sl.shapes.add_picture(str(im), Inches(x), Inches(1.2), width=Inches(wpic))
                x += wpic + 0.3
            if s.get("caption"):
                nb = sl.shapes.add_textbox(Inches(0.4), Inches(6.6), Inches(12.5), Inches(0.7)); nb.text_frame.word_wrap = True
                nb.text_frame.paragraphs[0].text = s["caption"]; nb.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    prs.save(str(out))


def main():
    R = Results(); S = slides(R)
    render_pdf(S, HERE / "Progress_Slides_Fall2026.pdf"); render_pptx(S, HERE / "Progress_Slides_Fall2026.pptx")
    print(f"wrote {len(S)} slides: Progress_Slides_Fall2026.pdf / .pptx")


if __name__ == "__main__":
    main()
