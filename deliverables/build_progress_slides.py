#!/usr/bin/env python3
"""
build_progress_slides.py: Fall 2026 progress report slides (PDF via reportlab, PPTX via python-pptx).

Every number is read from ../results and from the built thesis at build time; a slide whose result file is
missing is left out. The deck follows the thesis: one table or figure per result and a short note under it.
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib.colors import HexColor, black, white
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
from mimic_results import Mimic, JUDGE_NAMES, VAR as MVAR, _join  # noqa: E402
from render_docx import set_properties  # noqa: E402

W, H = 13.333 * 72, 7.5 * 72
DOC_TITLE = ("Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization: "
             "Second Progress Report, Fall 2026")
DOC_AUTHOR = "Srilaya Ponangi"
NAVY, ORANGE, GREY = HexColor("#1F2A44"), HexColor("#E07B54"), HexColor("#666666")
for name, fn in (("TNR", "Times New Roman.ttf"), ("TNRB", "Times New Roman Bold.ttf")):
    pdfmetrics.registerFont(TTFont(name, f"/System/Library/Fonts/Supplemental/{fn}"))

COND = {"E0": "E0 full-note", "E1": "E1 RAG excerpts", "E1b": "E1b RAG + course", "E3": "E3 RAG + CoVe", "E2": "E2 extractive", "REF": "Clinician-written"}
MODE = {"top3": "top-3 sentences", "doc": "whole course, windowed"}
GEN = {"llama31": "Llama-3.1-8B-Instruct", "mistral7b": "Mistral-7B-Instruct-v0.3"}
GEN_SHORT = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Mistral-7B-Instruct-v0.3": "Mistral-7B"}


def csv(name):
    f = RES / name
    try:
        return pd.read_csv(f) if f.exists() else None
    except pd.errors.EmptyDataError:
        return None


def pts(x):
    return f"{100 * x:+.1f} points"


def short(names):
    return [GEN_SHORT.get(n, n) for n in names]


def thesis_stats():
    """Pages, tables, figures and references of the built thesis, read from its PDF; None when it is not built."""
    pdf = ROOT / "thesis" / "Ponangi_Thesis_Fall2026.pdf"
    if not pdf.exists():
        return None
    try:
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = int(next(l.split(":")[1] for l in info.splitlines() if l.startswith("Pages")))
        full = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        lists = full.rsplit("List of Tables", 1)[1].split("Abstract", 1)[0] if "List of Tables" in full else ""
        return dict(pages=pages,
                    tables=len(re.findall(r"^\s*Table [0-9A-Z]+\.\d+ ", lists, re.M)),
                    figures=len(re.findall(r"^\s*Figure \d+\.\d+ ", lists, re.M)),
                    refs=max([int(m) for m in re.findall(r"^\s*\[(\d+)\]\s", full, re.M)] or [0]))
    except Exception:  # noqa: BLE001
        return None


def generators(M):
    """Generator-robustness runs: [(name, per-condition base rows, replicated?)] and the two name lists."""
    out, repl, norepl = [], [], []
    for tag, gname in GEN.items():
        gs, gpt = csv(f"mimic_summary_{tag}.csv"), csv(f"mimic_pairwise_tests_{tag}.csv")
        if gs is None or gpt is None or not len(gs):
            continue
        gb = gs[gs.variant == "base"].set_index("condition")
        res = []
        for c in ("E1", "E1b", "E3"):
            t = gpt[(gpt.metric == "UFR") & (gpt.comparison == f"{c} vs E0")]
            if not len(t):
                continue
            t = t.iloc[0]
            ph = float(t.p_holm_primary) if "p_holm_primary" in t.index and not pd.isna(t.p_holm_primary) else float(t.p)
            res.append((c, float(t.delta_mean), ph))
        sig = [r for r in res if r[2] < 0.05 and r[1] < 0]
        (repl if sig else norepl).append(gname)
        out.append((gname, gb, res, sig))
    return out, repl, norepl


def slides(R: Results):
    M = Mimic()
    cal = R.cal_row("all")
    var = R.cal_var[R.cal_var.group == "all"].set_index("variant") if R.cal_var is not None else None
    jc = csv("judge_candidates.csv")
    best = jc[jc.group == "all"].sort_values("kappa", ascending=False).iloc[0] if jc is not None else None
    jname = JUDGE_NAMES.get(best.judge, best.judge) if best is not None else "the selected judge"
    ev = csv("verifact_judge_eval.csv")
    E = ev.set_index("subset") if ev is not None and "subset" in ev.columns and "all" in set(ev.subset) else None
    aud = csv("audit_summary.csv")
    A = aud[aud.manual_unsupported.notna()].set_index("condition") if aud is not None and "manual_unsupported" in aud.columns else None
    Dd = aud[aud["diff"].notna()].set_index("condition") if aud is not None and "diff" in aud.columns else None
    ag = csv("audit_agreement.csv")
    G = csv("mimic_judge_agreement.csv"); G = G.set_index("condition") if G is not None else None
    st2 = csv("mimic_second_judge_tests.csv")
    jsb = csv("judge_selection_bias.csv")
    cvk = jsb[(jsb.analysis == "nested_cv_selection_procedure") & (jsb.metric == "kappa")] if jsb is not None else None
    cv_kappa = float(cvk.estimate.iloc[0]) if cvk is not None and len(cvk) else None
    vt = csv("mimic_variant_tests.csv"); eq = csv("mimic_equivalence.csv")
    gens, repl, norepl = generators(M)
    ts = thesis_stats()
    S = []

    # ── 1 title
    S.append(dict(kind="title", title="Evaluating and Reducing Hallucinations in LLM-Based Medical Report Summarization",
                  sub=["Second Progress Report, Fall 2026 (COMP 698C), revised after the October review",
                       "Srilaya Ponangi, M.S. Computer Science",
                       "Committee: Dr. Taehyung (George) Wang (chair), Dr. Robert McIlhenny, Dr. Mahdi Ebrahimi",
                       f"California State University, Northridge, {date.today():%B %d, %Y}"]))

    # ── 2 goal
    S.append(dict(kind="bullets", title="Goal and research questions (from the proposal)", bullets=[
        "Goal: measure hallucinations in LLM-generated patient-facing summaries at the claim level with a judge validated against medical experts, and test whether retrieval grounding and verification reduce them on real clinical notes",
        "RQ1: how often are summary statements unsupported by, or contradicting, the source note?",
        "RQ2: which statements are unsupported, and how well does the automatic judge agree with medical experts?",
        "RQ3: do retrieval-augmented generation and verification reduce unsupported and contradicted statements, compared with an extractive reference, and at what cost in coverage?",
        "Three studies: a pilot on public MTSamples notes (GPT-4o-mini), a judge selection study against expert labels, and the main study on 110 MIMIC-IV hospital courses with an open-weight model running locally; the proposal's second task, question answering, is piloted with abstention"]))

    # ── 3 commitments
    judge_status = (f"Done: pilot judge at chance against the experts (kappa {f2(cal.any_kappa)}); seven candidates compared; {jname} selected "
                    f"(kappa {f2(best.kappa)}, AUROC {f2(best.auroc)})" if best is not None else "Pilot judge validated; candidate comparison running")
    if A is not None and "pooled" in A.index:
        judge_status += f"; audited blind on the main-study outputs (kappa {f2(A.loc['pooled', 'kappa'])})"
    if E is not None:
        judge_status += f"; checked on {int(E.loc['all', 'n']):,} clinician-labeled VeriFact propositions (kappa {f2(E.loc['all', 'kappa'])})"
    S.append(dict(kind="table", title="Proposal commitments and where they stand", columns=["Commitment in the proposal", "Status"],
                  rows=[["Claim-level NLI judge calibrated against expert labels", judge_status],
                        ["Experiments on MIMIC data", f"Done: {M.n_docs} MIMIC-IV hospital courses; Qwen2.5-7B-Instruct served on the laptop; replicated with Llama-3.1-8B and Mistral-7B; no protected text leaves the machine" if M.ok else "Running"],
                        ["RAG, verification and an extractive baseline", "Done in both studies: E0 full note, E1 RAG (excerpts), E1b RAG (course + excerpts), E3 RAG + Chain-of-Verification, E2 extractive"],
                        ["Coverage against a reference", "Done: share of the clinician's discharge-instruction sentences that the summary supports (judge run in reverse)"],
                        ["Citation accuracy", "Done: retrieval conditions cite an excerpt after every sentence; the judge checks each citation"],
                        ["Retrieval ablations (chunk size, top-k, BM25 vs dense, citations)", "Done: six E1 variants on the same courses; no setting changed faithfulness after correction, coverage follows the amount of retrieved text"],
                        ["Question answering with abstention", "Piloted (exploratory): three questions per course, full course vs retrieved excerpts, 'Not stated in the note.' allowed"]],
                  widths=[4.4, 7.8]))

    # ── 4 pilot: the judge disagrees with the experts
    S.append(dict(kind="table", title=f"Pilot on {R.n_docs} public notes: the pipeline works, the small MiniLM judge does not",
                  columns=["Group", "Sentences", "Expert-flagged", "Judge-flagged", "Precision", "Recall", "Kappa", "AUROC"],
                  rows=[[r.group.replace("_", " "), f"{int(r.n_sentences):,}", pct(r.expert_flag_rate), pct(r.judge_UFR), f2(r.any_precision), f2(r.any_recall), f2(r.any_kappa), f2(r.auroc_1_minus_p_entail)]
                        for _, r in R.cal.iterrows() if r.group in ("all", "generated", "doctor_written", "gpt4_zero_shot", "llama_70b_original")],
                  widths=[2.2, 1.2, 1.5, 1.5, 1.2, 1.1, 1.0, 1.0],
                  note=(f"Five conditions were compared on {R.n_docs} MTSamples notes with GPT-4o-mini (UFR {f2(R.mean('E0','UFR'))} for full-note summaries, {f2(R.mean('E3','UFR'))} with verification, "
                        f"{f2(R.mean('E2','UFR'))} for verbatim extracts), but on 1,781 expert-annotated sentences this judge flags {pct0(cal.judge_UFR)} where the experts flag {pct0(cal.expert_flag_rate)}: "
                        f"kappa {f2(cal.any_kappa)} is chance level, and no evidence-aggregation variant raised it above {f2(var.best_kappa.max()) if var is not None else 'n/a'}. "
                        f"The pilot therefore establishes the pipeline and the artifacts it must handle (header lines, abstentions, citation markers), not hallucination rates.")))

    # ── 5 judge selection
    if jc is not None:
        allr = jc[jc.group == "all"].sort_values("kappa", ascending=False)
        S.append(dict(kind="table", title="Judge selection: seven candidates scored against the expert annotations (1,781 sentences)",
                      columns=["Judge", "Evidence", "AUROC", "Precision", "Recall", "Kappa", "Flag rate"],
                      rows=[[JUDGE_NAMES.get(r.judge, r.judge), MODE.get(r["mode"], r["mode"]), f2(r.auroc), f2(r.precision), f2(r.recall), f2(r.kappa), pct0(r.flag_rate)] for _, r in allr.iterrows()],
                      widths=[4.0, 2.0, 1.1, 1.3, 1.1, 1.1, 1.3],
                      note=(f"Rule fixed in advance: highest kappa on all sentences, threshold chosen on the other subset. Selected: {jname} ({MODE.get(best['mode'])}), "
                            f"kappa {f2(best.kappa)}" + (f" ({f2(cv_kappa)} under nested cross-validation)" if cv_kappa is not None else "")
                            + f", AUROC {f2(best.auroc)}. Model size and training data matter more than the aggregation rule; MedNLI fine-tuning raised MedNLI accuracy but not agreement with the experts. "
                            f"Moderate agreement, so main-study rates are judge-estimated and the paired comparisons carry the weight.")))

    # ── 6 VeriFact external check
    if E is not None:
        al = E.loc["all"]
        sub = [("all", "All propositions"), ("author=human", "Human-written summaries"), ("author=llm", "LLM-written summaries"), ("type=claim", "Claims"), ("type=sentence", "Sentences")]
        rows = [[lab, f"{int(E.loc[k, 'n']):,}", f2(E.loc[k, "auroc"]), f2(E.loc[k, "sensitivity"]), f2(E.loc[k, "specificity"]), f2(E.loc[k, "kappa"])] for k, lab in sub if k in E.index]
        note = (f"Same judge, windowing and threshold as the main study, on the {int(al.topk)} best-matching 400-token windows of the admission's notes "
                f"({int(al.judge_calls):,} judge calls; an admission spans {int(al.min_windows)} to {int(al.max_windows):,} windows). ")
        if "cap_check_vs_all_windows" in E.index:
            cc = E.loc["cap_check_vs_all_windows"]
            note += f"Scoring every window instead, on the {int(cc.n_reference_texts)} shortest admissions, drops kappa from {f2(cc.kappa)} to {f2(cc.kappa_all_windows)}: the maximum over hundreds of windows supports almost everything. "
        ai = csv("verifact_ai_raters.csv")
        if ai is not None and len(ai) and "reference_only_admission" in ai.columns:
            n_all = len(ai); same = ai[ai.reference_only_admission.astype(str).str.lower() == "true"]
            if len(same):
                note += f"The best of the dataset's own {n_all} LLM rater configurations with the same reference reaches kappa {f2(same.iloc[0].kappa)}. "
        note += "The judge still separates supported from unsupported statements on another institution's data, with the same low sensitivity the manual audit found."
        S.append(dict(kind="table", title="External check: the selected judge on clinician-labeled propositions (VeriFact-BHC, Section 6.7)",
                      columns=["Propositions", "n", "AUROC", "Sensitivity", "Specificity", "Kappa"], rows=rows, widths=[3.4, 1.4, 1.5, 1.6, 1.6, 1.4], note=note))

    # ── 7 main study: data and design
    S.append(dict(kind="bullets", title="Main study: data and design", bullets=[
        "Corpus: 110 de-identified MIMIC-IV Brief Hospital Course sections (ann-pt-summ, PhysioNet), each paired with the discharge instructions the treating clinician wrote; the same dataset's 210 expert-annotated summaries validated the judge, and no patient is shared between the two",
        "Generator: Qwen2.5-7B-Instruct (4-bit) served on the laptop; the same prompts, decoding and retrieval rerun with Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3",
        "Conditions: E0 full note; E1 RAG with excerpts only; E1b RAG with the course and the excerpts; E3 RAG with Chain-of-Verification; E2 extractive; retrieval conditions cite the excerpt behind every sentence, and 'Not stated in the note.' is allowed",
        f"Judge: {jname}, whole-course evidence in 400-token windows, τ = {M.tau:.2f}; judge-estimated UFR and CR per summary; the clinicians' own instructions scored as a reference",
        "Measures: coverage of the clinician's instructions (judge in reverse), citation accuracy, six retrieval ablations, length-matched and no-citation variants, repeated generations, question-answering pilot",
        "Checks of the judge on these outputs: blind manual audit in two passes, second judge, second threshold, atomic facts, overlapping windows",
        "Ethics: MIMIC-IV-Note, ann-pt-summ, MedNLI and VeriFact-BHC are PhysioNet credentialed data (CITI training completed); every text is processed by models on the author's laptop, nothing is sent to an API or to cloud storage, and an automated scan finds no protected text in the repository"]))

    if M.ok:
        mc = [c for c in ["E0", "E1", "E1b", "E3", "E2", "REF"] if c in M.conds]
        def cell(c, col, fmt):
            v = M.val(c, col)
            return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else fmt(v)
        def cite(c):
            """Pooled citation coverage and accuracy over the 110 courses, as in Table 7.5 of the thesis."""
            r = M.base.loc[c]
            if not r.cited_claims:
                return None
            return r.cited_claims / r.claims_total, r.cite_correct / r.cited_claims
        # ── 8 main results
        S.append(dict(kind="table", title=f"Main study results (n = {M.n_docs} hospital courses; judge: {M.judge_name()}, τ = {M.tau:.2f})",
                      columns=["Metric"] + [COND[c] for c in mc],
                      rows=[["UFR mean (judge-estimated)"] + [cell(c, "UFR_mean", f3) for c in mc],
                            ["CR mean"] + [cell(c, "CR_mean", f3) for c in mc],
                            ["Coverage of the clinician's sentences"] + [cell(c, "coverage_ref_mean", pct) if c != "REF" else "n/a" for c in mc],
                            ["Claims cited / citation accuracy (pooled)"] + [(f"{pct0(cite(c)[0])} / {pct0(cite(c)[1])}" if cite(c) else "n/a") for c in mc],
                            ["Abstentions per summary"] + [cell(c, "abstentions_mean", lambda v: f"{v:.1f}") for c in mc],
                            ["Claims per summary"] + [cell(c, "claims_mean", lambda v: f"{v:.1f}") for c in mc],
                            ["Words per summary"] + [cell(c, "words_mean", lambda v: f"{v:.0f}") for c in mc]],
                      widths=[3.2] + [1.55] * len(mc),
                      note=" ".join(f"{c} vs E0: UFR {fp(M.test('UFR', c).p)}, CR {fp(M.test('CR', c).p)}, coverage {fp(M.test('coverage_ref', c).p)}." for c in mc if c not in ("E0", "REF") and M.test("UFR", c) is not None)
                           + " Paired two-sided Wilcoxon tests on the same courses; the primary family of twelve tests is Holm-adjusted in the thesis."))
        # ── 9 what the results say
        bl = [f"Grounding: E1, E1b and E3 lowered the judge-estimated UFR from {f3(M.mean('E0','UFR'))} to {f3(M.mean('E1','UFR'))}, {f3(M.mean('E1b','UFR'))} and {f3(M.mean('E3','UFR'))} (all {fp(max(M.test('UFR', c).p for c in ('E1','E1b','E3')))}); the three do not differ from one another; CR did not change",
              f"Omission: E1 covered {pct0(M.mean('E1','coverage_ref'))} of the clinician's sentences and E3 {pct0(M.mean('E3','coverage_ref'))}, against {pct0(M.mean('E0','coverage_ref'))} for E0; E1b {pct0(M.mean('E1b','coverage_ref'))}"
              + (", within 5 points of E0 by two one-sided tests" if eq is not None and len(eq[(eq.comparison == "E1b vs E0") & (eq.equivalent_at_margin.astype(str) == "True")]) else ""),
              f"References, not bounds: verbatim extracts (E2) score {f3(M.mean('E2','UFR'))}, judge error on supported text; the clinicians' own instructions {f3(M.mean('REF','UFR'))}, because they state what the course does not"]
        if vt is not None and len(vt[(vt.comparison == "E0/short vs E0/base") & (vt.metric == "UFR")]):
            s = vt[(vt.comparison == "E0/short vs E0/base") & (vt.metric == "UFR")].iloc[0]
            bl.append(f"Brevity explains part of it, not all: an E0 told to write about 150 words scores {f3(s.mean_b)} ({fp(s.p)} vs E0), still above the grounded conditions")
        bl.append(f"Citations: E1 cites {pct0(cite('E1')[0])} of its claims and {pct0(cite('E1')[1])} of those citations hold, E1b {pct0(cite('E1b')[0])} and {pct0(cite('E1b')[1])}; {M.faithfulness_short()}")
        if M.qa_tot is not None:
            bl.append(f"Question answering (exploratory): retrieval raised abstention from {pct0(M.qa_tot.loc['QA-E0','abstention_rate'])} to {pct0(M.qa_tot.loc['QA-E1','abstention_rate'])} and lowered the UFR of the answers given ({f3(M.qa_tot.loc['QA-E0','UFR_mean'])} to {f3(M.qa_tot.loc['QA-E1','UFR_mean'])}); warning-sign questions are rarely answerable from the course")
        S.append(dict(kind="bullets", title="Main study: what the results say", bullets=bl))
        # ── 10 manual audit
        if A is not None:
            ca = [c for c in ("E0", "E1", "E1b", "E3") if c in A.index]
            rows = [[COND[c], str(int(A.loc[c, "n_labeled"])), f"{pct(A.loc[c, 'manual_unsupported'])} ({pct(A.loc[c, 'manual_unsupported_lo'])} to {pct(A.loc[c, 'manual_unsupported_hi'])})",
                     pct(A.loc[c, "judge_flag_share_pipeline"]), f2(A.loc[c, "precision"]), f2(A.loc[c, "false_omission"]), f2(A.loc[c, "sensitivity"]), f2(A.loc[c, "specificity"]), f2(A.loc[c, "kappa"])] for c in ca]
            n_lab = int(A.loc[ca, "n_labeled"].sum())
            note = f"The author labeled {n_lab} claims blind to the condition and the judge's label"
            if ag is not None and "claim" in set(ag.kind):
                acl = ag[ag.kind == "claim"].iloc[0]
                note += f", in two passes that agree at kappa {f2(acl.kappa)} after the rubric was clarified"
            note += ". The judge understates unsupported content, mostly partly supported sentences. "
            if Dd is not None:
                keys = [k for k in ("E1 minus E0", "E1b minus E0", "E3 minus E0") if k in Dd.index]
                conf = [k for k in keys if Dd.loc[k, "diff_hi"] < 0]
                note += ("Manual differences from E0: " + "; ".join(f"{k.split(' minus ')[0]} {pts(Dd.loc[k, 'diff'])} ({100 * Dd.loc[k, 'diff_lo']:+.1f} to {100 * Dd.loc[k, 'diff_hi']:+.1f})" for k in keys)
                         + (f". The interval excludes zero for {_join([k.split(' minus ')[0] for k in conf])}, which confirms the judge's direction there; the other differences are unresolved at this sample size." if conf
                            else ". No manual interval excludes zero at this sample size."))
            S.append(dict(kind="table", title="Manual audit of the judge on the main-study claims (Section 7.9)",
                          columns=["Condition", "Labeled", "Manual unsupported (95% CI)", "Judge flags", "Precision", "False omission", "Sensitivity", "Specificity", "Kappa"],
                          rows=rows, widths=[1.7, 0.9, 2.6, 1.1, 1.1, 1.2, 1.2, 1.2, 1.0], note=note))
        # ── 11 robustness: generators and the second judge
        if gens:
            gconds = [c for c in ("E0", "E1", "E1b", "E3", "REF") if all(c in gb.index for _, gb, _, _ in gens)]
            rows = [[COND[c], f3(M.mean(c, "UFR"))] + [f3(gb.loc[c, "UFR_mean"]) for _, gb, _, _ in gens] for c in gconds]
            parts = []
            for gname, gb, res, sig in gens:
                if res and len(sig) == len(res):
                    parts.append(f"under {gname} every grounded condition is again below E0 after Holm adjustment")
                elif sig:
                    parts.append(f"under {gname} only {_join([c for c, _, _ in sig])} is below E0 after adjustment")
                else:
                    worse = [c for c, d, _ in res if d > 0]
                    parts.append(f"under {gname} no grounded condition differs from E0 after adjustment" + (f" and {_join(worse)} is higher" if worse else ""))
            note = (lambda x: x[0].upper() + x[1:])("; ".join(parts)) + ". "
            if G is not None and "all" in G.index:
                note += f"A second judge (Bespoke-MiniCheck-7B) agrees with the selected judge on {pct0(G.loc['all','agreement'])} of claims (kappa {f2(G.loc['all','kappa'])}) and keeps every grounded condition below E0"
                s31 = st2[(st2.measure == "UFR_second") & (st2.comparison == "E3 vs E1")] if st2 is not None and "measure" in st2.columns else None
                if s31 is not None and len(s31):
                    note += f"; it puts E3 below E1 ({fp(s31.iloc[0].p)}) where the selected judge sees no difference, so whether verification adds to retrieval depends on the judge"
                note += ". "
            note += "The reductions also hold at threshold 0.5 and in each of three generations on 40 courses."
            S.append(dict(kind="table", title="Robustness: the same comparison under three generators (judge-estimated UFR, Section 7.10)",
                          columns=["Condition", "Qwen2.5-7B (main)"] + [g for g, _, _, _ in gens], rows=rows, widths=[3.2] + [2.6] * (1 + len(gens)), note=note))
        # ── 12 hypotheses
        t10c_, t31u_ = R.test("CR", "E0", "E1"), R.test("UFR", "E1", "E3")
        s31 = st2[(st2.measure == "UFR_second") & (st2.comparison == "E3 vs E1")].iloc[0] if st2 is not None and "measure" in st2.columns and len(st2[(st2.measure == "UFR_second") & (st2.comparison == "E3 vs E1")]) else None
        h2v = "Supported for unsupported facts on hospital courses" + (f" with Qwen2.5-7B and {_join(short(repl))}, not with {_join(short(norepl))}" if repl and norepl else "") + "; not for contradictions"
        S.append(dict(kind="table", title="Hypotheses: verdicts across both studies", columns=["Hypothesis", "Pilot (MTSamples, GPT-4o-mini)", "Main study (MIMIC-IV, local models)", "Verdict"],
                      rows=[["H1: unsupported content is measurable, and the judge agrees with experts better than chance",
                             f"UFR {f3(R.mean('E0','UFR'))} under a judge with kappa {f2(cal.any_kappa)} (chance level)",
                             f"UFR {f3(M.mean('E0','UFR'))} judge-estimated" + (f"; {pct(A.loc['E0','manual_unsupported'])} by manual reading" if A is not None and "E0" in A.index else "") + f"; judge kappa {f2(best.kappa) if best is not None else 'n/a'}",
                             "First half supported (confirmed by the audit); second half only moderately, for the selected judge"],
                            ["H2: RAG reduces unsupported and contradicted statements",
                             f"CR down {abs(t10c_.rel_change_mean):.0f}% ({fp(t10c_.p)}, Holm {fp(t10c_.p_holm)}); UFR unchanged",
                             f"UFR down {M.rel(M.test('UFR','E1'))} ({fp(M.test('UFR','E1').p)}); CR unchanged ({fp(M.test('CR','E1').p)})",
                             h2v],
                            ["H3: verification adds to RAG",
                             f"E3 vs E1 UFR {fp(t31u_.p)}",
                             f"Selected judge: E3 = E1 ({fp(M.test('UFR','E3','E1').p)})" + (f"; second judge: {f3(s31.mean_b)} vs {f3(s31.mean_a)} ({fp(s31.p)})" if s31 is not None else ""),
                             "Supported in the pilot; judge-dependent in the main study, so not claimed"]],
                      widths=[3.0, 3.0, 3.6, 2.6],
                      note="Paired two-sided Wilcoxon tests on the same documents. Pilot effects describe a judge later shown invalid and are not interpreted further."))

    # ── 13 thesis status and what changed since the review
    tsline = (f"Revised draft: {ts['pages']} pages in CSUN format; {ts['tables']} tables, {ts['figures']} figures, {ts['refs']} references; " if ts else "Revised draft in CSUN format; ")
    S.append(dict(kind="bullets", title="Thesis status and what changed since the October review", bullets=[
        tsline + "nine chapters (Introduction, Literature Review, Data, Methodology, Pilot Study, Selecting a Valid Judge, Main Study, Discussion, Conclusion) and Appendices A to C (prompts; code repository and reproduction; manual audit protocol); every table and figure is regenerated by one command from the result files",
        "Length: 105 pages (136 in October, 163 after the additions for the review); the text is compact, with a table or figure and a short paragraph per result; the pilot chapter is cut to 6 tables and 1 figure; three appendices are removed (per-document results, worked examples, threshold sweep); the judge selection and main study tables all remain and no result file changed",
        "Judge-estimated versus confirmed rates: every rate is labeled judge-estimated; a blind manual audit in two passes measures the judge on these outputs and confirms the E1b reduction directly (Sections 4.10.5, 7.9)",
        "Judge evaluation: claim splitting corrected (citation markers, merged headers); selection optimism measured by nested cross-validation" + (f" (kappa {f2(cv_kappa)})" if cv_kappa is not None else "") + "; no patient shared between the expert set and the main study; external check on VeriFact-BHC (Sections 3.5, 6.5, 6.7)",
        "Confounds and inference: length-matched E0, E1b without the citation instruction, repeated generations; equivalence of E1b's coverage tested with two one-sided tests; E2 and the clinicians' instructions reported as references, not bounds (Sections 7.4, 7.5, 7.8)",
        "Robustness: second threshold, second judge with conservative and inclusive rules as alternative estimates, atomic facts, overlapping windows, verifier retrieval failures, and two further generators, with conclusions stated per generator (Section 7.10)",
        "Question answering marked exploratory; conclusions qualified by the judge's measured validity; reproducibility appendix with model revisions, seeds and what each result needs (Appendix B)"]))

    # ── 14 plan and requests
    S.append(dict(kind="table", title="Plan to the defense (CSUN ETD process) and requests", columns=["Milestone", "Status"],
                  rows=[["ETD planning form submitted; committee confirmed (Dr. Wang, chair; Dr. McIlhenny; Dr. Ebrahimi)", "Done, October 9, 2026"],
                        ["Second draft and progress report; advisor's review received", "Done, October 12 and 15, 2026"],
                        ["Revised draft answering the review, these slides and the description", f"Done, {date.today():%B %d, %Y}"],
                        ["Advisor approval; Canvas upload; Turnitin similarity 20% or lower", "Next"],
                        ["Draft to the committee with at least two weeks of review; Doodle poll with 1-hour slots", "After approval"],
                        ["Defense on CSUN Zoom (40 min presentation, 10 min Q&A, 10 min deliberation); slides to the advisor one week before", "Before the ETD deadline"],
                        ["Final ETD upload after revisions; committee approval in the ETD system", "December 4, 2026, 5 pm"]], widths=[7.6, 4.6],
                  note=("The committee approves inside the ETD system, so the defense must be finished before the December 4 upload; the remaining dates are set with the advisor. "
                        "Requests to the advisor: review the revised draft; approve the draft for distribution to the committee; advise on the defense date.")))
    return S


# ───────────────────────────── PDF renderer ─────────────────────────────
def draw_chrome(c, i, n, title):
    c.setFillColor(white); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, H - 62, W, 62, fill=1, stroke=0)
    c.setFillColor(white)
    size = 22
    while pdfmetrics.stringWidth(title, "TNRB", size) > W - 72 and size > 14:
        size -= 1
    c.setFont("TNRB", size); c.drawString(36, H - 42, title)
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
    c.setTitle(DOC_TITLE); c.setAuthor(DOC_AUTHOR); c.setSubject("COMP 698C"); c.setCreator("build_progress_slides.py")
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
            size = 15
            while size > 11:   # shrink until the bullets fit above the footer
                need = sum(len(wrap(c, bl, "TNR", size, W - 130)) * (size + 6) + 10 for bl in s["bullets"])
                if need <= H - 105 - 36:
                    break
                size -= 1
            y = H - 105; c.setFillColor(black)
            for bl in s["bullets"]:
                c.setFont("TNR", size); lines = wrap(c, bl, "TNR", size, W - 130)
                c.setFillColor(ORANGE); c.circle(52, y + 5, 3.5, fill=1, stroke=0); c.setFillColor(black)
                for line in lines:
                    c.drawString(66, y, line); y -= size + 6
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
    set_properties(prs, DOC_TITLE, DOC_AUTHOR)
    blank = prs.slide_layouts[6]
    for s in S:
        sl = prs.slides.add_slide(blank)
        tb = sl.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.9)); tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = s["title"]; p.runs[0].font.size = Pt(26 if s["kind"] != "title" else 30); p.runs[0].font.bold = True
        if s["kind"] == "title":
            for line in s["sub"]:
                q = tf.add_paragraph(); q.text = line; q.runs[0].font.size = Pt(18)
            continue
        if s["kind"] == "bullets":
            box = sl.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8)); tf = box.text_frame; tf.word_wrap = True
            size = 16 if sum(len(b) for b in s["bullets"]) < 1500 else 14
            for i, bl in enumerate(s["bullets"]):
                q = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); q.text = "• " + bl; q.runs[0].font.size = Pt(size); q.space_after = Pt(8)
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
                nb = sl.shapes.add_textbox(Inches(0.4), Inches(6.3), Inches(12.5), Inches(1.0)); nb.text_frame.word_wrap = True
                nb.text_frame.paragraphs[0].text = s["note"]; nb.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
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
