"""ch_judge.py: Chapter 6: Selecting a Valid Judge (candidate judges scored against the expert annotations; MedNLI adaptation)."""
import json
from pathlib import Path

import pandas as pd

from results_loader import Results, f2, f3, pct, pct0, fp

ROOT = Path(__file__).resolve().parent.parent
JUDGE_NAMES = {"minilm_nli": "MiniLM cross-encoder (pilot judge)", "deberta_large_nli": "DeBERTa-v3-large NLI",
               "ce_deberta_large_nli": "DeBERTa-v3-large cross-encoder", "minicheck_deberta": "MiniCheck DeBERTa-v3-large",
               "minicheck_roberta": "MiniCheck RoBERTa-large", "mednli_deberta_large": "DeBERTa-v3-large NLI + MedNLI",
               "bespoke_minicheck_7b": "Bespoke-MiniCheck-7B"}
MODE_NAMES = {"top3": "top-3 sentences", "doc": "whole course"}
MODE_PHRASE = {"top3": "top-3 sentence evidence", "doc": "whole-course evidence"}


def P(t):
    return ("p", t)


def load():
    f = ROOT / "results" / "judge_candidates.csv"
    jc = pd.read_csv(f) if f.exists() else None
    ft = Path.home() / "Desktop" / "Thesis" / "models" / "mednli-deberta-v3-large" / "mednli_finetune_summary.json"
    ftj = json.load(open(ft)) if ft.exists() else None
    return jc, ftj


def best_row(jc):
    a = jc[jc.group == "all"].sort_values("kappa", ascending=False)
    return a.iloc[0]


def blocks(R: Results) -> list:
    jc, ftj = load()
    b = [("h1", "Chapter 6 Selecting a Valid Judge"), ("h2", "6.1 Purpose")]
    b += [P("The pilot judge agrees with medical experts at chance level (Chapter 5), so the main study needs a judge whose labels "
            "can be interpreted. This chapter scores the candidates of Section 4.9 against the same 1,781 expert-annotated "
            "sentences with the protocol of Section 4.7, adapts the strongest general model to clinical language with MedNLI "
            "[[cite:romanov2018]], and applies a rule fixed in advance: the candidate with the highest Cohen's kappa on all "
            "sentences, with its threshold chosen on the other subset of summaries, becomes the judge of the main study, and its "
            "error profile is carried into every main-study result.")]
    if jc is None:
        b += [P("The candidate comparison has not been run yet.")]
        return b
    b += [("h2", "6.2 Agreement of the Candidate Judges with the Experts")]
    allr = jc[jc.group == "all"].copy()
    allr["jn"] = allr.judge.map(JUDGE_NAMES).fillna(allr.judge); allr["mn"] = allr["mode"].map(MODE_NAMES)
    allr = allr.sort_values(["kappa"], ascending=False)
    best = best_row(jc)
    pilot = allr[(allr.judge == "minilm_nli") & (allr["mode"] == "top3")].iloc[0]
    gen = jc[jc.group == "generated"].sort_values("kappa", ascending=False); doc = jc[jc.group == "doctor_written"].sort_values("kappa", ascending=False)
    _A = jc[(jc.group == "all") & (jc.judge != "minilm_nli")].pivot_table(index="judge", columns="mode", values="kappa")
    _A = _A.dropna(subset=[c for c in ("doc", "top3") if c in _A.columns])
    bg, bdw = gen.iloc[0], doc.iloc[0]
    b += [P(f"[[tab:judges]] and [[fig:judges]] report, for every candidate and evidence mode, the AUROC of the support score and the "
            f"agreement statistics at the threshold chosen on the other subset. The pilot judge reaches a kappa of {f2(pilot.kappa)} "
            f"even with a tuned threshold. The best candidate, {JUDGE_NAMES.get(best.judge, best.judge)} with "
            f"{MODE_PHRASE.get(best['mode'], best['mode'])}, reaches an AUROC of {f2(best.auroc)} and a kappa of {f2(best.kappa)} "
            f"(precision {f2(best.precision)}, recall {f2(best.recall)}, threshold {best.tau_from_other_subset:.2f}), flagging "
            f"{pct0(best.flag_rate)} of sentences where the experts flag {pct0(best.expert_rate)}. Agreement is higher on "
            f"doctor-written instructions (best kappa {f2(bdw.kappa)}) than on LLM-generated summaries (best kappa {f2(bg.kappa)}), "
            f"where the experts flagged fewer sentences. Two lessons emerge: model size and training data matter more than the "
            f"aggregation rule, since every large model improves on the pilot judge in every mode while Chapter 5 showed that "
            f"re-aggregating the small model's evidence changes little; and for {int((_A['doc'] > _A['top3']).sum())} of the "
            f"{len(_A)} large candidates, letting the model see the whole hospital course beats three retrieved sentences, because "
            f"patient-facing paraphrase draws on several parts of the course at once."),
          ("table", dict(label="judges", caption="Candidate judges scored against the medical-expert annotations on all 1,781 sentences. Thresholds are chosen on the other subset (doctor-written for the generated summaries and vice versa; for the all-sentences row the two are pooled). Flag rate is the share of sentences the judge marks unsupported; the experts marked 20.4 percent.",
                         columns=["Judge", "Evidence", "AUROC", "Precision", "Recall", "Specificity", "F1", "Kappa", "Flag rate", "τ"],
                         widths=[1.55, 0.85, 0.55, 0.6, 0.5, 0.7, 0.4, 0.5, 0.45, 0.4], font=8, align=["left", "left"] + ["center"] * 8,
                         rows=[[r.jn, r.mn, f2(r.auroc), f2(r.precision), f2(r.recall), f2(r.specificity), f2(r.f1), f2(r.kappa), pct0(r.flag_rate), f"{r.tau_from_other_subset:.2f}"] for _, r in allr.iterrows()])),
          ("figure", dict(label="judges", path="results/fig_judges.png", width=6.3, caption="AUROC and kappa of each candidate judge and evidence mode against the expert annotations, all sentences."))]
    bmc = jc[(jc.judge == "bespoke_minicheck_7b") & (jc.group == "all")]
    if len(bmc):
        bdm = bmc[bmc["mode"] == "doc"]; deb = jc[(jc.judge == "deberta_large_nli") & (jc["mode"] == "doc") & (jc.group == "all")].iloc[0]
        if len(bdm):
            bdr = bdm.iloc[0]
            b += [P(f"The language-model checker, Bespoke-MiniCheck-7B, which reads the whole course in one pass and answers yes or no, "
                    f"reaches a kappa of {f2(bdr.kappa)} (AUROC {f2(bdr.auroc)}), "
                    + (f"above the best encoder ({f2(deb.kappa)}), so the pre-specified rule selects it." if bdr.kappa > deb.kappa else
                       f"below the best encoder ({f2(deb.kappa)}): a model that tops the general fact-checking leaderboards does not agree "
                       f"with medical experts better than a strong NLI encoder on patient-facing sentences, so the disagreement is not "
                       f"a matter of model size or of reading the whole document."))]
    b += [("h2", "6.3 Adaptation to Clinical Language with MedNLI")]
    if ftj is not None:
        mr = jc[(jc.judge == "mednli_deberta_large") & (jc.group == "all")]
        mdoc = mr[mr["mode"] == "doc"]
        bdoc = jc[(jc.judge == "deberta_large_nli") & (jc["mode"] == "doc") & (jc.group == "all")]
        zs = f" from {pct(ftj['zero_shot_dev_accuracy'])}" if "zero_shot_dev_accuracy" in ftj else ""
        txt = (f"The strongest general model was fine-tuned on MedNLI, 11,232 clinician-written premise and hypothesis pairs from "
               f"MIMIC-III notes (Sections 3.4 and 4.9). Its MedNLI development accuracy rose{zs} to {pct(ftj['dev_accuracy'])}")
        if len(mdoc) and len(bdoc):
            bd_, md = bdoc.iloc[0], mdoc.iloc[0]
            txt += (f", but its agreement with the experts did not improve: kappa {f2(md.kappa)} against {f2(bd_.kappa)} for the same "
                    f"model before adaptation ([[tab:judges]]), with {'fewer' if md.flag_rate < bd_.flag_rate else 'more'} sentences "
                    f"flagged and {'lower' if md.recall < bd_.recall else 'higher'} recall. One explanation consistent with this is "
                    f"granularity: MedNLI pairs a single note sentence with a short clinical statement, whereas patient-facing "
                    f"sentences combine several facts from different parts of a course, so better clinical vocabulary does not help "
                    f"the model decide which part of a multi-fact sentence lacks support. The experiment does not establish this "
                    f"explanation, and Section 6.6 tests one implication of it.")
        else:
            txt += "; its agreement with the experts is reported in [[tab:judges]]."
        b += [P(txt)]
    else:
        b += [P("The MedNLI adaptation is described in Section 4.9; its results are added to [[tab:judges]] when the fine-tuning run completes.")]
    b += [("h2", "6.4 Decision")]
    b += [P(f"By the pre-specified rule the judge of the main study is {JUDGE_NAMES.get(best.judge, best.judge)} with "
            f"{MODE_PHRASE.get(best['mode'], best['mode'])} and a support threshold of {best.tau_from_other_subset:.2f}. Its agreement "
            f"with the experts is moderate, not high: a kappa of {f2(best.kappa)}, a precision of {f2(best.precision)} (roughly "
            f"{100 - round(100 * best.precision)} of every 100 sentences it flags were not flagged by the experts) and a recall of "
            f"{f2(best.recall)} (it misses about {100 - round(100 * best.recall)} of every 100 expert-flagged sentences). Every rate "
            f"in Chapter 7 is therefore an estimate from an instrument with this error profile, and because the judge was chosen on "
            f"the sentences on which the profile is measured, the figures are optimistic; Section 6.5 estimates by how much.")]
    b += independence_blocks()
    b += atomic_blocks()
    b += verifact_blocks()
    return b


SOURCE_NAMES = {"doctor_written": "Doctor-written", "llama_70b_original": "Llama-2 70B, original data", "llama_70b_cleaned": "Llama-2 70B, cleaned data",
                "gpt4_zero_shot": "GPT-4, zero-shot", "gpt4_orig": "GPT-4, original examples", "gpt4_cleaned": "GPT-4, cleaned examples"}


def _res(name):
    f = ROOT / "results" / name
    return pd.read_csv(f) if f.exists() else None


def _cand(note):
    """'deberta_large_nli__doc at 0.2' -> ('DeBERTa-v3-large NLI', 'whole-course evidence', 0.2)."""
    c, t = str(note).split(" at ")
    j, m = c.split("__")
    return JUDGE_NAMES.get(j, j), MODE_PHRASE.get(m, m), float(t)


def independence_blocks():
    ov, sb, fq, ps = _res("data_overlap.csv"), _res("judge_selection_bias.csv"), _res("judge_selection_frequency.csv"), _res("judge_per_source.csv")
    if ov is None and sb is None:
        return []
    b = [("h2", "6.5 Optimism of the Selection and Independence of the Evaluation")]
    O = dict(ov.values.tolist()) if ov is not None else None
    if O is not None:
        b += [P(f"The agreement figures above are measured on sentences that are not independent, and the judge was chosen on the same "
                f"sentences. Matching every course to its MIMIC-IV-Note discharge note (Section 4.9) establishes three facts. Each of the "
                f"{int(O['main-study documents (doctor-written + validation courses)'])} main-study courses belongs to a different patient "
                f"and admission, so the paired observations of Chapter 7 are independent at patient level. The 100 LLM-generated "
                f"summaries of the expert set were written for {int(O['distinct hospital courses behind the 100 LLM-generated summaries'])} "
                f"other courses, {'none of whose patients appears' if int(O['generated-summary courses whose patient also appears in the main study']) == 0 else 'some of whose patients appear'} "
                f"in the main study. But the doctor-written summaries of the expert set are the clinicians' instructions for the "
                f"main-study courses themselves, so the judge was selected partly on text written for the courses whose generated "
                f"summaries it later scores; the generated summaries themselves were never part of the selection.")]
    if sb is not None:
        def g(analysis, metric, col="estimate"):
            r = sb[(sb.analysis == analysis) & (sb.metric == metric)]
            return float(r[col].iloc[0]) if len(r) else float("nan")
        def note(analysis):
            r = sb[(sb.analysis == analysis) & sb.note.notna()]
            return r.note.iloc[0] if len(r) else None
        bs = "selected_judge_course_bootstrap"; cv = "nested_cv_selection_procedure"
        k_app = g("apparent_selected_judge", "kappa"); k_cv = g(cv, "kappa")
        sel_share = None
        if fq is not None and len(fq):
            sel = fq[fq.candidate == "deberta_large_nli__doc"]
            sel_share = float(sel.share.iloc[0]) if len(sel) else 0.0
        b += [P(f"Two analyses measure how optimistic the apparent kappa of {f2(k_app)} is. Resampling hospital courses rather than "
                f"sentences gives it a 95 percent interval of {f2(g(bs, 'kappa', 'lo'))} to {f2(g(bs, 'kappa', 'hi'))}. Nested "
                f"cross-validation then re-runs the whole selection (candidate, evidence mode and threshold) on four of five folds of "
                f"courses and applies the choice to the fifth, twenty times over: the procedure reaches a kappa of {f2(k_cv)} "
                f"({f2(g(cv, 'kappa', 'lo'))} to {f2(g(cv, 'kappa', 'hi'))} over the repetitions) and an AUROC of {f2(g(cv, 'auroc'))}, "
                f"so the optimism of the apparent figure is about {f2(k_app - k_cv)} in kappa"
                + (f", and the procedure chose the selected configuration in {pct0(sel_share)} of the 100 folds" if sel_share is not None else "")
                + ".")]
        t1, t2 = "select_on_doctor_test_on_generated", "select_on_generated_test_on_doctor"
        n1, n2 = note(t1), note(t2)
        if n1 and n2:
            j1, m1, tau1 = _cand(n1); j2, m2, tau2 = _cand(n2)
            same_rank = abs(g(t1, "auroc") - g(bs, "auroc")) < 0.05 and abs(g(t2, "auroc") - g(bs, "auroc")) < 0.05
            b += [P(f"A patient-disjoint transfer test asks the same question across patients. Selected on the doctor-written sentences "
                    f"alone, the procedure chooses {j1} with {m1} and reaches a kappa of {f2(g(t1, 'kappa'))} on the LLM-generated "
                    f"summaries of other patients; selected on the generated summaries, it chooses {j2} with {m2} and reaches "
                    f"{f2(g(t2, 'kappa'))} on the doctor-written sentences. Agreement on unseen patients and a different kind of summary "
                    f"is {'lower than' if max(g(t1, 'kappa'), g(t2, 'kappa')) < k_app - 0.02 else 'close to'} the apparent figure, and "
                    f"the AUROC {'changes little, so the judge' + chr(39) + 's ranking ability transfers while its threshold depends on the kind of summary' if same_rank else 'also changes'}; "
                    f"with a different winner in each direction, this is instability of the choice rather than validation.")]
    if ps is not None and len(ps):
        order = [k for k in SOURCE_NAMES if k in set(ps.group)]
        P_ = ps.set_index("group")
        lo, hi = P_.kappa.idxmin(), P_.kappa.idxmax()
        b += [P(f"[[tab:persource]] gives the selected judge's agreement by source of the summaries: kappa ranges from {f2(P_.loc[lo, 'kappa'])} "
                f"to {f2(P_.loc[hi, 'kappa'])}, and precision is lowest where the experts found the least unsupported content, as expected "
                f"when a judge with a fixed error rate meets texts with fewer real errors. The main study's generator is not among the "
                f"sources, which is why the manual audit of Section 7.9 measures the judge on its summaries directly. The selected "
                f"judge's figures are therefore an optimistic estimate of its accuracy; the comparisons of Chapter 7 do not need it to "
                f"be accurate in absolute terms, but they do need its errors to be similar across conditions, which the audit tests."),
              ("table", dict(label="persource", caption="Agreement of the selected judge (DeBERTa-v3-large NLI, whole-course evidence, τ = 0.20) with the experts, by source of the summaries.",
                             columns=["Source", "Sentences", "Expert rate", "Flag rate", "Precision", "Recall", "Kappa", "AUROC"],
                             widths=[1.85, 0.75, 0.7, 0.65, 0.7, 0.6, 0.55, 0.6], font=9, align=["left"] + ["center"] * 7,
                             rows=[[SOURCE_NAMES.get(k, k), str(int(P_.loc[k, 'n'])), pct0(P_.loc[k, 'expert_rate']), pct0(P_.loc[k, 'flag_rate']),
                                    f2(P_.loc[k, 'precision']), f2(P_.loc[k, 'recall']), f2(P_.loc[k, 'kappa']), f2(P_.loc[k, 'auroc'])] for k in order]))]
    return b


def _join_names(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def atomic_blocks():
    at = _res("atomic_results.csv")
    ex = _res("atomic_example.csv")
    a = sn = None; fx = None
    if at is not None and len(at):
        def row(analysis, group="all"):
            r = at[(at.analysis == analysis) & (at.group == group)] if "group" in at else at[at.analysis == analysis]
            return r.iloc[0] if len(r) else None
        a, sn = row("expert_agreement_atomic_min"), row("expert_agreement_sentence")
        fxr = at[at.analysis == "expert_facts_per_sentence"]
        fx = fxr.iloc[0] if len(fxr) else None
    if a is None and (ex is None or not len(ex)):
        return []
    b = [("h2", "6.6 Sentence-Level Versus Atomic-Fact Scoring")]
    txt = ("A patient-facing sentence often combines several facts, and the judge labels the whole sentence. To test whether finer "
           "units agree better with the experts, the local model split each expert-set sentence into atomic facts (prompt in "
           "Appendix A), every fact was scored by the selected judge against the whole course, and a sentence counts as flagged when "
           "its weakest fact falls below a threshold chosen on the other subset."
           + (f" Sentences contain {fx['mean']:.1f} facts on average and {pct0(fx.share_multi)} contain more than one." if fx is not None else ""))
    if a is not None and sn is not None:
        better = a.kappa > sn.kappa + 0.02; worse = a.kappa < sn.kappa - 0.02
        txt += (f" Scored this way the sentences agree with the experts at a kappa of {f2(a.kappa)} (AUROC {f2(a.auroc)}) against "
                f"{f2(sn.kappa)} ({f2(sn.auroc)}) for whole sentences, so "
                + ("decomposition improves agreement, which supports the granularity explanation of Section 6.3 without proving it." if better else
                   ("decomposition lowers agreement: each fact, read alone, loses the context its sentence supplied." if worse else
                    "decomposition leaves agreement essentially unchanged, and the granularity explanation of Section 6.3 is not "
                    "supported in this form.")))
    b += [P(txt)]
    if ex is not None and len(ex):
        rows = []
        for i, (sent, g) in enumerate(ex.groupby("sentence", sort=False)):
            for j, r in enumerate(g.itertuples()):
                rows.append([f"{sent} ({r.sentence_support:.2f})" if j == 0 else "", r.fact, f"{r.fact_support:.2f}"])
        mixed = sum(1 for _, g in ex.groupby("sentence", sort=False) if (g.fact_support >= g.tau).any() and (g.fact_support < g.tau).any())
        b += [P(f"[[tab:atomicex]] shows the decomposition on public text, the longest full-note summary sentence of each of the first "
                f"{ex.sentence.nunique()} pilot documents"
                + (f"; in {mixed} of them some facts are supported and others are not, which a single sentence-level score cannot express."
                   if mixed else "; in these examples the facts of each sentence agree in support.")),
              ("table", dict(label="atomicex", caption=f"Atomic decomposition of three pilot summary sentences (public MTSamples data). The number after each sentence and each fact is the selected judge's support probability against the whole note; values below {ex.tau.iloc[0]:.2f} are flagged.",
                             columns=["Sentence (support)", "Atomic fact", "Support"], widths=[2.9, 2.9, 0.7], font=8.5,
                             align=["left", "left", "center"], rows=rows))]
    return b


def _and_join(parts):
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _retriever_choice_sentence(k, n_cap_texts=None):
    """How the window ranking was chosen, from results/verifact_retriever_check.csv; empty when absent or only one ranking tried."""
    rc = _res("verifact_retriever_check.csv")
    if not k or rc is None or not len(rc) or "setting" not in rc.columns:
        return ""
    names = {f"top{k}_dense": "the bi-encoder alone", f"top{k}_bm25": "BM25 alone", f"top{k}_hybrid": "the interleaved ranking"}
    rk = rc[rc.setting.isin(names)].set_index("setting")
    if len(rk) < 2:
        return ""
    best = rk.kappa.idxmax()
    nt = int(rk.n_reference_texts.iloc[0]) if "n_reference_texts" in rk.columns and pd.notna(rk.n_reference_texts.iloc[0]) else None
    where = "these admissions" if nt is None or nt == n_cap_texts else f"the {nt} shortest of them"
    return (f"The ranking was chosen on {where} before the full run (" + "; ".join(f"{names[s]} kappa {f2(r.kappa)}" for s, r in rk.iterrows())
            + f"), so {names[best]} was used throughout. ")


def verifact_blocks():
    """6.7 External check on clinician-labeled propositions. Rendered only when results/verifact_judge_eval.csv exists."""
    ev = _res("verifact_judge_eval.csv")
    if ev is None or not len(ev) or "subset" not in ev.columns or "all" not in set(ev.subset):
        return []
    E = ev.set_index("subset")
    al = E.loc["all"]
    adm = str(al.get("premise", "patient")) == "admission"
    k = int(al.get("topk", 0) or 0)
    retr = {"dense": "the judge selection's bi-encoder", "bm25": "BM25",
            "hybrid": "the judge selection's bi-encoder and BM25, interleaved"}.get(str(al.get("retriever", "")), "")
    b = [("h2", "6.7 External Check on Clinician-Labeled Propositions (VeriFact-BHC)")]
    b += [P(f"As an external check, the selected judge was run unchanged on MIMIC-III-Ext-VeriFact-BHC [[cite:chung2026verifact]]: "
            f"{int(al.n):,} valid propositions from human-written and LLM-written brief hospital course summaries of 100 MIMIC-III "
            f"patients, each adjudicated by clinicians as supported, not supported or not addressed against the patient's notes. These "
            f"propositions come from another institution, another decade and other models' summaries. The judge read "
            + ("the notes of the summarized admission" if adm else "each patient's notes") + " with the main study's windowing"
            + (f"; an admission spans {int(al.min_windows)} to {int(al.max_windows):,} windows of 400 tokens (median {int(al.median_windows)}) "
               f"against one to three for a main-study course, so each proposition was scored against the {k} windows ranked highest by "
               f"{retr}, with the maximum over those windows as in the main study ({int(al.judge_calls):,} judge calls), the "
               f"retrieve-then-judge design of the dataset's own raters"
               if k else "")
            + f". A proposition counts as flagged when its best window's support falls below the main study's threshold ({al.tau:.2f}), "
            f"and clinician labels other than supported count as unsupported, as in the dataset's own analysis"
            + ("; because the clinicians judged against the whole record, the comparison below uses the dataset's raters with the same "
               "admission-only restriction." if adm else "."))]
    if k and "cap_check_vs_all_windows" in E.index:
        cc = E.loc["cap_check_vs_all_windows"]
        b += [P(f"On the {int(cc.n_reference_texts)} shortest admissions ({int(cc.n):,} propositions), where every window was also scored, "
                f"the cap changed the judge's flag for {pct0(1 - cc.flag_agreement)} of propositions"
                + (", all from supported to flagged" if pd.notna(cc.get("n_newly_flagged", None)) and int(cc.n_newly_flagged) == int(cc.n_flag_changed) else "")
                + f", and agreement with the clinicians was kappa {f2(cc.kappa)} with the cap against {f2(cc.kappa_all_windows)} without "
                f"it (sensitivity {f2(cc.sensitivity)} against {f2(cc.sensitivity_all_windows)}): "
                + ("the maximum over every window of a long record finds support for almost everything, so the cap also keeps the "
                   "judge close to the main study's regime of one to three windows per course. " if cc.kappa_all_windows < cc.kappa else "")
                + _retriever_choice_sentence(k, int(cc.n_reference_texts))
                + (f"The best-supporting window was the last of the {k} retrieved for {pct0(al.share_best_at_last_rank)} of propositions, "
                   f"so the figures below measure the judge together with this retrieval step."
                   if pd.notna(al.get("share_best_at_last_rank", None)) else ""))]
    perf = (f"On all propositions the judge reaches an AUROC of {f2(al.auroc)} and, at the main study's threshold, sensitivity "
            f"{f2(al.sensitivity)}, specificity {f2(al.specificity)} and kappa {f2(al.kappa)} against the clinician labels. ")
    auth = {i.split("=", 1)[1]: E.loc[i] for i in E.index if i.startswith("author=")}
    typ = {i.split("=", 1)[1]: E.loc[i] for i in E.index if i.startswith("type=")}
    labs = {i.split("=", 1)[1]: E.loc[i] for i in E.index if i.startswith("label=")}
    anames = {"human": "human-written summaries", "llm": "LLM-written summaries"}
    tnames = {"claim": "claims", "sentence": "sentences"}
    if auth:
        perf += "Kappa is " + _and_join(f"{f2(r.kappa)} for {anames.get(k, k)}" for k, r in auth.items())
        perf += (" and " + _and_join(f"{f2(r.kappa)} for {tnames.get(k, k)}" for k, r in typ.items()) if typ else "") + ". "
    if labs:
        perf += ("The judge flags "
                 + _and_join(f"{pct0(r.flag_rate)} of the propositions the clinicians adjudicated {k.lower()}" if j == 0
                             else f"{pct0(r.flag_rate)} of those adjudicated {k.lower()}" for j, (k, r) in enumerate(labs.items()))
                 + ". ")
    ai = _res("verifact_ai_raters.csv")
    if ai is not None and len(ai):
        n_all = len(ai)
        if "reference_only_admission" in ai.columns:
            ai = ai[ai.reference_only_admission.astype(str).str.lower() == str(adm).lower()]
        if len(ai):
            bt = ai.iloc[0]
            perf += (f"For context, the best of the dataset's own {n_all} retrieval-plus-LLM rater configurations with the same reference "
                     f"reaches kappa {f2(bt.kappa)} (sensitivity {f2(bt.sensitivity)}, specificity {f2(bt.specificity)}); the comparison "
                     f"is indicative, since those raters are large language models reading retrieved facts with their chart times while "
                     f"the judge here is a sentence-pair entailment model. ")
    perf += ("The check shows that the judge, chosen on one corpus, still separates supported from unsupported statements on another, "
             "with the same low sensitivity the audit found; it does not re-validate the threshold, and a kappa on retrieved "
             "admission-note evidence is not comparable in size to the course-level agreement of Section 6.2.")
    b += [P(perf)]
    return b
