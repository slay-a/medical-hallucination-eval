"""ch_judge.py — Chapter 6: Selecting a Valid Judge (candidate judges scored against the expert annotations; MedNLI adaptation)."""
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
    b += [P("Chapter 5 showed that the pilot judge, a six-layer MiniLM cross-encoder trained on general-domain NLI data, agrees "
            "with medical experts at chance level. The main study on MIMIC-IV hospital courses (Chapter 7) needs a judge whose "
            "labels can be interpreted, so this chapter scores candidate judges against the same 1,781 expert-annotated sentences "
            "with the protocol of Section 4.7 and the candidates of Section 4.9, and then adapts the strongest general model to "
            "clinical language with MedNLI [[cite:romanov2018]]. The decision rule was fixed in advance: the candidate with the "
            "highest Cohen's kappa on all sentences, with its decision threshold chosen on the other subset of summaries, becomes "
            "the judge of the main study; its precision, recall and AUROC are reported alongside every main-study result so that "
            "the residual measurement error is visible.")]
    if jc is None:
        b += [P("The candidate comparison has not been run yet.")]
        return b
    b += [("h2", "6.2 Agreement of the Candidate Judges with the Experts")]
    allr = jc[jc.group == "all"].copy()
    allr["jn"] = allr.judge.map(JUDGE_NAMES).fillna(allr.judge); allr["mn"] = allr["mode"].map(MODE_NAMES)
    allr = allr.sort_values(["kappa"], ascending=False)
    best = best_row(jc)
    b += [P(f"[[tab:judges]] reports, for every candidate and evidence mode, the area under the ROC curve of the support score, "
            f"and precision, recall, F1, specificity and kappa at the threshold chosen on the other subset. [[fig:judges]] shows "
            f"the same comparison graphically. (The DeBERTa-v3-large NLI model is the one trained on MultiNLI, FEVER-NLI, ANLI, LingNLI "
            f"and WANLI, Section 4.9.) The pilot judge reaches an AUROC of "
            f"{f2(allr[(allr.judge=='minilm_nli')&(allr['mode']=='top3')].auroc.iloc[0])} and a kappa of "
            f"{f2(allr[(allr.judge=='minilm_nli')&(allr['mode']=='top3')].kappa.iloc[0])} even with a tuned threshold. The best "
            f"candidate, {JUDGE_NAMES.get(best.judge, best.judge)} with {MODE_PHRASE.get(best['mode'], best['mode'])}, reaches "
            f"an AUROC of {f2(best.auroc)} and a kappa of {f2(best.kappa)}, with precision {f2(best.precision)} and recall "
            f"{f2(best.recall)} at a threshold of {best.tau_from_other_subset:.2f} on the support probability, and it flags "
            f"{pct0(best.flag_rate)} of sentences where the experts flag {pct0(best.expert_rate)}."),
          ("table", dict(label="judges", caption="Candidate judges scored against the medical-expert annotations on all 1,781 sentences. Thresholds are chosen on the other subset (doctor-written for the generated summaries and vice versa; for the all-sentences row the two are pooled). Flag rate is the share of sentences the judge marks unsupported; the experts marked 20.4 percent.",
                         columns=["Judge", "Evidence", "AUROC", "Precision", "Recall", "Specificity", "F1", "Kappa", "Flag rate", "τ"],
                         widths=[1.55, 0.85, 0.55, 0.6, 0.5, 0.7, 0.4, 0.5, 0.45, 0.4], font=8, align=["left", "left"] + ["center"] * 8,
                         rows=[[r.jn, r.mn, f2(r.auroc), f2(r.precision), f2(r.recall), f2(r.specificity), f2(r.f1), f2(r.kappa), pct0(r.flag_rate), f"{r.tau_from_other_subset:.2f}"] for _, r in allr.iterrows()])),
          ("figure", dict(label="judges", path="results/fig_judges.png", width=6.3, caption="AUROC and kappa of each candidate judge and evidence mode against the expert annotations, all sentences."))]
    gen = jc[jc.group == "generated"].sort_values("kappa", ascending=False); doc = jc[jc.group == "doctor_written"].sort_values("kappa", ascending=False)
    _A = jc[(jc.group == "all") & (jc.judge != "minilm_nli")].pivot_table(index="judge", columns="mode", values="kappa")
    _A = _A.dropna(subset=[c for c in ("doc", "top3") if c in _A.columns])
    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    mode_total = WORDS.get(len(_A), str(len(_A))); mode_wins = WORDS.get(int((_A["doc"] > _A["top3"]).sum()), str(int((_A["doc"] > _A["top3"]).sum())))
    bg, bd = gen.iloc[0], doc.iloc[0]
    b += [P(f"Agreement differs between the two kinds of summaries. On LLM-generated summaries, where experts flagged "
            f"{pct0(bg.expert_rate)} of sentences, the best candidate ({JUDGE_NAMES.get(bg.judge, bg.judge)}, {MODE_PHRASE.get(bg['mode'])}) reaches "
            f"kappa {f2(bg.kappa)} and AUROC {f2(bg.auroc)}; on doctor-written instructions, where experts flagged {pct0(bd.expert_rate)}, "
            f"the best candidate ({JUDGE_NAMES.get(bd.judge, bd.judge)}, {MODE_PHRASE.get(bd['mode'])}) reaches kappa {f2(bd.kappa)} and AUROC "
            f"{f2(bd.auroc)}. Two design lessons emerge. Model size and training data matter more than the aggregation rule: the "
            f"large models improve on the pilot judge in every mode, whereas Chapter 5 showed that changing the aggregation rule "
            f"of the small model changes little. And for {mode_wins} of the {mode_total} large candidates the whole-course evidence "
            f"mode, which lets the model see every sentence of the hospital course, gives a higher kappa than three retrieved "
            f"sentences, because paraphrased patient-facing sentences draw on several parts of the course at once.")]
    bmc = jc[(jc.judge == "bespoke_minicheck_7b") & (jc.group == "all")]
    if len(bmc):
        bd = bmc[bmc["mode"] == "doc"]; bt = bmc[bmc["mode"] == "top3"]; deb = jc[(jc.judge == "deberta_large_nli") & (jc["mode"] == "doc") & (jc.group == "all")].iloc[0]
        bdr = bd.iloc[0] if len(bd) else None
        b += [P(f"The language-model checker behaves differently from the encoders. Bespoke-MiniCheck-7B, which reads the whole hospital course in "
                f"one pass and answers yes or no, reaches an AUROC of {f2(bdr.auroc) if bdr is not None else '—'} and a kappa of "
                f"{f2(bdr.kappa) if bdr is not None else '—'} with whole-course evidence (precision {f2(bdr.precision) if bdr is not None else '—'}, recall "
                f"{f2(bdr.recall) if bdr is not None else '—'}, flag rate {pct0(bdr.flag_rate) if bdr is not None else '—'}) and "
                f"{f2(bt.iloc[0].kappa) if len(bt) else '—'} with three retrieved sentences. "
                + (f"That is {'above' if bdr.kappa > deb.kappa else 'below'} the best encoder ({f2(deb.kappa)}), so "
                   + ("the pre-specified rule selects it for the main study." if bdr.kappa > deb.kappa else
                      "a model that tops the general fact-checking leaderboards does not agree with medical experts better than a strong NLI encoder on "
                      "patient-facing sentences; the disagreement is not a matter of model size or reading the whole document.") if bdr is not None else ""))]
    b += [("h2", "6.3 Adaptation to Clinical Language with MedNLI")]
    if ftj is not None:
        mr = jc[(jc.judge == "mednli_deberta_large") & (jc.group == "all")]
        lr_txt = f"{ftj['lr']:.0e}".replace("e-0", "e-")
        epochs_txt = "one epoch" if ftj["epochs"] == 1 else f"{ftj['epochs']} epochs"
        opt_txt = "Adafactor" if ftj.get("optimizer") == "adafactor" else "AdamW"
        zs = f" from {pct(ftj['zero_shot_dev_accuracy'])} to" if "zero_shot_dev_accuracy" in ftj else " to"
        zst = f" from {pct(ftj['zero_shot_test_accuracy'])} to" if "zero_shot_test_accuracy" in ftj else " to"
        hours = ftj.get("train_minutes", 0) / 60
        mdoc = mr[mr["mode"] == "doc"]; mtop = mr[mr["mode"] == "top3"]
        agree = (f" Against the expert annotations the adapted model reaches an AUROC of {f2(mdoc.auroc.iloc[0]) if len(mdoc) else '—'} with "
                 f"whole-course evidence and {f2(mtop.auroc.iloc[0]) if len(mtop) else '—'} with top-3 sentence evidence, with a kappa of "
                 f"{f2(mr.kappa.max())} at best ([[tab:judges]])." if len(mr) else " Its agreement with the experts is reported in [[tab:judges]].")
        bdoc = jc[(jc.judge == "deberta_large_nli") & (jc["mode"] == "doc") & (jc.group == "all")]
        comp = ""
        if len(mdoc) and len(bdoc):
            bd, md = bdoc.iloc[0], mdoc.iloc[0]
            comp = (f" Compared with the same model before adaptation (AUROC {f2(bd.auroc)}, kappa {f2(bd.kappa)}, precision {f2(bd.precision)}, "
                    f"recall {f2(bd.recall)}), the adapted model flags {'fewer' if md.flag_rate < bd.flag_rate else 'more'} sentences ({pct0(md.flag_rate)} "
                    f"versus {pct0(bd.flag_rate)}, where the experts flag {pct0(md.expert_rate)}), "
                    f"{'keeps the same precision' if abs(md.precision - bd.precision) < 0.005 else ('gains' if md.precision > bd.precision else 'loses') + f' {abs(md.precision - bd.precision):.2f} in precision'} "
                    f"and {'loses' if md.recall < bd.recall else 'gains'} {abs(md.recall - bd.recall):.2f} in "
                    f"recall, so that its kappa is {'lower' if md.kappa < bd.kappa else 'higher'}. A nine-point gain on MedNLI did not transfer to the "
                    f"expert task. One explanation consistent with this is a difference in granularity: MedNLI pairs a single note sentence with a "
                    f"short clinical statement, whereas the expert-set sentences are patient-facing paraphrases that often combine several facts "
                    f"drawn from different parts of a hospital course, so that better clinical vocabulary does not help the model decide which part "
                    f"of a multi-fact sentence lacks support. The experiment does not establish this explanation; other differences between the "
                    f"two tasks, such as the patient-facing register and the length of the evidence, could produce the same result. Section 6.6 "
                    f"tests one implication of it by scoring atomic facts instead of sentences.")
        b += [P(f"The strongest general model was fine-tuned on MedNLI, 11,232 clinician-written premise and hypothesis pairs from "
                f"MIMIC-III notes (Section 3.4), for {epochs_txt} with the {opt_txt} optimizer, a learning rate of {lr_txt} and an "
                f"effective batch size of {ftj['batch']} (Section 4.9), which took {hours:.1f} hours on the laptop's GPU while it was "
                f"shared with the language-model server. Its MedNLI development accuracy rose{zs} {pct(ftj['dev_accuracy'])} and its test "
                f"accuracy{zst} {pct(ftj['test_accuracy'])}." + agree + comp)]
    else:
        b += [P("The MedNLI adaptation is described in Section 4.9; its results are added to [[tab:judges]] when the fine-tuning run completes.")]
    b += [("h2", "6.4 Decision")]
    b += [P(f"By the pre-specified rule the judge of the main study is {JUDGE_NAMES.get(best.judge, best.judge)} with "
            f"{MODE_PHRASE.get(best['mode'], best['mode'])} and a support threshold of {best.tau_from_other_subset:.2f}."
            + (f" The MedNLI-adapted model was not selected because its best kappa ({f2(jc[(jc.judge == 'mednli_deberta_large') & (jc.group == 'all')].kappa.max())}) "
               f"is below that of the unadapted model." if ftj is not None and best.judge != "mednli_deberta_large" and (jc.judge == "mednli_deberta_large").any() else "")
            + f" Its agreement with medical experts is "
            f"moderate rather than high: a kappa of {f2(best.kappa)} means substantial disagreement remains, its precision of "
            f"{f2(best.precision)} means that roughly {100 - round(100 * best.precision)} of every 100 sentences it flags were not "
            f"flagged by the experts, and its recall of {f2(best.recall)} means that it misses about {100 - round(100 * best.recall)} of "
            f"every 100 expert-flagged sentences. These figures are carried into Chapter 7: every unsupported fact rate reported "
            f"there is an estimate produced by an instrument with this known error profile, and the comparisons between "
            f"conditions are again paired comparisons on the same documents under the same instrument. Because the judge was "
            f"chosen on the sentences on which these figures are measured, they are optimistic; Section 6.5 estimates by how much.")]
    b += independence_blocks()
    b += atomic_blocks()
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
    if ov is not None:
        O = dict(ov.values.tolist())
        b += [P(f"The agreement figures of Section 6.2 are measured on sentences that are not independent, and the selected judge was chosen "
                f"on the same sentences. Three facts about the data, established by matching every hospital course to its MIMIC-IV-Note "
                f"discharge note (Section 4.9), determine how far they can be trusted. First, each of the {int(O['main-study documents (doctor-written + validation courses)'])} "
                f"main-study courses matches exactly one discharge note, and the notes belong to {int(O['distinct patients among matched main-study courses'])} "
                f"different patients and {int(O['distinct admissions among matched main-study courses'])} different admissions, so the paired "
                f"observations of Chapter 7 are independent at the level of the patient. Second, the 100 LLM-generated summaries of the expert set "
                f"were written from {int(O['distinct hospital courses behind the 100 LLM-generated summaries'])} further courses, five summaries per "
                f"course, and {'none of their patients or admissions appears' if int(O['generated-summary courses whose patient also appears in the main study']) == 0 and int(O['generated-summary courses whose admission also appears in the main study']) == 0 else 'some of their patients appear'} "
                f"in the main study. Third, the doctor-written summaries of the expert set are the clinicians' instructions for the main-study "
                f"courses themselves. The judge was therefore selected partly on text written for the same courses whose generated summaries it "
                f"later scores; the generated summaries of the main study were never part of the selection, but their courses were.")]
    if sb is not None:
        def g(analysis, metric, col="estimate"):
            r = sb[(sb.analysis == analysis) & (sb.metric == metric)]
            return float(r[col].iloc[0]) if len(r) else float("nan")
        def note(analysis):
            r = sb[(sb.analysis == analysis) & sb.note.notna()]
            return r.note.iloc[0] if len(r) else None
        bs = "selected_judge_course_bootstrap"; cv = "nested_cv_selection_procedure"
        k_app = g("apparent_selected_judge", "kappa")
        b += [P(f"Resampling hospital courses instead of sentences, which respects the dependence of sentences written for the same course, "
                f"the selected judge's kappa of {f2(k_app)} has a 95 percent interval from {f2(g(bs, 'kappa', 'lo'))} to {f2(g(bs, 'kappa', 'hi'))}, "
                f"its AUROC of {f2(g(bs, 'auroc'))} an interval from {f2(g(bs, 'auroc', 'lo'))} to {f2(g(bs, 'auroc', 'hi'))}, its precision "
                f"from {f2(g(bs, 'precision', 'lo'))} to {f2(g(bs, 'precision', 'hi'))} and its recall from {f2(g(bs, 'recall', 'lo'))} to "
                f"{f2(g(bs, 'recall', 'hi'))}.")]
        k_cv, opt = g(cv, "kappa"), k_app - g(cv, "kappa")
        sel_share = None
        if fq is not None and len(fq):
            sel = fq[fq.candidate == "deberta_large_nli__doc"]
            sel_share = float(sel.share.iloc[0]) if len(sel) else 0.0
            others = []
            for c, sh in zip(fq.candidate, fq.share):
                if c != "deberta_large_nli__doc":
                    j, m = c.split("__")
                    others.append(f"{JUDGE_NAMES.get(j, j)} with {MODE_PHRASE.get(m, m)} ({pct0(sh)})")
            other_txt = f", and otherwise {_join_names(others)}" if others else ""
        b += [P(f"Choosing the best of {int(len(JUDGE_NAMES))} candidates in two evidence modes and over a grid of thresholds on the same "
                f"sentences inflates the agreement of the winner. Nested cross-validation estimates the agreement that the selection "
                f"procedure itself achieves on courses it has not seen: the courses are split into five folds, the whole procedure "
                f"(candidate, evidence mode and threshold chosen by maximum kappa) is run on four folds and its choice is applied to the "
                f"fifth, and the split is repeated twenty times. The procedure reaches a kappa of {f2(k_cv)} (from {f2(g(cv, 'kappa', 'lo'))} to "
                f"{f2(g(cv, 'kappa', 'hi'))} over the repetitions), an AUROC of {f2(g(cv, 'auroc'))}, precision {f2(g(cv, 'precision'))} and recall "
                f"{f2(g(cv, 'recall'))}, against the apparent kappa of {f2(k_app)} of the judge it selects on all sentences, so the optimism "
                f"of the apparent figure is about {f2(opt)} in kappa."
                + (f" The procedure chose the selected configuration, {JUDGE_NAMES['deberta_large_nli']} with whole-course evidence, in "
                   f"{pct0(sel_share)} of the 100 folds{other_txt}." if sel_share is not None else ""))]
        t1, t2 = "select_on_doctor_test_on_generated", "select_on_generated_test_on_doctor"
        n1, n2 = note(t1), note(t2)
        if n1 and n2:
            j1, m1, tau1 = _cand(n1); j2, m2, tau2 = _cand(n2)
            b += [P(f"The patient-disjoint transfer test asks the same question across patients. Selected on the doctor-written sentences "
                    f"alone, which belong to the main-study courses, the procedure chooses {j1} with {m1} at a threshold of {tau1:.2f} (kappa "
                    f"{f2(g(t1, 'training_kappa'))} on those sentences) and reaches a kappa of {f2(g(t1, 'kappa'))} and an AUROC of "
                    f"{f2(g(t1, 'auroc'))} on the LLM-generated summaries of the {int(O['distinct hospital courses behind the 100 LLM-generated summaries']) if ov is not None else 20} "
                    f"other courses. Selected on the generated summaries, it chooses {j2} with {m2} at {tau2:.2f} (training kappa "
                    f"{f2(g(t2, 'training_kappa'))}) and reaches {f2(g(t2, 'kappa'))} and {f2(g(t2, 'auroc'))} on the doctor-written "
                    f"sentences. In both directions the agreement on unseen patients and on a different kind of summary is "
                    f"{'lower than' if max(g(t1, 'kappa'), g(t2, 'kappa')) < k_app - 0.02 else 'close to'} the apparent agreement, and "
                    f"the AUROC, which does not depend on the threshold, "
                    f"{'changes little' if abs(g(t1, 'auroc') - g(bs, 'auroc')) < 0.05 and abs(g(t2, 'auroc') - g(bs, 'auroc')) < 0.05 else 'also changes'}: "
                    f"{'the ranking ability of the judge transfers, while its threshold and the expected flag rate depend on the kind of summary' if abs(g(t1, 'auroc') - g(bs, 'auroc')) < 0.05 and abs(g(t2, 'auroc') - g(bs, 'auroc')) < 0.05 else 'both its ranking ability and its threshold depend on the kind of summary'}.")]
    if ps is not None and len(ps):
        order = [k for k in SOURCE_NAMES if k in set(ps.group)]
        P_ = ps.set_index("group")
        lo, hi = P_.kappa.idxmin(), P_.kappa.idxmax()
        b += [P(f"[[tab:persource]] reports the selected judge's agreement separately for each source of summaries. Agreement is not "
                f"uniform: kappa ranges from {f2(P_.loc[lo, 'kappa'])} on {SOURCE_NAMES.get(lo, lo).lower().replace('gpt', 'GPT').replace('llama', 'Llama')} "
                f"to {f2(P_.loc[hi, 'kappa'])} on {SOURCE_NAMES.get(hi, hi).lower().replace('gpt', 'GPT').replace('llama', 'Llama')}, and precision is lowest where the experts "
                f"found the least unsupported content, as expected when a judge with a fixed error rate is applied to texts with fewer real "
                f"errors. Each LLM source has only 20 courses, so these per-source figures are imprecise. The main study's generator, "
                f"Qwen2.5-7B-Instruct, is not among the sources; how the judge behaves on its summaries is measured directly by the "
                f"manual audit of Section 7.9."),
              ("table", dict(label="persource", caption="Agreement of the selected judge (DeBERTa-v3-large NLI, whole-course evidence, τ = 0.20) with the experts, by source of the summaries.",
                             columns=["Source", "Sentences", "Expert rate", "Flag rate", "Precision", "Recall", "Kappa", "AUROC"],
                             widths=[1.85, 0.75, 0.7, 0.65, 0.7, 0.6, 0.55, 0.6], font=9, align=["left"] + ["center"] * 7,
                             rows=[[SOURCE_NAMES.get(k, k), str(int(P_.loc[k, 'n'])), pct0(P_.loc[k, 'expert_rate']), pct0(P_.loc[k, 'flag_rate']),
                                    f2(P_.loc[k, 'precision']), f2(P_.loc[k, 'recall']), f2(P_.loc[k, 'kappa']), f2(P_.loc[k, 'auroc'])] for k in order]))]
    b += [P("For the main study these analyses mean two things. The selected judge's agreement figures are an optimistic estimate "
            "of its accuracy, and its error rates on a new generator are not known from the expert set alone. The comparisons of "
            "Chapter 7 do not require the judge to be accurate in absolute terms, but they do require its errors to be similar "
            "across the conditions being compared, which is the assumption the manual audit tests.")]
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
    b += [P("Patient-facing sentences often combine several facts, and the judge assigns one label to the whole sentence. To test "
            "whether finer units agree better with the experts, the local model split each expert-set sentence into atomic facts, "
            "each carrying a single piece of information (prompt in Appendix A), and every fact was scored by the selected judge "
            "against the whole hospital course. A sentence counts as flagged when its weakest fact falls below a threshold chosen, "
            "as before, on the other subset of summaries."
            + (f" Sentences contain {fx['mean']:.1f} facts on average and {pct0(fx.share_multi)} of them contain more than one; "
               f"{pct(fx.share_fallback)} could not be decomposed and were kept whole." if fx is not None else ""))]
    if a is not None and sn is not None:
        better = a.kappa > sn.kappa + 0.02; worse = a.kappa < sn.kappa - 0.02
        b += [P(f"Scored by their weakest atomic fact, the sentences agree with the experts at a kappa of {f2(a.kappa)} and an AUROC of "
            f"{f2(a.auroc)} (precision {f2(a.precision)}, recall {f2(a.recall)}, flag rate {pct0(a.flag_rate)}), against {f2(sn.kappa)} "
            f"and {f2(sn.auroc)} for the same judge scoring whole sentences. "
            + ("Atomic decomposition therefore improves agreement, which supports the granularity explanation of Section 6.3 without "
               "proving it, and Chapter 7 reports the main-study rates at the level of facts as well as sentences." if better else
               ("Atomic decomposition therefore lowers agreement: requiring every fact to be supported multiplies the chances of a "
                "false flag, because each fact, read alone, loses the context that its sentence supplied." if worse else
                "Atomic decomposition therefore leaves agreement essentially unchanged.")
               + " The granularity explanation of Section 6.3 is not supported in this form: if agreement is limited by granularity, "
               "splitting sentences into facts with the same judge does not remove the limitation. Chapter 7 reports the main-study "
               "rates at the level of facts as a robustness check."))]
    if ex is not None and len(ex):
        rows = []
        for i, (sent, g) in enumerate(ex.groupby("sentence", sort=False)):
            for j, r in enumerate(g.itertuples()):
                rows.append([f"{sent} ({r.sentence_support:.2f})" if j == 0 else "", r.fact, f"{r.fact_support:.2f}"])
        mixed = sum(1 for _, g in ex.groupby("sentence", sort=False) if (g.fact_support >= g.tau).any() and (g.fact_support < g.tau).any())
        b += [P(f"[[tab:atomicex]] illustrates the decomposition on public text: the longest full-note summary sentence of each of the "
                f"first {ex.sentence.nunique()} pilot documents, written by GPT-4o-mini from an MTSamples note, split by the local model and "
                f"scored by the selected judge against the whole note. "
                + (f"In {mixed} of the {ex.sentence.nunique()} sentences some facts are supported and others are not, which a single "
                   f"sentence-level score cannot express." if mixed else
                   "In these examples the facts of each sentence agree in support, so the sentence-level score loses nothing."))]
        b += [("table", dict(label="atomicex", caption=f"Atomic decomposition of three pilot summary sentences (public MTSamples data). The number after each sentence and each fact is the selected judge's support probability against the whole note; values below {ex.tau.iloc[0]:.2f} are flagged.",
                             columns=["Sentence (support)", "Atomic fact", "Support"], widths=[2.9, 2.9, 0.7], font=8.5,
                             align=["left", "left", "center"], rows=rows))]
    return b
