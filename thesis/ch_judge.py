"""ch_judge.py — Chapter 6: Selecting a Valid Judge (candidate judges scored against the expert annotations; MedNLI adaptation)."""
import json
from pathlib import Path

import pandas as pd

from results_loader import Results, f2, f3, pct, pct0, fp

ROOT = Path(__file__).resolve().parent.parent
JUDGE_NAMES = {"minilm_nli": "MiniLM cross-encoder (pilot judge)", "deberta_large_nli": "DeBERTa-v3-large NLI",
               "ce_deberta_large_nli": "DeBERTa-v3-large cross-encoder", "minicheck_deberta": "MiniCheck DeBERTa-v3-large",
               "minicheck_roberta": "MiniCheck RoBERTa-large", "mednli_deberta_large": "DeBERTa-v3-large NLI + MedNLI"}
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
            "with the protocol of Section 4.7 and the candidates of Section 4.10, and then adapts the strongest general model to "
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
            f"and WANLI, Section 4.10.) The pilot judge reaches an AUROC of "
            f"{f2(allr[(allr.judge=='minilm_nli')&(allr['mode']=='top3')].auroc.iloc[0])} and a kappa of "
            f"{f2(allr[(allr.judge=='minilm_nli')&(allr['mode']=='top3')].kappa.iloc[0])} even with a tuned threshold. The best "
            f"candidate, {JUDGE_NAMES.get(best.judge, best.judge)} with {MODE_PHRASE.get(best['mode'], best['mode'])}, reaches "
            f"an AUROC of {f2(best.auroc)} and a kappa of {f2(best.kappa)}, with precision {f2(best.precision)} and recall "
            f"{f2(best.recall)} at a threshold of {best.tau_from_other_subset:.2f} on the support probability, and it flags "
            f"{pct0(best.flag_rate)} of sentences where the experts flag {pct0(best.expert_rate)}."),
          ("table", dict(label="judges", caption="Candidate judges scored against the medical-expert annotations on all 1,781 sentences. Thresholds are chosen on the other subset (doctor-written for the generated summaries and vice versa; for the all-sentences row the two are pooled). Flag rate is the share of sentences the judge marks unsupported; the experts marked 20.4 percent.",
                         columns=["Judge", "Evidence", "AUROC", "Precision", "Recall", "Specificity", "F1", "Kappa", "Flag rate", "τ"],
                         widths=[1.7, 0.85, 0.5, 0.6, 0.5, 0.7, 0.4, 0.5, 0.45, 0.3], font=8, align=["left", "left"] + ["center"] * 8,
                         rows=[[r.jn, r.mn, f2(r.auroc), f2(r.precision), f2(r.recall), f2(r.specificity), f2(r.f1), f2(r.kappa), pct0(r.flag_rate), f"{r.tau_from_other_subset:.2f}"] for _, r in allr.iterrows()])),
          ("figure", dict(label="judges", path="results/fig_judges.png", width=6.3, caption="AUROC and kappa of each candidate judge and evidence mode against the expert annotations, all sentences."))]
    gen = jc[jc.group == "generated"].sort_values("kappa", ascending=False); doc = jc[jc.group == "doctor_written"].sort_values("kappa", ascending=False)
    bg, bd = gen.iloc[0], doc.iloc[0]
    b += [P(f"Agreement differs between the two kinds of summaries. On LLM-generated summaries, where experts flagged "
            f"{pct0(bg.expert_rate)} of sentences, the best candidate ({JUDGE_NAMES.get(bg.judge, bg.judge)}, {MODE_PHRASE.get(bg['mode'])}) reaches "
            f"kappa {f2(bg.kappa)} and AUROC {f2(bg.auroc)}; on doctor-written instructions, where experts flagged {pct0(bd.expert_rate)}, "
            f"the best candidate ({JUDGE_NAMES.get(bd.judge, bd.judge)}, {MODE_PHRASE.get(bd['mode'])}) reaches kappa {f2(bd.kappa)} and AUROC "
            f"{f2(bd.auroc)}. Two design lessons emerge. Model size and training data matter more than the aggregation rule: the "
            f"large models improve on the pilot judge in every mode, whereas Chapter 5 showed that changing the aggregation rule "
            f"of the small model changes little. And for the large models the whole-course evidence mode, which lets the model "
            f"see every sentence of the hospital course in windows, is competitive with or better than three retrieved "
            f"sentences, because paraphrased patient-facing sentences draw on several parts of the course at once.")]
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
                    f"recall, so that its kappa is {'lower' if md.kappa < bd.kappa else 'higher'}. A nine-point gain on MedNLI, whose hypotheses are short "
                    f"clinical statements paired with a single note sentence, therefore did not transfer to the expert task, whose sentences are "
                    f"patient-facing paraphrases that draw on several parts of a hospital course. This points to the granularity of the comparison, "
                    f"not clinical vocabulary, as the factor that limits agreement with the experts.")
        b += [P(f"The strongest general model was fine-tuned on MedNLI, 11,232 clinician-written premise and hypothesis pairs from "
                f"MIMIC-III notes (Section 3.4), for {epochs_txt} with the {opt_txt} optimizer, a learning rate of {lr_txt} and an "
                f"effective batch size of {ftj['batch']} (Section 4.10), which took {hours:.1f} hours on the laptop's GPU while it was "
                f"shared with the language-model server. Its MedNLI development accuracy rose{zs} {pct(ftj['dev_accuracy'])} and its test "
                f"accuracy{zst} {pct(ftj['test_accuracy'])}." + agree + comp)]
    else:
        b += [P("The MedNLI adaptation is described in Section 4.10; its results are added to [[tab:judges]] when the fine-tuning run completes.")]
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
            f"conditions are again paired comparisons on the same documents under the same instrument.")]
    return b
