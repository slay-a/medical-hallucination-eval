#!/usr/bin/env python3
"""
calibration_annptsumm.py — Validate the NLI judge against medical-expert annotations.

Data: ann-pt-summ v1.0.1 (Hegselmann et al., 2024; PhysioNet, credentialed access).
  hallucinations_generated_di.jsonl          100 LLM-generated patient summaries
                                             (lines 1-20 llama_70b_original, 21-40 llama_70b_cleaned,
                                              41-60 gpt4_zero_shot, 61-80 gpt4_orig, 81-100 gpt4_cleaned)
  hallucinations_mimic_di.jsonl              100 doctor-written summaries
  hallucinations_mimic_di_validation.jsonl    10 doctor-written summaries (validation split)
Each line: {"text": context, "summary": summary, "labels": [{"start","end","length","text","label"}, ...]}
where every label span marks an unsupported fact agreed by two medical experts.

Privacy: the data are MIMIC-derived and never leave this machine.  Only local models
are used.  Aggregate statistics are written to results/; per-sentence details are
written to results_private/, which is excluded from version control.

Usage:  python calibration_annptsumm.py [--data-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he                    # noqa: E402
from preprocessing import is_markdown_header      # noqa: E402

RESULTS = HERE / "results"
PRIVATE = HERE / "results_private"
ARCHIVE_DIR = ("medical-expert-annotations-of-unsupported-facts-in-doctor-written-and-llm-generated-"
               "patient-summaries-1.0.1")
DEFAULT_DATA = Path.home() / "Desktop" / "Thesis" / "ann-pt-summ-1.0.1-partial" / ARCHIVE_DIR / "hallucination_datasets"
SYSTEMS = ["llama_70b_original", "llama_70b_cleaned", "gpt4_zero_shot", "gpt4_orig", "gpt4_cleaned"]


def read_jsonl(path: Path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary_sentences(summary: str, nlp, min_len: int = 10):
    """Sentences with character offsets; drops very short lines and markdown headers."""
    out = []
    for s in nlp(summary).sents:
        t = s.text.strip()
        if len(t) < min_len or is_markdown_header(t):
            continue
        # offsets of the stripped text
        lead = len(s.text) - len(s.text.lstrip())
        start = s.start_char + lead
        out.append((t, start, start + len(t)))
    return out


def judge(claims, context_sents, k=3):
    """Same retrieval + NLI rule as the main pipeline, batched."""
    if not claims:
        return []
    enc, ce = he.get_bi_encoder(), he.get_cross_encoder()
    if not context_sents:
        return [("Not-Supported", 0.0, 0.0)] * len(claims)
    S = enc.encode(context_sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    C = enc.encode(claims, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    top = np.argsort(-(C @ S.T), axis=1)[:, :min(k, len(context_sents))]
    pairs, owners = [], []
    for i in range(len(claims)):
        for j in top[i]:
            pairs.append([context_sents[j], claims[i]]); owners.append(i)
    logits = np.atleast_2d(ce.predict(pairs, batch_size=64, show_progress_bar=False))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True)); probs /= probs.sum(axis=1, keepdims=True)
    owners = np.array(owners)
    out = []
    for i in range(len(claims)):
        p = probs[owners == i]
        pe, pc = float(p[:, he.IDX_ENTAILMENT].max()), float(p[:, he.IDX_CONTRADICTION].max())
        if pe >= he.ENTAILMENT_THRESHOLD and pe > pc:
            lab = "Supported"
        elif pc >= he.CONTRADICTION_THRESHOLD and pc > pe:
            lab = "Contradicted"
        else:
            lab = "Not-Supported"
        out.append((lab, pe, pc))
    return out


def confusion(expert: np.ndarray, flag: np.ndarray) -> dict:
    tp = int((expert & flag).sum()); fp = int((~expert & flag).sum())
    fn = int((expert & ~flag).sum()); tn = int((~expert & ~flag).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) else np.nan
    kappa = cohen_kappa_score(expert, flag) if len(set(expert)) > 1 and len(set(flag)) > 1 else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, specificity=spec, f1=f1,
                accuracy=(tp + tn) / max(1, tp + fp + fn + tn),
                balanced_accuracy=np.nanmean([rec, spec]), kappa=kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    PRIVATE.mkdir(exist_ok=True)

    gen = read_jsonl(data_dir / "hallucinations_generated_di.jsonl")
    doc = read_jsonl(data_dir / "hallucinations_mimic_di.jsonl")
    val = read_jsonl(data_dir / "hallucinations_mimic_di_validation.jsonl")
    print(f"loaded generated={len(gen)} doctor={len(doc)} validation={len(val)}")
    print("record keys:", sorted(gen[0].keys()), "| label keys:", sorted(gen[0]["labels"][0].keys()) if gen[0]["labels"] else "n/a")

    items = []
    for i, r in enumerate(gen):
        items.append(dict(group=SYSTEMS[i // 20], source="generated", sid=f"gen_{i}", **r))
    for i, r in enumerate(doc):
        items.append(dict(group="doctor_written", source="doctor", sid=f"doc_{i}", **r))
    for i, r in enumerate(val):
        items.append(dict(group="doctor_written", source="doctor", sid=f"val_{i}", **r))

    # offset sanity check
    mism = tot = 0
    for it in items:
        for lb in it["labels"]:
            tot += 1
            if it["summary"][lb["start"]:lb["end"]].strip() != str(lb.get("text", "")).strip():
                mism += 1
    print(f"label spans: {tot}; offset/text mismatches: {mism}")
    label_values = pd.Series([lb["label"] for it in items for lb in it["labels"]]).value_counts()
    print("expert label types:\n", label_values.to_string())

    nlp = he.get_nlp(); he.get_bi_encoder(); he.get_cross_encoder()
    rows = []
    for n, it in enumerate(items):
        sents = summary_sentences(it["summary"], nlp)
        ctx = he.sentencize(it["text"])
        res = judge([s for s, _, _ in sents], ctx, k=he.TOP_K_EVIDENCE)
        for (text, a, b), (lab, pe, pc) in zip(sents, res):
            overl = [lb for lb in it["labels"] if max(a, lb["start"]) < min(b, lb["end"])]
            rows.append(dict(sid=it["sid"], group=it["group"], source=it["source"], sent_start=a, sent_end=b,
                             sentence=text, judge_label=lab, p_entailment=round(pe, 4), p_contradiction=round(pc, 4),
                             expert_flag=bool(overl), n_expert_spans=len(overl),
                             expert_labels="|".join(sorted({lb["label"] for lb in overl}))))
        if (n + 1) % 30 == 0:
            print(f"   judged {n+1}/{len(items)} summaries")
    df = pd.DataFrame(rows)
    df.to_csv(PRIVATE / "calibration_sentences.csv", index=False)   # contains MIMIC-derived text: private
    df["flag_any"] = df.judge_label != "Supported"
    df["flag_contra"] = df.judge_label == "Contradicted"
    print(f"sentences judged: {len(df)}; expert-flagged: {int(df.expert_flag.sum())} ({df.expert_flag.mean():.3f})")

    # ── overall / per-group table ─────────────────────────────────────────────
    groups = [("all", df), ("generated", df[df.source == "generated"]), ("doctor_written", df[df.source == "doctor"])]
    groups += [(g, df[df.group == g]) for g in SYSTEMS]
    out = []
    for name, d in groups:
        e, fa, fc = d.expert_flag.to_numpy(bool), d.flag_any.to_numpy(bool), d.flag_contra.to_numpy(bool)
        ca, cc = confusion(e, fa), confusion(e, fc)
        try:
            au_e = roc_auc_score(e, 1 - d.p_entailment) if len(set(e)) > 1 else np.nan
            au_c = roc_auc_score(e, d.p_contradiction) if len(set(e)) > 1 else np.nan
        except ValueError:
            au_e = au_c = np.nan
        out.append(dict(group=name, n_summaries=d.sid.nunique(), n_sentences=len(d), expert_flag_rate=e.mean(),
                        judge_UFR=fa.mean(), judge_CR=fc.mean(),
                        **{f"any_{k}": v for k, v in ca.items()}, **{f"contra_{k}": v for k, v in cc.items()},
                        auroc_1_minus_p_entail=au_e, auroc_p_contra=au_c))
    pd.DataFrame(out).to_csv(RESULTS / "calibration_overall.csv", index=False)

    # ── by expert label type: recall of the judge ─────────────────────────────
    lab_rows = []
    fl = df[df.expert_flag].copy()
    fl["expert_labels"] = fl.expert_labels.str.split("|")
    fl = fl.explode("expert_labels")
    for lab, d in fl.groupby("expert_labels"):
        lab_rows.append(dict(expert_label=lab, n_sentences=len(d), judge_recall_any=d.flag_any.mean(),
                             judge_recall_contradicted=d.flag_contra.mean(), mean_p_entail=d.p_entailment.mean()))
    lab_rows.append(dict(expert_label="__span_counts__", n_sentences=int(label_values.sum()), judge_recall_any=np.nan,
                         judge_recall_contradicted=np.nan, mean_p_entail=np.nan))
    pd.DataFrame(lab_rows).to_csv(RESULTS / "calibration_by_label.csv", index=False)
    label_values.rename_axis("expert_label").reset_index(name="n_spans").to_csv(RESULTS / "calibration_label_counts.csv", index=False)

    # ── threshold sweep ───────────────────────────────────────────────────────
    sw = []
    for tau in np.round(np.arange(0.30, 0.96, 0.05), 2):
        for name, d in groups[:3]:
            e = d.expert_flag.to_numpy(bool)
            flag = ~((d.p_entailment >= tau) & (d.p_entailment > d.p_contradiction)).to_numpy(bool)
            c = confusion(e, flag)
            sw.append(dict(tau=tau, group=name, judge_flag_rate=flag.mean(), **c))
    pd.DataFrame(sw).to_csv(RESULTS / "calibration_threshold_sweep.csv", index=False)

    # ── summary-level agreement ───────────────────────────────────────────────
    per = df.groupby(["sid", "group", "source"]).agg(n_sent=("sentence", "size"), expert_frac=("expert_flag", "mean"),
                                                     expert_spans=("n_expert_spans", "sum"), judge_UFR=("flag_any", "mean"),
                                                     judge_CR=("flag_contra", "mean")).reset_index()
    per.drop(columns=[]).to_csv(PRIVATE / "calibration_per_summary.csv", index=False)
    sl = []
    for name, d in (("all", per), ("generated", per[per.source == "generated"]), ("doctor_written", per[per.source == "doctor"])):
        for jm in ("judge_UFR", "judge_CR"):
            rho, p = spearmanr(d[jm], d.expert_frac); r, pp = pearsonr(d[jm], d.expert_frac)
            sl.append(dict(group=name, judge_metric=jm, n=len(d), spearman_rho=rho, spearman_p=p, pearson_r=r, pearson_p=pp))
    pd.DataFrame(sl).to_csv(RESULTS / "calibration_summary_level.csv", index=False)
    # system ranking: expert vs judge
    sysm = per.groupby("group")[["expert_frac", "judge_UFR", "judge_CR"]].mean().reset_index()
    sysm.to_csv(RESULTS / "calibration_by_system_means.csv", index=False)
    print("\nper-system means:\n", sysm.round(3).to_string(index=False))
    print("\noverall:\n", pd.DataFrame(out)[["group", "n_sentences", "expert_flag_rate", "judge_UFR", "judge_CR", "any_precision",
                                              "any_recall", "any_f1", "any_kappa", "auroc_1_minus_p_entail"]].round(3).to_string(index=False))
    print("\nby expert label:\n", pd.DataFrame(lab_rows).round(3).to_string(index=False))
    print("\ncalibration complete")


if __name__ == "__main__":
    main()
