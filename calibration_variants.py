#!/usr/bin/env python3
"""
calibration_variants.py — Does the way evidence is aggregated change how well the NLI
judge agrees with medical experts?  Evaluated on the ann-pt-summ expert-annotated set.

Variants (all use the same cross-encoder and the same summary sentences):
  max3       current pipeline: score claim against each of the top-3 context sentences, take the max
  concat3    concatenate the top-3 sentences (document order) into one premise
  concat5    concatenate the top-5 sentences into one premise
  maxall     score claim against every context sentence, take the max (SummaC-ZS style)
  concat_ctx whole context as one premise (truncated by the model's 512-token limit)

Outputs: results/calibration_variants.csv (aggregate only) and
         results_private/calibration_variants_sentences.csv (contains MIMIC-derived text; git-ignored)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA, SYSTEMS, read_jsonl, summary_sentences, confusion  # noqa: E402

RESULTS, PRIVATE = HERE / "results", HERE / "results_private"


def probs_for(pairs, ce):
    logits = np.atleast_2d(ce.predict(pairs, batch_size=64, show_progress_bar=False))
    p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    return p[:, he.IDX_ENTAILMENT], p[:, he.IDX_CONTRADICTION]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=str(DEFAULT_DATA)); args = ap.parse_args()
    d = Path(args.data_dir)
    items = []
    for i, r in enumerate(read_jsonl(d / "hallucinations_generated_di.jsonl")):
        items.append(dict(group=SYSTEMS[i // 20], source="generated", sid=f"gen_{i}", **r))
    for i, r in enumerate(read_jsonl(d / "hallucinations_mimic_di.jsonl")):
        items.append(dict(group="doctor_written", source="doctor", sid=f"doc_{i}", **r))
    for i, r in enumerate(read_jsonl(d / "hallucinations_mimic_di_validation.jsonl")):
        items.append(dict(group="doctor_written", source="doctor", sid=f"val_{i}", **r))
    nlp, enc, ce = he.get_nlp(), he.get_bi_encoder(), he.get_cross_encoder()
    PRIVATE.mkdir(exist_ok=True)

    rows = []
    for n, it in enumerate(items):
        sents = summary_sentences(it["summary"], nlp)
        ctx = he.sentencize(it["text"])
        if not sents or not ctx:
            continue
        claims = [s for s, _, _ in sents]
        S = enc.encode(ctx, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        C = enc.encode(claims, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        order = np.argsort(-(C @ S.T), axis=1)
        out = {k: ([], []) for k in ("max3", "concat3", "concat5", "maxall", "concat_ctx")}
        # max3 / maxall: all pairs at once
        pairs, owners, in_top3 = [], [], []
        for i in range(len(claims)):
            top3 = set(order[i][:3].tolist())
            for j in range(len(ctx)):
                pairs.append([ctx[j], claims[i]]); owners.append(i); in_top3.append(j in top3)
        pe, pc = probs_for(pairs, ce); owners = np.array(owners); in_top3 = np.array(in_top3)
        for i in range(len(claims)):
            m = owners == i
            out["maxall"][0].append(pe[m].max()); out["maxall"][1].append(pc[m].max())
            m3 = m & in_top3
            out["max3"][0].append(pe[m3].max()); out["max3"][1].append(pc[m3].max())
        # concatenations
        for name, k in (("concat3", 3), ("concat5", 5)):
            prem = [" ".join(ctx[j] for j in sorted(order[i][:k].tolist())) for i in range(len(claims))]
            pe, pc = probs_for([[p, c] for p, c in zip(prem, claims)], ce)
            out[name][0].extend(pe.tolist()); out[name][1].extend(pc.tolist())
        pe, pc = probs_for([[it["text"], c] for c in claims], ce)
        out["concat_ctx"][0].extend(pe.tolist()); out["concat_ctx"][1].extend(pc.tolist())
        for i, (text, a, b) in enumerate(sents):
            flagged = any(max(a, lb["start"]) < min(b, lb["end"]) for lb in it["labels"])
            row = dict(sid=it["sid"], group=it["group"], source=it["source"], sentence=text, expert_flag=flagged)
            for k in out:
                row[f"{k}_pe"], row[f"{k}_pc"] = float(out[k][0][i]), float(out[k][1][i])
            rows.append(row)
        if (n + 1) % 30 == 0:
            print(f"  {n+1}/{len(items)} summaries")
    df = pd.DataFrame(rows)
    df.to_csv(PRIVATE / "calibration_variants_sentences.csv", index=False)

    res = []
    for k in ("max3", "concat3", "concat5", "maxall", "concat_ctx"):
        for gname, g in (("all", df), ("generated", df[df.source == "generated"]), ("doctor_written", df[df.source == "doctor"])):
            e = g.expert_flag.to_numpy(bool); pe, pc = g[f"{k}_pe"].to_numpy(), g[f"{k}_pc"].to_numpy()
            flag = ~((pe >= 0.5) & (pe > pc)); c = confusion(e, flag)
            best_f1 = best_kappa = (np.nan, np.nan)
            for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
                f = ~((pe >= tau) & (pe > pc)); cc = confusion(e, f)
                if not np.isnan(cc["f1"]) and (np.isnan(best_f1[1]) or cc["f1"] > best_f1[1]):
                    best_f1 = (tau, cc["f1"])
                if not np.isnan(cc["kappa"]) and (np.isnan(best_kappa[1]) or cc["kappa"] > best_kappa[1]):
                    best_kappa = (tau, cc["kappa"])
            per = g.assign(flag=flag).groupby("sid").agg(ef=("expert_flag", "mean"), jf=("flag", "mean"))
            rho, p = spearmanr(per.jf, per.ef)
            res.append(dict(variant=k, group=gname, n_sentences=len(g), judge_flag_rate=flag.mean(), expert_flag_rate=e.mean(),
                            auroc_1_minus_pe=roc_auc_score(e, 1 - pe), auroc_pc=roc_auc_score(e, pc),
                            precision=c["precision"], recall=c["recall"], specificity=c["specificity"], f1=c["f1"], kappa=c["kappa"],
                            best_f1_tau=best_f1[0], best_f1=best_f1[1], best_kappa_tau=best_kappa[0], best_kappa=best_kappa[1],
                            summary_spearman=rho, summary_spearman_p=p))
    out = pd.DataFrame(res)
    out.to_csv(RESULTS / "calibration_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.group == "all"].round(3).to_string(index=False))
    print("\ngenerated only:\n", out[out.group == "generated"][["variant", "judge_flag_rate", "auroc_1_minus_pe", "precision", "recall", "kappa", "best_kappa_tau", "best_kappa", "summary_spearman"]].round(3).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
