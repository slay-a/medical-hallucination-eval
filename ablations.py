#!/usr/bin/env python3
"""
ablations.py — Offline robustness analyses for the claim-level NLI judge.

No API calls.  Uses the locally cached models (all-MiniLM-L6-v2 bi-encoder and
cross-encoder/nli-MiniLM2-L6-H768) and the stored generations in results/.

A. NLI decision-threshold sweep on stored probabilities      -> results/ablation_thresholds.csv
B. Evidence top-k ablation (k = 1, 3, 5), re-retrieve+re-label -> results/ablation_topk.csv
C. Cleaned-evidence ablation (MTSamples line-break repair)   -> results/ablation_cleaned_evidence.csv
                                                                results/claims_cleaned_evidence.csv
D. Coverage proxy (source sentences represented in summary)  -> results/coverage_per_sample.csv
                                                                results/coverage_summary.csv
E. Judge-error analysis on verbatim E2 sentences (negation)  -> results/e2_error_analysis.csv
F. Keyword-assisted error taxonomy of unsupported claims     -> results/error_taxonomy_counts.csv
                                                                results/error_taxonomy_examples.csv
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he                      # noqa: E402  (models, sentencize, load_samples)
from preprocessing import is_markdown_header, sentencize_cleaned  # noqa: E402
from recompute_metrics import per_sample_metrics, paired_stats, order_conditions  # noqa: E402

RESULTS = HERE / "results"
CSV_PATH = (HERE / he.INPUT_CSV).resolve()
GEN_CONDS = ["E0", "E1"]           # LLM-generated conditions whose claims are re-labelled
ALL_CONDS = ["E0", "E1", "E2"]


def decide(pe: float, pc: float, tau: float) -> str:
    if pe >= tau and pe > pc:
        return "Supported"
    if pc >= tau and pc > pe:
        return "Contradicted"
    return "Not-Supported"


def label_claims_against(claims, source_sents, k):
    """Retrieve top-k source sentences per claim (cosine) and NLI-label them in one batch."""
    if not claims:
        return []
    if not source_sents:
        return [("Not-Supported", 0.0, 0.0, [])] * len(claims)
    enc, ce = he.get_bi_encoder(), he.get_cross_encoder()
    S = enc.encode(source_sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    C = enc.encode(claims, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = C @ S.T
    kk = min(k, len(source_sents))
    top = np.argsort(-sims, axis=1)[:, :kk]
    pairs, owners = [], []
    for i in range(len(claims)):
        for j in top[i]:
            pairs.append([source_sents[j], claims[i]])
            owners.append(i)
    logits = np.atleast_2d(ce.predict(pairs, batch_size=64, show_progress_bar=False))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    owners = np.array(owners)
    out = []
    for i in range(len(claims)):
        p = probs[owners == i]
        pe, pc = float(p[:, he.IDX_ENTAILMENT].max()), float(p[:, he.IDX_CONTRADICTION].max())
        out.append((decide(pe, pc, 0.5), pe, pc, [source_sents[j] for j in top[i]]))
    return out


def extractive_topk(sentences, k=5):
    """Centroid-based extractive summary (same rule as e2_extractive_eval.py)."""
    if len(sentences) <= k:
        return list(sentences)
    enc = he.get_bi_encoder()
    E = enc.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sim = E @ E.T
    np.fill_diagonal(sim, 0.0)
    mean_sims = sim.sum(axis=1) / (len(sentences) - 1)
    idx = sorted(np.argsort(mean_sims)[::-1][:k])
    return [sentences[i] for i in idx]


def metrics_and_tests(records, conds):
    df = pd.DataFrame(records)
    ps = per_sample_metrics(df, sorted(df.doc_id.unique()), conds)
    wide = {m: ps.pivot(index="doc_id", columns="condition", values=m) for m in ("UFR", "CR")}
    return ps, wide


def main() -> None:
    t0 = time.time()
    claims = pd.read_csv(RESULTS / "claims_all.csv")
    if "is_header" not in claims.columns:
        claims["is_header"] = claims["claim"].astype(str).map(is_markdown_header)
    claims = claims[~claims.is_header].copy()
    samples = he.load_samples(str(CSV_PATH), n=he.N_SAMPLES)
    nlp = he.get_nlp()
    he.get_bi_encoder(); he.get_cross_encoder()

    # ═══════════════ A. threshold sweep (stored probabilities) ═══════════════
    rows = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        c2 = claims.copy()
        c2["label"] = [decide(pe, pc, tau) for pe, pc in zip(c2.p_entailment, c2.p_contradiction)]
        ps, wide = metrics_and_tests(c2[["doc_id", "condition", "label"]].to_dict("records"), ALL_CONDS)
        row = dict(tau=tau)
        for c in ALL_CONDS:
            for m in ("UFR", "CR"):
                row[f"{c}_{m}_mean"] = ps.loc[ps.condition == c, m].mean()
        for m in ("UFR", "CR"):
            st = paired_stats(wide[m]["E0"], wide[m]["E1"])
            row[f"p_E1_vs_E0_{m}"] = st["p"]; row[f"delta_E1_vs_E0_{m}"] = st["delta_mean"]
            st2 = paired_stats(wide[m]["E0"], wide[m]["E2"])
            row[f"p_E2_vs_E0_{m}"] = st2["p"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "ablation_thresholds.csv", index=False)
    print(f"[A] thresholds done ({time.time()-t0:.0f}s)")

    # ═══════════════ B + C + D: per-document work ═══════════════
    topk_records = {k: [] for k in (1, 3, 5)}
    agree_k3 = []
    clean_records, clean_claim_rows = [], []
    cov_rows = []
    NEG = re.compile(r"\b(no|not|denies|denied|deny|without|negative|never|none|nontender|non-tender|unremarkable|absent|absence)\b", re.I)

    for idx, row in samples.iterrows():
        doc_id = int(idx)
        source = str(row.get("transcription", "")).strip()
        orig_sents = he.sentencize(source)
        clean_sents = sentencize_cleaned(source, nlp)
        doc_claims = claims[claims.doc_id == doc_id]

        # ---- B: top-k on original sentences
        for k in (1, 3, 5):
            for c in ALL_CONDS:
                cl = doc_claims[doc_claims.condition == c]
                res = label_claims_against(cl.claim.tolist(), orig_sents, k)
                for (lab, pe, pc, ev), (_, orig) in zip(res, cl.iterrows()):
                    topk_records[k].append(dict(doc_id=doc_id, condition=c, label=lab))
                    if k == 3:
                        agree_k3.append(lab == orig.label)

        # ---- C: cleaned evidence (E0/E1 claims re-labelled; E2 regenerated from cleaned sentences)
        for c in GEN_CONDS:
            cl = doc_claims[doc_claims.condition == c]
            res = label_claims_against(cl.claim.tolist(), clean_sents, he.TOP_K_EVIDENCE)
            for (lab, pe, pc, ev), (_, orig) in zip(res, cl.iterrows()):
                clean_records.append(dict(doc_id=doc_id, condition=c, label=lab))
                clean_claim_rows.append(dict(doc_id=doc_id, condition=c, claim=orig.claim, evidence=" | ".join(ev),
                                             label=lab, p_entailment=round(pe, 4), p_contradiction=round(pc, 4),
                                             original_label=orig.label))
        e2c = extractive_topk(clean_sents, k=5)
        res = label_claims_against(e2c, clean_sents, he.TOP_K_EVIDENCE)
        for (lab, pe, pc, ev), sent in zip(res, e2c):
            clean_records.append(dict(doc_id=doc_id, condition="E2", label=lab))
            clean_claim_rows.append(dict(doc_id=doc_id, condition="E2", claim=sent, evidence=" | ".join(ev), label=lab,
                                         p_entailment=round(pe, 4), p_contradiction=round(pc, 4), original_label=""))

        # ---- D: coverage proxy on original sentences
        enc = he.get_bi_encoder()
        if orig_sents:
            S = enc.encode(orig_sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            for c in ALL_CONDS:
                cl = doc_claims[doc_claims.condition == c].claim.tolist()
                if not cl:
                    continue
                Cm = enc.encode(cl, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                best = (S @ Cm.T).max(axis=1)
                cov_rows.append(dict(doc_id=doc_id, condition=c, n_source_sents=len(orig_sents), n_claims=len(cl),
                                     coverage_at_0_5=float((best >= 0.5).mean()), coverage_at_0_6=float((best >= 0.6).mean()),
                                     coverage_at_0_7=float((best >= 0.7).mean()), mean_best_similarity=float(best.mean())))
        if doc_id % 10 == 9:
            print(f"   processed {doc_id+1}/50 documents ({time.time()-t0:.0f}s)")

    # ---- B outputs
    rows = []
    for k in (1, 3, 5):
        ps, wide = metrics_and_tests(topk_records[k], ALL_CONDS)
        r = dict(k=k)
        for c in ALL_CONDS:
            for m in ("UFR", "CR"):
                r[f"{c}_{m}_mean"] = ps.loc[ps.condition == c, m].mean()
                r[f"{c}_{m}_median"] = ps.loc[ps.condition == c, m].median()
        for m in ("UFR", "CR"):
            st = paired_stats(wide[m]["E0"], wide[m]["E1"]); r[f"p_E1_vs_E0_{m}"] = st["p"]; r[f"delta_E1_vs_E0_{m}"] = st["delta_mean"]
            st = paired_stats(wide[m]["E0"], wide[m]["E2"]); r[f"p_E2_vs_E0_{m}"] = st["p"]
        r["label_agreement_with_stored"] = float(np.mean(agree_k3)) if k == 3 else np.nan
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "ablation_topk.csv", index=False)
    print(f"[B] top-k done; k=3 agreement with stored labels = {np.mean(agree_k3):.3f}")

    # ---- C outputs
    pd.DataFrame(clean_claim_rows).to_csv(RESULTS / "claims_cleaned_evidence.csv", index=False)
    ps, wide = metrics_and_tests(clean_records, ALL_CONDS)
    rows = []
    for c in ALL_CONDS:
        sub = ps[ps.condition == c]
        rows.append(dict(condition=c, UFR_mean=sub.UFR.mean(), UFR_median=sub.UFR.median(), CR_mean=sub.CR.mean(),
                         CR_median=sub.CR.median(), n_claims=int(sub.n_claims.sum()),
                         n_contradicted=int(sub.n_contradicted.sum()), n_not_supported=int(sub.n_not_supported.sum())))
    df_c = pd.DataFrame(rows)
    for m in ("UFR", "CR"):
        st = paired_stats(wide[m]["E0"], wide[m]["E1"]); df_c[f"p_E1_vs_E0_{m}"] = st["p"]; df_c[f"delta_E1_vs_E0_{m}"] = st["delta_mean"]
        st = paired_stats(wide[m]["E0"], wide[m]["E2"]); df_c[f"p_E2_vs_E0_{m}"] = st["p"]
    df_c.to_csv(RESULTS / "ablation_cleaned_evidence.csv", index=False)
    print("[C] cleaned-evidence done")

    # ---- D outputs
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(RESULTS / "coverage_per_sample.csv", index=False)
    summ = cov.groupby("condition")[["coverage_at_0_5", "coverage_at_0_6", "coverage_at_0_7", "mean_best_similarity", "n_claims"]].mean().reset_index()
    w = cov.pivot(index="doc_id", columns="condition", values="coverage_at_0_6")
    for a, b in (("E0", "E1"), ("E0", "E2"), ("E1", "E2")):
        st = paired_stats(w[a], w[b])
        summ[f"p_cov06_{b}_vs_{a}"] = st["p"]
    summ.to_csv(RESULTS / "coverage_summary.csv", index=False)
    print("[D] coverage done")

    # ═══════════════ E. negation analysis ═══════════════
    all_claims = pd.read_csv(RESULTS / "claims_all.csv")
    all_claims = all_claims[~all_claims.is_header].copy()
    all_claims["has_negation"] = all_claims.claim.astype(str).str.contains(NEG)
    ct = all_claims.groupby(["condition", "has_negation", "label"]).size().reset_index(name="count")
    e2 = all_claims[all_claims.condition == "E2"]
    e2c = e2[e2.label == "Contradicted"]
    extra = pd.DataFrame([
        dict(condition="E2", has_negation="", label="Contradicted_mean_p_entail", count=round(e2c.p_entailment.mean(), 4)),
        dict(condition="E2", has_negation="", label="Contradicted_mean_p_contra", count=round(e2c.p_contradiction.mean(), 4)),
        dict(condition="E2", has_negation="", label="Contradicted_both_above_0.9", count=int(((e2c.p_entailment > 0.9) & (e2c.p_contradiction > 0.9)).sum())),
        dict(condition="E2", has_negation="", label="Contradicted_with_negation", count=int(e2c.has_negation.sum())),
        dict(condition="E2", has_negation="", label="Contradicted_total", count=len(e2c)),
    ])
    pd.concat([ct, extra], ignore_index=True).to_csv(RESULTS / "e2_error_analysis.csv", index=False)
    print("[E] negation analysis done")

    # ═══════════════ F. keyword-assisted taxonomy ═══════════════
    TAXONOMY = [
        ("Medication or dosage", r"\b(medication|medicine|medicines|drug|mg\b|dose|dosage|prescri\w*|tablet|pill|antibiotic|insulin|inhaler|aspirin|ibuprofen|acetaminophen|statin|taking|take (it|them|your))\b"),
        ("Follow-up or scheduling", r"\b(follow[- ]?up|appointment|schedule|reschedul\w*|return (to|in|for)|come back|see (your|the|a) (doctor|provider|physician|specialist)|contact (your|the|a)|call (your|the|911)|reach out|referr\w*|visit)\b"),
        ("Generic advice or patient education", r"\b(important to|make sure|be sure|remember to|monitor|keep track|stay (hydrated|active)|healthy|lifestyle|get (plenty of )?rest|avoid|maintain|manage your|take care|it is (important|essential|recommended)|please|encourag\w*|seek (medical|immediate))\b"),
        ("Diagnosis or assessment", r"\b(diagnos\w*|assessment|indicat\w*|consistent with|suggest\w*|likely|impression|suspected|rule out|condition)\b"),
        ("Procedure or treatment", r"\b(procedure|surgery|surgical|performed|underwent|operation|treated|treatment|therapy|biopsy|transfusion|intubat\w*|catheter|repair|removal|injection)\b"),
        ("Findings, exam, labs or imaging", r"\b(exam\w*|reveal\w*|showed|shows|found|finding|level|levels|blood pressure|heart rate|temperature|lab|labs|test\w*|x-ray|ct|mri|ultrasound|echocardiogram|normal|abnormal|elevated|swelling|tender\w*)\b"),
        ("History, symptoms or timeline", r"\b(history|symptom\w*|complain\w*|report\w*|pain|ago|years?|months?|weeks?|days?|since|previously|prior|started|began|ongoing|experienc\w*)\b"),
    ]
    tax = [(n, re.compile(p, re.I)) for n, p in TAXONOMY]

    def categorize(text: str) -> str:
        for name, rx in tax:
            if rx.search(text):
                return name
        return "Other or unclassified"

    uns = all_claims[(all_claims.condition.isin(GEN_CONDS)) & (all_claims.label != "Supported")].copy()
    uns["category"] = uns.claim.astype(str).map(categorize)
    counts = uns.groupby(["condition", "category", "label"]).size().reset_index(name="count")
    tot = uns.groupby(["condition", "category"]).size().reset_index(name="count").assign(label="All unsupported")
    pd.concat([counts, tot], ignore_index=True).to_csv(RESULTS / "error_taxonomy_counts.csv", index=False)
    ex = (uns.assign(L=uns.claim.str.len()).query("40 <= L <= 170")
             .sort_values(["condition", "category", "doc_id"]).groupby(["condition", "category"]).head(3))
    ex[["condition", "category", "doc_id", "label", "claim", "p_entailment", "p_contradiction"]].to_csv(RESULTS / "error_taxonomy_examples.csv", index=False)
    print("[F] taxonomy done")
    print(uns.groupby(["condition", "category"]).size().unstack(0).fillna(0).astype(int))
    print(f"\nAll ablations complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
