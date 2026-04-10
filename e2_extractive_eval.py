#!/usr/bin/env python3
"""
e2_extractive_eval.py — Add E2 (Extractive baseline) to the evaluation pipeline.

E2: centroid-based extractive summarisation using all-MiniLM-L6-v2.
  - Embed every sentence in the source document.
  - Rank each sentence by its mean cosine similarity to every other sentence
    (centroid / most-representative criterion — no query needed).
  - Take the top-5 sentences (in original order) as the "summary".
  - No LLM call, no generation.

The same NLI pipeline used for E0/E1 is then applied:
  claim segmentation (spaCy) → evidence retrieval (cos-sim) → NLI label
  → UFR / CR aggregation.

Results are written into the existing CSVs:
  results/summaries.csv           — E2 columns appended
  results/claims_all.csv          — E2 claim rows appended
  results/comparison_per_sample.csv — E2 columns added
  results/aggregate_statistics.csv  — rebuilt with E0/E1/E2

Box plots (plot_ufr_boxplot.png, plot_cr_boxplot.png) are regenerated with
three boxes.

A 3-way comparison table (E0 vs E1 vs E2) is printed to stdout.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuration (must match hallucination_eval.py) ────────────────────────
SCRIPT_DIR   = Path(__file__).parent
INPUT_CSV    = (SCRIPT_DIR / "../mtsamples.csv").resolve()
RESULTS_DIR  = SCRIPT_DIR / "results"
N_SAMPLES    = 50
RANDOM_SEED  = 42
TARGET_TYPES = ["Discharge Summary", "Consult - History and Phy."]

TOP_K_EVIDENCE       = 3
TOP_K_EXTRACTIVE     = 5    # sentences to keep per document

IDX_CONTRADICTION    = 0
IDX_ENTAILMENT       = 1
IDX_NEUTRAL          = 2
ENTAILMENT_THRESHOLD    = 0.5
CONTRADICTION_THRESHOLD = 0.5

# ─── Lazy model loading ───────────────────────────────────────────────────────
_nlp           = None
_bi_encoder    = None
_cross_encoder = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        log.info("Loading spaCy en_core_web_sm …")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading bi-encoder all-MiniLM-L6-v2 …")
        _bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _bi_encoder


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder/nli-MiniLM2-L6-H768 …")
        _cross_encoder = CrossEncoder(
            "cross-encoder/nli-MiniLM2-L6-H768", num_labels=3
        )
    return _cross_encoder


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_samples() -> pd.DataFrame:
    """Identical filtering + sampling logic as hallucination_eval.py."""
    df = pd.read_csv(INPUT_CSV)
    df.columns = [c.strip() for c in df.columns]
    mask = df["medical_specialty"].str.strip().isin(TARGET_TYPES)
    filtered = (
        df[mask]
        .dropna(subset=["transcription"])
        .pipe(lambda d: d[d["transcription"].str.strip().ne("")])
        .reset_index(drop=True)
    )
    sampled = filtered.sample(
        n=min(N_SAMPLES, len(filtered)), random_state=RANDOM_SEED
    ).reset_index(drop=True)
    log.info(f"Loaded {len(sampled)} samples (seed={RANDOM_SEED})")
    return sampled


# ─── Text utilities ───────────────────────────────────────────────────────────
def sentencize(text: str, min_len: int = 10) -> List[str]:
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]


# ─── Centroid-based extractive summarisation ──────────────────────────────────
def extractive_summary(source_text: str, k: int = TOP_K_EXTRACTIVE) -> str:
    """
    Embed all source sentences; rank each by mean cosine similarity to every
    other sentence (centroid proxy).  Return the top-k sentences in original
    document order joined as a string.
    """
    sentences = sentencize(source_text)
    if not sentences:
        return ""
    if len(sentences) <= k:
        return " ".join(sentences)

    enc = get_bi_encoder()
    # normalize_embeddings=True → dot product == cosine similarity
    embeddings = enc.encode(
        sentences, convert_to_numpy=True, normalize_embeddings=True
    )

    # Full cosine similarity matrix via dot product
    sim_matrix = embeddings @ embeddings.T          # (n, n)
    np.fill_diagonal(sim_matrix, 0.0)               # exclude self-similarity
    n = len(sentences)
    mean_sims = sim_matrix.sum(axis=1) / (n - 1)   # mean sim to all others

    top_idx = np.argsort(mean_sims)[::-1][:k]
    top_idx_ordered = sorted(top_idx)               # preserve narrative order
    return " ".join(sentences[i] for i in top_idx_ordered)


# ─── Evidence retrieval ───────────────────────────────────────────────────────
def retrieve_top_k(
    query: str, candidates: List[str], k: int
) -> List[Tuple[str, float]]:
    if not candidates:
        return []
    from sentence_transformers import util as st_util
    enc = get_bi_encoder()
    q_emb  = enc.encode(query, convert_to_tensor=True)
    c_embs = enc.encode(candidates, convert_to_tensor=True)
    scores = st_util.cos_sim(q_emb, c_embs)[0].cpu().numpy()
    top_idx = np.argsort(scores)[::-1][: min(k, len(candidates))]
    return [(candidates[i], float(scores[i])) for i in top_idx]


# ─── NLI labelling ────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def label_claim(
    claim: str, evidence: List[str]
) -> Tuple[str, float, float]:
    if not evidence:
        return "Not-Supported", 0.0, 0.0
    ce     = get_cross_encoder()
    pairs  = [[ev, claim] for ev in evidence]
    logits = ce.predict(pairs)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    probs          = np.array([softmax(row) for row in logits])
    best_entail    = float(probs[:, IDX_ENTAILMENT].max())
    best_contradict = float(probs[:, IDX_CONTRADICTION].max())

    if best_entail >= ENTAILMENT_THRESHOLD and best_entail > best_contradict:
        label = "Supported"
    elif best_contradict >= CONTRADICTION_THRESHOLD and best_contradict > best_entail:
        label = "Contradicted"
    else:
        label = "Not-Supported"
    return label, best_entail, best_contradict


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(labels: List[str]) -> Dict:
    n = len(labels)
    if n == 0:
        return dict(UFR=0.0, CR=0.0, n_claims=0,
                    n_supported=0, n_contradicted=0, n_not_supported=0)
    ns  = labels.count("Supported")
    nc  = labels.count("Contradicted")
    nns = labels.count("Not-Supported")
    return dict(
        UFR=(nc + nns) / n, CR=nc / n, n_claims=n,
        n_supported=ns, n_contradicted=nc, n_not_supported=nns,
    )


# ─── NLI evaluation pipeline ─────────────────────────────────────────────────
def evaluate_summary(
    summary: str, source_text: str
) -> Tuple[List[Dict], Dict]:
    source_sents = sentencize(source_text)
    claims       = sentencize(summary)
    records, labels = [], []

    for claim in claims:
        ev_with_scores = retrieve_top_k(claim, source_sents, k=TOP_K_EVIDENCE)
        evidence  = [ev for ev, _ in ev_with_scores]
        ev_scores = [s  for _, s  in ev_with_scores]
        label, p_entail, p_contra = label_claim(claim, evidence)
        labels.append(label)
        records.append(dict(
            claim           = claim,
            evidence        = " | ".join(evidence),
            evidence_scores = str([round(s, 4) for s in ev_scores]),
            label           = label,
            p_entailment    = round(p_entail,  4),
            p_contradiction = round(p_contra,  4),
        ))

    return records, compute_metrics(labels)


# ─── CSV update helpers ───────────────────────────────────────────────────────
def update_summaries_csv(e2_rows: List[Dict]) -> pd.DataFrame:
    path   = RESULTS_DIR / "summaries.csv"
    df     = pd.read_csv(path)
    e2_df  = pd.DataFrame(e2_rows)

    # Drop any stale E2 columns from a previous run
    e2_cols = [c for c in df.columns if c.startswith("e2_") or c.startswith("E2_")]
    df = df.drop(columns=e2_cols, errors="ignore")

    df = df.merge(e2_df, on="doc_id", how="left")
    df.to_csv(path, index=False)
    log.info(f"  Updated {path}")
    return df


def update_claims_csv(e2_claim_rows: List[Dict]) -> None:
    path       = RESULTS_DIR / "claims_all.csv"
    df_exist   = pd.read_csv(path)
    df_exist   = df_exist[df_exist["condition"] != "E2"]     # remove stale E2 rows
    df_e2      = pd.DataFrame(e2_claim_rows)
    df_combined = pd.concat([df_exist, df_e2], ignore_index=True)
    df_combined.to_csv(path, index=False)
    log.info(f"  Updated {path} (+{len(df_e2)} E2 claim rows)")


def build_comparison_csv(df_summary: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS_DIR / "comparison_per_sample.csv"
    want = [
        "doc_id", "specialty", "description", "source_word_count",
        "e0_summary_words",
        "E0_UFR", "E0_CR", "E0_n_claims",
        "E0_n_supported", "E0_n_contradicted", "E0_n_not_supported",
        "e1_summary_words",
        "E1_UFR", "E1_CR", "E1_n_claims",
        "E1_n_supported", "E1_n_contradicted", "E1_n_not_supported",
        "e2_summary_words",
        "E2_UFR", "E2_CR", "E2_n_claims",
        "E2_n_supported", "E2_n_contradicted", "E2_n_not_supported",
    ]
    df = df_summary[[c for c in want if c in df_summary.columns]].copy()
    if {"E0_UFR", "E1_UFR"} <= set(df.columns):
        df["delta_UFR"]    = df["E1_UFR"] - df["E0_UFR"]
        df["delta_CR"]     = df["E1_CR"]  - df["E0_CR"]
    if {"E0_UFR", "E2_UFR"} <= set(df.columns):
        df["E2_delta_UFR"] = df["E2_UFR"] - df["E0_UFR"]
        df["E2_delta_CR"]  = df["E2_CR"]  - df["E0_CR"]
    df.to_csv(path, index=False)
    log.info(f"  Updated {path}")
    return df


def build_aggregate_stats(df_summary: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS_DIR / "aggregate_statistics.csv"
    rows = []
    for metric in ("UFR", "CR"):
        e0c, e1c, e2c = f"E0_{metric}", f"E1_{metric}", f"E2_{metric}"
        if not {e0c, e1c, e2c} <= set(df_summary.columns):
            continue
        e0 = df_summary[e0c].dropna()
        e1 = df_summary[e1c].dropna()
        e2 = df_summary[e2c].dropna()
        d10 = e1 - e0    # E1 minus E0
        d20 = e2 - e0    # E2 minus E0
        rows.append(dict(
            Metric          = metric,
            E0_Mean         = round(e0.mean(),    4),
            E0_Std          = round(e0.std(),     4),
            E0_Median       = round(e0.median(),  4),
            E1_Mean         = round(e1.mean(),    4),
            E1_Std          = round(e1.std(),     4),
            E1_Median       = round(e1.median(),  4),
            E2_Mean         = round(e2.mean(),    4),
            E2_Std          = round(e2.std(),     4),
            E2_Median       = round(e2.median(),  4),
            Delta_Mean      = round(d10.mean(),   4),   # E1 − E0
            Delta_Std       = round(d10.std(),    4),
            Pct_Improved    = round((d10 < 0).mean() * 100, 1),
            E2_Delta_Mean   = round(d20.mean(),   4),   # E2 − E0
            E2_Delta_Std    = round(d20.std(),    4),
            E2_Pct_Improved = round((d20 < 0).mean() * 100, 1),
        ))
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(path, index=False)
    log.info(f"  Updated {path}")
    return stats_df


# ─── Box plots ────────────────────────────────────────────────────────────────
def regenerate_boxplots(df_cmp: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_E0 = "#E07B54"    # warm orange  — baseline
    C_E1 = "#4C8BB5"    # steel blue   — RAG
    C_E2 = "#5DBF6E"    # leaf green   — extractive
    GREY = "#888888"

    def make_boxplot(col_e0, col_e1, col_e2, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        data   = [df_cmp[col_e0].values,
                  df_cmp[col_e1].values,
                  df_cmp[col_e2].values]
        labels = ["E0  Baseline", "E1  RAG", "E2  Extractive"]
        colors = [C_E0, C_E1, C_E2]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.42,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            flierprops=dict(marker="o", markerfacecolor=GREY,
                            markersize=4, linestyle="none"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        rng = np.random.default_rng(42)
        for i, (d, color) in enumerate(zip(data, colors), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(d))
            ax.scatter(
                np.full(len(d), i) + jitter, d,
                color=color, alpha=0.45, s=18, zorder=3,
            )

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()

        out = RESULTS_DIR / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"  Saved {out}")
        plt.close(fig)

    make_boxplot(
        "E0_UFR", "E1_UFR", "E2_UFR",
        "Unsupported Fact Rate",
        "UFR Distribution — Baseline vs RAG vs Extractive",
        "plot_ufr_boxplot.png",
    )
    make_boxplot(
        "E0_CR", "E1_CR", "E2_CR",
        "Contradiction Rate",
        "CR Distribution — Baseline vs RAG vs Extractive",
        "plot_cr_boxplot.png",
    )


# ─── 3-way comparison table ───────────────────────────────────────────────────
def print_3way_table(stats: pd.DataFrame, n_samples: int) -> None:
    sep = "=" * 82
    print(f"\n{sep}")
    print("  3-WAY COMPARISON — E0 Baseline  ·  E1 RAG  ·  E2 Extractive")
    print(sep)
    print(f"  Samples evaluated : {n_samples}\n")

    hdr = (
        f"{'Metric':<8} "
        f"{'E0 Mean':>9} {'E1 Mean':>9} {'E2 Mean':>9}  "
        f"{'E1−E0 Δ':>9} {'E2−E0 Δ':>9}  "
        f"{'E1 %↓':>7} {'E2 %↓':>7}"
    )
    print(hdr)
    print("─" * len(hdr))
    for _, r in stats.iterrows():
        print(
            f"{r['Metric']:<8} "
            f"{r['E0_Mean']:>9.4f} {r['E1_Mean']:>9.4f} {r['E2_Mean']:>9.4f}  "
            f"{r['Delta_Mean']:>+9.4f} {r['E2_Delta_Mean']:>+9.4f}  "
            f"{r['Pct_Improved']:>6.1f}% {r['E2_Pct_Improved']:>6.1f}%"
        )
    print(
        "\n  Δ = condition − E0  (negative = fewer hallucinations than baseline)"
    )
    print("  %↓ = % of samples where condition improved over E0")
    print(
        "\n  UFR  Unsupported Fact Rate = (Contradicted + Not-Supported) / total claims"
    )
    print("  CR   Contradiction Rate    = Contradicted / total claims")
    print(f"\n{sep}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("─" * 60)
    log.info("E2 Extractive Baseline Evaluation")
    log.info(f"  Input  : {INPUT_CSV}")
    log.info(f"  Output : {RESULTS_DIR}")
    log.info("─" * 60)

    samples = load_samples()

    # Warm up models once
    get_nlp()
    get_bi_encoder()
    get_cross_encoder()

    e2_summary_rows: List[Dict] = []
    e2_claim_rows:   List[Dict] = []

    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="E2"):
        doc_id      = int(idx)
        source_text = str(row.get("transcription", "")).strip()
        specialty   = str(row.get("medical_specialty", "")).strip()
        description = str(row.get("description", "")).strip()

        if not source_text:
            log.warning(f"doc_id={doc_id} has empty transcription — skipping.")
            continue

        # Build extractive summary (no LLM)
        summary = extractive_summary(source_text, k=TOP_K_EXTRACTIVE)

        # NLI evaluation
        claim_records, metrics = evaluate_summary(summary, source_text)

        for c in claim_records:
            e2_claim_rows.append(dict(
                doc_id      = doc_id,
                condition   = "E2",
                specialty   = specialty,
                description = description,
                **c,
            ))

        e2_summary_rows.append(dict(
            doc_id             = doc_id,
            e2_summary         = summary,
            e2_summary_words   = len(summary.split()),
            E2_UFR             = metrics["UFR"],
            E2_CR              = metrics["CR"],
            E2_n_claims        = metrics["n_claims"],
            E2_n_supported     = metrics["n_supported"],
            E2_n_contradicted  = metrics["n_contradicted"],
            E2_n_not_supported = metrics["n_not_supported"],
        ))

    # ── Persist results ────────────────────────────────────────────────────────
    log.info("Updating output files …")
    df_summary = update_summaries_csv(e2_summary_rows)
    update_claims_csv(e2_claim_rows)
    df_cmp     = build_comparison_csv(df_summary)
    stats_df   = build_aggregate_stats(df_summary)

    # ── Plots ──────────────────────────────────────────────────────────────────
    log.info("Regenerating box plots …")
    regenerate_boxplots(df_cmp)

    # ── Print table ────────────────────────────────────────────────────────────
    print_3way_table(stats_df, len(df_summary))

    log.info("E2 evaluation complete.")


if __name__ == "__main__":
    main()
