#!/usr/bin/env python3
"""
Hallucination Evaluation in LLM-Based Medical Summarization
============================================================
Compares:
  E0 — Baseline GPT-4o-mini (no retrieval)
  E1 — RAG-augmented GPT-4o-mini (top-3 chunks retrieved)

Metrics per summary:
  UFR  — Unsupported Fact Rate  = (Not-Supported + Contradicted) / total_claims
  CR   — Contradiction Rate     = Contradicted / total_claims

Pipeline per sample:
  1. Generate summary (E0 or E1)
  2. Sentencize summary into claims  (spaCy)
  3. For each claim: retrieve top-3 source sentences  (all-MiniLM-L6-v2 cosine sim)
  4. NLI-label each (claim, evidence) pair  (cross-encoder/nli-MiniLM2-L6-H768)
  5. Aggregate UFR / CR
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
OPENAI_MODEL: str = "gpt-4o-mini"
INPUT_CSV: str = "../mtsamples.csv"       # relative to script location
N_SAMPLES: int = 50
RANDOM_SEED: int = 42

TOP_K_EVIDENCE: int = 3   # source sentences retrieved per claim
TOP_K_CHUNKS: int = 3     # RAG chunks retrieved per summary
CHUNK_SIZE: int = 5        # sentences per RAG chunk

# Target medical_specialty values
TARGET_TYPES = ["Discharge Summary", "Consult - History and Phy."]

# NLI label indices for cross-encoder/nli-MiniLM2-L6-H768
# Model trained on MNLI → outputs [contradiction, entailment, neutral]
IDX_CONTRADICTION = 0
IDX_ENTAILMENT    = 1
IDX_NEUTRAL       = 2
ENTAILMENT_THRESHOLD    = 0.5   # min softmax prob to call Supported
CONTRADICTION_THRESHOLD = 0.5   # min softmax prob to call Contradicted

# GPT generation params
GPT_TEMPERATURE = 0.3
GPT_MAX_TOKENS  = 450
GPT_TIMEOUT_S   = 60
GPT_RETRIES     = 3

# ──────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADING  (avoids slow imports at module level)
# ──────────────────────────────────────────────────────────────────────────────
_nlp            = None
_bi_encoder     = None
_cross_encoder  = None
_openai_client  = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        log.info("Loading spaCy model en_core_web_sm …")
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
            "cross-encoder/nli-MiniLM2-L6-H768",
            num_labels=3,
        )
    return _cross_encoder


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=API_KEY, timeout=GPT_TIMEOUT_S)
    return _openai_client


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_samples(csv_path: str, n: int = N_SAMPLES) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} total rows from {csv_path}")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Filter specialty
    mask = df["medical_specialty"].str.strip().isin(TARGET_TYPES)
    filtered = df[mask].dropna(subset=["transcription"])
    filtered = filtered[filtered["transcription"].str.strip().ne("")].reset_index(drop=True)
    log.info(f"  {len(filtered)} rows after filtering for target specialties")

    if len(filtered) == 0:
        raise ValueError(
            f"No rows matched specialties {TARGET_TYPES}. "
            f"Available: {df['medical_specialty'].unique()[:10]}"
        )

    sampled = filtered.sample(n=min(n, len(filtered)), random_state=RANDOM_SEED).reset_index(drop=True)
    log.info(f"  Using {len(sampled)} samples (seed={RANDOM_SEED})")
    return sampled


# ──────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def sentencize(text: str, min_len: int = 10) -> List[str]:
    """Return non-trivial sentences from text using spaCy."""
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= min_len]


def chunk_sentences(sentences: List[str], size: int = CHUNK_SIZE) -> List[str]:
    """Group consecutive sentences into fixed-size text chunks."""
    return [
        " ".join(sentences[i : i + size])
        for i in range(0, len(sentences), size)
        if sentences[i : i + size]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL  (bi-encoder cosine similarity)
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_top_k(
    query: str,
    candidates: List[str],
    k: int,
) -> List[Tuple[str, float]]:
    """Return the top-k candidates most similar to query, with scores."""
    if not candidates:
        return []
    from sentence_transformers import util as st_util
    enc = get_bi_encoder()
    q_emb = enc.encode(query, convert_to_tensor=True)
    c_embs = enc.encode(candidates, convert_to_tensor=True)
    scores = st_util.cos_sim(q_emb, c_embs)[0].cpu().numpy()
    top_idx = np.argsort(scores)[::-1][: min(k, len(candidates))]
    return [(candidates[i], float(scores[i])) for i in top_idx]


# ──────────────────────────────────────────────────────────────────────────────
# NLI LABELLING
# ──────────────────────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def label_claim(claim: str, evidence: List[str]) -> Tuple[str, float, float]:
    """
    Score claim against each evidence sentence; take the max entailment /
    contradiction probability across all evidence items.

    Returns:
        label       — 'Supported' | 'Contradicted' | 'Not-Supported'
        p_entail    — best entailment probability
        p_contradict — best contradiction probability
    """
    if not evidence:
        return "Not-Supported", 0.0, 0.0

    ce = get_cross_encoder()
    pairs = [[ev, claim] for ev in evidence]
    logits = ce.predict(pairs)                # shape (n_evidence, 3)

    # Ensure 2-D even for a single evidence sentence
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]

    probs = np.array([softmax(row) for row in logits])  # (n, 3)

    best_entail    = float(probs[:, IDX_ENTAILMENT].max())
    best_contradict = float(probs[:, IDX_CONTRADICTION].max())

    if best_entail >= ENTAILMENT_THRESHOLD and best_entail > best_contradict:
        label = "Supported"
    elif best_contradict >= CONTRADICTION_THRESHOLD and best_contradict > best_entail:
        label = "Contradicted"
    else:
        label = "Not-Supported"

    return label, best_entail, best_contradict


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(labels: List[str]) -> Dict[str, float]:
    n = len(labels)
    if n == 0:
        return dict(UFR=0.0, CR=0.0, n_claims=0,
                    n_supported=0, n_contradicted=0, n_not_supported=0)
    ns  = labels.count("Supported")
    nc  = labels.count("Contradicted")
    nns = labels.count("Not-Supported")
    return dict(
        UFR=(nc + nns) / n,
        CR=nc / n,
        n_claims=n,
        n_supported=ns,
        n_contradicted=nc,
        n_not_supported=nns,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GPT CALLS
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_SYSTEM = (
    "You are a medical scribe assistant. "
    "Produce patient-facing summaries that are accurate and grounded "
    "only in the provided clinical note. Do not add or invent information."
)

BASELINE_USER = """\
Write a patient-facing summary (150-250 words) of the following clinical note.
Structure: (1) Reason for visit / diagnosis, (2) Key findings, \
(3) Treatment or procedures performed, (4) Follow-up instructions.
Avoid medical jargon where possible.

=== CLINICAL NOTE ===
{text}
=== END ===

Patient-Facing Summary:"""

RAG_SYSTEM = (
    "You are a medical scribe assistant. "
    "Use ONLY the supplied excerpts to write the summary. "
    "Do NOT add any information not present in the excerpts."
)

RAG_USER = """\
Using ONLY the excerpts below, write a patient-facing summary (150-250 words).
Structure: (1) Reason for visit / diagnosis, (2) Key findings, \
(3) Treatment or procedures performed, (4) Follow-up instructions.
Avoid medical jargon where possible.

=== RETRIEVED EXCERPTS ===
{context}
=== END ===

Patient-Facing Summary:"""


def call_gpt(system: str, user: str) -> str:
    client = get_openai_client()
    for attempt in range(GPT_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=GPT_TEMPERATURE,
                max_tokens=GPT_MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 2 ** attempt
            log.warning(f"GPT call failed (attempt {attempt+1}/{GPT_RETRIES}): {exc} — retry in {wait}s")
            if attempt < GPT_RETRIES - 1:
                time.sleep(wait)
    log.error("GPT call failed after all retries; returning empty string.")
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION PIPELINE  (shared by E0 and E1)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_summary(
    summary: str,
    source_text: str,
) -> Tuple[List[Dict], Dict]:
    """
    Split summary into claims, find evidence from source, NLI-label each claim.

    Returns:
        claim_records — list of dicts with claim / evidence / label / scores
        metrics       — UFR, CR, counts
    """
    source_sents = sentencize(source_text)
    claims       = sentencize(summary)

    records = []
    labels  = []

    for claim in claims:
        evidence_with_scores = retrieve_top_k(claim, source_sents, k=TOP_K_EVIDENCE)
        evidence = [ev for ev, _ in evidence_with_scores]
        ev_scores = [s for _, s in evidence_with_scores]

        label, p_entail, p_contradict = label_claim(claim, evidence)
        labels.append(label)

        records.append(dict(
            claim            = claim,
            evidence         = " | ".join(evidence),
            evidence_scores  = str([round(s, 4) for s in ev_scores]),
            label            = label,
            p_entailment     = round(p_entail, 4),
            p_contradiction  = round(p_contradict, 4),
        ))

    return records, compute_metrics(labels)


# ──────────────────────────────────────────────────────────────────────────────
# RAG RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────
def build_rag_context(source_text: str, description: str) -> str:
    """
    Chunk source doc, retrieve top-K chunks most relevant to the document
    description + first 200 chars of the transcription (simulates the query
    a summariser would make before writing).
    """
    sents  = sentencize(source_text)
    chunks = chunk_sentences(sents, size=CHUNK_SIZE)

    if not chunks:
        return source_text[:3500]

    # Query = description + opening lines of note (topic signal)
    query = f"{description}. {source_text[:300]}"
    top   = retrieve_top_k(query, chunks, k=TOP_K_CHUNKS)

    # Return as numbered excerpts so the model sees clear boundaries
    parts = [f"[Excerpt {i+1}]\n{text}" for i, (text, _) in enumerate(top)]
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Resolve paths ─────────────────────────────────────────────────────────
    script_dir  = Path(__file__).parent
    csv_path    = (script_dir / INPUT_CSV).resolve()
    output_dir  = script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Input  CSV : {csv_path}")
    log.info(f"Output dir : {output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    samples = load_samples(str(csv_path), n=N_SAMPLES)

    # ── Pre-load all models once ───────────────────────────────────────────────
    get_nlp()
    get_bi_encoder()
    get_cross_encoder()
    # (OpenAI client is lazy but fine to warm up)
    get_openai_client()

    # ── Result accumulators ───────────────────────────────────────────────────
    summary_rows: List[Dict] = []
    claim_rows:   List[Dict] = []

    # ── Per-sample loop ───────────────────────────────────────────────────────
    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="Samples"):
        doc_id      = int(idx)
        source_text = str(row.get("transcription", "")).strip()
        specialty   = str(row.get("medical_specialty", "")).strip()
        description = str(row.get("description", "")).strip()

        if not source_text:
            log.warning(f"Sample {doc_id} has empty transcription — skipping.")
            continue

        log.info(f"[{idx+1}/{len(samples)}] {specialty} | {description[:70]}")

        # ──────────────────────────────────────────────────────────────────────
        # E0: BASELINE SUMMARY
        # ──────────────────────────────────────────────────────────────────────
        e0_summary = call_gpt(
            system=BASELINE_SYSTEM,
            user=BASELINE_USER.format(text=source_text[:4500]),
        )
        e0_claims, e0_metrics = evaluate_summary(e0_summary, source_text)

        for c in e0_claims:
            claim_rows.append(dict(
                doc_id      = doc_id,
                condition   = "E0",
                specialty   = specialty,
                description = description,
                **c,
            ))

        # ──────────────────────────────────────────────────────────────────────
        # E1: RAG SUMMARY
        # ──────────────────────────────────────────────────────────────────────
        rag_context = build_rag_context(source_text, description)
        e1_summary  = call_gpt(
            system=RAG_SYSTEM,
            user=RAG_USER.format(context=rag_context[:4000]),
        )
        e1_claims, e1_metrics = evaluate_summary(e1_summary, source_text)

        for c in e1_claims:
            claim_rows.append(dict(
                doc_id      = doc_id,
                condition   = "E1",
                specialty   = specialty,
                description = description,
                **c,
            ))

        # ──────────────────────────────────────────────────────────────────────
        # Collate summary-level row
        # ──────────────────────────────────────────────────────────────────────
        summary_rows.append(dict(
            doc_id             = doc_id,
            specialty          = specialty,
            description        = description,
            source_word_count  = len(source_text.split()),
            # E0
            e0_summary         = e0_summary,
            e0_summary_words   = len(e0_summary.split()),
            E0_UFR             = e0_metrics["UFR"],
            E0_CR              = e0_metrics["CR"],
            E0_n_claims        = e0_metrics["n_claims"],
            E0_n_supported     = e0_metrics["n_supported"],
            E0_n_contradicted  = e0_metrics["n_contradicted"],
            E0_n_not_supported = e0_metrics["n_not_supported"],
            # E1
            e1_summary         = e1_summary,
            e1_summary_words   = len(e1_summary.split()),
            E1_UFR             = e1_metrics["UFR"],
            E1_CR              = e1_metrics["CR"],
            E1_n_claims        = e1_metrics["n_claims"],
            E1_n_supported     = e1_metrics["n_supported"],
            E1_n_contradicted  = e1_metrics["n_contradicted"],
            E1_n_not_supported = e1_metrics["n_not_supported"],
        ))

        # ── Incremental saves every 10 samples ────────────────────────────────
        if (len(summary_rows) % 10) == 0:
            _save_partial(summary_rows, claim_rows, output_dir)
            log.info(f"  Partial save at {len(summary_rows)} samples.")

    # ── Final saves ───────────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_claims  = pd.DataFrame(claim_rows)

    df_summary.to_csv(output_dir / "summaries.csv",    index=False)
    df_claims.to_csv(output_dir  / "claims_all.csv",   index=False)

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison = _build_comparison_table(df_summary)
    comparison.to_csv(output_dir / "comparison_per_sample.csv", index=False)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    stats = _build_stats_table(df_summary)
    stats.to_csv(output_dir / "aggregate_statistics.csv", index=False)

    # ── Print final table ─────────────────────────────────────────────────────
    _print_results(stats, df_summary, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _save_partial(summary_rows, claim_rows, output_dir: Path) -> None:
    pd.DataFrame(summary_rows).to_csv(output_dir / "summaries_partial.csv",  index=False)
    pd.DataFrame(claim_rows).to_csv(  output_dir / "claims_partial.csv",     index=False)


def _build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "doc_id", "specialty", "description",
        "source_word_count",
        "e0_summary_words", "E0_UFR", "E0_CR",
        "E0_n_claims", "E0_n_supported", "E0_n_contradicted", "E0_n_not_supported",
        "e1_summary_words", "E1_UFR", "E1_CR",
        "E1_n_claims", "E1_n_supported", "E1_n_contradicted", "E1_n_not_supported",
    ]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["delta_UFR"] = df["E1_UFR"] - df["E0_UFR"]
    df["delta_CR"]  = df["E1_CR"]  - df["E0_CR"]
    return df


def _build_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("UFR", "CR"):
        e0_col = f"E0_{metric}"
        e1_col = f"E1_{metric}"
        if e0_col not in df.columns or e1_col not in df.columns:
            continue

        e0_vals = df[e0_col].dropna()
        e1_vals = df[e1_col].dropna()
        delta   = e1_vals - e0_vals

        rows.append(dict(
            Metric            = metric,
            E0_Mean           = round(e0_vals.mean(), 4),
            E0_Std            = round(e0_vals.std(),  4),
            E0_Median         = round(e0_vals.median(), 4),
            E1_Mean           = round(e1_vals.mean(), 4),
            E1_Std            = round(e1_vals.std(),  4),
            E1_Median         = round(e1_vals.median(), 4),
            Delta_Mean        = round(delta.mean(),   4),
            Delta_Std         = round(delta.std(),    4),
            Pct_Improved      = round((delta < 0).mean() * 100, 1),
        ))
    return pd.DataFrame(rows)


def _print_results(stats: pd.DataFrame, df: pd.DataFrame, output_dir: Path) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  HALLUCINATION EVALUATION  —  E0 (Baseline) vs E1 (RAG)")
    print(sep)
    print(f"  Samples evaluated : {len(df)}")
    print(f"  Output directory  : {output_dir}\n")

    # Aggregate table
    print(stats.to_string(index=False))

    print(f"""
Column glossary
  UFR  Unsupported Fact Rate = (Contradicted + Not-Supported) / total claims
  CR   Contradiction Rate    = Contradicted / total claims
  Delta = E1 − E0  (negative = RAG improved the metric)
  Pct_Improved = % samples where E1 < E0 for that metric

Output files
  results/summaries.csv            — per-sample summaries + metrics (E0 & E1)
  results/claims_all.csv           — per-claim labels for every sample/condition
  results/comparison_per_sample.csv — side-by-side E0 vs E1 per document
  results/aggregate_statistics.csv  — this table
{sep}
""")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
