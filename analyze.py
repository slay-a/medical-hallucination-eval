#!/usr/bin/env python3
"""
analyze.py — Post-hoc analysis of hallucination evaluation results.

Generates:
  - 4 PNG plots in results/
  - Wilcoxon signed-rank tests (UFR and CR)
  - 5 example claims fixed by RAG (Contradicted→Supported)
  - 5 example persistent hallucinations (Not-Supported in both E0 and E1)
"""

import os
import textwrap
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

RESULTS = os.path.join(os.path.dirname(__file__), "results")
CLAIMS_CSV = os.path.join(RESULTS, "claims_all.csv")
CMP_CSV    = os.path.join(RESULTS, "comparison_per_sample.csv")

# ── palette ────────────────────────────────────────────────────────────────
C_E0 = "#E07B54"   # warm orange  → baseline
C_E1 = "#4C8BB5"   # steel blue   → RAG
GREY = "#888888"

def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)


def section(title):
    bar = "─" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════
section("Loading data")
claims = pd.read_csv(CLAIMS_CSV)
cmp    = pd.read_csv(CMP_CSV)
print(f"  claims rows : {len(claims):,}  ({claims['condition'].value_counts().to_dict()})")
print(f"  samples     : {len(cmp)}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Box plots — UFR and CR
# ══════════════════════════════════════════════════════════════════════════
section("Generating plots")

def boxplot_metric(metric_e0, metric_e1, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    data   = [cmp[metric_e0].values, cmp[metric_e1].values]
    labels = ["E0  Baseline", "E1  RAG"]
    colors = [C_E0, C_E1]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker="o", markerfacecolor=GREY, markersize=4, linestyle="none"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # overlay jittered points
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d,
                   color=color, alpha=0.45, s=18, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, filename)

boxplot_metric("E0_UFR", "E1_UFR",
               "Unsupported Fact Rate",
               "UFR Distribution — Baseline vs RAG",
               "plot_ufr_boxplot.png")

boxplot_metric("E0_CR", "E1_CR",
               "Contradiction Rate",
               "CR Distribution — Baseline vs RAG",
               "plot_cr_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════
# 3. Scatter plot — E0 CR vs E1 CR per sample
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5.5))

ax.scatter(cmp["E0_CR"], cmp["E1_CR"],
           color=C_E1, alpha=0.7, s=45, zorder=3, label="Sample")

# diagonal
lim_max = max(cmp["E0_CR"].max(), cmp["E1_CR"].max()) * 1.05
ax.plot([0, lim_max], [0, lim_max], color=GREY, linewidth=1.3,
        linestyle="--", label="No change (y = x)")

# shade region below diagonal (RAG improved)
ax.fill_between([0, lim_max], [0, 0], [0, lim_max],
                color=C_E1, alpha=0.07, label="RAG improved (below line)")

ax.set_xlabel("E0 Contradiction Rate  (Baseline)", fontsize=11)
ax.set_ylabel("E1 Contradiction Rate  (RAG)",       fontsize=11)
ax.set_title("Per-Sample CR: Baseline vs RAG", fontsize=13,
             fontweight="bold", pad=10)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_aspect("equal", adjustable="box")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "plot_cr_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. Bar chart — % samples improved for UFR and CR
# ══════════════════════════════════════════════════════════════════════════
pct_ufr_improved = (cmp["delta_UFR"] < 0).mean() * 100
pct_cr_improved  = (cmp["delta_CR"]  < 0).mean() * 100
pct_ufr_worse    = (cmp["delta_UFR"] > 0).mean() * 100
pct_cr_worse     = (cmp["delta_CR"]  > 0).mean() * 100
pct_ufr_same     = (cmp["delta_UFR"] == 0).mean() * 100
pct_cr_same      = (cmp["delta_CR"]  == 0).mean() * 100

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x      = np.array([0, 1])
width  = 0.25

improved = [pct_ufr_improved, pct_cr_improved]
same     = [pct_ufr_same,     pct_cr_same]
worse    = [pct_ufr_worse,    pct_cr_worse]

b1 = ax.bar(x - width, improved, width, label="Improved (E1 < E0)",
            color="#4CB87A", alpha=0.85)
b2 = ax.bar(x,          same,    width, label="No change",
            color=GREY,    alpha=0.6)
b3 = ax.bar(x + width,  worse,   width, label="Worse (E1 > E0)",
            color="#D9534F", alpha=0.85)

# value labels
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if h > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["UFR", "CR"], fontsize=12)
ax.set_ylabel("% of samples", fontsize=11)
ax.set_title("Samples Improved / Unchanged / Worse — RAG vs Baseline",
             fontsize=12, fontweight="bold", pad=10)
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "plot_pct_improved_bar.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. Wilcoxon signed-rank tests
# ══════════════════════════════════════════════════════════════════════════
section("Wilcoxon Signed-Rank Tests  (two-sided, H0: no difference)")

for metric, col_e0, col_e1 in [
    ("UFR", "E0_UFR", "E1_UFR"),
    ("CR",  "E0_CR",  "E1_CR"),
]:
    e0 = cmp[col_e0].values
    e1 = cmp[col_e1].values
    diff = e1 - e0
    n_nonzero = (diff != 0).sum()
    if n_nonzero < 2:
        print(f"  {metric}: not enough non-zero differences to test (n={n_nonzero})")
        continue
    stat, p = wilcoxon(e0, e1, alternative="two-sided", zero_method="wilcox")
    direction = "E1 < E0  (RAG reduced)" if np.median(diff) < 0 else "E1 > E0  (RAG increased)"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"\n  {metric}")
    print(f"    W statistic : {stat:.1f}")
    print(f"    p-value     : {p:.4f}  {sig}")
    print(f"    median Δ    : {np.median(diff):+.4f}  ({direction})")
    print(f"    mean  Δ     : {np.mean(diff):+.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 6. RAG-fixed — docs where E0 had Contradicted claims but E1 had Supported
# ══════════════════════════════════════════════════════════════════════════
# E0 and E1 generate different summaries so claims won't match verbatim.
# Instead: find documents that have ≥1 Contradicted claim in E0 AND ≥1
# Supported claim in E1, then show the most contradicted E0 claim alongside
# the most-supported E1 claim for the same document.
section("5 Claim Pairs: E0 Contradicted vs E1 Supported  (RAG improved same doc)")

e0_all = claims[claims["condition"] == "E0"].copy()
e1_all = claims[claims["condition"] == "E1"].copy()

docs_e0_contra = set(e0_all[e0_all["label"] == "Contradicted"]["doc_id"])
docs_e1_supp   = set(e1_all[e1_all["label"] == "Supported"]["doc_id"])
improved_docs  = sorted(docs_e0_contra & docs_e1_supp)

print(f"\n  Docs with E0 Contradicted AND E1 Supported claim: {len(improved_docs)}")

rng = np.random.default_rng(42)
chosen_docs = rng.choice(improved_docs, size=min(5, len(improved_docs)), replace=False)

for rank, doc_id in enumerate(chosen_docs, 1):
    # pick E0 claim with highest contradiction probability
    e0_doc = e0_all[(e0_all["doc_id"] == doc_id) & (e0_all["label"] == "Contradicted")]
    e0_row = e0_doc.sort_values("p_contradiction", ascending=False).iloc[0]
    # pick E1 claim with highest entailment probability
    e1_doc = e1_all[(e1_all["doc_id"] == doc_id) & (e1_all["label"] == "Supported")]
    e1_row = e1_doc.sort_values("p_entailment", ascending=False).iloc[0]

    ev_e0 = str(e0_row["evidence"]).split(" | ")[0][:120]
    ev_e1 = str(e1_row["evidence"]).split(" | ")[0][:120]
    print(f"\n  [{rank}] doc_id={doc_id}  ({e0_row['description'][:60]})")
    print(f"  E0 claim : {textwrap.fill(str(e0_row['claim']), 78, subsequent_indent='             ')}")
    print(f"  E0 evid  : {textwrap.fill(ev_e0, 78, subsequent_indent='             ')}")
    print(f"  E0 label : Contradicted  (p_contra={e0_row['p_contradiction']:.3f})")
    print(f"  E1 claim : {textwrap.fill(str(e1_row['claim']), 78, subsequent_indent='             ')}")
    print(f"  E1 evid  : {textwrap.fill(ev_e1, 78, subsequent_indent='             ')}")
    print(f"  E1 label : Supported     (p_entail={e1_row['p_entailment']:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# 7. Persistent hallucinations — docs where BOTH E0 and E1 have Not-Supported
#    claims. Show the least-supported E0 claim (lowest p_entailment) plus
#    the corresponding worst E1 claim for the same document.
# ══════════════════════════════════════════════════════════════════════════
section("5 Persistent Hallucinations  (Not-Supported in both E0 and E1)")

docs_e0_ns = set(e0_all[e0_all["label"] == "Not-Supported"]["doc_id"])
docs_e1_ns = set(e1_all[e1_all["label"] == "Not-Supported"]["doc_id"])
persist_docs = sorted(docs_e0_ns & docs_e1_ns)

print(f"\n  Docs with Not-Supported claims in BOTH E0 and E1: {len(persist_docs)}")

chosen_persist = rng.choice(persist_docs, size=min(5, len(persist_docs)), replace=False)

for rank, doc_id in enumerate(chosen_persist, 1):
    e0_ns = e0_all[(e0_all["doc_id"] == doc_id) & (e0_all["label"] == "Not-Supported")]
    e0_row = e0_ns.sort_values("p_entailment").iloc[0]   # least supported
    e1_ns = e1_all[(e1_all["doc_id"] == doc_id) & (e1_all["label"] == "Not-Supported")]
    e1_row = e1_ns.sort_values("p_entailment").iloc[0]

    ev_e0 = str(e0_row["evidence"]).split(" | ")[0][:120]
    ev_e1 = str(e1_row["evidence"]).split(" | ")[0][:120]
    print(f"\n  [{rank}] doc_id={doc_id}  ({e0_row['description'][:60]})")
    print(f"  E0 claim : {textwrap.fill(str(e0_row['claim']), 78, subsequent_indent='             ')}")
    print(f"  E0 evid  : {textwrap.fill(ev_e0, 78, subsequent_indent='             ')}")
    print(f"  E0 label : Not-Supported  (p_entail={e0_row['p_entailment']:.3f})")
    print(f"  E1 claim : {textwrap.fill(str(e1_row['claim']), 78, subsequent_indent='             ')}")
    print(f"  E1 evid  : {textwrap.fill(ev_e1, 78, subsequent_indent='             ')}")
    print(f"  E1 label : Not-Supported  (p_entail={e1_row['p_entailment']:.3f})")


# ══════════════════════════════════════════════════════════════════════════
# 8. Summary
# ══════════════════════════════════════════════════════════════════════════
section("Summary")
print(f"  Plots written to: {RESULTS}/")
print(f"    plot_ufr_boxplot.png")
print(f"    plot_cr_boxplot.png")
print(f"    plot_cr_scatter.png")
print(f"    plot_pct_improved_bar.png")
print(f"\n  UFR: {pct_ufr_improved:.0f}% of samples improved with RAG")
print(f"  CR : {pct_cr_improved:.0f}% of samples improved with RAG")
print(f"  Docs with E0-Contradicted / E1-Supported pairs: {len(improved_docs)}")
print(f"  Docs with persistent Not-Supported in both E0+E1: {len(persist_docs)}")
