#!/usr/bin/env python3
"""
analyze.py — Figures and example tables for the thesis, built from results/*.csv.

Runs offline.  Every figure is written to results/fig_*.png at 200 dpi.  Figures whose
inputs are missing (e.g. ablations not yet run) are skipped with a message.

Also writes:
  results/examples_fixed_by_rag.csv   E0 claims labeled Contradicted whose closest E1 claim is Supported
  results/examples_persistent.csv     claims unsupported in both E0 and E1 (closest pairs)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preprocessing import is_markdown_header  # noqa: E402

RES = HERE / "results"
COL = {"E0": "#E07B54", "E1": "#4C8BB5", "E1b": "#8E6BB5", "E2": "#5DBF6E", "E3": "#B5A14C"}
NAME = {"E0": "E0 Baseline LLM", "E1": "E1 RAG (excerpts only)", "E1b": "E1b RAG (note + excerpts)",
        "E2": "E2 Extractive", "E3": "E3 RAG + CoVe"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "figure.dpi": 100})


def csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    out = RES / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {out.name}")


def conds_in(df):
    return [c for c in ["E0", "E1", "E1b", "E2", "E3"] if f"{c}_UFR" in df.columns]


# ───────────────────────── 1. pipeline diagram ─────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")

    def box(x, y, w, h, text, fc="#F4F4F4", ec="#444444", fs=9.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, **kw):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.1, color=kw.get("color", "#333333"),
                                     linestyle=kw.get("ls", "-"), connectionstyle=kw.get("cs", "arc3,rad=0")))

    box(0.2, 2.9, 1.6, 1.0, "Source clinical\nnote (MTSamples)", fc="#FFFFFF", bold=True)
    gens = [(5.4, "E0  GPT-4o-mini\nfull note, no retrieval", "#F9E1D6"),
            (4.2, "E1  GPT-4o-mini\ntop-3 retrieved chunks only", "#DCE8F2"),
            (3.0, "E1b  GPT-4o-mini\nfull note + retrieved chunks", "#E6DDF2"),
            (1.8, "E3  E1 draft -> per-claim verification\nagainst the note -> revision", "#F1EBCF"),
            (0.6, "E2  Extractive\ncentroid top-5 sentences", "#DDF1E0")]
    for y, text, fc in gens:
        box(2.4, y, 2.5, 0.95, text, fc=fc, fs=8.8)
        arrow(1.8, 3.4, 2.4, y + 0.47)
    box(5.5, 2.9, 1.4, 1.0, "Summary\n(claims)", fc="#FFFFFF")
    for y, _, _ in gens:
        arrow(4.9, y + 0.47, 5.5, 3.4)
    box(7.3, 4.6, 2.5, 0.85, "Claim segmentation (spaCy)\nheader lines removed", fc="#F4F4F4")
    box(7.3, 3.45, 2.5, 0.85, "Evidence retrieval\nall-MiniLM-L6-v2, top-3 sentences", fc="#F4F4F4")
    box(7.3, 2.3, 2.5, 0.85, "NLI cross-encoder\nnli-MiniLM2-L6-H768", fc="#F4F4F4")
    box(7.3, 1.15, 2.5, 0.85, "Labels -> UFR, CR per summary\nWilcoxon, bootstrap CI", fc="#FFF6CC", bold=True)
    arrow(6.9, 3.4, 7.3, 5.02)
    for y in (4.6, 3.45, 2.3):
        arrow(8.55, y, 8.55, y - 0.3)
    ax.text(8.55, 5.85, "Claim-level NLI judge (identical for every condition)", ha="center", fontsize=10, fontweight="bold")
    ax.text(1.0, 2.35, "evidence sentences\nretrieved from the same note", ha="center", fontsize=8.5, style="italic", color="#555555")
    arrow(1.0, 2.9, 7.3, 3.87, color="#777777", ls="--", cs="arc3,rad=0.25")
    save(fig, "fig_pipeline.png")


# ───────────────────────── 2. distributions ─────────────────────────
def fig_boxplots(cmp):
    conds = conds_in(cmp)
    for metric, ylabel, fname in (("UFR", "Unsupported Fact Rate", "fig_ufr_boxplot.png"), ("CR", "Contradiction Rate", "fig_cr_boxplot.png")):
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        data = [cmp[f"{c}_{metric}"].dropna().values for c in conds]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        rng = np.random.default_rng(42)
        for i, (d, c) in enumerate(zip(data, conds), start=1):
            bp["boxes"][i - 1].set_facecolor(COL[c]); bp["boxes"][i - 1].set_alpha(0.75)
            ax.scatter(np.full(len(d), i) + rng.uniform(-0.15, 0.15, len(d)), d, s=12, color=COL[c], alpha=0.55, zorder=3, edgecolor="none")
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels([NAME[c] for c in conds], fontsize=9.5)
        ax.set_ylabel(ylabel); ax.set_ylim(-0.03, 1.03); ax.yaxis.grid(True, ls="--", alpha=0.4); ax.set_axisbelow(True)
        ax.set_title(f"{ylabel} per summary (n = {len(cmp)} documents)")
        save(fig, fname)


def fig_scatter(cmp):
    pairs = [("E0", b) for b in conds_in(cmp) if b != "E0"]
    ncol = 2 if len(pairs) > 2 else len(pairs); nrow = (len(pairs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[len(pairs):]:
        ax.axis("off")
    for ax, (a, b) in zip(axes_flat, pairs):
        x, y = cmp[f"{a}_CR"], cmp[f"{b}_CR"]
        ax.fill_between([0, 1], [0, 1], [0, 0], color=COL[b], alpha=0.08)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.scatter(x, y, s=28, color=COL[b], alpha=0.75, edgecolor="black", lw=0.4)
        below = int((y < x).sum()); ax.text(0.03, 0.92, f"{below}/{len(cmp)} below diagonal\n({b} lower CR)", transform=ax.transAxes, fontsize=9)
        lim = max(0.05, float(max(x.max(), y.max())) * 1.1)
        ax.set_xlim(-0.01, lim); ax.set_ylim(-0.01, lim)
        ax.set_xlabel(f"{a} contradiction rate"); ax.set_ylabel(f"{b} contradiction rate"); ax.set_title(f"{b} vs {a}, per document")
        ax.grid(True, ls="--", alpha=0.35)
    save(fig, "fig_cr_scatter.png")


def fig_pct_improved(tests):
    keep = [c for c in tests.comparison.unique() if c.endswith("vs E0") or c in ("E2 vs E1", "E3 vs E1", "E1b vs E1")]
    t = tests[tests.comparison.isin(keep)]
    comps = list(dict.fromkeys(t.comparison))
    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, 1.6 * len(comps) + 3), 3.8), sharey=True)
    for ax, metric in zip(axes, ("UFR", "CR")):
        sub = t[t.metric == metric].set_index("comparison").loc[comps]
        bottom = np.zeros(len(sub))
        for col, lab, colr in (("pct_improved", "improved (lower)", "#5DBF6E"), ("pct_unchanged", "unchanged", "#BBBBBB"), ("pct_worse", "worse (higher)", "#E07B54")):
            ax.bar(range(len(sub)), sub[col], bottom=bottom, color=colr, label=lab, width=0.6, edgecolor="white")
            for i, (v, b) in enumerate(zip(sub[col], bottom)):
                if v >= 8:
                    ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
            bottom += sub[col].values
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(comps, fontsize=8.5, rotation=20); ax.set_title(f"{metric}: share of documents"); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Percent of documents"); axes[1].legend(loc="upper center", bbox_to_anchor=(-0.1, -0.15), ncol=3, frameon=False)
    save(fig, "fig_pct_improved.png")


def fig_label_distribution(claims):
    c = claims[~claims.is_header & ~claims.get('is_abstention', False)]
    conds = [x for x in ["E0", "E1", "E1b", "E2", "E3"] if x in set(c.condition)]
    ct = pd.crosstab(c.condition, c.label, normalize="index").loc[conds]
    ct = ct[[l for l in ["Supported", "Not-Supported", "Contradicted"] if l in ct.columns]]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    left = np.zeros(len(ct))
    colors = {"Supported": "#5DBF6E", "Not-Supported": "#BBBBBB", "Contradicted": "#E07B54"}
    for lab in ct.columns:
        ax.barh(range(len(ct)), ct[lab] * 100, left=left, color=colors[lab], label=lab, edgecolor="white")
        for i, (v, l) in enumerate(zip(ct[lab] * 100, left)):
            if v > 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9)
        left += ct[lab].values * 100
    n = c.groupby("condition").size().loc[conds]
    ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"{NAME[x]}\n({n[x]} claims)" for x in conds], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of claims (%)"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    save(fig, "fig_label_distribution.png")


def fig_specialty(cmp):
    conds = conds_in(cmp)
    g = cmp.groupby("specialty")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ("UFR", "CR")):
        means = g[[f"{c}_{metric}" for c in conds]].mean(); sems = g[[f"{c}_{metric}" for c in conds]].sem()
        x = np.arange(len(means)); w = 0.8 / len(conds)
        for i, c in enumerate(conds):
            ax.bar(x + i * w - 0.4 + w / 2, means[f"{c}_{metric}"], w, yerr=sems[f"{c}_{metric}"], color=COL[c], label=NAME[c], capsize=3, edgecolor="black", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n = {int(n)})" for s, n in g.size().items()], fontsize=9)
        ax.set_ylabel(f"Mean {metric} (± s.e.m.)"); ax.set_title(f"{metric} by note type"); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5)
    save(fig, "fig_specialty.png")


def fig_length_claims(cmp):
    conds = conds_in(cmp)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (col, lab) in zip(axes, (("summary_words", "Words per summary"), ("n_claims", "Claims per summary (headers removed)"))):
        data = []
        for c in conds:
            key = f"{c.lower()}_{col}" if col == "summary_words" else f"{c}_{col}"
            data.append(cmp[key].dropna().values if key in cmp.columns else np.array([]))
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], conds):
            patch.set_facecolor(COL[c]); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds); ax.set_ylabel(lab); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].set_title("Summary length"); axes[1].set_title("Number of claims")
    save(fig, "fig_length_claims.png")


def fig_header_effect(eff):
    e = eff.copy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ("UFR", "CR")):
        sub = e[e.metric == metric].set_index("condition")
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w / 2, sub.mean_before, w, color="#BBBBBB", label="headers counted as claims (original)", edgecolor="black", lw=0.4)
        ax.bar(x + w / 2, sub.mean_after, w, color=[COL[c] for c in sub.index], label="headers removed (this thesis)", edgecolor="black", lw=0.4)
        for i, (b, a) in enumerate(zip(sub.mean_before, sub.mean_after)):
            ax.text(i - w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8.5); ax.text(i + w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(sub.index); ax.set_ylabel(f"Mean {metric}"); ax.set_title(f"Effect of the header filter on {metric}")
        ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    h = [mpatches.Patch(color="#BBBBBB", label="headers counted as claims (original pipeline)"), mpatches.Patch(color="#888888", label="headers removed (corrected)")]
    axes[1].legend(handles=h, frameon=False, fontsize=8.5, loc="upper right")
    save(fig, "fig_header_effect.png")


def fig_thresholds(tau):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    conds = [c for c in ["E0", "E1", "E1b", "E2", "E3"] if f"{c}_UFR_mean" in tau.columns]
    for ax, metric in zip(axes, ("UFR", "CR")):
        for c in conds:
            ax.plot(tau.tau, tau[f"{c}_{metric}_mean"], marker="o", color=COL[c], label=NAME[c])
        ax.set_xlabel("NLI decision threshold τ"); ax.set_ylabel(f"Mean {metric}"); ax.set_title(f"{metric} versus decision threshold"); ax.grid(True, ls="--", alpha=0.35)
        ax2 = ax.twinx(); ax2.plot(tau.tau, tau[f"p_E1_vs_E0_{metric}"], ls=":", color="black", marker="x", label="p (E1 vs E0)")
        ax2.set_yscale("log"); ax2.set_ylabel("Wilcoxon p, E1 vs E0 (log)"); ax2.axhline(0.05, color="red", lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=8.5, loc="center left")
    save(fig, "fig_threshold_ablation.png")


def fig_topk(topk):
    conds = [c for c in ["E0", "E1", "E1b", "E2", "E3"] if f"{c}_UFR_mean" in topk.columns]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ("UFR", "CR")):
        x = np.arange(len(topk)); w = 0.8 / len(conds)
        for i, c in enumerate(conds):
            ax.bar(x + i * w - 0.4 + w / 2, topk[f"{c}_{metric}_mean"], w, color=COL[c], label=NAME[c], edgecolor="black", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"k = {int(k)}" for k in topk.k]); ax.set_ylabel(f"Mean {metric}"); ax.set_title(f"{metric} versus evidence sentences retrieved (k)")
        ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5)
    save(fig, "fig_topk_ablation.png")


def fig_e2_probs(claims):
    e2 = claims[(claims.condition == "E2") & (~claims.is_header) & (~claims.get('is_abstention', False))]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    colors = {"Supported": "#5DBF6E", "Not-Supported": "#999999", "Contradicted": "#E07B54"}
    for lab, d in e2.groupby("label"):
        ax.scatter(d.p_entailment, d.p_contradiction, s=26, color=colors.get(lab, "black"), label=f"{lab} (n = {len(d)})", alpha=0.8, edgecolor="black", lw=0.3)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1); ax.axvline(0.5, color="gray", lw=0.7, ls=":"); ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("max entailment probability over evidence"); ax.set_ylabel("max contradiction probability over evidence")
    ax.set_title("E2 verbatim source sentences: judge probabilities"); ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    save(fig, "fig_e2_probs.png")


def fig_calibration(cal, sweep):
    groups = [g for g in ["all", "generated", "doctor_written", "llama_70b_original", "llama_70b_cleaned", "gpt4_zero_shot", "gpt4_orig", "gpt4_cleaned"] if g in set(cal.group)]
    c = cal.set_index("group").loc[groups]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]; x = np.arange(len(groups)); w = 0.2
    ax.bar(x - 1.5 * w, c.expert_flag_rate, w, label="expert-flagged share of sentences", color="#333333", edgecolor="black", lw=0.4)
    ax.bar(x - 0.5 * w, c.judge_UFR, w, label="judge-flagged share (UFR)", color="#BBBBBB", edgecolor="black", lw=0.4)
    ax.bar(x + 0.5 * w, c.any_precision, w, label="precision of judge flags", color="#4C8BB5", edgecolor="black", lw=0.4)
    ax.bar(x + 1.5 * w, c.any_recall, w, label="recall of judge flags", color="#E07B54", edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("_", "\n") for g in groups], fontsize=8.5); ax.set_ylim(0, 1.05)
    ax.set_title("Judge versus medical experts, sentence level"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax = axes[1]
    priv = HERE / "results_private" / "calibration_sentences.csv"
    if priv.exists():
        d = pd.read_csv(priv)
        bins = np.linspace(0, 1, 21)
        ax.hist(d.loc[~d.expert_flag, "p_entailment"], bins=bins, density=True, alpha=0.6, color="#5DBF6E", label=f"not flagged by experts (n = {int((~d.expert_flag).sum())})")
        ax.hist(d.loc[d.expert_flag, "p_entailment"], bins=bins, density=True, alpha=0.6, color="#E07B54", label=f"flagged by experts (n = {int(d.expert_flag.sum())})")
        ax.axvline(0.5, color="black", ls=":", lw=1); ax.set_xlabel("judge entailment probability p(e)"); ax.set_ylabel("density")
        ax.set_title("Entailment probability by expert judgment"); ax.legend(frameon=False, fontsize=8.5)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "per-sentence data not available", ha="center")
    save(fig, "fig_calibration.png")


def fig_variants(var):
    v = var[var.group == "all"].set_index("variant")
    order = [k for k in ["max3", "concat3", "concat5", "maxall", "concat_ctx"] if k in v.index]
    names = {"max3": "max over\ntop-3", "concat3": "concat\ntop-3", "concat5": "concat\ntop-5", "maxall": "max over\nall sentences", "concat_ctx": "whole\ncontext"}
    v = v.loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]; x = np.arange(len(order)); w = 0.26
    ax.bar(x - w, v.auroc_1_minus_pe, w, label="AUROC, 1 − p(e)", color="#4C8BB5", edgecolor="black", lw=0.4)
    ax.bar(x, v.best_kappa, w, label="best κ over thresholds", color="#5DBF6E", edgecolor="black", lw=0.4)
    ax.bar(x + w, v.judge_flag_rate, w, label="judge flag rate at τ = 0.5", color="#BBBBBB", edgecolor="black", lw=0.4)
    ax.axhline(v.expert_flag_rate.iloc[0], color="black", ls="--", lw=1, label="expert flag rate")
    ax.set_xticks(x); ax.set_xticklabels([names[k] for k in order], fontsize=9); ax.set_ylim(0, 1); ax.set_title("Evidence aggregation variants, all 210 summaries")
    ax.legend(frameon=False, fontsize=8); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    ax = axes[1]
    for g, colr in (("generated", "#4C8BB5"), ("doctor_written", "#E07B54")):
        s = var[var.group == g].set_index("variant").loc[order]
        ax.plot(range(len(order)), s.summary_spearman, marker="o", color=colr, label=f"{g}")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([names[k] for k in order], fontsize=9); ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Spearman ρ, judge UFR vs expert share"); ax.set_title("Summary-level rank agreement"); ax.legend(frameon=False, fontsize=8.5); ax.grid(True, ls="--", alpha=0.35)
    save(fig, "fig_variants.png")


def fig_coverage(cov_ps):
    conds = [c for c in ["E0", "E1", "E1b", "E2", "E3"] if c in set(cov_ps.condition)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (col, lab) in zip(axes, (("coverage_at_0_6", "Source sentences covered (cos ≥ 0.6)"), ("mean_best_similarity", "Mean best similarity of source sentences"))):
        data = [cov_ps.loc[cov_ps.condition == c, col].values for c in conds]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], conds):
            patch.set_facecolor(COL[c]); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds); ax.set_ylabel(lab); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].set_title("Coverage proxy"); axes[1].set_title("Soft coverage")
    save(fig, "fig_coverage.png")


def fig_taxonomy(tc):
    t = tc[tc.label == "All unsupported"].pivot(index="category", columns="condition", values="count").fillna(0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(t)); w = 0.38
    gens = [x for x in ["E0", "E1", "E1b", "E3"] if x in t.columns]; w = 0.8 / len(gens)
    for i, c in enumerate(gens):
        ax.barh(y + (i - (len(gens) - 1) / 2) * w, t[c], w, color=COL[c], label=NAME[c], edgecolor="black", lw=0.4)
    ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=9.5); ax.set_xlabel("Unsupported or contradicted claims"); ax.legend(frameon=False)
    ax.set_title("Keyword-assisted categories of claims not supported by the note"); ax.xaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    fig.set_size_inches(8, 4.2 + 0.6 * max(0, len(gens) - 2))
    save(fig, "fig_taxonomy.png")


def fig_judges(jc):
    a = jc[jc.group == "all"].copy()
    names = {"minilm_nli": "MiniLM\n(pilot)", "deberta_large_nli": "DeBERTa-L\nNLI", "ce_deberta_large_nli": "DeBERTa-L\ncross-enc.", "minicheck_deberta": "MiniCheck\nDeBERTa",
             "minicheck_roberta": "MiniCheck\nRoBERTa", "mednli_deberta_large": "DeBERTa-L\n+MedNLI", "bespoke_minicheck_7b": "Bespoke\nMiniCheck-7B"}
    order = [k for k in names if k in set(a.judge)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in zip(axes, ("auroc", "kappa"), ("AUROC of the support score", "Cohen's kappa (threshold from other subset)")):
        x = np.arange(len(order)); w = 0.38
        for i, (mode, colr) in enumerate((("top3", "#4C8BB5"), ("doc", "#E07B54"))):
            vals = [a[(a.judge == k) & (a["mode"] == mode)][metric].iloc[0] if len(a[(a.judge == k) & (a["mode"] == mode)]) else np.nan for k in order]
            ax.bar(x + (i - 0.5) * w, vals, w, color=colr, label={"top3": "top-3 sentences", "doc": "whole course, windowed"}[mode], edgecolor="black", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels([names[k] for k in order], fontsize=8); ax.set_title(title); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(order) - 0.4)
        if metric == "auroc":
            ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_ylim(0.4, 1.0)
        else:
            ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend(frameon=False, fontsize=8.5)
    save(fig, "fig_judges.png")


def fig_mimic(per):
    order = [c for c in ["E0", "E1", "E1b", "E3", "E2", "REF"] if c in set(per.condition)]
    colm = dict(COL, REF="#999999"); names = dict(NAME, REF="Doctor-written")
    base = per[per.variant == "base"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, lab in zip(axes, ("UFR", "CR"), ("Unsupported Fact Rate", "Contradiction Rate")):
        data = [base.loc[base.condition == c, metric].dropna().values for c in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black", lw=2), flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(colm.get(c, "#999999")); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([names.get(c, c).replace(" (", "\n(") for c in order], fontsize=8); ax.set_ylabel(lab)
        ax.set_ylim(-0.03, 1.03); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0].set_title("Main study: UFR per document"); axes[1].set_title("Main study: CR per document")
    save(fig, "fig_mimic_box.png")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    order2 = [c for c in order if c != "REF"]
    data = [base.loc[base.condition == c, "coverage_ref"].dropna().values for c in order2]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], order2):
        patch.set_facecolor(colm.get(c, "#999999")); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(order2) + 1)); ax.set_xticklabels(order2); ax.set_ylabel("Share of clinician's sentences supported"); ax.set_ylim(-0.03, 1.03)
    ax.set_title("Coverage of the doctor-written discharge instructions"); ax.yaxis.grid(True, ls="--", alpha=0.35); ax.set_axisbelow(True)
    save(fig, "fig_mimic_coverage.png")


# ───────────────────────── examples ─────────────────────────
def _tok(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def examples(claims):
    c = claims[~claims.is_header & ~claims.get('is_abstention', False)]
    fixed, persistent = [], []
    for doc_id, d in c.groupby("doc_id"):
        e0 = d[d.condition == "E0"]; e1 = d[d.condition == "E1"]
        for _, r0 in e0.iterrows():
            best, best_j = 0.0, None
            for _, r1 in e1.iterrows():
                a, b = _tok(r0.claim), _tok(r1.claim)
                j = len(a & b) / max(1, len(a | b))
                if j > best:
                    best, best_j = j, r1
            if best_j is None or best < 0.2:
                continue
            rec = dict(doc_id=doc_id, specialty=r0.specialty, jaccard=round(best, 2), e0_claim=r0.claim, e0_label=r0.label,
                       e1_claim=best_j.claim, e1_label=best_j.label, e0_evidence=r0.evidence)
            if r0.label == "Contradicted" and best_j.label == "Supported":
                fixed.append(rec)
            elif r0.label != "Supported" and best_j.label != "Supported":
                persistent.append(rec)
    pd.DataFrame(fixed).sort_values("jaccard", ascending=False).to_csv(RES / "examples_fixed_by_rag.csv", index=False)
    pd.DataFrame(persistent).sort_values("jaccard", ascending=False).to_csv(RES / "examples_persistent.csv", index=False)
    print(f"  examples: {len(fixed)} fixed-by-RAG pairs, {len(persistent)} persistent pairs")


def main():
    cmp, tests, claims = csv("comparison_per_sample.csv"), csv("pairwise_tests.csv"), csv("claims_all.csv")
    if "is_header" not in claims.columns:
        claims["is_header"] = claims.claim.astype(str).map(is_markdown_header)
    print("Figures:")
    fig_pipeline(); fig_boxplots(cmp); fig_scatter(cmp); fig_pct_improved(tests); fig_label_distribution(claims)
    fig_specialty(cmp); fig_length_claims(cmp); fig_e2_probs(claims)
    eff = csv("header_filter_effect.csv");   fig_header_effect(eff) if eff is not None else print("  skip header effect")
    tau = csv("ablation_thresholds.csv");    fig_thresholds(tau) if tau is not None else print("  skip thresholds (not run)")
    topk = csv("ablation_topk.csv");         fig_topk(topk) if topk is not None else print("  skip top-k (not run)")
    cal = csv("calibration_overall.csv");    fig_calibration(cal, csv("calibration_threshold_sweep.csv")) if cal is not None else print("  skip calibration (not run)")
    var = csv("calibration_variants.csv"); fig_variants(var) if var is not None else print("  skip variants (not run)")
    cov = csv("coverage_per_sample.csv");    fig_coverage(cov) if cov is not None else print("  skip coverage (not run)")
    tc = csv("error_taxonomy_counts.csv");   fig_taxonomy(tc) if tc is not None else print("  skip taxonomy (not run)")
    jc = csv("judge_candidates.csv");        fig_judges(jc) if jc is not None else print("  skip judges (not run)")
    per_path = HERE / "results_private" / "mimic_per_summary.csv"
    fig_mimic(pd.read_csv(per_path)) if per_path.exists() else print("  skip main-study figures (not evaluated)")
    examples(claims)
    print("done")


if __name__ == "__main__":
    main()
