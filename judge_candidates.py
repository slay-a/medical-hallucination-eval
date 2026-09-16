#!/usr/bin/env python3
"""
judge_candidates.py — Head-to-head evaluation of candidate claim judges against the ann-pt-summ
medical-expert annotations.  Local models only; MIMIC-derived text never leaves this machine.

Candidates (all run on the same 1,781 summary sentences):
  minilm_nli           cross-encoder/nli-MiniLM2-L6-H768          (the judge used so far; 3-way NLI, 22M)
  deberta_large_nli    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (3-way NLI, 435M)
  ce_deberta_large_nli cross-encoder/nli-deberta-v3-large          (3-way NLI, 435M)
  minicheck_deberta    lytang/MiniCheck-DeBERTa-v3-Large           (grounding fact-checker, 2-way, 435M)
  minicheck_roberta    lytang/MiniCheck-RoBERTa-Large              (grounding fact-checker, 2-way, 355M)

Evidence modes per candidate:
  top3    the three most similar context sentences (all-MiniLM-L6-v2) concatenated into one premise
  doc     the whole hospital course, split into windows of consecutive sentences (<= 400 tokens); max support over windows

Outputs: results/judge_candidates.csv (aggregate), results_private/judge_candidates_sentences.csv (per sentence; git-ignored)
Usage:   python judge_candidates.py [--only name,name] [--data-dir DIR]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hallucination_eval as he  # noqa: E402
from calibration_annptsumm import DEFAULT_DATA, SYSTEMS, read_jsonl, summary_sentences, confusion  # noqa: E402

RESULTS, PRIVATE = HERE / "results", HERE / "results_private"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CANDIDATES = {
    "minilm_nli": dict(kind="nli3", name="cross-encoder/nli-MiniLM2-L6-H768", order=["contradiction", "entailment", "neutral"]),
    "deberta_large_nli": dict(kind="nli3", name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", order=None),
    "ce_deberta_large_nli": dict(kind="nli3", name="cross-encoder/nli-deberta-v3-large", order=["contradiction", "entailment", "neutral"]),
    "minicheck_deberta": dict(kind="minicheck", name="lytang/MiniCheck-DeBERTa-v3-Large"),
    "minicheck_roberta": dict(kind="minicheck", name="lytang/MiniCheck-RoBERTa-Large"),
    # local fine-tuned clinical NLI model produced by finetune_mednli.py (outside the repository)
    "mednli_deberta_large": dict(kind="nli3", name=str(Path.home() / "Desktop" / "Thesis" / "models" / "mednli-deberta-v3-large"), order=None),
    # Bespoke-MiniCheck-7B (InternLM2.5-7B fine-tuned for grounding checks; top of the LLM-AggreFact leaderboard), served locally
    # by minicheck_server.py (MLX, 8-bit) and queried over localhost; reads the whole hospital course in one pass
    "bespoke_minicheck_7b": dict(kind="llm_yesno", name="bespokelabs/Bespoke-MiniCheck-7B", url="http://127.0.0.1:8081"),
}


def load_items(data_dir: Path):
    items = []
    for i, r in enumerate(read_jsonl(data_dir / "hallucinations_generated_di.jsonl")):
        items.append(dict(group=SYSTEMS[i // 20], source="generated", sid=f"gen_{i}", **r))
    for i, r in enumerate(read_jsonl(data_dir / "hallucinations_mimic_di.jsonl")):
        items.append(dict(group="doctor_written", source="doctor", sid=f"doc_{i}", **r))
    for i, r in enumerate(read_jsonl(data_dir / "hallucinations_mimic_di_validation.jsonl")):
        items.append(dict(group="doctor_written", source="doctor", sid=f"val_{i}", **r))
    return items


class Judge:
    def __init__(self, key, spec):
        self.key, self.kind = key, spec["kind"]
        if self.kind == "llm_yesno":
            import urllib.request
            self.url = spec["url"]
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                info = json.loads(r.read())
            print(f"[{key}] using local scoring server: {info.get('model')}", flush=True)
            self.tok = None; self.model = None; self.max_len = None
            return
        self.tok = AutoTokenizer.from_pretrained(spec["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(spec["name"]).to(DEVICE).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if self.kind == "nli3":
            names = [id2label[i] for i in range(len(id2label))]
            if any(n.startswith("label_") for n in names):
                names = spec["order"]
            self.i_ent = names.index("entailment"); self.i_con = names.index("contradiction")
        self.max_len = 512

    @torch.no_grad()
    def probs(self, premises, hypotheses, batch=16):
        out = []
        idx = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            enc = self.tok([premises[i] for i in b], [hypotheses[i] for i in b], truncation="only_first", max_length=self.max_len,
                           padding=True, return_tensors="pt").to(DEVICE)
            p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            out.append((b, p))
        res = np.zeros((len(premises), out[0][1].shape[1]))
        for b, p in out:
            res[b] = p
        return res

    def support_scores(self, premises, hypotheses):
        """Return (p_support, p_contra) arrays."""
        if self.kind == "llm_yesno":
            import urllib.request
            py = np.zeros(len(premises))
            for s in range(0, len(premises), 512):
                body = json.dumps({"docs": list(premises[s:s + 512]), "claims": list(hypotheses[s:s + 512])}).encode()
                req = urllib.request.Request(self.url + "/score", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=36000) as r:
                    out = json.loads(r.read())
                py[s:s + 512] = out["p_yes"]
            return py, np.zeros(len(premises))          # a binary grounding checker has no contradiction channel
        p = self.probs(premises, hypotheses)
        if self.kind == "nli3":
            return p[:, self.i_ent], p[:, self.i_con]
        return p[:, 1], np.zeros(len(premises))          # MiniCheck: label 1 = supported

    def windows(self, sentences, max_tokens=400):
        if self.kind == "llm_yesno":                     # the 32k-context checker reads the whole course at once
            return [" ".join(sentences)]
        wins, cur, cur_len = [], [], 0
        for s in sentences:
            n = len(self.tok.tokenize(s))
            if cur and cur_len + n > max_tokens:
                wins.append(" ".join(cur)); cur, cur_len = [], 0
            cur.append(s); cur_len += n
        if cur:
            wins.append(" ".join(cur))
        return wins or [" ".join(sentences)]


def evaluate(df, score_col):
    """Threshold-free and thresholded agreement for a support score (higher = more supported)."""
    rows = []
    e_all = df.expert_flag.to_numpy(bool); s_all = df[score_col].to_numpy()
    for gname, mask in (("all", np.ones(len(df), bool)), ("generated", (df.source == "generated").to_numpy()), ("doctor_written", (df.source == "doctor").to_numpy())):
        e, s = e_all[mask], s_all[mask]
        auroc = roc_auc_score(e, 1 - s)
        # thresholds chosen on the *other* subset to avoid optimism
        other = ~mask if gname != "all" else mask
        eo, so = e_all[other], s_all[other]
        best_tau, best_k = 0.5, -1
        for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
            k = cohen_kappa_score(eo, so < tau) if len(set(so < tau)) > 1 else -1
            if k > best_k:
                best_k, best_tau = k, tau
        flag = s < best_tau; c = confusion(e, flag)
        c50 = confusion(e, s < 0.5)
        rows.append(dict(group=gname, n=int(mask.sum()), expert_rate=e.mean(), auroc=auroc, tau_from_other_subset=best_tau,
                         flag_rate=flag.mean(), precision=c["precision"], recall=c["recall"], specificity=c["specificity"], f1=c["f1"], kappa=c["kappa"],
                         kappa_at_0_5=c50["kappa"], flag_rate_at_0_5=(s < 0.5).mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(CANDIDATES)
    items = load_items(Path(args.data_dir))
    nlp, enc = he.get_nlp(), he.get_bi_encoder()
    PRIVATE.mkdir(exist_ok=True)
    print(f"device={DEVICE}; {len(items)} summaries; candidates={keys}", flush=True)

    # sentence table with retrieval done once
    base = []
    for it in items:
        sents = summary_sentences(it["summary"], nlp); ctx = he.sentencize(it["text"])
        if not sents or not ctx:
            continue
        S = enc.encode(ctx, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        C = enc.encode([s for s, _, _ in sents], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        order = np.argsort(-(C @ S.T), axis=1)
        for i, (text, a, b) in enumerate(sents):
            flagged = any(max(a, lb["start"]) < min(b, lb["end"]) for lb in it["labels"])
            base.append(dict(sid=it["sid"], group=it["group"], source=it["source"], sentence=text, expert_flag=flagged,
                             top3=" ".join(ctx[j] for j in sorted(order[i][:3].tolist())), ctx_sents=ctx))
    df = pd.DataFrame(base)
    print(f"{len(df)} sentences; expert-flagged {df.expert_flag.mean():.3f}", flush=True)

    agg, prev = [], None
    if (RESULTS / "judge_candidates.csv").exists():
        prev = pd.read_csv(RESULTS / "judge_candidates.csv")
        prev = prev[~prev.judge.isin(keys)]
    for key in keys:
        t0 = time.time(); spec = CANDIDATES[key]
        try:
            J = Judge(key, spec)
        except Exception as exc:  # noqa: BLE001
            print(f"[{key}] could not load: {exc}", flush=True); continue
        # top-3 concatenated premise
        ps, pc = J.support_scores(df.top3.tolist(), df.sentence.tolist())
        df[f"{key}__top3_sup"], df[f"{key}__top3_con"] = ps, pc
        # whole document, windowed, max support over windows
        premises, hyps, owners = [], [], []
        for i, row in df.iterrows():
            for w in J.windows(row.ctx_sents):
                premises.append(w); hyps.append(row.sentence); owners.append(i)
        ps_w, pc_w = J.support_scores(premises, hyps)
        owners = np.array(owners)
        sup_doc = np.zeros(len(df)); con_doc = np.zeros(len(df))
        for i in range(len(df)):
            m = owners == i
            sup_doc[i] = ps_w[m].max(); con_doc[i] = pc_w[m].max()
        df[f"{key}__doc_sup"], df[f"{key}__doc_con"] = sup_doc, con_doc
        for mode in ("top3", "doc"):
            for r in evaluate(df, f"{key}__{mode}_sup"):
                agg.append(dict(judge=key, model=spec["name"], mode=mode, **r))
        print(f"[{key}] done in {time.time()-t0:.0f}s; pairs={len(df) + len(premises)}", flush=True)
        a = pd.DataFrame(agg)
        print(a[(a.judge == key) & (a.group == "all")][["mode", "auroc", "kappa", "precision", "recall", "flag_rate", "tau_from_other_subset"]].round(3).to_string(index=False), flush=True)
        del J; gc.collect(); (torch.mps.empty_cache() if DEVICE == "mps" else None)
        out = pd.concat([prev, pd.DataFrame(agg)], ignore_index=True) if prev is not None else pd.DataFrame(agg)
        out.to_csv(RESULTS / "judge_candidates.csv", index=False)
        df.drop(columns=["ctx_sents"]).to_csv(PRIVATE / "judge_candidates_sentences.csv", index=False)
    print("\n=== summary (all sentences) ===")
    out = pd.read_csv(RESULTS / "judge_candidates.csv")
    print(out[out.group == "all"][["judge", "mode", "auroc", "kappa", "precision", "recall", "f1", "flag_rate", "expert_rate"]].round(3).to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
