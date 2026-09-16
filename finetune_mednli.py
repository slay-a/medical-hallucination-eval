#!/usr/bin/env python3
"""
finetune_mednli.py — Adapt an NLI cross-encoder to clinical language with MedNLI (PhysioNet, credentialed).

MedNLI (Romanov & Shivade, 2018): 11,232 train / 1,395 dev / 1,422 test premise-hypothesis pairs written by
clinicians from MIMIC-III notes, labels entailment / contradiction / neutral.  The data stay on this machine.

Default base model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (already a strong general NLI model);
the fine-tuned model is saved OUTSIDE the repository (~/Desktop/Thesis/models/...) and can then be scored against
the ann-pt-summ expert labels with:  python judge_candidates.py --only mednli_deberta_large

Usage: python finetune_mednli.py [--base MODEL] [--epochs 2] [--lr 1e-5] [--batch 4] [--grad-accum 4] [--max-len 256] [--freeze-embeddings]
"""
import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["entailment", "neutral", "contradiction"]
DATA_ROOT = Path.home() / "Desktop" / "Thesis" / "mednli"
OUT_DEFAULT = Path.home() / "Desktop" / "Thesis" / "models" / "mednli-deberta-v3-large"
DEVICE = os.environ.get("FT_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")   # FT_DEVICE=cpu forces the CPU


def load_split(name):
    f = glob.glob(str(DATA_ROOT / "**" / f"mli_{name}_v1.jsonl"), recursive=True)[0]
    rows = [json.loads(l) for l in open(f)]
    return [(r["sentence1"], r["sentence2"], r["gold_label"]) for r in rows if r["gold_label"] in LABELS]


class PairDS(Dataset):
    def __init__(self, rows, tok, max_len, label2id):
        self.rows, self.tok, self.max_len, self.label2id = rows, tok, max_len, label2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, h, l = self.rows[i]
        return p, h, self.label2id[l]


def collate(batch, tok, max_len):
    p, h, y = zip(*batch)
    enc = tok(list(p), list(h), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y)
    return enc


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = n = 0
    for enc in loader:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**{k: v for k, v in enc.items() if k != "labels"}).logits
        correct += int((logits.argmax(-1) == enc["labels"]).sum()); n += len(enc["labels"])
    model.train()
    return correct / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4, help="micro-batch size (memory-bound on a 24 GB Apple GPU)")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step (effective batch = batch x grad-accum)")
    ap.add_argument("--freeze-embeddings", action="store_true", help="do not update the 128k-token embedding matrix (saves ~1.5 GB of optimizer state)")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base)
    # map the base model's label names onto MedNLI's labels (keeps the pre-trained classifier head)
    base_id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    if set(base_id2label.values()) >= set(LABELS):
        label2id = {v: k for k, v in base_id2label.items() if v in LABELS}
    else:
        label2id = {l: i for i, l in enumerate(LABELS)}
    model.config.id2label = {i: l for l, i in label2id.items()}; model.config.label2id = label2id
    model.to(DEVICE).train()
    print(f"device={DEVICE} base={args.base} label2id={label2id}", flush=True)

    train, dev, test = load_split("train"), load_split("dev"), load_split("test")
    print(f"MedNLI train={len(train)} dev={len(dev)} test={len(test)}", flush=True)
    mk = lambda rows, shuffle: DataLoader(PairDS(rows, tok, args.max_len, label2id), batch_size=args.batch, shuffle=shuffle,
                                          collate_fn=lambda b: collate(b, tok, args.max_len))
    tr, dv, te = mk(train, True), mk(dev, False), mk(test, False)
    print(f"zero-shot dev accuracy: {evaluate(model, dv):.4f}  test: {evaluate(model, te):.4f}", flush=True)

    if args.freeze_embeddings:
        for prm in model.get_input_embeddings().parameters():
            prm.requires_grad_(False)
    params = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    accum = max(1, args.grad_accum)
    steps = args.epochs * math.ceil(len(tr) / accum)          # optimizer steps
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, out = -1.0, Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); step = 0; micro = 0
    print(f"micro-batch {args.batch} x accumulation {accum} = effective batch {args.batch * accum}; {steps} optimizer steps", flush=True)
    for ep in range(args.epochs):
        n_micro = len(tr)
        for i, enc in enumerate(tr):
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            loss = model(**enc).loss / accum
            loss.backward(); micro += 1
            if micro % accum == 0 or i == n_micro - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print(f"  epoch {ep+1} step {step}/{steps} loss {loss.item() * accum:.4f} ({time.time()-t0:.0f}s)", flush=True)
        acc_dev, acc_test = evaluate(model, dv), evaluate(model, te)
        print(f"epoch {ep+1}: dev accuracy {acc_dev:.4f}  test accuracy {acc_test:.4f}", flush=True)
        if acc_dev > best:
            best = acc_dev; model.save_pretrained(out); tok.save_pretrained(out)
            json.dump(dict(base=args.base, epochs=ep + 1, dev_accuracy=acc_dev, test_accuracy=acc_test, lr=args.lr,
                           batch=args.batch * accum, micro_batch=args.batch, grad_accum=accum, freeze_embeddings=args.freeze_embeddings,
                           train_minutes=round((time.time() - t0) / 60, 1)),
                      open(out / "mednli_finetune_summary.json", "w"), indent=1)
            print(f"  saved to {out}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
