#!/usr/bin/env python3
"""
minicheck_server.py — local scoring server for Bespoke-MiniCheck-7B (Tang, Laban and Durrett, 2024; Bespoke Labs, 2024)
running on Apple silicon through MLX.  Runs in the Python 3.13 environment (.venv-llm); the evaluation pipeline
(.venv, Python 3.9) talks to it over localhost, so no text ever leaves the machine.

The model answers "Yes" or "No" to the MiniCheck prompt; the support score is the probability mass of the first
generated token on "yes" (case-insensitive), exactly as in the MiniCheck reference implementation.  Pairs that
share a document reuse the document's key-value cache, so scoring many claims against one hospital course costs
one prefill of the course plus a few dozen tokens per claim.

Usage:  .venv-llm/bin/python minicheck_server.py --model ~/Desktop/Thesis/models/bespoke-minicheck-7b-mlx-8bit --port 8081
API:    POST /score  {"docs": [...], "claims": [...]}  ->  {"p_yes": [...], "p_no": [...]}
        GET  /health
"""
import argparse
import json
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from tokenizers import Tokenizer

# InternLM2 chat format (identical to the model's chat template); the BOS token is added by the tokenizer's post-processor.
# The tokenizer is read directly from tokenizer.json with the `tokenizers` library because the model's remote tokenizer
# class splits text into single characters under transformers 5.
CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

SYSTEM_PROMPT = ("Determine whether the provided claim is consistent with the corresponding document. Consistency in this context "
                 "implies that all information presented in the claim is substantiated by the document. If not, it should be "
                 "considered inconsistent. Please assess the claim's consistency with the document by responding with either "
                 "\"Yes\" or \"No\".")
USER_PROMPT = "Document: {doc}\nClaim: {claim}"
MAX_DOC_TOKENS = 30000
PREFILL_CHUNK = 512


class Scorer:
    def __init__(self, path):
        self.model, _ = load(path, tokenizer_config={"trust_remote_code": True})
        self.tk = Tokenizer.from_file(str(path).rstrip("/") + "/tokenizer.json")
        self.yes_ids, self.no_ids = [], []
        for tid in range(self.tk.get_vocab_size()):
            d = self.tk.decode([tid]).strip().lower()
            if d == "yes":
                self.yes_ids.append(tid)
            elif d == "no":
                self.no_ids.append(tid)
        self.yes_ids, self.no_ids = mx.array(self.yes_ids), mx.array(self.no_ids)
        probe = self.prompt_ids("The patient was given aspirin.", "The patient received aspirin.")
        print(f"model loaded; {len(self.yes_ids)} 'yes' tokens, {len(self.no_ids)} 'no' tokens; probe prompt {len(probe)} tokens, "
              f"tail {self.tk.decode(probe[-12:])!r}", flush=True)

    def prompt_ids(self, doc, claim):
        text = CHAT_TEMPLATE.format(system=SYSTEM_PROMPT, user=USER_PROMPT.format(doc=doc, claim=claim))
        return list(self.tk.encode(text).ids)

    def truncate_doc(self, doc):
        ids = self.tk.encode(doc, add_special_tokens=False).ids
        if len(ids) > MAX_DOC_TOKENS:
            return self.tk.decode(ids[:MAX_DOC_TOKENS])
        return doc

    def _forward(self, ids, cache):
        out = None
        for s in range(0, len(ids), PREFILL_CHUNK):
            out = self.model(mx.array(ids[s:s + PREFILL_CHUNK])[None], cache=cache)
            mx.eval(out)
        return out

    def score_group(self, doc, claims):
        doc = self.truncate_doc(doc)
        prompts = [self.prompt_ids(doc, c) for c in claims]
        n = len(prompts[0])
        for p in prompts[1:]:
            k = 0
            while k < min(n, len(p)) and p[k] == prompts[0][k]:
                k += 1
            n = min(n, k)
        n = max(0, n - 1)   # keep the last shared token in the suffix so the boundary is always re-processed
        cache = make_prompt_cache(self.model)
        if n:
            self._forward(prompts[0][:n], cache)
        res = []
        for p in prompts:
            suffix = p[n:]
            logits = self._forward(suffix, cache)[0, -1, :].astype(mx.float32)
            lp = logits - mx.logsumexp(logits)
            py = mx.exp(lp[self.yes_ids]).sum().item(); pn = mx.exp(lp[self.no_ids]).sum().item()
            res.append((py, pn))
            trim_prompt_cache(cache, len(suffix))
        return res

    def score(self, docs, claims):
        groups = defaultdict(list)
        for i, (d, c) in enumerate(zip(docs, claims)):
            groups[d].append(i)
        py = [0.0] * len(docs); pn = [0.0] * len(docs)
        for d, idx in groups.items():
            for i, (a, b) in zip(idx, self.score_group(d, [claims[i] for i in idx])):
                py[i], pn[i] = a, b
        return py, pn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    scorer = Scorer(args.model)
    t0 = time.time(); py, pn = scorer.score(["The patient was given aspirin 81 mg daily and discharged home."] * 2,
                                            ["The patient received aspirin.", "The patient was given warfarin."])
    print(f"self-test: supported claim p_yes={py[0]:.3f}, unsupported claim p_yes={py[1]:.3f} ({time.time()-t0:.1f}s)", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            body = json.dumps({"status": "ok", "model": args.model}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
            try:
                py, pn = scorer.score(req["docs"], req["claims"]); body = json.dumps({"p_yes": py, "p_no": pn}).encode(); code = 200
            except Exception as exc:  # noqa: BLE001
                body = json.dumps({"error": str(exc)}).encode(); code = 500
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    print(f"serving on http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
