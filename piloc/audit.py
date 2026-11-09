"""Audit three prompt-injection detectors on PI-Loc.

Steps
  1. detection: P(injection) for every item and every detector
  2. matched set: 50 injections caught by protectai (seeded sample)
  3. explanations: occlusion, KernelSHAP, attention, gradient x input (word level)
  4. localization and faithfulness metrics per item
  5. sanitization on the anchor detector

Everything runs on CPU and is deterministic for a given seed.
Outputs go to results/ as JSON.
"""
import argparse, json, math, os, random, time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REV = json.load(open(os.path.join(os.path.dirname(__file__), "..", "revisions.json")))
DETECTORS = {
    "protectai-deberta-v2": "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset-deberta": "deepset/deberta-v3-base-injection",
    "fmops-distilbert": "fmops/distilbert-prompt-injection",
}
ANCHOR = "protectai-deberta-v2"
EXPLAINERS = ["occlusion", "kernelshap", "attention", "gradxinput"]
MAXLEN = 512
TOPFRAC = 0.20


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Detector:
    def __init__(self, name):
        repo = DETECTORS[name]
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(repo, revision=REV[repo])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=REV[repo], attn_implementation="eager").eval()
        # all three models use label index 1 for the injection class (checked in tests)
        self.inj = 1

    @torch.inference_mode()
    def prob(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            x = self.tok(texts[i:i + bs], return_tensors="pt", padding=True,
                         truncation=True, max_length=MAXLEN)
            out.append(self.model(**x).logits.float().softmax(-1)[:, self.inj])
        return torch.cat(out).numpy()

    # ---- word <-> token mapping -------------------------------------------------
    def token_word_map(self, text, words):
        """Return encoding and, for each token, the index of the word it belongs to (or -1)."""
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=MAXLEN,
                       return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        wmap = []
        for (a, b) in offs:
            w = -1
            if b > a:
                for k, (ws, we) in enumerate(words):
                    if a < we and b > ws:
                        w = k
                        break
            wmap.append(w)
        return enc, wmap

    def attention_scores(self, text, words):
        enc, wmap = self.token_word_map(text, words)
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        att = out.attentions[-1][0].mean(0)[0]  # last layer, mean over heads, from position 0
        return pool(att.numpy(), wmap, len(words))

    def gradxinput_scores(self, text, words):
        enc, wmap = self.token_word_map(text, words)
        emb = self.model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        kw = {k: v for k, v in enc.items() if k != "input_ids"}
        p = self.model(inputs_embeds=emb, **kw).logits.float().softmax(-1)[0, self.inj]
        p.backward()
        s = (emb.grad[0] * emb[0]).sum(-1).abs().detach().numpy()
        return pool(s, wmap, len(words))


def pool(tok_scores, wmap, n):
    w = np.zeros(n)
    for s, k in zip(tok_scores, wmap):
        if k >= 0:
            w[k] = max(w[k], s)
    return w


def word_spans(text):
    """Whitespace words with character offsets."""
    out, i = [], 0
    for w in text.split(" "):
        if w:
            out.append((i, i + len(w)))
        i += len(w) + 1
    return out


def drop_words(text, words, keep_mask):
    return " ".join(text[a:b] for (a, b), k in zip(words, keep_mask) if k)


def occlusion_scores(det, text, words, p0):
    variants = [drop_words(text, words, [j != i for j in range(len(words))]) for i in range(len(words))]
    return p0 - det.prob(variants, bs=32)


def kernelshap_scores(det, text, words, n_coal, rng):
    """KernelSHAP with sampled coalitions and weighted ridge regression."""
    n = len(words)
    Z = []
    for _ in range(n_coal):
        k = rng.randint(1, n - 1)
        on = rng.sample(range(n), k)
        z = np.zeros(n); z[on] = 1
        Z.append(z)
    Z = np.array(Z)
    texts = [drop_words(text, words, z.astype(bool)) for z in Z]
    y = det.prob(texts, bs=32)
    k = Z.sum(1)
    wts = (n - 1) / (np.array([math.comb(n, int(x)) for x in k], dtype=float) * k * (n - k))
    wts = wts / wts.max()
    X = np.hstack([np.ones((len(Z), 1)), Z])
    W = np.diag(wts)
    lam = 1e-3
    beta = np.linalg.solve(X.T @ W @ X + lam * np.eye(n + 1), X.T @ W @ y)
    return beta[1:]


def auprc(scores, mask):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(mask, scores))


def evaluate_item(det, it, expl, p0):
    text = it["text"]; words = word_spans(text)
    s, e = it["span"]
    mask = np.array([1 if (a < e and b > s) else 0 for a, b in words])
    n = len(words); kspan = int(mask.sum())
    order = np.argsort(-expl, kind="stable")
    ktop = max(1, math.ceil(TOPFRAC * n))
    top = np.zeros(n, bool); top[order[:ktop]] = True
    p_removed = float(det.prob([drop_words(text, words, ~top)])[0])
    p_kept = float(det.prob([drop_words(text, words, top)])[0])
    return dict(
        auprc=auprc(expl, mask),
        p_at_span=float(mask[order[:kspan]].mean()),
        pointing=int(mask[order[0]]),
        frac_in_span=float(mask.mean()),
        comprehensiveness=p0 - p_removed,
        retained=p_kept,
    )


def sanitize(det, it, expl, p0, k):
    text = it["text"]; words = word_spans(text)
    s, e = it["span"]
    mask = np.array([1 if (a < e and b > s) else 0 for a, b in words])
    n = len(words)
    order = np.argsort(-expl, kind="stable")
    m = max(1, math.ceil(k * n))
    rm = np.zeros(n, bool); rm[order[:m]] = True
    p_after = float(det.prob([drop_words(text, words, ~rm)])[0])
    return dict(p_before=p0, p_after=p_after, drop=p0 - p_after,
                span_removed=float(rm[mask == 1].mean()),
                collateral=float(rm[mask == 0].mean()))


def main(seed=47, n_match=50, data="data/piloc.jsonl", out="results", threads=2):
    set_seed(seed)
    torch.set_num_threads(threads)
    os.makedirs(out, exist_ok=True)
    items = [json.loads(l) for l in open(data)]
    log = {"seed": seed, "n_match": n_match, "torch": torch.__version__, "start": time.time()}

    # 1. detection ------------------------------------------------------------------
    dets = {k: Detector(k) for k in DETECTORS}
    det_path = os.path.join(out, "detection_scores.json")
    if os.path.exists(det_path):
        scores = json.load(open(det_path))
    else:
        scores = {k: [float(x) for x in d.prob([it["text"] for it in items])] for k, d in dets.items()}
        json.dump(scores, open(det_path, "w"))
    # 2. matched set ------------------------------------------------------------------
    caught = [i for i, it in enumerate(items) if it["cls"] == "inj" and scores[ANCHOR][i] >= 0.5]
    rng = random.Random(seed)
    match = sorted(rng.sample(caught, n_match))
    json.dump({"caught_by_anchor": len(caught), "matched_ids": [items[i]["id"] for i in match]},
              open(os.path.join(out, "matched_set.json"), "w"), indent=1)

    # 3-4. explanations + metrics -----------------------------------------------------
    pi_path = os.path.join(out, "explain_per_item.json")
    ex_path = os.path.join(out, "anchor_explanations.json")
    per_item = json.load(open(pi_path)) if os.path.exists(pi_path) else []
    expl_store = json.load(open(ex_path)) if os.path.exists(ex_path) else {}
    done = {(r["detector"], r["id"]) for r in per_item}
    for dname, det in dets.items():
        for i in match:
            it = items[i]
            if (dname, it["id"]) in done:
                continue
            words = word_spans(it["text"])
            p0 = scores[dname][i]
            krng = random.Random(seed * 1000 + i)
            ncoal = 96 if dname == ANCHOR else 48
            ex = {
                "occlusion": occlusion_scores(det, it["text"], words, p0),
                "kernelshap": kernelshap_scores(det, it["text"], words, ncoal, krng),
                "attention": det.attention_scores(it["text"], words),
                "gradxinput": det.gradxinput_scores(it["text"], words),
            }
            for en, v in ex.items():
                r = evaluate_item(det, it, v, p0)
                r.update(detector=dname, explainer=en, id=it["id"], p0=p0)
                per_item.append(r)
                if dname == ANCHOR:
                    expl_store[f"{it['id']}|{en}"] = [float(x) for x in v]
            # checkpoint after every item so that an interrupted run can resume
            json.dump(per_item, open(pi_path, "w"))
            json.dump(expl_store, open(ex_path, "w"))
            print(f"[{time.strftime('%H:%M:%S')}] {dname} {it['id']} done", flush=True)

    # 5. sanitization -------------------------------------------------------------------
    san = []
    det = dets[ANCHOR]
    for i in match:
        it = items[i]; p0 = scores[ANCHOR][i]
        for en in ["occlusion", "attention"]:
            for k in [0.10, 0.20]:
                r = sanitize(det, it, np.array(expl_store[f"{it['id']}|{en}"]), p0, k)
                r.update(id=it["id"], explainer=en, k=k)
                san.append(r)
    json.dump(san, open(os.path.join(out, "sanitization_per_item.json"), "w"))
    log["end"] = time.time()
    json.dump(log, open(os.path.join(out, "run_log.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--n-match", type=int, default=50)
    ap.add_argument("--data", default="data/piloc.jsonl")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    main(a.seed, a.n_match, a.data, a.out, a.threads)
