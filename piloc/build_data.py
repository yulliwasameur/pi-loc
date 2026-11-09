"""Build the PI-Loc benchmark.

PI-Loc models indirect prompt injection: a benign document (a "carrier") with an
injected instruction spliced at a sentence boundary. Every injection has an exact
character span. The builder is deterministic for a given seed and pinned dataset
revisions.

Outputs
  data/piloc.jsonl           full benchmark (text + labels + spans)
  data/piloc_manifest.jsonl  pointers only (source ids, offsets), no text
  data/build_stats.json      counts and SHA-256 of piloc.jsonl
"""
import argparse, hashlib, json, os, random, re

from datasets import load_dataset
from langdetect import DetectorFactory, detect

REV = json.load(open(os.path.join(os.path.dirname(__file__), "..", "revisions.json")))

# Words that make a benign sentence look like an injection. Used twice:
# carriers must NOT contain them, hard negatives MUST contain them.
TRIGGER = re.compile(
    r"\b(ignore|ignoring|disregard|forget|instruction|instructions|override|"
    r"system prompt|prompt|previous|jailbreak|pretend|act as|you are now|bypass|"
    r"reveal|password|rules)\b", re.I)

SENT_END = re.compile(r"(?<=[.!?])\s+")


def n_words(s):
    return len(s.split())


def is_english(s):
    try:
        return detect(s) == "en"
    except Exception:
        return False


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_payloads():
    """Injection payloads (label 1) and benign trigger sentences for hard negatives.

    Payloads: positive rows of the two public injection corpora.
    Hard-negative sentences: declarative sentences from SQuAD v1.1 passages that
    contain trigger lexicon (e.g. "previous", "rules", "ignore"). They are benign
    by construction. Their passages contain a trigger word, so they can never be
    used as carriers (carriers contain none).
    """
    pos, neg = {}, {}
    for name in ["deepset/prompt-injections", "xTRam1/safe-guard-prompt-injection"]:
        d = load_dataset(name, revision=REV[name])
        for split in sorted(d.keys()):
            for i, row in enumerate(d[split]):
                t = norm(row["text"])
                if row["label"] == 1 and 5 <= n_words(t) <= 55:
                    pos.setdefault(t, {"dataset": name, "split": split, "row": i})
    sqd = load_dataset("rajpurkar/squad", revision=REV["rajpurkar/squad"])
    for split in ["train", "validation"]:
        seen = set()
        for i, row in enumerate(sqd[split]):
            c = norm(row["context"])
            if c in seen or not TRIGGER.search(c):
                continue  # passages without trigger words stay available as carriers
            seen.add(c)
            for j, sent in enumerate(SENT_END.split(c)):
                if 8 <= n_words(sent) <= 40 and TRIGGER.search(sent):
                    neg.setdefault(sent, {"dataset": "rajpurkar/squad", "split": split,
                                          "row": i, "sentence": j})
    DetectorFactory.seed = 0
    pos = {t: s for t, s in sorted(pos.items()) if is_english(t)}
    return pos, neg


def load_carriers():
    out = {"wiki": {}, "email": {}}
    sq = load_dataset("rajpurkar/squad", revision=REV["rajpurkar/squad"])["train"]
    for i, row in enumerate(sq):
        c = norm(row["context"])
        if 70 <= n_words(c) <= 180 and not TRIGGER.search(c):
            out["wiki"].setdefault(c, {"dataset": "rajpurkar/squad", "split": "train", "row": i})
    ae = load_dataset("Yale-LILY/aeslc", revision=REV["Yale-LILY/aeslc"])["train"]
    for i, row in enumerate(ae):
        c = norm(row["email_body"])
        if 70 <= n_words(c) <= 180 and not TRIGGER.search(c):
            out["email"].setdefault(c, {"dataset": "Yale-LILY/aeslc", "split": "train", "row": i})
    return {k: sorted(v.items()) for k, v in out.items()}


def splice(carrier, insert, rng):
    """Insert `insert` at a sentence boundary of `carrier`; return text and span."""
    sents = SENT_END.split(carrier)
    k = rng.randint(0, len(sents))  # 0 = before first sentence, len = after last
    left = " ".join(sents[:k])
    right = " ".join(sents[k:])
    prefix = left + (" " if left else "")
    text = prefix + insert + (" " + right if right else "")
    s = len(prefix)
    e = s + len(insert)
    assert text[s:e] == insert
    return text, s, e, k / max(1, len(sents))


def build(seed=47, n_inj=520, n_hn=180, n_clean=120, outdir="data"):
    rng = random.Random(seed)
    pos, neg = load_payloads()
    carriers = load_carriers()
    pos_items = sorted(pos.items())
    neg_items = sorted(neg.items())
    rng.shuffle(pos_items)
    rng.shuffle(neg_items)
    pools = {k: list(v) for k, v in carriers.items()}
    for k in pools:
        rng.shuffle(pools[k])
    need = n_inj + n_hn + n_clean
    assert len(pos_items) >= n_inj and len(neg_items) >= n_hn
    assert sum(len(v) for v in pools.values()) >= need

    def next_carrier(i):
        fam = "wiki" if i % 2 == 0 else "email"
        c, src = pools[fam].pop()
        return fam, c, src

    items, idx = [], 0
    for j in range(n_inj):
        fam, c, csrc = next_carrier(idx); idx += 1
        p, psrc = pos_items[j]
        text, s, e, rel = splice(c, p, rng)
        items.append(dict(id=f"inj-{j:03d}", cls="inj", label=1, family=fam, text=text,
                          span=[s, e], rel_pos=round(rel, 4), carrier_src=csrc, insert_src=psrc))
    for j in range(n_hn):
        fam, c, csrc = next_carrier(idx); idx += 1
        p, psrc = neg_items[j]
        text, s, e, rel = splice(c, p, rng)
        items.append(dict(id=f"hn-{j:03d}", cls="hn", label=0, family=fam, text=text,
                          span=None, insert_span=[s, e], rel_pos=round(rel, 4),
                          carrier_src=csrc, insert_src=psrc))
    for j in range(n_clean):
        fam, c, csrc = next_carrier(idx); idx += 1
        items.append(dict(id=f"clean-{j:03d}", cls="clean", label=0, family=fam, text=c,
                          span=None, carrier_src=csrc, insert_src=None))

    # integrity checks
    inj_payloads = [it["text"][it["span"][0]:it["span"][1]] for it in items if it["cls"] == "inj"]
    assert len(set(inj_payloads)) == n_inj, "payloads must be pairwise distinct"
    assert len({json.dumps(it["carrier_src"], sort_keys=True) for it in items}) == need

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "piloc.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    with open(os.path.join(outdir, "piloc_manifest.jsonl"), "w") as f:
        for it in items:
            m = {k: v for k, v in it.items() if k != "text"}
            f.write(json.dumps(m, sort_keys=True) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    stats = dict(seed=seed, n_items=len(items), n_inj=n_inj, n_hn=n_hn, n_clean=n_clean,
                 distinct_positive_payloads=len(pos), benign_trigger_sentences_available=len(neg),
                 carriers_available={k: len(v) for k, v in carriers.items()},
                 sha256_piloc_jsonl=sha, revisions=REV)
    json.dump(stats, open(os.path.join(outdir, "build_stats.json"), "w"), indent=1)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    print(json.dumps(build(a.seed, outdir=a.out), indent=1))
