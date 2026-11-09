"""Integrity checks on the built benchmark (fast, no model needed)."""
import hashlib, json, os, re
from piloc.build_data import TRIGGER

ROOT = os.path.dirname(os.path.dirname(__file__))
ITEMS = [json.loads(l) for l in open(os.path.join(ROOT, "data", "piloc.jsonl"))]
STATS = json.load(open(os.path.join(ROOT, "data", "build_stats.json")))


def test_counts():
    from collections import Counter
    c = Counter(it["cls"] for it in ITEMS)
    assert c == {"inj": 520, "hn": 180, "clean": 120}


def test_sha_matches_build_stats():
    sha = hashlib.sha256(open(os.path.join(ROOT, "data", "piloc.jsonl"), "rb").read()).hexdigest()
    assert sha == STATS["sha256_piloc_jsonl"]


def test_spans_exact_and_payloads_distinct():
    pay = []
    for it in ITEMS:
        if it["cls"] == "inj":
            s, e = it["span"]
            assert 0 <= s < e <= len(it["text"])
            pay.append(it["text"][s:e])
    assert len(set(pay)) == 520


def test_carriers_unique_and_trigger_free():
    srcs = [json.dumps(it["carrier_src"], sort_keys=True) for it in ITEMS]
    assert len(set(srcs)) == len(ITEMS)
    for it in ITEMS:
        if it["cls"] == "clean":
            assert not TRIGGER.search(it["text"])
        else:
            key = "span" if it["cls"] == "inj" else "insert_span"
            s, e = it[key]
            carrier = (it["text"][:s] + it["text"][e:]).strip()
            assert not TRIGGER.search(carrier)


def test_hard_negatives_contain_trigger():
    for it in ITEMS:
        if it["cls"] == "hn":
            s, e = it["insert_span"]
            assert TRIGGER.search(it["text"][s:e])


def test_manifest_has_no_text():
    for l in open(os.path.join(ROOT, "data", "piloc_manifest.jsonl")):
        assert "text" not in json.loads(l)
