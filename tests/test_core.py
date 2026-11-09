"""Unit tests for the helpers used by the audit (no model needed)."""
import numpy as np
from piloc.audit import word_spans, drop_words, pool
from piloc.stats import wilson, boot_ci


def test_word_spans_roundtrip():
    t = "Hello world, this is  PI-Loc."
    ws = word_spans(t)
    assert [t[a:b] for a, b in ws] == ["Hello", "world,", "this", "is", "PI-Loc."]


def test_drop_words():
    t = "a b c d"
    ws = word_spans(t)
    assert drop_words(t, ws, [True, False, True, False]) == "a c"


def test_pool_max():
    assert pool(np.array([0.1, 0.5, 0.2, 0.9]), [-1, 0, 0, 1], 2).tolist() == [0.5, 0.9]


def test_wilson_bounds():
    lo, hi = wilson(47, 50)
    assert 0.8 < lo < 0.94 < hi < 1.0


def test_boot_ci_contains_mean():
    x = np.linspace(0, 1, 50)
    lo, hi = boot_ci(x)
    assert lo < x.mean() < hi
