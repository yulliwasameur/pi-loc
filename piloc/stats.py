"""Aggregate PI-Loc results into the numbers reported in the paper.

Reads  results/detection_scores.json, explain_per_item.json, sanitization_per_item.json
Writes results/summary.json  (single source for every number in the manuscript)
"""
import json, math, os, sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

B = 5000
SEED = 47


def boot_ci(x, seed=SEED, b=B):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(b, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def detection(items, scores):
    y = np.array([it["label"] for it in items])
    cls = np.array([it["cls"] for it in items])
    out = {}
    for d, s in scores.items():
        s = np.array(s)
        neg = s[y == 0]
        thr = float(np.quantile(neg, 0.95, method="higher"))
        # threshold so that at most 5% of negatives score strictly above it
        tpr_cal = float((s[cls == "inj"] > thr).mean())
        fpr_cal = float((neg > thr).mean())
        out[d] = dict(
            roc_auc=float(roc_auc_score(y, s)),
            tpr_inj=float((s[cls == "inj"] >= 0.5).mean()),
            fpr_hn=float((s[cls == "hn"] >= 0.5).mean()),
            fpr_clean=float((s[cls == "clean"] >= 0.5).mean()),
            tpr_at_fpr05=tpr_cal, fpr_at_cal=fpr_cal, threshold_cal=thr,
            by_family={f: float((s[(cls == "inj") & (np.array([it["family"] for it in items]) == f)] >= 0.5).mean())
                       for f in ["wiki", "email"]},
        )
    return out


def explain(per_item):
    out = {}
    dets = sorted({r["detector"] for r in per_item})
    exps = ["occlusion", "kernelshap", "gradxinput", "attention"]
    for d in dets:
        out[d] = {}
        for e in exps:
            rs = [r for r in per_item if r["detector"] == d and r["explainer"] == e]
            rs.sort(key=lambda r: r["id"])
            au = [r["auprc"] for r in rs]; co = [r["comprehensiveness"] for r in rs]
            pg = [r["pointing"] for r in rs]
            out[d][e] = dict(
                n=len(rs), auprc=float(np.mean(au)), auprc_ci=boot_ci(au),
                p_at_span=float(np.mean([r["p_at_span"] for r in rs])),
                pointing=float(np.mean(pg)), pointing_ci=wilson(sum(pg), len(pg)),
                comprehensiveness=float(np.mean(co)), comprehensiveness_ci=boot_ci(co),
                retained=float(np.mean([r["retained"] for r in rs])),
            )
        chance = [r["frac_in_span"] for r in per_item if r["detector"] == d and r["explainer"] == "occlusion"]
        out[d]["chance_auprc"] = float(np.mean(chance))
    return out


def paired_tests(per_item, d):
    def vec(e, key):
        rs = sorted([r for r in per_item if r["detector"] == d and r["explainer"] == e], key=lambda r: r["id"])
        return np.array([r[key] for r in rs], float)
    res = {}
    exps = ["occlusion", "kernelshap", "gradxinput", "attention"]
    for key in ["auprc", "pointing", "comprehensiveness"]:
        res[key] = {}
        for i, a in enumerate(exps):
            for b in exps[i + 1:]:
                x, y = vec(a, key), vec(b, key)
                if np.allclose(x, y):
                    p = 1.0
                else:
                    p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
                res[key][f"{a}_vs_{b}"] = dict(mean_diff=float((x - y).mean()), p=p)
    return res


def sanit(san):
    out = {}
    for e in ["occlusion", "attention"]:
        for k in [0.10, 0.20]:
            rs = [r for r in san if r["explainer"] == e and abs(r["k"] - k) < 1e-9]
            out[f"{e}@{int(k*100)}"] = {m: float(np.mean([r[m] for r in rs]))
                                       for m in ["p_before", "p_after", "drop", "span_removed", "collateral"]}
            out[f"{e}@{int(k*100)}"]["n"] = len(rs)
    return out


def main(data="data/piloc.jsonl", res="results"):
    items = [json.loads(l) for l in open(data)]
    scores = json.load(open(os.path.join(res, "detection_scores.json")))
    per_item = json.load(open(os.path.join(res, "explain_per_item.json")))
    san = json.load(open(os.path.join(res, "sanitization_per_item.json")))
    ms = json.load(open(os.path.join(res, "matched_set.json")))
    summary = dict(
        build=json.load(open(os.path.join(os.path.dirname(data), "build_stats.json"))),
        detection=detection(items, scores),
        matched=dict(n=len(ms["matched_ids"]), caught_by_anchor=ms["caught_by_anchor"]),
        explain=explain(per_item),
        tests=paired_tests(per_item, "protectai-deberta-v2"),
        sanitization=sanit(san),
    )
    json.dump(summary, open(os.path.join(res, "summary.json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    s = main(*sys.argv[1:])
    print(json.dumps(s["detection"], indent=1))
