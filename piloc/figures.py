"""Draw the three figures of the paper from results/ (no number is typed by hand)."""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "svg.hashsalt": "piloc", "path.simplify": False})
DET = ["protectai-deberta-v2", "deepset-deberta", "fmops-distilbert"]
EXP = ["occlusion", "kernelshap", "gradxinput", "attention"]
LAB = {"occlusion": "Occlusion", "kernelshap": "KernelSHAP", "gradxinput": "Grad.$\\times$input",
       "attention": "Attention"}
DARK, MID, LIGHT, ACC = "#1f4e79", "#9dc3e6", "#c9d9ea", "#c55a11"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.5); ax.set_axisbelow(True)


def main(res="results", out="figures"):
    os.makedirs(out, exist_ok=True)
    S = json.load(open(os.path.join(res, "summary.json")))
    meta = {"metadata": {"CreationDate": None}}

    # Fig. 1 — detection
    cols = [("TPR (inj.)", "tpr_inj", DARK), ("FPR (hard neg.)", "fpr_hn", MID),
            ("FPR (clean)", "fpr_clean", LIGHT), ("TPR@FPR=0.05", "tpr_at_fpr05", ACC)]
    fig, ax = plt.subplots(figsize=(3.45, 1.9)); x = np.arange(3); w = 0.2
    for i, (lab, key, c) in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, [S["detection"][d][key] for d in DET], w, label=lab, color=c,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([d.replace("-", "-\n", 1) for d in DET])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Rate"); style(ax)
    ax.legend(ncol=2, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_detection.pdf"), **meta); plt.close(fig)

    # Fig. 2 — localization AUPRC with CIs on the anchor detector
    E = S["explain"]["protectai-deberta-v2"]
    m = np.array([E[e]["auprc"] for e in EXP]); lo = np.array([E[e]["auprc_ci"][0] for e in EXP])
    hi = np.array([E[e]["auprc_ci"][1] for e in EXP]); ch = E["chance_auprc"]
    fig, ax = plt.subplots(figsize=(3.45, 1.7)); x = np.arange(4)
    ax.bar(x, m, 0.55, color=DARK, edgecolor="black", linewidth=0.3)
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", capsize=2.5, linewidth=0.6)
    ax.axhline(ch, ls="--", color=ACC, linewidth=0.8, label=f"chance AUPRC = {ch:.3f}")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_xticks(x); ax.set_xticklabels([LAB[e] for e in EXP]); ax.set_ylabel("Localization AUPRC")
    ax.set_ylim(0, max(0.75, hi.max() + 0.08)); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_loc.pdf"), **meta); plt.close(fig)

    # Fig. 3 — per-item spread
    P = json.load(open(os.path.join(res, "explain_per_item.json")))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.8))
    rng = np.random.default_rng(47)
    for ax, key, title in [(axs[0], "auprc", "Localization AUPRC"), (axs[1], "comprehensiveness", "Comprehensiveness")]:
        for j, e in enumerate(EXP):
            v = [r[key] for r in P if r["detector"] == "protectai-deberta-v2" and r["explainer"] == e]
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=3, color=DARK, alpha=0.6, linewidths=0)
            ax.plot([j - 0.25, j + 0.25], [np.mean(v)] * 2, color=ACC, linewidth=1.2)
        if key == "auprc":
            ax.axhline(ch, ls="--", color="grey", linewidth=0.6)
        ax.set_xticks(range(4)); ax.set_xticklabels(["Occ.", "SHAP", "G$\\times$I", "Attn."])
        ax.set_title(title, fontsize=7); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_dist.pdf"), **meta); plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
