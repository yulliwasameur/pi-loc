# PI-Loc — do prompt-injection detectors look at the injection?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24085151.svg)](https://doi.org/10.5281/zenodo.24085151)

Code, data and results for the paper:

> Y. Ameur and S. Bouzefrane, *Do Prompt-Injection Detectors Look at the Injection? A Localization-Faithful Explainability Audit of Agent Guardrails*, submitted to IEEE Access, 2026.

A **guardrail** is a small classifier placed in front of an AI agent. It reads a document (a web page, an e-mail, a tool output) and says whether it contains a **prompt injection**: a hidden instruction such as "ignore your previous instructions and…". This repository checks two things:

1. Does the guardrail still catch the injection when it is **hidden inside a normal document**?
2. When it flags a document, does its **explanation** point at the injected sentence, and does that explanation really drive the decision?

## What is inside

| Folder | Content |
|---|---|
| `piloc/build_data.py` | Builds the PI-Loc benchmark (820 documents, exact injection spans) from four public datasets at pinned revisions. |
| `piloc/audit.py` | Runs three open detectors and four explainers (occlusion, KernelSHAP, attention, gradient × input), then measures localization, faithfulness and an explanation-guided "sanitization". |
| `piloc/stats.py` | Computes every number of the paper, with bootstrap confidence intervals and paired Wilcoxon tests, into `results/summary.json`. |
| `piloc/figures.py` | Draws the three figures from the results. |
| `piloc/check_paper.py` | Writes `paper/numbers.tex` (one LaTeX macro per reported number) and checks that the manuscript contains no result typed by hand. |
| `data/` | Benchmark manifest, build statistics, and (see `DATA_CARD.md`) the benchmark itself. |
| `results/` | Raw per-item results and `summary.json`. |
| `figures/` | Figures used in the paper. |
| `paper/` | LaTeX source of the submitted manuscript. |
| `tests/` | Integrity and unit tests. |

## Reproduce everything

CPU only, about 2.5 hours on 2 cores, about 2 GB of downloads.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` rebuilds PI-Loc, checks its SHA-256, runs the audit, recomputes the statistics, redraws the figures and checks the manuscript. All randomness uses seed 47. The detectors and datasets are pinned to exact Hugging Face revisions in `revisions.json`.

## Detectors audited

| Name in the paper | Hugging Face model | Revision |
|---|---|---|
| protectai-deberta-v2 | `protectai/deberta-v3-base-prompt-injection-v2` | see `revisions.json` |
| deepset-deberta | `deepset/deberta-v3-base-injection` | see `revisions.json` |
| fmops-distilbert | `fmops/distilbert-prompt-injection` | see `revisions.json` |

## Ethics

No new attack string was written: every injection payload comes from a public, labelled dataset. The work audits defences; it does not make attacks stronger.

## Licenses

- Code: MIT (`LICENSE`).
- Results, figures, documentation, spans and labels: CC BY 4.0 (`LICENSE-DATA`).
- Text reused from third-party datasets keeps its own license; see `DATA_CARD.md`.

## How this work was produced

An AI assistant (Claude, Anthropic) wrote most of the code and ran the experiments under the author's direction; the author reviewed the code and the results. See the Acknowledgment of the paper.

## Citation

See `CITATION.cff`.
