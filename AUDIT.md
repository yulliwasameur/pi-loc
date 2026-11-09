# Audit of the PI-Loc artefacts (09/11/2026)

## Why the code was rebuilt

The first version of this study (2 October 2026) was produced in an AI-assisted session. Its archive (code, data, raw results) was lost. Code that cannot be shown cannot be cited, so the whole study was **re-implemented from the method described in the paper and re-run** on 9 November 2026. The paper now reports only the numbers of this new run. The October numbers are listed below for transparency; they are not used anywhere.

## Checks performed

| # | Check | Result |
|---|---|---|
| 1 | Benchmark rebuilt in a fresh process gives the same file (SHA-256 of `data/piloc.jsonl`) | identical |
| 2 | Audit re-run in a fresh process on two matched items, all four explainers | every metric identical |
| 3 | Detection re-run on 20 items in different batches | max difference 4e-11 (batch padding only) |
| 4 | Integrity tests (`tests/test_data.py`): counts, exact spans, distinct payloads, unique and trigger-free carriers, hard negatives contain trigger words, manifest without text | pass |
| 5 | Unit tests (`tests/test_core.py`) and model label tests (`tests/test_models.py`) | pass (14 tests) |
| 6 | Every result in the manuscript comes from `paper/numbers.tex`, generated from `results/summary.json` (`piloc/check_paper.py`) | pass |
| 7 | Longest document: 361 tokens (DeBERTa), 396 (DistilBERT), below the 512-token limit | pass |
| 8 | Calibrated threshold really keeps false positives at or below 5% | 4.7% for all three detectors |
| 9 | Licenses of models and datasets read from their Hugging Face cards | two datasets without a stated license, see `DATA_CARD.md` |
| 10 | References of the paper checked against arXiv (titles, authors) | corrected TracLLM and AttnTrace author lists |

## What changed compared with the lost October version

| Measure | 02/10 (lost archive) | 09/11 (this run) |
|---|---|---|
| protectai ROC-AUC | 0.906 | 0.893 |
| protectai recall (seuil 0,5) | 0.471 | 0.471 |
| protectai recall à FPR ≤ 5 % | 0.619 | 0.610 |
| deepset ROC-AUC | 0.631 | 0.766 |
| deepset recall à FPR ≤ 5 % | 0.054 | 0.258 |
| fmops ROC-AUC | 0.515 | 0.749 |
| fmops recall à FPR ≤ 5 % | 0.021 | 0.223 |
| AUPRC du hasard | 0.144 | 0.129 |
| Occlusion AUPRC | 0.456 | 0.444 |
| Attention AUPRC | 0.604 | 0.636 |
| Occlusion pointing | 0.940 | 0.980 |
| Attention pointing | 0.400 | 0.420 |
| Occlusion comprehensiveness | 0.925 | 0.981 |
| Attention comprehensiveness | 0.938 | 0.958 |
| KernelSHAP AUPRC | 0.251 | 0.185 |
| Grad×input AUPRC | 0.409 | 0.389 |
| Assainissement attention 20 % : P après | 0.027 | 0.023 |
| Assainissement attention 20 % : dégâts | 0.114 | 0.121 |
| Assainissement occlusion 20 % : dégâts | 0.164 | 0.168 |
| Payloads distincts disponibles | 2 642 | 2 665 |
The main findings hold: protectai misses about half of embedded injections, the two other detectors flag everything, attention covers the span best and occlusion pinpoints it best, and explanations of the saturated detectors are not faithful.

Two statements of the October text had to change:
- deepset and fmops are **not** near chance. Their ranking carries some signal (ROC-AUC about 0.75), but at a 5% false-positive budget they still miss about three injections out of four.
- Occlusion is now **slightly more faithful** than attention (the October text said the opposite).

## Known differences in method details

The October description did not fix every detail. The rebuild makes these choices explicit in the code:
- hard-negative sentences come from SQuAD passages that contain a trigger word (8 to 40 words);
- English payloads are selected with `langdetect` (seeded);
- carriers alternate between the two families;
- KernelSHAP uses ridge regularization 1e-3 and samples coalition sizes uniformly.
