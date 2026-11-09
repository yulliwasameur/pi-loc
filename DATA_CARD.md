# PI-Loc data card

## What PI-Loc is

PI-Loc is a small benchmark to test prompt-injection detectors in the **indirect** setting. Each document looks like what an AI agent reads from a tool, such as a web passage or an e-mail. Some documents contain an injected instruction. For each injection we know its **exact position**, to the character.

| Class | Items | Label | What it contains |
|---|---|---|---|
| `inj` | 520 | 1 | A benign carrier with an injection payload inserted at a sentence boundary. The `span` field gives `[start, end)` in characters. |
| `hn` (hard negative) | 180 | 0 | A benign carrier with a benign sentence that contains trigger words (for example "previous", "rules", "ignore"). It tests whether a detector reacts to words or to instructions. |
| `clean` | 120 | 0 | A benign carrier, unchanged. |

Carriers alternate between encyclopedic passages and e-mails (`family` = `wiki` or `email`). Each carrier is used once.

## How it is built

`python -m piloc.build_data --seed 47` rebuilds the benchmark from pinned dataset revisions (see `revisions.json`). The SHA-256 of `data/piloc.jsonl` is stored in `data/build_stats.json`. The tests check it.

1. **Payloads.** Positive (injection) rows of `deepset/prompt-injections` and `xTRam1/safe-guard-prompt-injection`, 5 to 55 words, deduplicated, kept only if `langdetect` says English. No attack string was written by us.
2. **Carriers.** SQuAD v1.1 training passages and AESLC training e-mails, 70 to 180 words, without any trigger word.
3. **Hard-negative sentences.** Sentences of 8 to 40 words, taken from SQuAD passages that contain a trigger word. These passages are therefore never carriers.
4. **Splicing.** A sentence boundary is drawn at random (seeded), and the payload or sentence is inserted there.

## Files

| File | Content |
|---|---|
| `data/piloc.jsonl` | The full benchmark, with text. |
| `data/piloc_manifest.jsonl` | The same items **without text**: source dataset, split, row, offsets. Use it with the builder if you cannot redistribute the text. |
| `data/build_stats.json` | Counts, seed, revisions, SHA-256. |

## Licenses of the sources

PI-Loc reuses text from four public datasets. Their licenses apply to the text in `piloc.jsonl`.

| Source | Use in PI-Loc | License (Hugging Face card, checked on 09/11/2026) |
|---|---|---|
| `deepset/prompt-injections` | payloads | Apache-2.0 |
| `xTRam1/safe-guard-prompt-injection` | payloads | **not stated** |
| `rajpurkar/squad` (v1.1) | carriers, hard-negative sentences | CC BY-SA 4.0 |
| `Yale-LILY/aeslc` | carriers | **unknown** (e-mails from the public Enron corpus) |

The GitHub repository publishes the manifest and the builder; anyone can rebuild the exact file with one command. The full `piloc.jsonl` is published in the Zenodo archive (https://doi.org/10.5281/zenodo.24085151). Two sources do not state a license: the text reused from them is provided for research reproducibility only, and remains under whatever terms its owners set. Our own additions (spans, labels, manifest) are under CC BY 4.0.

## Intended use and limits

- **Use**: measuring detection recall, localization and faithfulness of prompt-injection guardrails.
- **Not for**: training attack generators. All payloads are already public.
- **Limits**: English only; injections are spliced at sentence boundaries, not written naturally into the text; two carrier genres only.
