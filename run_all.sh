#!/usr/bin/env bash
# Reproduce every number and figure of the paper (CPU only, about 2.5 h on 2 cores).
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
python -m piloc.build_data --seed 47 --out data          # 1. benchmark
python -m pytest -q tests                                # 2. integrity, unit and model tests
python -m piloc.audit --seed 47 --out results            # 3. detection, explanations, sanitization
python -m piloc.stats data/piloc.jsonl results           # 4. results/summary.json
python -m piloc.figures results figures                  # 5. figures/*.pdf
cp figures/*.pdf paper/
python -m piloc.check_paper macros results/summary.json paper/numbers.tex   # 6. numbers of the paper
python -m piloc.check_paper paper/main.tex                                  # 7. no hand-typed result
