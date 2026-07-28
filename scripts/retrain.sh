#!/usr/bin/env bash
# Run this whenever you've refreshed/re-curated data/features_clean.parquet by hand
# (real new data, or labels folded back in from label_delay_backfill.py) and want
# to retrain. Skips regenerating the synthetic dataset -- assumes real data is
# already in place at data/features_clean.parquet and data/labels_delayed.parquet.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Transformation Gate (re-check the data you're about to train on) ==="
python3 src/quality/transformation_gate.py

echo -e "\n=== Train + register a new candidate version ==="
python3 -m src.modeling.train

echo -e "\n=== Champion-challenger gate + promotion ==="
python3 scripts/evaluate_and_promote.py

echo -e "\nIf the server is already running, refresh it with:"
echo "  curl -X POST http://localhost:8000/admin/reload"
