#!/usr/bin/env bash
# Full pipeline, end to end. Every step exits non-zero on failure, so this script
# stops at the first broken gate -- exactly like Airflow blocking downstream tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 0: generate/refresh synthetic data ==="
python3 data/generate_data.py

echo -e "\n=== Step 1: Ingestion Gate ==="
python3 src/quality/ingestion_gate.py

echo -e "\n=== Step 2: Transformation Gate ==="
python3 src/quality/transformation_gate.py

echo -e "\n=== Step 3: Train + register a new candidate version ==="
python3 -m src.modeling.train

echo -e "\n=== Step 4: Champion-challenger gate + promotion ==="
python3 scripts/evaluate_and_promote.py

echo -e "\nPipeline complete. Start the API with:"
echo "  uvicorn src.serving.app:app --reload --port 8000"
