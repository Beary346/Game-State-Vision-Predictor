#!/usr/bin/env bash
set -euo pipefail

echo "=== Bronze: preprocessing raw screenshots ==="
python -m src.pipeline.bronze --image data/bronze/test_screenshot.png --output data/silver/processed.jpg

echo "=== Silver: cleaning / EDA ==="
# TODO: python -m src.pipeline.silver --input data/silver/processed.jpg --output data/gold/

echo "=== Gold: training ==="
# TODO: python -m src.pipeline.gold --input data/gold/ --output models/

echo "Done."
