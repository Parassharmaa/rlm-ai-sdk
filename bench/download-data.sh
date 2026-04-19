#!/usr/bin/env bash
# Download benchmark datasets. Run from repo root:
#   bash bench/download-data.sh
set -euo pipefail
mkdir -p bench/data

echo "Downloading LongBench-v2 (code-repo QA)..."
curl -fSL -o bench/data/longbench-v2.json \
  "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json"

echo "Done. bench/data/longbench-v2.json ($(du -sh bench/data/longbench-v2.json | cut -f1))"
