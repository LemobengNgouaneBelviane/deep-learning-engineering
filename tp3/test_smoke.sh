#!/usr/bin/env bash
set -euo pipefail

# Run from this script's directory so python finds cnn_classification.py
cd "$(dirname "$0")"

python3 cnn_classification.py --smoke
