#!/usr/bin/env bash
# Reproduces the main table and every number in the report from the released artifacts.
# Usage: bash run_all.sh        (SKIP_INSTALL=1 skips pip; PYTHON=python3 selects the interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
ARTIFACTS_ID="${ARTIFACTS_ID:-12CViLk8V_4WIqX6KzsYZF5SNH9h3Rnib}"
mkdir -p results/logs artifacts
[ "${SKIP_INSTALL:-0}" = "1" ] || $PY -m pip install -r requirements.txt
$PY -c "import tensorflow" 2>/dev/null || { echo "ERROR: TensorFlow cannot be imported (wrong Python, or blocked by an OS policy). Use Python 3.12, or run in WSL/Linux/Colab. See README."; exit 1; }
if [ ! -f artifacts/waste_experiments/exp1_yolov5s/test_detections.pkl ]; then
  $PY -m gdown "$ARTIFACTS_ID" -O artifacts/artifacts.zip
  $PY src/extract_artifacts.py artifacts/artifacts.zip artifacts
fi
$PY src/prepare_data.py                                   2>&1 | tee results/logs/prepare_data.log
$PY src/controls.py            --artifacts artifacts      2>&1 | tee results/logs/controls.log
$PY src/evaluate_baselines.py  --artifacts artifacts      2>&1 | tee results/logs/evaluate_baselines.log
$PY src/evaluate_hybrid.py     --artifacts artifacts      2>&1 | tee results/logs/evaluate_hybrid.log
$PY src/analysis_failures.py   --artifacts artifacts      2>&1 | tee results/logs/analysis_failures.log
$PY src/make_main_table.py                                2>&1 | tee results/logs/make_main_table.log
echo "DONE. Main table: results/main_table.md"
