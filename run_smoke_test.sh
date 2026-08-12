#!/usr/bin/env bash
# GPU smoke test: full pipeline end-to-end at ~500 docs / ~200 train steps
# (configs/smoke.yaml layered on defaults; science lengths 128/960 unchanged).
# Target: < ~15 min total on one A100. Runs pytest first.
#
# Usage:  ./run_smoke_test.sh
#         PYTHON=.venv/bin/python ./run_smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke_$(date +%Y%m%d-%H%M%S).log"
echo "logging to $LOG"

step() {
    local name="$1"; shift
    echo "=== [$name] $(date +%H:%M:%S) ===" | tee -a "$LOG"
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "$LOG"
    echo "=== [$name] done in $((SECONDS - t0))s ===" | tee -a "$LOG"
}

step pytest            "$PYTHON" -m pytest tests/ -q
step build_dataset     "$PYTHON" scripts/build_dataset.py     --smoke-test
step generate_synth    "$PYTHON" scripts/generate_synthetic.py --smoke-test
step train_classifier  "$PYTHON" scripts/train_classifier.py  --smoke-test
step evaluate          "$PYTHON" scripts/evaluate.py          --smoke-test
step run_controls      "$PYTHON" scripts/run_controls.py      --smoke-test
step annotate_tmin     "$PYTHON" scripts/annotate_tmin.py     --smoke-test

echo
echo "SMOKE TEST PASSED — artifacts:"
ls -la results/auc_vs_t.png results/auc_vs_t.csv results/controls_overlay.png \
       results/controls.csv results/annotations.parquet results/tmin_histogram.png \
       checkpoints/uniform_t/final.pt
echo
echo "NOTE: smoke artifacts live in the same data/, checkpoints/, results/ paths"
echo "as real runs. Before the full run, clear them:  rm -rf data checkpoints results"
