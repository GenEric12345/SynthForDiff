#!/usr/bin/env bash
# Full experiment at default sizes (30k/5k/5k docs) on one A100.
# Rough total: ~10-18 h, dominated by generation (~4-8 h) and annotation (~2-4 h).
#
# Safe to re-run after an interruption:
#   - build_dataset is skipped if data/dataset.parquet already exists
#   - generate_synthetic natively skips already-generated doc_ids
#   - training resumes from the latest checkpoint (--resume auto)
#
# Usage:  ./run_full_pipeline.sh
#         RUN_FIXED_T=1 ./run_full_pipeline.sh    # also train t=0.5 / t=0.9 controls
#         PYTHON=.venv/bin/python ./run_full_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
RUN_FIXED_T="${RUN_FIXED_T:-0}"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/full_$(date +%Y%m%d-%H%M%S).log"
echo "logging to $LOG"

step() {
    local name="$1"; shift
    echo "=== [$name] $(date +%H:%M:%S) ===" | tee -a "$LOG"
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "$LOG"
    echo "=== [$name] done in $((SECONDS - t0))s ===" | tee -a "$LOG"
}

if [ -f data/dataset.parquet ]; then
    echo "data/dataset.parquet exists — skipping build_dataset" | tee -a "$LOG"
else
    step build_dataset "$PYTHON" scripts/build_dataset.py
fi

step generate_synth   "$PYTHON" scripts/generate_synthetic.py
step train_classifier "$PYTHON" scripts/train_classifier.py --resume auto

if [ "$RUN_FIXED_T" = "1" ]; then
    step train_fixed_t0.5 "$PYTHON" scripts/train_classifier.py --fixed-t 0.5 --resume auto
    step train_fixed_t0.9 "$PYTHON" scripts/train_classifier.py --fixed-t 0.9 --resume auto
fi

step evaluate      "$PYTHON" scripts/evaluate.py
step run_controls  "$PYTHON" scripts/run_controls.py
step annotate_tmin "$PYTHON" scripts/annotate_tmin.py

echo
echo "PIPELINE COMPLETE — key artifacts:"
ls -la results/auc_vs_t.png results/auc_vs_t.csv results/controls_overlay.png \
       results/controls.csv results/annotations.parquet results/tmin_histogram.png \
       checkpoints/uniform_t/final.pt
