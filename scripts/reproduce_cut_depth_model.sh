#!/bin/bash
# Rebuild the frame table and retrain the shipped cut-depth model (models/cut_depth_combined).
#
# Usage: scripts/reproduce_cut_depth_model.sh [data_root] [work_dir]
#   data_root  merged recordings, <root>/<session>/rosbags/lesion_N/cut_*_rect_parsed/...
#              (default ~/Resection_Data_Visual_Prop_merged; see scripts/data/merge_parts.py)
#   work_dir   frame table, 14 GB crop cache and run outputs (default ~/aliss_runs/contact_full)
# Environment:
#   VENV    training venv (default ~/aliss_runs/contact_venv; scripts/setup_train_venv.sh)
#   SEED    training seed (default 0, the shipped model)
#   MODELS  models to cross-validate and export (default: all four, ~45 min on an RTX 5080;
#           "combined" alone takes ~10 min)
#
# Steps:
#   1. build_contact_dataset: one row per frame with depth labels and spot features (~15 min)
#   2. contact_train: 5-fold leave-one-session-out CV, metrics, then a final model per kind
#      trained on all sessions and exported to ONNX (opset 12, loads in OpenCV 4.6 cv2.dnn)
set -euo pipefail
PKG=$(cd "$(dirname "$0")/.." && pwd)
DATA=${1:-$HOME/Resection_Data_Visual_Prop_merged}
WORK=${2:-$HOME/aliss_runs/contact_full}
VENV=${VENV:-$HOME/aliss_runs/contact_venv}
SEED=${SEED:-0}
MODELS=${MODELS:-spot image combined temporal}
PY="$VENV/bin/python -W ignore"

cd "$PKG"
mkdir -p "$WORK"
FRAMES=$WORK/frames_depth.csv
if [ -f "$FRAMES" ]; then
  echo "using existing $FRAMES (delete it to rebuild)"
else
  $PY -m visual_proprioception.contact_dataset --data-root "$DATA" --out "$FRAMES" \
    --labels depth --depth-pos 1.0 --depth-neg 2.0 --lateral 2.0 \
    --retry-window 8.0 --retry-deeper 0.5 2>&1 | tee "$WORK/frames_depth_build.log"
fi

RUN=$WORK/runD_seed$SEED
# shellcheck disable=SC2086
$PY -m visual_proprioception.contact_train --frames "$FRAMES" --out-dir "$RUN" \
  --models $MODELS --epochs 8 --epoch-frames 16000 --seed "$SEED" 2>&1 | tee "$RUN.log"

echo
echo "Done. Compare $RUN/metrics.json with $PKG/models/cut_depth_combined/metrics.json."
echo "To ship the new combined model:"
echo "  cp $RUN/combined/model.{json,onnx} $RUN/metrics.json $PKG/models/cut_depth_combined/"
