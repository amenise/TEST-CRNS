#!/usr/bin/env bash
# Full experimental pipeline for the DINOv2 ViT-S/14 backbone (added after results/backbone_probe.csv).
# Steps run one after another (one GPU job at a time on a 4 GB card); every step is resumable.
# Prerequisite: python -m birdshift.cli.train_source --backbone dinov2 (source head + cached features).
# Usage: bash scripts/run_pipeline_dinov2.sh   (log: runs/logs/pipeline_dinov2.log)
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=runs/logs/pipeline_dinov2.log
mkdir -p runs/logs

step() {
  echo "=== STEP $1 START $(date +%H:%M:%S)" | tee -a "$LOG"
  shift
  "$PY" -u "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "=== STEP END rc=$rc $(date +%H:%M:%S)" | tee -a "$LOG"
  return $rc
}

step shift      -m birdshift.cli.run_shift --backbone dinov2 || exit 1
step geo        -m birdshift.cli.run_geo --backbones dinov2 --reuse-prior || exit 1
step hp         -m birdshift.cli.select_hp --backbone dinov2 || exit 1
step tta        -m birdshift.cli.run_tta --backbone dinov2 || exit 1
step tta_an     -m birdshift.cli.analyze_tta --backbone dinov2 || exit 1
step failure    -m birdshift.cli.run_failure --backbone dinov2 --parts layers noniid || exit 1
step ctta       -m birdshift.cli.run_ctta --backbone dinov2 --sequences brief year || exit 1
step ctta_long  -m birdshift.cli.run_ctta --backbone dinov2 --sequences brief_long || exit 1
step ctta_seeds -m birdshift.cli.run_ctta --backbone dinov2 --sequences brief \
  --methods source tent eata rdumb anchored anchored-noanchor anchored-noteacher --seeds 1 2 || exit 1
step ctta_rec   -m birdshift.cli.run_ctta --backbone dinov2 --sequences brief_recurring --methods eata anchored --seeds 0 || exit 1
step ctta_csv   -m birdshift.cli.run_ctta --backbone dinov2 --rebuild-csv || exit 1
step ctta_an    -m birdshift.cli.analyze_ctta --backbone dinov2 || exit 1
step detector   -m birdshift.cli.run_detector --backbone dinov2 || exit 1
step fusion     -m birdshift.cli.fuse_tta --backbones dinov2 || exit 1
echo "=== PIPELINE DONE $(date +%H:%M:%S)" | tee -a "$LOG"
