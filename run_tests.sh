#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="/home/ws-ids-es3-01/repo/jadadic_bm2cp"
TRAIN_PY="$REPO_ROOT/opencood/tools/train.py"
GPU_ID=0
export PYTHONPATH="$PWD"


INTERVAL_SECS=$((5*60))

YAMLS=(
  "$REPO_ROOT/opencood/hypes_yaml/WP1/pointpillar_single_lidar_baseline_wp1.yaml"
  "$REPO_ROOT/opencood/hypes_yaml/WP1/pointpillar_single_lidar_radar_baseline_wp1.yaml"
  "$REPO_ROOT/opencood/hypes_yaml/WP1/pointpillar_single_radar_baseline_wp1.yaml"
)

for yml in "${YAMLS[@]}"; do
  start_ts=$(date +%s)
  name="$(basename "$yml" .yaml)"

  echo "[$(date '+%F %T')] Starte Training: $name"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python "$TRAIN_PY" --hypes_yaml "$yml"

  echo "[$(date '+%F %T')] Fertig: $name"

  sleep 60

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  remain=$((INTERVAL_SECS - elapsed))
  (( remain > 0 )) && sleep "$remain"
done

echo "[$(date '+%F %T')] Alle Trainings abgeschlossen"