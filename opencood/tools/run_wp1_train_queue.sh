#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/ws-ids-es3-01/repo/jadadic_bm2cp"
CONDA_SH="/home/ws-ids-es3-01/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/home/ws-ids-es3-01/miniconda3/envs/jadadic_bm2cp"
LOG_DIR="$REPO_ROOT/opencood/logs/WP1_tmux_queue"

CONFIGS=(
  "opencood/hypes_yaml/WP1/pointpillar_single_lidar_baseline_wp1.yaml"
  "opencood/hypes_yaml/WP1/pointpillar_single_lidar_radar_baseline_wp1.yaml"
  "opencood/hypes_yaml/WP1/pointpillar_single_radar_baseline_wp1.yaml"
)

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script nicht gefunden: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
# Some conda activate scripts reference optional vars that may be unset.
set +u
conda activate "$CONDA_ENV"
set -u

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config nicht gefunden: $cfg" >&2
    exit 1
  fi

  run_name="$(basename "$cfg" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/${run_name}_${ts}.log"

  echo "============================================================"
  echo "[$(date '+%F %T')] Starte Training: $cfg"
  echo "Log: $log_file"
  echo "============================================================"

  python opencood/tools/train.py --hypes_yaml "$cfg" 2>&1 | tee "$log_file"

  echo "[$(date '+%F %T')] Training abgeschlossen: $cfg"
  echo

done

echo "[$(date '+%F %T')] Alle WP1-Trainings abgeschlossen."
