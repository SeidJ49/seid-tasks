#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/nj644/dev/anaconda3/envs/bm2cp_v2/bin/python}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-val}"
MAX_BATCHES="${MAX_BATCHES:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-wp3-gradcos}"

run_case() {
  local name="$1"
  local run_dir="$2"
  local epoch="$3"
  local out_dir="$4"
  local config="${run_dir}/config.yaml"
  local checkpoint="${run_dir}/net_epoch${epoch}.pth"

  if [[ ! -f "${config}" ]]; then
    echo "[skip] ${name}: missing ${config}"
    return
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[skip] ${name}: missing ${checkpoint}"
    return
  fi

  echo "[run] ${name}: epoch ${epoch}"
  "${PYTHON_BIN}" opencood/tools/wp3_loss_gradient_attribution.py \
    -y "${config}" \
    --checkpoint "${checkpoint}" \
    --split "${SPLIT}" \
    --output_dir "${out_dir}" \
    --max_batches "${MAX_BATCHES}" \
    --device "${DEVICE}"
}

run_case \
  "current_error_strong" \
  "opencood/logs/pillarnet_radar_distill_rcs_proto_headresponse_error_full_wp3_2026_08_03_10_56_41" \
  8 \
  "debug_wp3_gradcos_current_error_epoch8"

run_case \
  "proto_headresponse" \
  "opencood/logs/pillarnet_radar_distill_rcs_proto_headresponse_full_wp3_2026_08_02_14_24_11" \
  12 \
  "debug_wp3_gradcos_proto_headresponse_epoch12"

run_case \
  "afd_proto" \
  "opencood/logs/pillarnet_radar_distill_rcs_afd_proto_full_wp3_2026_07_31_23_13_16" \
  12 \
  "debug_wp3_gradcos_afd_proto_epoch12"

run_case \
  "radardistill_rcs" \
  "opencood/logs/pillarnet_radar_distill_rcs_wp3_2026_07_26_19_12_03" \
  20 \
  "debug_wp3_gradcos_radardistill_rcs_epoch20"
