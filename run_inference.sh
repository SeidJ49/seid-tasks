#!/bin/bash

MODEL_DIR="/home/ws-ids-es3-01/PycharmProjects/hamdard_bm2cp/opencood/logs/old_truckscences/paper/pointpillar_single_lidar_radar_baseline_attention_mlp_his_sweep_2025_10_19_23_55_39"
for epoch in {30..23..-1}
do
    echo "Starte Inferenz mit Epoch $epoch"
    python opencood/tools/inference_epoch.py --model_dir "$MODEL_DIR" --epoch $epoch
done
