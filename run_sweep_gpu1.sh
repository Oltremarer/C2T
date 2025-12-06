#!/bin/bash
# Parameter sweep script for GPU 1
# Tests queue_threshold values: 30.0, 40.0

python run_queue_threshold_sweep.py \
  --scenario jinan \
  --seed 42 \
  --episodes 170 \
  --max_steps 200 \
  --feature_norm_path model_weights/c2t_reward/jinan/feature_norm_v3.json \
  --reward_feature_norm_path model_weights/c2t_reward/jinan/feature_norm_v3.json \
  --reward_model_path model_weights/c2t_reward/jinan/reward_model_v3.pth \
  --reward_norm_path model_weights/c2t_reward/jinan/reward_norm_v3.json \
  --queue_thresholds "30.0,40.0" \
  --gpu_id 1 \
  --use_wandb \
  --wandb_project c2t-sweep \
  --wandb_run_name c2t-v3-smartmask-gpu1

echo "[GPU1] Sweep completed!"

