#!/bin/bash
# Parameter sweep script for GPU 0
# Tests queue_threshold values: 10.0, 20.0

python run_queue_threshold_sweep.py \
  --scenario jinan \
  --seed 42 \
  --episodes 170 \
  --max_steps 200 \
  --feature_norm_path model_weights/c2t_reward/jinan/feature_norm_v3.json \
  --reward_feature_norm_path model_weights/c2t_reward/jinan/feature_norm_v3.json \
  --reward_model_path model_weights/c2t_reward/jinan/reward_model_v3.pth \
  --reward_norm_path model_weights/c2t_reward/jinan/reward_norm_v3.json \
  --queue_thresholds "10.0,20.0" \
  --gpu_id 0 \
  --use_wandb \
  --wandb_project c2t-sweep \
  --wandb_run_name c2t-v3-smartmask-gpu0

echo "[GPU0] Sweep completed!"

