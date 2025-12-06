#!/usr/bin/env python3
"""
Parameter sweep script for queue_threshold in C²T PPO training.

This script automatically runs multiple experiments with different queue_threshold values
to find the optimal balance between safety and efficiency.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(queue_threshold: float, base_args: dict, run_suffix: str = ""):
    """Run a single experiment with given queue_threshold."""
    cmd = [
        sys.executable, "run_c2t_ppo.py",
        "--scenario", base_args["scenario"],
        "--seed", str(base_args["seed"]),
        "--episodes", str(base_args["episodes"]),
        "--max_steps", str(base_args["max_steps"]),
        "--feature_norm_path", base_args["feature_norm_path"],
        "--reward_feature_norm_path", base_args["reward_feature_norm_path"],
        "--reward_model_path", base_args["reward_model_path"],
        "--reward_norm_path", base_args["reward_norm_path"],
        "--queue_threshold", str(queue_threshold),
        "--enable_intrinsic_reward",
    ]
    
    if base_args.get("intrinsic_warmup"):
        cmd.extend(["--intrinsic_warmup", str(base_args["intrinsic_warmup"])])
    
    if base_args.get("intrinsic_clip_max"):
        cmd.extend(["--intrinsic_clip_max", str(base_args["intrinsic_clip_max"])])
    
    if base_args.get("use_wandb", False):
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", base_args.get("wandb_project", "c2t")])
        wandb_name = base_args.get("wandb_run_name", "c2t-sweep")
        cmd.extend(["--wandb_run_name", f"{wandb_name}-q{queue_threshold:.1f}{run_suffix}"])
    
    # Set device: if gpu_id is specified, use cuda:gpu_id; otherwise use base_args["device"]
    if base_args.get("gpu_id") is not None:
        cmd.extend(["--device", f"cuda:{base_args['gpu_id']}"])
    elif base_args.get("device"):
        cmd.extend(["--device", base_args["device"]])
    
    print(f"\n{'='*80}")
    print(f"[Sweep] Running experiment with queue_threshold={queue_threshold}")
    print(f"[Sweep] Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n[Sweep] ✓ Experiment with queue_threshold={queue_threshold} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Sweep] ✗ Experiment with queue_threshold={queue_threshold} failed with exit code {e.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for queue_threshold in C²T PPO training."
    )
    
    # Base experiment arguments (passed to run_c2t_ppo.py)
    parser.add_argument("--scenario", type=str, default="jinan", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=170)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--feature_norm_path", type=str, required=True)
    parser.add_argument("--reward_feature_norm_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--reward_norm_path", type=str, required=True)
    parser.add_argument("--intrinsic_warmup", type=int, default=20)
    parser.add_argument("--intrinsic_clip_max", type=float, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto, cuda, cpu). If --gpu_id is set, this is ignored.")
    
    # Sweep-specific arguments
    parser.add_argument("--queue_thresholds", type=str, default="10.0,15.0,20.0,25.0,30.0,35.0",
                        help="Comma-separated list of queue_threshold values to test.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="c2t-sweep")
    parser.add_argument("--wandb_run_name", type=str, default="c2t-v3-smartmask")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="Specific GPU ID to use (0, 1, etc.). If not set, uses default device.")
    
    args = parser.parse_args()
    
    # Parse queue_threshold values
    try:
        queue_thresholds = [float(x.strip()) for x in args.queue_thresholds.split(",")]
    except ValueError:
        print(f"[Sweep] Error: Invalid queue_thresholds format: {args.queue_thresholds}")
        print("[Sweep] Expected format: '10.0,15.0,20.0' (comma-separated floats)")
        sys.exit(1)
    
    print(f"[Sweep] Starting parameter sweep for queue_threshold")
    print(f"[Sweep] Values to test: {queue_thresholds}")
    print(f"[Sweep] Total experiments: {len(queue_thresholds)}")
    
    # Prepare base arguments
    base_args = {
        "scenario": args.scenario,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "feature_norm_path": args.feature_norm_path,
        "reward_feature_norm_path": args.reward_feature_norm_path,
        "reward_model_path": args.reward_model_path,
        "reward_norm_path": args.reward_norm_path,
        "intrinsic_warmup": args.intrinsic_warmup,
        "intrinsic_clip_max": args.intrinsic_clip_max,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    
    # Sequential execution (each experiment runs one at a time on the specified GPU)
    results = []
    gpu_info = f"cuda:{args.gpu_id}" if args.gpu_id is not None else (args.device or "auto")
    print(f"[Sweep] Running on device: {gpu_info}")
    for q_thresh in queue_thresholds:
        success = run_experiment(q_thresh, base_args)
        results.append((q_thresh, success))
    
    # Summary
    print(f"\n{'='*80}")
    print("[Sweep] Parameter sweep completed!")
    print(f"{'='*80}")
    print("\nResults summary:")
    for q_thresh, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  queue_threshold={q_thresh:5.1f}: {status}")
    
    successful = [q for q, s in results if s]
    if successful:
        print(f"\n[Sweep] Successful experiments: {len(successful)}/{len(results)}")
        print(f"[Sweep] Tested values: {successful}")
        print(f"[Sweep] Compare results in WandB project: {args.wandb_project}")
        print(f"[Sweep] Key metrics to compare:")
        print(f"  - metric/global_queue (lower is better)")
        print(f"  - reward/env (higher/less negative is better)")
        print(f"  - debug/mask_active_rate (should be 0.3-0.7 for good balance)")
        print(f"  - metric/global_red_violations (lower is better)")
        print(f"  - metric/global_harsh_brakes (lower is better)")
    else:
        print("\n[Sweep] ⚠ All experiments failed. Please check error messages above.")


if __name__ == "__main__":
    main()

