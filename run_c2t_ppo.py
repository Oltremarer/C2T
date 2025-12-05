import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from utils import config
from utils.c2t_features import extract_c2t_features, apply_normalization, load_normalization_params
from utils.cityflow_env import CityFlowEnv, TrafficMetrics
from utils.pipeline import path_check, copy_conf_file, copy_cityflow_file
from models.ppo_agent import PPOAgent, PPOConfig
from reward_engineering.train_reward_model import C2TRewardNet


SCENARIO_LIBRARY = {
    "jinan": {"template": "Jinan", "roadnet": "3_4", "traffic_file": "anon_3_4_jinan_real.json", "steps": 3600},
    "hangzhou": {"template": "Hangzhou", "roadnet": "4_4", "traffic_file": "anon_4_4_hangzhou_real.json", "steps": 3600},
    "newyork": {"template": "NewYork", "roadnet": "28_7", "traffic_file": "anon_28_7_newyork_real_double.json", "steps": 3600},
}


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def schedule_lambda(ep_idx: int, warmup_episodes: int) -> float:
    if warmup_episodes <= 0:
        return 1.0
    return min(1.0, ep_idx / warmup_episodes)


class ScalarRunningStat:
    def __init__(self):
        self.mean = 0.0
        self.m2 = 1.0
        self.count = 1e-4

    def update(self, value: float):
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def normalize(self, value: float) -> float:
        variance = self.m2 / max(self.count, 1.0)
        std = float(np.sqrt(max(variance, 1e-8)))
        return (value - self.mean) / std


def prepare_environment(args):
    scenario_key = args.scenario.lower()
    if scenario_key not in SCENARIO_LIBRARY:
        raise ValueError(f"Unknown scenario {args.scenario}.")
    scenario_meta = SCENARIO_LIBRARY[scenario_key]

    base_env_conf = deepcopy(config.dic_traffic_env_conf)
    num_row, num_col = map(int, scenario_meta["roadnet"].split("_"))
    base_env_conf.update({
        "MODEL_NAME": "C2TPPO",
        "PROJECT_NAME": "c2t",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "NUM_INTERSECTIONS": num_row * num_col,
        "TRAFFIC_FILE": scenario_meta["traffic_file"],
        "ROADNET_FILE": f"roadnet_{scenario_meta['roadnet']}.json",
        "RUN_COUNTS": args.max_steps or scenario_meta["steps"],
        "CITYFLOW_SEED": args.seed,
        # Set reward weights: negative to encourage reducing queue/pressure
        "DIC_REWARD_INFO": {
            "queue_length": -1.0,  # Negative: lower queue = higher reward
            "pressure": -1.0,      # Negative: lower pressure = higher reward
        },
    })

    dic_path = deepcopy(config.DIC_PATH)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    work_dir = Path("records") / "c2t_ppo" / scenario_key / timestamp
    model_dir = Path(args.save_dir) / scenario_key
    dic_path["PATH_TO_WORK_DIRECTORY"] = str(work_dir)
    dic_path["PATH_TO_MODEL"] = str(model_dir)
    dic_path["PATH_TO_DATA"] = str(Path("data") / scenario_meta["template"] / scenario_meta["roadnet"])

    attempt = 0
    while True:
        try:
            path_check(dic_path)
            break
        except FileExistsError:
            attempt += 1
            suffix = f"{timestamp}-{attempt}"
            work_dir = Path("records") / "c2t_ppo" / scenario_key / suffix
            model_dir = Path(args.save_dir) / scenario_key / suffix
            dic_path["PATH_TO_WORK_DIRECTORY"] = str(work_dir)
            dic_path["PATH_TO_MODEL"] = str(model_dir)
    copy_conf_file(dic_path, config.DIC_BASE_AGENT_CONF, base_env_conf)
    copy_cityflow_file(dic_path, base_env_conf)

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=base_env_conf,
        dic_path=dic_path
    )
    env.reset()
    return env, base_env_conf, dic_path, scenario_key, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent with C²T intrinsic rewards.")
    parser.add_argument("--scenario", type=str, default="jinan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--intrinsic_warmup", type=int, default=20,
                        help="Episode count to linearly increase intrinsic weight from 0 to 1.")
    parser.add_argument("--ttc_threshold", type=float, default=1.5,
                        help="Minimum TTC threshold for safety mask.")
    parser.add_argument("--queue_threshold", type=float, default=20.0,
                        help="Maximum queue length per junction to enable intrinsic reward. "
                             "If queue exceeds this, mask is set to 0 to prevent rewarding gridlock.")
    parser.add_argument("--intrinsic_clip_max", type=float, default=None,
                        help="Maximum value for normalized intrinsic reward (None = no clip). "
                             "Useful if RewardNet outputs are too high and overwhelm env reward.")
    parser.add_argument("--enable_intrinsic_reward", action="store_true")
    parser.add_argument("--feature_norm_path", type=str, required=True,
                        help="Path to PPO feature normalization params.")
    parser.add_argument("--reward_feature_norm_path", type=str, default=None,
                        help="Optional path for reward model feature normalization.")
    parser.add_argument("--reward_model_path", type=str, required=True,
                        help="Path to trained reward model weights.")
    parser.add_argument("--reward_norm_path", type=str, required=True,
                        help="Path to reward output normalization stats (mean/std).")
    parser.add_argument("--log_dir", type=str, default="runs/c2t")
    parser.add_argument("--save_dir", type=str, default="model_weights/c2t_ppo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="c2t")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    env, env_conf, dic_path, scenario_key, timestamp = prepare_environment(args)
    metrics = TrafficMetrics(env)
    junction_ids = [inter.inter_name for inter in env.list_intersection]

    feature_norm = load_normalization_params(args.feature_norm_path)
    reward_feature_norm = (load_normalization_params(args.reward_feature_norm_path)
                           if args.reward_feature_norm_path else feature_norm)
    with open(args.reward_norm_path, "r", encoding="utf-8") as f_in:
        reward_norm_stats = json.load(f_in)

    device = torch.device(args.device)
    reward_net = C2TRewardNet(input_dim=len(reward_feature_norm["mean"]))
    reward_net.load_state_dict(torch.load(args.reward_model_path, map_location=device))
    reward_net.to(device)
    reward_net.eval()

    agent = PPOAgent(
        input_dim=len(feature_norm["mean"]),
        action_dim=len(env_conf["PHASE"]),
        config=PPOConfig(),
        device=args.device
    )

    log_path = Path(args.log_dir) / scenario_key / timestamp
    writer = SummaryWriter(log_dir=log_path)
    if args.use_wandb:
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name or f"C2TPPO-{scenario_key}-{timestamp}",
                   config=vars(args))

    env_reward_norm = ScalarRunningStat()
    intrinsic_norm = ScalarRunningStat()
    intrinsic_norm.mean = reward_norm_stats.get("mean", 0.0)
    intrinsic_norm.m2 = (reward_norm_stats.get("std", 1.0) ** 2) * intrinsic_norm.count

    for episode in range(1, args.episodes + 1):
        env.reset()
        metrics.refresh_structure_mappings()
        metrics.reset()
        metrics.update()
        step_rewards_env = []
        step_rewards_intr = []
        step_env_norm_vals = []
        step_intr_norm_vals = []
        mask_values = []
        lambda_t = schedule_lambda(episode, args.intrinsic_warmup) if args.enable_intrinsic_reward else 0.0

        for step in range(args.max_steps):
            # metrics correspond to current state before action
            safety_metrics = metrics.get_safety_metrics()
            c2t_state = env.get_c2t_state()

            actions = []
            step_info = []
            for jid in junction_ids:
                junction_state = c2t_state[jid]
                safety = safety_metrics["per_junction"].get(jid, {
                    "min_ttc": metrics.max_ttc,
                    "harsh_brakes": 0,
                    "red_violations": 0,
                    "total_queue": 0.0
                })
                raw_feat = extract_c2t_features(junction_state, safety)
                # PPO: feed raw features, internal RunningMeanStd will normalize online
                ppo_feat = raw_feat
                action, log_prob, value = agent.choose_action(ppo_feat)
                actions.append(action)
                step_info.append((ppo_feat, raw_feat, log_prob, value))

            _, env_rewards, _, _ = env.step(actions)
            # update metrics for new state
            metrics.update()
            safety_metrics_next = metrics.get_safety_metrics()

            for idx, jid in enumerate(junction_ids):
                ppo_feat, raw_feat, log_prob, value = step_info[idx]
                env_reward = env_rewards[idx] if isinstance(env_rewards, list) else env_rewards
                env_reward_norm.update(env_reward)
                r_env_norm = env_reward_norm.normalize(env_reward)

                intrinsic = 0.0
                if args.enable_intrinsic_reward:
                    reward_feat = apply_normalization(raw_feat, reward_feature_norm)
                    with torch.no_grad():
                        tensor_feat = torch.from_numpy(reward_feat).float().unsqueeze(0).to(device)
                        intrinsic = reward_net(tensor_feat).cpu().item()
                intrinsic_norm.update(intrinsic)
                r_intr_norm = intrinsic_norm.normalize(intrinsic)

                # Smart mask: only enable intrinsic reward when BOTH safe (TTC) AND not congested (queue)
                local_ttc = safety_metrics_next["per_junction"].get(
                    jid, {"min_ttc": metrics.max_ttc, "total_queue": 0.0}
                )["min_ttc"]
                junction_queue = safety_metrics_next["per_junction"].get(
                    jid, {"min_ttc": metrics.max_ttc, "total_queue": 0.0}
                )["total_queue"]
                
                is_safe_ttc = local_ttc > args.ttc_threshold
                is_not_congested = junction_queue < args.queue_threshold
                local_mask = 1.0 if (is_safe_ttc and is_not_congested) else 0.0
                mask_values.append(local_mask)

                # Clip intrinsic reward if specified (to prevent it from overwhelming env reward)
                if args.intrinsic_clip_max is not None:
                    r_intr_norm = np.clip(r_intr_norm, a_min=None, a_max=args.intrinsic_clip_max)
                
                # Apply mask: disable intrinsic reward when unsafe or congested
                total_reward = r_env_norm + lambda_t * local_mask * r_intr_norm
                agent.store(ppo_feat, actions[idx], total_reward, log_prob, value, False)

                step_rewards_env.append(env_reward)
                step_rewards_intr.append(intrinsic)
                step_env_norm_vals.append(r_env_norm)
                step_intr_norm_vals.append(r_intr_norm)

        agent.finish_path(last_value=0.0)
        agent.update()

        # compute average mask activation rate over the episode (for analysis)
        mask_mean = float(np.mean(mask_values)) if mask_values else 0.0
        # estimate intrinsic raw std from running stats
        intrinsic_variance = intrinsic_norm.m2 / max(intrinsic_norm.count, 1.0)
        intrinsic_std = float(np.sqrt(max(intrinsic_variance, 1e-8)))

        logs = {
            "reward/env": np.mean(step_rewards_env) if step_rewards_env else 0.0,
            "reward/intrinsic": np.mean(step_rewards_intr) if step_rewards_intr else 0.0,
            "lambda_t": lambda_t,
            "mask_mean": mask_mean,
            "metric/global_min_ttc": safety_metrics_next["global"]["min_ttc"],
            "metric/global_harsh_brakes": safety_metrics_next["global"]["total_harsh_brakes"],
            "metric/global_red_violations": safety_metrics_next["global"]["total_red_violations"],
            "metric/global_queue": safety_metrics_next["global"]["total_queue"],
            # debug signals for analysis
            "debug/env_norm_val": np.mean(step_env_norm_vals) if step_env_norm_vals else 0.0,
            "debug/intr_norm_val": np.mean(step_intr_norm_vals) if step_intr_norm_vals else 0.0,
            "debug/mask_active_rate": mask_mean,
            "debug/raw_intrinsic_std": intrinsic_std,
        }
        for key, value in logs.items():
            writer.add_scalar(key, value, episode)
        if args.use_wandb:
            wandb.log(logs, step=episode)

        ckpt_path = Path(dic_path["PATH_TO_MODEL"]) / f"ppo_agent_ep{episode}.pth"
        torch.save(agent.model.state_dict(), ckpt_path)
        print(f"[C2T-PPO] Episode {episode} finished. Checkpoint saved to {ckpt_path}")

    writer.close()
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

