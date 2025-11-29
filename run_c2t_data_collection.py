import argparse
import json
import random
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from utils import config
from utils.pipeline import path_check, copy_cityflow_file, copy_conf_file
from utils.cityflow_env import CityFlowEnv, TrafficMetrics
from utils.c2t_features import extract_c2t_features
from utils.captioner import generate_structured_caption

SCENARIO_LIBRARY = {
    "jinan": {
        "template": "Jinan",
        "roadnet": "3_4",
        "traffic_file": "anon_3_4_jinan_real.json",
        "default_steps": 3600
    },
    "hangzhou": {
        "template": "Hangzhou",
        "roadnet": "4_4",
        "traffic_file": "anon_4_4_hangzhou_real.json",
        "default_steps": 3600
    },
    "newyork": {
        "template": "NewYork",
        "roadnet": "28_7",
        "traffic_file": "anon_28_7_newyork_real_double.json",
        "default_steps": 3600
    }
}


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect C²T preference buffer.")
    parser.add_argument("--scenario", type=str, default="Jinan", help="Scenario key (e.g., Jinan, Hangzhou).")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--policy_type", type=str, default="random", choices=["random"],
                        help="Baseline policy used to explore the environment.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to roll out.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max decision steps per episode.")
    parser.add_argument("--traffic_file", type=str, default=None, help="Optional override for the traffic file.")
    parser.add_argument("--flush_every", type=int, default=200, help="Flush buffer to disk every N samples.")
    parser.add_argument("--output_dir", type=str, default="data/c2t_buffer", help="Root directory for collected data.")
    return parser.parse_args()


def prepare_environment(args):
    scenario_key = args.scenario.lower()
    if scenario_key not in SCENARIO_LIBRARY:
        raise ValueError(f"Unknown scenario '{args.scenario}'. Supported: {list(SCENARIO_LIBRARY.keys())}")
    scenario_meta = SCENARIO_LIBRARY[scenario_key]
    traffic_file = args.traffic_file or scenario_meta["traffic_file"]

    base_env_conf = deepcopy(config.dic_traffic_env_conf)
    num_row, num_col = map(int, scenario_meta["roadnet"].split("_"))
    base_env_conf.update({
        "MODEL_NAME": "C2TDataCollector",
        "PROJECT_NAME": "c2t",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "NUM_INTERSECTIONS": num_row * num_col,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": f"roadnet_{scenario_meta['roadnet']}.json",
        "RUN_COUNTS": args.max_steps or scenario_meta["default_steps"],
        "CITYFLOW_SEED": args.seed
    })

    dic_path = deepcopy(config.DIC_PATH)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    work_dir = Path("records") / "c2t_data" / scenario_key / timestamp
    model_dir = Path("model") / "c2t_data" / scenario_key / timestamp
    dic_path["PATH_TO_WORK_DIRECTORY"] = str(work_dir)
    dic_path["PATH_TO_MODEL"] = str(model_dir)
    dic_path["PATH_TO_DATA"] = str(Path("data") / scenario_meta["template"] / scenario_meta["roadnet"])

    path_check(dic_path)
    copy_conf_file(dic_path, config.DIC_BASE_AGENT_CONF, base_env_conf)
    copy_cityflow_file(dic_path, base_env_conf)

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=base_env_conf,
        dic_path=dic_path
    )
    return env, base_env_conf, scenario_key


def sample_action(policy_type: str, num_phases: int):
    if policy_type == "random":
        return random.randint(0, max(num_phases - 1, 0))
    raise NotImplementedError(f"Policy '{policy_type}' is not implemented.")


def serialize_state(state_dict):
    return {
        "junction_id": state_dict["junction_id"],
        "phase_id": int(state_dict["phase_id"]),
        "phase_duration": float(state_dict["phase_duration"]),
        "lanes": {
            lane_id: {
                "queue_len": float(metrics["queue_len"]),
                "avg_speed": float(metrics["avg_speed"]),
                "avg_wait": float(metrics["avg_wait"])
            }
            for lane_id, metrics in state_dict.get("lanes", {}).items()
        }
    }


def flush_buffer(buffer, output_file: Path):
    if not buffer:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f_out:
        for row in buffer:
            f_out.write(json.dumps(row) + "\n")
    buffer.clear()


def main():
    args = parse_args()
    set_global_seeds(args.seed)
    env, env_conf, scenario_key = prepare_environment(args)
    env.reset()
    metrics = TrafficMetrics(env)
    metrics.refresh_structure_mappings()
    metrics.update()

    output_root = Path(args.output_dir) / scenario_key
    buffer_file = output_root / f"buffer_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    print(f"[C2T] Writing samples to {buffer_file}")

    num_phases = len(env_conf["PHASE"])
    buffer = []

    for episode in range(args.episodes):
        env.reset()
        metrics.refresh_structure_mappings()
        metrics.reset()
        max_steps = args.max_steps or env_conf["RUN_COUNTS"]

        for step_idx in range(max_steps):
            actions = [sample_action(args.policy_type, num_phases) for _ in env.list_intersection]
            env.step(actions)
            # update safety metrics once per environment transition
            metrics.update()
            safety_metrics = metrics.get_safety_metrics()
            state_dict = env.get_c2t_state()
            per_junction_metrics = safety_metrics.get("per_junction", {})

            for junction_id, junction_state in state_dict.items():
                junction_metrics = per_junction_metrics.get(junction_id, {
                    "min_ttc": metrics.max_ttc,
                    "harsh_brakes": 0,
                    "red_violations": 0,
                    "total_queue": 0.0
                })
                feature_vec = extract_c2t_features(junction_state, junction_metrics)
                caption = generate_structured_caption(
                    feature_vec,
                    junction_metrics,
                    junction_id=junction_id,
                    phase_id=junction_state["phase_id"],
                    phase_duration=junction_state["phase_duration"]
                )

                record = {
                    "scenario": scenario_key,
                    "episode": episode,
                    "timestep": step_idx,
                    "junction_id": junction_id,
                    "state": serialize_state(junction_state),
                    "features": feature_vec.tolist(),
                    "caption": caption,
                    "safety": junction_metrics,
                    "global_metrics": safety_metrics.get("global", {})
                }
                buffer.append(record)

            if len(buffer) >= args.flush_every:
                flush_buffer(buffer, buffer_file)

        flush_buffer(buffer, buffer_file)
    print("[C2T] Data collection complete.")


if __name__ == "__main__":
    main()

