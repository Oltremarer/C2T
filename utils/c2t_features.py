import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

REF_SPEED = 15.0  # m/s reference for normalization


def _safe_array(values):
    if not values:
        return np.zeros(1, dtype=np.float32)
    return np.array(values, dtype=np.float32)


def extract_c2t_features(state_dict_for_junction: Dict[str, Any],
                         safety_metrics_for_junction: Dict[str, Any]) -> np.ndarray:
    """
    Build a numeric feature vector used across Captioner, Reward Model, and PPO.
    """
    lanes = state_dict_for_junction.get("lanes", {})
    queue = _safe_array([lane["queue_len"] for lane in lanes.values()])
    waits = _safe_array([lane["avg_wait"] for lane in lanes.values()])
    speeds = _safe_array([lane["avg_speed"] for lane in lanes.values()])

    queue_total = float(queue.sum())
    queue_mean = float(queue.mean())
    queue_std = float(queue.std())
    queue_max = float(queue.max())
    queue_norm = float((queue / (queue_total + 1e-6)).std()) if queue_total > 0 else 0.0

    wait_mean = float(waits.mean())
    wait_max = float(waits.max())
    wait_norm = float(wait_mean / (queue_mean + 1e-6)) if queue_mean > 0 else 0.0

    speed_mean = float(speeds.mean())
    speed_min = float(speeds.min())
    speed_norm = float(speed_mean / REF_SPEED) if REF_SPEED > 0 else 0.0

    safety_min_ttc = float(safety_metrics_for_junction.get("min_ttc", 0.0))
    safety_brakes = float(safety_metrics_for_junction.get("harsh_brakes", 0))
    safety_red = float(safety_metrics_for_junction.get("red_violations", 0))
    safety_queue = float(safety_metrics_for_junction.get("total_queue", queue_total))

    phase_id = float(state_dict_for_junction.get("phase_id", 0))
    phase_duration = float(state_dict_for_junction.get("phase_duration", 0.0))

    features = np.array([
        phase_id,
        phase_duration,
        queue_total,
        queue_mean,
        queue_std,
        queue_max,
        queue_norm,
        wait_mean,
        wait_max,
        wait_norm,
        speed_mean,
        speed_min,
        speed_norm,
        safety_min_ttc,
        safety_brakes,
        safety_red,
        safety_queue
    ], dtype=np.float32)
    return features


def fit_normalization(feature_matrix: np.ndarray) -> Dict[str, Any]:
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_normalization(features: np.ndarray, normalization_params: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(normalization_params["mean"], dtype=np.float32)
    std = np.asarray(normalization_params["std"], dtype=np.float32)
    return (np.asarray(features, dtype=np.float32) - mean) / std


def save_normalization_params(path: str, normalization_params: Dict[str, Any]):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f_out:
        json.dump(normalization_params, f_out, indent=2)


def load_normalization_params(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f_in:
        return json.load(f_in)

