from typing import Dict, Any
import numpy as np

# feature layout after one-hot:
# 0: phase_duration
# 1-8: phase_one_hot (MAX_PHASES=8)
# 9: queue_total
# 14: wait_mean
# 17: speed_mean
IDX_QUEUE_TOTAL = 9
IDX_WAIT_MEAN = 14
IDX_SPEED_MEAN = 17


def generate_structured_caption(features: np.ndarray,
                                metrics_for_junction: Dict[str, Any],
                                junction_id: str,
                                phase_id: int,
                                phase_duration: float) -> str:
    """
    Deterministic caption template used for LLM preference queries.
    """
    total_queue = float(metrics_for_junction.get("total_queue", float(features[IDX_QUEUE_TOTAL])))
    avg_delay = float(features[IDX_WAIT_MEAN])
    avg_speed = float(features[IDX_SPEED_MEAN])
    min_ttc = float(metrics_for_junction.get("min_ttc", 0.0))
    harsh_brakes = int(metrics_for_junction.get("harsh_brakes", 0))
    red_violations = int(metrics_for_junction.get("red_violations", 0))
    window = max(phase_duration, 1.0)

    caption = (
        f"At junction {junction_id}, phase {phase_id} has been active for {phase_duration:.1f} seconds. "
        f"The total queue length is {total_queue:.1f} vehicles, with average delay {avg_delay:.1f} seconds. "
        f"The approaching traffic moves at {avg_speed:.2f} m/s on average. "
        f"The minimum time-to-collision is {min_ttc:.2f} seconds, and there are {harsh_brakes} harsh brakes "
        f"in the last {window:.1f} seconds. "
        f"Red-light violations counted: {red_violations}."
    )
    return caption

