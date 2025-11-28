import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROMPT_TEMPLATE = """You are a traffic safety and operations expert.
Compare these two traffic states:

State A:
{caption_a}

State B:
{caption_b}

Criteria (in order of importance):
1) Safety: higher TTC, fewer harsh brakes, fewer red-light violations.
2) Efficiency: lower queue length and lower delay.
3) Smoothness: fewer abrupt changes in traffic conditions.

Which state is better overall?
Respond ONLY with a JSON object: {{'better_state': 'A'}} or {{'better_state': 'B'}}."""


@dataclass
class PreferenceRecord:
    features_a: List[float]
    features_b: List[float]
    caption_a: str
    caption_b: str
    label: int  # 1 if A better, 0 if B better
    junction_id: str
    informative_score: float


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_buffer_records(buffer_dir: Path) -> Dict[str, List[dict]]:
    per_junction: Dict[str, List[dict]] = defaultdict(list)
    files = sorted(buffer_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No buffer files found under {buffer_dir}")
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jid = record.get("junction_id")
                if not jid:
                    continue
                per_junction[jid].append(record)
    return per_junction


def informative_score(rec_a: dict, rec_b: dict) -> float:
    safety_a = rec_a.get("safety", {})
    safety_b = rec_b.get("safety", {})
    queue_diff = abs(safety_a.get("total_queue", 0.0) - safety_b.get("total_queue", 0.0))
    ttc_diff = abs(safety_a.get("min_ttc", 0.0) - safety_b.get("min_ttc", 0.0))
    brake_diff = abs(safety_a.get("harsh_brakes", 0.0) - safety_b.get("harsh_brakes", 0.0))
    red_diff = abs(safety_a.get("red_violations", 0.0) - safety_b.get("red_violations", 0.0))
    return float(max(queue_diff, ttc_diff, brake_diff, red_diff))


def pick_informative_pairs(records_by_junction: Dict[str, List[dict]],
                           target_pairs: int,
                           rng: random.Random,
                           min_score: float) -> List[Tuple[dict, dict, float]]:
    candidate_pairs: List[Tuple[dict, dict, float]] = []
    for junction_id, records in records_by_junction.items():
        if len(records) < 2:
            continue
        # shuffle to add randomness
        shuffled = records.copy()
        rng.shuffle(shuffled)
        num_samples = min(len(shuffled) * 3, 500)
        for _ in range(num_samples):
            rec_a, rec_b = rng.sample(shuffled, 2)
            score = informative_score(rec_a, rec_b)
            if score < min_score:
                continue
            candidate_pairs.append((rec_a, rec_b, score))

    candidate_pairs.sort(key=lambda tup: tup[2], reverse=True)
    return candidate_pairs[:target_pairs]


def mock_label(rec_a: dict, rec_b: dict) -> int:
    def score(rec: dict) -> float:
        safety = rec.get("safety", {})
        queue = safety.get("total_queue", 0.0)
        ttc = safety.get("min_ttc", 0.0)
        brakes = safety.get("harsh_brakes", 0.0)
        red = safety.get("red_violations", 0.0)
        delay = rec.get("features", [0.0] * 10)[7] if rec.get("features") else 0.0
        return 0.5 * ttc - 0.3 * queue - 0.1 * delay - 0.05 * brakes - 0.05 * red

    score_a = score(rec_a)
    score_b = score(rec_b)
    return 1 if score_a >= score_b else 0


def call_openai_chat(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("openai package is required for real LLM labeling. Install via `pip install openai`.") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a traffic safety and operations expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion["choices"][0]["message"]["content"]


def parse_llm_response(response_text: str) -> Optional[int]:
    response_text = response_text.strip()
    try:
        parsed = json.loads(response_text.replace("'", '"'))
    except json.JSONDecodeError:
        return None
    better_state = parsed.get("better_state")
    if better_state == "A":
        return 1
    if better_state == "B":
        return 0
    return None


def create_preference_record(rec_a: dict, rec_b: dict, label: int, score: float) -> PreferenceRecord:
    return PreferenceRecord(
        features_a=rec_a["features"],
        features_b=rec_b["features"],
        caption_a=rec_a["caption"],
        caption_b=rec_b["caption"],
        label=label,
        junction_id=rec_a["junction_id"],
        informative_score=score
    )


def write_preferences(output_path: Path, records: List[PreferenceRecord]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f_out:
        for rec in records:
            f_out.write(json.dumps(rec.__dict__) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate preference labels using LLM comparisons.")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario key, e.g., jinan or hangzhou.")
    parser.add_argument("--buffer_dir", type=str, default=None,
                        help="Directory containing C²T buffer jsonl files.")
    parser.add_argument("--output_dir", type=str, default="data/c2t_prefs",
                        help="Directory to store preference jsonl files.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="LLM model name for OpenAI chat completion.")
    parser.add_argument("--max_pairs", type=int, default=1000, help="Maximum number of preference pairs to label.")
    parser.add_argument("--min_score", type=float, default=2.0, help="Minimum informative score to consider a pair.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_mock_labeler", action="store_true",
                        help="If set, bypass LLM and generate labels via heuristic.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    scenario_key = args.scenario.lower()
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else Path("data") / "c2t_buffer" / scenario_key
    records_by_junction = load_buffer_records(buffer_dir)
    rng = random.Random(args.seed)
    candidate_pairs = pick_informative_pairs(records_by_junction, args.max_pairs, rng, args.min_score)
    if not candidate_pairs:
        raise RuntimeError("No informative pairs found. Consider lowering --min_score or collecting more data.")

    labeled_records: List[PreferenceRecord] = []
    for rec_a, rec_b, score in candidate_pairs:
        prompt = PROMPT_TEMPLATE.format(caption_a=rec_a["caption"], caption_b=rec_b["caption"])
        if args.use_mock_labeler:
            label = mock_label(rec_a, rec_b)
        else:
            response_text = call_openai_chat(prompt, args.model_name, temperature=0.0)
            label = parse_llm_response(response_text)
            if label is None:
                print("Invalid LLM response, skipping pair.")
                continue
        labeled_records.append(create_preference_record(rec_a, rec_b, label, score))

    output_dir = Path(args.output_dir) / scenario_key
    output_file = output_dir / "prefs.jsonl"
    write_preferences(output_file, labeled_records)
    print(f"[PreferenceLabeling] Wrote {len(labeled_records)} labeled pairs to {output_file}")


if __name__ == "__main__":
    main()

