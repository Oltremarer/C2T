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


class LLMProvider:
    """Abstract interface for LLM providers."""
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt.
            temperature: Sampling temperature (0.0 for deterministic).
        
        Returns:
            The generated text response.
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_base: Optional[str] = None):
        try:
            import openai
            self.openai = openai
        except ImportError as exc:
            raise RuntimeError("openai package is required. Install via `pip install openai`.") from exc
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY must be provided via argument or environment variable.")
        
        # Support custom API base (e.g., for OpenAI-compatible local servers)
        if api_base:
            self.openai.api_base = api_base
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.openai.api_key = self.api_key
        completion = self.openai.ChatCompletion.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a traffic safety and operations expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion["choices"][0]["message"]["content"]


class LocalLLMProvider(LLMProvider):
    """Local model provider using transformers."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for local models. Install via `pip install transformers torch`.") from exc
        
        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[LocalLLM] Loading model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"[LocalLLM] Model loaded successfully.")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        import torch
        
        system_msg = "You are a traffic safety and operations expert."
        full_prompt = f"{system_msg}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


def create_llm_provider(llm_type: str, model_name: Optional[str] = None, model_path: Optional[str] = None,
                       api_key: Optional[str] = None, api_base: Optional[str] = None,
                       device: str = "auto") -> LLMProvider:
    """
    Factory function to create an LLM provider.
    
    Args:
        llm_type: One of "openai", "local", or "mock".
        model_name: Model name for OpenAI API (e.g., "gpt-4o-mini").
        model_path: Path to local model (for "local" type).
        api_key: API key for OpenAI (optional, can use env var).
        api_base: Custom API base URL (for OpenAI-compatible servers).
        device: Device for local models ("auto", "cuda", or "cpu").
    
    Returns:
        An LLMProvider instance.
    """
    if llm_type == "openai":
        if not model_name:
            raise ValueError("--model_name is required when --llm_type=openai")
        return OpenAIProvider(model_name=model_name, api_key=api_key, api_base=api_base)
    elif llm_type == "local":
        if not model_path:
            raise ValueError("--model_path is required when --llm_type=local")
        return LocalLLMProvider(model_path=model_path, device=device)
    elif llm_type == "mock":
        return None  # Handled separately in main()
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}. Choose from: openai, local, mock")


def parse_llm_response(response_text: str) -> Optional[int]:
    response_text = response_text.strip()
    # Handle common Markdown code fences like ```json ... ```
    if response_text.startswith("```"):
        # remove leading ```xxx and trailing ```
        # find first '{' and last '}' to extract JSON object
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            response_text = response_text[start : end + 1]
        else:
            # fallback: strip fences heuristically
            lines = [ln for ln in response_text.splitlines() if not ln.strip().startswith("```")]
            response_text = "\n".join(lines).strip()

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
    
    # LLM provider selection
    parser.add_argument("--llm_type", type=str, default="openai", choices=["openai", "local", "mock"],
                        help="LLM provider type: 'openai' for OpenAI API, 'local' for local models, 'mock' for heuristic.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="Model name for OpenAI API (e.g., 'gpt-4o-mini', 'gpt-4').")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local model directory (required for --llm_type=local).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (optional, can use OPENAI_API_KEY env var).")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Custom API base URL (for OpenAI-compatible local servers, e.g., vLLM).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device for local models (default: auto-detect).")
    
    parser.add_argument("--max_pairs", type=int, default=1000, help="Maximum number of preference pairs to label.")
    parser.add_argument("--min_score", type=float, default=2.0, help="Minimum informative score to consider a pair.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_mock_labeler", action="store_true",
                        help="[Deprecated] Use --llm_type=mock instead. If set, bypass LLM and generate labels via heuristic.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    # Handle deprecated --use_mock_labeler flag
    if args.use_mock_labeler:
        print("[WARNING] --use_mock_labeler is deprecated. Use --llm_type=mock instead.")
        args.llm_type = "mock"

    scenario_key = args.scenario.lower()
    buffer_dir = Path(args.buffer_dir) if args.buffer_dir else Path("data") / "c2t_buffer" / scenario_key
    records_by_junction = load_buffer_records(buffer_dir)
    rng = random.Random(args.seed)
    candidate_pairs = pick_informative_pairs(records_by_junction, args.max_pairs, rng, args.min_score)
    if not candidate_pairs:
        raise RuntimeError("No informative pairs found. Consider lowering --min_score or collecting more data.")

    # Initialize LLM provider
    llm_provider = None
    if args.llm_type != "mock":
        llm_provider = create_llm_provider(
            llm_type=args.llm_type,
            model_name=args.model_name,
            model_path=args.model_path,
            api_key=args.api_key,
            api_base=args.api_base,
            device=args.device
        )
        print(f"[PreferenceLabeling] Using LLM provider: {args.llm_type}")

    labeled_records: List[PreferenceRecord] = []
    total_pairs = len(candidate_pairs)
    for idx, (rec_a, rec_b, score) in enumerate(candidate_pairs, 1):
        prompt = PROMPT_TEMPLATE.format(caption_a=rec_a["caption"], caption_b=rec_b["caption"])
        
        if args.llm_type == "mock":
            label = mock_label(rec_a, rec_b)
        else:
            try:
                response_text = llm_provider.generate(prompt, temperature=0.0)
                label = parse_llm_response(response_text)
                if label is None:
                    print(f"[{idx}/{total_pairs}] Invalid LLM response, skipping pair. Response: {response_text[:100]}")
                    continue
            except Exception as e:
                print(f"[{idx}/{total_pairs}] LLM call failed: {e}, skipping pair.")
                continue
        
        labeled_records.append(create_preference_record(rec_a, rec_b, label, score))
        if idx % 50 == 0:
            print(f"[PreferenceLabeling] Processed {idx}/{total_pairs} pairs, {len(labeled_records)} valid labels so far.")

    output_dir = Path(args.output_dir) / scenario_key
    output_file = output_dir / "prefs.jsonl"
    write_preferences(output_file, labeled_records)
    print(f"[PreferenceLabeling] Wrote {len(labeled_records)} labeled pairs to {output_file}")


if __name__ == "__main__":
    main()

