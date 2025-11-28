import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from utils.c2t_features import apply_normalization, fit_normalization, load_normalization_params, save_normalization_params
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.c2t_features import apply_normalization, fit_normalization, load_normalization_params, save_normalization_params


class PreferenceDataset(Dataset):
    def __init__(self, features_a: np.ndarray, features_b: np.ndarray, labels: np.ndarray):
        self.features_a = features_a.astype(np.float32)
        self.features_b = features_b.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features_a[idx]),
            torch.from_numpy(self.features_b[idx]),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class C2TRewardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_preferences(prefs_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_a, features_b, labels = [], [], []
    if not prefs_path.exists():
        raise FileNotFoundError(f"Preference file not found: {prefs_path}")
    with prefs_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            features_a.append(record["features_a"])
            features_b.append(record["features_b"])
            labels.append(record["label"])
    if not features_a:
        raise RuntimeError("No preference pairs loaded.")
    return np.array(features_a), np.array(features_b), np.array(labels)


def build_dataloader(features_a: np.ndarray, features_b: np.ndarray, labels: np.ndarray,
                     batch_size: int) -> DataLoader:
    dataset = PreferenceDataset(features_a, features_b, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_reward_model(model: C2TRewardNet, dataloader: DataLoader, optimizer, device, epochs: int) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            feat_a, feat_b, labels = batch
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            labels = labels.to(device)

            reward_a = model(feat_a)
            reward_b = model(feat_b)
            logits = reward_a - reward_b
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[RewardTrain] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    return avg_loss


def compute_reward_stats(model: C2TRewardNet, features: np.ndarray, device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        tens = torch.from_numpy(features.astype(np.float32)).to(device)
        rewards = model(tens).cpu().numpy()
    return {
        "mean": float(rewards.mean()),
        "std": float(rewards.std() if rewards.std() > 1e-6 else 1.0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the C²T reward model from preference data.")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario key (e.g., jinan).")
    parser.add_argument("--prefs_path", type=str, default=None,
                        help="Path to preference jsonl file. Defaults to data/c2t_prefs/<scenario>/prefs.jsonl")
    parser.add_argument("--output_dir", type=str, default="model_weights/c2t_reward",
                        help="Directory to store trained model weights and norms.")
    parser.add_argument("--feature_norm_path", type=str, default=None,
                        help="Path to precomputed feature normalization params.")
    parser.add_argument("--save_feature_norm_path", type=str, default=None,
                        help="Optional path to save fitted feature normalization if not provided.")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    scenario_key = args.scenario.lower()
    prefs_path = Path(args.prefs_path) if args.prefs_path else Path("data") / "c2t_prefs" / scenario_key / "prefs.jsonl"
    features_a, features_b, labels = load_preferences(prefs_path)

    if args.feature_norm_path and Path(args.feature_norm_path).exists():
        feature_norm = load_normalization_params(args.feature_norm_path)
    else:
        stacked = np.vstack([features_a, features_b])
        feature_norm = fit_normalization(stacked)
        norm_save_path = args.save_feature_norm_path or (Path("model_weights") / "c2t_reward" /
                                                         scenario_key / "feature_norm.json")
        save_normalization_params(norm_save_path, feature_norm)
        print(f"[RewardTrain] Saved feature normalization to {norm_save_path}")

    features_a_norm = apply_normalization(features_a, feature_norm)
    features_b_norm = apply_normalization(features_b, feature_norm)

    dataloader = build_dataloader(features_a_norm, features_b_norm, labels, args.batch_size)

    device = torch.device(args.device)
    model = C2TRewardNet(input_dim=features_a.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_reward_model(model, dataloader, optimizer, device, args.epochs)

    output_root = Path(args.output_dir) / scenario_key
    output_root.mkdir(parents=True, exist_ok=True)
    reward_model_path = output_root / "reward_model.pth"
    torch.save(model.state_dict(), reward_model_path)
    print(f"[RewardTrain] Saved model weights to {reward_model_path}")

    reward_stats = compute_reward_stats(model, features_a_norm, device)
    reward_norm_path = output_root / "reward_norm.json"
    with reward_norm_path.open("w", encoding="utf-8") as f_out:
        json.dump(reward_stats, f_out, indent=2)
    print(f"[RewardTrain] Saved reward normalization to {reward_norm_path}")


if __name__ == "__main__":
    main()

