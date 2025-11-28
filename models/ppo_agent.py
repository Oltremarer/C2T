import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = torch.tensor(epsilon, dtype=torch.float32)

    def update(self, x: torch.Tensor):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return (x - self.mean) / torch.sqrt(self.var + eps)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOAgent:
    def __init__(self, input_dim: int, action_dim: int, config: PPOConfig, device: str = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))

        self.reset_storage()

    def reset_storage(self):
        self.storage = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def choose_action(self, state_features: np.ndarray) -> Tuple[int, float]:
        state = torch.from_numpy(state_features).float().unsqueeze(0)
        self.obs_rms.update(state)
        state_norm = self.obs_rms.normalize(state).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state_norm)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.cpu().item()), float(log_prob.cpu().item()), float(value.cpu().item())

    def store(self, state: np.ndarray, action: int, reward: float, log_prob: float,
              value: float, done: bool):
        self.storage["states"].append(state)
        self.storage["actions"].append(action)
        self.storage["log_probs"].append(log_prob)
        self.storage["rewards"].append(reward)
        self.storage["values"].append(value)
        self.storage["dones"].append(done)

    def finish_path(self, last_value: float = 0.0):
        rewards = np.array(self.storage["rewards"] + [last_value])
        values = np.array(self.storage["values"] + [last_value])
        dones = np.array(self.storage["dones"] + [0])

        gae = 0
        returns = []
        for step in reversed(range(len(self.storage["rewards"]))):
            delta = rewards[step] + self.config.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        advantages = np.array(returns) - np.array(self.storage["values"])
        self.storage["advantages"] = advantages
        self.storage["returns"] = returns

    def update(self):
        states = torch.tensor(self.storage["states"], dtype=torch.float32)
        states = self.obs_rms.normalize(states).to(self.device)
        actions = torch.tensor(self.storage["actions"], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.storage["log_probs"], dtype=torch.float32).to(self.device)
        returns = torch.tensor(self.storage["returns"], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(self.storage["advantages"], dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.shape[0]
        batch_size = min(self.config.batch_size, dataset_size)
        for _ in range(self.config.epochs):
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = perm[start:start + batch_size]
                logits, values = self.model(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions[idx])

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns[idx])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        self.reset_storage()


