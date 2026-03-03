from __future__ import annotations

"""
Rollout buffer with Generalized Advantage Estimation (GAE) for PPO.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class BufferConfig:
    obs_dim: int
    rollout_steps: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self, cfg: BufferConfig) -> None:
        self.cfg = cfg
        T = cfg.rollout_steps
        obs_dim = cfg.obs_dim
        self.obs = torch.zeros((T, obs_dim), dtype=torch.float32, device=cfg.device)
        self.actions = torch.zeros(T, dtype=torch.long, device=cfg.device)
        self.rewards = torch.zeros(T, dtype=torch.float32, device=cfg.device)
        self.dones = torch.zeros(T, dtype=torch.float32, device=cfg.device)
        self.values = torch.zeros(T, dtype=torch.float32, device=cfg.device)
        self.log_probs = torch.zeros(T, dtype=torch.float32, device=cfg.device)

        self.advantages = torch.zeros(T, dtype=torch.float32, device=cfg.device)
        self.returns = torch.zeros(T, dtype=torch.float32, device=cfg.device)

        self.ptr = 0
        self.full = False

    def store(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.cfg.rollout_steps:
            raise RuntimeError("RolloutBuffer is full; call finalize() before reusing.")
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.ptr += 1
        if self.ptr == self.cfg.rollout_steps:
            self.full = True

    def is_full(self) -> bool:
        return self.full

    @torch.no_grad()
    def finalize(self, last_value: float) -> None:
        """Compute GAE advantages and returns."""
        T = self.ptr
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        last_adv = 0.0
        last_val = float(last_value)

        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]
            next_value = last_val if t == T - 1 else self.values[t + 1].item()
            delta = (
                self.rewards[t].item()
                + gamma * next_value * mask
                - self.values[t].item()
            )
            last_adv = delta + gamma * lam * mask * last_adv
            self.advantages[t] = last_adv
            self.returns[t] = self.advantages[t] + self.values[t]

        # Normalize advantages for stability
        adv_mean = self.advantages[:T].mean()
        adv_std = self.advantages[:T].std(unbiased=False) + 1e-8
        self.advantages[:T] = (self.advantages[:T] - adv_mean) / adv_std

    def get(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Yield mini-batches over the rollout."""
        if not self.full:
            raise RuntimeError("Buffer not full; cannot sample mini-batches yet.")

        T = self.cfg.rollout_steps
        indices = torch.randperm(T, device=self.cfg.device)
        for start in range(0, T, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
            }

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

