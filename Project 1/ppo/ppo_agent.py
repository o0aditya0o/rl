from __future__ import annotations

"""
PPO agent: loss computation and parameter updates.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from .actor_critic import ActorCritic
from .buffer import RolloutBuffer


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    epochs: int = 10
    batch_size: int = 64
    target_kl: Optional[float] = None


class PPOAgent:
    def __init__(self, model: ActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr)

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run multiple epochs of PPO updates over the rollout buffer."""
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "num_updates": 0,
        }

        for epoch in range(self.cfg.epochs):
            for batch in buffer.get(self.cfg.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                dist, values = self.model(obs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Ratio for clipped surrogate objective
                ratios = torch.exp(log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(
                    ratios, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped)
                value_pred_clipped = old_values + torch.clamp(
                    values - old_values, -self.cfg.clip_range, self.cfg.clip_range
                )
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Approximate KL for monitoring / early stopping
                approx_kl = 0.5 * (old_log_probs - log_probs).pow(2).mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl"] += approx_kl.item()
                metrics["num_updates"] += 1

            if self.cfg.target_kl is not None and metrics["num_updates"] > 0:
                mean_kl = metrics["kl"] / metrics["num_updates"]
                if mean_kl > 1.5 * self.cfg.target_kl:
                    break

        if metrics["num_updates"] > 0:
            for k in ["policy_loss", "value_loss", "entropy", "kl"]:
                metrics[k] /= metrics["num_updates"]
        return metrics

