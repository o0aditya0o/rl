from __future__ import annotations

"""
Actor-Critic network definitions for PPO.

Shared MLP backbone with separate policy and value heads for discrete-action
gym environments.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ActorCriticConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"  # or "relu"


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class ActorCritic(nn.Module):
    """Shared-body MLP with separate policy and value heads."""

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config

        layers = []
        input_dim = config.obs_dim
        for h in config.hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(_get_activation(config.activation))
            input_dim = h

        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, config.action_dim)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.body(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


def build_actor_critic(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    activation: str = "tanh",
) -> ActorCritic:
    """Convenience builder."""
    cfg = ActorCriticConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )
    return ActorCritic(cfg)


