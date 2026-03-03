from __future__ import annotations

"""
Utility helpers for PPO training: seeding, logging, evaluation.
"""

import csv
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LoggerConfig:
    log_dir: str
    filename: str = "metrics.csv"


class CSVLogger:
    def __init__(self, cfg: LoggerConfig) -> None:
        self.cfg = cfg
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.path = os.path.join(cfg.log_dir, cfg.filename)
        self._file = open(self.path, mode="w", newline="")
        self._writer = None

    def log(self, row: dict) -> None:
        if self._writer is None:
            fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()


def evaluate_policy(
    env,
    policy,
    episodes: int = 5,
    device: Optional[torch.device] = None,
) -> float:
    """Run the policy without exploration noise and return average return."""
    returns = []
    for _ in range(episodes):
        obs, _info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, _v = policy(obs_tensor)
                action = dist.probs.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        returns.append(ep_ret)
    return float(np.mean(returns))

