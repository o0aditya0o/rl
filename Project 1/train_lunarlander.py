from __future__ import annotations

import argparse
import os
from typing import Tuple

import gymnasium as gym
import torch

from ppo.actor_critic import build_actor_critic
from ppo.buffer import BufferConfig, RolloutBuffer
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.utils import CSVLogger, LoggerConfig, evaluate_policy, set_seed


def make_env(env_id: str) -> gym.Env:
    return gym.make(env_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO training on LunarLander-v2")
    parser.add_argument("--env_id", type=str, default="LunarLander-v2")
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--rollout_steps", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="Project 1/logs/lunarlander")
    parser.add_argument("--eval_interval", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def get_obs_act_dims(env: gym.Env) -> Tuple[int, int]:
    obs_space = env.observation_space
    act_space = env.action_space
    assert len(obs_space.shape) == 1
    obs_dim = obs_space.shape[0]
    assert hasattr(act_space, "n")
    action_dim = act_space.n
    return obs_dim, action_dim


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    env = make_env(args.env_id)
    eval_env = make_env(args.env_id)

    obs_dim, action_dim = get_obs_act_dims(env)

    model = build_actor_critic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(128, 128),
        activation="tanh",
    ).to(device)

    buffer_cfg = BufferConfig(
        obs_dim=obs_dim,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=str(device),
    )
    buffer = RolloutBuffer(buffer_cfg)

    ppo_cfg = PPOConfig(
        clip_range=args.clip_range,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_kl=0.01,
    )
    agent = PPOAgent(model, ppo_cfg)

    logger = CSVLogger(LoggerConfig(log_dir=args.log_dir))

    obs, _info = env.reset(seed=args.seed)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    global_step = 0
    episode_return = 0.0
    episode_length = 0
    episode_idx = 0

    try:
        while global_step < args.total_steps:
            buffer.reset()
            while not buffer.is_full():
                with torch.no_grad():
                    dist, value = model(obs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _info = env.step(
                    action.item()
                )
                done = terminated or truncated

                buffer.store(
                    obs=obs,
                    action=action.item(),
                    reward=reward,
                    done=done,
                    value=value.item(),
                    log_prob=log_prob.item(),
                )

                episode_return += float(reward)
                episode_length += 1
                global_step += 1

                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

                if done:
                    obs, _info = env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    logger.log(
                        {
                            "global_step": global_step,
                            "episode": episode_idx,
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                        }
                    )
                    episode_return = 0.0
                    episode_length = 0
                    episode_idx += 1

            # Bootstrap value from last obs
            with torch.no_grad():
                dist, value = model(obs)
            buffer.finalize(last_value=value.item())

            metrics = agent.update(buffer)

            row = {
                "global_step": global_step,
                "episode": episode_idx,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "kl": metrics["kl"],
            }

            if global_step % args.eval_interval < args.rollout_steps:
                avg_return = evaluate_policy(eval_env, model, episodes=5, device=device)
                row["eval_return"] = avg_return

            logger.log(row)

        # Save final model
        os.makedirs("Project 1/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "Project 1/checkpoints/lunarlander_ppo.pt")
    finally:
        logger.close()
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

