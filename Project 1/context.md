# Project 1: RL Zero - PPO Implementation Roadmap

The objective of this project is to implement **Proximal Policy Optimization (PPO)** from scratch. PPO is the backbone of modern LLM alignment (RLHF), and understanding its mechanics at a low level is essential for post-training expertise.

## 🛠 The Implementation Roadmap

### 1. Actor-Critic Architecture
The first step is building the neural network foundations.
- **Actor (Policy)**: Outputs action probabilities.
- **Critic (Value)**: Estimates state values $V(s)$.
- Base shared layers vs. separate networks for stability.

### 2. Trajectory Collection (The Buffer)
A rollout buffer to store transitions $(s, a, r, s', log\_p, v)$ during training phases.
- Handling terminal states and truncation.
- Vectorized environments for throughput.

### 3. Generalized Advantage Estimation (GAE)
Implementing GAE to calculate the "Advantage" of an action—how much better it was than expected. This is the core signal for the policy update.

### 4. The PPO Loss Function
The "Proximal" part of PPO that prevents the policy from changing too drastically.
- Probability Ratio Calculation.
- Clipped Surrogate Objective.
- Entropy Bonus for exploration.
- Value Loss for state estimation.

### 5. Training Loop & Optimization
Stitching it all together into an iterative loop of:
1. Collecting experience.
2. Computing advantages/returns.
3. Updating parameters over multiple epochs.
4. Monitoring performance (Reward curves, KL divergence).

## 🚀 Environment Target
We will target **CartPole-v1** first for rapid iteration, then scale to **LunarLander-v2**.

## 📚 Key Reference
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

## Implementation Notes

- **Network architecture**: We use a shared MLP backbone with two hidden layers
  (64 units for CartPole, 128 units for LunarLander) and separate policy/value
  heads, with Tanh activations as in the PPO paper.
- **GAE + returns**: Implemented per-step using $(\gamma, \\lambda) = (0.99, 0.95)$
  and normalized advantages for stability.
- **PPO objective**: Clipped surrogate with clip range $\\epsilon = 0.2$, plus
  a clipped value loss and an entropy bonus for exploration.
- **Optimization & stability**: Adam optimizer, gradient clipping at 0.5, and
  optional KL-based early stopping on updates.
