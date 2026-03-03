# LLM Post-Training & RL Curriculum

This repository contains a sequence of projects designed to bridge the gap between core Reinforcement Learning (RL) foundations and cutting-edge LLM post-training techniques (SFT, DPO, GRPO).

## Curriculum Overview

The transition to post-training requires two main pillars:
1.  **RL Intuition**: Understanding how policy updates and reward signals interact.
2.  **LLM Alignment**: Applying these signals to high-dimensional language models efficiently.

---

### Project 1: "RL Zero" - PPO from Scratch
**Goal:** Implement **Proximal Policy Optimization (PPO)** in PyTorch for standard gymnasium environments.
- **Key Concepts:** Actor-Critic, Clipped Surrogate Objective, GAE, Entropy exploration.

### Project 2: "Instruction Tuner" - SFT & PEFT
**Goal:** Supervised Fine-Tuning (SFT) on a 1B-3B model (e.g., Llama-3.2-1B) for specialized domains.
- **Key Concepts:** LoRA, QLoRA, ChatML formatting, Gradient Checkpointing.

### Project 3: "Preference Alignment" - DPO
**Goal:** Align an SFT model using Direct Preference Optimization (DPO).
- **Key Concepts:** Pairwise preferences, Reference model stabilization, KL-divergence.

### Project 4: "The Reasoning Engine" - GRPO (DeepSeek Style)
**Goal:** Group Relative Policy Optimization (GRPO) for math/coding reasoning.
- **Key Concepts:** Rule-based rewards (verifiability), Group Normalization, CoT.

### Project 5: Evaluation & Benchmarking
**Goal:** Measure success using LLM-as-a-Judge and standard benchmarks like GSM8K.
