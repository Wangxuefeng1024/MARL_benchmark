# MARL Benchmarks

Hi there! This repository collects some popular MARL environments, and implements several algorithms on top of them.
I provide detailed information for each environment, you can check it under its directory.

## Environments
- [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac),
- [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs),
- [Traffic Junction (TJ)](https://github.com/IC3Net/IC3Net),
- [Google Research Football (GRF)](https://github.com/google-research/football),
- [Multi-agent Mujoco](https://github.com/schroederdewitt/multiagent_mujoco),
- Payoff Matrix

## Corresponding Algorithms
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning (VDN)](https://arxiv.org/abs/1706.05296)
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (QMIX)](https://arxiv.org/abs/1803.11485)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)
- [Weighted QMIX: Expanding Monotonic Value Function Factorisation (WQMIX)](https://arxiv.org/abs/2006.10800) 
- [TarMAC: Targeted Multi-Agent Communication](http://proceedings.mlr.press/v97/das19a/das19a.pdf)

## TODO List

- [ ] Qtran
- [ ] PAC
- [ ] Model-based algorithms
- [ ] Other SOTA MARL algorithms
- [ ] Update results

## Code Structure

- `./scripts`: contains code for runnning the training code.

- `./runs`: contains training logs

- `./Envs`: contains environment files are used throughout the code. And installation procedure for each one.

- `./model`: used for saving the trained models.

- `./network`: neural network for each algorithm.

- `./policy`: contains the algorithms of DQN, PPO, DDPG, REINFORCE.

## Quick Start

```shell
$ python scripts/sc_main.py --map=3m --algo=qmix
```

Directly run the `sc_main.py`, then the algorithm will start **training** on map `3m`. 


