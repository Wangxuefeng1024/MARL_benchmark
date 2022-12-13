# Multi-Agent Mujoco

This is the multi-agent version of Mujoco. Each agent/joint has a continuous observation and continuous action space, along with some basic simulated physics.
Used in the paper [Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2003.06709).


![](https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/mamujoco.jpg)

## Baselines
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)

## Results

<img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/Hopper.png" width="300px"> <img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/Swimmer.png" width="300px"> <img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/Walker.png" width="300px"> <img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/Ant.png" width="300px"> 

## Getting Started

### requirement

```shell
$ gym == 0.10.8
$ Mujoco == 2.1
$ Multi-Agent Mujoco (https://github.com/schroederdewitt/multiagent_mujoco)
```
### quick start

```shell
$ python runner/mujoco_runner.py --algo maddpg

