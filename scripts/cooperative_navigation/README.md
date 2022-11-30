# Cooperative Navigation

This is a simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

`Cooperative Navigation` (simple_spread): N agents try to cover N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions.

## Selecting Baselines
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)
- [TarMAC: Targeted Multi-Agent Communication (TarMAC)](http://proceedings.mlr.press/v97/das19a/das19a.pdf)

## Environment Settings

## Results


## Getting Started

```shell
$ python runner/cooperative_navigation.py --algo tarmac --n_agent 10
```



