# Predator Prey

This is a simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.
Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

`Predator Prey` (simple_tag): Predator-prey environment. Good agents (green) are faster and want to avoid being hit by adversaries (red). Adversaries are slower and want to hit good agents. In this repo, we set this seeting as a cooperative one, which means we only train predator to tag the prey. Each agent observes several closest predators and closet preys.

## Baselines
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)
- [TarMAC: Targeted Multi-Agent Communication (TarMAC)](http://proceedings.mlr.press/v97/das19a/das19a.pdf)

## Results

![](https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/pp_tarmac.png)


## Getting Started

### requirement

```shell
$ gym == 0.10.8
```
### quick start

```shell
$ python runner/predator_prey.py --algo tarmac --n_agent 10
```



