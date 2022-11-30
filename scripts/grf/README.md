# Google Football Research

This repository contains an RL environment based on open-source game Gameplay Football.
It was created by the Google Brain team for research purposes.

![](https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/episode_done_20221108-154022258943.gif)


## Baselines
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning (QMIX)](https://arxiv.org/abs/1803.11485)

## Results

<img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/3vs1_winrates.png" width="300px"> <img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/rps_winrate.png" width="300px"> <img src="https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/ps_winrates.png" width="300px"> 



## Getting Started

### requirement

- gym-0.11.0
- [gfootball](https://github.com/google-research/football)

### reference

We follow the following observation & reward wrapper:

https://github.com/PKU-MARL/Multi-Agent-Transformer

### quick start

```shell
$ python runner/football_runner.py --algo qmix --scenario academy_3_vs_1_with_keeper
```





