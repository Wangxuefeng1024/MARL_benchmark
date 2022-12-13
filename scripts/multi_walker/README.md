# Multi Walker

This environment is part of the [SISL environments](https://pettingzoo.farama.org/environments/sisl/). In this environment, bipedal robots attempt to carry a package placed on top of them towards the right. By default, the number of robots is set to 3.

Each walker receives a reward equal to the change in position of the package from the previous timestep, multiplied by the forward_reward scaling factor. The maximum achievable total reward depends on the terrain length; as a reference, for a terrain length of 75, the total reward under an optimal policy is around 300.

![](https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/sisl_multiwalker.gif)

## Baselines
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf)
- [TarMAC: Targeted Multi-Agent Communication (TarMAC)](http://proceedings.mlr.press/v97/das19a/das19a.pdf)

## Results

![](https://github.com/Wangxuefeng1024/MARL_benchmark/blob/main/results/muilti_walker.png)


## Getting Started

### requirement

```shell
$ gym == 0.10.8
$ pettingzoo
```
### quick start

```shell
$ python runner/multiwalker_runner.py --algo tarmac --n_agent 3
```



