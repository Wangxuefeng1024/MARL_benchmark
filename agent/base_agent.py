import torch
import os
from Algos.iac.network import CNN_Actor, Actor, CNN_Critic, Critic, Cen_Critic, ICMNetwork
from tensorboardX import SummaryWriter
import numpy as np
from torch.distributions import Categorical

import torch.nn.functional as F
from torch.distributions import MultivariateNormal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn

class base_agent:
    def __init__(self, env, args):
        # set up the base information
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.map +"_"+str(args.n_agents))

        # self.state_shape = args.state_shape
        self.obs_shape = args.n_states
        self.env = env
        input_shape = self.obs_shape
        # whether to use last action & agent ID
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.win_rates = []
        self.episode_rewards = []
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.env + '/' + args.map

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        print('Init algo ', args.algo)

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        raise NotImplementedError

    def _get_inputs(self, batch, transition_idx):
        raise NotImplementedError

    def generate_episode(self, episode_num=None, evaluate=False):
        raise NotImplementedError

    def choose_action(self, obs, agent_num):
        raise NotImplementedError

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        raise NotImplementedError

    def get_q_values(self, batch, max_episode_len):
        raise NotImplementedError

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # obtain the prob of all actions
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # mask the unavailable actions

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len