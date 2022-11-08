import torch
import torch.nn as nn
import os
import numpy as np
from copy import deepcopy
# import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from utils.sc_memory import PPO_RolloutBuffer

# set device to cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAPPO:
    def __init__(self, state_dim, action_dim, n_agents, args):

        self.continuous = args.continuous

        if self.continuous:
            self.action_std = 0.6

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs
        self.args = args
        self.buffer = PPO_RolloutBuffer()
        # Networks
        if args.algo == "tarmac":
            from network.tarmac import Actor, Critic
        elif args.algo == "commnet":
            from network.commnet import Actor, Critic
        else:
            from network.tarmac import Actor, Critic
        self.actor = Actor(state_dim, action_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(self.n_agents, state_dim, action_dim, args)
        self.old_actor = Actor(state_dim, action_dim, args)
        self.actor.to(device)
        self.critic.to(device)
        self.old_actor.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.c_lr)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        #
        #
        # self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        #
        # self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.continuous:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.old_actor.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.continuous:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.continuous:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.old_actor.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.old_actor.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def evaluate(self, state, action):

        if self.continuous:
            action_mean = self.actor(state)

            action_var = self.actor.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.actor.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # actor loss
            action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm)
            self.actor_optimizer.step()

            # critic loss
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm)
            self.critic_optimizer.step()
            # final loss of clipped objective PPO
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class CenPPO:
    def __init__(self, state_dim, action_dim, n_agents, args):

        self.continuous = args.continuous

        if self.continuous:
            self.action_std = 0.6

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs
        self.args = args
        self.buffer = PPO_RolloutBuffer()
        # Networks
        if args.algo == "tarmac":
            from network.tarmac import Actor, Critic
        elif args.algo == "commnet":
            from network.commnet import Actor, Critic
        else:
            from network.tarmac import Actor, Critic
        self.actor = Actor(state_dim, action_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(self.n_agents, state_dim, action_dim, args)
        self.old_actor = Actor(state_dim, action_dim, args)
        self.actor.to(device)
        self.critic.to(device)
        self.old_actor.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.c_lr)

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.continuous:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.old_actor.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.continuous:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.continuous:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.old_actor.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.old_actor.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def evaluate(self, state, action):

        if self.continuous:
            action_mean = self.actor(state)

            action_var = self.actor.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.actor.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # actor loss
            action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm)
            self.actor_optimizer.step()

            # critic loss
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm)
            self.critic_optimizer.step()
            # final loss of clipped objective PPO
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class MAPPO_GRF:
    def __init__(self, env, args):

        self.continuous = args.continuous
        self.win_rates = []
        self.episode_rewards = []
        if self.continuous:
            self.action_std = 0.6
        self.env = env
        self.epsilon = args.epsilon
        self.n_agents = args.n_agents
        self.action_dim = 19
        self.state_dim = args.obs_shape
        self.gamma = args.gamma
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.eps_clip = 0.05
        self.K_epochs = args.k_epochs
        self.args = args
        self.buffer = PPO_RolloutBuffer()
        # Networks
        if args.algo == "tarmac":
            from network.tarmac import Actor, Critic
        elif args.algo == "commnet":
            from network.commnet import Actor, Critic
        else:
            from network.maddpg import Actor, Critic
        self.actor = Actor(self.state_dim, self.action_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(self.n_agents, self.state_dim, self.action_dim)
        self.old_actor = Actor(self.state_dim, self.action_dim, args)
        self.actor.to(device)
        self.critic.to(device)
        self.old_actor.to(device)
        self.action_var = torch.full((self.action_dim,), 0.6 * 0.6).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.c_lr)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        #
        #
        # self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        #
        # self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.continuous:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.old_actor.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.continuous:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, ava_actions):

        if self.continuous:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.act(state, ava_actions)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.act(state, ava_actions)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.item(), action_logprob.item()

    def act(self, state, ava_actions):
        if self.continuous:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            action_probs[ava_actions == 0] = 0.0
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate_value(self, state, action, action_onehot):
        batch_size = state.shape[0]
        if self.continuous:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.actor.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state.view(batch_size, -1).float(), action_onehot.view(batch_size, -1).float())

        return action_logprobs, state_values, dist_entropy

    def train(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_action_onehot = torch.squeeze(torch.stack(self.buffer.action_one_hot, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            for i in range(self.n_agents):
            logprobs, state_values, dist_entropy = self.evaluate_value(old_states, old_actions, old_action_onehot)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(torch.mean(logprobs) - torch.mean(old_logprobs).detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # actor loss
            action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm)
            self.actor_optimizer.step()

            # critic loss
            value_loss = 0.5 * self.MseLoss(state_values, rewards)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm)
            self.critic_optimizer.step()
            # final loss of clipped objective PPO
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        # if self.args.last_action:
        #     if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
        #         inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        #     else:
        #         inputs.append(u_onehot[:, transition_idx - 1])
        #     inputs_next.append(u_onehot[:, transition_idx])
        # if self.args.reuse_network:
        #     # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
        #     # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
        #     # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
        #     inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        #     inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next


    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        # o, u, r, s, avail_u, u_onehot, terminate, action_logs = [], [], [], [], [], [], [], []
        # self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # self.init_hidden(1)
        # done = False
        obs, state, ava = self.env.reset()
        # s.append(np.array(state).flatten())
        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            # time.sleep(0.2)
            # check_obs = self.env.observation()
            # cc, aa = tranf_obs(check_obs, self.feature_encoder)
            # observation = self.env.observation()[1:self.args.n_agents+1]
            # obs, ava_actions = tranf_obs(observation, self.feature_encoder)
            # # self.env.render()
            # # obs = obs.squeeze()
            # # obs = self.feature_encoder.encode(observation)
            # # obs = football_observation_wrapper(observation)
            # # obs = self.env.observation(observation)
            # # aa = [-i-1 for i in range(self.n_agents*2)]
            # state = obs.flatten()
            actions, actions_onehot, log_action = [], [], []
            for agent_id in range(self.n_agents):

                # avail_action = self.env.get_avail_agent_actions(agent_id)
                action, action_log = self.select_action(obs[agent_id], ava[agent_id])
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                log_action.append(torch.tensor(action_log).to(device))
                actions.append(torch.tensor(np.int(action)).to(device))
                actions_onehot.append(torch.from_numpy(action_onehot).to(device))
                # avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            all_actions = deepcopy(actions)
            all_actions.insert(0, 0)
            self.buffer.states.append(torch.tensor(obs[1:]).to(device))
            # self.env.env.env.render()
            obs, state, rewards, dones, infos, ava = self.env.step(all_actions)
            terminated = True if True in dones else False
            # if done:
            #     print('debug here')
            # next_obs = next_obs[1:self.args.n_agents+1]
            # rewards = calc_reward(reward, observation[0], next_obs[0], step)

            # rewards = sum(rewards)
            # next_obs = football_observation_wrapper(next_obs)
            # reward = football_reward_wrapper(next_obs, reward)
            for value in infos:
                if value['score_reward'] > 0:
                    win_tag = True
            # o.append(obs)
            self.buffer.logprobs.append(torch.stack(log_action))
            # action_logs.append(log_action)
            # s.append(np.array(state).flatten())
            self.buffer.actions.append(torch.stack(actions))
            self.buffer.action_one_hot.append(torch.stack(actions_onehot))
            self.buffer.avail_actions.append(torch.tensor(ava).to(device))
            step += 1
            self.buffer.rewards.append(torch.sum(torch.tensor(rewards)).float().to(device))
            self.buffer.is_terminals.append(terminated)
            # padded.append([0.])
            episode_reward += np.sum(rewards)

        #     # if self.args.epsilon_anneal_scale == 'step':
        #     #     epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        #     # not win_tag and step < self.args.episode_limit:
        # # last obs
        # # obs = self.env.observation()[1:self.args.n_agents+1]
        # # obs, ava_actions = tranf_obs(obs, self.feature_encoder)
        # # obs = obs.squeeze()
        # # state = obs.flatten()
        # # o.append(obs)
        # # s.append(state)
        # o_next = o[1:]
        # s_next = s[1:]
        # o = o[:-1]
        # s = s[:-1]
        # # get avail_action for last obs，because target_q needs avail_action in training
        # # avail_actions = []
        # if not terminated:
        #     actions, actions_onehot = [], []
        #     for agent_id in range(self.n_agents):
        #         # avail_action = self.env.get_avail_agent_actions(agent_id)
        #         action = self.select_action(obs[agent_id], ava[agent_id])
        #         # generate onehot vector of th action
        #         action_onehot = np.zeros(self.args.n_actions)
        #         action_onehot[action] = 1
        #         actions.append(np.int(action))
        #         actions_onehot.append(action_onehot)
        #         # avail_actions.append(avail_action)
        #         last_action[agent_id] = action_onehot
        #
        #     obs, state, rewards, dones, infos, ava = self.env.step(actions)
        # else:
        #     ava = np.zeros([self.args.n_agents, self.args.n_actions])
        #     # avail_actions.append(ava)
        # # observation = self.env.observation()[1:self.args.n_agents + 1]
        # # obs, ava_actions = tranf_obs(observation, self.feature_encoder)
        # avail_u.append(ava)
        # avail_u_next = avail_u[1:]
        # avail_u = avail_u[:-1]
        #
        # # if step < self.episode_limit，padding
        # for i in range(step, self.episode_limit):
        #     o.append(np.zeros((self.n_agents, self.state_dim)))
        #     u.append(np.zeros([self.n_agents, 1]))
        #     s.append(np.zeros(self.state_dim))
        #     r.append([0.])
        #     o_next.append(np.zeros((self.n_agents, self.state_dim)))
        #     s_next.append(np.zeros(self.state_dim))
        #     u_onehot.append(np.zeros((self.n_agents, self.action_dim)))
        #     avail_u.append(np.zeros((self.n_agents, self.action_dim)))
        #     avail_u_next.append(np.zeros((self.n_agents, self.action_dim)))
        #     # padded.append([1.])
        #     terminate.append([1.])

        # episode = dict(o=o.copy(),
        #                s=s.copy(),
        #                u=u.copy(),
        #                r=r.copy(),
        #                avail_u=avail_u.copy(),
        #                # o_next=o_next.copy(),
        #                # s_next=s_next.copy(),
        #                # avail_u_next=avail_u_next.copy(),
        #                u_onehot=u_onehot.copy(),
        #                # padded=padded.copy(),
        #                terminated=terminate.copy()
        #                )
        # add episode dim
        # for key in episode.keys():
        #     episode[key] = np.array([episode[key]])
        # episode['r'] = episode['r'].reshape((1, self.args.episode_limit, 1))
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return self.buffer, episode_reward, win_tag, step

    # def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
    #
    #     # different episode has different length, so we need to get max length of the batch
    #     max_episode_len = self._get_max_episode_len(batch)
    #     for key in batch.keys():
    #         if key != 'z':
    #             batch[key] = batch[key][:, :max_episode_len]
    #     self.learn(batch, max_episode_len, train_step, epsilon)
    #     if train_step > 0 and train_step % self.args.save_cycle == 0:
    #         self.save_model(train_step)

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
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

