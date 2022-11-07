import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from utils.sc_memory import PPO_RolloutBuffer

# set device to cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
#         super(ActorCritic, self).__init__()
#
#         self.has_continuous_action_space = has_continuous_action_space
#
#         if has_continuous_action_space:
#             self.action_dim = action_dim
#             self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
#         # actor
#         if has_continuous_action_space:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#             )
#         else:
#             self.actor = nn.Sequential(
#                 nn.Linear(state_dim, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, action_dim),
#                 nn.Softmax(dim=-1)
#             )
#         # critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
#
#     def set_action_std(self, new_action_std):
#         if self.has_continuous_action_space:
#             self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
#         else:
#             print("--------------------------------------------------------------------------------------------")
#             print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
#             print("--------------------------------------------------------------------------------------------")
#
#     def forward(self):
#         raise NotImplementedError
#
#     def act(self, state):
#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)
#             cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
#             dist = MultivariateNormal(action_mean, cov_mat)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)
#
#         action = dist.sample()
#         action_logprob = dist.log_prob(action)
#
#         return action.detach(), action_logprob.detach()
#
#     def evaluate(self, state, action):
#
#         if self.has_continuous_action_space:
#             action_mean = self.actor(state)
#
#             action_var = self.action_var.expand_as(action_mean)
#             cov_mat = torch.diag_embed(action_var).to(device)
#             dist = MultivariateNormal(action_mean, cov_mat)
#
#             # For Single Action Environments.
#             if self.action_dim == 1:
#                 action = action.reshape(-1, self.action_dim)
#         else:
#             action_probs = self.actor(state)
#             dist = Categorical(action_probs)
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_values = self.critic(state)
#
#         return action_logprobs, state_values, dist_entropy


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