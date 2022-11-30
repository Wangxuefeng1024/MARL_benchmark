import torch.nn as nn
from utils.namedtuple_memory import ReplayMemory_rnn, Experience_rnn
import torch, os
import numpy as np
from torch.autograd import Variable
from utils.util import soft_update, hard_update
from network.tarmac import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical


class TarMAC():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.args = args
        self.batch_size = self.args.batch_size
        self.n_agents = n_agents
        self.device = device
        self.full_comm = args.full_comm
        # Networks
        self.actor = Actor(s_dim, a_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.actor_target = Actor(s_dim, a_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(self.n_agents, s_dim, a_dim, args)
        self.critic_target = Critic(self.n_agents, s_dim, a_dim, args)

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.args.a_lr)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.args.a_lr)

        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.args.epsilon_decay
        self.use_cuda = torch.cuda.is_available()

        self.c_loss = None
        self.a_loss = None
        # self.action_log = list()
        self.memory = ReplayMemory_rnn(1e5)

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.args.algo) + '/' + str(self.args.scenario) + '/' + str(
            self.args.n_agents) + "/actor_100000.pth"
        # model_critic_path = "./trained_model/" + str(self.args.algo) + "/critic_" + str(self.args.model_episode) + ".pth"
        if os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            # critic = torch.load(model_critic_path)
            self.actor.load_state_dict(actor)
            # self.critic.load_state_dict(critic)

    def save_model(self, episode):
        if not os.path.exists("../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/"):
            os.mkdir("../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/")
        torch.save(self.actor.state_dict(),
                   "../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                       self.args.n_agents) + "/actor_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(),
                   "../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                       self.args.n_agents) + "/critic_" + str(episode) + ".pth")

    def init_hidden(self):
        self.actor.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)
        self.actor_target.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)

    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor(np.array(obs)).to(self.device)

        with torch.no_grad():
            action, hidden_state, previous_hidden = self.actor(obs)
        action = action.detach().cpu().numpy()
        if noisy:
            for agent_idx in range(self.n_agents):
                action[agent_idx] += np.random.randn(self.a_dim) * self.var[agent_idx]

                if self.var[agent_idx] > 0.05:
                    self.var[agent_idx] *= 0.999998  # 0.999998

        action = np.clip(action, -1., 1.)
        # print(action)
        return action, hidden_state, previous_hidden

    def update(self, i_episode):
        self.train_num = i_episode
        if self.train_num <= self.args.episode_before_train:
            self.init_hidden()
            return None, None

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)

        batch = Experience_rnn(*zip(*transitions))
        with torch.no_grad():
            state_batch = torch.stack(batch.states).type(FloatTensor)
            hidden_states = torch.stack(batch.hidden_states)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack(batch.next_states).type(
                FloatTensor)  # torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            whole_state = state_batch.view(self.batch_size, self.n_agents, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            action_batch.view(self.batch_size, self.n_agents, -1)
            previous_hidden = torch.stack(batch.previous_hidden).type(FloatTensor)
            # if self.args.scenario == 'traffic_junction':

            next_whole_batch = self.actor_target(non_final_next_states, hidden_data=hidden_states).view(
                self.batch_size, -1)
        if len(reward_batch.shape) == 1:
            reward_batch = reward_batch.unsqueeze(1).expand(self.args.batch_size, self.args.n_agents)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        whole_state = whole_state.view(self.batch_size, -1)
        non_final_next_states = non_final_next_states.view(self.batch_size, -1)
        current_Q = self.critic(whole_state, previous_hidden).view(-1, self.n_agents)
        target_Q = self.critic_target(non_final_next_states, hidden_states).view(-1,
                                                                                                   self.n_agents)  # .view(-1, self.n_agents * self.n_actions)
        target_Q = target_Q * self.GAMMA + reward_batch
        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()

        whole_action = self.actor(whole_state, hidden_data=previous_hidden).view(self.batch_size, -1)
        actor_loss = -self.critic(whole_state, previous_hidden).mean() * 0.3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.actor_optimizer.step()

        c_loss.append(loss_Q.item())
        a_loss.append(actor_loss.item())
        self.init_hidden()
        if self.train_num % 200 == 0:
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

        return sum(c_loss) / len(c_loss), sum(a_loss) / len(a_loss)


class TJ_TarMAC():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.args = args
        self.batch_size = self.args.batch_size
        self.n_agents = n_agents
        self.device = device
        self.full_comm = args.full_comm
        self.episode = 0
        # Networks
        self.actor = Actor(s_dim, a_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.actor.to(self.device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.args.a_lr)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.args.epsilon_decay
        self.use_cuda = torch.cuda.is_available()
        self.a_loss = None
        # self.action_log = list()
        self.memory = ReplayMemory_rnn(1e5)
        self.GAMMA = 0.95
        self.tau = 0.01
        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0
        self.saved_log_probs = []
        self.rewards = []

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.args.algo) + '/' + str(
            self.args.scenario) + "/actor_200000.pth"

        if os.path.exists(model_actor_path):
            print('load_model!')
            actor = torch.load(model_actor_path)
            self.actor.load_state_dict(actor)

    def save_model(self, episode):
        if not os.path.exists("../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/"):
            os.mkdir("../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/")
        torch.save(self.actor.state_dict(),
                   "../model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                       self.args.n_agents) + "/actor_" + str(episode) + ".pth")

    def init_hidden(self):
        self.actor.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)

    def choose_action(self, obs):
        obs = Variable(torch.Tensor(obs).to(self.device))
        # with torch.no_grad():
        action_prob = self.actor(obs)
        # action_prob = action_prob.detach().cpu().numpy()
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = - (action_prob * action_prob.log()).sum()
        # print(action)
        return action, log_prob, entropy

    def update(self, episode, rewards, log_probs, entropies):
        self.episode = episode
        R = torch.zeros(1, 1).to(device)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.GAMMA * R + rewards[i]
            loss = loss - (log_probs[i] * Variable(R).cuda()).sum() - (
                    0.001 * entropies[i].cuda()).sum()
        loss = loss / len(rewards)

        self.actor_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        self.init_hidden()
        return loss.item()
