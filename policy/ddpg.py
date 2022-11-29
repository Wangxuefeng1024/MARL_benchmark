# from algo.maddpg.network import Critic, Actor
from copy import deepcopy
from torch.optim import Adam
# from agent import base_agent
from utils.namedtuple_memory import ReplayMemory, Experience
from utils.utils import soft_update, hard_update
import os, torch
import torch.nn as nn
import numpy as np

scale_reward = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, dim_obs, dim_act, n_agents, args):
        self.args = args
        # self.mode = args.mode
        self.actors = []
        self.critics = []
        from network.maddpg import Actor, Critic
        self.actors = [Actor(dim_obs, dim_act, args) for _ in range(n_agents)]
        # self.critic = Critic(n_agents, dim_obs, dim_act)
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = args.episode_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]

        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self):
        if self.args.model_episode:
            path_flag = True
            for idx in range(self.n_agents):
                path_flag = path_flag \
                            and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth")) \
                            and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
                                                +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                for idx in range(self.n_agents):
                    actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                    self.actors[idx].load_state_dict(actor.state_dict())
                    self.critics[idx].load_state_dict(critic.state_dict())

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode):
        if not os.path.exists(self.args.model_dir + self.args.task + "/" + str(self.args.algo)):
            os.mkdir(self.args.model_dir + self.args.task + "/" + str(self.args.algo))
        for i in range(self.n_agents):
            torch.save(self.actors[i],
                       self.args.model_dir + self.args.task + "/" + str(self.args.algo) +'/'+ 'actor[' + str(i) + ']' + '_' + str(episode) + '.pth')
            torch.save(self.critics[i],
                       self.args.model_dir + self.args.task + "/" + str(self.args.algo) +'/'+ 'critic[' + str(i) + ']' + '_' + str(episode) + '.pth')

    def update(self, i_episode):

        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        for agent in range(self.n_agents):

            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i,:]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())
            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze() # .view(-1, self.n_agents * self.n_actions)

            # scale_reward: to scale reward in Q functions
            # reward_sum = sum([reward_batch[:,agent_idx] for agent_idx in range(self.n_agents)])
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch.unsqueeze(1))# + reward_sum.unsqueeze(1) * 0.1

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action).mean()
            # actor_loss += (action_i ** 2).mean() * 1e-3
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.actor_optimizer[agent].step()
            # self.critic_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.train_num % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return sum(a_loss).item()/self.n_agents, sum(c_loss).item()/self.n_agents

    def choose_action(self, state, action_mask = None, noisy=True):

        obs = torch.from_numpy(np.stack(state)).float().to(device)

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        with torch.no_grad():
            actions = torch.zeros(self.n_agents, self.n_actions)
            for i in range(self.n_agents):
                single_obs = obs[i].detach()
                act = self.actors[i](single_obs.unsqueeze(0)).squeeze()

                if noisy:
                    act += torch.from_numpy(np.random.randn(self.args.n_actions) * self.var[i]).type(FloatTensor)

                    if self.episode_done > self.episodes_before_train and \
                            self.var[i] > 0.05:
                        self.var[i] *= 0.999998
                act = torch.clamp(act, -1.0, 1.0)
                if action_mask is not None:
                    action = torch.zeros_like(act)
                    action = action + act[action_mask[i]]
                    actions[i, :] = action
                actions[i, :] = act
            self.steps_done += 1
            return actions.data.cpu().numpy()

class Cen_DDPG:

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.args = args
        self.batch_size = self.args.batch_size
        self.n_agents = n_agents
        self.device = device
        self.full_comm = args.full_comm
        # Networks
        if args.algo == "tarmac":
            from network.tarmac import Actor, Critic
        elif args.algo == "commnet":
            from network.commnet import Actor, Critic
        else:
            from network.tarmac import Actor, Critic
        self.actor = Actor(s_dim, a_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.actor_target = Actor(s_dim, a_dim, args)  # Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(self.n_agents, s_dim, a_dim, args)
        self.critic_target = Critic(self.n_agents, s_dim, a_dim, args)

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.args.a_lr)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.args.c_lr)

        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.args.epsilon_decay
        self.use_cuda = torch.cuda.is_available()

        self.c_loss = None
        self.a_loss = None
        # self.action_log = list()
        self.memory = ReplayMemory(1e5)

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
        if not os.path.exists("./trained_model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/"):
            os.mkdir("./trained_model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                self.args.n_agents) + "/")
        torch.save(self.actor.state_dict(),
                   "./trained_model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                       self.args.n_agents) + "/actor_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(),
                   "./trained_model/" + str(self.args.algo) + "/" + self.args.scenario + "/" + str(
                       self.args.n_agents) + "/critic_" + str(episode) + ".pth")

    def init_hidden(self):
        self.actor.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)
        self.actor_target.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)

    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor(np.array(obs)).to(self.device)

        with torch.no_grad():
            action, hidden_state, previous_hidden = self.actor(obs)
        action = action.detach().cpu().numpy()
        if noisy and self.args.scenario == "simple_spread":
            for agent_idx in range(self.n_agents):
                action[agent_idx] += np.random.randn(2) * self.var[agent_idx]

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

        batch = Experience(*zip(*transitions))
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
        current_Q = self.critic(whole_state, whole_action, previous_hidden).view(-1, self.n_agents)
        target_Q = self.critic_target(non_final_next_states, next_whole_batch, hidden_states).view(-1,
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
        actor_loss = -self.critic(whole_state, whole_action, previous_hidden).mean() * 0.3
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