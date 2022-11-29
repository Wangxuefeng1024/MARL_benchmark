import torch.nn as nn
from utils.namedtuple_memory import ReplayMemory, Experience
import torch, os
import numpy as np, random
from torch.distributions import Categorical

from utils.util import soft_update, hard_update
from network.commnet import Actor, Critic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

class CommNet():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = args
        self.batch_size = self.config.batch_size
        self.n_agents = n_agents
        self.device = device
        # Networks
        self.actor = Actor(s_dim,a_dim,n_agents,args) #Actor(s_dim, a_dim, n_agents)
        self.actor_target = Actor(s_dim,a_dim,n_agents,args) #Actor(s_dim, a_dim, n_agents)
        self.critic = Critic(s_dim,a_dim,n_agents,args) #Actor(s_dim, a_dim, n_agents)
        self.critic_target = Critic(s_dim,a_dim,n_agents,args) #Actor(s_dim, a_dim, n_agents)

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.a_lr)

        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay

        self.c_loss = None
        self.a_loss = None
        self.action_log = list()
        self.memory = ReplayMemory(1e5)
        self.GAMMA = self.config.gamma
        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0

    def load_model(self):
        model_actor_path = "../model/" + str(self.config.algo) + "/actor_" + str(200000) + ".pth"
        model_critic_path = "../model/" + str(self.config.algo) + "/critic_" + str(200000) + ".pth"
        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            critic = torch.load(model_critic_path)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)

    def save_model(self, episode):
        if not os.path.exists("../model/" + str(self.config.algo)+ "/" + self.config.scenario + "/" + str(self.config.n_agents)+"/"):
            os.mkdir("../model/" + str(self.config.algo)+ "/" + self.config.scenario + "/" + str(self.config.n_agents)+"/")
        torch.save(self.actor.state_dict(),
                   "../model/" + str(self.config.algo)+ "/"  + self.config.scenario + "/" + str(self.config.n_agents) + "/actor_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(),
                   "../model/" + str(self.config.algo)+ "/"  + self.config.scenario + "/" + str(self.config.n_agents) + "/critic_" + str(episode) + ".pth")


    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor(np.array(obs)).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).detach().cpu().numpy()

        if noisy and self.config.scenario=='simple_spread':
            for agent_idx in range(self.n_agents):
                action[agent_idx] += np.random.randn(2) * self.var[agent_idx]

                if self.var[agent_idx] > 0.05:
                    self.var[agent_idx] *= 0.999998#0.999998

            action = np.clip(action, -1., 1.)
        # print(action)
        return action

    def prep_train(self):
        self.actor.train()
        self.critic.train()

    def prep_eval(self):
        self.actor.eval()
        self.critic.eval()

    def get_batches(self):
        experiences = random.sample(self.replay_buffer, self.config.batch_size)

        state_batches = np.array([_[0] for _ in experiences])
        action_batches = np.array([_[1] for _ in experiences])
        reward_batches = np.array([_[2] for _ in experiences])
        next_state_batches = np.array([_[3] for _ in experiences])
        done_batches = np.array([_[4] for _ in experiences])
        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def update(self,i_episode):

        # if len(self.memory.memory) < self.batch_size:
        #     return None, None
        self.use_cuda = torch.cuda.is_available()

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.stack(batch.states).type(FloatTensor)
        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        non_final_next_states = torch.stack(batch.next_states).type(FloatTensor)#torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
        whole_state = state_batch.view(self.batch_size, self.n_agents, -1)
        whole_action = action_batch.view(self.batch_size, self.n_agents, -1)
        action_batch.view(self.batch_size, self.n_agents, -1)
        next_whole_batch = self.actor_target(non_final_next_states).view(self.batch_size, self.n_agents, -1)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        current_Q = self.critic(whole_state,whole_action).view(-1, self.n_agents)
        target_Q = self.critic_target(non_final_next_states,next_whole_batch).view(-1, self.n_agents) # .view(-1, self.n_agents * self.n_actions)
        target_Q = target_Q * self.GAMMA + reward_batch
        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        loss_Q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        whole_action = self.actor(whole_state).view(self.batch_size, self.n_agents, -1)
        actor_loss = -self.critic(whole_state, whole_action).mean()*100
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.actor_optimizer.step()
        c_loss.append(loss_Q.item())
        a_loss.append(actor_loss.item())
        self.train_num = i_episode
        if self.train_num % 200 == 0:
            soft_update(self.actor, self.actor_target, self.config.tau)
            soft_update(self.critic, self.critic_target, self.config.tau)

        return sum(c_loss) / len(c_loss), sum(a_loss) / len(a_loss)

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_action_std(self):
        return np.array(self.action_log).std(axis=-1).mean()

class TJ_CommNet():

    def __init__(self, s_dim, a_dim, n_agents, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.config = args
        self.batch_size = self.config.batch_size
        self.n_agents = n_agents
        self.device = device
        # Networks
        self.actor = Actor(s_dim,a_dim,n_agents,args)#Actor(s_dim, a_dim, n_agents)
        self.actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.a_lr)

        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.config.epsilon_decay
        self.a_loss = None
        self.action_log = list()
        self.memory = ReplayMemory(1e5)
        self.GAMMA = self.config.gamma
        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0

    def load_model(self):
        model_actor_path = "../model/" + str(self.config.algo)+"/" + self.config.scenario + "/" + str(self.config.n_agents) + "/actor_" + str(200000) + ".pth"
        if os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            self.actor.load_state_dict(actor)

    def save_model(self, episode):
        if not os.path.exists("../model/" + str(self.config.algo)+ "/" + self.config.scenario + "/" + str(self.config.n_agents)+"/"):
            os.mkdir("../model/" + str(self.config.algo)+ "/" + self.config.scenario + "/" + str(self.config.n_agents)+"/")
        torch.save(self.actor.state_dict(),
                   "../model/" + str(self.config.algo)+ "/"  + self.config.scenario + "/" + str(self.config.n_agents) + "/actor_" + str(episode) + ".pth")

    def choose_action(self, obs):
        obs = torch.Tensor(np.array(obs)).to(self.device)

        action_prob = self.actor(obs)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = - (action_prob*action_prob.log()).sum()
        # print(action)
        return action, log_prob, entropy

    def update(self, episode, rewards, log_probs, entropies):
        self.episode = episode
        R = torch.zeros(1, 1).to(device)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.GAMMA * R + rewards[i]
            loss = loss - (log_probs[i] * R).sum() - (
                        0.001 * entropies[i].cuda()).sum()
        loss = loss / len(rewards)

        self.actor_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        return loss.item()