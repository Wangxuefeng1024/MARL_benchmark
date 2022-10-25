import torch.nn as nn
from algo.memory import ReplayMemory_rnn, Experience_rnn, ReplayMemory_eva, Experience_eva
import torch, os
import numpy as np
from torch.autograd import Variable
# from Algos.utils import soft_update, device, tranform_adj
from Algos.TarMAC.network import Actor, Critic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical
from torch.distributions import Categorical
from Utils.utils import tranf_obs
from Utils.encoder_basic import FeatureEncoder
from Utils.rewarder_basic import calc_reward
from Utils.sc_memory import ReplayBuffer
from copy import deepcopy
import torch
import os
import torch.functional as F
# from network.ppo_net import PPOActor
# from network.ppo_net import PPOCritic

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
        self.actor = Actor(s_dim, a_dim, args)#Actor(s_dim, a_dim, n_agents)
        self.actor_target = Actor(s_dim, a_dim, args)#Actor(s_dim, a_dim, n_agents)
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
        self.memory = ReplayMemory_eva(1e5)

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.args.algo)+'/'+str(self.args.scenario)+'/'+str(self.args.n_agents)  + "/actor_100000.pth"
        # model_critic_path = "./trained_model/" + str(self.args.algo) + "/critic_" + str(self.args.model_episode) + ".pth"
        if os.path.exists(model_actor_path):
            print("load model!")
            actor = torch.load(model_actor_path)
            # critic = torch.load(model_critic_path)
            self.actor.load_state_dict(actor)
            # self.critic.load_state_dict(critic)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.args.algo)+ "/" + self.args.scenario + "/" + str(self.args.n_agents)+"/"):
            os.mkdir("./trained_model/" + str(self.args.algo)+ "/" + self.args.scenario + "/" + str(self.args.n_agents)+"/")
        torch.save(self.actor.state_dict(),
                   "./trained_model/" + str(self.args.algo)+ "/"  + self.args.scenario + "/" + str(self.args.n_agents) + "/actor_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(),
                   "./trained_model/" + str(self.args.algo)+ "/"  + self.args.scenario + "/" + str(self.args.n_agents) + "/critic_" + str(episode) + ".pth")


    def init_hidden(self):
        self.actor.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)
        self.actor_target.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)

    def choose_action(self, obs, noisy=True):
        obs = torch.Tensor(np.array(obs)).to(self.device)

        with torch.no_grad():
            action, hidden_state, previous_hidden = self.actor(obs)
        action = action.detach().cpu().numpy()
        if noisy and self.args.scenario=="simple_spread":
            for agent_idx in range(self.n_agents):
                action[agent_idx] += np.random.randn(2) * self.var[agent_idx]

                if self.var[agent_idx] > 0.05:
                    self.var[agent_idx] *= 0.999998#0.999998

        action = np.clip(action, -1., 1.)
        # print(action)
        return action, hidden_state, previous_hidden

    def update(self,i_episode):
        self.train_num = i_episode
        if self.train_num <= self.args.episode_before_train:
            self.init_hidden()
            return None, None

        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)

        batch = Experience_eva(*zip(*transitions))
        with torch.no_grad():
            state_batch = torch.stack(batch.states).type(FloatTensor)
            hidden_states = torch.stack(batch.hidden_states)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack(batch.next_states).type(FloatTensor)#torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
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
        current_Q = self.critic(whole_state,whole_action, previous_hidden).view(-1, self.n_agents)
        target_Q = self.critic_target(non_final_next_states, next_whole_batch, hidden_states).view(-1, self.n_agents) # .view(-1, self.n_agents * self.n_actions)
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
        actor_loss = -self.critic(whole_state, whole_action, previous_hidden).mean()*0.3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.actor_optimizer.step()

        c_loss.append(loss_Q.item())
        a_loss.append(actor_loss.item())
        self.init_hidden()
        # if self.train_num % 200 == 0:
        #     soft_update(self.actor, self.actor_target, self.tau)
        #     soft_update(self.critic, self.critic_target, self.tau)

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
        self.actor = Actor(s_dim, a_dim, args)#Actor(s_dim, a_dim, n_agents)
        self.actor.to(self.device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.args.a_lr)
        self.replay_buffer = list()
        self.epsilon = 1.
        self.depsilon = self.epsilon / self.args.epsilon_decay
        self.use_cuda = torch.cuda.is_available()
        self.a_loss = None
        # self.action_log = list()
        self.memory = ReplayMemory_eva(1e5)
        self.GAMMA = 0.95
        self.tau = 0.01
        self.var = [1.0 for i in range(n_agents)]
        self.episode = 0
        self.saved_log_probs = []
        self.rewards = []

    def load_model(self):
        model_actor_path = "./trained_model/" + str(self.args.algo)+'/'+str(self.args.scenario)+ "/actor_200000.pth"

        if os.path.exists(model_actor_path):
            print('load_model!')
            actor = torch.load(model_actor_path)
            self.actor.load_state_dict(actor)

    def save_model(self, episode):
        if not os.path.exists("./trained_model/" + str(self.args.algo)+ "/" + self.args.scenario + "/" + str(self.args.n_agents)+"/"):
            os.mkdir("./trained_model/" + str(self.args.algo)+ "/" + self.args.scenario + "/" + str(self.args.n_agents)+"/")
        torch.save(self.actor.state_dict(),
                   "./trained_model/" + str(self.args.algo)+ "/"  + self.args.scenario + "/" + str(self.args.n_agents) + "/actor_" + str(episode) + ".pth")

    def init_hidden(self):
        self.actor.hidden_state = torch.zeros(self.args.n_agents, self.args.rnn_hidden_size).to(device)

    def choose_action(self, obs):
        obs = Variable(torch.Tensor(obs).to(self.device))

        # with torch.no_grad():
        action_prob, _, _ = self.actor(obs)
        # action_prob = action_prob.detach().cpu().numpy()
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
            loss = loss - (log_probs[i] * Variable(R).cuda()).sum() - (
                        0.001 * entropies[i].cuda()).sum()
        loss = loss / len(rewards)

        self.actor_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        self.init_hidden()
        return loss.item()

class gfoot_TarMAC:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.n_states
        self.obs_shape = args.obs_shape
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.feature_encoder = FeatureEncoder()
        input_shape = self.obs_shape
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = ReplayBuffer(args)
        self.save_path = self.args.result_dir + '/' + args.algo + '/' + args.scenario
        # 根据参数决定RNN的输入维度
        # if args.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents
        # 神经网络
        self.eval_rnn = Actor(self.obs_shape, self.n_actions, args)  #Actor(s_dim, a_dim, n_agents) # 每个agent选动作的网络
        self.target_rnn = Actor(self.obs_shape, self.n_actions, args)

        # self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        # self.target_qmix_net = QMixNet(args)
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            # self.eval_qmix_net.cuda()
            # self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.scenario
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_actor = self.model_dir + '/rnn_net_params.pkl'
                # path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_actor, map_location=map_location))
                # self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                # print('Successfully load the model: {} and {}'.format(path_actor, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        # self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_rnn.parameters(), lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo TarMAC')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        # q_total_eval = self.eval_qmix_net(q_evals, s)
        # q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_targets * (1 - terminated)

        td_error = (q_evals - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_rnn.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            # self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

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

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            # if len(inputs.shape) == 2:
            inputs = inputs.view(-1, self.n_agents, self.obs_shape)
            inputs_next = inputs_next.view(-1, self.n_agents, self.obs_shape)
            q_eval = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

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
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.init_hidden(1)
        # done = False
        obs = self.env.reset()

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            # time.sleep(0.2)
            # check_obs = self.env.observation()
            # cc, aa = tranf_obs(check_obs, self.feature_encoder)
            observation = self.env.observation()[1:self.args.n_agents+1]
            obs, ava_actions = tranf_obs(observation, self.feature_encoder)
            # self.env.render()
            # obs = obs.squeeze()
            # obs = self.feature_encoder.encode(observation)
            # obs = football_observation_wrapper(observation)
            # obs = self.env.observation(observation)
            # aa = [-i-1 for i in range(self.n_agents*2)]
            state = obs.flatten()
            actions, actions_onehot = [], []
            # for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
            action = self.choose_action(obs, ava_actions, epsilon)
            # generate onehot vector of th action
            for agent_id in range(self.n_agents):
                action_onehot = np.zeros(self.args.n_actions)
                # try:
                # actionss = action[agent_id]
                action_onehot[action[agent_id]] = 1
                # except:
                #     print('debug here')
                actions.append(np.int(action[agent_id]))
                try:
                    actions_onehot.append(action_onehot)
                except:
                    print('debug here')
                # avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            all_actions = deepcopy(actions)
            all_actions.insert(0, 0)
            next_obs, reward, terminated, info = self.env.step(all_actions)
            # if done:
            #     print('debug here')
            next_obs = next_obs[1:self.args.n_agents+1]
            rewards = calc_reward(reward, observation[0], next_obs[0], step)

            rewards = sum(rewards)
            # next_obs = football_observation_wrapper(next_obs)
            # reward = football_reward_wrapper(next_obs, reward)
            if info['score_reward'] > 0:
                win_tag = True
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(ava_actions)
            step += 1
            r.append([rewards])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += rewards

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            # not win_tag and step < self.args.episode_limit:
        # last obs
        obs = self.env.observation()[1:self.args.n_agents+1]
        obs, ava_actions = tranf_obs(obs, self.feature_encoder)
        obs = obs.squeeze()
        state = obs.flatten()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        # avail_actions = []
        # for agent_id in range(self.n_agents):
        #     avail_action = self.env.get_avail_agent_actions(agent_id)
        #     avail_actions.append(avail_action)
        # observation = self.env.observation()[1:self.args.n_agents + 1]
        # obs, ava_actions = tranf_obs(observation, self.feature_encoder)
        avail_u.append(ava_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step

    def choose_action(self, obs, ava_actions, epsilon):
        inputs = obs.copy()
        ava_actions_ind = []
        for i in range(self.n_agents):
            avail_actions_i= np.nonzero(ava_actions[i]) # index of actions which can be choose
            ava_actions_ind.append(avail_actions_i)
        # transform agent_num to onehot vector

        # ava_actions_ind_a = np.array(ava_actions_ind[0]).squeeze()
        # if self.args.last_action:
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))
        hidden_state = self.eval_hidden

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value

        q_value = self.eval_rnn(inputs, hidden_state)

        # choose action from q value
        # ava_actions = ava_actions.unsqueeze(0)
        q_value = q_value.squeeze()
        q_value[ava_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = []
            for z in range(len(ava_actions_ind)):
                action_i = np.random.choice(np.array(ava_actions_ind[z]).squeeze())
                action.append(action_i)  # action是一个整数
        else:
            action = torch.argmax(q_value, dim=1)
        return action

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.save_model(train_step)

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

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action




class MAPPO:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.n_states
        self.obs_shape = args.obs_shape
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.feature_encoder = FeatureEncoder()
        input_shape = self.obs_shape
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = ReplayBuffer(args)
        self.save_path = self.args.result_dir + '/' + args.algo + '/' + args.scenario
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        # if args.last_action:
        #     actor_input_shape += self.n_actions
        # if args.reuse_network:
        #     actor_input_shape += self.n_agents
        self.args = args

        self.policy_rnn = Actor(self.obs_shape, self.n_actions, args)
        self.eval_critic = Critic(self.n_agents, self.obs_shape, self.n_actions, args)
        # self.target_critic = PPOCritic(critic_input_shape, self.args)

        if self.args.cuda:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            # self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.scenario

        # if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
        #     if os.path.exists(self.model_dir + '/rnn_params.pkl'):
        #         path_rnn = self.model_dir + '/rnn_params.pkl'
        #         path_coma = self.model_dir + '/critic_params.pkl'
        #         map_location = 'cuda:0' if self.args.use_gpu else 'cpu'
        #         self.policy_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
        #         self.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
        #         print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
        #     else:
        #         raise Exception("No model!")

        # self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.ac_parameters = list(self.policy_rnn.parameters()) + list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop(self.ac_parameters, lr=args.lr)
        elif args.optimizer == "Adam":
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=args.lr)

        self.args = args

        self.policy_rnn.policy_hidden = None
        self.eval_critic_hidden = None
        # self.target_critic_hidden = None

    def _get_critic_input_shape(self):
        # state
        input_shape = self.state_shape
        # obs
        input_shape += self.obs_shape
        # agent_id
        input_shape += self.n_agents

        # input_shape += self.n_actions * self.n_agents * 2  # 54
        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated, s = batch['u'], batch['r'], batch['avail_u'], batch['terminated'], batch['s']

        mask = (1 - batch["padded"].float())

        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            s = s.cuda()

        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        old_values, _ = self._get_values(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)
        old_action_prob = self._get_action_prob(batch, max_episode_len)

        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)

            values, target_values = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            returns = torch.zeros_like(r)
            deltas = torch.zeros_like(r)
            advantages = torch.zeros_like(r)

            prev_return = 0.0
            prev_value = 0.0
            prev_advantage = 0.0
            for transition_idx in reversed(range(max_episode_len)):
                returns[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_return * (
                            1 - terminated[:, transition_idx]) * mask[:, transition_idx]
                deltas[:, transition_idx] = r[:, transition_idx] + self.args.gamma * prev_value * (
                            1 - terminated[:, transition_idx]) * mask[:, transition_idx] \
                                            - values[:, transition_idx]
                advantages[:, transition_idx] = deltas[:,
                                                transition_idx] + self.args.gamma * self.args.lamda * prev_advantage * (
                                                            1 - terminated[:, transition_idx]) * mask[:, transition_idx]

                prev_return = returns[:, transition_idx]
                prev_value = values[:, transition_idx]
                prev_advantage = advantages[:, transition_idx]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()

            if self.args.cuda:
                advantages = advantages.cuda()

            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0

            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages

            entropy = dist.entropy()
            entropy[mask == 0] = 0.0

            policy_loss = torch.min(surr1, surr2) + self.args.entropy_coeff * entropy

            policy_loss = - (policy_loss * mask).sum() / mask.sum()

            error_clip = torch.clamp(values - old_values.detach(), -self.args.clip_param,
                                     self.args.clip_param) + old_values.detach() - returns
            error_original = values - returns

            value_loss = 0.5 * torch.max(error_original ** 2, error_clip ** 2)
            value_loss = (mask * value_loss).sum() / mask.sum()

            loss = policy_loss + value_loss

            self.ac_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, self.args.grad_norm_clip)
            self.ac_optimizer.step()

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
        #     self.target_critic.load_state_dict(self.eval_critic.state_dict())

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx], \
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        # u_onehot = batch['u_onehot'][:, transition_idx]
        # if transition_idx != max_episode_len - 1:
        #     u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        # else:
        #     u_onehot_next = torch.zeros(*u_onehot.shape)
        # s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        # s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]
        # u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        # u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        #
        # if transition_idx == 0:
        #     u_onehot_last = torch.zeros_like(u_onehot)
        # else:
        #     u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
        #     u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []

        # inputs.append(s)
        # inputs_next.append(s_next)

        inputs.append(obs)
        inputs_next.append(obs_next)

        # inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.cuda:
                inputs = inputs.cuda()
                # inputs_next = inputs_next.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()
                # self.target_critic_hidden = self.target_critic_hidden.cuda()
            if len(inputs.shape) == 2:
                inputs = inputs.view(-1, self.n_agents, self.obs_shape)
                inputs_next = inputs_next.view(-1, self.n_agents, self.obs_shape)
            v_eval, self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden)
            # v_target, self.target_critic_hidden = self.eval_critic(inputs_next, self.target_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            # v_target = v_target.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)
            # v_targets.append(v_target)

        v_evals = torch.stack(v_evals, dim=1)
        # v_targets = torch.stack(v_targets, dim=1)
        return v_evals, v_targets

    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        # if self.args.last_action:
        #     if transition_idx == 0:
        #         inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        #     else:
        #         inputs.append(u_onehot[:, transition_idx - 1])
        # if self.args.reuse_network:
        #     inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                self.policy_rnn.policy_hidden = self.policy_rnn.policy_hidden.cuda()
            if len(inputs.shape) == 2:
                inputs = inputs.view(-1, self.n_agents, self.obs_shape)

            outputs = self.policy_rnn(inputs, self.policy_rnn.policy_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = action_prob + 1e-10

        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0

        action_prob = action_prob + 1e-10

        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.policy_rnn.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        # self.target_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.policy_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_params.pkl')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        # self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.init_hidden(1)
        # done = False
        obs = self.env.reset()

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            # time.sleep(0.2)
            # check_obs = self.env.observation()
            # cc, aa = tranf_obs(check_obs, self.feature_encoder)
            observation = self.env.observation()[1:self.args.n_agents+1]
            obs, ava_actions = tranf_obs(observation, self.feature_encoder)
            # self.env.render()
            # obs = obs.squeeze()
            # obs = self.feature_encoder.encode(observation)
            # obs = football_observation_wrapper(observation)
            # obs = self.env.observation(observation)
            # aa = [-i-1 for i in range(self.n_agents*2)]
            state = obs.flatten()
            actions, actions_onehot = [], []
            # for agent_id in range(self.n_agents):
                # avail_action = self.env.get_avail_agent_actions(agent_id)
            action = self.choose_action(obs, ava_actions)
            # generate onehot vector of th action
            for agent_id in range(self.n_agents):
                action_onehot = np.zeros(self.args.n_actions)
                # try:
                # actionss = action[agent_id]
                action_onehot[action[agent_id]] = 1
                # except:
                #     print('debug here')
                actions.append(np.int(action[agent_id]))
                try:
                    actions_onehot.append(action_onehot)
                except:
                    print('debug here')
                # avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            all_actions = deepcopy(actions)
            all_actions.insert(0, 0)
            next_obs, reward, terminated, info = self.env.step(all_actions)
            # if done:
            #     print('debug here')
            next_obs = next_obs[1:self.args.n_agents+1]
            rewards = calc_reward(reward, observation[0], next_obs[0], step)

            rewards = sum(rewards)
            # next_obs = football_observation_wrapper(next_obs)
            # reward = football_reward_wrapper(next_obs, reward)
            if info['score_reward'] > 0:
                win_tag = True
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(ava_actions)
            step += 1
            r.append([rewards])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += rewards

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            # not win_tag and step < self.args.episode_limit:
        # last obs
        obs = self.env.observation()[1:self.args.n_agents+1]
        obs, ava_actions = tranf_obs(obs, self.feature_encoder)
        obs = obs.squeeze()
        state = obs.flatten()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        # avail_actions = []
        # for agent_id in range(self.n_agents):
        #     avail_action = self.env.get_avail_agent_actions(agent_id)
        #     avail_actions.append(avail_action)
        # observation = self.env.observation()[1:self.args.n_agents + 1]
        # obs, ava_actions = tranf_obs(observation, self.feature_encoder)
        avail_u.append(ava_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step

    def choose_action(self, obs, avail_actions, evaluate=False):
        inputs = obs.copy()
        # avail_actions_ind = np.nonzero(avail_actions)[0]
        # agent_id = np.zeros(self.n_agents)
        # agent_id[agent_num] = 1.
        # if self.args.last_action:
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))
        with torch.no_grad():
            policy_hidden_state = self.policy_rnn.policy_hidden

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32)
        if self.args.cuda:
            inputs = inputs.cuda()
            policy_hidden_state = policy_hidden_state.cuda()

        policy_q_value = self.policy_rnn(inputs, policy_hidden_state)
        action_prob = torch.nn.functional.softmax(policy_q_value.cpu(), dim=-1)
        action_prob[avail_actions == 0.0] = 0.0
        if evaluate:
            action = torch.argmax(action_prob)
        else:
            action = Categorical(action_prob).sample().long()

        return action

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.save_model(train_step)

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

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch