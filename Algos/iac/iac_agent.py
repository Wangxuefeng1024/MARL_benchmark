import torch
import os
from Algos.iac.network import CNN_Actor, Actor, CNN_Critic, Critic, Cen_Critic, ICMNetwork
from tensorboardX import SummaryWriter
import numpy as np
from torch.distributions import Categorical
from Utils.sc_memory import img_ReplayBuffer, ReplayBuffer
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
from .intrinsic_reward import IntrinsicReward

class ICM(IntrinsicReward):
    """
    Intrinsic curiosity module (ICM) class

    Paper:
    Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).
    Curiosity-driven exploration by self-supervised prediction.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 16-17).

    Link: http://openaccess.thecvf.com/content_cvpr_2017_workshops/w5/html/Pathak_Curiosity-Driven_Exploration_by_CVPR_2017_paper.html
    """

    def __init__(
        self,
        state_size,
        action_size,
        args,
        hidden_dim=128,
        state_rep_size=64,
        learning_rate=1e-5,
        eta=2,
        discrete_actions=False,
    ):
        """
        Initialise parameters for MARL training
        :param state_size: dimension of state input
        :param action_size: dimension of action input
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of state representation in network
        :param learning_rate: learning rate for ICM parameter optimisation
        :param eta: curiosity loss weighting factor
        :param discrete_actions: flag if discrete actions are used (one-hot encoded)
        """
        super(ICM, self).__init__(state_size, action_size, eta)
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size
        self.learning_rate = learning_rate
        self.discrete_actions = discrete_actions
        self.args = args
        self.model_dev = "cpu"

        self.model = ICMNetwork(
            state_size, action_size, args, hidden_dim, state_rep_size, discrete_actions
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.forward_loss = None
        self.inverse_loss = None

    def _prediction(self, state, action, next_state, use_cuda):
        """
        Compute prediction
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: use CUDA tensors
        :return: (batch of) forward loss, inverse_loss
        """
        if use_cuda:
            fn = lambda x: x.cuda()
            device = "gpu"
        else:
            fn = lambda x: x.cpu()
            device = "cpu"
        if not self.model_dev == device:
            self.model = fn(self.model)
            self.model_dev = device

        predicted_action, predicted_next_state_rep, next_state_rep = self.model(
            state, next_state, action
        )
        if self.discrete_actions:
            # discrete one-hot encoded action
            action_targets = []
            for z in range(self.args.n_agents):
                targeted_action = np.zeros(self.args.n_actions)
                targeted_action[action[z]] = 1
                action_targets.append(targeted_action)
            action_targets = torch.tensor(action_targets).cuda().float()
            # action_targets = [np.zeros(self.args.n_actions)[action[z]] for z in range(self.args.n_agents)]
            inverse_loss = F.cross_entropy(predicted_action, action_targets, reduction="none")
        else:
            inverse_loss = ((predicted_action - action) ** 2).sum(-1)
        forward_loss = 0.5 * ((next_state_rep - predicted_next_state_rep) ** 2).sum(-1)
        return forward_loss.mean(-1), inverse_loss.mean(-1)

    def compute_intrinsic_reward(self, state, action, next_state, use_cuda, train=False):
        """
        Compute intrinsic reward for given input
        :param state: (batch of) current state(s)
        :param action: (batch of) applied action(s)
        :param next_state: (batch of) next/reached state(s)
        :param use_cuda: use CUDA tensors
        :param train: flag if model should be trained
        :return: (batch of) intrinsic reward(s)
        """
        forward_loss, inverse_loss = self._prediction(state, action, next_state, use_cuda)

        self.forward_loss = forward_loss
        self.inverse_loss = inverse_loss

        if train:
            self.optimizer.zero_grad()
            loss = forward_loss + inverse_loss
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        return self.eta * forward_loss

    def get_losses(self):
        """
        Get losses of last computation if existing
        :return: list of (batch of) loss(es)
        """
        if self.forward_loss is not None:
            return [self.forward_loss, self.inverse_loss]
        else:
            return []



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.intrinsic_rewards = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.intrinsic_rewards[:]

class IAC:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # self.state_shape = args.state_shape
        self.obs_shape = args.n_states
        self.env = env
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = ReplayBuffer(args)
        # 神经网络
        self.eval_rnn = Actor(input_shape, args)
        self.target_rnn = Actor(input_shape, args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.env
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.eval_parameters = list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo IQL')

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
        u, r, avail_u, avail_u_next, terminated = batch['u'], batch['r'].repeat(1, 1, self.n_agents),  batch['avail_u'], \
                                                  batch['avail_u_next'], batch['terminated'].repeat(1, 1, self.n_agents)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            u = u.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        targets = r + self.args.gamma * q_targets * (1 - terminated)

        td_error = (q_evals - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('loss is ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        return loss

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
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

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def choose_action(self, obs, agent_num):
        inputs = obs.copy()
        avail_actions_ind = np.ones(self.n_actions)  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        # if self.args.last_action:
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))
        # hidden_state = self.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            # hidden_state = hidden_state.cuda()

        # get q value

        q_value = self.eval_rnn(inputs.squeeze())

        # choose action from q value

        # q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action

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

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.save_model(train_step)

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
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

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
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

class im_IAC:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # self.state_shape = args.state_shape
        # self.flatten_state_shape = args.state_shape[0]*args.state_shape[1]*args.state_shape[2]
        self.obs_shape = args.n_states
        self.K_epochs = 8
        # self.flatten_obs_shape = args.obs_shape[0]*args.obs_shape[1]*args.obs_shape[2]
        self.env = env
        self.gamma = 0.99
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        # if args.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = RolloutBuffer()
        # 神经网络
        self.actor = CNN_Actor(input_shape, args)
        self.critic = CNN_Critic(input_shape, args)
        self.args = args
        self.writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + str(args.n_agents))
        if self.args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            # self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.algo
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        # self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.epsilon = args.epsilon
        self.eps_clip = 0.2
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.MseLoss = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        # # if args.optimizer == "RMS":
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': self.args.lr_actor},
        #     {'params': self.critic.parameters(), 'lr': self.args.lr_critic}
        # ])

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo IQL')

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def choose_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.from_numpy(obs).to(device)
        if self.args.continuous:
            action_mean = self.actor(obs)
            cov_mat = self.cov_mat.to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(obs)
            dist = Categorical(action_probs)

        action = dist.sample()
        action = action.detach().cpu().numpy()
        action = np.clip(action, -1., 1.)
        action_logprob = dist.log_prob(torch.from_numpy(action).to(device))

        return action, action_logprob.detach().cpu().numpy()

    def evaluate(self, state, action):
        state_values = self.critic(state.view(-1,3,84,84))
        if self.args.continuous:
            action_mean = self.actor(state)

            action_var = self.cov_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.n_actions == 1:
                action = action.reshape(-1, self.n_actions)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()


        return action_logprobs, state_values, dist_entropy

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.actor.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        # buffer_rewards = np.sum(np.array(self.buffer.rewards).reshape(-1, self.n_agents), axis=-1)
        # buffer_terminals = np.array(self.buffer.is_terminals).reshape(-1, self.n_agents)
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
        old_states = torch.squeeze(torch.from_numpy(np.array(self.buffer.states))).detach().to(device)
        old_actions = torch.squeeze(torch.from_numpy(np.array(self.buffer.actions))).detach().to(device)
        old_logprobs = torch.squeeze(torch.from_numpy(np.array(self.buffer.logprobs))).detach().to(device)

        a_loss = []
        c_loss = []
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

            # final loss of clipped objective PPO
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            a_loss.append(actor_loss.item())
            c_loss.append(critic_loss.item())
        # clear buffer
        self.buffer.clear()
        return np.mean(c_loss), np.mean(a_loss)


class PPO_IAC:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # self.state_shape = args.state_shape
        # self.flatten_state_shape = args.state_shape[0]*args.state_shape[1]*args.state_shape[2]
        self.obs_shape = args.n_states
        self.K_epochs = 8
        # self.flatten_obs_shape = args.obs_shape[0]*args.obs_shape[1]*args.obs_shape[2]
        self.env = env
        self.gamma = 0.99
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        # if args.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = RolloutBuffer()
        # 神经网络
        self.actor = Actor(input_shape, args)
        self.critic = Critic(input_shape, args)
        self.args = args
        self.writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + str(args.n_agents))
        if self.args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            # self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.algo
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        # self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.epsilon = args.epsilon
        self.eps_clip = 0.2
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.MseLoss = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        # # if args.optimizer == "RMS":
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': self.args.lr_actor},
        #     {'params': self.critic.parameters(), 'lr': self.args.lr_critic}
        # ])

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo IAC')

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def choose_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.from_numpy(obs).to(device)
        if self.args.continuous:
            action_mean = self.actor(obs.float())
            cov_mat = self.cov_mat.to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1., 1.)
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        else:
            action_probs = self.actor(obs.float())
            dist = Categorical(action_probs)
            action = dist.sample().view(1)
            action = action.detach().cpu().numpy()
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        return action, action_logprob.detach().cpu().numpy()

    def evaluate(self, state, action):
        state_values = self.critic(state.float())
        if self.args.continuous:
            action_mean = self.actor(state.float())

            # action_var = self.cov_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, self.cov_mat.to(device))

            # For Single Action Environments.
            if self.n_actions == 1:
                action = action.reshape(-1, self.n_actions)
        else:
            action_probs = self.actor(state.float())
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.actor.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        # buffer_rewards = np.sum(np.array(self.buffer.rewards).reshape(-1, self.n_agents), axis=-1)
        # buffer_terminals = np.array(self.buffer.is_terminals).reshape(-1, self.n_agents)
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0).expand(self.args.n_agents, rewards.shape[0])
            rewards = rewards.reshape(-1, self.args.n_agents)
        # convert list to tensor
        old_states = torch.squeeze(torch.from_numpy(np.array(self.buffer.states))).detach().to(device)
        old_actions = torch.squeeze(torch.from_numpy(np.array(self.buffer.actions))).detach().to(device)
        old_logprobs = torch.squeeze(torch.from_numpy(np.array(self.buffer.logprobs))).detach().to(device)

        a_loss = []
        c_loss = []
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

            # final loss of clipped objective PPO
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            a_loss.append(actor_loss.item())
            c_loss.append(critic_loss.item())
        # clear buffer
        self.buffer.clear()
        return np.mean(c_loss), np.mean(a_loss)


class Central_PPO:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # self.state_shape = args.state_shape
        # self.flatten_state_shape = args.state_shape[0]*args.state_shape[1]*args.state_shape[2]
        self.obs_shape = args.n_states
        self.K_epochs = 8
        # self.flatten_obs_shape = args.obs_shape[0]*args.obs_shape[1]*args.obs_shape[2]
        self.env = env
        self.gamma = 0.99
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        # if args.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = RolloutBuffer()
        # 神经网络
        self.actor = Actor(input_shape, args)
        self.critic = Cen_Critic(input_shape, args)
        self.args = args
        self.writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + str(args.n_agents))
        if self.args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            # self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.algo
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        # self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.epsilon = args.epsilon
        self.eps_clip = 0.2
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.MseLoss = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        # # if args.optimizer == "RMS":
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': self.args.lr_actor},
        #     {'params': self.critic.parameters(), 'lr': self.args.lr_critic}
        # ])

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo IAC')

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def choose_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.from_numpy(obs).to(device)
        if self.args.continuous:
            action_mean = self.actor(obs.float())
            cov_mat = self.cov_mat.to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1., 1.)
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        else:
            action_probs = self.actor(obs.float())
            dist = Categorical(action_probs)
            action = dist.sample().view(1)
            action = action.detach().cpu().numpy()
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        return action, action_logprob.detach().cpu().numpy()

    def evaluate(self, state, action):
        state_values = self.critic(state.float())
        if self.args.continuous:
            action_mean = self.actor(state.float())

            # action_var = self.cov_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, self.cov_mat.to(device))

            # For Single Action Environments.
            if self.n_actions == 1:
                action = action.reshape(-1, self.n_actions)
        else:
            action_probs = self.actor(state.float())
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.actor.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        # buffer_rewards = np.sum(np.array(self.buffer.rewards).reshape(-1, self.n_agents), axis=-1)
        # buffer_terminals = np.array(self.buffer.is_terminals).reshape(-1, self.n_agents)
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0).expand(self.args.n_agents, rewards.shape[0])
            rewards = rewards.reshape(-1, self.args.n_agents)
        # convert list to tensor
        old_states = torch.squeeze(torch.from_numpy(np.array(self.buffer.states))).detach().to(device)
        old_actions = torch.squeeze(torch.from_numpy(np.array(self.buffer.actions))).detach().to(device)
        old_logprobs = torch.squeeze(torch.from_numpy(np.array(self.buffer.logprobs))).detach().to(device)

        a_loss = []
        c_loss = []
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

            # final loss of clipped objective PPO
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            a_loss.append(actor_loss.item())
            c_loss.append(critic_loss.item())
        # clear buffer
        self.buffer.clear()
        return np.mean(c_loss), np.mean(a_loss)

class ICM_PPO:
    def __init__(self, env, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # self.state_shape = args.state_shape
        # self.flatten_state_shape = args.state_shape[0]*args.state_shape[1]*args.state_shape[2]
        self.obs_shape = args.n_states
        self.K_epochs = 8
        # self.flatten_obs_shape = args.obs_shape[0]*args.obs_shape[1]*args.obs_shape[2]
        self.env = env
        self.gamma = 0.99
        self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        # if args.last_action:
        #     input_shape += self.n_actions
        # if args.reuse_network:
        #     input_shape += self.n_agents
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = RolloutBuffer()
        # 神经网络
        self.actor = Actor(input_shape, args)
        self.critic = Cen_Critic(input_shape, args)
        self.args = args
        self.ICM = ICM(
                    args.n_states,
                    args.n_actions,
                    args,
                    128,
                    128,
                    1e-4,
                    2,
                    True,
                )

        self.writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + str(args.env))
        if self.args.cuda:
            self.actor.cuda()
            self.critic.cuda()
            # self.target_rnn.cuda()
        self.model_dir = args.model_dir + '/' + args.algo
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_rnn, map_location=map_location))
                print('Successfully load the model: {}'.format(path_rnn))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        # self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.epsilon = args.epsilon
        self.eps_clip = 0.2
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        self.MseLoss = torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)
        # # if args.optimizer == "RMS":
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': self.args.lr_actor},
        #     {'params': self.critic.parameters(), 'lr': self.args.lr_critic}
        # ])

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo IAC')

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def choose_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        obs = torch.from_numpy(obs).to(device)
        if self.args.continuous:
            action_mean = self.actor(obs.float())
            cov_mat = self.cov_mat.to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action = action.detach().cpu().numpy()
            action = np.clip(action, -1., 1.)
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        else:
            action_probs = self.actor(obs.float())
            dist = Categorical(action_probs)
            action = dist.sample().view(1)
            action = action.detach().cpu().numpy()
            action_logprob = dist.log_prob(torch.from_numpy(action).to(device))
        return action, action_logprob.detach().cpu().numpy()

    def evaluate(self, state, action):
        state_values = self.critic(state.float())
        if self.args.continuous:
            action_mean = self.actor(state.float())

            # action_var = self.cov_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, self.cov_mat.to(device))

            # For Single Action Environments.
            if self.n_actions == 1:
                action = action.reshape(-1, self.n_actions)
        else:
            action_probs = self.actor(state.float())
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.actor.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

    # def compute_intrinsic_rewards(self, states, actions, next_states):


    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        # buffer_rewards = np.sum(np.array(self.buffer.rewards).reshape(-1, self.n_agents), axis=-1)
        # buffer_terminals = np.array(self.buffer.is_terminals).reshape(-1, self.n_agents)
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(0).expand(self.args.n_agents, rewards.shape[0])
            rewards = rewards.reshape(-1, self.args.n_agents)
        # convert list to tensor
        old_states = torch.squeeze(torch.from_numpy(np.array(self.buffer.states))).detach().to(device)
        old_actions = torch.squeeze(torch.from_numpy(np.array(self.buffer.actions))).detach().to(device)
        old_logprobs = torch.squeeze(torch.from_numpy(np.array(self.buffer.logprobs))).detach().to(device)

        a_loss = []
        c_loss = []
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

            # final loss of clipped objective PPO
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            a_loss.append(actor_loss.item())
            c_loss.append(critic_loss.item())
        # clear buffer
        self.buffer.clear()
        return np.mean(c_loss), np.mean(a_loss)


