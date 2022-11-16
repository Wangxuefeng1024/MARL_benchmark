import torch
import os
from copy import deepcopy
from Algos.qmix.network import DRQN, QMixNet
from utils.sc_memory import ReplayBuffer
import numpy as np
from torch.distributions import Categorical
from utils.encoder_basic import FeatureEncoder
import glob
import random

class QMIX:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.n_states
        self.obs_shape = args.n_obs
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        input_shape = self.obs_shape
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = ReplayBuffer(args)
        self.save_path = self.args.result_dir + '/' + args.algo + '/' + args.map
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        # 神经网络
        self.eval_rnn = DRQN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = DRQN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_actor = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_actor, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_actor, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo QMIX')

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

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
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
        num = str(train_step)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
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

    def update(self):
        pass

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
                                                   avail_action, epsilon)
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

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value

        q_value, self.eval_hidden[:, agent_num, :] = self.eval_rnn(inputs, hidden_state)

        # choose action from q value

        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        loss = self.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.save_model(train_step)
        return loss

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

class gfoot_qmix:
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
        self.save_path = self.args.result_dir + '/' + args.algo + '/' + args.map
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        # 神经网络
        self.eval_rnn = DRQN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = DRQN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.scenario
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_actor = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_actor, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_actor, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo QMIX')

    def learn(self, batch, max_episode_len, train_step):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
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

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        return loss.item()

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
        num = str(train_step)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
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
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.init_hidden(1)
        # done = False
        obs, state, ava = self.env.reset()
        o.append(obs[1:])
        s.append(np.array(state)[1:].flatten())

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.args.episode_limit:
            actions, actions_onehot = [], []
            for agent_id in range(self.n_agents):
                action = self.choose_action(obs[agent_id+1], ava[agent_id+1], last_action[agent_id], agent_id, epsilon)
                # generate onehot vector of each action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot
            all_actions = deepcopy(actions)
            all_actions.insert(0, 0)
            obs, state, rewards, dones, infos, ava = self.env.step(all_actions)
            terminated = True if True in dones else False
            for value in infos:
                if value['score_reward'] > 0:
                    win_tag = True
            o.append(obs[1:])
            s.append(np.array(state)[1:].flatten())
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(ava[1:])
            step += 1
            r.append([np.sum(rewards)])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += np.sum(rewards)

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # split obs, obs_next, state, next_state
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        if not terminated:
            actions, actions_onehot = [], []
            for agent_id in range(self.n_agents):
                action = self.choose_action(obs[agent_id+1], ava[agent_id+1], last_action[agent_id], agent_id, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot
            all_actions = deepcopy(actions)
            all_actions.insert(0, 0)
            obs, state, rewards, dones, infos, ava = self.env.step(all_actions)
            ava = ava[1:]
        else:
            ava = np.zeros([self.args.n_agents, self.args.n_actions])
            # avail_actions.append(ava)
        avail_u.append(ava)
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
        # episode['r'] = episode['r'].reshape((1, self.args.episode_limit, 1))
        if not evaluate:
            self.epsilon = epsilon

        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step

    def choose_action(self, obs, ava_actions,last_action, agent_num, epsilon):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(ava_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        q_value, self.eval_hidden[:, agent_num, :] = self.eval_rnn(inputs, hidden_state)

        # choose action from q value
        q_value = q_value.squeeze()
        q_value[ava_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        loss = self.learn(batch, max_episode_len, train_step)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.save_model(train_step)
        return loss

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


class qmix_matrix:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        assert self.args.base_net in ['mlp', 'rnn']
        if self.args.base_net == 'mlp':
            from utils.matrix_memory import ReplayMemoryForMLP
            self.replay_memory = ReplayMemoryForMLP(self.args)
        else:
            from utils.matrix_memory import ReplayMemoryForRNN
            self.replay_memory = ReplayMemoryForRNN(self.args)

        self.training_steps = self.args.training_steps
        self.playing_steps = self.args.playing_steps

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.n_states
        self.obs_shape = args.n_obs
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.episode_limit = args.episode_limit
        input_shape = self.obs_shape
        self.win_rates = []
        self.episode_rewards = []
        self.buffer = ReplayBuffer(args)
        self.save_path = self.args.result_dir + '/' + args.algo + '/' + args.map
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        # 神经网络
        self.eval_rnn = DRQN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = DRQN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.algo + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_actor = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_actor, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_actor, path_qmix))
            else:
                raise Exception("No model!")

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def choose_action(self, observation, h_in=None, state=None):
        actions = []
        q_evals = []
        h_outs = []
        for a in range(self.args.n_agents):
            obs = observation[self.agents[a].agent_id]
            obs = torch.from_numpy(obs).float()
            obs = obs.unsqueeze(0)
            if self.args.base_net == 'rnn':
                _h_in = h_in[self.agents[a].agent_id]
                _h_in = _h_in.unsqueeze(0)
                q_eval, h_out = self.agents[a].get_q_value(obs, _h_in)
                h_outs.append(h_out)
            else:
                q_eval = self.agents[a].get_q_value(obs)
            action = self.choose_action_with_epsilon_greedy(q_eval)
            action = torch.tensor([action])
            action = action.unsqueeze(0)
            actions.append(action)

            q_eval = q_eval.gather(1, action)
            q_evals.append(q_eval)

        if self.args.play:
            q_evals = torch.stack(q_evals, dim=1)
            state = torch.tensor(state, dtype=torch.float)
            if self.args.algorithm == 'vdn':
                q_total_eval = self.trainer.get_q_value(q_evals)
            elif self.args.algorithm == 'qmix':
                q_total_eval = self.trainer.get_q_value(q_evals, state)
            else:
                q_total_eval = None
            if self.args.base_net == 'rnn':
                h_outs = torch.stack(h_outs)
                return actions, h_outs, q_total_eval.item()
            else:
                return actions, q_total_eval.item()
        else:
            if self.args.base_net == 'rnn':
                h_outs = torch.stack(h_outs)
                return actions, h_outs
            else:
                return actions

    def choose_action_with_epsilon_greedy(self, q_val):
        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0, self.args.n_actions - 1)
        else:
            return q_val.argmax().item()

    def train(self, batch, step):
        q_evals = []
        max_q_prime_evals = []
        for a in range(self.args.n_agents):
            obs = batch['observation'][:, a]
            obs_prime = batch['next_observation'][:, a]
            action = batch['action'].squeeze(1).unsqueeze(2)[:, a]

            if self.args.base_net == 'rnn':
                _h_in = batch['hidden_in'][:, a]
                q_eval, _ = self.agents[a].get_q_value(obs, _h_in)
            else:
                q_eval = self.agents[a].get_q_value(obs)
            q_eval = q_eval.gather(1, action)
            q_evals.append(q_eval)

            if self.args.base_net == 'rnn':
                _h_out = batch['hidden_out'][:, a]
                max_q_prime_eval, _ = self.agents[a].get_target_q_value(obs_prime, _h_out)
                max_q_prime_eval = max_q_prime_eval.max(1)[0].unsqueeze(1)
            else:
                max_q_prime_eval = self.agents[a].get_target_q_value(obs_prime).max(1)[0].unsqueeze(1)
            max_q_prime_evals.append(max_q_prime_eval)

        q_evals = torch.stack(q_evals, dim=1)
        max_q_prime_evals = torch.stack(max_q_prime_evals, dim=1)

        state = batch['state']
        next_state = batch['next_state']
        reward = batch['reward']
        done_mask = batch['done_mask']

        if self.args.algorithm == 'vdn':
            loss = self.train_agents(q_evals, max_q_prime_evals, reward, done_mask)
        elif self.args.algorithm == 'qmix':
            loss = self.train_agents(q_evals, max_q_prime_evals, state, next_state, reward, done_mask)
        else:
            loss = 0.0

        if step > 0 and step % self.args.target_network_update_interval == 0:
            for a in range(self.args.n_agents):
                self.agents[a].update_net()
            self.trainer.update_net()

        return loss

    def save_model(self):
        model_params = {}
        for a in range(self.args.n_agents):
            model_params['agent_{}'.format(str(a))] = self.agents[a].get_net_params()
        model_params['mixer'] = self.trainer.get_net_params()

        model_save_filename = os.path.join(
            model_save_path, "{0}_{1}.pth".format(
                self.args.algorithm, self.args.base_net
            )
        )
        torch.save(model_params, model_save_filename)

    def load_model(self):
        saved_model = glob.glob(os.path.join(
            model_save_path, "{0}_{1}.pth".format(
                self.args.algorithm, self.args.base_net
            )
        ))
        model_params = torch.load(saved_model[0])

        for a in range(self.args.n_agents):
            self.agents[a].update_net(model_params['agent_{}'.format(str(a))])
        self.trainer.update_net(model_params['mixer'])


    def run(self):
        step = 0
        while step < self.training_steps:
            state, observations = self.env.reset()
            done = False
            if self.args.base_net == 'rnn':
                h_out = self.init_hidden(1)
            while not done:
                h_in = h_out
                actions, h_out = self.agents.choose_action(observations, h_in)

                next_state, next_observations, reward, done = self.env.step(actions)
                print('step: {0}, state: {1}, actions: {2}, reward: {3}'.format(step, state, actions, reward))
                done_mask = 0.0 if done else 1.0


                self.replay_memory.put(
                    [state, observations, actions, reward, next_state, next_observations, h_in, h_out, done_mask]
                )


                if self.replay_memory.size() >= self.args.batch_size:
                    batch = {}
                    if self.args.base_net == 'rnn':
                        s, o, a, r, s_prime, o_prime, hidden_in, hidden_out, done_mask = self.replay_memory.sample(
                            self.args.batch_size
                        )
                        batch['hidden_in'] = hidden_in
                        batch['hidden_out'] = hidden_out
                    else:
                        s, o, a, r, s_prime, o_prime, done_mask = self.replay_memory.sample(self.args.batch_size)
                    batch['state'] = s
                    batch['observation'] = o
                    batch['action'] = a
                    batch['reward'] = r
                    batch['next_state'] = s_prime
                    batch['next_observation'] = o_prime
                    batch['done_mask'] = done_mask

                    loss = self.train(batch, step)

                    if step % self.args.print_interval == 0:
                        print("step: {0}, loss: {1}".format(step, loss))

                state = next_state
                observations = next_observations
                step += 1

                if done:
                    break
        self.agents.save_model()

    def play(self):
        self.agents.load_model()

        q_value_list, iteration, selected_q_value_list, q_value_list_0, q_value_list_1, q_value_list_2, \
        iteration_0, iteration_1, iteration_2 = None, None, None, None, None, None, None, None, None

        if self.args.env_name == 'one_step_payoff_matrix':
            q_value_list = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        elif self.args.env_name == 'two_step_payoff_matrix':
            q_value_list_0 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_0 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            q_value_list_1 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_1 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            q_value_list_2 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
            iteration_2 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        else:
            raise Exception("Wrong env name.")

        step = 0
        while step < self.playing_steps:
            state, observations = self.env.reset()
            done = False

            if self.args.base_net == 'rnn':
                h_out = init_hidden(self.args)
            state_num = 0
            while not done:
                if self.args.base_net == 'rnn':
                    h_in = h_out
                    actions, h_out, q_total_evals = self.agents.choose_action(observations, h_in=h_in, state=state)
                else:
                    actions, q_total_evals = self.agents.choose_action(observations, state=state)
                next_state, next_observations, reward, done = self.env.step(actions)

                state = next_state
                observations = next_observations

                if self.args.env_name == 'one_step_payoff_matrix':
                    q_value_list[actions[0]][actions[1]] += q_total_evals
                    iteration[actions[0]][actions[1]] += 1
                elif self.args.env_name == 'two_step_payoff_matrix':
                    if state_num == 0:
                        if actions[0] == 0:
                            state_num = 1
                        if actions[0] == 1:
                            state_num = 2
                        q_value_list_0[actions[0]][actions[1]] += q_total_evals
                        iteration_0[actions[0]][actions[1]] += 1
                    else:
                        if state_num == 1:
                            q_value_list_1[actions[0]][actions[1]] += q_total_evals
                            iteration_1[actions[0]][actions[1]] += 1
                        elif state_num == 2:
                            q_value_list_2[actions[0]][actions[1]] += q_total_evals
                            iteration_2[actions[0]][actions[1]] += 1

                step += 1

                if done:
                    break

        if self.args.env_name == 'one_step_payoff_matrix':
            for i in range(self.args.n_actions):
                for j in range(self.args.n_actions):
                    q_value_list[i][j] /= iteration[i][j]
            print(q_value_list)
        elif self.args.env_name == 'two_step_payoff_matrix':
            for i in range(self.args.n_actions):
                for j in range(self.args.n_actions):
                    q_value_list_0[i][j] /= iteration_0[i][j]
                    q_value_list_1[i][j] /= iteration_1[i][j]
                    q_value_list_2[i][j] /= iteration_2[i][j]
            print(q_value_list_0)
            print(q_value_list_1)
            print(q_value_list_2)