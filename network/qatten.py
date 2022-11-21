import torch.nn as nn
import torch
import torch.nn.functional as F

class DRQN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(DRQN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QattenNet(nn.Module):
    def __init__(self, args):
        super(QattenNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.hyper_hidden_dim, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.hyper_hidden_dim, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.n_states, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.n_states, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )
        self.state_encoder = nn.Sequential(nn.Linear(args.n_states, args.hyper_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hyper_hidden_dim, args.hyper_hidden_dim)
                                     )

        self.key = nn.Linear(args.n_obs, args.hyper_hidden_dim).cuda()
        self.query = nn.Linear(args.n_states, args.hyper_hidden_dim).cuda()
        self.state_projection = nn.Sequential(nn.Linear(args.n_states, args.hyper_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.hyper_hidden_dim, 1)
                                     ).cuda()

    def forward(self, q_values, states, observations):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)

        states = states.reshape(-1, 1, self.args.n_states)  # (episode_num * max_episode_len, state_shape)
        all_states = states.expand(-1, self.args.n_agents, self.args.n_states)# (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        # q_values = q_values.view(episode_num, -1, 1)

        k = F.relu(self.key(observations)).view(-1, self.args.n_agents, self.args.hyper_hidden_dim)
        q = F.relu(self.query(all_states)).view(-1, self.args.n_agents, self.args.hyper_hidden_dim).permute(0,2,1)
        v = q_values.view(-1, self.args.n_agents, 1)

        weights = torch.bmm(k, q)/((self.args.hyper_hidden_dim)**0.5)
        # attention = torch.where(adj > 0, weights, zero_vec)
        attention = F.softmax(weights, dim=-1)
        # get the weighted sum value
        Weighted_Q = torch.bmm(attention, v)

        new_states = self.state_encoder(states.view(-1, self.args.n_states))


        w1 = torch.abs(self.hyper_w1(new_states))  # (1920, 160)
        b1 = self.hyper_b1(new_states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(torch.bmm(Weighted_Q.permute(0,2,1), w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(new_states))  # (1920, 32)
        b2 = self.hyper_b2(new_states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        q_total = q_total + self.state_projection(states.view(episode_num, -1, self.args.n_states))
        return q_total

