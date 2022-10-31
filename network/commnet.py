import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, n_agents, args):
        super(Actor, self).__init__()
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_agents = n_agents
        self.args = args
        self.n_actions = a_dim
        self.input_shape = s_dim
        self.input_size = s_dim * self.n_agents
        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)#nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding0 = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def forward(self, obs):
        size = obs.view(-1, self.n_agents, self.input_shape).shape
        size0 = size[0]
        obs_encoding = torch.relu(self.encoding(obs.view(size0*self.n_agents, self.input_shape)))#.contiguous()  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        h_out = self.f_obs(obs_encoding)
        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)

                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = (1 - torch.eye(self.n_agents))
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c,h)

        weights = torch.relu(self.decoding0(h))
        if self.args.scenario == 'simple_spread':
            weights = torch.tanh(self.decoding(weights))
        else:
            weights = torch.softmax(self.decoding(weights), dim=-1)
        # print(weights)
        return weights


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, n_agents, args):
        super(Critic, self).__init__()
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_agents = n_agents
        self.args = args
        self.n_actions = a_dim
        self.input_shape = s_dim + a_dim
        self.input_size = s_dim * self.n_agents
        self.encoding = nn.Linear(self.input_shape , self.rnn_hidden_dim)
        self.f_obs = nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.f_comm = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)#nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.decoding = nn.Linear(self.rnn_hidden_dim, 1)


    def forward(self, obs, act):
        obs = obs.view(-1, self.n_agents, self.input_shape-self.n_actions)
        size0 = obs.shape[0]
        act = act.view(-1, self.n_agents, self.n_actions)
        x = torch.cat((obs, act), dim=-1)
        obs_encoding = torch.relu(self.encoding(x.view(size0*self.n_agents, self.input_shape)))#.contiguous()  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

        h_out = self.f_obs(obs_encoding)

        for k in range(2):
            if k == 0:
                h = h_out
                c = torch.zeros_like(h)
            else:
                h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim)

                c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim)
                c = c.repeat(1, self.n_agents, 1)
                mask = (1 - torch.eye(self.n_agents))
                mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim).view(self.n_agents, -1)
                if self.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim)
                c = c.mean(dim=-2)
                h = h.reshape(-1, self.rnn_hidden_dim)
                c = c.reshape(-1, self.rnn_hidden_dim)
            h = self.f_comm(c,h)

        weights = self.decoding(h).view(size0,self.n_agents,-1)
        return weights
