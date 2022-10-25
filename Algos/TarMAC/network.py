import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, args):
        super(Actor, self).__init__()
        # that's the encoder
        self.args = args
        self.dim_obs = dim_observation
        self.policy_hidden = torch.zeros(args.n_agents, args.rnn_hidden_size).to(device)
        # self.FC1 = nn.Linear(dim_observation + args.rnn_hidden_size, args.rnn_hidden_size)

        # GRU policy
        self.GRU = nn.GRU(input_size=dim_observation, hidden_size=args.rnn_hidden_size, batch_first=True)
        # Attention module
        # set output of the key, query and value is 0.5*hidden_size
        self.key = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size//2)
        self.query = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size//2)
        self.value = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size //2)
        # action policy
        self.FC2 = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size)
        self.FC3 = nn.Linear(args.rnn_hidden_size, dim_action)

    def forward(self, obs, hidden_data=None):
        if hidden_data==None:
            batch_size = 1
            hidden_state = self.hidden_state.view(-1, self.args.rnn_hidden_size)
        else:
            batch_size = obs.shape[0]
            hidden_state = hidden_data
        # n:agents' number    b: batch size     l:observation's dim     h: hidden size
        obs = obs.view(batch_size*self.args.n_agents, -1, self.dim_obs)                                  #(b*n)*1*l
        hidden_state = hidden_state.view(-1, batch_size*self.args.n_agents, self.args.rnn_hidden_size)   #1*(b*n)*l
        # start from GRU, output is result and hidden states h, the initial h is all zero

        result, h = self.GRU(obs, hidden_state)
        # generate key(signature in the paper), query, value
        h = h.view(batch_size, -1, self.args.rnn_hidden_size)
        k = F.tanh(self.key(h)).view(batch_size, -1, self.args.rnn_hidden_size//2)
        q = F.tanh(self.query(h)).view(batch_size, -1, self.args.rnn_hidden_size//2).permute(0,2,1)
        v = F.tanh(self.value(h)).view(batch_size, -1, self.args.rnn_hidden_size//2)
        # compute the attention
        weights = torch.bmm(k, q)/((self.args.rnn_hidden_size//2)**0.5)
        # zero_vec = -9e15*torch.ones_like(weights)
        # attention = torch.where(adj > 0, weights, zero_vec)
        attention = F.softmax(weights, dim=-1)
        # get the weighted sum value
        out = torch.bmm(attention, v)
        # concate the signature and weighted sum value as presented in paper
        final_hidden = torch.cat([k, out], dim=-1).view(-1, batch_size*self.args.n_agents, self.args.rnn_hidden_size)
        # now let's repeat this process again (just like paper said "multi-round")
        result_2, h_2 = self.GRU(obs, final_hidden)
        result_2 = result_2.view(batch_size, -1, self.args.rnn_hidden_size)
        # generate key(signature in the paper), query, value
        h_2 = h_2.view(batch_size, -1, self.args.rnn_hidden_size)
        k = F.tanh(self.key(h_2)).view(batch_size, -1, self.args.rnn_hidden_size//2)
        q = F.tanh(self.query(h_2)).view(batch_size, -1, self.args.rnn_hidden_size//2).permute(0,2,1)
        v = F.tanh(self.value(h_2)).view(batch_size, -1, self.args.rnn_hidden_size//2)
        # compute the attention
        weights = torch.bmm(k, q)/((self.args.rnn_hidden_size//2)**0.5)
        # zero_vec = -9e15 * torch.ones_like(weights)
        # attention = torch.where(adj > 0, weights, zero_vec)
        attention = F.softmax(weights, dim=-1)
        # get the weighted sum value
        out = torch.bmm(attention, v)
        # concate the signature and weighted sum value as presented in paper
        final_hidden = torch.cat([k, out], dim=-1)
        # finally, lets ouput the features
        result_2 = F.relu(self.FC2(result_2))
        if self.args.scenario in ["simple_spread", "simple_reference"]:
            result_2 = F.tanh(self.FC3(result_2)).squeeze()
        elif self.args.scenario == "predator_prey":
            result_2 = F.softmax(self.FC3(result_2), dim=-1).squeeze()
        else:
            result_2 = F.softmax(self.FC3(result_2), dim=-1).squeeze()
        # result_2 = F.tanh(self.FC3(result_2)).squeeze()
        # remember replace the hidden state
        if batch_size == 1:
            self.policy_hidden = final_hidden.squeeze()
            return result_2
        else:
            return result_2

class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action, args):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.args = args
        self.dim_obs = dim_observation
        self.hidden_state = torch.zeros(args.n_agents, args.rnn_hidden_size).to(device)
        # self.FC1 = nn.Linear(dim_observation + args.rnn_hidden_size, args.rnn_hidden_size)

        # GRU policy
        self.GRU = nn.GRU(input_size=self.dim_obs, hidden_size=args.rnn_hidden_size, batch_first=True)
        # Attention module
        # set output of the key, query and value is 0.5*hidden_size
        self.key = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size // 2)
        self.query = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size // 2)
        self.value = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size // 2)
        # action policy
        self.FC2 = nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size)
        self.FC3 = nn.Linear(args.rnn_hidden_size, 1)

    def forward(self, obs, hidden_data):

        batch_size = obs.shape[0]
        hidden_state = hidden_data
        # obs = torch.cat([obs.view(batch_size, self.args.n_agents, -1), action.view(batch_size,self.args.n_agents, -1)], dim=-1)
        # n:agents' number    b: batch size     l:observation's dim     h: hidden size
        obs = obs.view(batch_size * self.args.n_agents, -1, self.dim_obs)  # (b*n)*1*l
        hidden_state = hidden_state.view(-1, batch_size * self.args.n_agents, self.args.rnn_hidden_size)  # 1*(b*n)*l
        # start from GRU, output is result and hidden states h, the initial h is all zero

        result, h = self.GRU(obs, hidden_state)
        # generate key(signature in the paper), query, value
        h = h.view(batch_size, -1, self.args.rnn_hidden_size)
        k = F.tanh(self.key(h)).view(batch_size, -1, self.args.rnn_hidden_size // 2)
        q = F.tanh(self.query(h)).view(batch_size, -1, self.args.rnn_hidden_size // 2).permute(0, 2, 1)
        v = F.tanh(self.value(h)).view(batch_size, -1, self.args.rnn_hidden_size // 2)
        # compute the attention
        weights = torch.bmm(k, q) / ((self.args.rnn_hidden_size // 2) ** 0.5)
        # zero_vec = -9e15*torch.ones_like(weights)
        # attention = torch.where(adj > 0, weights, zero_vec)
        attention = F.softmax(weights, dim=-1)
        # get the weighted sum value
        out = torch.bmm(attention, v)
        # concate the signature and weighted sum value as presented in paper
        final_hidden = torch.cat([k, out], dim=-1).view(-1, batch_size * self.args.n_agents, self.args.rnn_hidden_size)
        # now let's repeat this process again (just like paper said "multi-round")
        result_2, h_2 = self.GRU(obs, final_hidden)
        result_2 = result_2.view(batch_size, -1, self.args.rnn_hidden_size)
        # generate key(signature in the paper), query, value
        h_2 = h_2.view(batch_size, -1, self.args.rnn_hidden_size)
        k = F.tanh(self.key(h_2)).view(batch_size, -1, self.args.rnn_hidden_size // 2)
        q = F.tanh(self.query(h_2)).view(batch_size, -1, self.args.rnn_hidden_size // 2).permute(0, 2, 1)
        v = F.tanh(self.value(h_2)).view(batch_size, -1, self.args.rnn_hidden_size // 2)
        # compute the attention
        weights = torch.bmm(k, q) / ((self.args.rnn_hidden_size // 2) ** 0.5)
        # zero_vec = -9e15 * torch.ones_like(weights)
        # attention = torch.where(adj > 0, weights, zero_vec)
        attention = F.softmax(weights, dim=-1)
        # get the weighted sum value
        out = torch.bmm(attention, v)
        # concate the signature and weighted sum value as presented in paper

        # finally, lets ouput the features
        result_2 = F.relu(self.FC2(result_2))
        value = self.FC3(result_2)

        return value, final_hidden