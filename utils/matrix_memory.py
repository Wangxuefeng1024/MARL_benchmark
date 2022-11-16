import torch
import collections
import random


class ReplayMemoryForMLP:
    def __init__(self, args):
        self.args = args
        self.buffer = collections.deque(maxlen=self.args.replay_memory_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, o_lst, a_lst, r_lst, s_prime_lst, o_prime_lst, done_mask_lst = [], [], [], [], [], [], []

        for transition in mini_batch:
            s, o, a, r, s_prime, o_prime, done_mask = transition
            s_lst.append(s)
            o_lst.append(o)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            o_prime_lst.append(o_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(o_lst, dtype=torch.float), \
               torch.tensor(a_lst), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(o_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


def init_hidden(args):
    h_out = torch.zeros([args.n_agents, args.rnn_hidden_dim], dtype=torch.float)
    return h_out


class ReplayMemoryForRNN:
    def __init__(self, args):
        self.args = args
        self.buffer = collections.deque(maxlen=args.replay_memory_size)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, o_lst, a_lst, r_lst, s_prime_lst, o_prime_lst, h_in_lst, h_out_lst, done_mask_lst = \
            [], [], [], [], [], [], [], [], []

        for transition in mini_batch:
            s, o, a, r, s_prime, o_prime, h_in, h_out, done_mask = transition
            s_lst.append(s)
            o_lst.append(o)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            o_prime_lst.append(o_prime)
            h_in = h_in.reshape([self.args.n_agents, self.args.rnn_hidden_dim])
            h_out = h_out.reshape([self.args.n_agents, self.args.rnn_hidden_dim])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(o_lst, dtype=torch.float), \
               torch.tensor(a_lst), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(o_prime_lst, dtype=torch.float), \
               torch.stack(h_in_lst).detach(), \
               torch.stack(h_out_lst).detach(), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)