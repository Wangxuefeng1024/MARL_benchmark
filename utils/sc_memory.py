import numpy as np
import threading
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.n_states
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

class img_ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape[0]*self.args.state_shape[1]*self.args.state_shape[2]
        self.obs_shape = self.args.obs_shape[0]*self.args.obs_shape[1]*self.args.obs_shape[2]
        self.size = 2
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': [],
                        'u': [],
                        's': [],
                        'r': [],
                        'o_next': [],
                        's_next': [],
                        'u_onehot': [],
                        'terminated': []
                        }
        # thread lock
    def clean(self):
        self.buffers = {'o': [],
                        'u': [],
                        's': [],
                        'r': [],
                        'o_next': [],
                        's_next': [],
                        'u_onehot': [],
                        'terminated': []
                        }

class PPO_RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_one_hot = []
        self.avail_actions = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_one_hot[:]
        del self.avail_actions[:]



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
        self.buffer = collections.deque(maxlen=args.buffer_size)

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
