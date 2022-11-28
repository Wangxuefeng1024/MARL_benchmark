from collections import namedtuple
import random

Experience = namedtuple('Experience',
						('states', 'actions', 'next_states', 'rewards'))

Experience_rnn = namedtuple('Experience',
						('states', 'actions', 'next_states', 'rewards', 'data', 'hidden_states', 'previous_hidden'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        Experience = namedtuple('Experience',
                                ('states', 'actions', 'next_states', 'rewards'))

        self.memory[self.position] = Experience(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        # print(len(self.memory),batch_size)
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

class ReplayMemory_rnn:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience_rnn(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        # print(len(self.memory),batch_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)