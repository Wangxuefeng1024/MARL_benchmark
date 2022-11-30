import numpy as np


class TwoStepPayOffMatrix:
    def __init__(self, n=2, value_list=None):
        if value_list is None:
            value_list = [0. for _ in range(n * n)]
        self.N = n
        assert len(value_list[0]) == self.N * self.N
        assert len(value_list[1]) == self.N * self.N
        self.matrix = None
        self.matrix_1 = np.array(value_list[0]).reshape(self.N, self.N)
        self.matrix_2 = np.array(value_list[1]).reshape(self.N, self.N)
        self.states = np.eye(4)
        self.state_num = 0
        self.done = False

    def reset(self):
        self.state_num = 0
        self.done = False
        self.matrix = None
        return (
            self.states[self.state_num], [self.states[self.state_num], self.states[self.state_num]]
        )

    def step(self, actions):
        if self.state_num == 0 and actions[0] == 0:
            self.matrix = self.matrix_1
            self.state_num = 1
            reward = 0.
        elif self.state_num == 0 and actions[0] == 1:
            self.matrix = self.matrix_2
            self.state_num = 2
            reward = 0.
        elif self.state_num == 1 or self.state_num == 2:
            self.state_num = 3
            reward = self.matrix[actions[0]][actions[1]]
            self.done = True

        return (
            self.states[self.state_num], [self.states[self.state_num], self.states[self.state_num]], reward, self.done
        )
