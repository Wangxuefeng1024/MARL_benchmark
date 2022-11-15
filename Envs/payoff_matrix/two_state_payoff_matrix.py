import numpy as np

# value_list1 = [4, -2, -2; -2, 0, 0; -2, 0, 0]
# value_list2 = [-2, 0, 0; 4, -2, -2; -2, 0, 0]

class OneStepPayOffMatrix:
    def __init__(self, n=3, value_list=None):
        if value_list is None:
            value_list = [0. for _ in range(n*n)]
        self.N = n
        assert len(value_list) == self.N * self.N
        self.matrix = np.array(value_list).reshape(self.N, self.N)
        self.states = np.eye(2)
        self.t = 0
        self.done = False

    def reset(self):
        self.t = 0
        self.done = False
        return [
            self.states[self.t], [self.states[self.t], self.states[self.t]]
        ]

    def step(self, actions):
        self.t += 1
        self.done = True
        return [
            self.states[self.t], [self.states[self.t], self.states[self.t]],
            self.matrix[actions[0]][actions[1]], self.done
        ]
