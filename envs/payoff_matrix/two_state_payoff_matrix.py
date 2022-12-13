import numpy as np

# value_list1 = [4, -2, -2; -2, 0, 0; -2, 0, 0]
# value_list2 = [-2, 0, 0; 4, -2, -2; -2, 0, 0]

class TwoStatePayOffMatrix:
    def __init__(self, n=3, value_list=None):
        if value_list is None:
            value_list = [0. for _ in range(n*n)]
        self.N = n
        assert len(value_list[0]) == self.N * self.N
        self.matrix0 = np.array(value_list[0]).reshape(self.N, self.N)
        self.matrix1 = np.array(value_list[1]).reshape(self.N, self.N)
        self.agent_obs1 = [[1,1],[1,2]]
        self.agent_obs2 = [[2,2],[2,2]]
        self.states = np.eye(2)
        self.t = 0
        self.done = False

    def reset(self):
        self.t = 0
        self.done = False
        if np.random.uniform(0, 1, 1) < 0.5:
            self.matrix = self.matrix0
            self.agent_obs = [self.agent_obs1[0], self.agent_obs2[0]]
            self.state = [0, 1]
            self.STATE = 0
        else:
            self.matrix = self.matrix1
            self.agent_obs = [self.agent_obs1[1], self.agent_obs2[1]]
            self.state = [1, 0]
            self.STATE = 1
        return [
            np.array(self.state), np.array([self.agent_obs[0], self.agent_obs[1]])
        ]

    def step(self, actions):
        self.t += 1
        self.done = True
        return [
            np.array([self.state[0] + 1, self.state[1] + 1]), np.array([self.agent_obs[0], self.agent_obs[1]]),
            self.matrix[actions[0]][actions[1]], self.done
        ]
