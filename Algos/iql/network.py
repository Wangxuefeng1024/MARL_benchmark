import torch
import torch.nn as nn
import torch.nn.functional as f
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(Actor, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, input_shape, args):
        super(NatureCNN, self).__init__()
        self.args = args
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.randn([input_shape[0], input_shape[1], input_shape[2]])).shape[1] * 16
            # n_flatten_2 = self.cnn(torch.randn([input_shape[0], input_shape[1], input_shape[2]]))
        self.linear = nn.Sequential(nn.Linear(n_flatten, self.args.rnn_hidden_dim), nn.ReLU())

    def forward(self, observations):
        output = self.cnn(observations)
        output = self.linear(output.flatten())
        return f.relu(output)

class CNN_Actor(nn.Module):
    def __init__(self, input, args):
        super(CNN_Actor, self).__init__()
        self.args = args
        self.n_input_channels = input[0]
        self.CNN_encoder = NatureCNN(input, args)
        # self.qmix_encoder = Actor(input, args)
        self.linear_1 = nn.Linear(self.args.rnn_hidden_dim+self.args.n_actions+self.args.n_agents, self.args.rnn_hidden_dim)
        self.linear_2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions)

    def forward(self, input, last_action, agent_id):
        # input = torch.cat([input, last_action, agent_id], dim=-1)
        last_action = torch.from_numpy(last_action).type(torch.float).to(device)
        agent_id = torch.from_numpy(agent_id).type(torch.float).to(device)
        output = self.CNN_encoder(input)
        output = torch.relu(self.linear_1(torch.cat([output,last_action,agent_id], dim=-1)))
        output = self.linear_2(output)
        return output
