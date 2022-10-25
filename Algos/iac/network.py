import torch
import torch.nn as nn
import torch.nn.functional as f
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Critic of Central-V
class Actor(nn.Module):
    def __init__(self, input_shape, args):
        super(Actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, args.n_actions)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        if self.args.continuous:
            q = torch.tanh(self.fc3(x))
        else:
            q = f.softmax(self.fc3(x), dim=-1)
        return q


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

class Cen_Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Cen_Critic, self).__init__()
        self.args = args
        self.input_dim = input_shape * self.args.n_agents
        self.fc1 = nn.Linear(self.input_dim, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, self.args.n_agents)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
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
        ).cuda()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = 784
            # n_flatten_2 = self.cnn(torch.randn([input_shape[0], input_shape[1], input_shape[2]]))
        self.linear = nn.Sequential(nn.Linear(n_flatten, self.args.rnn_hidden_dim), nn.ReLU()).cuda()

    def forward(self, observations):
        output = self.cnn(observations.float())
        if len(observations.shape) == 4:
            output = output
        else:
            output = output.flatten()
        output = self.linear(output)
        return f.relu(output)

class CNN_Actor(nn.Module):
    def __init__(self, input, args):
        super(CNN_Actor, self).__init__()
        self.args = args
        self.n_input_channels = input[0]
        self.CNN_encoder = NatureCNN(input, args).cuda()
        # self.qmix_encoder = Actor(input, args)
        self.linear_1 = nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim).cuda()
        self.linear_2 = nn.Linear(self.args.rnn_hidden_dim, self.args.n_actions).cuda()

    def forward(self, input):
        # input = torch.cat([input, last_action, agent_id], dim=-1)
        if len(input.shape)==4:
            input = input.view(-1,3,84,84)
        else:
            input = input.view(3,84,84)
        output = self.CNN_encoder(input)
        output = torch.relu(self.linear_1(output))

        if self.args.continuous:
            output = torch.tanh(self.linear_2(output))
        else:
            output = f.softmax(self.linear_2(output), dim=-1)
        return output

class CNN_Critic(nn.Module):
    def __init__(self, input, args):
        super(CNN_Critic, self).__init__()
        self.args = args
        self.n_input_channels = input[0]
        self.CNN_encoder = NatureCNN(input, args)
        # self.qmix_encoder = Actor(input, args)
        self.linear_1 = nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        self.linear_2 = nn.Linear(self.args.rnn_hidden_dim, 1)

    def forward(self, input):
        # input = torch.cat([input, last_action, agent_id], dim=-1)
        output = self.CNN_encoder(input)
        output = torch.relu(self.linear_1(output))
        output = self.linear_2(output)
        return output

class ICMNetwork(nn.Module):
    """Intrinsic curiosity module (ICM) network"""

    def __init__(
        self, state_size, action_size, args, hidden_dim=128, state_rep_size=64, discrete_actions=True
    ):
        """
        Initialize parameters and build model.
        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param hidden_dim: hidden dimension of networks
        :param state_rep_size: dimension of internal state feature representation
        :param discrete_actions: flag if discrete actions are used (one-hot encoded)
        """
        super(ICMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_rep_size = state_rep_size
        self.discrete_actions = discrete_actions
        self.n_agents = args.n_agents
        # state representation
        self.state_rep = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_rep_size),
        )

        # inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(state_rep_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )

        # forward model
        self.forward_model = nn.Sequential(
            nn.Linear(state_rep_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_rep_size),
        )

    def forward(self, state, next_state, action):
        """
        Compute forward pass over ICM network
        :param state: current state
        :param next_state: reached state
        :param action: applied action
        :return: predicted_action, predicted_next_state_rep, next_state_rep
        """
        # compute state representations
        state_rep = self.state_rep(torch.tensor(state).to(device).float())
        next_state_rep = self.state_rep(torch.tensor(next_state).to(device).float())
        action = torch.tensor(action).to(device)
        # inverse model output
        inverse_input = torch.cat([state_rep, next_state_rep], 1)
        predicted_action = self.inverse_model(inverse_input)

        # forward model output
        forward_input = torch.cat([state_rep, action.float()], 1)
        predicted_next_state_rep = self.forward_model(forward_input)

        return predicted_action, predicted_next_state_rep, next_state_rep
