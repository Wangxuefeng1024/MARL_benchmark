from Envs.payoff_matrix.one_step_payoff_matrix import OneStepPayOffMatrix
from Envs.payoff_matrix.two_step_payoff_matrix import TwoStepPayOffMatrix
from policy.qmix import QMIX
from policy.vdn import VDN
import argparse

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


# def init_hidden(episode_num):
#     # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
#     eval_hidden = torch.zeros((episode_num, n_agents, args.rnn_hidden_dim))
#     target_hidden = torch.zeros((episode_num, n_agents, self.args.rnn_hidden_dim))


def run(self):
    step = 0
    while step < self.training_steps:
        state, observations = self.env.reset()
        done = False
        if self.args.base_net == 'rnn':
            h_out = self.init_hidden(self.args)
        while not done:
            if self.args.base_net == 'rnn':
                h_in = h_out
                actions, h_out = self.agents.choose_action(observations, h_in)
            else:
                actions = self.agents.choose_action(observations)
            next_state, next_observations, reward, done = self.env.step(actions)
            print('step: {0}, state: {1}, actions: {2}, reward: {3}'.format(step, state, actions, reward))
            done_mask = 0.0 if done else 1.0

            if self.args.base_net == 'rnn':
                self.replay_memory.put(
                    [state, observations, actions, reward, next_state, next_observations, h_in, h_out, done_mask]
                )
            else:
                self.replay_memory.put(
                    [state, observations, actions, reward, next_state, next_observations, done_mask]
                )

            if self.replay_memory.size() >= self.args.batch_size:
                batch = {}
                if self.args.base_net == 'rnn':
                    s, o, a, r, s_prime, o_prime, hidden_in, hidden_out, done_mask = self.replay_memory.sample(
                        self.args.batch_size
                    )
                    batch['hidden_in'] = hidden_in
                    batch['hidden_out'] = hidden_out
                else:
                    s, o, a, r, s_prime, o_prime, done_mask = self.replay_memory.sample(self.args.batch_size)
                batch['state'] = s
                batch['observation'] = o
                batch['action'] = a
                batch['reward'] = r
                batch['next_state'] = s_prime
                batch['next_observation'] = o_prime
                batch['done_mask'] = done_mask

                loss = self.agents.train(batch, step)

                if step % self.args.print_interval == 0:
                    print("step: {0}, loss: {1}".format(step, loss))

            state = next_state
            observations = next_observations
            step += 1

            if done:
                break
    self.agents.save_model()


def play(self):
    self.agents.load_model()

    q_value_list, iteration, selected_q_value_list, q_value_list_0, q_value_list_1, q_value_list_2, \
    iteration_0, iteration_1, iteration_2 = None, None, None, None, None, None, None, None, None

    if self.args.env_name == 'one_step_payoff_matrix':
        q_value_list = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        iteration = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
    elif self.args.env_name == 'two_step_payoff_matrix':
        q_value_list_0 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        iteration_0 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        q_value_list_1 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        iteration_1 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        q_value_list_2 = [[0. for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
        iteration_2 = [[0 for _ in range(self.args.n_actions)] for _ in range(self.args.n_actions)]
    else:
        raise Exception("Wrong env name.")

    step = 0
    while step < self.playing_steps:
        state, observations = self.env.reset()
        done = False

        if self.args.base_net == 'rnn':
            h_out = self.init_hidden(self.args)
        state_num = 0
        while not done:
            if self.args.base_net == 'rnn':
                h_in = h_out
                actions, h_out, q_total_evals = self.agents.choose_action(observations, h_in=h_in, state=state)
            else:
                actions, q_total_evals = self.agents.choose_action(observations, state=state)
            next_state, next_observations, reward, done = self.env.step(actions)

            state = next_state
            observations = next_observations

            if self.args.env_name == 'one_step_payoff_matrix':
                q_value_list[actions[0]][actions[1]] += q_total_evals
                iteration[actions[0]][actions[1]] += 1
            elif self.args.env_name == 'two_step_payoff_matrix':
                if state_num == 0:
                    if actions[0] == 0:
                        state_num = 1
                    if actions[0] == 1:
                        state_num = 2
                    q_value_list_0[actions[0]][actions[1]] += q_total_evals
                    iteration_0[actions[0]][actions[1]] += 1
                else:
                    if state_num == 1:
                        q_value_list_1[actions[0]][actions[1]] += q_total_evals
                        iteration_1[actions[0]][actions[1]] += 1
                    elif state_num == 2:
                        q_value_list_2[actions[0]][actions[1]] += q_total_evals
                        iteration_2[actions[0]][actions[1]] += 1

            step += 1

            if done:
                break

    if self.args.env_name == 'one_step_payoff_matrix':
        for i in range(self.args.n_actions):
            for j in range(self.args.n_actions):
                q_value_list[i][j] /= iteration[i][j]
        print(q_value_list)
    elif self.args.env_name == 'two_step_payoff_matrix':
        for i in range(self.args.n_actions):
            for j in range(self.args.n_actions):
                q_value_list_0[i][j] /= iteration_0[i][j]
                q_value_list_1[i][j] /= iteration_1[i][j]
                q_value_list_2[i][j] /= iteration_2[i][j]
        print(q_value_list_0)
        print(q_value_list_1)
        print(q_value_list_2)


def main(args):
    # One Step Pay-off Matrix or Two Step Pay-off Matrix
    if args.env_name == 'one_step_payoff_matrix':
        args.state_shape = 2
        args.obs_shape = 2
        args.n_actions = 3
        value_list = [10.4, 0., 10., 0., 10., 10., 10., 10., 10.]
        env = OneStepPayOffMatrix(value_list=value_list)
    elif args.env_name == 'two_step_payoff_matrix':
        args.state_shape = 4
        args.obs_shape = 4
        args.n_actions = 2
        value_list = [[7., 7., 7., 7.], [0., 1., 1., 8.]]
        env = TwoStepPayOffMatrix(value_list=value_list)
    else:
        raise Exception("Wrong env name.")
    print()
    print("* Environment Name: {}".format(args.env_name))
    print("* Initial Value List: {}".format(value_list))
    print()

    if args.algo == "vdn":
        runner = VDN(env, args)
    elif args.algo == "qmix":
        runner = QMIX(env, args)
    if args.play:
        play(runner)
    else:
        run(runner)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='5', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn dimension')
    # parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn dimension')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='hyper dimension')
    parser.add_argument('--qmix_hidden_dim', type=int, default=32, help='qmix dimension')
    parser.add_argument('--critic_dim', type=int, default=128, help='critic dimension')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--n_agents', type=int, default=5, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--test_episodes', type=int, default=20, help='random seed')
    parser.add_argument('--train_steps', type=int, default=1, help='random seed')

    parser.add_argument('--algo', type=str, default='qmix', help='the algorithm to train the agent')

    parser.add_argument('--max_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--tensorboard', type=bool, default=True, help='whether to use tensorboard')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='critic learning rate')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='actor learning rate')

    parser.add_argument('--epsilon', type=float, default=1.0, help='discount factor')
    parser.add_argument('--anneal_epsilon', type=float, default=0.000019, help='discount factor')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='discount factor')

    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='how often to evaluate the model')
    parser.add_argument('--save_cycle', type=int, default=5000, help='how often to save the model')
    parser.add_argument('--grad_norm_clip', type=int, default=10, help='prevent gradient explosion')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--buffer_size', type=int, default=int(2e4), help='buffer size')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--two_hyper_layers', type=bool, default=True, help='whether to use two hyper layers')
    parser.add_argument('--td_lambda', type=float, default=0.8, help='value of td lambda')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--reminder', type=str, default="normal", help='something helps to remind')
    parser.add_argument('--epsilon_anneal_scale', type=str, default="step", help='something helps to remind')
    args = parser.parse_args()
    main(args)
