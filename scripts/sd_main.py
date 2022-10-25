
from Envs.social_dilemmas.envs.env_creator import get_env_creator
import torch
from tensorboardX import SummaryWriter
from Algos.qmix.qmix_agent import QMIX
from Algos.iql.iql_agent import IQL
from Algos.coma.coma_agent import COMA
from copy import deepcopy
import argparse
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    BoolTensor = torch.cuda.BoolTensor if args.cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    torch.manual_seed(args.seed)
    if args.tensorboard:
        writer = SummaryWriter(
            log_dir='runs/' + args.algo + "/"  + str(
                args.n_agents) + args.reminder)


    env = get_env_creator(env=args.scenario,
                        num_agents=args.n_agents,
                        use_collective_reward=False,
                        inequity_averse_reward=False,
                        alpha=0.0,
                        beta=0.0,
                        num_switches=6,)

    if args.algo == "qmix":
        model = QMIX(env, args)
    elif args.algo == "iql":
        model = IQL(env, args)
    elif args.algo == "coma":
        model = COMA(env, args)
    else:
        model = QMIX(env, args)
    print(model)
    # episode_id = 0
    # train_steps = 0
    # total_step = 0
    # we set 2000000 total timesteps now
    time_steps, train_steps, evaluate_steps = 0, 0, -1
    while time_steps < args.max_steps:
        print('time_steps {}'.format(time_steps))
        # evaluate 20 episodes after training every 100 episodes
        # o, s, u, u_onehot, avail_u, r, terminate, padded = [], [], [], [], [], [], [], []
        if time_steps // args.evaluate_cycle > evaluate_steps:
            win_rate, episode_reward = model.evaluate()
            model.win_rates.append(win_rate)
            model.episode_rewards.append(episode_reward)
            evaluate_steps += 1
            if args.tensorboard:
                writer.add_scalar(tag='agent/win rates', global_step=evaluate_steps, scalar_value=win_rate)
                writer.add_scalar(tag='agent/reward', global_step=evaluate_steps, scalar_value=episode_reward)

        episodes = []
        # 收集self.args.n_episodes个episodes
        for episode_idx in range(args.n_episodes):
            episode, _, _, steps = model.generate_episode(episode_idx)
            episodes.append(episode)
            time_steps += steps
        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        model.buffer.store_episode(episode_batch)
        for train_step in range(args.train_steps):
            mini_batch = model.buffer.sample(min(model.buffer.current_size, model.args.batch_size))
            model.train(mini_batch, train_steps)
            train_steps += 1
        # if episode_id % args.save_cycle == 0:
        #     model.save_model(train_steps)
    win_rate, episode_reward = model.evaluate()
    print('win_rate is ', win_rate)
    model.win_rates.append(win_rate)
    model.episode_rewards.append(episode_reward)
    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # the environment setting
    # parser.add_argument('--difficulty', type=str, default='5', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')

    parser.add_argument('--scenario', type=str, default='harvest', help='harvest, cleanup, switch(not available now)')

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
    parser.add_argument('--n_actions', type=int, default=5, help='the number of actions')

    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--tensorboard', type=bool, default=True, help='whether to use tensorboard')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='critic learning rate')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='actor learning rate')

    parser.add_argument('--epsilon', type=float, default=1.0, help='discount factor')
    parser.add_argument('--anneal_epsilon', type=float, default=0.000019, help='discount factor')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='discount factor')
    parser.add_argument("--use-collective-reward", type=bool, default=False, help="Give each agent the collective reward across all agents")
    parser.add_argument("--inequity-averse-reward", type=bool, default=False, help="Use inequity averse rewards from 'Inequity aversion \improves cooperation in intertemporal social dilemmas'")
    parser.add_argument("--alpha", type=float, default=5, help="Advantageous inequity aversion factor")
    parser.add_argument("--beta", type=float, default=0.05, help="Disadvantageous inequity aversion factor")
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


