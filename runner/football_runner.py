import sys
sys.path.append('../')
import torch
from tensorboardX import SummaryWriter
from envs.football.football_env import FootballEnv
from policy.qmix import gfoot_qmix
import argparse
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # set seed
    torch.manual_seed(args.seed)

    # set tensorboard
    if args.tensorboard:
        writer = SummaryWriter(
            log_dir='../runs/' + args.algo + "/" + args.scenario)

    # create environments
    if args.scenario == 'academy_pass_and_shoot_with_keeper':
        args.n_agents = 2
        args.obs_shape = 98
        args.n_states = args.obs_shape*args.n_agents
        args.episode_limit = 150

    elif args.scenario == 'academy_run_pass_and_shoot_with_keeper':
        args.n_agents = 2
        args.obs_shape = 98
        args.n_states = args.obs_shape*args.n_agents
        args.episode_limit = 200

    elif args.scenario == 'academy_3_vs_1_with_keeper':
        args.n_agents = 3
        args.obs_shape = 105
        args.n_states = args.obs_shape*args.n_agents
        args.episode_limit = 250
    else:
        args.n_agents = 2
        args.obs_shape = 115
        args.n_states = args.obs_shape*args.n_agents

    env = FootballEnv(args=args)

    # create model
    if args.algo == "qmix":
        model = gfoot_qmix(env, args)
    else:
        model = gfoot_qmix(env, args)
    print(model)

    time_steps, train_steps, evaluate_steps = 0, 0, -1
    total_rewards = 0

    # generate episodic rollout and store into the replay buffer
    while time_steps < args.max_steps:
        print("[time_steps %05d] reward %6.4f" % (time_steps, total_rewards))
        episodes = []
        # evaluate 20 episodes after training every 100 episodes
        if time_steps // args.evaluate_cycle > evaluate_steps:
            win_rate, episode_reward = model.evaluate()
            model.win_rates.append(win_rate)
            model.episode_rewards.append(episode_reward)
            evaluate_steps += 1
            print('win rates is:', win_rate)
            if args.tensorboard:
                writer.add_scalar(tag='agent/win rates', global_step=train_steps, scalar_value=win_rate)
                writer.add_scalar(tag='agent/reward', global_step=train_steps, scalar_value=episode_reward)

        episode, _, _, steps = model.generate_episode(train_steps)
        episodes.append(episode)
        time_steps += steps
        episode_batch = episodes[0]
        total_rewards = np.sum(episode_batch['r'])
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        model.buffer.store_episode(episode_batch)
        writer.add_scalar(tag='agent/rewards', global_step=time_steps, scalar_value=total_rewards)
        for train_step in range(args.train_steps):
            mini_batch = model.buffer.sample(min(model.buffer.current_size, model.args.batch_size))
            loss = model.train(mini_batch, train_steps)
            writer.add_scalar(tag='agent/loss', global_step=time_steps, scalar_value=loss)
            train_steps += 1
        if train_steps % args.save_cycle == 0:
            model.save_model(train_steps)
        train_steps += 1
    win_rate, episode_reward = model.evaluate()
    print('win_rate is ', win_rate)
    model.win_rates.append(win_rate)
    model.episode_rewards.append(episode_reward)
    env.close()

def _t2n(x):
    return x.detach().cpu().numpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper', help='academy_pass_and_shoot_with_keeper, academy_run_pass_and_shoot_with_keeper, academy_3_vs_1_with_keeper')
    parser.add_argument('--reward', type=str, default='scoring', help='the reward type of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn dimension')
    parser.add_argument('--n_actions', type=int, default=19, help='number of actions')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='hyper dimension')
    parser.add_argument('--qmix_hidden_dim', type=int, default=32, help='qmix dimension')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--test_episodes', type=int, default=20, help='how many episodes need to evaluate')

    parser.add_argument('--rnn_hidden_size', type=int, default=64, help='random seed')

    parser.add_argument('--algo', type=str, default='qmix', help='the algorithm to train the agent')

    parser.add_argument('--max_steps', type=int, default=10000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--tensorboard', type=bool, default=True, help='whether to use tensorboard')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--c_lr', type=float, default=1e-4, help='critic learning rate')
    parser.add_argument('--a_lr', type=float, default=1e-4, help='actor learning rate')
    parser.add_argument('--grad_norm', type=float, default=1, help='actor learning rate')

    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='discount factor')
    parser.add_argument('--clip_param', type=float, default=0.2, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='discount factor')
    parser.add_argument('--lamda', type=float, default=0.95, help='discount factor')
    parser.add_argument('--anneal_epsilon', type=float, default=0.00004, help='discount factor')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='discount factor')
    parser.add_argument('--ppo_n_epochs', type=int, default=10, help='how often to evaluate the model')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=20000, help='how often to evaluate the model')
    parser.add_argument('--save_cycle', type=int, default=5000, help='how often to save the model')
    parser.add_argument('--grad_norm_clip', type=int, default=10, help='prevent gradient explosion')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--buffer_size', type=int, default=int(1e4), help='buffer size')
    parser.add_argument('--k_epochs', type=int, default=8, help='buffer size')

    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--continuous', type=bool, default=False, help='whether to load the pretrained model')

    parser.add_argument('--two_hyper_layers', type=bool, default=True, help='whether to use two hyper layers')
    parser.add_argument('--td_lambda', type=float, default=0.8, help='value of td lambda')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--reminder', type=str, default="normal", help='something helps to remind')
    parser.add_argument('--epsilon_anneal_scale', type=str, default="step", help='something helps to remind')
    args = parser.parse_args()
    main(args)