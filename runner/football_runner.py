import torch
from tensorboardX import SummaryWriter
from Envs.football.football_env import FootballEnv
from policy.qmix import gfoot_qmix
from policy.ppo import MAPPO_GRF
# from Algos.iql.iql_agent import IQL
# from Algos.coma.coma_agent import COMA
# from Algos.TarMAC.tarmac_agent import gfoot_TarMAC, MAPPO
import argparse
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    BoolTensor = torch.cuda.BoolTensor if args.cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    torch.manual_seed(args.seed)
    if args.tensorboard:
        writer = SummaryWriter(
            log_dir='runs/' + args.algo + "/" + args.scenario)
    # create environments
    # set the number of agents by the sceanrio
    if args.scenario in ['academy_pass_and_shoot_with_keeper', 'academy_run_pass_and_shoot_with_keeper']:
        args.n_agents = 2
        args.obs_shape = 98
        args.n_states = args.obs_shape*args.n_agents
    elif args.scenario == 'academy_3_vs_1_with_keeper':
        args.n_agents = 3
        args.obs_shape = 105
        args.n_states = args.obs_shape*args.n_agents
    else:
        args.n_agents = 2

        args.obs_shape = 115
        args.n_states = args.obs_shape*args.n_agents

    # args.n_states = 29
    # args.obs_shape = 33

    env = FootballEnv(args=args)
    # create model
    if args.algo == "qmix":
        model = gfoot_qmix(env, args)
    # elif args.algo == "iql":
    #     model = IQL(env, args)
    # elif args.algo == "coma":
    #     model = COMA(env, args)
    # elif args.algo == "tarmac":
    #     model = gfoot_TarMAC(env, args)
    elif args.algo == "mappo":
        model = MAPPO_GRF(env, args)
    # else:
    #     model = gfoot_qmix(env, args)
    print(model)



    # get the whole rollout now
    # we utilize on-policy ppo or dqn now, for the dqn we are sample the whole episodes to the replay buffer
    # episode_id = 0
    # train_steps = 0
    total_step = 0

    time_steps, train_steps, evaluate_steps = 0, 0, -1
    total_rewards = 0
    while time_steps < args.max_steps:
        print("[time_steps %05d] reward %6.4f" % (time_steps, total_rewards))

        episodes = []
        # evaluate 20 episodes after training every 100 episodes
        #
        # if time_steps // args.evaluate_cycle > evaluate_steps:
        #     win_rate, episode_reward = model.evaluate()
        #     model.win_rates.append(win_rate)
        #     model.episode_rewards.append(episode_reward)
        #     evaluate_steps += 1
        #     print('win rates is:', win_rate)
        #     if args.tensorboard:
        #         writer.add_scalar(tag='agent/win rates', global_step=train_steps, scalar_value=win_rate)
        #         writer.add_scalar(tag='agent/reward', global_step=train_steps, scalar_value=episode_reward)

        episode, _, _, steps = model.generate_episode(train_steps)
        episodes.append(episode)
        time_steps += steps

        # if args.algo != "mappo":
        episode_batch = episodes[0]
        total_rewards = np.sum(episode_batch['r'])

        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            # if args.algo != "mappo":
        model.buffer.store_episode(episode_batch)
        writer.add_scalar(tag='agent/rewards', global_step=time_steps, scalar_value=total_rewards)
        for train_step in range(args.train_steps):
            if args.algo != "mappo":
                mini_batch = model.buffer.sample(min(model.buffer.current_size, model.args.batch_size))
                loss = model.train(mini_batch, train_steps)
                writer.add_scalar(tag='agent/loss', global_step=time_steps, scalar_value=loss)
            if args.algo == "mappo":
                model.train()
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
    parser.add_argument('--difficulty', type=str, default='5', help='the difficulty of the game')
    parser.add_argument('--scenario', type=str, default='academy_pass_and_shoot_with_keeper', help='academy_pass_and_shoot_with_keeper, academy_run_pass_and_shoot_with_keeper, academy_3_vs_1_with_keeper')
    parser.add_argument('--reward', type=str, default='scoring', help='the map of the game')

    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='5m_vs_6m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    # parser.add_argument('--state_shape', type=int, default=230, help='dimension of global state')
    # parser.add_argument('--obs_shape', type=int, default=115, help='dimension of local observation')
    parser.add_argument('--episode_limit', type=int, default=200, help='episode_limit')

    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn dimension')
    parser.add_argument('--n_actions', type=int, default=19, help='number of actions')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='hyper dimension')
    parser.add_argument('--qmix_hidden_dim', type=int, default=32, help='qmix dimension')
    parser.add_argument('--critic_dim', type=int, default=128, help='critic dimension')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--n_agents', type=int, default=3, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--test_episodes', type=int, default=20, help='random seed')
    parser.add_argument('--train_steps', type=int, default=1, help='random seed')
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
    parser.add_argument('--anneal_epsilon', type=float, default=0.000019, help='discount factor')
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