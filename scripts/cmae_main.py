import supersuit as ss
import torch
import argparse
from Algos.iac.iac_agent import PPO_IAC, Central_PPO, ICM_PPO
from Envs.sparse.push_box import PushBox
from Envs.sparse.rooms import Rooms
from Envs.sparse.secret_rooms import SecretRooms
from tensorboardX import SummaryWriter
import numpy as np

def main(args):
    if args.env == "pushbox":
        env = PushBox()
        args.n_actions = 4
        args.n_states = 4
        args.n_agents = 2
    elif args.env == "rooms":
        env = Rooms()
        args.n_actions = 4
        args.n_agents = 2
        args.n_states = 3
    elif args.env == "secret_rooms":
        env = SecretRooms()
        args.n_actions = 4
        args.n_agents = 2
        args.n_states = 3
    else:
        env = SecretRooms()
        args.n_actions = 4
        args.n_agents = 2
        args.n_states = 3

    torch.manual_seed(args.seed)
    # if args.tensorboard and args.mode == "train":
    writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + args.env)
    if args.algo == "iac":
        model = PPO_IAC(env, args)
    elif args.algo == "coma":
        model = Central_PPO(env, args)
    elif args. algo == "icm":
        model = ICM_PPO(env, args)
    else:
        model = Central_PPO(env, args)
    episode = 0
    while True:
        # obtain the rollout for current envrionments
        obs = env.reset()
        episode += 1
        done = False
        accum_reward = 0
        rewards = []
        # model.init_hidden(1)
        for i in range(args.episode_length):
            actions = []
            act_logprobs = []
            # step_reward = 0
            if done:
                break
            for j in range(args.n_agents):
                act, act_logprob = model.choose_action(np.array(obs[j]))
                actions.append(act)
                act_logprobs.append(act_logprob)
            next_obs, reward, done = env.step(np.array(actions))
            if args.algo == "icm":
                new_reward = model.ICM.compute_intrinsic_reward(obs, actions, next_obs, True, True)
                fake_reward = new_reward + reward
            # if j == args.n_agents-1:
            model.buffer.is_terminals.append(done)
            # rewards.append(reward)
            if args.algo == "icm":
                model.buffer.rewards.append(fake_reward)
            else:
                model.buffer.rewards.append(reward)
            model.buffer.actions.append(actions)
            model.buffer.states.append(obs)
            model.buffer.logprobs.append(act_logprobs)
            accum_reward += reward
            obs = next_obs
            if done:
                break
            # act = np.array(act)
            #env.render()
            # env.step(act)
        writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward)
        # rollout is done, update the network
        c_loss, a_loss = model.update()

        # loss = torch.mean(loss).item()
        print(" a_loss %3.2f c_loss %3.2f" % (c_loss, a_loss), end='')
        print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
        # if writer:
        writer.add_scalars('agent/loss', global_step=episode,
                           tag_scalar_dict={'actor': a_loss, 'critic': c_loss})
        if episode >=2000000:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='5', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')

    parser.add_argument('--env', type=str, default='secret_rooms', help='the map of the game')

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='rnn dimension')
    parser.add_argument('--episode_length', type=int, default=100, help='rnn dimension')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='hyper dimension')
    parser.add_argument('--qmix_hidden_dim', type=int, default=32, help='qmix dimension')
    parser.add_argument('--critic_dim', type=int, default=128, help='critic dimension')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    # parser.add_argument('--n_agents', type=int, default=10, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--test_episodes', type=int, default=20, help='random seed')
    parser.add_argument('--train_steps', type=int, default=1, help='random seed')
    # parser.add_argument('--continuous', type=bool, default=True, help='whether to use the last action to choose action')

    parser.add_argument('--algo', type=str, default='icm', help='the algorithm to train the agent')

    parser.add_argument('--max_steps', type=int, default=2000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--tensorboard', type=bool, default=True, help='whether to use tensorboard')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-4, help='critic learning rate')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='actor learning rate')

    parser.add_argument('--epsilon', type=float, default=0.9, help='discount factor')
    parser.add_argument('--anneal_epsilon', type=float, default=0.000019, help='discount factor')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='discount factor')

    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='how often to evaluate the model')
    parser.add_argument('--save_cycle', type=int, default=5000, help='how often to save the model')
    parser.add_argument('--grad_norm_clip', type=int, default=10, help='prevent gradient explosion')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--episode_limit', type=int, default=100, help='batch size')

    parser.add_argument('--buffer_size', type=int, default=int(2e4), help='buffer size')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--two_hyper_layers', type=bool, default=True, help='whether to use two hyper layers')
    parser.add_argument('--continuous', type=bool, default=False, help='whether to use two hyper layers')

    parser.add_argument('--td_lambda', type=float, default=0.8, help='value of td lambda')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--reminder', type=str, default="normal", help='something helps to remind')
    parser.add_argument('--epsilon_anneal_scale', type=str, default="step", help='something helps to remind')
    args = parser.parse_args()
    main(args)