import sys
sys.path.append('../')
import torch
import numpy as np
import datetime
import argparse
from envs.cooperative_navigation.make_env import make_env
from policy.ddpg import MADDPG
from policy.tarmac import TarMAC
from tensorboardX import SummaryWriter
from utils.utils import reward_from_state

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args)
    n_agents = args.n_agents
    n_actions = env.world.dim_p
    n_states = env.observation_space[0].shape[0]

    # set seed
    torch.manual_seed(args.seed)

    # set tensorboard
    writer = SummaryWriter(log_dir='../runs/'+ args.algo + "/" + "cooperative_navigation/" + str(args.n_agents))

    # set algorithm
    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)
    else:
        model = TarMAC(n_states, n_actions, n_agents, args)

    print(model)

    # generate rollout and store them in the replay buffer
    episode = 0
    total_step = 0
    while episode < args.max_episodes:
        state = env.reset()
        episode += 1
        step = 0
        accum_reward = 0 # accum_reward is the summation of all agents
        rewardA = 0      # rewradA is the reward of first agent
        while True:
            if args.algo == "tarmac":
                action, hidden_state, previous_hidden = model.choose_action(state, noisy=True)
            else:
                action = model.choose_action(state, noisy=True)

            next_state, reward, done, info = env.step(action)

            step += 1
            total_step += 1
            reward = np.array(reward)
            rew1 = reward_from_state(next_state, n_agents, env)
            reward = rew1 + (np.array(reward, dtype=np.float32) / 100.)

            accum_reward += sum(reward)
            rewardA += reward[0]

            if args.algo == "maddpg" or args.algo == "commnet":
                obs = torch.from_numpy(np.stack(state)).float().to(device)
                obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                if step != args.episode_length - 1:
                    next_obs = obs_
                else:
                    next_obs = None
                rw_tensor = torch.FloatTensor(reward).to(device)
                ac_tensor = torch.FloatTensor(action).to(device)
                if args.algo == "commnet" and next_obs is not None:
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                if args.algo == "maddpg":
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
            elif args.algo == "tarmac":
                obs = torch.from_numpy(np.stack(state)).float().to(device)
                obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)
                if step != args.episode_length - 1 or args.scenario == 'traffic_junction':
                    next_obs = obs_
                else:
                    next_obs = None
                rw_tensor = torch.FloatTensor(reward).to(device)
                ac_tensor = torch.FloatTensor(action).to(device)
                if next_obs is not None:
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, hidden_state, previous_hidden)

            state = next_state

            if args.episode_length < step or not (False in done):
                # update model
                a_loss, c_loss = model.update(episode)
                print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                if args.tensorboard:
                    writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                    writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())

                    if c_loss and a_loss:
                        writer.add_scalars('agent/loss', global_step=episode,
                                           tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

                if c_loss and a_loss:
                    print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')

                if episode % args.save_interval == 0:
                    model.save_model(episode)
                env.reset()
                break

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="simple_spread", type=str,
                        help="simple_spread/traffic_junction/predator_prey/simple_reference")
    parser.add_argument('--max_episodes', default=100000, type=int)
    parser.add_argument('--n_actions', default=2, type=int)
    parser.add_argument('--n_agents', default=3, type=int)

    parser.add_argument('--algo', default='maddpg', type=str,
                        help="mhop/eva/eva_2/commnet/maddpg/tarmac/sarnet/eva_4/eva_5/eva_6/eva_7/eva_9/i2cfc/eva_10/dgn")

    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--episode_length', default=50, type=int, help="how many timesteps in each episode")
    parser.add_argument('--episode_start_to_train', default=10, type=int)
    parser.add_argument('--memory_length', default=int(3e4), type=int)
    parser.add_argument('--tau', default=0.001, type=float, help="soft update parameter")
    parser.add_argument('--distance', default=1, type=float, help="the communication range of each agent")
    parser.add_argument('--rnn_hidden_dim', default=128, type=int)
    parser.add_argument('--rnn_hidden_size', default=128, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=101, type=int)
    parser.add_argument('--lamba', default=0.3, type=float)
    parser.add_argument('--full_comm', default=True, type=bool)
    parser.add_argument('--msg_hidden_dim', default=500, type=int)
    parser.add_argument('--msg_out_size', default=500, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float, help="learning rate of actor")
    parser.add_argument('--c_lr', default=0.0001, type=float, help="learning rate of critic")
    parser.add_argument('--batch_size', default=200, type=int, help="the number of samples when update the model")
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--attention_layer', default=2, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=10000, type=int, help="how often store the model")
    parser.add_argument("--model_episode", default=150000, type=int)
    parser.add_argument('--episode_before_train', default=10, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--reminder', default='normal', type=str)

    args = parser.parse_args()

    main(args)