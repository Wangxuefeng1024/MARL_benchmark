import os, sys
import gym
import torch
import datetime
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from policy.ddpg import MADDPG, Cen_DDPG

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.task)
    args.n_states = env.observation_space.shape or env.observation_space.n
    args.n_actions = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    print("Observations shape:", args.n_states)
    print("Actions shape:", args.n_actions)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    n_agents = args.n_agents
    n_actions = args.n_actions[0]
    n_states = args.n_states[0]
    args.n_actions = args.n_actions[0]
    args.n_states = args.n_states[0]

    writer = SummaryWriter(log_dir='../runs/'+ "mujoco/" + args.task + "/" + args.algo)

    # set algorithm
    if args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)
    else:
        model = Cen_DDPG(n_states, n_actions, n_agents, args)

    print(model)
    # set seed
    torch.manual_seed(args.seed)

    episode = 0
    total_step = 0
    win_times = 0
    while episode < args.max_episodes:
        state = env.reset()
        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0
        while True:
            if args.algo == "tarmac":
                action, hidden_state, previous_hidden = model.choose_action(state, noisy=True)
            else:
                action = model.choose_action(state, noisy=True)

            next_state, reward, done, info = env.step(action)

            step += 1
            total_step += 1
            reward = np.array(reward)

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
                if args.scenario == 'traffic_junction' and not (
                        torch.equal(obs, torch.zeros_like(obs)) and torch.equal(obs_, torch.zeros_like(obs))):
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, hidden_state, previous_hidden)
            state = next_state

            if args.episode_length < step or not (False in done):
                c_loss, a_loss = model.update(episode)

                print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                if not (False in done):
                    win_times += 1
                if args.tensorboard:
                    writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                    writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())

                    if args.scenario == "traffic_junction" and episode % 100 == 0:
                        writer.add_scalar('agent/win_rates', global_step=episode, scalar_value=win_times / 100)

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
    parser.add_argument('--task', type=str, default='Swimmer-v2')
    parser.add_argument('--algo', type=str, default='maddpg')
    parser.add_argument('--save_dir', type=str, default='mlp')
    # parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--num_agents_list', type=list, default=[9, 8])
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--memory_length', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--a_lr', type=float, default=3e-4)
    parser.add_argument('--c_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--update-per-step', type=int, default=0.025)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--episode_before_train', type=int, default=0)
    # parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    args = parser.parse_args()

    main(args)
