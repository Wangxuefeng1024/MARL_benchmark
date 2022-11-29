import os, sys
import gym
import torch
import datetime
import argparse
import numpy as np
from multiagent_mujoco.mujoco_multi import MujocoMulti

from torch.utils.tensorboard import SummaryWriter
from policy.ddpg import MADDPG, Cen_DDPG

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_args = {"scenario": args.task,
                  "agent_conf": "2x3",
                  "agent_obsk": 0,
                  "episode_limit": args.episode_length}
    env = MujocoMulti(env_args=env_args)

    args.n_states = len(env.get_obs()[0])
    args.n_actions = env.action_space[0].shape[0]
    # args.max_action = env.action_space.high[0]
    # args.exploration_noise = args.exploration_noise * args.max_action
    # print("Observations shape:", args.n_states)
    # print("Actions shape:", args.n_actions)
    # print("Action range:", np.min(env.action_space.low),
    #       np.max(env.action_space.high))
    n_agents = args.n_agents
    # n_actions = args.n_actions[0]
    # n_states = args.n_states[0]
    # args.n_actions = args.n_actions[0]
    # args.n_states = args.n_states[0]

    writer = SummaryWriter(log_dir='../runs/'+ "mujoco/" + args.task + "/" + args.algo)

    # set algorithm
    if args.algo == "maddpg":
        model = MADDPG(args.n_states, args.n_actions, n_agents, args)
    else:
        model = Cen_DDPG(args.n_states, args.n_actions, n_agents, args)

    print(model)
    # set seed
    torch.manual_seed(args.seed)

    episode = 0
    total_step = 0
    win_times = 0
    while episode < args.max_episodes:
        env.reset()
        terminated = False
        episode_reward = 0
        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0
        while not terminated and step <= args.episode_length:
            obs = env.get_obs()
            state = env.get_state()
            actions_inds = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                actions_inds.append(avail_actions_ind)
            if args.algo == "tarmac":
                action, hidden_state, previous_hidden = model.choose_action(obs, actions_inds)
            else:
                action = model.choose_action(obs, actions_inds)
                # actions.append(action)
            reward, terminated, _ = env.step(action)
            episode_reward += reward
            env.render()
            step += 1
            total_step += 1

            accum_reward += reward
            rewardA += reward

            if args.algo == "maddpg" or args.algo == "commnet":
                obs = torch.from_numpy(np.stack(obs)).float().to(device)
                obs_ = torch.from_numpy(np.stack(env.get_obs())).float().to(device)
                if step != args.episode_length - 1:
                    next_obs = obs_
                else:
                    next_obs = None
                rw_tensor = torch.tensor(reward).float().to(device)
                ac_tensor = torch.FloatTensor(action).to(device)
                if args.algo == "commnet" and next_obs is not None:
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
                if args.algo == "maddpg":
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor)
            elif args.algo == "tarmac":
                obs = torch.from_numpy(np.stack(state)).float().to(device)
                obs_ = torch.from_numpy(np.stack(env.get_obs())).float().to(device)
                if step != args.episode_length - 1 or args.scenario == 'traffic_junction':
                    next_obs = obs_
                else:
                    next_obs = None
                rw_tensor = torch.FloatTensor(reward).to(device)
                ac_tensor = torch.FloatTensor(action).to(device)
                if args.scenario == 'traffic_junction' and not (
                        torch.equal(obs, torch.zeros_like(obs)) and torch.equal(obs_, torch.zeros_like(obs))):
                    model.memory.push(obs.data, ac_tensor, next_obs, rw_tensor, hidden_state, previous_hidden)


        c_loss, a_loss = model.update(episode)

        print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
        # if args.tensorboard:
        writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
        writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())

        if c_loss and a_loss:
            writer.add_scalars('agent/loss', global_step=episode,
                               tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

        if c_loss and a_loss:
            print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')

        if episode % args.save_interval == 0:
            model.save_model(episode)

    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='mujoco')
    parser.add_argument('--task', type=str, default="HalfCheetah-v2")
    parser.add_argument('--algo', type=str, default='maddpg')
    parser.add_argument('--save_dir', type=str, default='mlp')
    parser.add_argument('--episode_length', type=int, default=300)
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
    parser.add_argument('--max_episodes', type=int, default=50000)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--update-per-step', type=int, default=0.025)
    parser.add_argument('--n-step', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=600)
    # parser.add_argument('--episode_before_train', type=int, default=1000)
    parser.add_argument('--model_dir', type=str, default='./trained_model/mujoco/')
    parser.add_argument('--episode_before_train', type=int, default=2)
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
