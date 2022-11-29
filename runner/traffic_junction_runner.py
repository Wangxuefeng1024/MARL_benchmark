import sys
sys.path.append('../')
import argparse, datetime
from tensorboardX import SummaryWriter
import torch
import numpy as np

import os
cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

from policy.commnet import TJ_CommNet
from policy.tarmac import TJ_TarMAC

from Envs.tj.traffic_junction_env import TrafficJunctionEnv

from utils.util import _flatten_obs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficJunctionEnv()
    env.multi_agent_init(args)
    n_agents = args.n_agents
    n_actions = 2
    if args.difficulty == 'hard':
        n_states = 77
        args.episode_length = 80
    else:
        n_states = 29
        args.episode_length = 60

    torch.manual_seed(args.seed)
    if args.tensorboard:
        writer = SummaryWriter(
            log_dir='../runs/' + "/traffic_junction/" + args.difficulty + "/" + args.algo + "/" )

    if args.algo == "commnet":
        model = TJ_CommNet(n_states, n_actions, n_agents, args)
    elif args.algo == "tarmac":
        model = TJ_TarMAC(n_states, n_actions, n_agents, args)
    else:
        model = TJ_CommNet(n_states, n_actions, n_agents, args)

    print(model)
    episode_length = 60 if args.difficulty == "medium" else 80
    episode = 0
    total_step = 0
    win_times = 0
    while episode < args.max_episodes:
        state = env.reset()
        actions, log_probs, entropys, rewards, adjs, hidden_datas, pre_features, first_features, second_features = [], [], [], [], [], [], [], [], []
        state = _flatten_obs(state)
        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0

        while True:
            if args.mode == "train":

                action, log_prob, entropy = model.choose_action(state)

                actions.append(action)
                entropys.append(entropy)
                log_probs.append(log_prob)
                next_state, reward, done, info = env.step(action.cpu().detach().numpy())

                step += 1
                total_step += 1
                reward = np.array(reward)
                rewards.append(torch.tensor(reward, device=device).float())
                next_state = _flatten_obs(next_state)
                accum_reward += sum(reward)
                rewardA += reward[0]
                state = next_state

                # if args.batch_size < step or (True in done):
                if episode_length < step:

                    loss = model.update(episode, rewards, log_probs, entropys)
                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    if not (False in done):
                        win_times += 1
                    if args.tensorboard:
                        writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                        writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())

                        if episode % 100 == 0:
                            writer.add_scalar('agent/win_rates', global_step=episode, scalar_value=win_times / 100)

                        if loss:
                            writer.add_scalars('agent/loss', global_step=episode,
                                               tag_scalar_dict={'actor': loss})

                    if loss:
                        print(" a_loss %3.2f" % (loss), end='')

                    if episode % args.save_interval == 0 and args.mode == "train":
                        model.save_model(episode)
                    break

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="traffic_junction", type=str, help="traffic_junction")
    parser.add_argument('--max_episodes', default=200000, type=int)
    parser.add_argument('--n_actions', default=2, type=int)
    parser.add_argument('--n_agents', default=20, type=int)
    parser.add_argument('--state_shape', default=29, type=int)
    parser.add_argument('--algo', default='tarmac', type=str,
                        help="commnet/tarmac")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--difficulty', default="medium", type=str, help="easy/medium/hard")
    parser.add_argument('--episode_length', default=80, type=int)
    parser.add_argument('--episode_start_to_train', default=1, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--rnn_hidden_dim', default=128, type=int)
    parser.add_argument('--rnn_hidden_size', default=128, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=101, type=int)
    parser.add_argument('--lamba', default=0.3, type=float)
    parser.add_argument('--full_comm', default=True, type=bool)
    parser.add_argument('--msg_hidden_dim', default=500, type=int)
    parser.add_argument('--msg_out_size', default=500, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=80, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--attention_layer', default=2, type=int)
    parser.add_argument('--hard', default=True, type=bool)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--threshold', default=0.011, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=10000, type=int)
    parser.add_argument("--model_episode", default=150000, type=int)
    parser.add_argument('--episode_before_train', default=10, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--reminder', default='normal', type=str)

    args = parser.parse_args()
    main(args)
