import sys

sys.path.append('../')
from env.make_env import make_env
from env.predator_prey.make_env import pp_make_env
import argparse, datetime
from tensorboardX import SummaryWriter
import torch
import numpy as np

from algo.bicnet.bicnet_agent import BiCNet
from algo.evaluator.eva_agent import TJ_EVA, compute_adj, transfer_adj
from algo.eva_2.eva2_agent import TJ_EVA_2
from algo.eva_3.eva3_agent import TJ_EVA_3
import os
cpu_num = 2
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
from algo.eva_4.eva4_agent import TJ_EVA_4
from algo.eva_5.eva5_agent import TJ_EVA_5
from algo.eva_6.eva6_agent import TJ_EVA_6
from algo.eva_8.eva8_agent import TJ_EVA_8
from algo.eva_11.eva11_agent import TJ_EVA_11
from algo.eva_15.eva15_agent import TJ_EVA_15
from algo.eva_16.eva16_agent import TJ_EVA_16
from algo.commnet.commnet_agent import TJ_CommNet
from algo.G2A.g2a_agent import TJ_G2A
from algo.DICG.dicg_agent import TJ_DICG
from algo.DGN.dgn_agent import TJ_DGN
from algo.i2c.i2c import I2C, get_comm_pairs
from algo.i2c_fc.i2c_fc_agent import TJ_I2C_FC
from algo.SARNet_2.sarnet_agent_2 import TJ_SARNet_2
from env.tj.traffic_junction_env import TrafficJunctionEnv
from algo.Multi_Hop.mhop_agent import MHOP

from algo.TarMAC_2.tarmac_agent_2 import TJ_TarMAC_2
from algo.utils import _flatten_obs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.scenario == "predator_prey":
        env = pp_make_env(args)
        n_agents = args.n_agents
        n_actions = 5

        n_states = env.observation_space[0].shape[0]
    elif args.scenario == "traffic_junction":
        env = TrafficJunctionEnv()
        env.multi_agent_init(args)
        n_agents = args.n_agents
        n_actions = 2
        if args.difficulty == 'hard':
            n_states = 77
        else:
            n_states = 29
    else:
        env = make_env(args)
        n_agents = args.n_agents
        n_actions = env.world.dim_p

        n_states = env.observation_space[0].shape[0]

    torch.manual_seed(args.seed)

    if args.tensorboard and args.mode == "train":
        writer = SummaryWriter(
            log_dir='runs/' + args.algo + "/" + args.log_dir + str(args.episode_length) + args.scenario + str(
                args.n_agents) + '_' + str(args.attention_layer) + args.reminder)

    if args.algo == "bicnet":
        model = BiCNet(n_states, n_actions, n_agents, args)

    elif args.algo == "mhop":
        model = MHOP(n_states, n_actions, n_agents, args)

    elif args.algo == "eva":
        model = TJ_EVA(n_states, n_actions, n_agents, args)

    elif args.algo == "eva_2":
        model = TJ_EVA_2(n_states, n_actions, n_agents, args)

    elif args.algo == "eva_3":
        model = TJ_EVA_3(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_4":
        model = TJ_EVA_4(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_5":
        model = TJ_EVA_5(n_states, n_actions, n_agents, args)

    elif args.algo == "eva_6":
        model = TJ_EVA_6(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_8":
        model = TJ_EVA_8(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_11":
        model = TJ_EVA_11(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_15":
        model = TJ_EVA_15(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_16":
        model = TJ_EVA_16(n_states, n_actions, n_agents, args)
    elif args.algo == "commnet":
        model = TJ_CommNet(n_states, n_actions, n_agents, args)

    elif args.algo == "gnn":
        model = GNN(n_states, n_actions, n_agents, args)

    elif args.algo == "tarmac":
        model = TJ_TarMAC_2(n_states, n_actions, n_agents, args)
    elif args.algo == "g2a":
        model = TJ_G2A(n_states, n_actions, n_agents, args)

    elif args.algo == "sarnet":
        model = TJ_SARNet_2(n_states, n_actions, n_agents, args)
    elif args.algo == "dicg":
        model = TJ_DICG(n_states, n_actions, n_agents, args)
    elif args.algo == "dgn":
        model = TJ_DGN(n_states, n_actions, n_agents, args)
    elif args.algo == "i2c":
        model = I2C(n_states, n_actions, n_agents, args)
    elif args.algo == "i2cfc":
        model = TJ_I2C_FC(n_states, n_actions, n_agents, args)

    else:
        model = TJ_CommNet(n_states, n_actions, n_agents, args)

    print(model)
    episode_length = 60 if args.difficulty == "medium" else 80
    episode = 0
    total_step = 0
    win_times = 0
    while episode < args.max_episodes:
        state = env.reset()
        adjs = []
        hidden_datas = []
        first_features = []
        pre_features = []
        actions, log_probs, entropys, rewards, adjs, hidden_datas, pre_features, first_features, second_features = [], [], [], [], [], [], [], [], []
        if args.scenario == "traffic_junction":
            state = _flatten_obs(state)
        episode += 1
        step = 0
        accum_reward = 0
        rewardA = 0

        while True:
            if args.mode == "train":
                if args.algo == "eva_2" or args.algo == 'eva_8' :
                    adj_matrix = transfer_adj(env, args.n_agents)
                    action, log_prob, entropy, adj, hidden_data, pre_feature, first_feature = model.choose_action(state, adj_matrix, episode)
                elif args.algo == "eva_15" or args.algo == "eva_16":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    action, log_prob, entropy, adj, hidden_data, pre_feature, first_feature, second_feature = model.choose_action(state, adj_matrix, episode)
                elif args.algo == "mhop":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix, episode)
                elif args.algo == "eva" or args.algo == "eva_3" or args.algo == "eva_4" or args.algo == "eva_5" or args.algo == "eva_6" or args.algo == "eva_11":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix)
                elif args.algo =="dicg" or args.algo == "tarmac" or args.algo == "sarnet" or args.algo =="dgn":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix)
                elif args.algo == "g2a":
                    action, log_prob, entropy = model.choose_action(state)
                elif args.algo == "i2cfc":
                    action, log_prob, entropy = model.choose_action(state)
                else:
                    action, log_prob, entropy = model.choose_action(state)

                # if args.scenario == 'traffic_junction':
                #     action, pi = select_action(action)
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
                if args.algo == 'eva_2' or args.algo == 'eva_8':
                    adjs.append(adj)
                    hidden_datas.append(hidden_data)
                    pre_features.append(pre_feature)
                    first_features.append(first_feature)
                elif args.algo == 'eva_15' or args.algo == 'eva_16':
                    adjs.append(adj)
                    hidden_datas.append(hidden_data)
                    pre_features.append(pre_feature)
                    first_features.append(first_feature)
                    second_features.append(second_feature)
                # if args.batch_size < step or (True in done):
                if episode_length < step:
                    if args.algo == 'eva_2' or args.algo =='eva_8':
                        loss = model.update(episode, rewards, log_probs, entropys, torch.stack(adjs),torch.stack(hidden_datas), torch.stack(pre_features), torch.stack(first_features))
                    elif args.algo == 'eva_15' or args.algo =='eva_16':
                        loss = model.update(episode, rewards, log_probs, entropys, torch.stack(adjs),torch.stack(hidden_datas), torch.stack(pre_features), torch.stack(first_features), torch.stack(second_features))
                    else:
                        loss = model.update(episode, rewards, log_probs, entropys)
                    print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                    if not (False in done):
                        win_times += 1
                    if args.tensorboard:
                        writer.add_scalar(tag='agent/reward', global_step=episode, scalar_value=accum_reward.item())
                        writer.add_scalar(tag='agent/reward_0', global_step=episode, scalar_value=rewardA.item())

                        if args.scenario == "traffic_junction" and episode % 100 == 0:
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


def tj_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.scenario == "predator_prey":
        env = pp_make_env(args)
        n_agents = args.n_agents
        n_actions = 5

        n_states = env.observation_space[0].shape[0]
    elif args.scenario == "traffic_junction":
        env = TrafficJunctionEnv()
        env.multi_agent_init(args)
        n_agents = args.n_agents
        n_actions = 2
        n_states = 29
    else:
        env = make_env(args)
        n_agents = args.n_agents
        n_actions = env.world.dim_p

        n_states = env.observation_space[0].shape[0]

    torch.manual_seed(args.seed)

    if args.tensorboard:
        writer = SummaryWriter(log_dir='runs/' + args.algo + "/" + 'tj_test_seed_' + str(args.seed) + '_' + str(
            args.n_agents) + args.reminder)

    if args.algo == "bicnet":
        model = BiCNet(n_states, n_actions, n_agents, args)

    elif args.algo == "mhop":
        model = MHOP(n_states, n_actions, n_agents, args)

    elif args.algo == "eva":
        model = TJ_EVA(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_2":
        model = TJ_EVA_2(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_3":
        model = TJ_EVA_3(n_states, n_actions, n_agents, args)
    elif args.algo == "eva_8":
        model = TJ_EVA_8(n_states, n_actions, n_agents, args)

    elif args.algo == "commnet":
        model = TJ_CommNet(n_states, n_actions, n_agents, args)

    elif args.algo == "gnn":
        model = GNN(n_states, n_actions, n_agents, args)
    elif args.algo == "dgn":
        model = TJ_DGN(n_states, n_actions, n_agents, args)
    elif args.algo == "maddpg":
        model = MADDPG(n_states, n_actions, n_agents, args)

    elif args.algo == "tarmac":
        model = TJ_TarMAC(n_states, n_actions, n_agents, args)

    elif args.algo == "sarnet":
        model = TJ_SARNet(n_states, n_actions, n_agents, args)

    elif args.algo == "i2c":
        model = I2C(n_states, n_actions, n_agents, args)

    elif args.algo == "i2cfc":
        model = TJ_I2C_FC(n_states, n_actions, n_agents, args)

    elif args.algo == "full_com":
        model = FULL_COMM(n_states, n_actions, n_agents, args)

    else:
        model = MADDPG(n_states, n_actions, n_agents, args)

    print(model)

    episode = 0
    model.load_model()
    success = 0
    with torch.no_grad():
        for i in range(args.max_episodes):
            state = env.reset()
            collision = 0
            step = 0

            while step <= args.episode_length:

                if args.algo == "eva_2" or args.algo == "eva_8":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    state = _flatten_obs(state)
                    action, log_prob, entropy, adj, hidden_data, pre_feature, first_feature = model.choose_action(state,
                                                                                                                  adj_matrix,
                                                                                                                  episode)
                elif args.algo == "mhop":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    state = _flatten_obs(state)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix, episode)
                elif args.algo == "eva" or args.algo == "eva_3":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    state = _flatten_obs(state)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix)
                elif args.algo == "tarmac" or args.algo == "sarnet":
                    adj_matrix = transfer_adj(env, args.n_agents)
                    state = _flatten_obs(state)
                    action, log_prob, entropy = model.choose_action(state, adj_matrix)
                elif args.algo == "i2cfc":
                    state = _flatten_obs(state)
                    action, log_prob, entropy = model.choose_action(state)
                else:
                    state = _flatten_obs(state)
                    action, log_prob, entropy = model.choose_action(state)

                next_state, reward, done, info = env.step(action.cpu().detach().numpy())
                env.render()
                step += 1

                state = next_state
                if env.has_failed == 1:
                    collision += 1

            if collision == 0:
                success += 1
            if i % 100 == 0:
                print('success time is ', success)
            writer.add_scalar(tag='agent/win_rate', global_step=i, scalar_value=success)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="traffic_junction", type=str, help="traffic_junction")
    parser.add_argument('--max_episodes', default=200000, type=int)
    parser.add_argument('--n_actions', default=2, type=int)
    parser.add_argument('--n_agents', default=20, type=int)
    parser.add_argument('--state_shape', default=29, type=int)
    parser.add_argument('--algo', default='eva_8', type=str,
                        help="mhop/eva/commnet/bicnet/maddpg/tarmac/sarnet/i2c/i2cfc")
    parser.add_argument('--mode', default="train", type=str, help="train/eval")
    parser.add_argument('--difficulty', default="hard", type=str, help="easy/medium/hard")
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
