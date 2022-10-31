import numpy as np
import torch

def football_observation_wrapper(obs):
    # obs2 = obs[0]['left_team_roles']
    n_agents = len(obs[0]['left_team_roles']) - 1
    agent_id = range(1, n_agents + 1)
    new_obs = []
    for i in agent_id:
        single_obs = []
        single_obs = np.array(single_obs)
        single_agent_obs = obs[i]
        # aa  = single_agent_obs['ball_owned_team']
        single_obs = np.append(single_obs, [abs(1-single_agent_obs['ball'][0]), abs(0-single_agent_obs['ball'][1])])
        single_obs = np.append(single_obs, [single_agent_obs['ball']])
        single_obs = np.append(single_obs, [single_agent_obs['ball_rotation']])
        single_obs = np.append(single_obs, [single_agent_obs['ball_owned_team']])
        single_obs = np.append(single_obs, [single_agent_obs['ball_direction']])
        single_obs = np.append(single_obs, [single_agent_obs['ball_direction']])
        single_obs = np.append(single_obs, [single_agent_obs['right_team_direction']])
        single_obs = np.append(single_obs, [single_agent_obs['right_team']])
        single_obs = np.append(single_obs, [single_agent_obs['right_team_roles']])
        single_obs = np.append(single_obs, [single_agent_obs['left_team_direction'][i]])
        single_obs = np.append(single_obs, [single_agent_obs['left_team'][i]])
        for z in agent_id:
            if z != i:
                single_obs = np.append(single_obs, [single_agent_obs['left_team_direction'][z]])
                single_obs = np.append(single_obs, [single_agent_obs['left_team'][z]])
        # single_obs = np.array(single_obs).flatten()
        new_obs.append(single_obs)
    return new_obs

def football_reward_wrapper(obs, reward):
    prev_reward = reward*10
    dist = obs[0][0] + obs[0][1]
    final_reward = prev_reward - dist
    return final_reward

def tranf_obs(obs, fe):  # 将obs和h_out 编码成state_dict,state_dict_tensor
    # h_in = h_out
    dict_obs = []
    final_obs = []
    ava_actions = []
    # for i in range(n_rollout_threads):
    x = []
    n_agents = len(obs)
    for j in range(n_agents):
        state_dict1, ava_action = fe.encode(obs[j])  # 长度为7的字典
        state_dict_tensor1 = state_to_tensor(state_dict1)
        x.append(state_dict1)
        ava_actions.append(ava_action)
    state_dict_tensor = {}
    for k, v in state_dict_tensor1.items():
        # state_dict_tensor[k] = torch.cat((state_dict_tensor1[k], state_dict_tensor2[k]), 0)
        state_dict_tensor[k] = torch.Tensor([x[s][k] for s in range(len(obs))])
    # state_dict_tensor['hidden'] = h_in  # ((1,1,256),(1,1,256))
    dict_obs.append(state_dict_tensor)
    # for i in range(n_rollout_threads):
    final_obs.append(obs_transform(dict_obs[0], n_agents))
    final_obs = torch.Tensor(final_obs).numpy().squeeze()  # [n_threads, state_shape]
    return final_obs, ava_actions

def state_to_tensor(state_dict):  # state_dict:{'player':(29,),'ball':(18,),'left_team':(10,7),'left_closest':(7,),'right_team':(11,7),'player':(7,)}
    # pdb.set_trace() #debug

    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(
        0)  # 在第0维增加一个维度；[[   state_dict["player"]  ]] #shape(1,1,29)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,18)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,10,7)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(
        0)  # shape(1,1,7)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(
        0)  # shape(1,1,11,7)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(
        0)  # shape(1,1,7)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(
        0)  # shape(1,1,12)  tensor([[[1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.]]])

    state_dict_tensor = {
        "player": player_state,
        "ball": ball_state,
        "left_team": left_team_state,
        "left_closest": left_closest_state,
        "right_team": right_team_state,
        "right_closest": right_closest_state,
        "avail": avail,
        # "hidden" : h_in # ([1,1,256], [1,1,256])
    }
    return state_dict_tensor

def obs_transform(state_dict_tensor, n_agents):
    '''

    :param state_dict_tensor: 7 kind of state dict with tensor for each element
    :return: flattern_obs for multi-agents [num_agent, obs_shape] (3 x 115)
    '''
    flattern_obs_0 = []
    flattern_obs_1 = []
    flattern_obs = [[] for _ in range(n_agents)]
    for i in range(len(flattern_obs)):
        for k, v in enumerate(state_dict_tensor):
            if k != 'hidden':  # hideen这一维度去掉
                flattern_obs[i].insert(0, state_dict_tensor[v][i].reshape([-1]))
                # flattern_obs_1.append(state_dict_tensor[v][1].reshape([-1]))
        flattern_obs[i] = torch.hstack(flattern_obs[i])

    # flattern_obs_0 = torch.hstack(flattern_obs_0)
    # flattern_obs_1 = torch.hstack(flattern_obs_1)
    flattern_obs = torch.stack((flattern_obs), dim=0)

    return flattern_obs.numpy()

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

def reward_from_state(n_state, n_agents, env):
    rew = []

    for state in n_state:

        obs_landmark = np.array(state[4:4+n_agents*2])
        agent_reward = 0
        for i in range(n_agents):

            sub_obs = obs_landmark[i*2: i*2+2]

            dist = np.sqrt(sub_obs[0]**2 + sub_obs[1]**2)


            # if dist < 0.4: agent_reward += 0.3
            if dist < 0.2: agent_reward += 0.5
            if dist < 0.1: agent_reward += 1.

            other_agents_pos = [env.agents[j].state.p_pos for j in range(n_agents) if j!= i]
            to_other_distance = [((other_agents_pos[s][0]-env.agents[i].state.p_pos[0])**2+(other_agents_pos[s][1]-env.agents[i].state.p_pos[1])**2)**0.5 for s in range(len(other_agents_pos))]
            for p in range(len(to_other_distance)):
                if to_other_distance[p] <= 1:
                    agent_reward -= 0.25
        rew.append(agent_reward)
    return rew
