import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
	def __init__(self,n_agent, dim_observation, dim_action):
		super(Critic,self).__init__()
		self.n_agent = n_agent
		self.dim_observation = dim_observation
		self.dim_action = dim_action
		obs_dim = self.dim_observation * n_agent
		act_dim = self.dim_action * n_agent
		
		self.FC1 = nn.Linear(obs_dim,256)
		self.FC2 = nn.Linear(256+act_dim,64)
		self.FC3 = nn.Linear(64,1)
		
	# obs:batch_size * obs_dim
	def forward(self, obs, acts):
		result = F.relu(self.FC1(obs))
		combined = torch.cat([result, acts], dim=1)
		result = F.relu(self.FC2(combined))
		return self.FC3(result)


		
# class Actor(nn.Module):
# 	def __init__(self,dim_observation,dim_action):
# 		#print('model.dim_action',dim_action)
# 		super(Actor,self).__init__()
# 		self.FC1 = nn.Linear(dim_observation,500)
# 		self.FC2 = nn.Linear(500,128)
# 		self.FC3 = nn.Linear(128,dim_action)
#
#
# 	def forward(self,obs):
# 		result = F.relu(self.FC1(obs))
# 		result = F.relu(self.FC2(result))
# 		result = F.tanh(self.FC3(result))
# 		return result

class Actor(nn.Module):
	def __init__(self, dim_observation, dim_action, args):
		# print('model.dim_action',dim_action)
		super(Actor, self).__init__()
		self.args = args
		self.FC1 = nn.Linear(dim_observation, 64)
		self.FC2 = nn.Linear(64, 64)
		self.FC3 = nn.Linear(64, dim_action)

	def forward(self, obs):
		result = F.relu(self.FC1(obs))
		result = F.relu(self.FC2(result))
		if self.args.scenario in ["simple_spread", "simple_reference"]:
			result = F.tanh(self.FC3(result)).squeeze()
		elif self.args.scenario == "predator_prey":
			result = F.softmax(self.FC3(result), dim=-1).squeeze()
		else:
			result = F.softmax(self.FC3(result), dim=-1).squeeze()
		return result

	def act(self, state):
		if self.continuous:
			action_mean = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(action_mean, cov_mat)
		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)

		action = dist.sample()
		action_logprob = dist.log_prob(action)

		return action.detach(), action_logprob.detach()

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

