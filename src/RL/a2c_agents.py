import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self,num_inputs,num_actions,hidden_size,learning_rate = 3e-4):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)

    def forward(self,state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist),dim=1)
        return value,policy_dist

    def get_action(self,state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        v, probs = self.forward(state)
        value = v.detach().numpy()[0,0]
        dist = probs.detach().numpy()

        action = np.random.choice(self.num_actions,p = np.squeeze(dist))
        log_prob = torch.log(probs.squeeze(0))[action]
        entropy = -np.sum(np.mean(dist)*np.log(dist))
        return action, value,log_prob, entropy
        



# def update_policy(policy_network,rewards,values,log_probs,E,new_state):
#     Qval,_ = net.forward(new_state)
#     Qval = Qval.detach().numpy()[0,0]
 
#     Qvals = np.zeros_like(values)
#     for t in reversed(range(len(rewards))):
#         Qval = rewards[t] + GAMMA * Qval
#         Qvals[t] = Qval
    
#     values = torch.FloatTensor(values)
#     Qvals = torch.FloatTensor(Qvals)
#     log_probs = torch.stack(log_probs)

#     advangate = Qvals - values
#     actor_loss = (-log_probs * advantage).mean()
#     critic_loss = 0.5*advantage.pow(2).mean()
#     ac_loss = actor_loss + critic_loss + 0.001* E
#     policy_network.optimizer.zero_grad()
#     ac_loss.backward()
#     policy_network.optimizer.step()


# if a:

#         log_probs = []
#         rewards = []
#         values = []
#         state = game.get_state().reshape(-1)

#         pbar = tqdm(range(steps),bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}')
#         trewards  = 0
#         entropy_term = 0
#         for j in pbar:
#             action,log_prob,enropy = net.get_action(state)
#             acec = np.zeros((nactions));avec[action] = 1
#             new_state,reward,done,_ = game.act(avec)
#             rewards.append(reward)
#             values.append(value)
#             log_probs.append(log_prob)
#             entropy_term += entropy
#             state = new_state

#    update_policy(net,rewards,values,log_probs,entropy_term,new_state)

