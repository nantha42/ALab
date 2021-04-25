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
        

class ActorCriticGRU(nn.Module):
    def __init__(self,num_inputs,num_actions,hidden_size,learning_rate = 3e-4,nlayers=2):
        super(ActorCriticGRU, self).__init__()
        input_size = num_inputs
        self.num_actions = num_actions
        self.layers = nlayers
        self.hidden_size = hidden_size
        self.hidden = torch.zeros(self.layers,1,self.hidden_size)
        self.gru = nn.GRU(input_size,self.hidden_size,self.layers)

        self.critic_linear1 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)
    
    def reset(self):
        self.hidden = torch.zeros(self.layers,1,self.hidden_size)

    def forward(self,state):
        x,self.hidden = self.gru(state,self.hidden)
        value = F.relu(self.critic_linear1(x))
        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.softmax(policy_dist,dim=-1)
        return value,policy_dist

    # def forward(self,state):
    #     value = F.relu(self.critic_linear1(state))
    #     value = self.critic_linear2(value)

    #     policy_dist = F.relu(self.actor_linear1(state))
    #     policy_dist = F.softmax(self.actor_linear2(policy_dist),dim=1)
    #     return value,policy_dist

    def get_action(self,state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0))
        v, probs = self.forward(state)
        value = v.detach().numpy()[0,0,0]
        probs = probs[:,0,:]
        dist = probs.detach().numpy()

        action = np.random.choice(self.num_actions,p = np.squeeze(dist))
        log_prob = torch.log(probs.squeeze(0))[action]
        entropy = -np.sum(np.mean(dist)*np.log(dist))
        return action, value,log_prob, entropy
        

