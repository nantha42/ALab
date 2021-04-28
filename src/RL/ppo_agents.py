import os
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import Sequential
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self,batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indices = np.arange(n_states,dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
				np.array(self.actions),\
				np.array(self.probs),\
				np.array(self.vals),\
				np.array(self.rewards),\
				np.array(self.dones),\
				batches

    def store_memory(self,state,action,probs,vals,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
		
    def clear_memory(self ):
        self.states =  []
        self.probs =   []
        self.actions = []
        self.rewards = []
        self.dones =   []
        self.vals =    []


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpoint_file = ""
    
    def save_checkpoint(self):
        if self.checkpoint_file != "":
            T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        if self.checkpoint_file != "":
            self.load_state_dict(self.state_dict(),self.checkpoint_file)

class ActorNet(nn.Module):
    def __init__(self,input_dims,n_actions,):
        super().__init__()
        self.actor = Sequential(
            nn.Linear(input_dims,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,n_actions),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(),lr = 0.0003)
    
    def forward(self,x):
        dist = self.actor(x)
        dist = Categorical(dist)
        return dist

class CriticNet(nn.Module):
    def __init__(self,input_dims,n_actions,):
        super().__init__()
        self.critic = Sequential(
            nn.Linear(input_dims,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1),
        )
        self.optimizer = optim.Adam(self.parameters(),lr = 0.0003)
    
    def forward(self,x):
        v = self.critic(x)
        return v 

class ActorGRUNet(Network):
    def __init__(self,input_dims,n_actions,nlayers=2):
        super().__init__()
        input_size = input_dims
        self.num_actions =n_actions 
        self.layers = nlayers
        self.hidden_size = 128 
        self.gru = nn.GRU(input_size,self.hidden_size,self.layers)
        self.hidden = T.zeros(self.layers,1,self.hidden_size)
        self.actor = Sequential(
            nn.Linear(128,n_actions),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(),lr = 0.003)

    def reset(self) :
        self.hidden = T.zeros(self.layers,1,self.hidden_size)

    def forward(self,x):
        x,self.hidden = self.gru(x,self.hidden)
        dist = self.actor(x)
        dist = Categorical(dist)
        return dist

class NormalNet(Network):
    def __init__(self,input_dims,n_actions,):
        super().__init__()
        self.actor = Sequential(
            nn.Linear(input_dims,128),
            nn.ReLU(),
            nn.Linear(128,n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = Sequential(
            nn.Linear(input_dims,128),
            nn.ReLU(),
            nn.Linear(128,1),
        )
        self.optimizer = optim.Adam(self.parameters(),lr = 0.003)
    
    def forward(self,state):
        dist = Categorical(self.actor(state) )
        value = self.critic(state)
        return dist,value
    
class CAgent:
    def __init__(self,actor,critic,batch_size=5):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 5
        self.gae_lambda = 0.95 

        self.actor= actor 
        self.critic = critic
        self.memory = PPOMemory(batch_size)
    
    def remember(self,state,action,probs,vals,reward, done = 0):
        self.memory.store_memory(state,action,probs,vals,reward,done)
    
    def save_model(self):
        self.net.save_checkpoint()

    def load_model(self):
        self.net.load_checkpoint()

    def choose_action(self,state):
        state = T.from_numpy(state).float().unsqueeze(0)
        dist= self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        entropy = -T.sum(T.mean(dist.probs) * T.log(dist.probs))
        return action,probs, value, entropy
    
    def learn(self,entropy = 0):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr),dtype = np.float32)

            for t in range(len(reward_arr)-1):
                a_t = 0
                discount = 1
                for i in range(t,len(reward_arr)-1):
                    a_t = a_t + discount*(reward_arr[i] + self.gamma*values[i+1]*(1-int(dones_arr[i])) -values[i] )
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
	       
            advantage = T.tensor(advantage)
            values = T.tensor(values)
            sum_total_loss = 0
            for batch in batches:
                states = T.tensor(state_arr[batch]).float()
                old_probs = T.tensor(old_probs_arr[batch])
                actions = T.tensor(action_arr[batch])

                dist= self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp()
                weighted_probs = advantage[batch]* prob_ratio 
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+ self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss + 0.001*entropy
            
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                sum_total_loss += total_loss.item()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            # print(sum_total_loss)

        self.memory.clear_memory()

class Agent:
    def __init__(self,network,batch_size=5):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 5
        self.gae_lambda = 0.95 

        self.net = network
        self.memory = PPOMemory(batch_size)
    
    def remember(self,state,action,probs,vals,reward, done = 0):
        self.memory.store_memory(state,action,probs,vals,reward,done)
    
    def save_model(self):
        self.net.save_checkpoint()

    def load_model(self):
        self.net.load_checkpoint()

    def choose_action(self,state):
        state = T.from_numpy(state).float().unsqueeze(0)
        dist,value = self.net(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        entropy = -T.sum(T.mean(dist.probs) * T.log(dist.probs))
        return action,probs, value, entropy
    
    def learn(self,entropy = 0):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr),dtype = np.float32)

            for t in range(len(reward_arr)-1):
                a_t = 0
                discount = 1
                for i in range(t,len(reward_arr)-1):
                    a_t = a_t + discount*(reward_arr[i] + self.gamma*values[i+1]*(1-int(dones_arr[i])) -values[i] )
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
	       
            advantage = T.tensor(advantage)
            values = T.tensor(values)

            for batch in batches:
                states = T.tensor(state_arr[batch]).float()
                old_probs = T.tensor(old_probs_arr[batch])
                actions = T.tensor(action_arr[batch])

                dist,critic_value = self.net(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp()/old_probs.exp()
                weighted_probs = advantage[batch]* prob_ratio 
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+ self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss + 0.001*entropy
            
                self.net.optimizer.zero_grad()
                total_loss.backward()
                self.net.optimizer.step()
        self.memory.clear_memory()
