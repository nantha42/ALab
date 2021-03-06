import os
from icecream import ic
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb


lr = 0.0003
# wandb.init(project='ppotesting', entity='rnanthak42')
# config = wandb.config
# config.learning_rate = 0.0003



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

class ActorNet(Network):
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

class CriticNet(Network):
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
    def __init__(self,actor,critic,batch_size=5,model_name = None):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 5 
        self.gae_lambda = 0.95 
        self.actor= actor 
        self.critic = critic
        self.memory = PPOMemory(batch_size)
        if model_name != None:
            self.actor.checkpoint_file = "../../../models/PPO/Actor" + model_name
            self.critic.checkpoint_file = "../../../models/PPO/Critic" + model_name
    
    def remember(self,state,action,probs,vals,reward, done = 0):
        self.memory.store_memory(state,action,probs,vals,reward,done)
    
    def save_model(self):
        # self.net.save_checkpoint()
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


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

                # wandb.log({"surr1":weighted_probs.mean(),
                #         "surr2":weighted_clipped_probs.mean(),
                #         "KL divergence":prob_ratio.mean(),
                #         "aloss": actor_loss.mean(),
                #         "closs": critic_loss.mean(),
                #         "total_loss":total_loss.mean()
                # })
 
            
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                sum_total_loss += total_loss.item()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            # print(sum_total_loss)

        self.memory.clear_memory()

class PPORMemory:
    def __init__(self,batch_size):
        self.states = []
        self.states_prime = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.hidden_in= []
        self.hidden_out= []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0,n_states,self.batch_size)
        indices = np.arange(n_states,dtype = np.int64)
        # np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.states_prime),\
				np.array(self.actions),\
				np.array(self.probs),\
				np.array(self.vals),\
				np.array(self.rewards),\
                np.array(self.hidden_in),\
                np.array(self.hidden_out),\
				np.array(self.dones),\
				batches

    def store_memory(self,state,state_prime,action,probs,vals,reward,hidden_in,hidden_out,done):
        self.states.append(state)
        self.states_prime.append(state_prime)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.hidden_in.append(hidden_in)
        self.hidden_out.append(hidden_out)
        self.dones.append(done)
		
    def clear_memory(self ):
        self.states =  []
        self.states_prime = []
        self.probs =   []
        self.actions = []
        self.rewards = []
        self.dones =   []
        self.hidden_in = []
        self.hidden_out = []
        self.vals =    []


class ActorGRUNet(Network):
    def __init__(self,input_dims,n_actions,nlayers=2):
        super().__init__()
        input_size = input_dims
        self.num_actions =n_actions 
        self.layers = nlayers
        self.hidden_size = 16 
        self.gru = nn.GRU(input_size,self.hidden_size,self.layers)
        # self.hidden = T.zeros(self.layers,1,self.hidden_size)
        self.actor = Sequential(
            nn.Linear(16,n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(),lr = lr)

    def forward(self,x,hidden):
        grout,h = self.gru(x,hidden)
        grout = grout.squeeze(0)
        dist = self.actor(grout)
        dist = Categorical(dist)
        return dist,h, grout

class CriticGRUNet(Network):
    def __init__(self,input_dims,n_actions,nlayers=2):
        super().__init__()
        input_size = input_dims
        self.num_actions =n_actions 
        self.layers = nlayers
        self.hidden_size = 16 
        # self.hidden = T.zeros(self.layers,1,self.hidden_size)
        self.critic = Sequential(
            nn.Linear(self.hidden_size,1),
        )
        self.optimizer = optim.Adam(self.parameters(),lr =lr) 

    def forward(self,x):
        v = self.critic(x)
        return v


class MemoryAgent:
    def __init__(self,actor,critic,batch_size=5,model_name = None):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = 3 
        self.gae_lambda = 0.95 
        
        self.actor= actor 
        self.critic = critic
        
        self.actor_hidden = T.zeros(self.actor.layers,1,self.actor.hidden_size)

        self.memory = PPORMemory(batch_size)

        if model_name != None:
            self.actor.checkpoint_file = "../../../models/PPO/MActor" + model_name
            self.critic.checkpoint_file = "../../../models/PPO/MCritic" + model_name
    
    def reset(self):
        # self.actor_hidden = T.zeros(self.actor.layers,1,self.actor.hidden_size)
        pass

    def remember(self,state,new_state,action,probs,vals,reward, hidden_in,hidden_out,done = 0):
        self.memory.store_memory(state,new_state,action,probs,vals,reward,hidden_in,hidden_out,done)
    
    def save_model(self):
        # self.net.save_checkpoint()
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


    def choose_action(self,state,hidden_in):
        state = T.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
        dist,hidden_out,grout= self.actor(state,hidden_in)
        value = self.critic(grout)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        entropy = -T.sum(T.mean(dist.probs) * T.log(dist.probs))
        return action,probs, value, entropy, hidden_out,

    def get_value(self,state,hidden):
        state = state.unsqueeze(1)
        dist, _,grout = self.actor(state,hidden)
        value = self.critic(grout)
        return value
    
    def get_prob(self,state,hidden):
        state = state.unsqueeze(1)
        dist, _, _ = self.actor(state,hidden)
        return dist

    def learn(self,entropy =0 ):
        state_arr, state_prime_arr,action_arr, old_probs_arr, vals_arr, \
                reward_arr, hidden_in,hidden_out, dones_arr, batches = \
                    self.memory.generate_batches()
                
        state_arr       =  T.from_numpy(state_arr).float() 
        state_prime_arr =  T.from_numpy(state_prime_arr).float() 
        action_arr      =  T.from_numpy(action_arr).float() 
        old_probs_arr   =  T.from_numpy(old_probs_arr).float() 
        vals_arr        =  T.from_numpy(vals_arr).float() 
        reward_arr      =  T.from_numpy(reward_arr).float().unsqueeze(1)
        dones_arr       =  T.from_numpy(dones_arr).float().unsqueeze(1)
        
        first_hidden = hidden_in[0].detach() 
        second_hidden = hidden_out[0].detach()
        for _ in range(self.n_epochs) :
            v_prime = self.get_value(state_prime_arr,second_hidden).squeeze(1)
            td_target = reward_arr + self.gamma * v_prime*dones_arr
            # ic(v_prime.shape)
            # ic(td_target.shape)
            v_s = self.get_value(state_arr,first_hidden).squeeze(1)
            # ic(v_s.shape)
            delta = td_target - v_s
            # ic(delta.shape)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = self.gamma * self.gae_lambda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = T.tensor(advantage_lst,dtype = T.float)
            # print("Advantage",advantage.shape)
            pi = self.get_prob(state_arr,first_hidden).probs
            
            # ic(action_arr.shape)
            # ic(pi.shape)
            pi_a = pi.squeeze(1).gather(1,action_arr.unsqueeze(1).long())
            # ic(pi_a.shape)
            # ic(old_probs_arr.shape)
            ratio = T.exp(T.log(pi_a) - old_probs_arr.unsqueeze(1))
            # ic(ratio.shape)
            surr1 = ratio*advantage
            # ic(advantage.shape)
            # print(advantage.shape,ratio.shape,ratio.mean(),advantage.mean())
            surr2 = T.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage
            # ic(surr1.shape,surr2.shape)
            # exit()
            aloss = -T.min(surr1,surr2) 
            closs = F.smooth_l1_loss(v_s,td_target.detach())
            loss = aloss + closs 
            # wandb.log({"surr1":surr1.mean(),
            #             "surr2":surr2.mean(),
            #             "KL divergence":ratio.mean(),
            #             "total_loss":loss.mean(),
            #             "aloss": aloss.mean(),
            #             "closs": closs.mean(),
            #             "entropy":entropy
            # })
            # # print(f"surr1 {surr1.mean()} surr2 {surr2.mean()} ","KL divergence",ratio.mean().item(),f"aloss: {aloss.mean()} closs: {closs.mean()}")
            # exit()
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            loss.mean().backward(retain_graph = True)
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        self.memory.clear_memory()

# v_prime.shape: torch.Size([50, 1])
# ic| td_target.shape: torch.Size([50, 1])
# ic| v_s.shape: torch.Size([50, 1])
# ic| delta.shape: torch.Size([50, 1])
# Advantage torch.Size([50])
# ic| pi.shape: torch.Size([50, 6])
# ic| pi_a.shape: torch.Size([1, 50])
# 