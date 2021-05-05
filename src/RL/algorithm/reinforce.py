from agents import *
import torch as T
from tqdm import tqdm
import time
from Env import *
from utils import *


class Trainer:
    def __init__(self,model
                learning_rate = 0.001,):
        self.model = model 
        self.optimizer = T.optim.Adam(self.model.parameters(),lr=learning_rate) 
        self.rewards = []
        self.log_probs = []
    
    def store_records(self,reward,log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def clear_memory(self):
        self.rewards = []
        self.log_probs = []
    
    def update(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(self.rewards)):
            Gt = 0 
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()