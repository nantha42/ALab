import torch as T
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import numpy as np
import pygame as py
import time
from .utils import RLGraph


class LSTMTrainer:
    def __init__(self, model,
                 learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.data = []

    def store_records(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [
        ], [], [], [], [], [], [], []
        for T in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = T
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in,
                                              h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())
        gamma = 0.99
        for i in range(5):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma* v_prime * done_mask 
            v_s = self.v(s,first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            
class PPOGRU:
    def __init__(self, model,
                 learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.data = []

    def store_records(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst,  prob_a_lst, h_in_lst,  done_lst = [
        ], [], [], [], [], [], [], []
        for T in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = T
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r,  done_mask, prob_a = T.tensor(s_lst, dtype=T.float), T.tensor(a_lst), \
            T.tensor(r_lst), \
            T.tensor(done_lst, dtype=T.float), T.tensor(prob_a_lst)

        self.data = []
        return s, a, r,  done_mask, prob_a, h_in_lst[0] 
    
    def update(self):
        s, a, r,  done_mask, prob_a, (h1_in) = self.make_batch()
        hidden = h1_in.detach()
        gamma = 0.99
        k_epoch = 5
        for i in range(k_epoch):
            discounted_rewards  = []
            Gt = 0
            for t in reversed(range(len(r))):
                Gt = r[t] + done_mask[t]*gamma*Gt
                discounted_rewards.append(Gt)

            discounted_rewards.reverse()
            discounted_rewards = T.tensor(discounted_rewards,dtype=T.float)

            pi = self.model.forward(s,hidden)
            print(pi.shape)
            pi_a = pi.squeeze(1).gather(1,a)
            print(pi_a.shape)
            ratio = T.exp( T.log(pi_a) - T.log(prob_a) )

            surr1 = ratio * discounted_rewards 
            eps_clip = 0.1
            surr2 = T.clamp(ratio, 1-eps_clip, 1+eps_clip)*discounted_rewards
            loss = -min(surr1, surr2)
            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

