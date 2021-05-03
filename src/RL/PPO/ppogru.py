#PPO-LSTM
from utils import *
import gym
from icecream import ic
from tqdm import tqdm
from Env import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

class Config:
    def __init__(self,model_name):
        self.MODEL_NAME = model_name + ".pth"
        self.TYPE = "Default" 
        self.NACTIONS = 6 
        self.PLOT_FILENAME = model_name + ".png"
        self.HIST_FILENAME = model_name + ".pkl" 
        self.NLAYERS = 4 
        self.HIDDEN_SIZE = 64 
        self.VSIZE = 5
        self.GSIZE = (14,14)
        self.LOADMODEL = False



#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 5
T_horizon     = 20


class PPO(nn.Module):
    def __init__(self,input_size,output_size):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(input_size,64)
        self.gru   = nn.GRU(64,32)
        self.fc_pi = nn.Linear(32,output_size)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, gru_hidden = self.gru(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob,gru_hidden 
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, gru_hidden = self.gru(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, h1_in, h1_out = self.make_batch()
        first_hidden  = h1_in.detach()
        second_hidden = h1_out.detach()
        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            aloss = -torch.min(surr1,surr2) 
            closs = F.smooth_l1_loss(v_s,td_target.detach())
            loss = aloss + closs

            # print("KL divergence",ratio.mean().item(),f"aloss: {aloss.mean()} closs: {closs.mean()}")

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


def train(config):
    gsize = config.GSIZE
    vsize = config.VSIZE
    nactions = config.NACTIONS
    model_name = config.MODEL_NAME
    type = config.TYPE 
    nlayers = config.NLAYERS
    load_model = config.LOADMODEL
    HIDDEN_SIZE = config.HIDDEN_SIZE
    HIST_FILENAME = config.HIST_FILENAME
    PLOT_FILENAME = config.PLOT_FILENAME

    hs = 32
    kr,kc = gsize
    game = PowerGame(kr,kc,vsize)
    game.enable_draw = False
    entropy_term = 0
    steps = STEPS 
    batch_size = 50
    episodes = EPISODES 
    recorder = RLGraph()
 
    agent = PPO(vsize*vsize,nactions)
    score = 0.0
    print_interval = 20
 
    for n_epi in range(episodes):
        game.reset()
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float))
        s = game.get_state().reshape(-1)
        done = False
        game.enable_draw = True if n_epi%5 == 0 else False
        score = 0 
        pbar = tqdm(range(10),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for _ in pbar:
            for t in range(STEPS//10):
                h_in = h_out
                c_time = time.time()
                prob, h_out = agent.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                avec = np.zeros((nactions))
                avec[a] = 1
                s_prime,r = game.act(avec)
                s_prime = s_prime.reshape(-1)
                done = 0

                agent.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime
                score += r
                game.step()
                if done:
                    break
                    
            recorder.newdata(score)
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
            pbar.set_description(f"Episodes: {n_epi:4} Rewards: {score:2}")
            agent.train_net()

    env.close()

if __name__ == '__main__':
    EPISODES = 5000
    STEPS = 500

    c = Config("MemAgent-S5")
    c.HIDDEN_SIZE =  64 
    c.TYPE = "Memory" 
    c.VSIZE = 5
    c.NACTIONS = 6
    c.NLAYERS = 2
    c.GSIZE= (14,14)
    c.LOADMODEL = False 

    train(c)