import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants
GAMMA = 0.9

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def attention(q,k,v):
    matmul_qk = q @ k.transpose(-1,-2)
    scaled_attention_logits = matmul_qk/ torch.sqrt(torch.tensor(k.shape[-1],dtype=torch.float32))
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits,dim=-1)
    attention_weights = torch.nn.functional.dropout(attention_weights,0.2)
    output = attention_weights @ v 
    return [output, attention_weights.detach().clone()]

class PolicyAttenNetwork(nn.Module):   
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyAttenNetwork, self).__init__()
        self.hidden_size = hidden_size 
        self.num_actions = num_actions
        self.WQ = nn.Linear(num_inputs,hidden_size)
        self.WK = nn.Linear(num_inputs,hidden_size)
        self.WV = nn.Linear(num_inputs,hidden_size)
        self.cache_size = 5
        self.kmem = None 
        self.vmem = None 
        self.linear1 = nn.Linear(hidden_size,hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2,num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def reset(self):
        self.kmem = None 
        self.vmem = None 
 
    
    def forward(self,state):
        q = self.WQ(state)
        k = self.WK(state)
        v = self.WV(state)
        if self.kmem is None:
            self.kmem = k
            self.vmem = v
        if len(self.kmem) == self.cache_size:
            self.kmem = self.kmem[:-1,:]
            self.vmem = self.vmem[:-1,:]
        dk = k.detach().clone()
        dv = v.detach().clone()
        self.kmem = torch.cat([dk,self.kmem],dim=-2)
        self.vmem = torch.cat([dv,self.vmem],dim=-2)
        out,attn = attention(q,self.kmem,self.vmem)
        out = nn.functional.relu(self.linear1(out))
        out = nn.functional.softmax(self.linear2(out),dim=-1)
        return out

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

   

class PolicyMemoryNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyMemoryNetwork, self).__init__()
        self.hidden_size = hidden_size 
        input_size = num_inputs
        output_size = hidden_size//2
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        self.num_actions = num_actions
        self.linear1 = nn.Linear(output_size, output_size//2)
        self.linear2 = nn.Linear(output_size//2, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def reset(self):
        self.hidden = torch.zeros(1,self.hidden_size)

    def forward(self, state):
        input_combined = torch.cat((state,self.hidden),-1)
        self.hidden = torch.nn.functional.sigmoid(self.i2h(input_combined))
        o = self.i2o(input_combined)
        x = F.relu(self.linear1(o))
        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

