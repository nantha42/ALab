import torch as T
import torch.nn as nn
import pygame as py
import typing
import numpy as np
from torchsummary import summary
T.random.manual_seed(5)
np.random.seed(5)

# from algorithm.ppo import TrainerGRU, TrainerNOGRU,Simulator
# from algorithm.ppo import TrainerGRU, TrainerNOGRU_V,SimulatorV
# from environment.gatherer import Gatherer
from environment.gatherer import GathererState
from environment.collector import PowerGame 

from algorithm.reinforce import MultiEnvironmentSimulator
from algorithm.reinforce import Trainer, MultiAgentRunner, MultiAgentSimulator, Simulator
# from algorithm.reinforce import Trainer,  Simulator

class StateRAgent(nn.Module):
    def __init__(self, input_size,state_size,output_size=6,containers=1):
        super().__init__()
        self.input_size = input_size
        self.state_size= state_size 
        self.output_size = output_size
        self.pre = nn.Linear(input_size, 64)
        self.hidden = T.zeros((1,containers ,64)) #seq length, Batch, hiddensize
        self.hidden_states = []
        self.ncontainers = containers

        self.embedder = nn.Sequential(
            nn.Linear(state_size,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU()
        )
        #inputsize, hiddensize, num_layers
        self.gru = nn.GRU(64+5, 64, 1) # hidden size + embedding dimension
        self.layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Softmax(dim=-1)
        )
        self.type = "mem"
        self.hidden_vectors = None 
        self.activations = []

        def hook_fn(m,i,o):
            if type(o) == type((1,)):
                for u in o:
                    if len(self.activations) != u.shape[1]:
                        self.activations = [[] for _ in range(u.shape[1])]
                    for aa in range(u.shape[1]):
                        self.activations[aa].append(u[:,aa,:].reshape(-1))
            else:
                if len(self.activations) != o.shape[1]:
                    self.activations = [[] for _ in range(o.shape[1])]
                for aa in range(o.shape[1]):
                    self.activations[aa].append(o[:,aa,:].reshape(-1))

        for n,l in self._modules.items():
            l.register_forward_hook(hook_fn)

    def reset(self):
        self.hidden = T.zeros((1,self.ncontainers,  64)) #seq length, Batch, hiddensize
        self.activations = []
        self.hidden_states = [self.hidden]

    def forward(self, x,states):
        #x shape : 
        #states shape: -1,1,self.state_size
        self.activations = [] # clearing activations because 
        #                       heach step different environment uses the model

        x = x.reshape(-1,self.ncontainers, self.input_size)
        states = states.reshape(-1,self.ncontainers,self.state_size)
        x = self.pre(x)
        emb = self.embedder(states)
        x = T.cat((x,emb),dim=-1)
        x, self.hidden = self.gru(x, self.hidden)
        self.hidden_states.append(self.hidden)
        self.hidden_vectors = self.hidden.detach().clone().squeeze(0).numpy()
        o = self.layers(x)
        return o


class StateAgent(nn.Module):
    def __init__(self, input_size,state_size,output_size=6,containers=1):
        super().__init__()
        self.input_size = input_size
        self.state_size= state_size 
        self.output_size = output_size
        self.pre = nn.Linear(input_size, 64)
        self.ncontainers = containers

        self.embedder = nn.Sequential(
            nn.Linear(state_size,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU()
        )
        #inputsize, hiddensize, num_layers
        self.layers = nn.Sequential(
            nn.Linear(64+5, 32),
            nn.ReLU(),
            nn.Linear(32,output_size),
            nn.Softmax(dim=-1)
        )
        self.type = "reg"
        self.hidden_vectors = None 
        self.activations = []

        def hook_fn(m,i,o):
            if type(o) == type((1,)):
                for u in o:
                    if len(self.activations) != u.shape[1]:
                        self.activations = [[] for _ in range(u.shape[1])]
                    for aa in range(u.shape[1]):
                        self.activations[aa].append(u[:,aa,:].reshape(-1))
            else:
                if len(self.activations) != o.shape[1]:
                    self.activations = [[] for _ in range(o.shape[1])]
                for aa in range(o.shape[1]):
                    self.activations[aa].append(o[:,aa,:].reshape(-1))




        for n,l in self._modules.items():
            l.register_forward_hook(hook_fn)


    def reset(self):
        self.activations = []

    def forward(self, x,states):
        #x shape : 
        #states shape: -1,1,self.state_size
        self.activations = [] # clearing activations because 
        #                       heach step different environment uses the model

        x = x.reshape(-1,self.ncontainers, self.input_size)
        states = states.reshape(-1,self.ncontainers,self.state_size)
        x = self.pre(x)
        emb = self.embedder(states)
        x = T.cat((x,emb),dim=-1)
        o = self.layers(x)
        return o


class RAgent(nn.Module):
    def __init__(self, input_size,output_size=6):
        super().__init__()
        self.input_size = input_size
        self.pre = nn.Linear(input_size, 64)
        self.gru = nn.GRU(64, 64, 1)
        self.hidden = T.zeros((1, 1, 64))
        self.layers = nn.Sequential(
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
        self.type = "mem"
        self.hidden_vectors = None 
        self.activations = []

        def hook_fn(m,i,o):
            if type(o) == type((1,)):
                for u in o:
                    self.activations.append(u.reshape(-1))
            else:
                self.activations.append(o.reshape(-1))

        for n,l in self._modules.items():
            l.register_forward_hook(hook_fn)

    def reset(self):
        self.hidden = T.zeros((1, 1, 64))
        self.activations = []
        self.hidden_states = [self.hidden]

    def forward(self, x):
        self.activations = []
        x = x.reshape(-1, 1, self.input_size)
        x = self.pre(x)
        x, self.hidden = self.gru(x, self.hidden)
        self.hidden_states.append(self.hidden)
        self.hidden_vectors = self.hidden.detach().clone().squeeze(0).numpy()
        o = self.layers(x)
        return o

class AgentCA(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.type = "reg"
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1)
        )
        self.vlayers = nn.Se;quential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64,1 ),
        )


    def forward(self, x):
        o = self.layers(x)
        v = self.vlayers(x)
        return o,v
    
class Agent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.type = "reg"
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        o = self.layers(x)
        return o


if __name__ == '__main__':
    # MULTI AGENTS TESTING
    # nagents =  1 
    # env = StateGatherer(gr = 20,gc = 20,vis = 5,nagents=nagents)
    # models = [StateRAgent(input_size=100) for i in range(nagents)]
    # trainers = [Trainer(m,learning_rate=0.001) for m in models ] 

    # s = MultiAgentSimulator(
    #     models,env,trainers,nactions=6,
    #     log_message="single agent cloned later",
    #     visual_activations = True 
    # )
    # train = 0 
    # s.run(1000,1000,train=train,render_once=1,saveonce=2)

    #MULTI ENVIRONMENT TESTING
    boxsize = 10
    na = 1
    n_envs = 0 
    environments = [GathererState(gr=10,gc=10,vis=5,nagents=na,boxsize=boxsize,spawn_limit=10,volcano=False) for i in range(n_envs)]
    env1 = GathererState(gr=5,gc=5,vis=5,nagents=na,boxsize=boxsize,spawn_limit=1,volcano=False) 
    for agent in env1.agents:
        agent.circular_world = True
    environments.append(env1) 
    # environments = [env]

    #model = StateRAgent(input_size=100,state_size=3,containers=len(environments))
    #model1 = StateRAgent(input_size=100,state_size=3,containers=len(environments))
#    model2 = StateRAgent(input_size=100,state_size=3,containers=len(environments))

    model = StateRAgent(input_size=100,state_size=3,containers=len(environments))
    model1 = StateAgent(input_size=100,state_size=3,containers=len(environments))
 
#          1625202536.430398

#    model.load_state_dict(T.load("logs/models/1625317286.807864.pth"))
#    model1.load_state_dict(T.load("logs/models/1625317286.815844.pth"))

    models = [model]
    s = MultiEnvironmentSimulator(
        models,environments,nactions=6,
        log_message="Testing for fallout",
        visual_activations=True)

    train = 1 
    s.run(1000,500,train=train,render_once=1,saveonce=2)
