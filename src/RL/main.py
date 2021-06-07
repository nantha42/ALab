
from numpy.core.overrides import verify_matching_signatures
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
from environment.gatherer import Gatherer
from environment.collector import PowerGame 

from algorithm.reinforce import Trainer, MultiAgentRunner, MultiAgentSimulator, Simulator
# from algorithm.reinforce import Trainer,  Simulator

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
        self.vlayers = nn.Sequential(
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
 



# runner = Runner(
#         agent,env,trainer,
#         nactions = 6,
#         log_message="Continuing Training Model",
#         visual_activations=True
#         )

# runner.run(1000,5000,train=False,render_once=10,saveonce=7)


if __name__ == '__main__':
    # env = PowerGame(gr=20, gc=20, vis=5,neural_image=True)
    # agent.load_state_dict(T.load("logs/models/1621665458.pth"))

    # runner = Runner(
    #         agent,env,trainer,
    #         nactions = 6,
    #         log_message="Continuing Training Model",
    #         visual_activations=True
    #         )
   
    # runner.run(1000,5000,train=False,render_once=10,saveonce=7)

    # MULTI AGENTS TESTING
    nagents =  1 
    env = Gatherer(gr = 20,gc = 20,vis = 5,nagents=nagents)
    # models_names = ["1622821726.961184","1622821726.963818","1622821726.9635148"] 
    models_names = ["1622821726.963818","1622821726.9635148"] 
    # models_names = ["1622962499.238822","1622962499.241493","1622962499.242031"]
    models = [RAgent(input_size=100) for i in range(nagents)]

    # for m,n in zip(models,models_names):
    #     m.load_state_dict(T.load("logs/models/"+n+".pth"))

    trainers = [Trainer(m,learning_rate=0.001) for m in models ] 
    s = MultiAgentSimulator(
        models,env,trainers,nactions=7,
        log_message="single agent cloned later",
        visual_activations = True 
    )
    train = 1 
    s.run(1000,2000,train=train,render_once=10,saveonce=2)

    #SINGLE AGENT TESTING REINFORCE
    # s = Simulator(
    #     agent,env,trainer,
    #     nactions=6,
    #     log_message="Scaling up processors reduced collected gains",
    #     visual_activations= True 
    # )
    # print(s.visual_activations)
    # s.run(1000,500,train=True,render_once=1,saveonce=1)

    #PPO TESTING
    # env = PowerGame(gr=10,gc=10,vis=5)
    # model = AgentCA(input_size=25) 
    # # model.load_state_dict(T.load("logs/models/1622623059.6184058.pth"))
    # # model = Agent(input_size=49) 
    # trainer = TrainerNOGRU_V(model,learning_rate=0.001)
    # # trainer = TrainerNOGRU(model,learning_rate=0.001)
    # env.enable_draw = False
    # s = SimulatorV(
    #     model,env,trainer,
    #     nactions=6,
    #     log_message="nogru with 3 kepochs",
    #     visual_activations = True
    # )
    # s.run(1000,500,train=True,render_once=3,saveonce=3)

    #REINFORCE
    # model = Agent(input_size=25)
    # trainer = Trainer(model,learning_rate=0.001)
    # s = Simulator(
    #     model,env,trainer,
    #     nactions=6,
    #     log_message="Reinforce",
    #     visual_activations=False
    # )
    # s.run(1000,500,train=True,render_once=4,saveonce=3)