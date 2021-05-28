
import torch as T
import torch.nn as nn
import pygame as py
import typing
import numpy as np
from torchsummary import summary

from algorithm.reinforce import Trainer, MultiAgentRunner, MultiAgentSimulator
from environment.gatherer import Gatherer


class RAgent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.pre = nn.Linear(input_size, 64)
        self.gru = nn.GRU(64, 64, 1)
        self.hidden = T.zeros((1, 1, 64))
        self.layers = nn.Sequential(
            nn.Linear(64, 7),
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

    def forward(self, x):
        x = x.reshape(1, -1, self.input_size)
        x = self.pre(x)
        x, self.hidden = self.gru(x, self.hidden)
        self.hidden_vectors = self.hidden.detach().clone().squeeze(0).numpy()
        o = self.layers(x)
        return o

class Agent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        print(x.shape)
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
    env = Gatherer(gr = 20,gc = 20,vis = 7,nagents=2)
    model = RAgent(input_size = 196)
    model1 = RAgent(input_size = 196)
    # agent.load_state_dict(T.load("logs/models/1621665458.pth"))

    trainer = Trainer(model, learning_rate=0.001)
    trainer1 = Trainer(model1, learning_rate=0.001)
    env.enable_draw = False
    # runner = Runner(
    #         agent,env,trainer,
    #         nactions = 6,
    #         log_message="Continuing Training Model",
    #         visual_activations=True
    #         )
   
    # runner.run(1000,5000,train=False,render_once=10,saveonce=7)
    s = MultiAgentSimulator(
        [model,model1],env,[trainer,trainer1],nactions=7,
        log_message="Will not work for logging",
        visual_activations = True
    )
    # s = Simulator(
    #     agent,env,trainer,
    #     nactions=6,
    #     log_message="Scaling up processors reduced collected gains",
    #     visual_activations= True 
    # )
    print(s.visual_activations)
    s.run(1000,5000,train=True,render_once=1,saveonce=7)