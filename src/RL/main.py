import torch as T 
import torch.nn as nn
import pygame as py 
import typing 
import numpy as np

from algorithm.reinforce import Trainer,Runner
from environment.power import  PowerGame



class Agent(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,64) ,
            nn.ReLU(),
            nn.Linear(64,6),
            nn.Softmax(dim=-1)
        )
    
    def forward(self,x):
        o = self.layers(x)
        return o

env = PowerGame(gr=10,gc=10,vis=5)
agent = Agent(5*5)
trainer = Trainer(agent,learning_rate=0.001)

env.enable_draw = False
runner = Runner(agent,env,trainer,nactions = 6)

runner.run(1000,500,train=True)