import torch as T 
import torch.nn as nn
import pygame as py 
import typing 
import numpy as np
from torchsummary import summary

from algorithm.reinforce import Trainer,Runner
from environment.power import  PowerGame


class RAgent(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.input_size = input_size
        self.pre = nn.Linear(input_size,64)
        self.gru = nn.GRU(64,64,1)
        self.hidden = T.zeros((1,1,64))
        self.layers = nn.Sequential(
            nn.Linear(64,6) ,
            nn.Softmax(dim=-1)
        )
        self.type = "mem"

    def reset(self):
        self.hidden = T.zeros((1,1,64)) 

    def forward(self,x):
        x = x.reshape(1,-1,self.input_size)
        x = self.pre(x)
        x,self.hidden = self.gru(x,self.hidden)
        o = self.layers(x)
        return o


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
        print(x.shape)
        o = self.layers(x)
        return o

env = PowerGame(gr=10,gc=10,vis=5)
agent = RAgent(5*5)
trainer = Trainer(agent,learning_rate=0.001)
env.enable_draw = False

runner = Runner(
        agent,env,trainer,
        nactions = 6,
        log_message="json dump indentation test"
        )

runner.run(1000,500,train=True,render_once=10)


