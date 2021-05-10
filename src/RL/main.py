import torch as T
import torch.nn as nn
import pygame as py
import typing
import numpy as np
from torchsummary import summary

from algorithm.reinforce import Trainer, Runner, Simulator
from environment.power import PowerGame


class RAgent(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.pre = nn.Linear(input_size, 64)
        self.gru = nn.GRU(64, 64, 1)
        self.hidden = T.zeros((1, 1, 64))
        self.layers = nn.Sequential(
            nn.Linear(64, 6),
            nn.Softmax(dim=-1)
        )
        self.type = "mem"
        self.hidden_vectors = None 

    def reset(self):
        self.hidden = T.zeros((1, 1, 64))

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
    env = PowerGame(gr=20, gc=20, vis=5,neural_image=True)
    agent = RAgent(5*5)
    agent.load_state_dict(T.load("logs/models/1620290798.pth"))
    trainer = Trainer(agent, learning_rate=0.001)
    env.enable_draw = False
    # runner = Runner(
    #         agent,env,trainer,
    #         nactions = 6,
    #         log_message="Continuing Training Model",
    #         visual_activations=True
    #         )
   
    # runner.run(1000,5000,train=False,render_once=10,saveonce=7)

    s = Simulator(
        agent,env,trainer,
        nactions=6,
        log_message="Continuing Training model",
        visual_activations=True
    )
    print(s.visual_activations)
    s.run(1000,5000,train=False,render_once=10,saveonce=7)