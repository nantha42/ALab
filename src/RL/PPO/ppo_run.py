from tqdm import tqdm
import gym 
import numpy as np 
from ppo_agents import * 
import matplotlib.pyplot as plt
import time
from Env import * 
from utils import *

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


def train1(config):
   
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 4
    n_epochs = 4
    print(env.observation_space.shape)
    actor = ActorNet(env.observation_space.shape[0],  env.action_space.n)
    critic = CriticNet(env.observation_space.shape[0],  env.action_space.n)

    agent = CAgent(actor,critic) 
    recorder = RLGraph()

    for i in range(300):
        observation= env.reset()
        done=False
        n_steps = 0
        score = 0
        while not done:
            action,prob,val,entropy = agent.choose_action(observation) 
            # env.render()
            observation_,reward,done,info = env.step(action)
            score+= reward 
            n_steps += 1
            agent.remember(observation,action,prob,val,reward,done)
            if n_steps % N == 0:
                agent.learn()
            observation = observation_
        recorder.newdata(score)
        recorder.plot("PPO/cart.png")

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
     
    if type == "Default":
        actor = ActorNet(vsize*vsize,  nactions)
        critic = CriticNet(vsize*vsize, nactions)
        agent = CAgent(actor,critic,batch_size = batch_size,model_name=model_name)
    
    if type == "Memory":
        actor = ActorGRUNet(vsize*vsize,  nactions)
        critic = CriticGRUNet(vsize*vsize, nactions)
        agent = MemoryAgent(actor,critic,batch_size=batch_size,model_name=model_name)

    if load_model :
        agent.load_model()
    agent.n_epochs = 5
    for i in range(episodes):
        hard_reset = False 
        game.reset(hard_reset) 
        state = game.get_state().reshape(-1)
        if i%5==0:
            game.enable_draw = True 
        else:
            game.enable_draw = False
        if type == "Memory" or type=="Atten":
            agent.reset()

        pbar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        trewards = 0
        entropy_term = 0
        for j in pbar: 
            if type == "Default":
                action, prob, value, entropy  = agent.choose_action(state)
            else:
                action, prob, value, entropy, ah  = agent.choose_action(state)

            # print(entropy)
            entropy_term += entropy.item()
            avec = np.zeros((nactions));avec[action] = 1
            new_state,reward = game.act(avec)
            if type == "Default":
                agent.remember(state,action,prob,value,reward/100.0)
            else:
                agent.remember(state,action,prob,value,reward/100.0,ah)

            state = new_state
            trewards += reward
            state = new_state.reshape(-1)
            game.step()
            pbar.set_description(f"Episodes: {i:4} Rewards: {trewards:2}")
        
        agent.learn(entropy_term)
        starting = time.time()
        recorder.newdata(trewards)
        show_once = 1 
        if i% show_once == show_once -1:
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
        agent.save_model()
    recorder.save(HIST_FILENAME)

if __name__ == '__main__':
    EPISODES = 5000
    STEPS = 500

    c = Config("GRUDlrAgent-S5")
    c.HIDDEN_SIZE =  64 
    c.TYPE = "Default" 
    c.VSIZE = 5
    c.NACTIONS = 6
    c.NLAYERS = 4
    c.GSIZE= (14,14)
    c.LOADMODEL = False 

    train(c)