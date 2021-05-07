import torch as T
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import numpy as np
import time
from .utils import RLGraph 


class Trainer:
    def __init__(self,model,
                learning_rate = 0.001):
        self.model = model 
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(self.model.parameters(),lr=learning_rate) 
        self.rewards = []
        self.log_probs = []
    
    def store_records(self,reward,log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def clear_memory(self):
        self.rewards = []
        self.log_probs = []
    
    def update(self): 
        discounted_rewards = []
        GAMMA = 0.99
        for t in range(len(self.rewards)):
            Gt = 0 
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = T.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = T.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()


class Runner:
    def __init__(self,model,environment,trainer,nactions=6,log_message=None,visual_activations = False):
        self.env = environment
        self.model = model
        self.trainer = trainer
        self.nactions = nactions 
        self.recorder = RLGraph()
        self.recorder.log_message = log_message
        self.activations = []
        self.visual_activations = visual_activations
        if visual_activations:
            def hook_fn(m,i,o):
                if type(o) == type((1,)):
                    for u in o:
                        self.activations.append(u.reshape(-1))
                else:
                    self.activations.append(o.reshape(-1))


            for n,l in self.model._modules.items():
                l.register_forward_hook(hook_fn)

    
    def run(self,episodes,steps,train=False,render_once=1e10,saveonce=10):
        if train:
            assert self.recorder.log_message is not None, "log_message is necessary during training, Instantiate Runner with log message"

        
        reset_model = False
        if hasattr(self.model,"type") and self.model.type == "mem":
            print("Recurrent Model")
            reset_model = True
        self.env.display_neural_image = self.visual_activations
        for _ in range(episodes):

            self.env.reset()
            self.env.enable_draw = True if not train or _ % render_once == render_once-1 else False

            if reset_model:
                self.model.reset()

            state = self.env.get_state().reshape(-1)
            bar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            trewards = 0

            for step in bar:
                
                state = T.from_numpy(state).float()
                actions = self.model(state)

                c = Categorical(actions)
                action = c.sample()
                log_prob = c.log_prob(action)

                u = np.zeros(self.nactions)
                u[action] = 1.0
                newstate,reward = self.env.act(u)
                state = newstate.reshape(-1)
                trewards += reward

                if train:
                    self.trainer.store_records(reward,log_prob)
                
                if self.visual_activations:
                    u = T.cat(self.activations,dim=0)
                    l = u.shape[0]
                    lim = int(np.cbrt(l))
                    u = u[:lim**3].reshape((lim,lim,lim))
                    self.env.neural_image_values = u.detach().numpy()
                    self.activations = []

                bar.set_description(f"Episode: {_:4} Rewards : {trewards}")
                self.env.step() 
            if train:
                self.trainer.update()
                self.trainer.clear_memory()
                self.recorder.newdata(trewards)
                if _ % saveonce == saveonce-1:
                    self.recorder.save_model(self.model)
                if _ % 5 == 4:
                    self.recorder.save()
                    self.recorder.plot()

        print("******* Run Complete *******")