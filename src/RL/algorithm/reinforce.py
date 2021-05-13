import torch as T
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import numpy as np
import pygame as py
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
        self.weights = []
        self.visual_activations = visual_activations
        self.current_max_reward = 0
        if visual_activations:
            def hook_fn(m,i,o):
                if type(o) == type((1,)):
                    for u in o:
                        self.activations.append(u.reshape(-1))
                else:
                    self.activations.append(o.reshape(-1))

            for n,l in self.model._modules.items():
                l.register_forward_hook(hook_fn)

    def update_weights(self):    
        self.weights = []
        for param in self.model.parameters():
            self.weights.append(T.tensor(param).clone().detach().reshape(-1))
        self.weights = T.cat(self.weights,dim=0).numpy()
        print("weights shape",self.weights.shape)
 
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
                    u = T.cat(self.activations,dim=0).reshape(-1)
                    self.env.neural_image_values = u.detach().numpy()
                    self.activations = []
                    if _ % 10 == 0 and step/steps == 0:
                        self.update_weights()
                        self.env.neural_weights = self.weights
                        self.env.weight_change = True
                    if type(self.model.hidden_vectors) != type(None):
                        self.env.hidden_state = self.model.hidden_vectors

                bar.set_description(f"Episode: {_:4} Rewards : {trewards}")
                if train:
                    self.env.step() 
                else:
                    self.env.step(speed=0)
                
            if train:
                self.trainer.update()
                self.trainer.clear_memory()
                self.recorder.newdata(trewards)
                if _ % saveonce == saveonce-1:
                    self.recorder.save()
                    self.recorder.plot()

                if _ % saveonce == saveonce-1 and self.recorder.final_reward >= self.current_max_reward:
                    self.recorder.save_model(self.model)
                    self.current_max_reward = self.recorder.final_reward
        print("******* Run Complete *******")



class Simulator(Runner):
    def __init__(self,model,environment,trainer,nactions=6,log_message=None,visual_activations = False):
        super().__init__(model,environment,trainer,nactions=6,log_message=log_message,visual_activations = visual_activations)
        py.init()        
        extra_width = 300 
        env_w = environment.win.get_width()
        env_h = environment.win.get_height()
        if visual_activations:
            self.w = 50 + env_w   + extra_width  
        else:
            self.w = 50 + env_w 
        self.h = 50 + env_h 
 
        self.window = py.display.set_mode((self.w,self.h),py.DOUBLEBUF,32)
        self.window.set_alpha(128)
        self.clock = py.time.Clock()
        self.enable_draw = True 


        self.hidden_state = None
        self.hidden_state_surfaces = []
        self.neural_image_values = []
        self.neural_image = None  # stores the surface
        self.neural_layout = None
        self.neural_layout_size = None
 
        self.display_neural_image = visual_activations 
        self.neural_weights = None
        self.neural_weight_surface = None
        self.weight_change = False

    def create_pack(self,arr):
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:int(s*s)].reshape((s,s))
        col_splits = np.random.randint(3,6)
        columns = []
        a_split = s/col_splits
        total = s
        starting = 0
        for i in range(col_splits-1):
            end = np.random.randint(a_split-20,a_split)
            columns.append([starting,starting+end])
            starting+= end 
        columns.append([starting,s])

        final_split = []
        row_splits = 5 
        for column in columns:
            starting = 0
            a_split = s/row_splits
            rows = []
            for i in range(row_splits-1):
                e = np.random.randint(a_split-30,a_split+10)
                rows.append([starting,starting+e,column[0],column[1]])
                starting += e
            rows.append([starting,s,column[0],column[1]])
            final_split.append(rows)
        surface_size = [[s,s],[row_splits,col_splits]]
        return final_split,surface_size
    
    def weight_from_pack(self):
        arr = self.neural_weights
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:(s*s)].reshape((s,s))
 
        out = []
        for col in self.neural_layout:
            rows = []
            for r in col:
                rows.append(arr[r[0]:r[1],r[2]:r[3]])
            out.append(rows)
        return out 
 
    def calculate_color(self,av,maxi,mini,poslimit,neglimit):
        cg = 0
        cr = 0
  
        if av < 0:
            cr = int( ((av)/mini)*neglimit)
            cg = 0
            # cg = poslimit- (cr)
        else:
            cg = int((av/maxi)*poslimit)
            # cr = neglimit - cg 
            cr = 0
        return cr,cg
 
    def calculate_limits(self,maxi,mini):
        if abs(maxi) > abs(mini):
            poslimit = 180 
            neglimit = int(255.0*( abs(mini)/ abs(maxi)))
        else:
            neglimit = 255 
            poslimit = int(255.0*( abs(maxi)/ abs(mini)  ))
        return poslimit,neglimit 

    def surf_neural_activation(self):
        """ Returns a Drawn surface for neural activation"""

        assert type(self.neural_image_values) == type(np.array([])), "neural_image_values should be in numpy array"

        points = []
        varr = self.neural_image_values
 
        varr = varr.reshape(-1)
        l = int(np.sqrt(varr.shape[0]))
        varr = varr[-l*l:].reshape((l,l))
        
        sz = 10
        activ_surf = py.Surface((l*sz+20,l*sz+20))
        maxi = np.max(varr)
        mini = np.min(varr)
        
        poslimit,neglimit = self.calculate_limits(maxi,mini)
        for r in range(l):
            for c in range(l):
                av = varr[r][c]
                cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                py.draw.rect(activ_surf,colorvalue,(10+c*sz,10+r*sz,sz,sz))
        return activ_surf,(activ_surf.get_width(),activ_surf.get_height())
    
    def surf_hidden_activation(self):
        if type(self.hidden_state) != type(None):
            maxi = np.max(self.hidden_state)
            mini = np.min(self.hidden_state)
            c,r = self.hidden_state.shape
            sz = 2 
            state_surface= py.Surface((c*sz,r*sz))
            swidth = state_surface.get_width()

            poslimit,neglimit = self.calculate_limits(maxi,mini)
            for i in range(c):
                for j in range(r):
                    av = self.hidden_state[i][j]
                    cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                    colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                    py.draw.rect(state_surface,colorvalue,(i*sz,j*sz,sz,sz))

            self.hidden_state_surfaces.append(state_surface)
            if len(self.hidden_state_surfaces) > 330/swidth-3*swidth:
                self.hidden_state_surfaces = self.hidden_state_surfaces[1:]

            l = len(self.hidden_state_surfaces)
            surf_w = self.hidden_state_surfaces[0].get_width()
            surf_h = self.hidden_state_surfaces[0].get_height()
            full_surf = py.Surface((20+l*surf_w,20+surf_h))
            for i in range(l):
                full_surf.blit( self.hidden_state_surfaces[i], (10+i*surf_w,10))
            return full_surf,(full_surf.get_width(),full_surf.get_height())
                
    def surf_neural_weights(self):
        if self.neural_weights is not None and self.weight_change:
            self.weight_change = False
            if self.neural_layout == None:
                self.neural_layout,self.neural_layout_size = self.create_pack(self.neural_weights)
            
            pix_size , gaps = self.neural_layout_size 
            gap_size = 1
            sz = 1
            p_x = pix_size[0]*sz + gaps[0]*gap_size + 20
            p_y = pix_size[1]*sz + gaps[1]*gap_size + 20
            neural_weight_surface = py.Surface((p_x,p_y))
            weights = self.weight_from_pack()
            startx = 10
            for col in weights:
                starty = 10
                for weight in col:
                    r,c = weight.shape
                    maxi = np.max(weight)
                    mini = np.min(weight)
                    poslimit,neglimit = self.calculate_limits(maxi,mini)
                    for i in range(r):
                        for j in range(c):
                            av = weight[i][j]
                            cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                            colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                            py.draw.rect(neural_weight_surface,colorvalue,(startx+j*sz,starty+i*sz,sz,sz))
                    starty += r*sz + gap_size 
                startx += c*sz + gap_size 
            print("returning ",neural_weight_surface)
            sizefor = [neural_weight_surface.get_width(),neural_weight_surface.get_height()]
            print("sie",sizefor)
            self.neural_weight_surface = [neural_weight_surface,sizefor]
            return neural_weight_surface,sizefor 
        else:
            return self.neural_weight_surface 

    def draw_neural_image(self):
        panel = py.Surface((500,self.window.get_height()))
        surf_activation,asize = self.surf_neural_activation()
        surf_weights,wsize = self.surf_neural_weights()
        surf_hidden,hsize = self.surf_hidden_activation()
        panel.blit(surf_activation,(0,0))
        if surf_weights is not None:
            panel.blit(surf_weights,(asize[0],0))
        panel.blit(surf_hidden,(0,max(asize[1],wsize[1])))
        self.window.blit(panel,(500,10))

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()
 

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
                
                if self.visual_activations :
                    u = T.cat(self.activations,dim=0).reshape(-1)
                    self.neural_image_values = u.detach().numpy()
                    self.activations = []
                    if _ % 10 == 0 and step/steps == 0:
                        self.update_weights()
                        self.neural_weights = self.weights
                        self.weight_change = True
                    if type(self.model.hidden_vectors) != type(None):
                        self.hidden_state = self.model.hidden_vectors

                bar.set_description(f"Episode: {_:4} Rewards : {trewards}")
                if train:
                    self.env.step() 
                else:
                    self.env.step(speed=0)
                
                self.event_handler()
                self.window.fill((0,0,0))
                if self.visual_activations and (not train  or _ % render_once == render_once-1):
                    self.draw_neural_image()
                    self.window.blit(self.env.win,(0,0))
                
            if train:
                self.trainer.update()
                self.trainer.clear_memory()
                self.recorder.newdata(trewards)
                if _ % saveonce == saveonce-1:
                    self.recorder.save()
                    self.recorder.plot()

                if _ % saveonce == saveonce-1 and self.recorder.final_reward >= self.current_max_reward:
                    self.recorder.save_model(self.model)
                    self.current_max_reward = self.recorder.final_reward
        print("******* Run Complete *******")

