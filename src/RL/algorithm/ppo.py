import torch as T
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import numpy as np
import pygame as py
import time
from .utils import RLGraph


class LSTMTrainer:
    def __init__(self, model,
                 learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.data = []

    def store_records(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [
        ], [], [], [], [], [], [], []
        for T in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = T
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in,
                                              h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())
        gamma = 0.99
        for i in range(5):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma* v_prime * done_mask 
            v_s = self.v(s,first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()


class TrainerGRU:
    def __init__(self, model,
                 learning_rate=0.0001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.data = []

    def store_records(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst,  prob_a_lst, h_in_lst,  done_lst = [
        ], [], [], [], [], [] 
        for transition in self.data:
            s, a, r,  prob_a, h_in,  done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r,  done_mask, prob_a = T.tensor(s_lst, dtype=T.float), T.tensor(a_lst), \
            T.tensor(r_lst), \
            T.tensor(done_lst, dtype=T.float), T.tensor(prob_a_lst)

        self.data = []
        return s, a, r,  done_mask, prob_a, h_in_lst[0] 
    
    def update(self):
        s, a, r,  done_mask, prob_a, (h1_in) = self.make_batch()
        hidden = h1_in.detach()
        gamma = 0.99
        
        k_epoch = 8 
        for i in range(k_epoch):
            discounted_rewards  = []
            Gt = 0
            for t in reversed(range(len(r))):
                Gt = r[t] + done_mask[t]*gamma*Gt
                discounted_rewards.append(Gt)

            discounted_rewards.reverse()
            discounted_rewards = T.tensor(discounted_rewards,dtype=T.float)
            #PROBLEM POSSIBLE OVERWRITING HIDDEN CAN DESTROY THE GRADIENT OF HIDDEN 
            self.model.hidden = hidden
            print(hidden.grad)
            pi = self.model.forward(s)
            pi_a = pi.squeeze(1).gather(1,a)
            print(pi_a.shape,prob_a.shape)
            ratio = T.exp( T.log(pi_a) - T.log(prob_a) )

            surr1 = ratio * discounted_rewards 
            eps_clip = 0.1
            surr2 = T.clamp(ratio, 1-eps_clip, 1+eps_clip)*discounted_rewards
            loss = -T.min(surr1, surr2)
            self.optimizer.zero_grad()
            loss = loss.mean()
            print("LOSS: ",loss.item())
            loss.backward()
            self.optimizer.step()
            print(hidden.grad)


class TrainerNOGRU:
    def __init__(self, model,
                 learning_rate=0.0001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.data = []

    def store_records(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst,  prob_a_lst, h_in_lst,  done_lst = [
        ], [], [], [], [], [] 
        for transition in self.data:
            s, a, r,  prob_a, h_in,  done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            prob_a_lst.append(prob_a)
            h_in_lst.append(h_in)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        # print("PROB A LIST",prob_a_lst)
        s, a, r,  done_mask, prob_a = T.tensor(s_lst, dtype=T.float), T.tensor(a_lst), \
            T.tensor(r_lst), \
            T.tensor(done_lst, dtype=T.float), T.tensor(prob_a_lst)
        self.data = []
        # print("AFTER",prob_a)
        return s, a, r,  done_mask, prob_a, [] 
    
    def update(self):
        s, a, r,  done_mask, prob_a, (h1_in) = self.make_batch()
        gamma = 0.99
        k_epoch = 1 
        for i in range(k_epoch):
            discounted_rewards  = []
            Gt = 0
            for t in reversed(range(len(r))):
                Gt = r[t] + done_mask[t]*gamma*Gt
                discounted_rewards.append(Gt)
            discounted_rewards.reverse()
            discounted_rewards = T.tensor(discounted_rewards,dtype = T.float)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards


            # print(discounted_rewards)

            # disc2 = []
            # for t in range(len(r)):
            #     Gt = 0
            #     pw = 0
            #     for u in r[t:]:
            #         Gt = Gt + gamma**pw*u
            #         pw = pw + 1
            #     disc2.append(Gt)

            # disc2 = T.tensor(disc2 ,dtype=T.float)
            # print(disc2)
            # print("PROBABILITIES",prob_a)
            # print(prob_a.shape,discounted_rewards.shape)
            # gradient = -T.log(prob_a)*discounted_rewards

            # self.model.hidden = hidden
            # print(hidden.grad)
            pi = self.model.forward(s)
            # print(pi.squeeze(1))
            # print(pi.shape,pi.squeeze(0).shape)
            # print(pi[:3].squeeze(1).gather(1,a[:3]))
            pi_a = pi.squeeze(1).gather(1,a)

            ratio = T.exp( T.log(pi_a) - T.log(prob_a) )
            surr1 = ratio * discounted_rewards 
            eps_clip = 0.2
            surr2 = T.clamp(ratio, 1-eps_clip, 1+eps_clip)*discounted_rewards
            loss = -T.min(surr1, surr2)
 
            # policy_gradient = []
            # for prob, Gt in zip(prob_a,discounted_rewards):
            #     policy_gradient.append(-(prob_a)* Gt)

            self.optimizer.zero_grad()
            # loss = gradient.sum()
            print("LOSS: ",loss.sum().item())
            loss.sum().backward(retain_graph=True)

            # policy_gradient = T.stack(policy_gradient).sum()
            # policy_gradient.backward()
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
        super().__init__(model,environment,trainer,nactions=nactions,log_message=log_message,visual_activations = visual_activations)
        py.init()        
        extra_width = 300 
        env_w = environment.win.get_width()
        env_h = environment.win.get_height()
        if visual_activations:
            self.w = 50 + env_w   + extra_width  
        else:
            self.w = 50 + env_w 
        self.h = 50 + env_h 
 
        self.font = py.font.SysFont("times",10)
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

    def render_text(self,surf,text,pos):
        text = self.font.render(text,True,(200,200,200))
        trect = text.get_rect()
        trect.topleft =  pos 
        surf.blit(text,trect)
        return surf


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

    def surf_create_graph(self,values,x_label,y_value,width):
        wid,hei = width,150
        surf_size = (wid,hei)
        surf = py.Surface(surf_size)
        length = len(values)
        text = self.font.render(x_label,True,(200,200,200))
        # mark_text = self.font.render("R: "+str(values[-1]), True, (200,200,200)) 
        mark_text = self.font.render(y_value,True, (200,200,200)) 
        maxi  =max(values)
        mini = min(values)
        poly = []
        for i in range(0,length,max(1,int(length/(wid-10) )) ):
            v = values[i]
            x = (i/length)*(wid-10)
            if (maxi-mini) != 0:
                y = (hei-text.get_height()) - ((hei-10)/(maxi -mini))*v
            else:
                y = (hei-text.get_height())
            poly.append((x,y))
        line_poly = list(poly)
        poly.append((poly[-1][0],poly[0][1]))
        poly.append(poly[0])
        py.draw.polygon(surf,(200,0,200),poly,0)
        if len(line_poly) > 1:
            py.draw.lines(surf,(0,255,255),False,line_poly,3)
        # surf = self.render_text(surf,"steps",(self.wid/2))
        trect = text.get_rect()
        trect.topright =  (text.get_width(),hei-text.get_height()) 
        surf.blit(text,trect)
        
        trect = mark_text.get_rect()
        trect.topleft = (0,0)
        surf.blit(mark_text,trect)
        
            # py.draw.circle(surf,(0,255,255),(x,y),1)
        return surf

    def draw_episode_reward(self,w=150):
        y_value = "R: "+ str(self.episode_rewards[-1])
        surf = self.surf_create_graph(self.episode_rewards,"steps",y_value,width=w)
        return surf
    

    def draw_neural_image(self):
        panel = py.Surface((500,self.window.get_height()))
        surf_activation,asize = self.surf_neural_activation()
        surf_weights,wsize = self.surf_neural_weights()
        surf_hidden,hsize = self.surf_hidden_activation()
        panel.blit(surf_activation,(0,0))
        if surf_weights is not None:
            panel.blit(surf_weights,(asize[0],0))
        panel.blit(surf_hidden,(0,max(asize[1],wsize[1])))
        surf_graph_1 = self.draw_episode_reward(w=hsize[0] )
        panel.blit(surf_graph_1, (0,max(asize[1],wsize[1])+ surf_hidden.get_height()+5) )
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
        assert not hasattr(self.model,"hidden_states"), "no hidden_states list attribute"
        self.env.display_neural_image = self.visual_activations
        
        for _ in range(episodes):
            self.episode_rewards = []
            self.env.reset()
            self.env.enable_draw = True if not train or _ % render_once == render_once-1 else False

            if reset_model:
                self.model.reset()

            state = self.env.get_state().reshape(-1)
            bar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            trewards = 0

            for step in bar:
                # if self.env.game_done:
                #     break
                state = T.from_numpy(state).float()
                # print(state)
                actions = self.model(state).view(-1)
                # print(actions)
                c = Categorical(actions)
                action = c.sample()
                prob = c.probs[action]

                # print(actions,prob)
                u = np.zeros(self.nactions)
                u[action] = 1.0
                newstate,reward = self.env.act(u)
                trewards += reward
                self.episode_rewards.append(trewards)
                if train:
                    if self.model.type == "mem":
                        self.trainer.store_records((state.tolist(),action,reward, prob,self.model.hidden_states[-2], False))
                    else:
                        self.trainer.store_records((state.tolist(),action,reward, prob,[], False))
                    # self.trainer.store_records((state.tolist(),action,reward, c.log_prob(action),self.model.hidden_states[-2], False))
                
                state = newstate.reshape(-1)
                if self.model.type == "mem" and self.visual_activations :
                    u = T.cat(self.activations,dim=0).reshape(-1)
                    self.neural_image_values = u.detach().numpy()
                    self.activations = []
                    if _ % 10 == 0 and step/steps == 0:
                        self.update_weights()
                        self.neural_weights = self.weights
                        self.weight_change = True
                    if type(self.model.hidden_vectors) != type(None):
                        self.hidden_state = self.model.hidden_vectors
                else:
                    self.activations = []

                bar.set_description(f"Episode: {_:4} Rewards : {trewards}")
                if train:
                    self.env.step() 
                else:
                    self.env.step(speed=0)
                
                self.event_handler()
                self.window.fill((0,0,0))
                if self.visual_activations and (not train  or _ % render_once == render_once-1):
                    if self.model.type == "mem":
                        self.draw_neural_image()
                    self.window.blit(self.env.win,(0,0))
                
            if train:
                self.trainer.update()
                self.recorder.newdata(trewards)
                if _ % saveonce == saveonce-1:
                    self.recorder.save()
                    self.recorder.plot()

                if _ % saveonce == saveonce-1 and self.recorder.final_reward >= self.current_max_reward:
                    self.recorder.save_model(self.model)
                    self.current_max_reward = self.recorder.final_reward
        print("******* Run Complete *******")

