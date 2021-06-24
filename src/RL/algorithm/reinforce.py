from os import environ
import torch as T
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import numpy as np
import pygame as py
import time
from .utils import RLGraph


class Trainer:
    def __init__(self, model,
                 learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.rewards = []
        self.log_probs = []

    def store_records(self, reward, log_prob):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.rewards = []
        self.log_probs = []

    def update(self):
        discounted_rewards = []
        # GAMMA = 0.99
        # for t in range(len(self.rewards)):
        #     Gt = 0
        #     pw = 0
        #     for r in self.rewards[t:]:
        #         Gt = Gt + GAMMA**pw * r
        #         pw = pw + 1
        #     discounted_rewards.append(Gt)
        Gt = 0
        gamma = 0.99
        for t in reversed(range(len(self.rewards))):
            Gt = self.rewards[t] + gamma*Gt
            discounted_rewards.append(Gt)

        discounted_rewards.reverse()

        discounted_rewards = T.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9)  # normalize discounted rewards
        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        self.optimizer.zero_grad()
        # print("Length",len(policy_gradient))
        policy_gradient = T.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()


class Runner:
    def __init__(self, model, environment, trainer, nactions=6, log_message=None, visual_activations=False):
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
            def hook_fn(m, i, o):
                if type(o) == type((1,)):
                    for u in o:
                        self.activations.append(u.reshape(-1))
                else:
                    self.activations.append(o.reshape(-1))

            for n, l in self.model._modules.items():
                l.register_forward_hook(hook_fn)

    def update_weights(self):
        self.weights = []
        for param in self.model.parameters():
            self.weights.append(T.tensor(param).clone().detach().reshape(-1))
        self.weights = T.cat(self.weights, dim=0).numpy()
        print("weights shape", self.weights.shape)

    def run(self, episodes, steps, train=False, render_once=1e10, saveonce=10):
        if train:
            assert self.recorder.log_message is not None, "log_message is necessary during training, Instantiate Runner with log message"

        reset_model = False
        if hasattr(self.model, "type") and self.model.type == "mem":
            print("Recurrent Model")
            reset_model = True
        self.env.display_neural_image = self.visual_activations
        for _ in range(episodes):

            self.env.reset()
            self.env.enable_draw = True if not train or _ % render_once == render_once-1 else False

            if reset_model:
                self.model.reset()

            state = self.env.get_state().reshape(-1)
            bar = tqdm(
                range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            trewards = 0

            for step in bar:

                state = T.from_numpy(state).float()
                actions = self.model(state)

                c = Categorical(actions)
                action = c.sample()
                log_prob = c.log_prob(action)

                u = np.zeros(self.nactions)
                u[action] = 1.0
                newstate, reward = self.env.act(u)
                state = newstate.reshape(-1)
                trewards += reward

                if train:
                    self.trainer.store_records(reward, log_prob)

                if self.visual_activations:
                    u = T.cat(self.activations, dim=0).reshape(-1)
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
    def __init__(self, model, environment, trainer, nactions=6, log_message=None, visual_activations=False):
        super().__init__(model, environment, trainer, nactions=nactions,
                         log_message=log_message, visual_activations=visual_activations)
        py.init()
        extra_width = 300
        env_w = environment.win.get_width()
        env_h = environment.win.get_height()
        if visual_activations:
            self.w = 50 + env_w + extra_width
        else:
            self.w = 50 + env_w
        self.h = 50 + env_h

        self.font = py.font.SysFont("times", 10)
        self.window = py.display.set_mode((self.w, self.h), py.DOUBLEBUF, 32)
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

    def render_text(self, surf, text, pos):
        text = self.font.render(text, True, (200, 200, 200))
        trect = text.get_rect()
        trect.topleft = pos
        surf.blit(text, trect)
        return surf

    def create_pack(self, arr):
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:int(s*s)].reshape((s, s))
        col_splits = np.random.randint(3, 6)
        columns = []
        a_split = s/col_splits
        total = s
        starting = 0
        for i in range(col_splits-1):
            end = np.random.randint(a_split-20, a_split)
            columns.append([starting, starting+end])
            starting += end
        columns.append([starting, s])

        final_split = []
        row_splits = 5
        for column in columns:
            starting = 0
            a_split = s/row_splits
            rows = []
            for i in range(row_splits-1):
                e = np.random.randint(a_split-30, a_split+10)
                rows.append([starting, starting+e, column[0], column[1]])
                starting += e
            rows.append([starting, s, column[0], column[1]])
            final_split.append(rows)
        surface_size = [[s, s], [row_splits, col_splits]]
        return final_split, surface_size

    def weight_from_pack(self):
        arr = self.neural_weights
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:(s*s)].reshape((s, s))

        out = []
        for col in self.neural_layout:
            rows = []
            for r in col:
                rows.append(arr[r[0]:r[1], r[2]:r[3]])
            out.append(rows)
        return out

    def calculate_color(self, av, maxi, mini, poslimit, neglimit):
        cg = 0
        cr = 0

        if av < 0:
            cr = int(((av)/mini)*neglimit)
            cg = 0
            # cg = poslimit- (cr)
        else:
            cg = int((av/maxi)*poslimit)
            # cr = neglimit - cg
            cr = 0
        return cr, cg

    def calculate_limits(self, maxi, mini):
        if abs(maxi) > abs(mini):
            poslimit = 180
            neglimit = int(255.0*(abs(mini) / abs(maxi)))
        else:
            neglimit = 255
            poslimit = int(255.0*(abs(maxi) / abs(mini)))
        return poslimit, neglimit

    def surf_neural_activation(self):
        """ Returns a Drawn surface for neural activation"""

        assert type(self.neural_image_values) == type(
            np.array([])), "neural_image_values should be in numpy array"

        points = []
        varr = self.neural_image_values

        varr = varr.reshape(-1)
        l = int(np.sqrt(varr.shape[0]))
        varr = varr[-l*l:].reshape((l, l))

        sz = 10
        activ_surf = py.Surface((l*sz+20, l*sz+20))
        maxi = np.max(varr)
        mini = np.min(varr)

        poslimit, neglimit = self.calculate_limits(maxi, mini)
        for r in range(l):
            for c in range(l):
                av = varr[r][c]
                cr, cg = self.calculate_color(
                    av, maxi, mini, poslimit, neglimit)
                colorvalue = (abs(cr), abs(cg), max(abs(cr), abs(cg)))
                py.draw.rect(activ_surf, colorvalue,
                             (10+c*sz, 10+r*sz, sz, sz))
        return activ_surf, (activ_surf.get_width(), activ_surf.get_height())

    def surf_hidden_activation(self):
        if type(self.hidden_state) != type(None):
            maxi = np.max(self.hidden_state)
            mini = np.min(self.hidden_state)
            c, r = self.hidden_state.shape
            sz = 2
            state_surface = py.Surface((c*sz, r*sz))
            swidth = state_surface.get_width()

            poslimit, neglimit = self.calculate_limits(maxi, mini)
            for i in range(c):
                for j in range(r):
                    av = self.hidden_state[i][j]
                    cr, cg = self.calculate_color(
                        av, maxi, mini, poslimit, neglimit)
                    colorvalue = (abs(cr), abs(cg), max(abs(cr), abs(cg)))
                    py.draw.rect(state_surface, colorvalue,
                                 (i*sz, j*sz, sz, sz))

            self.hidden_state_surfaces.append(state_surface)
            if len(self.hidden_state_surfaces) > 330/swidth-3*swidth:
                self.hidden_state_surfaces = self.hidden_state_surfaces[1:]

            l = len(self.hidden_state_surfaces)
            surf_w = self.hidden_state_surfaces[0].get_width()
            surf_h = self.hidden_state_surfaces[0].get_height()
            full_surf = py.Surface((20+l*surf_w, 20+surf_h))
            for i in range(l):
                full_surf.blit(
                    self.hidden_state_surfaces[i], (10+i*surf_w, 10))
            return full_surf, (full_surf.get_width(), full_surf.get_height())

    def surf_neural_weights(self):
        if self.neural_weights is not None and self.weight_change:
            self.weight_change = False
            if self.neural_layout == None:
                self.neural_layout, self.neural_layout_size = self.create_pack(
                    self.neural_weights)

            pix_size, gaps = self.neural_layout_size
            gap_size = 1
            sz = 1
            p_x = pix_size[0]*sz + gaps[0]*gap_size + 20
            p_y = pix_size[1]*sz + gaps[1]*gap_size + 20
            neural_weight_surface = py.Surface((p_x, p_y))
            weights = self.weight_from_pack()
            startx = 10
            for col in weights:
                starty = 10
                for weight in col:
                    r, c = weight.shape
                    maxi = np.max(weight)
                    mini = np.min(weight)
                    poslimit, neglimit = self.calculate_limits(maxi, mini)
                    for i in range(r):
                        for j in range(c):
                            av = weight[i][j]
                            cr, cg = self.calculate_color(
                                av, maxi, mini, poslimit, neglimit)
                            colorvalue = (abs(cr), abs(
                                cg), max(abs(cr), abs(cg)))
                            py.draw.rect(neural_weight_surface, colorvalue,
                                         (startx+j*sz, starty+i*sz, sz, sz))
                    starty += r*sz + gap_size
                startx += c*sz + gap_size
            print("returning ", neural_weight_surface)
            sizefor = [neural_weight_surface.get_width(
            ), neural_weight_surface.get_height()]
            print("sie", sizefor)
            self.neural_weight_surface = [neural_weight_surface, sizefor]
            return neural_weight_surface, sizefor
        else:
            return self.neural_weight_surface

    def surf_create_graph(self, values, x_label, y_value, width):
        wid, hei = width, 150
        surf_size = (wid, hei)
        surf = py.Surface(surf_size)
        length = len(values)
        text = self.font.render(x_label, True, (200, 200, 200))
        # mark_text = self.font.render("R: "+str(values[-1]), True, (200,200,200))
        mark_text = self.font.render(y_value, True, (200, 200, 200))
        maxi = max(values)
        mini = min(values)
        poly = []
        for i in range(0, length, max(1, int(length/(wid-10)))):
            v = values[i]
            x = (i/length)*(wid-10)
            if (maxi-mini) != 0:
                y = (hei-text.get_height()) - ((hei-10)/(maxi - mini))*v
            else:
                y = (hei-text.get_height())
            poly.append((x, y))
        line_poly = list(poly)
        poly.append((poly[-1][0], poly[0][1]))
        poly.append(poly[0])
        py.draw.polygon(surf, (200, 0, 200), poly, 0)
        if len(line_poly) > 1:
            py.draw.lines(surf, (0, 255, 255), False, line_poly, 3)
        # surf = self.render_text(surf,"steps",(self.wid/2))
        trect = text.get_rect()
        trect.topright = (text.get_width(), hei-text.get_height())
        surf.blit(text, trect)

        trect = mark_text.get_rect()
        trect.topleft = (0, 0)
        surf.blit(mark_text, trect)

        # py.draw.circle(surf,(0,255,255),(x,y),1)
        return surf

    def draw_episode_reward(self, w=150):
        y_value = "R: " + str(self.episode_rewards[-1])
        surf = self.surf_create_graph(
            self.episode_rewards, "steps", y_value, width=w)
        return surf

    def draw_neural_image(self):
        panel = py.Surface((500, self.window.get_height()))
        surf_activation, asize = self.surf_neural_activation()
        surf_weights, wsize = self.surf_neural_weights()
        surf_hidden, hsize = self.surf_hidden_activation()
        panel.blit(surf_activation, (0, 0))
        if surf_weights is not None:
            panel.blit(surf_weights, (asize[0], 0))
        panel.blit(surf_hidden, (0, max(asize[1], wsize[1])))
        surf_graph_1 = self.draw_episode_reward(w=hsize[0])
        panel.blit(surf_graph_1, (0, max(
            asize[1], wsize[1]) + surf_hidden.get_height()+5))
        self.window.blit(panel, (500, 10))

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()

    def run(self, episodes, steps, train=False, render_once=1e10, saveonce=10):
        if train:
            assert self.recorder.log_message is not None, "log_message is necessary during training, Instantiate Runner with log message"

        reset_model = False
        if hasattr(self.model, "type") and self.model.type == "mem":
            print("Recurrent Model")
            reset_model = True
        self.env.display_neural_image = self.visual_activations

        for _ in range(episodes):
            self.episode_rewards = []
            self.env.reset()
            self.env.enable_draw = True if not train or _ % render_once == render_once-1 else False

            if reset_model:
                self.model.reset()

            state = self.env.get_state().reshape(-1)
            bar = tqdm(
                range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            trewards = 0

            for step in bar:
                if self.env.game_done:
                    break
                state = T.from_numpy(state).float()
                actions = self.model(state)

                c = Categorical(actions)
                action = c.sample()
                log_prob = c.log_prob(action)

                u = np.zeros(self.nactions)
                u[action] = 1.0
                newstate, reward = self.env.act(u)
                state = newstate.reshape(-1)
                trewards += reward
                self.episode_rewards.append(trewards)
                if train:
                    self.trainer.store_records(reward, log_prob)
                if self.model.type == "mem" and self.visual_activations:
                    u = T.cat(self.activations, dim=0).reshape(-1)
                    self.neural_image_values = u.detach().numpy()
                    self.activations = []
                    if _ % 10 == 0 and step/steps == 0:
                        self.update_weights()
                        self.neural_weights = self.weights
                        self.weight_change = True
                    if self.model.type == "mem" and type(self.model.hidden_vectors) != type(None):
                        self.hidden_state = self.model.hidden_vectors

                bar.set_description(f"Episode: {_:4} Rewards : {trewards}")
                if train:
                    self.env.step()
                else:
                    self.env.step(speed=0)
                self.event_handler()
                self.window.fill((0, 0, 0))
                if self.visual_activations and (not train or _ % render_once == render_once-1):
                    self.draw_neural_image()
                self.window.blit(self.env.win, (0, 0))

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


class MultiAgentRunner:
    def __init__(self, models, environment, trainers, nactions=6, log_message=None, visual_activations=False):
        self.env = environment
        self.models = models
        self.trainers = trainers
        self.nactions = nactions
        self.recorders = []
        for m in range(len(self.models)):
            self.recorders.append(RLGraph())
            self.recorders[m].log_message = log_message
        self.activations = []
        self.weights = []
        self.visual_activations = visual_activations
        self.current_max_reward = 0

        # TODO add the hook inside the model itself

        # if visual_activations:
        #     def hook_fn(m,i,o):
        #         if type(o) == type((1,)):
        #             for u in o:
        #                 activation.append(u.reshape(-1))
        #         else:
        #             activation.append(o.reshape(-1))

        #     for model in self.models:
        #         activation = []
        #         for n,l in model._modules.items():
        #             l.register_forward_hook(hook_fn)

    def update_weights(self):
        self.weights = []
        for model in self.models:
            wghts = []
            for param in model.parameters():
                wghts.append(T.tensor(param).clone().detach().reshape(-1))
            wghts = T.cat(wghts, dim=0).numpy()
            print("weights shape", wghts.shape)


class MultiAgentSimulator(MultiAgentRunner):
    def __init__(self, models, environment, trainers, nactions=6, log_message=None, visual_activations=False):
        super().__init__(models, environment, trainers, nactions=nactions,
                         log_message=log_message, visual_activations=visual_activations)

        py.init()
        extra_width = 300
        env_w = environment.win.get_width()
        env_h = environment.win.get_height()

        if visual_activations:
            self.w = 50 + env_w + extra_width
        else:
            self.w = 50 + env_w

        self.h = 50 + env_h
        self.font = py.font.SysFont("times", 10)
        self.window = py.display.set_mode((self.w, self.h), py.DOUBLEBUF, 32)
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

    def render_text(self, surf, text, pos):
        text = self.font.render(text, True, (200, 200, 200))
        trect = text.get_rect()
        trect.topleft = pos
        surf.blit(text, trect)
        return surf

    def create_pack(self, arr):
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:int(s*s)].reshape((s, s))
        col_splits = np.random.randint(3, 6)
        columns = []
        a_split = s/col_splits
        total = s
        starting = 0
        for i in range(col_splits-1):
            end = np.random.randint(a_split-20, a_split)
            columns.append([starting, starting+end])
            starting += end
        columns.append([starting, s])

        final_split = []
        row_splits = 5
        for column in columns:
            starting = 0
            a_split = s/row_splits
            rows = []
            for i in range(row_splits-1):
                e = np.random.randint(a_split-30, a_split+10)
                rows.append([starting, starting+e, column[0], column[1]])
                starting += e
            rows.append([starting, s, column[0], column[1]])
            final_split.append(rows)
        surface_size = [[s, s], [row_splits, col_splits]]
        return final_split, surface_size

    def weight_from_pack(self):
        arr = self.neural_weights
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:(s*s)].reshape((s, s))

        out = []
        for col in self.neural_layout:
            rows = []
            for r in col:
                rows.append(arr[r[0]:r[1], r[2]:r[3]])
            out.append(rows)
        return out

    def calculate_color(self, av, maxi, mini, poslimit, neglimit):
        cg = 0
        cr = 0

        if av < 0:
            cr = int(((av)/mini)*neglimit)
            cg = 0
            # cg = poslimit- (cr)
        else:
            cg = int((av/maxi)*poslimit)
            # cr = neglimit - cg
            cr = 0
        return cr, cg

    def calculate_limits(self, maxi, mini):
        if abs(maxi) > abs(mini):
            poslimit = 180
            neglimit = int(255.0*(abs(mini) / abs(maxi)))
        else:
            neglimit = 255
            poslimit = int(255.0*(abs(maxi) / abs(mini)))
        return poslimit, neglimit

    def surf_neural_activation(self):
        """ Returns a Drawn surface for neural activation"""
        assert type(self.neural_image_values) == type(
            []), "neural_image_values should be in list"
        act_surfaces = []
        for neural_image in self.neural_image_values:
            varr = neural_image
            varr = varr.reshape(-1)
            l = int(np.sqrt(varr.shape[0]))
            varr = varr[-l*l:].reshape((l, l))
            sz = 4
            activ_surf = py.Surface((l*sz+5, l*sz+5))
            maxi = np.max(varr)
            mini = np.min(varr)
            poslimit, neglimit = self.calculate_limits(maxi, mini)
            for r in range(l):
                for c in range(l):
                    av = varr[r][c]
                    cr, cg = self.calculate_color(
                        av, maxi, mini, poslimit, neglimit)
                    colorvalue = (abs(cr), abs(cg), max(abs(cr), abs(cg)))
                    py.draw.rect(activ_surf, colorvalue,
                                 (10+c*sz, 10+r*sz, sz, sz))
            act_surfaces.append(activ_surf)

        # n_surfaces = len(act_surfaces)
        # w,h = act_surfaces[0].get_width(), act_surfaces[0].get_height()
        # panel_width = 250
        # rows_count = n_surfaces*(w+5)//panel_width
        # if rows_count==0: rows_count = 1

        n_surfaces = len(act_surfaces)
        w, h = act_surfaces[0].get_width(), act_surfaces[0].get_height()
        panel_width = 250
        rows_count = (w+5)*n_surfaces//panel_width
        if rows_count == 0:
            rows_count = 1
        col_count = (w+5)*n_surfaces//rows_count
        panel = py.Surface(((w+5)*(col_count), (h+5)*rows_count))
        r, c = 0, 0
        for i in range(len(act_surfaces)):
            panel.blit(act_surfaces[i], (r, c))
            r += w+5
            if r >= panel_width:
                r = 0
                c += (h+5)
        return panel, (panel.get_width(), panel.get_height())
        # return activ_surf,(activ_surf.get_width(),activ_surf.get_height())

    def surf_hidden_activation(self):
        if type(self.hidden_state) != type(None):
            maxi = np.max(self.hidden_state)
            mini = np.min(self.hidden_state)
            c, r = self.hidden_state.shape
            sz = 2
            state_surface = py.Surface((c*sz, r*sz))
            swidth = state_surface.get_width()
            poslimit, neglimit = self.calculate_limits(maxi, mini)
            for i in range(c):
                for j in range(r):
                    av = self.hidden_state[i][j]
                    cr, cg = self.calculate_color(
                        av, maxi, mini, poslimit, neglimit)
                    colorvalue = (abs(cr), abs(cg), max(abs(cr), abs(cg)))
                    py.draw.rect(state_surface, colorvalue,
                                 (i*sz, j*sz, sz, sz))
            self.hidden_state_surfaces.append(state_surface)
            if len(self.hidden_state_surfaces) > 330/swidth-3*swidth:
                self.hidden_state_surfaces = self.hidden_state_surfaces[1:]
            l = len(self.hidden_state_surfaces)
            surf_w = self.hidden_state_surfaces[0].get_width()
            surf_h = self.hidden_state_surfaces[0].get_height()
            full_surf = py.Surface((20+l*surf_w, 20+surf_h))
            for i in range(l):
                full_surf.blit(
                    self.hidden_state_surfaces[i], (10+i*surf_w, 10))

            return full_surf, (full_surf.get_width(), full_surf.get_height())

    def surf_neural_weights(self):
        if self.neural_weights is not None and self.weight_change:
            self.weight_change = False
            if self.neural_layout == None:
                self.neural_layout, self.neural_layout_size = self.create_pack(
                    self.neural_weights)
            pix_size, gaps = self.neural_layout_size
            gap_size = 1
            sz = 1
            p_x = pix_size[0]*sz + gaps[0]*gap_size + 20
            p_y = pix_size[1]*sz + gaps[1]*gap_size + 20
            neural_weight_surface = py.Surface((p_x, p_y))
            weights = self.weight_from_pack()
            startx = 10
            for col in weights:
                starty = 10
                for weight in col:
                    r, c = weight.shape
                    maxi = np.max(weight)
                    mini = np.min(weight)
                    poslimit, neglimit = self.calculate_limits(maxi, mini)
                    for i in range(r):
                        for j in range(c):
                            av = weight[i][j]
                            cr, cg = self.calculate_color(
                                av, maxi, mini, poslimit, neglimit)
                            colorvalue = (abs(cr), abs(
                                cg), max(abs(cr), abs(cg)))
                            py.draw.rect(neural_weight_surface, colorvalue,
                                         (startx+j*sz, starty+i*sz, sz, sz))
                    starty += r*sz + gap_size
                startx += c*sz + gap_size
            print("returning ", neural_weight_surface)
            sizefor = [neural_weight_surface.get_width(
            ), neural_weight_surface.get_height()]
            print("sie", sizefor)
            self.neural_weight_surface = [neural_weight_surface, sizefor]
            return neural_weight_surface, sizefor
        else:
            return self.neural_weight_surface

    def surf_create_graph_multi(self, values, x_label, y_value, width, colors):
        wid, hei = width, 150
        surf_size = (wid, hei)
        surf = py.Surface(surf_size)
        length = len(values[0])
        text = self.font.render(x_label, True, (200, 200, 200))
        mark_text = self.font.render(y_value, True, (200, 200, 200))
        maxi = np.max(values)
        mini = np.min(values)
        polys = []

        for val, col in zip(values, colors):
            poly = []
            # print(val)
            for i in range(0, length, max(1, int(length/wid-10))):
                v = val[i]
                x = (i/length)*(wid-10)
                if (maxi-mini) != 0:
                    y = (hei - text.get_height()) - ((hei-10)/(maxi - mini))*v
                else:
                    y = (hei - text.get_height())
                poly.append((x, y))
            line_poly = list(poly)
            polys.append(poly)
            # if len(poly)> 1:
            #     py.draw.lines(surf,col,False,poly,3)

        polys.sort(key=lambda x: (x[-1]))

        for poly, col in zip(polys, colors):
            poly.append((poly[-1][0], poly[0][1]))
            if len(poly) > 2:
                r, g, b = col
                dv = 50
                p_col = [r - min(dv, r), g-min(dv, g), b-min(dv, b)]
                py.draw.polygon(surf, p_col, poly, 0)
            if len(poly) > 1:
                py.draw.lines(surf, col, False, poly, 3)

        trect = text.get_rect()
        trect.topright = (text.get_width(), hei-text.get_height())
        surf.blit(text, trect)
        trect = mark_text.get_rect()
        trect.topleft = (0, 0)
        surf.blit(mark_text, trect)
        return surf

    def surf_create_graph(self, values, x_label, y_value, width):
        wid, hei = width, 150
        surf_size = (wid, hei)
        surf = py.Surface(surf_size)
        length = len(values)
        text = self.font.render(x_label, True, (200, 200, 200))
        # mark_text = self.font.render("R: "+str(values[-1]), True, (200,200,200))
        mark_text = self.font.render(y_value, True, (200, 200, 200))
        maxi = max(values)
        mini = min(values)
        poly = []
        for i in range(0, length, max(1, int(length/(wid-10)))):
            v = values[i]
            x = (i/length)*(wid-10)
            if (maxi-mini) != 0:
                y = (hei-text.get_height()) - ((hei-10)/(maxi - mini))*v
            else:
                y = (hei-text.get_height())
            poly.append((x, y))
        line_poly = list(poly)
        poly.append((poly[-1][0], poly[0][1]))
        poly.append(poly[0])
        py.draw.polygon(surf, (200, 0, 200), poly, 0)
        if len(line_poly) > 1:
            py.draw.lines(surf, (0, 255, 255), False, line_poly, 3)
        # surf = self.render_text(surf,"steps",(self.wid/2))
        trect = text.get_rect()
        trect.topright = (text.get_width(), hei-text.get_height())
        surf.blit(text, trect)
        trect = mark_text.get_rect()
        trect.topleft = (0, 0)
        surf.blit(mark_text, trect)
        return surf

    def draw_episode_reward(self, w=150):
        y_value = "R: " + str(self.episode_rewards[-1])
        surf = self.surf_create_graph(
            self.episode_rewards, "steps", y_value, width=w)
        return surf

    def draw_episode_rewards(self, w=250):
        # print(np.array(self.episode_rewards).shape)
        y_value = "R: " + str(np.max(self.episode_rewards[-1]))
        f = 255
        colors = []
        for agent in self.env.agents:
            colors.append(agent.color)
        reshaped = np.array(self.episode_rewards).squeeze(1).T
        surf = self.surf_create_graph_multi(
            reshaped, "steps", y_value, width=w, colors=colors)
        return surf

    def draw_neural_image(self):
        panel = py.Surface((500, self.window.get_height()))
        surf_activation, asize = self.surf_neural_activation()
        # surf_weights,wsize = self.surf_neural_weights()
        # surf_hidden,hsize = self.surf_hidden_activation()
        panel.blit(surf_activation, (0, 0))
        # if surf_weights is not None:
        #     panel.blit(surf_weights,(asize[0],0))
        # panel.blit(surf_hidden,(0,max(asize[1],wsize[1])))
        surf_graph_1 = self.draw_episode_rewards(
            w=(self.window.get_width() - self.env.win.get_width()-10))
        # panel.blit(surf_graph_1, (0,max(asize[1],wsize[1])+ surf_hidden.get_height()+5) )
        print("size", asize[1])
        panel.blit(surf_graph_1, (0, asize[1] + 10))
        self.window.blit(panel, (self.env.win.get_width()+10, 10))

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()

    def run(self, episodes, steps, train=False, render_once=1e10, saveonce=10):
        print(f"Episodes     : {episodes}   ")
        print(f"Steps        : {steps}      ")
        print(f"Training     : {train}      ")
        print(f"Render Once  : {render_once}")
        print(f"Save Once    : {saveonce}   ")

        if train:
            assert self.recorders[0].log_message is not None, "log_message is necessary during training, Instantiate Runner with log message"

        reset_model = False
        if hasattr(self.models[0], "type") and self.models[0].type == "mem":
            print("Recurrent Model")
            reset_model = True
        self.env.display_neural_image = self.visual_activations

        for _ in range(episodes):
            self.episode_rewards = []
            self.env.reset()
            self.env.enable_draw = True if not train or _ % render_once == render_once-1 else False

            if reset_model:
                for model in self.models:
                    model.reset()

            states = []
            for i in range(len(self.env.agents)):
                state = self.env.get_state(i)
                states.append(state)

            bar = tqdm(
                range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            trewards = np.zeros((1, len(self.models)))
            for step in bar:
                # print("-----------------")
                # print("Logs probs 0",len(self.trainers[0].log_probs),len(self.trainers[1].log_probs))
                if self.env.game_done:
                    # print("Quitting game")
                    break
                action_vecs = []
                log_probs = []

                for ind in range(len(states)):
                    state = T.from_numpy(states[ind]).float()
                    actions = self.models[ind](state)
                    c = Categorical(actions)
                    action = c.sample()
                    log_prob = c.log_prob(action)
                    log_probs.append(log_prob)
                    u = np.zeros(self.nactions)
                    u[action] = 1.0
                    action_vecs.append(u)

                newstates, rewards = self.env.act(action_vecs)
                states = newstates

                # all_dead = True
                # if train:
                #     for j in range(len(newstates)):
                #         if not self.env.agents[j].dead:
                #             self.trainers[j].store_records(rewards[j],log_probs[j])
                if train:
                    for j in range(len(newstates)):
                        self.trainers[j].store_records(
                            rewards[j], log_probs[j])
                        # all_dead = False
                # print("Logs probs",len(self.trainers[0].log_probs),len(self.trainers[1].log_probs))
                trewards += np.array(rewards)
                states = newstates
                # trewards += rewards
                self.episode_rewards.append(trewards.tolist())

                if self.visual_activations:
                    self.neural_image_values = []
                    for model in self.models:
                        activations = T.cat(
                            model.activations, dim=0).reshape(-1)
                        self.neural_image_values.append(
                            activations.detach().numpy())

                    # u = T.cat(self.activations,dim=0).reshape(-1)
                    # self.neural_image_values = u.detach().numpy()
                    # self.activations = []
                    # if _ % 10 == 0 and step/steps == 0:
                    #     self.update_weights()
                    #     self.neural_weights = self.weights
                    #     self.weight_change = True
                    # if type(self.model.hidden_vectors) != type(None):
                    #     self.hidden_state = self.model.hidden_vectors
                reward_string = ""
                for uu in range(len(trewards[0])):
                    reward_string += str(trewards[0][uu])
                    reward_string += " "
                bar.set_description(
                    f"Episode: {_:4} Rewards : {reward_string} ")
                # print("Logs probs 1",len(self.trainers[0].log_probs),len(self.trainers[1].log_probs))
                if train:
                    self.env.step()
                else:
                    self.env.step(speed=0)

                self.event_handler()
                self.window.fill((0, 0, 0))

                if self.visual_activations and (not train or _ % render_once == render_once-1):
                    self.draw_neural_image()

                if not train or _ % render_once == render_once-1:
                    self.window.blit(self.env.win, (0, 0))

                # print("Logs probs 2",len(self.trainers[0].log_probs),len(self.trainers[1].log_probs))
            # print("Calling Updates")
            if train:
                for t in range(len(self.trainers)):
                    self.trainers[t].update()
                    self.trainers[t].clear_memory()
                    self.recorders[t].newdata(trewards[0, t])
                    self.recorders[t].save()
                    self.recorders[t].plot()
                    if _ % saveonce == saveonce - 1 and trewards[0, t] >= self.recorders[t].final_reward:
                        self.recorders[t].save_model(self.models[t])

                # self.trainer.update()
                # self.trainer.clear_memory()
                # self.recorder.newdata(trewards)
                # if _ % saveonce == saveonce-1:
                #     self.recorder.save()
                #     self.recorder.plot()

        #         if _ % saveonce == saveonce-1 and self.recorder.final_reward >= self.current_max_reward:
        #             self.recorder.save_model(self.model)
        #             self.current_max_reward = self.recorder.final_reward
        print("******* Run Complete *******")


class Container:
    def __init__(self, environment, models):
        """args: environment, model"""

        self.env = environment
        self.models = models
        self.hidden_states = None
        self.trewards = 0
        self.trainers = []
        for m in self.models:
            self.trainers.append(Trainer(m, learning_rate=0.01))
        self.neural_activation = []

    def reset(self):
        """Resets the container and zeros the hiddenstate of the model"""
        self.env.reset()
        self.hidden_states = [T.zeros((1, 1, 64)) for _ in range(len(self.models))] 
        self.states, self.agent_states = [],[]
        for i in range(len(self.models)):
            s,ags = self.env.get_state(i).reshape(-1), self.env.get_agent_state(i).reshape(-1)
            self.states.append(s)
            self.agent_states.append(ags)

        self.trewards = np.zeros((1, len(self.models)))
        self.episode_rewards = []

    def update(self):
        """Train the models with the collected trajectory data for the
        given environment"""
        for t in self.trainers:
            t.update()
            t.clear_memory()

    def step(self, train=True):
        """Performs a single step over the environments with all the models"""
        action_vecs = []
        log_probs = []
        for i in range(len(self.models)):
            state = T.from_numpy(self.states[i]).float()
            agent_state = T.from_numpy(self.agent_states[i]).float()
            if hasattr(self.models[i], "type") and self.models[i].type == "mem":
                self.models[i].hidden = self.hidden_states[i]

            self.models[i].hidden = self.hidden_states[i]
            actions = self.models[i](state, agent_state)
            c = Categorical(actions)
            action = c.sample()
            log_prob = c.log_prob(action)
            log_probs.append(log_prob)
            u = np.zeros(self.models[i].output_size)
            u[action] = 1.0
            action_vecs.append(u)

        combined, rewards = self.env.act(action_vecs)
        self.trewards += np.array(rewards)
        self.episode_rewards.append(self.trewards.tolist())
        self.states,self.agent_statess = combined[:,0],combined[:,1]
        if train:
            for j in range(len(self.models)):
                self.trainers[j].store_records(
                    rewards[j],log_probs[j]
                )
        self.env.step()

class MultiEnvironmentSimulator(MultiAgentSimulator):
    def __init__(self, models, environments, nactions=6, log_message=None, visual_activations=False):
        super().__init__(models, environments[0], None, nactions=nactions,
                         log_message=log_message, visual_activations=visual_activations)
        py.init()
        nenvs = len(environments)
        self.containers = [Container(x, models) for x in environments]
        self.environments = environments  # StateGatherers with different states

    def run(self, episodes, steps, train=False, render_once=1e10, saveonce=10):
        if train:
            assert self.recorders[0].log_message is not None, "log_message is necessary during training, Instantiate Runner with log message"

        for c in self.containers:
            c.env.display_neural_image = self.visual_activations

        for _ in range(episodes):
            for c in self.containers:
                c.reset()
            # for _ in tqdm_steps:
            #     for c in self.containers:
            #         c.step()
            #         py.display.update()

            containers_trainings = [] 
            for c in self.containers:
                tqdm_steps = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
                frames = []
                for _ in tqdm_steps:
                    c.step()
                    # self.window.blit(c.env.win, p)
                    frames.append(c.env.win.copy())
                    tqdm_steps.set_description(f"Episode: {_:4}")
                containers_trainings.append(frames)
                # tqdm_steps.set_description(f"Episode: {_:4} Rewards : {trewards}")
                c.update()
            initial = [10,10]
            positions = [initial]
            for i in range(len(containers_trainings)-1):
                w,h = containers_trainings[i][0].get_width(),containers_trainings[i][0].get_height()
                lx,ly = positions[-1]
                if lx + w > 150:
                    positions.append([10,ly+h])
                else: 
                    positions.append([lx+w+10,ly])
            print(positions)


            for i in range(steps):
                for cont,pos in zip(containers_trainings,positions):
                    self.window.blit(cont[i],pos)
                py.display.update()
                # for c, p in zip(self.containers, place_pos):
                #     c.update()
                