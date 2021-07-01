import pygame as py
import typing
import numpy as np
from icecream import ic

ENABLE_DRAW = True



class Agent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.picked = 0  # 0 means no values picked
        self.state = None
        self.trail_positions = []
        self.energy = 100
        self.color = None
        self.dead = False
        self.processor_locations = []

    def reset(self):
        self.x, self.y = 0, 0
        self.picked = 0
        self.processor_locations = []
        self.dead = False

    def act(self, action_vector, g_res, g_stor, g_processor):
        """ Rewarding systems is defined in this function """
        # if self.energy<1:
        # self.dead = True
        # return 0
        left, up, right, down = action_vector[:4]
        v, h = g_res.shape
        pick = action_vector[4]
        # drop = action_vector[5]
        build_proc = action_vector[5]
        max_amount = 20
        reward = 0
        cx, cy = self.x, self.y
        self.trail_positions.append([cx, cy])
        picks = 0
        VOLCANO = -3 

        if len(self.trail_positions) > 7:
            self.trail_positions.pop(0)

        if left and self.y > 0:
            self.y -= 1

        elif right and self.y < h-1:
            self.y += 1

        elif up and self.x > 0:
            self.x -= 1

        elif down and self.x < v-1:
            self.x += 1

        elif pick:
            if g_res[cx][cy] == 1:
                self.picked += g_res[cx][cy]
                g_res[cx][cy] = 0
                reward += 1
                picks = 1
            elif g_res[cx][cy] == 2:
                g_res[cx][cy] = 0
                reward += 3

        elif build_proc > 0 and g_processor[cx][cy] == 0 and self.picked > 4:
            # construction of storage is possible if agent picked more than 3 resource items
            self.picked -= 4
            g_processor[cx][cy] = 1
            reward += 1
            self.processor_locations.append([cx, cy])
        return reward, picks

        
class StateAgent(Agent):
    #Inheritance in Action
    def __init__(self):
        super().__init__()
        self.collecteds = 0
        self.items = 0
        self.processed_items = 0
        self.rewards = 0
        self.circular_world = False
    
    def act(self,action_vector, g_res, g_processor):
        left, up, right, down = action_vector[:4]
        v, h = g_res.shape
        pick = action_vector[4]
        # drop = action_vector[5]
        build_proc = action_vector[5]
        max_amount = 20
        reward = 0
        cx, cy = self.x, self.y
        self.trail_positions.append([cx, cy])
        picks = 0
        VOLCANO = -3

        if len(self.trail_positions) > 7:
            self.trail_positions.pop(0)

        if self.dead:
            return 0, 0


        if g_res[cx][cy] == VOLCANO:
            self.dead = True
            return -1,0


        if self.circular_world:
            if left:
                if self.y>0:
                    self.y -= 1
                else:         
                    self.y = h-1;
    
    
            elif right: 
                if self.y< h-1:
                    self.y += 1
                else:
                    self.y = 0
    
            elif up :
                if self.x > 0:
                    self.x -= 1
                else:
                    self.x = v-1
    
            elif down :
                if self.x < v-1:
                    self.x += 1
                else:
                    self.x = 0
            elif pick:
                if g_res[cx][cy] == 1:
                    self.picked += g_res[cx][cy]
                    self.collecteds+=1 
                    g_res[cx][cy] = 0
                    reward += 1
                    picks = 1
                elif g_res[cx][cy] == 2:
                    self.items += 1
                    g_res[cx][cy] = 0
                    reward += 3

            elif build_proc > 0 and g_processor[cx][cy] == 0 and self.picked > 4:
                # construction of storage is possible if agent picked more than 3 resource items
                self.picked -= 4
                g_processor[cx][cy] = 1
                reward += 1
                self.processor_locations.append([cx, cy])

        else:
            if left and self.y>0:
                self.y -= 1
            elif right and self.y < h-1:
                self.y+=1
            elif up and self.x>0:
                self.x -=1
            elif down and self.x<v-1:
                self.x += 1

            elif pick:
                if g_res[cx][cy] == 1:
                    self.picked += g_res[cx][cy]
                    self.collecteds+=1 
                    g_res[cx][cy] = 0
                    reward += 1
                    picks = 1
                elif g_res[cx][cy] == 2:
                    self.items += 1
                    g_res[cx][cy] = 0
                    reward += 3

            elif build_proc > 0 and g_processor[cx][cy] == 0 and self.picked > 4:
                # construction of storage is possible if agent picked more than 3 resource items
                self.picked -= 4
                g_processor[cx][cy] = 1
                reward += 1
                self.processor_locations.append([cx, cy])
        self.rewards += reward
        return reward, picks


    
class Gatherer:
    def __init__(self, gr=10, gc=10, vis=7, nagents=1):
        py.init()
        self.box_size = 20
        self.font = py.font.SysFont("times", 20)
        self.vissize = vis

        self.w = 50 + gc*self.box_size + 50
        self.h = 50 + gr*self.box_size + 50
        self.win = py.Surface((self.w, self.h), py.DOUBLEBUF, 32)
        self.start_position = [0, 0]

        self.nagents = nagents

        self.grid_resource = np.zeros((gr, gc))
        self.grid_storages = np.zeros((gr, gc))
        self.grid_stored = np.zeros((gr, gc))
        self.grid_agents = np.zeros((nagents, gr, gc))
        self.agents = []
        all_colors = [[255, 255, 0], [28, 230, 255], [255, 52, 255], [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89], [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172], [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135], [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0], [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128], [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160], [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0], [
            221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153], [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111], [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191], [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9], [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255], [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102], [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160], [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76]]
        for a in range(nagents):
            self.agents.append(Agent())
            agent = self.agents[a]
            agent.color = all_colors[a]
            self.grid_agents[a][agent.x][agent.y]

        self.timer = 0
        self.res = 0
        self.items = 0
        self.collected = 0
        self.processors = 0

        self.agent_energy = 30
        self.enable_draw = ENABLE_DRAW
        self.processors = []
        self.current_step = 0
        self.total_rewards = 0
        self.trail_positions = []
        self.game_done = False
        # initializing agents

    def num_to_color(self, a):
        a = a[1:]
        values = []
        for i in range(0, 6, 2):
            v = a[i:i+2]
            values.append(eval("0x"+v))
        return values

    def render_text(self, text, pos):
        text = self.font.render(text, True, (200, 200, 200))
        trect = text.get_rect()
        trect.topleft = pos
        self.win.blit(text, trect)
        return text.get_width(),text.get_height()
    
    def get_state(self, id):
        agent = self.agents[id]
        x, y = agent.x, agent.y
        g_res = np.ones((1, self.vissize, self.vissize))*- \
            1  # TODO try with np.zeros
        g_stor_tanks = np.ones((1, self.vissize, self.vissize))*-1
        g_stored = np.ones((1, self.vissize, self.vissize))*-1
        g_agents = np.ones((1, self.vissize, self.vissize))*-1
        v, h = self.grid_resource.shape

        r = self.vissize - 5
        s = -2 - r//2
        r -= r//2
        e = 3 + r

        for i in range(s, e):
            for j in range(s, e):
                if(0 <= x+i < v and 0 <= y+j < h):
                    g_res[0][i-s, j-s] = self.grid_resource[x+i, y+j]
                    g_stor_tanks[0][i-s, j-s] = self.grid_storages[x+i, y+j]
                    g_stored[0][i-s, j-s] = self.grid_stored[x+i, y+j]

        # for i in range(s,e):
        #     for j in range(s,e):
        #         if( 0 <= x+i < v and 0 <= y+j < h):

        for a in self.agents:
            if a != agent:
                ox, oy = a.x, a.y
                xx, yy = agent.x, agent.y
                if xx-s <= ox < xx+e and yy-s <= oy < yy+e:
                    px = ox - xx if ox < xx else (-s+e)//2 + xx - ox
                    py = oy - yy if oy < yy else (-s + e)//2 + yy - oy
                    g_agents[0][px][py] = 1

        return np.concatenate([
            g_res, g_stor_tanks, g_stored, g_agents
        ], axis=0).reshape(-1)


    def act(self, action_vecs):
        """action_vec should is 2 dimension always"""
        rewards = []
        states = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            r, picks = agent.act(
                action_vecs[i], self.grid_resource, self.grid_storages, self.grid_stored)
            self.res -= picks
            rewards.append(r)
        for i in range(len(self.agents)):
            # unmatrixed shape of state appened to states
            states.append(self.get_state(i))
        return states, rewards

    def reset(self, hard=False):
        if hard:
            np.random.seed(5)
        v, h = self.grid_storages.shape
        self.game_done = False
        self.res = 0
        self.grid_resource = np.zeros((v, h))
        self.grid_storages = np.zeros((v, h))
        self.grid_stored = np.zeros((v, h))
        self.grid_agents = np.zeros((self.nagents, v, h))

        for agent in self.agents:
            agent.reset()
        self.current_step = 0
        self.processors = []

    def draw_box(self, i, j, color):
        v, h = self.grid_resource.shape
        r = 1
        bs = self.box_size  # box size
        sx, sy = self.start_position
        surf = py.Surface((bs, bs), py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.rect(surf, color, surf.get_rect())
        self.win.blit(surf, (sx+j*bs, sy+i*bs, bs, bs), special_flags=py.BLEND_RGBA_ADD)
#        py.draw.rect(self.win,py.Color(color),(sx+j*bs,sy+i*bs,bs,bs))

    def draw_triangle(self,i,j,color):
        v, h = self.grid_resource.shape
        a = np.array([[0.5,0],[0,1],[1,1],[0.5,0]])
        a = a*self.box_size
        r = 1
        bs = self.box_size  # box size
        sx, sy = self.start_position
        surf = py.Surface((bs, bs), py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.polygon(surf,color,a.tolist(),0)
        self.win.blit(surf, (sx+j*bs, sy+i*bs, bs, bs), special_flags = py.BLEND_RGBA_ADD)

    def draw_circle(self,i,j,color):
        v, h = self.grid_resource.shape
        r = 1
        bs = self.box_size  # box size
        sx, sy = self.start_position
        surf = py.Surface((bs, bs), py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.circle(surf,color,(self.box_size/2,self.box_size/2),self.box_size/2,self.box_size//2)
        self.win.blit(surf, (sx+j*bs, sy+i*bs, bs, bs), special_flags = py.BLEND_RGBA_ADD)


    def draw_polygon(self,i,j,color,points):
        v, h = self.grid_resource.shape
        r = 1
        bs = self.box_size  # box size
        sx, sy = self.start_position
        surf = py.Surface((bs, bs), py.SRCALPHA)
        surf = surf.convert_alpha()
        # py.draw.rect(surf, color, surf.get_rect())
        py.draw.polygon(surf,color,points,0)
        self.win.blit(surf, (sx+j*bs, sy+i*bs, bs, bs), special_flags = py.BLEND_RGBA_ADD)

    def draw_grid(self):
        # horizontal line
        v, h = self.grid_resource.shape
        r = 1
        bs = self.box_size  # box size
        sx, sy = self.start_position
        color = (100, 0, 0)
        color = (60, 0, 60)

        for i in range(v+1):
            py.draw.line(self.win, color, (sx, sy+i*bs), (sx+h*bs, sy+i*bs))

        for i in range(h+1):
            py.draw.line(self.win, color, (sx+i*bs, sy), (sx+i*bs, sy+v*bs))

    def draw_visibility(self):
        for agent in self.agents:
            vv, vh = self.vissize, self.vissize
            ax, ay = agent.x, agent.y
            lx, ly = self.grid_resource.shape
            for i in range(vv):
                for j in range(vh):
                    if 0 <= i+ax-vv//2 < lx and 0 <= j+ay-vh//2 < ly:
                        self.draw_box(i+ax-vv//2, j+ay-vh//2, [50, 50, 50])


    def draw_items(self):
        v, h = self.grid_resource.shape
        VOLCANO = -3
        # v=20 h = 30
        for i in range(v):
            for j in range(h):
                # if self.grid_resource[i][j] == 1:
                #     self.draw_box(i, j, (0, 255, 0))
                # elif self.grid_resource[i][j] == 2:
                #     self.draw_box(i, j, (0, 255, 255))
                # if self.grid_stored[i][j] > 0:
                #     self.draw_box(i, j, (255, 0, 0))
                if self.grid_resource[i][j] == 1:
                    self.draw_triangle(i,j,(0,255,0))
                elif self.grid_resource[i][j] == 2:
                    self.draw_triangle(i,j,(0,255,255))
                elif self.grid_resource[i][j] == VOLCANO: #volcano
                    self.draw_box(i,j,(255,165,0))

                if self.grid_stored[i][j] > 0:
                    self.draw_circle(i,j,(255,0,0))


    def draw_trail(self):
        # for agent in self.agents:
        #     for j in range(len(agent.trail_positions)):
        #         x, y = agent.trail_positions[j]
        #         c = int(30 + (1.2)**j)
        #         self.draw_box(x, y, (c, c, c))
        
        for agent in self.agents:
            for j in range(len(agent.trail_positions)):
                x,y = agent.trail_positions[len(agent.trail_positions)-j-1]
                c = 200-j*20 
                self.draw_box(x, y, (0, c, c))

    def draw(self):
        self.win.fill((0, 0, 0))
        self.draw_grid()
        # self.draw_trail()
        self.draw_items()
        # self.render_text(f"Collected :{self.collected:3}",(0,0))
        # self.render_text(f"Items     :{self.items:3}",(0,20))
        self.render_text(f"S:{self.current_step:4}", (10, 10))
        self.render_text(f"R:{self.total_rewards:3}",(self.win.get_width()-50,20))
        # self.render_text(f"Energy :{self.agent_energy:3}",(250,20))
        # self.get_state()
        # agent_colors = [(0,0,200),(200,0,200),(255,0,0),(255,255,0)]
        # for agent,color in zip(self.agents,agent_colors):
        #     self.draw_box(agent.x,agent.y,color)
        for agent in self.agents:
            self.draw_box(agent.x, agent.y, agent.color)
        # self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        self.draw_visibility()

    def spawn_resources(self):
        # initializing foods
        self.timer += 1
        v, h = self.grid_resource.shape
        spawn_limit = 100
        processor_locations = []
        for agent in self.agents:
            processor_locations.extend(agent.processor_locations)

        if self.timer % 2 == 0 and self.res < spawn_limit:
            while True:
                r = np.random.randint(0, v)
                c = np.random.randint(0, h)
                if self.grid_resource[r][c] == 0:
                    self.grid_resource[r][c] = 1
                    self.res += 1
                    break

        if self.timer % 10 == 0:
            # for i in range(v):
            #     for j in range(h):
            #         if self.grid_stored[i][j] == 1:
            for loc in processor_locations:
                px, py = loc
                for agent in self.agents:
                    req = 3
                    if agent.picked > req:
                        if px-1 > -1 and self.grid_resource[px-1][py] == 0:
                            self.grid_resource[px-1][py] = 2
                            agent.picked -= req
                        elif px+1 < v and self.grid_resource[px+1][py] == 0:
                            self.grid_resource[px+1][py] = 2
                            agent.picked -= req
                        elif py - 1 > -1 and self.grid_resource[px][py-1] == 0:
                            self.grid_resource[px][py-1] = 2
                            agent.picked -= req
                        elif py+1 < h and self.grid_resource[px][py+1] == 0:
                            self.grid_resource[px][py+1] = 2
                            agent.picked -= req

    def event_handler(self, event):
        pass

    def update(self):
        self.spawn_resources()

    def step(self, speed=0):
        self.event_handler(None)
        self.update()
        self.current_step += 1
        if self.enable_draw:
            self.draw()

class GathererState(Gatherer):
    """ Uses the agent of StateAgent class. StateAgent class returns the 
        collecteds,items and rewards totally obtained
        The state of the agent need to be altered and stored in a list
        and passed to the multienvironmentsimulator 
    """
    def __init__(self,gr=10,gc=10,vis=7,nagents=1,boxsize=2,spawn_limit=10):
        super().__init__(gr,gc,vis,nagents)
        self.spawn_limit = spawn_limit
        self.font = py.font.SysFont("times", boxsize)
        self.box_size = boxsize
        self.start_position = [0, self.box_size]
        self.w = boxsize + gc*self.box_size + 10
        self.h = boxsize + gr*self.box_size + 10
        self.win = py.Surface((self.w, self.h), py.DOUBLEBUF, 32)
        print("SURFACE SHAPES :",self.w,self.h)
        self.agents = []

        self.total_rewards = [0 for i in range(len(self.agents))]
        self.bool_volcano = True if np.random.randint(2)==1 else False
        all_colors = [[255, 255, 0], [28, 230, 255], [255, 52, 255], [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89], [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172], [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135], [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0], [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128], [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160], [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0], [
            221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153], [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111], [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191], [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9], [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255], [69, 109, 117], [183, 123, 104], [122, 135, 161], [120, 141, 102], [136, 85, 120], [250, 208, 159], [255, 138, 154], [209, 87, 160], [190, 196, 89], [69, 102, 72], [0, 134, 237], [136, 111, 76]]
        self.total_rewards = []
        for a in range(nagents):
            self.total_rewards.append(0)
            self.agents.append(StateAgent())
            agent = self.agents[a]
            agent.color = all_colors[a]
            self.grid_agents[a][agent.x][agent.y]
        self.initiate_volcano()
    
    

    def initiate_volcano(self):
        if not self.bool_volcano:
            return 
        v,h = self.grid_resource.shape
        n_volcano_pits = 3
        pit_size = 2 
        VOLCANO = -3
        vpos = []
        for _ in range(n_volcano_pits):
            i,j = np.random.randint(0,v),np.random.randint(0,h)
            while True: 
                i,j = np.random.randint(0,v),np.random.randint(0,h)
                collides = False
                for pos in vpos:
                    a,b = pos
                    if i+pit_size >= a and j+pit_size >= b and j <= b and i <= a:
                        collides = True

                if collides:continue
                else: break


            vpos.append([i,j])
            for a in range(pit_size):
                for b in range(pit_size):
                    self.grid_resource[(i+a)%v][(j+b)%h] = VOLCANO

                

    def reset(self):
        super().reset()
        self.total_rewards = [0 for i in range(len(self.agents))]
        self.bool_volcano = True if np.random.randint(2)==1 else False
        self.initiate_volcano()


    def act(self,action_vecs):
        """ The action_vec should is 2 dimension always.
            Returns states -->[sourrounding info, agent collections], rewards"""
        rewards = []
        states = []
        agent_states = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            r, picks = agent.act(
                action_vecs[i], self.grid_resource, self.grid_stored)
            self.res -= picks
            rewards.append(r)
        for i in range(len(self.agents)):
            # unmatrixed shape of state appened to states
            # returning the additional information about the state of the agent
            states.append(self.get_state(i))
            agent_states .append(self.get_agent_state(i))
            self.total_rewards[i] += rewards[i]
        states = np.array(states)
        agent_states = np.array(agent_states)
        return [states,agent_states] , rewards

    def get_state(self, id):
        """ Returns only the surrounding information 
        in a matrix"""
        agent = self.agents[id]
        x, y = agent.x, agent.y
        g_res = np.ones((1, self.vissize, self.vissize))*- \
            1  # TODO try with np.zeros
        g_stor_tanks = np.ones((1, self.vissize, self.vissize))*-1
        g_stored = np.ones((1, self.vissize, self.vissize))*-1
        g_agents = np.ones((1, self.vissize, self.vissize))*-1
        v, h = self.grid_resource.shape

        r = self.vissize - 5
        s = -2 - r//2
        r -= r//2
        e = 3 + r

        for i in range(s, e):
            for j in range(s, e):
                if(0 <= x+i < v and 0 <= y+j < h):
                    g_res[0][i-s, j-s] = self.grid_resource[x+i, y+j]
                    g_stor_tanks[0][i-s, j-s] = self.grid_storages[x+i, y+j]
                    g_stored[0][i-s, j-s] = self.grid_stored[x+i, y+j]

        # for i in range(s,e):
        #     for j in range(s,e):
        #         if( 0 <= x+i < v and 0 <= y+j < h):

        for a in self.agents:
            if a != agent:
                ox, oy = a.x, a.y
                xx, yy = agent.x, agent.y
                if xx-s <= ox < xx+e and yy-s <= oy < yy+e:
                    px = ox - xx if ox < xx else (-s+e)//2 + xx - ox
                    py = oy - yy if oy < yy else (-s + e)//2 + yy - oy
                    g_agents[0][px][py] = 1

        return np.concatenate([
            g_res, g_stor_tanks, g_stored, g_agents
        ], axis=0).reshape(-1) 
    
    def get_agent_state(self,id):
        """Returns the agents current collections and rewards obtained"""
        agent = self.agents[id]
        return np.array([agent.collecteds/100,agent.items/100,agent.rewards/1000]).reshape(-1)

    def draw(self):
        self.win.fill((0, 0, 0))
        self.draw_grid()
        self.draw_trail()
        self.draw_items()

        # self.render_text(f"Collected :{self.collected:3}",(0,0))
        # self.render_text(f"Items     :{self.items:3}",(0,20))

        w,_ = self.render_text(f"S: {self.current_step:4}", (0, 0))
        self.render_text(f"R: {sum(self.total_rewards):3}",(w+5,0))

        # self.render_text(f"Rewards :{self.total_rewards:3}",(250,20))
        # self.render_text(f"Energy :{self.agent_energy:3}",(250,20))
        # self.get_state()
        # agent_colors = [(0,0,200),(200,0,200),(255,0,0),(255,255,0)]
        # for agent,color in zip(self.agents,agent_colors):
        #     self.draw_box(agent.x,agent.y,color)

        for agent in self.agents:
            self.draw_box(agent.x, agent.y, agent.color)
        # self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        self.draw_visibility()

    def spawn_resources(self):
        # initializing foods
        self.timer += 1
        v, h = self.grid_resource.shape
        processor_locations = []
        for agent in self.agents:
            processor_locations.extend(agent.processor_locations)

        if self.timer % 2 == 0 and self.res < self.spawn_limit:
            while True:
                r = np.random.randint(0, v)
                c = np.random.randint(0, h)
                if self.grid_resource[r][c] == 0:
                    self.grid_resource[r][c] = 1
                    self.res += 1
                    break

        if self.timer % 10 == 0:
            # for i in range(v):
            #     for j in range(h):
            #         if self.grid_stored[i][j] == 1:
            for loc in processor_locations:
                px, py = loc
                for agent in self.agents:
                    req = 3
                    if agent.picked > req: #checks if the agents has enough resources
                        if px-1 > -1 and self.grid_resource[px-1][py] == 0:
                            self.grid_resource[px-1][py] = 2
                            agent.picked -= req
                        elif px+1 < v and self.grid_resource[px+1][py] == 0:
                            self.grid_resource[px+1][py] = 2
                            agent.picked -= req
                        elif py - 1 > -1 and self.grid_resource[px][py-1] == 0:
                            self.grid_resource[px][py-1] = 2
                            agent.picked -= req
                        elif py+1 < h and self.grid_resource[px][py+1] == 0:
                            self.grid_resource[px][py+1] = 2
                            agent.picked -= req

