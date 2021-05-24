import pygame as py 
import typing
import numpy as np 
from icecream import ic

ENABLE_DRAW = True 

#TODO environment act and agent act how to implement this 


class Agent:
    def __init__(self):
        self.x = 0 
        self.y = 0
        self.picked = 0 # 0 means no values picked  
        self.state = None
        self.trail_positions = []

    def act(self,action_vector,g_res,g_stor,g_stored):
        """ Rewarding systems is defined in this function """
        left,up,right,down = action[:4]
        v,h = self.grid.shape
        pick = action[4]
        drop = action[5]
        build_stor = action[6]
        max_amount = 20

        cx,cy = self.x, self.y
        self.trail_positions.append([cx,cy])
        if len(self.trail_positions) > 20:
            self.trail_positions.pop(0)
        
        if left and self.y > 0:  self.agent_pos[1]>0:
            self.y -= 1
        elif right and self.y < h-1: 
            self.y += 1
        elif up and self.x > 0: 
            self.x -=1 
        elif downand self.x < v-1 : 
            self.x += 1 
        elif pick and (self.g_res[cx][cy] > 0) 
            reward = 1
            self.picked = self.g_res[cx][cy]
            self.g_res[cx][cy] = 0
            reward = 1
        elif drop and self.picked:
            if g_stor[cx][cy] != 0:
                rem = (max_amount - g_stored[cx][cy])
                if rem >= self.picked:
                    reward = int(self.picked*1.9) #storing the resources within the maxamount will give higher results other wise
                    g_stored[cx][cy] += self.picked#will reduce reward 
                    self.picked = 0
                else:
                    reward = int(rem*1.5) # getting collective will yield more reward than brining it one at once
                    g_stored[cx][cy] = max_amount
                    self.picked -= rem
            else:
                g_res[cx][cy] += self.picked 
            self.picked= 0    
        elif build_stor and self.g_stor[cx][cy] == 0 and self.picked > 3:
            #construction of storage is possible if agent picked more than 3 resource items
            self.g_stor[cx][cy] = 1
            self.reward = 1

        return reward

class Gatherer:
    def __init__(self,gr=10,gc=10,vis=7,nagents = 1):
        py.init()
        self.box_size = 20 
        self.font = py.font.SysFont("times",20)
        self.vissize = vis

        self.w = 50 + gc*self.box_size + 50
        self.h = 50 + gr*self.box_size + 50
        self.win =  py.Surface((self.w,self.h),py.DOUBLEBUF,32)
        self.start_position= [50,50]
        self.nagents  = nagents

        self.grid_resource = np.zeros((gr,gc))
        self.grid_storages = np.zeros((gr,gc))
        self.grid_stored = np.zeros((gr,gc))

        for a in range(nagents):
            self.agents.append(Agent())
            agent = self.agents[0]
            self.grid_agents[0][agent.x][agent.y]
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
        #initializing agents

    def render_text(self,text,pos):
        text = self.font.render(text,True,(200,200,200))
        trect = text.get_rect()
        trect.topleft =  pos 
        self.win.blit(text,trect)

    def get_state(self,i):
        agent = self.agents[i]
        x,y = agent.x,agent.y
        g_res= np.ones((1,self.vissize,self.vissize))*-1  #TODO try with np.zeros
        g_stor_tanks = np.ones((1,self.vissize,self.vissize))*-1
        g_stored =  np.ones((1,self.vissize,self.vissize))*-1
        g_agents = np.ones((1,self.vissize,self.vissize))*-1
        v,h = self.grid_resource.shape
        
        r = self.vissize - 5
        s = -2 - r/2
        r -= r/2
        e = 3 + r
        for i in range(s,e):
            for j in range(s,e):
                if( 0 <= x+i < v and 0 <= y+j < h):
                    g_res[0][i-s,j-s] = self.grid_resource[x+i,y+j] 
                    g_stor_tanks[0][i-s,j-s] = self.grid_storages[x+i,y+j]
                    g_stored[0][i-s,j-s] = self.grid_stored[x+i,y+j]

        for i in range(s,e):
            for j in range(s,e):
                if( 0 <= x+i < v and 0 <= y+j < h):
        for a in self.agents:
            if a != self.agents[i]:
                ox,oy = a.x,a.y
                xx,yy  = agent.x,agent.y
                if xx-s <= ox < xx+e  and yy-s <= oy < yy+e:
                    px = ox - xx if ox < xx else (-s+e)//2 + xx- ox
                    py = oy - yy if oy < yy else (-s + e)//2 + yy - oy
                    g_agents[0][px][py] = 1 

        return np.concatenate([  
            g_res,g_stor_tanks,g_stored,g_agents
        ],axis=0).reshape(-1)

    def act(self,action_vecs):
        """action_vec should is 2 dimension always"""
        rewards = []
        states = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            r = agent.act(action_vecs[i],self.grid_resource,self.grid_storages,self.grid_stored )
            rewards.append(r)
        for i in range(len(self.agents)) :
            states.append(self.get_state(i))  # unmatrixed shape of state appened to states
        return states, rewards

    def reset(self,hard=False):
        if hard:
            np.random.seed(5)
        v,h = self.grid.shape
        self.agent_energy = 400
        self.game_done = False
        self.grid = np.zeros((v,h))
        self.agent_pos = np.array([0,0]) 
        self.res = 0
        self.total_rewards =0
        self.current_step = 0
        self.items = 0
        self.collected = 0
        self.processors = []

    def draw_box(self,i,j,color):
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        # print(sx,sy)
        # print((sx+j*bs,sy+i*bs,bs,bs))
        surf = py.Surface((bs,bs),py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.rect(surf,color,surf.get_rect())
        self.win.blit(surf,(sx+j*bs,sy+i*bs,bs,bs),special_flags=py.BLEND_RGBA_ADD)
#        py.draw.rect(self.win,py.Color(color),(sx+j*bs,sy+i*bs,bs,bs)) 


    def draw_grid(self):
        #horizontal line
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        color = (100,0,0)
        color = (60,0,60)

        for i in range(v+1):
            py.draw.line(self.win,color,(sx,sy+i*bs),(sx+h*bs,sy+i*bs))
            
        for i in range(h+1):
            py.draw.line(self.win,color,(sx+i*bs,sy),(sx+i*bs,sy+v*bs))


    def draw_visibility(self):
        if self.visibility is not None:
            vv,vh = self.visibility.shape
            ax,ay = self.agent_pos
            lx,ly = self.grid.shape
            for i in range(vv):
                for j in range(vh):
                    if 0<= i+ax-vv//2 < lx and 0<= j+ay-vh//2 < ly: 
                        self.draw_box(i+ax-vv//2,j+ay-vh//2,[50,50,50])

 
    def draw_items(self):
        v,h = self.grid.shape
        #v=20 h = 30
        for i in range(v):
            for j in range(h):
                if self.grid[i][j] == self.RES:
                    self.draw_box(i,j,(0,255,0))
                elif self.grid[i][j] == self.PROCESSOR:
                    self.draw_box(i,j,(255,0,0))
                elif self.grid[i][j] == self.ITEM:
                    self.draw_box(i,j,(0,255,255))

    def draw_trail(self):

        for i in range(len(self.trail_positions)):
            x,y = self.trail_positions[i]
            c = 30 + (1.4)**i
            self.draw_box(x,y,(0,0,c))

    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.draw_trail()
        self.draw_items()
        self.render_text(f"Collected :{self.collected:3}",(0,0))
        self.render_text(f"Items     :{self.items:3}",(0,20))
        self.render_text(f"Steps     :{self.current_step:4}",(250,0))
        # self.render_text(f"Rewards :{self.total_rewards:3}",(250,20))
        self.render_text(f"Energy :{self.agent_energy:3}",(250,20))
        self.get_state()
        self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        self.draw_visibility()
        py.display.update()

    def spawn_resources(self):
        #initializing foods
        self.timer +=1
        v,h = self.grid_resource.shape
        if self.timer%10 == 0 and self.res < 10:
            while True: 
                r = np.random.randint(0,v)
                c = np.random.randint(0,h)
                if self.grid_resource[r][c] == 0:
                    self.grid_resource[r][c] = 1 
                    self.res +=1
                    break

    def event_handler(self,event):
        pass 
    
    def update(self):
        self.spawn_resources()
    
    def step(self,speed=0):
        self.event_handler(None)
        self.update()
        self.current_step += 1
        if self.enable_draw:
            self.draw()
