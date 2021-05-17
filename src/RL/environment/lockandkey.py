import pygame as py 
import typing
import numpy as np 
from icecream import ic


ENABLE_DRAW = True 
class Agent:
    def __init__(self,id,pos,vsize,atype="P"):
        self.id = id 
        self.x,self.y = pos
        self.vissize = vsize  #not square number 
        self.visibility = None
        self.atype = atype 
        self.energy = 10
        self.locked = False

         
class LockAndKey:
    def __init__(self,players,lockers,gr=10,gc=10,vis=7):
        self.box_size = 20 
        self.font = py.font.SysFont("times",20)
        self.w = 50 + gc*self.box_size + 50
        self.h = 50 + gr*self.box_size + 50
        self.win =  py.Surface((self.w,self.h),py.DOUBLEBUF,32)
 
        self.grid = np.zeros((gr,gc))
        self.vissize = vis

        self.nlockers = lockers
        self.nplayers = players
        self.__initialize_agents()
        self.start_position= [50,50]
        self.timer = 0
        self.enable_draw = ENABLE_DRAW 
        self.visibility = None
        self.current_step = 0
        self.total_rewards = 0


        ###### GRID ITEMS #####
        self.PLAYER = 1
        self.LOCKER = 2
        self.LOCKED = 3

    def __initialize_agents(self):
        self.agents = []
        gr,gc = self.grid.shape
        ##initializing lockers
        for i in range(self.nplayers):
            x,y = np.random.randint(0,gr),np.random.randint(0,gc)
            self.agents.append(Agent(i,[x,y],self.vissize,atype="P"))

        for j in range(self.nlockers):
            x,y = np.random.randint(0,gr), np.random.randint(0,gc)
            self.agents.append(Agent(self.nplayers + j,[x,y],self.vissize,atype="L" ))
        

    def render_text(self,text,pos):
        text = self.font.render(text,True,(200,200,200))
        trect = text.get_rect()
        trect.topleft =  pos 
        self.win.blit(text,trect)

    def initial_state(self):
        
        for agent in self.agents:
            if agent.atype == "P":
                self.grid[agent.x][agent.y] = self.PLAYER
            else: 
                self.grid[agent.x][agent.y] = self.LOCKER

    def update_state(self,agentid):
        agent = self.agents[agentid]
        x,y = agent.x,agent.y 
        vis = np.ones((self.vissize,self.vissize))*-1
        v,h = self.grid.shape
        for i in range(-2,3):
            for j in range(-2,3):
                if(0 <= x+i < v and 0 <= y+j < h):
                    vis[i+2,j+2] = self.grid[x+i,y+j] 
        agent.visibility = vis

    def act(self,id,action:np.ndarray):
        left,up,right,down = action[:4]
        agent = self.agents[id]
        if agent.locked == False:
            v,h = self.grid.shape
            release = action[4]
            cx,cy = agent.x,agent.y 
            reward = 0
            if left > 0 and agent.y > 0 :
                agent.y -= 1 
            elif right > 0 and agent.y < h-1:
                agent.y += 1
            elif up > 0 and agent.x >0:
                agent.x -=1
            elif down>0 and self.agent_pos[0] < v-1:
                agent.x += 1
            elif release > 0 and self.grid[cx][cy] == self.LOCKED:
                for agenty in self.agents:
                    if agenty.x == cx and agenty.y == cy and agenty.locked:
                        agenty.locked = False
            if cx != agent.x or cy != agent.y:
                self.grid[cx][cy] = 0
                if agent.atype == 'L':
                    self.grid[agent.x][agent.y] = self.LOCKER 
                else: 
                    self.grid[agent.x][agent.y] = self.PLAYER

            self.update_state(id)
            self.total_rewards += reward
            return agent.visibility, reward
        else:
            self.update_state(id)
            self.total_rewards += reward
            return agent.visibility,reward
 
    def reset(self,hard=False):
        if hard:
            np.random.seed(5)
        v,h = self.grid.shape
        self.grid = np.zeros((v,h))
        self.__initialize_agents() 
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
        surf = py.Surface((bs,bs),py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.rect(surf,color,surf.get_rect())
        self.win.blit(surf,(sx+j*bs,sy+i*bs,bs,bs),special_flags=py.BLEND_RGBA_ADD)


    def draw_grid(self):
        #horizontal line
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        color = (100,0,0)

        for i in range(v+1):
            py.draw.line(self.win,color,(sx,sy+i*bs),(sx+h*bs,sy+i*bs))
            
        for i in range(h+1):
            py.draw.line(self.win,color,(sx+i*bs,sy),(sx+i*bs,sy+v*bs))


    def draw_visibility(self):
        for agent in self.agents:
            visibility = self.visibility
            if visibility is not None:
                vv,vh = visibility.shape
                ax,ay = agent.x,agent.y 
                lx,ly = self.grid.shape
                for i in range(vv):
                    for j in range(vh):
                        if 0<= i+ax-vv//2 < lx and 0<= j+ay-vh//2 < ly: 
                            self.draw_box(i+ax-vv//2,j+ay-vh//2,[50,50,50])

    def draw_items(self):
        v,h = self.grid.shape
        for i in range(v):
            for j in range(h):
                if self.grid[i][j] == self.PLAYER:
                    self.draw_box(i,j,(0,255,0))
                elif self.grid[i][j] == self.LOCKER:
                    self.draw_box(i,j,(255,0,0))


    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.draw_items()
        self.render_text(f"Steps     :{self.current_step:4}",(250,0))
        self.render_text(f"Rewards :{self.total_rewards:3}",(250,20))
        self.draw_visibility()
        py.display.update()

    def event_handler(self,event):
        pass

    def step(self,speed=0):
        self.event_handler(None)
        self.current_step += 1
        if self.enable_draw:
            self.draw()


if __name__ == '__main__':
    py.init()
    wind = py.display.set_mode((600,600),py.DOUBLEBUF,32)
    lk = LockAndKey(4,1,gr=15,gc=15)
    lk.initial_state()
    print(lk.grid)
    lk.step()
    while True:
        for event in py.event.get():
            if event.type == py.QUIT:
                exit()
        wind.blit(lk.win,(0,0))
        py.display.update()
