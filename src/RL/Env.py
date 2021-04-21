import pygame as py 
import typing
import numpy as np 
np.random.seed(5)

ENABLE_DRAW = True 

class PowerGame:
    def __init__(self,gr=10,gc=10,vis=7):
        py.init()
        self.box_size = 20 
        self.grid = np.zeros((gr,gc))
        self.vissize = vis
        self.w = 50 + gc*self.box_size + 50
        self.h = 50 + gr*self.box_size + 50
        self.win = py.display.set_mode((self.w,self.h),py.DOUBLEBUF,32)
        self.win.set_alpha(128)
        self.agents = None 
        self.test = [0,0]
        self.clock = py.time.Clock()
        self.start_position= [50,50]
        self.timer = 0
        self.res = 0
        self.collected = 0
        self.processors = 0
        self.agent_pos = np.array([0,0])
        self.enable_draw = ENABLE_DRAW 
        self.visibility = None
        self.processors = []
        #initializing agents
        self.RES = 9
        self.PROCESSOR = 8
        self.ITEM = 7
    
    def get_state(self) -> np.ndarray :
        x,y = self.agent_pos
        vis = np.ones((self.vissize,self.vissize))*-1
        v,h = self.grid.shape
        for i in range(-2,3):
            for j in range(-2,3):
                if(0 <= x+i < v and 0 <= y+j < h):
                    vis[i+2,j+2] = self.grid[x+i,y+j] 
        self.visibility = vis
        return self.visibility 


    def act(self,action:np.ndarray):
        left,up,right,down = action[:4]
        v,h = self.grid.shape
        collect = action[4]
        build_proc = action[5]
        cx,cy = self.agent_pos
        
        reward = 0
        if left > 0 and self.agent_pos[1]>0:
            self.agent_pos[1] -=1
        elif right > 0 and self.agent_pos[1]< h-1:
            self.agent_pos[1] += 1
        elif up > 0 and self.agent_pos[0] >0:
            self.agent_pos[0] -=1 
        elif down>0 and self.agent_pos[0] < v-1:
            self.agent_pos[0] +=1
        elif collect > 0 and (self.grid[cx][cy] == self.RES or self.grid[cx][cy] == self.ITEM ):
            if self.grid[cx][cy] == self.RES:
                self.res -= 1
                reward = 1
                self.collected += 1
            elif self.grid[cx][cy] == self.ITEM:
                reward = 5
            self.grid[cx][cy] = 0
            reward = 1

        elif build_proc > 0 and self.collected >= 7 and self.grid[cx][cy] == 0:
            self.grid[cx][cy] = self.PROCESSOR
            self.processors.append([cx,cy])
            self.collected  -= 7  # 7 resources = 1 processor
            self.reward = 2
        
        new_state = self.get_state()
        return new_state,reward
        

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

    def reset(self,hard=False):
        if hard:
            np.random.seed(5)
        v,h = self.grid.shape
        self.grid = np.zeros((v,h))
        self.agent_pos = np.array([0,0]) 
        self.res = 0
        self.collected = 0
        self.processors = []



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
 
    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.draw_items()

        # self.agent_pos[1] = ((self.agent_pos[1]+1)%30)
        # if self.agent_pos[1] == 0:
        #     self.agent_pos[0] = ((self.agent_pos[0]+1)%20)
        # # print(self.agent_pos)
        # print(self.get_state())
        self.get_state()
        self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        self.draw_visibility()
        py.display.update()
    
    def spawn_resources(self):
        #initializing foods
        self.timer +=1
        v,h = self.grid.shape
        if self.timer%10 == 0 and self.res < 10:
            while True: 
                r = np.random.randint(0,v)
                c = np.random.randint(0,h)
                if self.grid[r][c] == 0:
                    self.grid[r][c] = self.RES
                    self.res +=1
                    break
        if self.timer%10 == 0:
            for processor in self.processors:
                px,py = processor
                if self.collected > 3:
                    if px -1 > -1 and self.grid[px-1][py] == 0:
                        self.grid[px-1][py] = self.ITEM
                        self.collected -= 3
                    elif px+1 < v and self.grid[px+1][py] == 0:
                        self.grid[px+1][py] = self.ITEM
                        self.collected -= 3
                    elif py -1 > -1 and self.grid[px][py-1] == 0:
                        self.grid[px][py-1] = self.ITEM
                        self.collected -= 3
                    elif py+1 < h and self.grid[px][py+1] == 0:
                        self.grid[px][py+1] = self.ITEM
                        self.collected -= 3
                else:
                    break
        pass

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()
    
    def update(self):
        self.spawn_resources()
        pass

    def step(self,speed=0):
        self.event_handler()
        self.update()
        if self.enable_draw:
            self.draw()
            self.clock.tick(speed)

if __name__ == '__main__':
    r = PowerGame() 
    # print(r.get_state())
    while True:
        r.step()

