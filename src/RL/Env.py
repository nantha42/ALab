import pygame as py 
import typing
import numpy as np 
np.random.seed(5)



class PowerGame:
    def __init__(self):
        self.w = 600
        self.h = 600
        self.win = py.display.set_mode((900,600))
        self.agents = None 
        self.grid = np.zeros((20,30))
        self.test = [0,0]
        self.clock = py.time.Clock()
        self.box_size = 20 
        self.start_position= [50,50]
        self.timer = 0
        self.res = 0
        self.agent_pos = np.array([0,0])
        #initializing agents
        self.RES = 9
    
    def get_state(self) -> np.ndarray :
        x,y = self.agent_pos
        vis = np.ones((5,5))*-1
        v,h = self.grid.shape

        for i in range(-2,3):
            for j in range(-2,3):
                if(0 <= x+i < h and 0 <= y+j < v):
                    vis[i+2,j+2] = self.grid[x+i,y+i] 
        return vis 

    def act(self,action:np.ndarray):
        pass

    def draw_box(self,i,j,color):
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        py.draw.rect(self.win,color,(sx+j*bs,sy+i*bs,bs,bs)) 

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


    def draw_items(self):
        v,h = self.grid.shape
        #v=20 h = 30
        for i in range(v):
            for j in range(h):
                if self.grid[i][j] == self.RES:
                    self.draw_box(i,j,(0,255,0))
 
    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.draw_items()

        self.agent_pos[1] = ((self.agent_pos[1]+1)%20)
        if self.agent_pos[1] == 0:
            self.agent_pos[0] = ((self.agent_pos[0]+1)%30)
        print(self.agent_pos)
        print(self.get_state())
        self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        py.display.update()
 
    def spawn_resources(self):
        #initializing foods
        pass

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()
    
    def update(self):
        self.timer += 1 
        v,h = self.grid.shape
        if self.timer%10 == 0 and self.res < 10:
            r = np.random.randint(v-10,v)
            c = np.random.randint(h-10,h)
            self.grid[r][c] = self.RES
            self.res +=1
        pass

    def step(self):
        self.event_handler()
        self.update()
        self.draw()
        self.clock.tick(10)

if __name__ == '__main__':
    r = PowerGame() 
    print(r.get_state())
    while True:
        r.step()

