import pygame as py 
import numpy as np 


class PowerGame:
    def __init__(self):
        self.w = 600
        self.h = 600
        self.win = py.display.set_mode((900,600))
        self.agents = None 
        self.grid = np.zeros((5,6))
        self.test = [0,0]
        self.clock = py.time.Clock()
        #initializing agents

    def draw_box(self,i,j,color):
        v,h = self.grid.shape
        r = 1
        bs = 10 #box size
        sx,sy = 50,50
        py.draw.rect(self.win,color,(sx+j*bs,sy+i*bs,bs,bs)) 

    def draw_grid(self):
        #horizontal line
        v,h = self.grid.shape
        r = 1
        bs = 10 #box size
        sx,sy = 50,50
        color = (100,0,0)

        for i in range(v+1):
            py.draw.line(self.win,color,(sx,sy+i*bs),(sx+h*bs,sy+i*bs))
            
        for i in range(h+1):
            py.draw.line(self.win,color,(sx+i*bs,sy),(sx+i*bs,sy+v*bs))

        # for i in range((v)):
        #     py.draw.line(self.win,(255,255,255),(50,(i+1)*(self.w/h)*r),(50+(h-1)*(self.h/v)*r,(i+1)*(self.w/h)*r),1)
        # #vertical line
        # for i in range((h)):
        #     py.draw.line(self.win,(255,255,255),(50+(i)*(self.h/v)*r,(self.w/h)*r),(50+(i)*(self.h/v)*r,(v)*(self.w/h)*r),1)


    def spawn_power(self):
        #initializing foods
        pass

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()
    
    def update(self):
        pass

    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.test[1] = ((self.test[1]+1)%6)
        if self.test[1] == 0:
            self.test[0] = ((self.test[0]+1)%5)
        self.draw_box(self.test[0],self.test[1],(0,200,0,))
        py.display.update()

    def step(self):
        self.event_handler()
        self.update()
        self.draw()
        self.clock.tick(1)

if __name__ == '__main__':
    r = PowerGame() 
    while True:
        r.step()

